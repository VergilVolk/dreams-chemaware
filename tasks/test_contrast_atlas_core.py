from __future__ import annotations

import pandas as pd
import pytest

from contrast_atlas_core import retrieve


def fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    designs = {
        ("Q", "source_q", "q"): [2.0, 1.0, -1.0, -2.0],
        ("S", "source_s", "s"): [1.8, 0.8, -0.9, -1.7],
        ("O", "source_o", "o"): [-1.8, -0.8, 0.9, 1.7],
        ("L", "source_l", "l"): [1.5],
    }
    for (study, source, contrast), values in designs.items():
        for index, beta in enumerate(values):
            rows.append({
                "study_id": study, "source_family": source, "contrast_id": contrast,
                "entity_id": f"{study}{index}", "independent_unit_id": f"{study}u{index}",
                "identity_status": "library_supported" if index < 2 else "unresolved_entity",
                "beta": beta, "se": 0.2, "qc_pass": True, "detectable": True,
                "data_role": "development/consumed",
            })
    effects = pd.DataFrame(rows)
    loading_rows = []
    for row in rows:
        pattern = "p1" if int(row["entity_id"][-1]) < 2 else "p2"
        loading_rows.append({
            "study_id": row["study_id"], "contrast_id": row["contrast_id"],
            "entity_id": row["entity_id"], "pattern_id": pattern, "weight": 1.0,
            "calibrated": True, "phenotype_blind": True,
        })
    links = []
    for target, contrast, count in (("S", "s", 4), ("O", "o", 4), ("L", "l", 1)):
        for index in range(count):
            links.append({
                "query_study_id": "Q", "query_contrast_id": "q", "query_entity_id": f"Q{index}",
                "target_study_id": target, "target_contrast_id": contrast, "target_entity_id": f"{target}{index}",
                "match_probability": 0.95, "claim_level": "strict_entity", "phenotype_blind": True,
            })
    return effects, pd.DataFrame(loading_rows), pd.DataFrame(links)


def test_multilevel_retrieval_finds_similar_opposite_and_abstains() -> None:
    effects, loadings, links = fixtures()
    result, patterns = retrieve(effects, loadings, links, "Q", "q")
    all_world = result.loc[result["world"] == "all"].set_index("target_study_id")
    assert all_world.loc["S", "study_contrast_association"] == "similar_response"
    assert all_world.loc["O", "study_contrast_association"] == "opposite_response"
    assert all_world.loc["L", "study_contrast_association"] == "abstain_insufficient_comparable_evidence"
    assert bool(all_world.loc["S", "strict_entity_pass"])
    assert not bool(all_world.loc["L", "strict_entity_pass"])
    assert all_world.loc["S", "strict_entity_query_coverage"] == pytest.approx(1.0)
    assert all_world.loc["S", "strict_entity_target_coverage"] == pytest.approx(1.0)
    assert set(patterns["world"]) == {"all", "known_only", "unresolved_only"}
    assert patterns["uncertainty_model"].eq(
        "marginal_se_perfect_positive_dependence_upper_bound"
    ).all()
    assert patterns["pattern_se"].eq(0.2).all()


def test_evidence_table_rejects_effect_leakage() -> None:
    effects, loadings, links = fixtures()
    links["beta"] = 1.0
    with pytest.raises(RuntimeError, match="phenotype/effect"):
        retrieve(effects, loadings, links, "Q", "q")


def test_strict_links_reject_ambiguous_and_missing_endpoints() -> None:
    effects, loadings, links = fixtures()
    ambiguous = pd.concat([links, pd.DataFrame([{
        "query_study_id": "Q", "query_contrast_id": "q", "query_entity_id": "Q0",
        "target_study_id": "S", "target_contrast_id": "s", "target_entity_id": "S1",
        "match_probability": 0.91, "claim_level": "strict_entity", "phenotype_blind": True,
    }])], ignore_index=True)
    with pytest.raises(RuntimeError, match="one-to-one"):
        retrieve(effects, loadings, ambiguous, "Q", "q")

    missing = links.copy()
    missing.loc[0, "target_entity_id"] = "absent"
    with pytest.raises(RuntimeError, match="absent or QC-ineligible"):
        retrieve(effects, loadings, missing, "Q", "q")

    missing_loading = loadings.copy()
    missing_loading.loc[0, "entity_id"] = "absent"
    with pytest.raises(RuntimeError, match="pattern loadings contain absent"):
        retrieve(effects, missing_loading, links, "Q", "q")


def test_fixed_universe_retains_missing_evidence_as_abstention() -> None:
    universe, universe_loadings, universe_links = fixtures()
    evidence = universe.loc[universe["study_id"].eq("S")].copy()
    loadings = universe_loadings.loc[universe_loadings["study_id"].eq("S")].copy()
    links = universe_links.iloc[0:0].copy()
    result, patterns = retrieve(
        evidence, loadings, links, "Q", "q",
        universe_effects_raw=universe,
        identity_worlds=("all",),
    )
    assert set(result["target_study_id"]) == {"S", "O", "L"}
    assert result["study_contrast_association"].eq("abstain_insufficient_comparable_evidence").all()
    assert result["n_query_entities"].eq(0).all()
    assert set(patterns["study_id"]) == {"S"}


def synchronized_unit_effects(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    offsets = (-0.3, -0.1, 0.1, 0.3)
    for effect in effects.itertuples(index=False):
        for index, offset in enumerate(offsets):
            rows.append({
                "study_id": effect.study_id,
                "contrast_id": effect.contrast_id,
                "biological_unit_id": f"{effect.study_id}_patient_{index}",
                "entity_id": effect.entity_id,
                "effect": effect.beta + offset,
            })
    return pd.DataFrame(rows)


def test_synchronized_biological_units_provide_empirical_pattern_se() -> None:
    effects, loadings, links = fixtures()
    unit_effects = synchronized_unit_effects(effects)
    _, patterns = retrieve(
        effects, loadings, links, "Q", "q",
        identity_worlds=("all",),
        unit_effects_raw=unit_effects,
    )
    assert patterns["uncertainty_model"].eq(
        "synchronized_biological_unit_empirical_se"
    ).all()
    assert patterns["n_biological_units"].eq(4).all()
    assert patterns["n_independent_units"].ge(1).all()
    assert patterns["pattern_se"].to_numpy() == pytest.approx(0.12909944487358055)


def test_synchronized_biological_units_fail_closed_on_mismatch_or_missingness() -> None:
    effects, loadings, links = fixtures()
    mismatch = synchronized_unit_effects(effects)
    mismatch.loc[0, "effect"] += 1.0
    with pytest.raises(RuntimeError, match="means disagree"):
        retrieve(effects, loadings, links, "Q", "q", unit_effects_raw=mismatch)

    incomplete = synchronized_unit_effects(effects)
    incomplete = incomplete.drop(incomplete.index[0])
    with pytest.raises(RuntimeError, match="complete synchronized rectangle"):
        retrieve(effects, loadings, links, "Q", "q", unit_effects_raw=incomplete)
