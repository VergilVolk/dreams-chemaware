#!/usr/bin/env python3
"""Run one predeclared cross-source structure-hidden response positive control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_pan_redu_factorial_response_pair import (
    feature_id,
    permutation_test,
    read_mgf,
    read_quant,
    sample_stem,
    sha256,
    signed_score,
    sketch,
    spectral_matrix,
)


def fit_study(
    specification: dict[str, str], pair: dict[str, str], unit_map: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, object], dict[str, object]]:
    quant_path = Path(pair["quant_local_path"])
    mgf_path = Path(pair["mgf_local_path"])
    quant, sample_columns = read_quant(quant_path)
    rows = unit_map.loc[
        unit_map["contrast_id"].eq(specification["contrast_id"])
        & unit_map["dataset_accession"].eq(specification["dataset_accession"])
        & unit_map["biological_unit_id"].ne("")
    ].copy()
    rows["_stem"] = rows["filename"].map(sample_stem)
    column_by_stem = {sample_stem(column): column for column in sample_columns}
    rows = rows.loc[rows["_stem"].isin(column_by_stem)].copy()
    rows["_quant_column"] = rows["_stem"].map(column_by_stem)
    if rows["_quant_column"].duplicated().any():
        raise ValueError(f"{specification['study_id']}: a quantitative file maps more than once")

    crossing = rows.groupby("biological_unit_id")["level"].nunique()
    excluded_units = set(crossing.index[crossing > 1])
    rows = rows.loc[~rows["biological_unit_id"].isin(excluded_units)].copy()
    levels = sorted(rows["level"].unique(), key=str.casefold)
    if levels != ["female", "male"]:
        raise ValueError(f"{specification['study_id']}: expected female/male, observed {levels}")

    selected = quant[rows["_quant_column"].tolist()].to_numpy(np.float64)
    active = np.any(selected > 0, axis=1)
    quant = quant.iloc[active].copy()
    selected = selected[active]
    minimum = np.nanmin(np.where(selected > 0, selected, np.nan), axis=1)
    transformed = np.log2(selected + minimum[:, None] / 2.0)
    transformed -= np.median(transformed, axis=0, keepdims=True)
    values = pd.DataFrame(transformed, index=quant.index, columns=rows["_quant_column"])

    unit_vectors: list[np.ndarray] = []
    unit_levels: list[str] = []
    for unit, block in rows.groupby("biological_unit_id", sort=True):
        unit_vectors.append(np.median(values[block["_quant_column"].tolist()].to_numpy(), axis=1))
        unit_levels.append(str(block["level"].iloc[0]))
    matrix = np.stack(unit_vectors)
    labels = np.asarray(unit_levels)
    female = matrix[labels == "female"]
    male = matrix[labels == "male"]
    if min(len(female), len(male)) < 5:
        raise ValueError(f"{specification['study_id']}: fewer than five units in a level")
    beta = male.mean(axis=0) - female.mean(axis=0)
    se = np.sqrt(
        male.var(axis=0, ddof=1) / len(male)
        + female.var(axis=0, ddof=1) / len(female)
    )
    se = np.maximum(se, 1e-8)
    response = beta / np.sqrt(beta * beta + se * se)

    # Prevalence is computed over the admitted acquisition files, not DDA scan
    # frequency and not a phenotype-selected subset.
    prevalence = (selected > 0).mean(axis=1)
    spectra = read_mgf(mgf_path)
    shared = [entity for entity in quant.index.astype(str) if feature_id(entity) in spectra]
    position = {entity: index for index, entity in enumerate(quant.index.astype(str))}
    effects = pd.DataFrame({
        "study_id": specification["study_id"],
        "dataset_accession": specification["dataset_accession"],
        "source_family": specification["source_family"],
        "contrast_id": specification["contrast_id"],
        "entity_id": shared,
        "beta_male_minus_female": [float(beta[position[entity]]) for entity in shared],
        "se": [float(se[position[entity]]) for entity in shared],
        "bounded_response": [float(response[position[entity]]) for entity in shared],
        "prevalence": [float(prevalence[position[entity]]) for entity in shared],
        "identity_status": "unresolved_entity",
    })
    report = {
        "study_id": specification["study_id"],
        "dataset_accession": specification["dataset_accession"],
        "joined_acquisition_files": len(rows),
        "excluded_label_inconsistent_units": len(excluded_units),
        "biological_units": len(unit_vectors),
        "female_units": len(female),
        "male_units": len(male),
        "ms2_linked_entities": len(shared),
        "quant_sha256": sha256(quant_path),
        "mgf_sha256": sha256(mgf_path),
    }
    shared_positions = np.asarray([position[entity] for entity in shared], dtype=int)
    auxiliary = {
        "unit_matrix": matrix[:, shared_positions],
        "male_label": labels == "male",
        "prevalence": prevalence[shared_positions],
    }
    return effects, {entity: spectra[feature_id(entity)] for entity in shared}, report, auxiliary


def conditional_permutation_test(
    matrices: list,
    responses: list[np.ndarray],
    precursor_mz: list[np.ndarray],
    observed: float,
    permutations: int,
    width: float,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    strata = [np.floor(values / width).astype(int) for values in precursor_mz]
    null = np.empty(permutations, dtype=np.float64)
    for permutation in range(permutations):
        shuffled = []
        for response, study_strata in zip(responses, strata):
            candidate = response.copy()
            for label in np.unique(study_strata):
                index = np.flatnonzero(study_strata == label)
                candidate[index] = rng.permutation(candidate[index])
            shuffled.append(candidate)
        null[permutation] = signed_score(
            sketch(matrices[0], shuffled[0]), sketch(matrices[1], shuffled[1])
        )["signed_response_score"]
    return {
        "permutations": permutations,
        "precursor_mass_stratum_width_da": width,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "z_vs_mass_stratified_shuffle": float(
            (observed - null.mean()) / max(null.std(ddof=1), 1e-12)
        ),
        "one_sided_p": float((1 + np.count_nonzero(null >= observed)) / (permutations + 1)),
        "null_q95": float(np.quantile(null, 0.95)),
        "null_q99": float(np.quantile(null, 0.99)),
    }


def label_permutation_test(
    matrices: list,
    auxiliaries: list[dict[str, object]],
    observed: float,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    null = np.empty(permutations, dtype=np.float64)
    for permutation in range(permutations):
        permuted_responses = []
        for auxiliary in auxiliaries:
            matrix = np.asarray(auxiliary["unit_matrix"])
            male = rng.permutation(np.asarray(auxiliary["male_label"], dtype=bool))
            female_values = matrix[~male]
            male_values = matrix[male]
            beta = male_values.mean(axis=0) - female_values.mean(axis=0)
            se = np.sqrt(
                male_values.var(axis=0, ddof=1) / len(male_values)
                + female_values.var(axis=0, ddof=1) / len(female_values)
            )
            se = np.maximum(se, 1e-8)
            response = beta / np.sqrt(beta * beta + se * se)
            response *= np.sqrt(np.asarray(auxiliary["prevalence"], dtype=float))
            permuted_responses.append(response)
        null[permutation] = signed_score(
            sketch(matrices[0], permuted_responses[0]),
            sketch(matrices[1], permuted_responses[1]),
        )["signed_response_score"]
    return {
        "permutations": permutations,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "z_vs_independent_unit_label_shuffle": float(
            (observed - null.mean()) / max(null.std(ddof=1), 1e-12)
        ),
        "one_sided_p": float((1 + np.count_nonzero(null >= observed)) / (permutations + 1)),
        "null_q95": float(np.quantile(null, 0.95)),
        "null_q99": float(np.quantile(null, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--unit-map", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--max-mz", type=float, default=1200.0)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--mass-shuffle-width", type=float, default=5.0)
    parser.add_argument("--label-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "active" or len(config.get("studies", [])) != 2:
        raise ValueError("configuration must contain exactly two active studies")
    pairs = pd.read_csv(args.pairs, sep="\t", dtype=str, keep_default_na=False)
    pair_by_accession = pairs.set_index("dataset_accession").to_dict("index")
    unit_map = pd.read_csv(args.unit_map, sep="\t", dtype=str, keep_default_na=False)
    fitted = [
        fit_study(study, pair_by_accession[study["dataset_accession"]], unit_map)
        for study in config["studies"]
    ]
    effects = [result[0] for result in fitted]
    spectra = [result[1] for result in fitted]
    study_reports = [result[2] for result in fitted]
    auxiliaries = [result[3] for result in fitted]
    fragments = []
    precursors = []
    responses = []
    precursor_mz = []
    for effect, spectrum in zip(effects, spectra):
        ids = effect["entity_id"].tolist()
        fragment, precursor = spectral_matrix(ids, spectrum, args.bin_width, args.max_mz)
        fragments.append(fragment); precursors.append(precursor)
        precursor_mz.append(np.asarray([float(spectrum[entity]["precursor_mz"]) for entity in ids]))
        responses.append(
            effect["bounded_response"].to_numpy(float)
            * np.sqrt(effect["prevalence"].to_numpy(float))
        )

    fragment_score = signed_score(sketch(fragments[0], responses[0]), sketch(fragments[1], responses[1]))
    precursor_score = signed_score(sketch(precursors[0], responses[0]), sketch(precursors[1], responses[1]))
    rng = np.random.default_rng(args.seed)
    fragment_null = permutation_test(
        fragments, responses, fragment_score["signed_response_score"], args.permutations, rng
    )
    precursor_null = permutation_test(
        precursors, responses, precursor_score["signed_response_score"], args.permutations, rng
    )
    mass_conditioned_fragment_null = conditional_permutation_test(
        fragments, responses, precursor_mz, fragment_score["signed_response_score"],
        args.permutations, args.mass_shuffle_width, rng,
    )
    label_null = label_permutation_test(
        fragments, auxiliaries, fragment_score["signed_response_score"],
        args.label_permutations, rng,
    )
    rows = [
        {"representation": "sqrt_intensity_fragment_bins", **fragment_score, **fragment_null},
        {"representation": "precursor_mass_bins", **precursor_score, **precursor_null},
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(effects, ignore_index=True).to_csv(args.output_dir / "entity_effects.tsv", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "pair_scores.tsv", sep="\t", index=False)
    report = {
        "schema": "pan_redu_crosssectional_signed_response_pair_v1",
        "status": "development_negative_result_outer_label_null_failed",
        "decision": (
            "PASS" if label_null["one_sided_p"] <= 0.05
            else "FAIL_BIOLOGICAL_UNIT_LABEL_NULL"
        ),
        "contrast_family": config["contrast_family"],
        "effect_direction": config["effect_direction"],
        "denominators": {
            "independent_source_families": 2,
            "biological_units": int(sum(row["biological_units"] for row in study_reports)),
            "ms2_linked_unresolved_entities": int(sum(len(frame) for frame in effects)),
            "permutations_per_representation": args.permutations,
        },
        "study_reports": study_reports,
        "fragment_score": {**fragment_score, **fragment_null},
        "precursor_only_score": {**precursor_score, **precursor_null},
        "fragment_score_mass_conditioned_null": mass_conditioned_fragment_null,
        "fragment_score_biological_unit_label_null": label_null,
        "leakage_status": "contrast_and_direction_frozen_before_feature_effects_or_pair_score_were_opened",
        "claim_boundary": [
            "This is a two-source positive control for response retrieval, not repository-scale validation.",
            "The biological-unit label permutation is the decisive outer null; entity-to-spectrum shuffles alone cannot establish a biological response.",
            "All entities are structure-hidden, so the run cannot estimate known-only versus unresolved increment.",
            "The smaller ONR cohort and unadjusted sex association limit biological interpretation.",
            "The fragment-bin representation is a transparent baseline and must be replaced by frozen Noise/unified embeddings without retuning the biological model.",
        ],
        "inputs": {"config_sha256": sha256(args.config)},
        "parameters": {
            "bin_width_da": args.bin_width,
            "max_mz": args.max_mz,
            "permutations": args.permutations,
            "mass_shuffle_width_da": args.mass_shuffle_width,
            "label_permutations": args.label_permutations,
            "seed": args.seed,
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
