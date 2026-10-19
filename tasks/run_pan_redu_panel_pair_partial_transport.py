#!/usr/bin/env python3
"""Score one frozen source-disjoint Pan-ReDU panel pair by partial response transport."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_ad_cross_study_spectral_edges import directed_topk
from contrast_response_same_graph_baselines import (
    deterministic_greedy_capacity_score,
    signed_response_centroid_cosine,
)
from contrast_response_transport import fixed_capacity_transport_score, parse_bool
from evaluate_ad_cross_study_response_retrieval import (
    MATCH_COLUMNS,
    fixed_capacity_axes,
    measurement_pair_axes,
    measurement_preserving_derangement,
    optimal_measurement_pairs,
    reassign_response_profiles,
    response_availability_summary,
)
from prepare_pan_redu_partial_transport_pairs import canonical_polarity


KNOWN = {"confirmed_known", "library_supported"}
UNRESOLVED = {"unresolved_entity"}
PAIR_MANIFEST_SCHEMA = "pan_redu_source_disjoint_partial_transport_pairs_v2_polarity_safe"
PAIR_MANIFEST_STATUS = "PRETRUTH_SOURCE_DISJOINT_SAME_POLARITY_PANEL_PAIRS_FROZEN"
PAIR_OUTPUT_SCHEMA = "pan_redu_panel_pair_fixed_capacity_response_transport_v4"
EFFECT_REPORT_CONTRACTS = {
    "pan_redu_panel_contrast_effects_v1": (
        "development_effect_objects_not_cross_study_evidence"
    ),
    "adjusted_axis_contrast_effects_v1": (
        "development_adjusted_effect_object_not_cross_study_evidence"
    ),
}
REVERSED_WORLD = {
    "unresolved_response_reassigned_left": "unresolved_response_reassigned_right",
    "unresolved_response_reassigned_right": "unresolved_response_reassigned_left",
    "measurement_matched_unresolved_response_reassigned_left": (
        "measurement_matched_unresolved_response_reassigned_right"
    ),
    "measurement_matched_unresolved_response_reassigned_right": (
        "measurement_matched_unresolved_response_reassigned_left"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def local_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "run_pan_redu_panel_pair_partial_transport.py",
        "contrast_response_transport.py",
        "evaluate_ad_cross_study_response_retrieval.py",
        "build_ad_cross_study_spectral_edges.py",
        "contrast_response_same_graph_baselines.py",
        "prepare_pan_redu_partial_transport_pairs.py",
    )
    return {name: sha256(root / name) for name in names}


def validate_frozen_pair_manifest(
    pair_manifest: Path,
    topology: Path,
    panel_metadata: Path,
    pairs: pd.DataFrame,
) -> tuple[str, str, str, str]:
    pair_sha = sha256(pair_manifest)
    topology_sha = sha256(topology)
    panel_metadata_sha = sha256(panel_metadata)
    report_path = pair_manifest.parent / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != PAIR_MANIFEST_SCHEMA or report.get("status") != PAIR_MANIFEST_STATUS:
        raise RuntimeError("pair manifest is not a frozen source-disjoint contract")
    provenance = report.get("provenance", {})
    expected = {
        "panel_pairs_sha256": pair_sha,
        "topology_sha256": topology_sha,
        "panel_metadata_sha256": panel_metadata_sha,
    }
    for key, observed in expected.items():
        if provenance.get(key) != observed:
            raise RuntimeError(f"pair-manifest provenance drift: {key}")
    if report.get("required_polarity") != "positive":
        raise RuntimeError("pair manifest is not frozen to positive polarity")
    reported_pairs = int(report.get("denominators", {}).get("source_disjoint_panel_pairs", -1))
    if reported_pairs != len(pairs):
        raise RuntimeError("pair-manifest pair denominator drift")
    return pair_sha, topology_sha, panel_metadata_sha, sha256(report_path)


def load_embeddings(
    root: Path, panel_id: str,
) -> tuple[np.ndarray, np.ndarray, Path, str, str]:
    panel = root / panel_id
    report = json.loads((panel / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "phenotype_blind_frozen_spectral_embeddings":
        raise RuntimeError(f"embeddings are not frozen for {panel_id}")
    if report.get("panel_id") != panel_id:
        raise RuntimeError(f"embedding report panel mismatch for {panel_id}")
    path = panel / "anchor_embeddings.npz"
    expected = report.get("provenance", {}).get("embeddings_sha256")
    actual_sha256 = sha256(path)
    if not expected or actual_sha256 != expected:
        raise RuntimeError(f"embedding checksum drift for {panel_id}")
    with np.load(path, allow_pickle=False) as packed:
        ids = np.asarray(packed["anchor_id"], dtype=str)
        embeddings = np.asarray(packed["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or len(ids) != len(embeddings) or len(ids) != len(set(ids)):
        raise RuntimeError(f"invalid embedding archive for {panel_id}")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or float(np.max(np.abs(norms - 1))) > 2e-5:
        raise RuntimeError(f"non-unit or non-finite embeddings for {panel_id}")
    panel_report_sha256 = str(report.get("provenance", {}).get("panel_report_sha256", ""))
    if not panel_report_sha256:
        raise RuntimeError(f"embedding report lacks panel-object provenance for {panel_id}")
    return ids, embeddings, path, panel_report_sha256, actual_sha256


def load_panel_responses(
    effects_root: Path,
    families_root: Path,
    panel_id: str,
    definitions: pd.DataFrame,
    identity: pd.DataFrame | None,
) -> tuple[
    dict[str, pd.DataFrame], pd.DataFrame, Path, Path, Path, str, str,
    dict[str, str],
]:
    effect_dir = effects_root / panel_id
    effect_report_path = effect_dir / "report.json"
    report = json.loads(effect_report_path.read_text(encoding="utf-8"))
    effect_schema = str(report.get("schema", ""))
    effect_status = str(report.get("status", ""))
    expected_status = EFFECT_REPORT_CONTRACTS.get(effect_schema)
    if expected_status is None or effect_status != expected_status:
        raise RuntimeError(
            f"effects are not an admitted frozen schema/status for {panel_id}: "
            f"{effect_schema!r}/{effect_status!r}"
        )
    if report.get("panel_id") != panel_id:
        raise RuntimeError(f"effect report panel mismatch for {panel_id}")
    effect_path = effect_dir / "contrast_effects.tsv.gz"
    effect_sha256 = sha256(effect_path)
    if report.get("output_sha256", {}).get("contrast_effects") != effect_sha256:
        raise RuntimeError(f"effect-table checksum drift for {panel_id}")
    effects = pd.read_csv(effect_path, sep="\t", dtype={"anchor_id": str, "contrast_id": str})
    if effects.duplicated(["contrast_id", "anchor_id"]).any():
        raise RuntimeError(f"duplicate effect contrast/anchor rows for {panel_id}")
    family_dir = families_root / panel_id
    family_report = json.loads((family_dir / "report.json").read_text(encoding="utf-8"))
    if family_report.get("status") != "PHENOTYPE_BLIND_PANEL_ION_FAMILIES_FROZEN":
        raise RuntimeError(f"ion families are not frozen for {panel_id}")
    if family_report.get("panel_id") != panel_id:
        raise RuntimeError(f"ion-family report panel mismatch for {panel_id}")
    family_path = family_dir / "ion_family_assignments.tsv.gz"
    covariate_path = family_dir / "measurement_covariates.tsv.gz"
    family_provenance = family_report.get("provenance", {})
    family_sha256 = sha256(family_path)
    covariate_sha256 = sha256(covariate_path)
    if family_provenance.get("assignments_sha256") != family_sha256:
        raise RuntimeError(f"ion-family assignment checksum drift for {panel_id}")
    if family_provenance.get("measurement_covariates_sha256") != covariate_sha256:
        raise RuntimeError(f"measurement-covariate checksum drift for {panel_id}")
    families = pd.read_csv(family_path, sep="\t", dtype={"entity_id": str})
    covariates = pd.read_csv(covariate_path, sep="\t", dtype={"entity_id": str})
    if families.entity_id.duplicated().any() or covariates.entity_id.duplicated().any():
        raise RuntimeError(f"duplicate family/covariate entity IDs for {panel_id}")
    if set(families.entity_id) != set(covariates.entity_id):
        raise RuntimeError(f"family/covariate entity universe mismatch for {panel_id}")
    universe = families[[
        "entity_id", "ion_family_id", "ion_family_size", "capacity_weight",
    ]].merge(
        covariates[["entity_id", "precursor_mz", *MATCH_COLUMNS]],
        on="entity_id", how="inner", validate="one_to_one",
    )
    if universe.ion_family_id.isna().any() or universe[["precursor_mz", *MATCH_COLUMNS]].isna().any().any():
        raise RuntimeError(f"incomplete phenotype-blind panel universe for {panel_id}")
    if identity is not None:
        panel_identity = identity.loc[
            identity.panel_id.eq(panel_id), ["anchor_id", "identity_status"]
        ].rename(columns={"anchor_id": "entity_id"})
        if set(panel_identity.entity_id) != set(universe.entity_id):
            raise RuntimeError(f"identity/family entity universe mismatch for {panel_id}")
        universe = universe.merge(panel_identity, on="entity_id", how="inner", validate="one_to_one")
    else:
        # The fixed-capacity constructor requires a stable identity field even
        # when no library partition is available.  This sentinel is never used
        # to create known/unresolved worlds.
        universe["identity_status"] = "identity_partition_unavailable"
    responses: dict[str, pd.DataFrame] = {}
    for definition in definitions.itertuples(index=False):
        block = effects.loc[effects.contrast_id.eq(str(definition.contrast_id))].copy()
        if block.empty:
            raise RuntimeError(f"missing effect object {panel_id}/{definition.contrast_id}")
        definition_columns = ["contrast_field", "level_a", "level_b"]
        if missing := set(definition_columns) - set(block):
            raise RuntimeError(f"effect table lacks frozen definition columns for {panel_id}: {sorted(missing)}")
        observed_definition = block[definition_columns].astype(str).drop_duplicates()
        expected_definition = tuple(str(getattr(definition, column)) for column in definition_columns)
        if len(observed_definition) != 1 or tuple(observed_definition.iloc[0]) != expected_definition:
            raise RuntimeError(
                f"effect/topology contrast-definition drift for {panel_id}/{definition.contrast_id}"
            )
        if set(block.anchor_id.astype(str)) != set(universe.entity_id):
            raise RuntimeError(f"effect/family entity universe mismatch for {panel_id}/{definition.contrast_id}")
        block = block.loc[parse_bool(block.passes_effect_qc, "passes_effect_qc")].copy()
        block = block.rename(columns={
            "anchor_id": "entity_id", "beta_b_minus_a": "beta",
            "unit_prevalence": "prevalence",
        })
        block["entity_id"] = block.entity_id.astype(str)
        block = block.drop(columns=["precursor_mz"], errors="ignore").merge(
            universe, on="entity_id", how="left", validate="one_to_one",
        )
        if block.ion_family_id.isna().any() or block[MATCH_COLUMNS].isna().any().any():
            raise RuntimeError(f"incomplete ion-family or measurement-covariate coverage for {panel_id}")
        if identity is not None and block.identity_status.isna().any():
            raise RuntimeError(f"incomplete identity partition for {panel_id}")
        responses[str(definition.contrast_id)] = block[
            ["entity_id", "beta", "se", "prevalence", "precursor_mz", *MATCH_COLUMNS,
             "ion_family_id", "ion_family_size", "capacity_weight"]
            + (["identity_status"] if identity is not None else [])
        ]
    effect_panel_sha256 = str(report.get("input_sha256", {}).get("panel_report", ""))
    family_panel_sha256 = str(family_provenance.get("panel_report_sha256", ""))
    if not effect_panel_sha256 or not family_panel_sha256:
        raise RuntimeError(f"effect/family report lacks panel-object provenance for {panel_id}")
    return (
        responses, universe, effect_path, family_path, covariate_path,
        effect_panel_sha256, family_panel_sha256,
        {
            "effect_report_sha256": sha256(effect_report_path),
            "effect_schema": effect_schema,
            "effect_status": effect_status,
            "effects_sha256": effect_sha256,
            "families_sha256": family_sha256,
            "measurement_covariates_sha256": covariate_sha256,
        },
    )


def measurement_axes_or_empty(
    axes: dict[str, pd.DataFrame],
    pairs: pd.DataFrame,
    *,
    member: str,
    require_pairwise_common_availability: bool = False,
) -> dict[str, pd.DataFrame]:
    """Retain an empty fixed world as an abstention instead of deleting it."""

    if pairs.empty or (axes and all(frame.empty for frame in axes.values())):
        return {name: frame.iloc[0:0].copy() for name, frame in axes.items()}
    return measurement_pair_axes(
        axes, pairs, member=member,
        require_pairwise_common_availability=require_pairwise_common_availability,
    )


def identity_axes_or_empty(
    axes: dict[str, pd.DataFrame], statuses: set[str], *, renormalize: bool,
) -> dict[str, pd.DataFrame]:
    """Keep a missing identity stratum as an explicit abstaining world."""

    result: dict[str, pd.DataFrame] = {}
    for axis, frame in axes.items():
        selected = frame.loc[frame.identity_status.astype(str).isin(statuses)].copy()
        total = float(selected.fixed_capacity.sum())
        if total > 0 and renormalize:
            selected["fixed_capacity"] /= total
        result[axis] = selected
    return result


def reassignment_axes_or_empty(
    universe: pd.DataFrame,
    axes: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, bool, str]:
    """Derange unresolved response profiles or preserve an explicit abstention."""

    unresolved_entities = int(universe.identity_status.astype(str).isin(UNRESOLVED).sum())
    if unresolved_entities < 2:
        empty = {name: frame.iloc[0:0].copy() for name, frame in axes.items()}
        ledger = pd.DataFrame(columns=[
            "block_id", "source_entity_id", "target_entity_id", "standardized_distance",
        ])
        return empty, ledger, False, f"need_at_least_two_unresolved_entities_observed_{unresolved_entities}"
    mapping, ledger = measurement_preserving_derangement(universe)
    return reassign_response_profiles(axes, mapping), ledger, True, ""


def reverse_summary(summary: dict[str, object]) -> dict[str, object]:
    result = dict(summary)
    for left_key, right_key in (
        ("left_entities", "right_entities"),
        ("fixed_capacity_left_total", "fixed_capacity_right_total"),
    ):
        result[left_key], result[right_key] = summary[right_key], summary[left_key]
    return result


def validate_pair_topology_lineage(
    pair: pd.Series,
    left_def: pd.DataFrame,
    right_def: pd.DataFrame,
) -> None:
    if str(pair.left_panel_id) == str(pair.right_panel_id):
        raise RuntimeError("pair cannot compare a panel with itself")
    if str(pair.left_accession) == str(pair.right_accession):
        raise RuntimeError("pair is not dataset-accession disjoint")
    if str(pair.left_source_family) == str(pair.right_source_family):
        raise RuntimeError("pair is not source-family disjoint")
    for side, definitions in (("left", left_def), ("right", right_def)):
        if definitions.contrast_id.astype(str).duplicated().any():
            raise RuntimeError(f"duplicate topology contrast IDs for {side} panel")
        expected = {
            "panel_id": str(pair[f"{side}_panel_id"]),
            "dataset_accession": str(pair[f"{side}_accession"]),
            "preliminary_source_family": str(pair[f"{side}_source_family"]),
        }
        for column, value in expected.items():
            observed = set(definitions[column].astype(str))
            if observed != {value}:
                raise RuntimeError(
                    f"pair/topology {column} drift for {side}: expected {value!r}, "
                    f"observed {sorted(observed)!r}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--pair-index", required=True, type=int)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--panel-metadata", required=True, type=Path)
    parser.add_argument("--effects-root", required=True, type=Path)
    parser.add_argument("--embeddings-root", required=True, type=Path)
    parser.add_argument("--families-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--identity-table", type=Path)
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--minimum-similarity", type=float, default=0.70)
    parser.add_argument("--unmatched-cost", type=float, default=0.20)
    parser.add_argument("--minimum-entities", type=int, default=12)
    parser.add_argument("--measurement-match-caliper", type=float, default=1.0)
    parser.add_argument("--minimum-matched-entities", type=int, default=10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.neighbors < 1 or not (0 <= args.minimum_similarity <= 1):
        raise ValueError("invalid chemical-edge parameters")
    if not (0 < args.unmatched_cost <= 1):
        raise ValueError("unmatched cost must lie in (0,1]")
    if args.minimum_entities < 1 or args.minimum_matched_entities < 1:
        raise ValueError("minimum entity counts must be positive")
    pairs = pd.read_csv(args.pair_manifest, sep="\t", dtype=str)
    required_pair_columns = {
        "pair_index", "left_panel_id", "right_panel_id",
        "left_accession", "right_accession",
        "left_source_family", "right_source_family",
        "left_polarity", "right_polarity",
    }
    if missing := required_pair_columns - set(pairs):
        raise ValueError(f"pair manifest lacks columns: {sorted(missing)}")
    pair_manifest_sha, topology_sha, panel_metadata_sha, pair_manifest_report_sha = (
        validate_frozen_pair_manifest(
            args.pair_manifest, args.topology, args.panel_metadata, pairs,
        )
    )
    implementation_sha256 = local_source_hashes()
    selected = pairs.loc[pairs.pair_index.astype(int).eq(args.pair_index)]
    if len(selected) != 1:
        raise ValueError(f"pair index {args.pair_index} is absent or duplicated")
    pair = selected.iloc[0]
    pair_polarities = {str(pair.left_polarity), str(pair.right_polarity)}
    if pair_polarities != {"positive"}:
        raise RuntimeError(
            "current identity and ion-family contracts are positive-mode only; "
            f"refusing pair polarities {sorted(pair_polarities)}"
        )
    panel_metadata = pd.read_csv(
        args.panel_metadata, sep="\t", dtype=str, usecols=["panel_id", "polarity"],
    )
    panel_metadata["canonical_polarity"] = panel_metadata.polarity.map(canonical_polarity)
    panel_polarity = panel_metadata[["panel_id", "canonical_polarity"]].drop_duplicates()
    if panel_polarity.groupby("panel_id").canonical_polarity.nunique().max() != 1:
        raise RuntimeError("a panel maps to multiple polarities")
    observed_polarity = panel_polarity.set_index("panel_id").canonical_polarity.to_dict()
    for side in ("left", "right"):
        panel_id = str(pair[f"{side}_panel_id"])
        if observed_polarity.get(panel_id) != str(pair[f"{side}_polarity"]):
            raise RuntimeError(f"pair-manifest polarity drift for {panel_id}")
    topology_columns = [
        "panel_id", "contrast_id", "dataset_accession", "preliminary_source_family",
        "contrast_field", "level_a", "level_b", "topology_status",
    ]
    topology = pd.read_csv(args.topology, sep="\t", dtype=str, usecols=topology_columns)
    topology = topology.loc[topology.topology_status.str.startswith("PASS")].copy()
    left_def = topology.loc[topology.panel_id.eq(pair.left_panel_id)].sort_values("contrast_id")
    right_def = topology.loc[topology.panel_id.eq(pair.right_panel_id)].sort_values("contrast_id")
    if left_def.empty or right_def.empty:
        raise RuntimeError("pair refers to a panel without a topology-pass contrast")
    validate_pair_topology_lineage(pair, left_def, right_def)
    identity = None
    identity_panel_lineage: dict[str, str] = {}
    identity_partition_sha256 = ""
    identity_report_sha256 = ""
    if args.identity_table:
        identity_report_path = args.identity_table.parent / "report.json"
        identity_report = json.loads(identity_report_path.read_text(encoding="utf-8"))
        if identity_report.get("status") != "PHENOTYPE_BLIND_STRICT_LIBRARY_PARTITIONS_COMPLETE":
            raise RuntimeError("combined identity partition is not frozen")
        identity_partition_sha256 = sha256(args.identity_table)
        identity_report_sha256 = sha256(identity_report_path)
        if identity_report.get("provenance", {}).get("combined_partition_sha256") != identity_partition_sha256:
            raise RuntimeError("combined identity partition checksum drift")
        for source in identity_report.get("provenance", {}).get("sources", []):
            source_panel = str(source.get("panel_id", ""))
            panel_lineage = str(source.get("panel_report_sha256", ""))
            if not source_panel or not panel_lineage or source_panel in identity_panel_lineage:
                raise RuntimeError("combined identity report has invalid panel lineage")
            identity_panel_lineage[source_panel] = panel_lineage
        identity = pd.read_csv(args.identity_table, sep="\t", dtype=str)
        required = {"panel_id", "anchor_id", "identity_status"}
        if missing := required - set(identity):
            raise ValueError(f"identity table lacks {sorted(missing)}")
        if identity.duplicated(["panel_id", "anchor_id"]).any():
            raise ValueError("identity table has duplicate panel/anchor rows")
        if not set(identity.identity_status).issubset(KNOWN | UNRESOLVED):
            raise ValueError("identity table contains unsupported states")
    parameters = {
        "neighbors_each_direction": args.neighbors,
        "minimum_similarity": args.minimum_similarity,
        "unmatched_cost": args.unmatched_cost,
        "minimum_entities": args.minimum_entities,
        "minimum_matched_mass_to_answer": 0.05,
        "answer_rule": "matched mass >= 0.05 and signed optimal-face interval excludes zero",
        "transport_plan": (
            "chemical-optimal then maximum-matched-mass face; signed lower/upper endpoint plans retained"
        ),
        "capacity_rule": (
            "each phenotype-blind ion family receives 1/F capacity on the complete measured "
            "panel universe and divides it over every family member before response QC"
        ),
        "unavailable_response_rule": (
            "retain entity and fixed capacity; set response_score to zero and report availability"
        ),
        "measurement_match_caliper": args.measurement_match_caliper,
        "minimum_matched_entities": args.minimum_matched_entities,
        "measurement_matching": (
            "one maximum-cardinality caliper matching frozen per full phenotype-blind panel universe; "
            "the same complete pair set is retained for every contrast and missing response evidence is zeroed"
        ),
        "deterministic_reassignment": (
            "left and right complete response profiles are independently moved among unresolved entities "
            "by a measurement-neighbour derangement; chemistry, entities and capacities remain fixed; "
            "this is a sensitivity stress, not a calibrated null"
        ),
    }
    execution_contract = {
        "pair_output_schema": PAIR_OUTPUT_SCHEMA,
        "parameters": parameters,
        "shared_input_sha256": {
            "pair_manifest": pair_manifest_sha,
            "pair_manifest_report": pair_manifest_report_sha,
            "topology": topology_sha,
            "panel_metadata": panel_metadata_sha,
            "identity_partition": identity_partition_sha256,
            "identity_report": identity_report_sha256,
        },
        "implementation_sha256": implementation_sha256,
    }
    execution_contract_sha256 = canonical_sha256(execution_contract)
    (
        left_ids, left_embedding, left_embedding_path, left_embedding_panel_sha,
        left_embedding_sha256,
    ) = load_embeddings(
        args.embeddings_root, pair.left_panel_id,
    )
    (
        right_ids, right_embedding, right_embedding_path, right_embedding_panel_sha,
        right_embedding_sha256,
    ) = load_embeddings(
        args.embeddings_root, pair.right_panel_id,
    )
    forward = directed_topk(
        left_ids, left_embedding, right_ids, right_embedding,
        neighbors=args.neighbors, minimum_similarity=args.minimum_similarity, reverse=False,
    )
    reverse = directed_topk(
        right_ids, right_embedding, left_ids, left_embedding,
        neighbors=args.neighbors, minimum_similarity=args.minimum_similarity, reverse=True,
    )
    edges = pd.concat([forward, reverse], ignore_index=True).sort_values(
        "chemical_similarity", ascending=False,
    ).drop_duplicates(["left_entity_id", "right_entity_id"]).sort_values(
        ["left_entity_id", "right_entity_id"], kind="stable",
    )
    edges["edge_evidence"] = "official_dreams_cross_panel_topk_union"
    left_embedding_frame = pd.DataFrame(left_embedding, index=left_ids.astype(str))
    right_embedding_frame = pd.DataFrame(right_embedding, index=right_ids.astype(str))
    (
        left_responses, left_universe, left_effect_path, left_family_path, left_covariate_path,
        left_effect_panel_sha, left_family_panel_sha, left_asset_sha256,
    ) = load_panel_responses(
        args.effects_root, args.families_root, pair.left_panel_id, left_def, identity,
    )
    (
        right_responses, right_universe, right_effect_path, right_family_path, right_covariate_path,
        right_effect_panel_sha, right_family_panel_sha, right_asset_sha256,
    ) = load_panel_responses(
        args.effects_root, args.families_root, pair.right_panel_id, right_def, identity,
    )
    for panel_id, hashes in (
        (pair.left_panel_id, {left_embedding_panel_sha, left_effect_panel_sha, left_family_panel_sha}),
        (pair.right_panel_id, {right_embedding_panel_sha, right_effect_panel_sha, right_family_panel_sha}),
    ):
        if identity is not None:
            hashes.add(identity_panel_lineage.get(str(panel_id), ""))
        if len(hashes) != 1:
            raise RuntimeError(f"panel-object provenance disagreement for {panel_id}: {sorted(hashes)}")
    if set(left_ids) != set(left_universe.entity_id) or set(right_ids) != set(right_universe.entity_id):
        raise RuntimeError("embedding/family entity universe mismatch")
    # Freeze ion-family capacity on every measured entity before opening effect
    # QC.  A failed effect estimate contributes response_score=0 while retaining
    # its phenotype-blind capacity and chemical edges.
    left_fixed = fixed_capacity_axes(left_universe, left_universe, left_responses)
    right_fixed = fixed_capacity_axes(right_universe, right_universe, right_responses)
    world_specs: dict[str, dict[str, object]] = {
        "all": {
            "left": left_fixed, "right": right_fixed,
            "evidence_scope": "all", "reassigned_panel_id": "",
        },
    }
    match_ledgers: list[pd.DataFrame] = []
    reassignment_ledger = pd.DataFrame()
    reassignment_by_side: dict[str, dict[str, object]] = {}
    if identity is not None:
        left_pairs = optimal_measurement_pairs(left_universe, args.measurement_match_caliper)
        right_pairs = optimal_measurement_pairs(right_universe, args.measurement_match_caliper)
        for side, panel_id, frozen_pairs in (
            ("left", pair.left_panel_id, left_pairs),
            ("right", pair.right_panel_id, right_pairs),
        ):
            ledger = frozen_pairs.copy()
            ledger.insert(0, "frozen_pair_id", [
                f"{side}_measurement_pair_{index:07d}" for index in range(len(ledger))
            ])
            ledger.insert(0, "panel_id", panel_id)
            ledger.insert(0, "side", side)
            ledger["frozen_before_response_qc"] = True
            match_ledgers.append(ledger)

        left_known = identity_axes_or_empty(left_fixed, KNOWN, renormalize=True)
        right_known = identity_axes_or_empty(right_fixed, KNOWN, renormalize=True)
        left_unresolved = identity_axes_or_empty(left_fixed, UNRESOLVED, renormalize=True)
        right_unresolved = identity_axes_or_empty(right_fixed, UNRESOLVED, renormalize=True)
        left_matched_known = measurement_axes_or_empty(left_fixed, left_pairs, member="known")
        right_matched_known = measurement_axes_or_empty(right_fixed, right_pairs, member="known")
        left_matched_unresolved = measurement_axes_or_empty(
            left_fixed, left_pairs, member="unresolved",
        )
        right_matched_unresolved = measurement_axes_or_empty(
            right_fixed, right_pairs, member="unresolved",
        )
        left_matched_known_common = measurement_axes_or_empty(
            left_fixed, left_pairs, member="known", require_pairwise_common_availability=True,
        )
        right_matched_known_common = measurement_axes_or_empty(
            right_fixed, right_pairs, member="known", require_pairwise_common_availability=True,
        )
        left_matched_unresolved_common = measurement_axes_or_empty(
            left_fixed, left_pairs, member="unresolved", require_pairwise_common_availability=True,
        )
        right_matched_unresolved_common = measurement_axes_or_empty(
            right_fixed, right_pairs, member="unresolved", require_pairwise_common_availability=True,
        )

        left_reassigned, left_reassignment, left_reassignment_available, left_reassignment_reason = (
            reassignment_axes_or_empty(left_universe, left_fixed)
        )
        right_reassigned, right_reassignment, right_reassignment_available, right_reassignment_reason = (
            reassignment_axes_or_empty(right_universe, right_fixed)
        )
        for side, panel_id, universe, ledger, available, reason in (
            (
                "left", pair.left_panel_id, left_universe, left_reassignment,
                left_reassignment_available, left_reassignment_reason,
            ),
            (
                "right", pair.right_panel_id, right_universe, right_reassignment,
                right_reassignment_available, right_reassignment_reason,
            ),
        ):
            ledger.insert(0, "panel_id", panel_id)
            ledger.insert(0, "reassigned_side", side)
            ledger["full_phenotype_blind_universe_size"] = len(universe)
            ledger["reassignment_available"] = available
            ledger["unavailable_reason"] = reason
            reassignment_by_side[side] = {
                "panel_id": str(panel_id),
                "unresolved_entities": int(
                    universe.identity_status.astype(str).isin(UNRESOLVED).sum()
                ),
                "reassignment_available": bool(available),
                "unavailable_reason": reason,
            }
        reassignment_ledger = pd.concat(
            [left_reassignment, right_reassignment], ignore_index=True,
        )
        left_matched_reassigned = measurement_axes_or_empty(
            left_reassigned, left_pairs, member="unresolved",
        )
        right_matched_reassigned = measurement_axes_or_empty(
            right_reassigned, right_pairs, member="unresolved",
        )

        world_specs.update({
            "known_only": {
                "left": left_known, "right": right_known,
                "evidence_scope": "all", "reassigned_panel_id": "",
            },
            "known_on_all_common_face": {
                "left": left_fixed, "right": right_fixed,
                "evidence_scope": "known_known", "reassigned_panel_id": "",
            },
            "unknown_involving_on_all_common_face": {
                "left": left_fixed, "right": right_fixed,
                "evidence_scope": "unknown_involving", "reassigned_panel_id": "",
            },
            "measurement_matched_known": {
                "left": left_matched_known, "right": right_matched_known,
                "evidence_scope": "all", "reassigned_panel_id": "",
            },
            "measurement_matched_unresolved": {
                "left": left_matched_unresolved, "right": right_matched_unresolved,
                "evidence_scope": "all", "reassigned_panel_id": "",
            },
            "measurement_matched_known_common_availability": {
                "left": left_matched_known_common, "right": right_matched_known_common,
                "evidence_scope": "all", "reassigned_panel_id": "",
            },
            "measurement_matched_unresolved_common_availability": {
                "left": left_matched_unresolved_common,
                "right": right_matched_unresolved_common,
                "evidence_scope": "all", "reassigned_panel_id": "",
            },
            "unresolved_only": {
                "left": left_unresolved, "right": right_unresolved,
                "evidence_scope": "all", "reassigned_panel_id": "",
            },
            "unresolved_response_reassigned_left": {
                "left": left_reassigned, "right": right_fixed,
                "evidence_scope": "all", "reassigned_panel_id": str(pair.left_panel_id),
            },
            "unresolved_response_reassigned_right": {
                "left": left_fixed, "right": right_reassigned,
                "evidence_scope": "all", "reassigned_panel_id": str(pair.right_panel_id),
            },
            "measurement_matched_unresolved_response_reassigned_left": {
                "left": left_matched_reassigned, "right": right_matched_unresolved,
                "evidence_scope": "all", "reassigned_panel_id": str(pair.left_panel_id),
            },
            "measurement_matched_unresolved_response_reassigned_right": {
                "left": left_matched_unresolved, "right": right_matched_reassigned,
                "evidence_scope": "all", "reassigned_panel_id": str(pair.right_panel_id),
            },
        })
    worlds = list(world_specs)
    rows: list[dict[str, object]] = []
    contribution_tables: list[pd.DataFrame] = []
    for left_definition in left_def.itertuples(index=False):
        for right_definition in right_def.itertuples(index=False):
            for world_name in worlds:
                spec = world_specs[world_name]
                left = spec["left"][str(left_definition.contrast_id)]
                right = spec["right"][str(right_definition.contrast_id)]
                evidence_scope = str(spec["evidence_scope"])
                minimum_entities = (
                    args.minimum_matched_entities
                    if world_name.startswith("measurement_matched_")
                    else args.minimum_entities
                )
                if len(left) < minimum_entities or len(right) < minimum_entities:
                    left_capacity_total = float(left.fixed_capacity.sum())
                    right_capacity_total = float(right.fixed_capacity.sum())
                    summary: dict[str, object] = {
                        "left_entities": len(left), "right_entities": len(right),
                        "eligible_chemical_edges": 0, "transported_edges": 0,
                        "transported_endpoint_rows": 0,
                        "matched_mass": 0.0,
                        "unmatched_mass_mean": (
                            left_capacity_total + right_capacity_total
                        ) / 2.0,
                        "concordant_mass": 0.0, "discordant_mass": 0.0,
                        "signed_chemical_evidence": 0.0,
                        "signed_chemical_evidence_conservative_bound": 0.0,
                        "signed_chemical_evidence_lower": 0.0,
                        "signed_chemical_evidence_upper": 0.0,
                        "signed_chemical_evidence_midpoint": 0.0,
                        "signed_chemical_evidence_interval_width": 0.0,
                        "direction_identified": False,
                        "unsigned_chemical_evidence": 0.0,
                        "unsigned_chemical_evidence_lower_endpoint": 0.0,
                        "unsigned_chemical_evidence_upper_endpoint": 0.0,
                        "matched_mass_lower_endpoint": 0.0,
                        "matched_mass_upper_endpoint": 0.0,
                        "fixed_capacity_left_total": left_capacity_total,
                        "fixed_capacity_right_total": right_capacity_total,
                        "evidence_scope": evidence_scope,
                        "transport_objective": np.nan,
                        "chemical_optimality_tolerance": 0.0,
                        "matched_mass_optimality_tolerance": 0.0,
                        "optimal_face_marginal_residual_max": 0.0,
                    }
                    used = pd.DataFrame()
                    chemical_support = False
                    direction_answerable = False
                    answerable = False
                else:
                    summary, used, _ = fixed_capacity_transport_score(
                        left, right, edges,
                        minimum_similarity=args.minimum_similarity, unmatched_cost=args.unmatched_cost,
                        evidence_scope=evidence_scope,
                    )
                    chemical_support = bool(summary["matched_mass"] >= 0.05)
                    direction_answerable = bool(summary["direction_identified"])
                    answerable = chemical_support and direction_answerable
                greedy, _ = deterministic_greedy_capacity_score(
                    left, right, edges,
                    minimum_similarity=args.minimum_similarity,
                    evidence_scope=evidence_scope,
                )
                centroid_scope_supported = evidence_scope == "all"
                if centroid_scope_supported:
                    centroid = signed_response_centroid_cosine(
                        left, right, left_embedding_frame, right_embedding_frame,
                    )
                    centroid_score = float(centroid["signed_response_centroid_cosine"])
                    centroid_available = bool(centroid["intrinsic_score_available"])
                    centroid_left_norm = float(centroid["left_centroid_norm"])
                    centroid_right_norm = float(centroid["right_centroid_norm"])
                else:
                    centroid_score = 0.0
                    centroid_available = False
                    centroid_left_norm = 0.0
                    centroid_right_norm = 0.0
                left_availability = response_availability_summary(left)
                right_availability = response_availability_summary(right)
                metadata = {
                    "pair_index": args.pair_index, "world": world_name,
                    "left_panel_id": pair.left_panel_id, "right_panel_id": pair.right_panel_id,
                    "left_contrast_id": str(left_definition.contrast_id),
                    "right_contrast_id": str(right_definition.contrast_id),
                    "left_contrast_field": str(left_definition.contrast_field),
                    "right_contrast_field": str(right_definition.contrast_field),
                    "left_level_a": str(left_definition.level_a), "left_level_b": str(left_definition.level_b),
                    "right_level_a": str(right_definition.level_a), "right_level_b": str(right_definition.level_b),
                    "reassigned_panel_id": str(spec["reassigned_panel_id"]),
                    "chemical_support": chemical_support,
                    "direction_answerable": direction_answerable,
                    "answerable": answerable,
                    "greedy_signed_chemical_evidence": float(
                        greedy["signed_chemical_evidence"]
                    ),
                    "greedy_matched_mass": float(greedy["matched_mass"]),
                    "greedy_intrinsic_score_available": bool(
                        greedy["intrinsic_score_available"]
                    ),
                    "centroid_cosine": centroid_score,
                    "centroid_intrinsic_score_available": centroid_available,
                    "centroid_scope_supported": centroid_scope_supported,
                    "centroid_left_norm": centroid_left_norm,
                    "centroid_right_norm": centroid_right_norm,
                    **{f"left_response_{key}": value for key, value in left_availability.items()},
                    **{f"right_response_{key}": value for key, value in right_availability.items()},
                }
                rows.append({"query_side": "left", **metadata, **summary})
                reversed_metadata = {
                    **metadata,
                    "world": REVERSED_WORLD.get(world_name, world_name),
                    "left_panel_id": pair.right_panel_id, "right_panel_id": pair.left_panel_id,
                    "left_contrast_id": str(right_definition.contrast_id),
                    "right_contrast_id": str(left_definition.contrast_id),
                    "left_contrast_field": str(right_definition.contrast_field),
                    "right_contrast_field": str(left_definition.contrast_field),
                    "left_level_a": str(right_definition.level_a), "left_level_b": str(right_definition.level_b),
                    "right_level_a": str(left_definition.level_a), "right_level_b": str(left_definition.level_b),
                }
                for key in left_availability:
                    reversed_metadata[f"left_response_{key}"] = right_availability[key]
                    reversed_metadata[f"right_response_{key}"] = left_availability[key]
                reversed_metadata["centroid_left_norm"] = centroid_right_norm
                reversed_metadata["centroid_right_norm"] = centroid_left_norm
                rows.append({"query_side": "right", **reversed_metadata, **reverse_summary(summary)})
                if not used.empty:
                    used.insert(0, "world", world_name)
                    used.insert(0, "right_contrast_id", str(right_definition.contrast_id))
                    used.insert(0, "left_contrast_id", str(left_definition.contrast_id))
                    contribution_tables.append(used)
    args.output_dir.mkdir(parents=True)
    edge_path = args.output_dir / "phenotype_blind_chemical_edges.tsv.gz"
    score_path = args.output_dir / "ordered_contrast_scores.tsv.gz"
    contribution_path = args.output_dir / "transported_contributions.tsv.gz"
    match_path = args.output_dir / "measurement_match_ledger.tsv.gz"
    reassignment_path = args.output_dir / "unresolved_response_profile_reassignment.tsv.gz"
    edges.to_csv(edge_path, sep="\t", index=False, compression="gzip")
    pd.DataFrame(rows).to_csv(score_path, sep="\t", index=False, compression="gzip")
    contributions = pd.concat(contribution_tables, ignore_index=True) if contribution_tables else pd.DataFrame()
    contributions.to_csv(contribution_path, sep="\t", index=False, compression="gzip")
    matches = pd.concat(match_ledgers, ignore_index=True) if match_ledgers else pd.DataFrame()
    matches.to_csv(match_path, sep="\t", index=False, compression="gzip")
    reassignment_ledger.to_csv(reassignment_path, sep="\t", index=False, compression="gzip")
    panel_assets = {
        str(pair.left_panel_id): {
            "panel_report_sha256": left_embedding_panel_sha,
            "embeddings_sha256": left_embedding_sha256,
            **left_asset_sha256,
        },
        str(pair.right_panel_id): {
            "panel_report_sha256": right_embedding_panel_sha,
            "embeddings_sha256": right_embedding_sha256,
            **right_asset_sha256,
        },
    }
    report = {
        "schema": PAIR_OUTPUT_SCHEMA,
        "status": "DEVELOPMENT_PRETRUTH_PARTIAL_RESPONSE_TRANSPORT_COMPLETE",
        "pair": pair.to_dict(),
        "denominators": {
            "left_embedded_entities": len(left_ids), "right_embedded_entities": len(right_ids),
            "left_phenotype_blind_universe_entities": len(left_universe),
            "right_phenotype_blind_universe_entities": len(right_universe),
            "phenotype_blind_edges": len(edges), "left_contrasts": len(left_def),
            "right_contrasts": len(right_def), "worlds": len(worlds),
            "ordered_contrast_scores": len(rows), "answerable_scores": int(pd.DataFrame(rows).answerable.sum()),
        },
        "parameters": parameters,
        "identity_partition_available": identity is not None,
        "reassignment_by_side": reassignment_by_side,
        "execution_contract": execution_contract,
        "execution_contract_sha256": execution_contract_sha256,
        "panel_assets": panel_assets,
        "leakage_status": (
            "the chemical graph, ion-family capacities, measurement matching and response-profile "
            "reassignment use full phenotype-blind panel universes before contrast effect-QC; "
            "response scoring then reads effects; "
            "topology defines requested axes but no cross-study relevance truth or retrieval score is read"
        ),
        "provenance": {
            "pair_manifest_sha256": pair_manifest_sha,
            "pair_manifest_report_sha256": pair_manifest_report_sha,
            "topology_sha256": topology_sha,
            "panel_metadata_sha256": panel_metadata_sha,
            "identity_partition_sha256": identity_partition_sha256,
            "identity_report_sha256": identity_report_sha256,
            "implementation_sha256": implementation_sha256,
            "left_embeddings_sha256": left_embedding_sha256,
            "right_embeddings_sha256": right_embedding_sha256,
            "left_effects_sha256": left_asset_sha256["effects_sha256"],
            "right_effects_sha256": right_asset_sha256["effects_sha256"],
            "left_effect_report_sha256": left_asset_sha256["effect_report_sha256"],
            "right_effect_report_sha256": right_asset_sha256["effect_report_sha256"],
            "left_effect_schema": left_asset_sha256["effect_schema"],
            "right_effect_schema": right_asset_sha256["effect_schema"],
            "left_effect_status": left_asset_sha256["effect_status"],
            "right_effect_status": right_asset_sha256["effect_status"],
            "left_families_sha256": left_asset_sha256["families_sha256"],
            "right_families_sha256": right_asset_sha256["families_sha256"],
            "left_measurement_covariates_sha256": left_asset_sha256["measurement_covariates_sha256"],
            "right_measurement_covariates_sha256": right_asset_sha256["measurement_covariates_sha256"],
            "chemical_edges_sha256": sha256(edge_path), "scores_sha256": sha256(score_path),
            "contributions_sha256": sha256(contribution_path),
            "measurement_matches_sha256": sha256(match_path),
            "unresolved_response_profile_reassignment_sha256": sha256(reassignment_path),
        },
        "claim_boundary": [
            "This matrix is pre-truth development evidence, not a biological association claim.",
            "Without an identity partition only the all-entity world is available; no unknown-information claim is permitted.",
            "Common-face known and unknown bounds are separate extrema and are not assumed to be exactly additive on a degenerate face.",
            "Deterministic response-profile reassignment is sensitivity-only and cannot generate an inferential p-value.",
            "Sex, age or other ontology labels require design-native adjudication and biological-unit label nulls.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
