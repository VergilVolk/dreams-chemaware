#!/usr/bin/env python
"""Fail-closed cross-study entity/effect replication analysis.

The phenotype-blind link graph is frozen before validation effects are read.
For each discovery entity, weighted Bonferroni combines ambiguous candidate
links.  The pair null is a union of: not-the-same-entity, wrong effect
direction, or validation effect below a prespecified minimum magnitude.
The discovery and validation evidence are then combined as a partial
conjunction (maximum p-value) and adjusted across the full discovery universe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY_STATES = {
    "confirmed_known", "library_supported", "unresolved_entity",
    "artifact_or_redundant_ion",
}
KNOWN_STATES = {"confirmed_known", "library_supported"}
FORBIDDEN_LINK_COLUMNS = {
    "beta", "effect", "effect_size", "p_value", "p_two_sided", "phenotype",
    "outcome", "disease", "case_control", "group_label", "validation_label",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, required: set[str], label: str) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty {label}: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")
    return frame


def parse_bool(series: pd.Series, label: str) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not values.isin(mapping).all():
        bad = sorted(values[~values.isin(mapping)].unique().tolist())
        raise RuntimeError(f"{label} contains invalid booleans: {bad}")
    return values.map(mapping).astype(bool)


def validate_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if frame.duplicated(columns).any():
        raise RuntimeError(f"{label} contains duplicate keys: {columns}")


def normal_upper(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def effect_p_values(beta: float, se: float, direction: int, minimum: float) -> tuple[float, float]:
    if not math.isfinite(beta) or not math.isfinite(se) or se <= 0:
        return 1.0, 1.0
    directed = direction * beta
    direction_p = normal_upper(directed / se)
    minimum_p = normal_upper((directed - minimum) / se)
    return min(max(direction_p, 0.0), 1.0), min(max(minimum_p, 0.0), 1.0)


def discovery_effect_p_values(beta: float, se: float, minimum: float) -> tuple[float, float]:
    """Two-sided discovery tests because the observed sign defines validation direction."""
    if not math.isfinite(beta) or not math.isfinite(se) or se <= 0:
        return 1.0, 1.0
    direction_p = min(1.0, 2.0 * normal_upper(abs(beta) / se))
    minimum_p = min(1.0, 2.0 * normal_upper((abs(beta) - minimum) / se))
    return direction_p, minimum_p


def adjust_p(values: np.ndarray, method: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise RuntimeError("invalid p-values")
    n = len(values)
    if n == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    multiplier = float(np.sum(1.0 / np.arange(1, n + 1))) if method == "by" else 1.0
    adjusted_ranked = ranked * n * multiplier / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def validate_freeze(
    freeze_path: Path, discovery_manifest: Path, validation_manifest: Path, links_path: Path,
) -> dict:
    if not freeze_path.is_file():
        raise RuntimeError(f"missing link freeze manifest: {freeze_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("phenotype_blind") is not True:
        raise RuntimeError("link freeze must declare phenotype_blind=true")
    if freeze.get("validation_effects_unseen") is not True:
        raise RuntimeError("link freeze must declare validation_effects_unseen=true")
    expected = {
        "discovery_manifest_sha256": sha256_file(discovery_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "links_sha256": sha256_file(links_path),
    }
    for key, value in expected.items():
        if freeze.get(key) != value:
            raise RuntimeError(f"link freeze hash mismatch for {key}")
    return freeze


def run(args: argparse.Namespace) -> dict:
    if args.min_abs_effect <= 0:
        raise RuntimeError("--min-abs-effect must be strictly positive and prespecified")
    if not (0 < args.fdr < 1):
        raise RuntimeError("--fdr must lie in (0,1)")
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {args.output_dir}")

    discovery = read_csv(
        args.discovery_manifest, {"entity_id", "identity_status", "qc_pass"}, "discovery manifest",
    )
    validation = read_csv(
        args.validation_manifest, {"entity_id", "qc_pass"}, "validation manifest",
    )
    links = read_csv(
        args.links,
        {"discovery_entity_id", "validation_entity_id", "match_p", "match_weight"},
        "phenotype-blind links",
    )
    discovery_effects = read_csv(
        args.discovery_effects, {"entity_id", "beta", "se"}, "discovery effects",
    )
    validation_effects = read_csv(
        args.validation_effects, {"entity_id", "beta", "se"}, "validation effects",
    )

    forbidden = sorted(FORBIDDEN_LINK_COLUMNS & set(links.columns))
    if forbidden:
        raise RuntimeError(f"link table contains phenotype/effect columns: {forbidden}")
    validate_unique(discovery, ["entity_id"], "discovery manifest")
    validate_unique(validation, ["entity_id"], "validation manifest")
    validate_unique(links, ["discovery_entity_id", "validation_entity_id"], "link table")
    validate_unique(discovery_effects, ["entity_id"], "discovery effects")
    validate_unique(validation_effects, ["entity_id"], "validation effects")
    freeze = validate_freeze(
        args.link_freeze, args.discovery_manifest, args.validation_manifest, args.links,
    )

    discovery["qc_pass"] = parse_bool(discovery["qc_pass"], "discovery qc_pass")
    validation["qc_pass"] = parse_bool(validation["qc_pass"], "validation qc_pass")
    bad_states = sorted(set(discovery["identity_status"].astype(str)) - IDENTITY_STATES)
    if bad_states:
        raise RuntimeError(f"unrecognized identity_status values: {bad_states}")
    if (discovery["qc_pass"] & discovery["identity_status"].eq("artifact_or_redundant_ion")).any():
        raise RuntimeError("artifact_or_redundant_ion rows cannot have qc_pass=true")
    discovery = discovery.loc[discovery["qc_pass"]].copy()
    validation = validation.loc[validation["qc_pass"]].copy()
    if discovery.empty or validation.empty:
        raise RuntimeError("no QC-passed discovery or validation entities")

    discovery_ids = set(discovery["entity_id"].astype(str))
    validation_ids = set(validation["entity_id"].astype(str))
    links["discovery_entity_id"] = links["discovery_entity_id"].astype(str)
    links["validation_entity_id"] = links["validation_entity_id"].astype(str)
    if not set(links["discovery_entity_id"]).issubset(discovery_ids):
        raise RuntimeError("links contain discovery entities absent from the QC-passed manifest")
    if not set(links["validation_entity_id"]).issubset(validation_ids):
        raise RuntimeError("links contain validation entities absent from the QC-passed manifest")
    for column in ("match_p", "match_weight"):
        links[column] = pd.to_numeric(links[column], errors="raise")
    if ((links["match_p"] < 0) | (links["match_p"] > 1)).any():
        raise RuntimeError("match_p must lie in [0,1]")
    if ((links["match_weight"] <= 0) | (links["match_weight"] > 1)).any():
        raise RuntimeError("match_weight must lie in (0,1]")
    weight_sum = links.groupby("discovery_entity_id")["match_weight"].sum()
    if (weight_sum > 1 + 1e-10).any():
        raise RuntimeError("match weights must sum to <=1 for each discovery entity")

    for frame, label in ((discovery_effects, "discovery"), (validation_effects, "validation")):
        frame["entity_id"] = frame["entity_id"].astype(str)
        frame["beta"] = pd.to_numeric(frame["beta"], errors="raise")
        frame["se"] = pd.to_numeric(frame["se"], errors="raise")
        if (~np.isfinite(frame[["beta", "se"]].to_numpy(float))).any() or (frame["se"] <= 0).any():
            raise RuntimeError(f"{label} effects require finite beta and strictly positive se")
    discovery_effects = discovery_effects.set_index("entity_id")
    validation_effects = validation_effects.set_index("entity_id")
    if set(discovery["entity_id"].astype(str)) != set(discovery_effects.index):
        raise RuntimeError("discovery effects must exactly cover the QC-passed discovery universe")

    identity = discovery.set_index(discovery["entity_id"].astype(str))["identity_status"].astype(str)
    pair_rows: list[dict[str, object]] = []
    entity_rows: list[dict[str, object]] = []
    grouped_links = {key: value for key, value in links.groupby("discovery_entity_id", sort=False)}

    for discovery_id in discovery["entity_id"].astype(str):
        d_beta = float(discovery_effects.loc[discovery_id, "beta"])
        d_se = float(discovery_effects.loc[discovery_id, "se"])
        direction = 1 if d_beta > 0 else -1 if d_beta < 0 else 0
        if direction == 0:
            discovery_direction_p = discovery_minimum_p = 1.0
        else:
            discovery_direction_p, discovery_minimum_p = discovery_effect_p_values(
                d_beta, d_se, args.min_abs_effect,
            )
        candidates = grouped_links.get(discovery_id)
        weighted_pair_values: list[tuple[float, str]] = []
        if candidates is not None and direction != 0:
            for link in candidates.itertuples(index=False):
                validation_id = str(link.validation_entity_id)
                if validation_id not in validation_effects.index:
                    continue
                v_beta = float(validation_effects.loc[validation_id, "beta"])
                v_se = float(validation_effects.loc[validation_id, "se"])
                direction_p, minimum_p = effect_p_values(
                    v_beta, v_se, direction, args.min_abs_effect,
                )
                pair_p = max(float(link.match_p), direction_p, minimum_p)
                weighted_p = min(1.0, pair_p / float(link.match_weight))
                weighted_pair_values.append((weighted_p, validation_id))
                pair_rows.append({
                    "discovery_entity_id": discovery_id,
                    "validation_entity_id": validation_id,
                    "identity_status": identity.loc[discovery_id],
                    "discovery_beta": d_beta,
                    "discovery_se": d_se,
                    "validation_beta": v_beta,
                    "validation_se": v_se,
                    "effect_direction": direction,
                    "match_p": float(link.match_p),
                    "match_weight": float(link.match_weight),
                    "validation_direction_p": direction_p,
                    "validation_minimum_effect_p": minimum_p,
                    "pair_intersection_union_p": pair_p,
                    "weighted_pair_p": weighted_p,
                    "effect_difference_validation_minus_discovery": v_beta - d_beta,
                    "effect_difference_se": math.sqrt(d_se * d_se + v_se * v_se),
                })
        if weighted_pair_values:
            pair_union_p, selected_id = min(weighted_pair_values, key=lambda item: (item[0], item[1]))
            link_state = "candidate_set_evaluated"
        else:
            pair_union_p, selected_id = 1.0, ""
            link_state = "no_evaluable_link"
        replication_p = max(discovery_direction_p, discovery_minimum_p, pair_union_p)
        entity_rows.append({
            "entity_id": discovery_id,
            "identity_status": identity.loc[discovery_id],
            "discovery_beta": d_beta,
            "discovery_se": d_se,
            "effect_direction": direction,
            "discovery_direction_p": discovery_direction_p,
            "discovery_minimum_effect_p": discovery_minimum_p,
            "link_state": link_state,
            "n_candidate_links": 0 if candidates is None else int(len(candidates)),
            "selected_joint_candidate_for_audit": selected_id,
            "validation_candidate_union_p": pair_union_p,
            "replication_p": replication_p,
        })

    entities = pd.DataFrame(entity_rows)
    entities["replication_q_full"] = adjust_p(entities["replication_p"].to_numpy(float), args.adjustment)
    known_mask = entities["identity_status"].isin(KNOWN_STATES)
    entities["replication_q_known_only"] = np.nan
    entities.loc[known_mask, "replication_q_known_only"] = adjust_p(
        entities.loc[known_mask, "replication_p"].to_numpy(float), args.adjustment,
    )
    entities["replicated_full"] = entities["replication_q_full"] <= args.fdr
    entities["replicated_known_only"] = False
    entities.loc[known_mask, "replicated_known_only"] = (
        entities.loc[known_mask, "replication_q_known_only"] <= args.fdr
    )
    pairs = pd.DataFrame(pair_rows)

    args.output_dir.mkdir(parents=True)
    entities.to_csv(args.output_dir / "entity_replication_results.csv", index=False)
    pairs.to_csv(args.output_dir / "pair_replication_audit.csv", index=False)
    unresolved_mask = entities["identity_status"].eq("unresolved_entity")
    report = {
        "status": "ENTITY_EFFECT_TRANSPORT_COMPLETE",
        "analysis_role": args.analysis_role,
        "effect_scale": args.effect_scale,
        "minimum_absolute_effect": args.min_abs_effect,
        "adjustment": args.adjustment.upper(),
        "fdr": args.fdr,
        "universe": {
            "discovery_qc_entities": int(len(entities)),
            "validation_qc_entities": int(len(validation)),
            "candidate_links": int(len(links)),
            "evaluable_pairs": int(len(pairs)),
            "known_discovery_entities": int(known_mask.sum()),
            "unresolved_discovery_entities": int(unresolved_mask.sum()),
        },
        "outcomes": {
            "replicated_full_universe": int(entities["replicated_full"].sum()),
            "replicated_known_in_full_universe": int((entities["replicated_full"] & known_mask).sum()),
            "replicated_unresolved_in_full_universe": int((entities["replicated_full"] & unresolved_mask).sum()),
            "replicated_known_in_known_only_universe": int(entities["replicated_known_only"].sum()),
            "no_evaluable_link": int(entities["link_state"].eq("no_evaluable_link").sum()),
            "reverse_or_below_minimum_pairs": int(
                (pairs["validation_minimum_effect_p"] >= 0.5).sum() if len(pairs) else 0
            ),
        },
        "known_unknown_interpretation": (
            "Descriptive partition only. An unknown-entity increment claim additionally requires "
            "a matched known-only control and an independent biological or held-out functional endpoint."
        ),
        "truth_and_leakage": {
            "links_declared_phenotype_blind": True,
            "validation_effects_unseen_declared_in_freeze": True,
            "link_table_effect_columns_prohibited_by_schema": True,
            "link_freeze_note": freeze.get("note", ""),
        },
        "provenance": {
            "discovery_manifest_sha256": sha256_file(args.discovery_manifest),
            "validation_manifest_sha256": sha256_file(args.validation_manifest),
            "links_sha256": sha256_file(args.links),
            "link_freeze_sha256": sha256_file(args.link_freeze),
            "discovery_effects_sha256": sha256_file(args.discovery_effects),
            "validation_effects_sha256": sha256_file(args.validation_effects),
        },
        "claim_limit": (
            "A positive result supports cross-study replication of an unresolved analytical entity "
            "on the declared effect scale. It does not establish exact structure, source, flux, "
            "enzyme activity, causal function, or independent-patient replication unless the input "
            "design itself supplies those orthogonal conditions."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--link-freeze", type=Path, required=True)
    parser.add_argument("--discovery-effects", type=Path, required=True)
    parser.add_argument("--validation-effects", type=Path, required=True)
    parser.add_argument("--effect-scale", required=True)
    parser.add_argument("--min-abs-effect", type=float, required=True)
    parser.add_argument("--adjustment", choices=("by", "bh"), default="by")
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--analysis-role", choices=("development", "technical_replication", "independent_biology"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "outcomes": report["outcomes"]}, indent=2))


if __name__ == "__main__":
    main()

