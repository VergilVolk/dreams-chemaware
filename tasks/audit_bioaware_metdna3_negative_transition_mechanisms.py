#!/usr/bin/env python
"""Audit correction and introduction mechanisms for negative BioAware.

This stage is descriptive.  It joins the frozen source-LOSO transitions to the
candidate-level network evidence and reports which evidence dimensions favored
the proposed candidate over the displaced DreaMS candidate.  It deliberately
does not tune a new gate from these outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "spectral_score",
    "known_mass_candidate_fraction",
    "known_path_fraction",
    "known_inverse_depth_mean",
    "known_log_seed_support_mean",
    "known_log_degree",
    "edge0_complete_fraction",
    "edge0_bottleneck_mean",
    "edge1_complete_fraction",
    "edge1_bottleneck_mean",
    "predicted_edge_increment",
]
NETWORK_FEATURES = FEATURES[1:]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_text(values: pd.Series) -> str:
    cleaned = [str(value).strip() for value in values if pd.notna(value) and str(value).strip()]
    return cleaned[0] if cleaned else ""


def molecule_lookup(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"inchikey", "name", "smiles"}
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(f"MONA manifest missing columns: {sorted(missing)}")
    manifest["candidate_id"] = manifest["inchikey"].astype(str).str[:14]
    return manifest.groupby("candidate_id", sort=False).agg(
        molecule_name=("name", first_text),
        molecule_smiles=("smiles", first_text),
        mona_reference_spectra=("candidate_id", "size"),
    ).reset_index()


def role_rows(candidates: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    indexed = candidates.set_index(["query_id", "candidate_id"], verify_integrity=True)
    rows: list[dict] = []
    for transition in transitions.itertuples(index=False):
        if not (bool(transition.corrected) or bool(transition.introduced)):
            continue
        record = transition._asdict()
        for role, candidate_id in (
            ("truth", str(transition.truth_candidate_id)),
            ("baseline", str(transition.baseline_candidate_id)),
            ("proposal", str(transition.proposed_candidate_id)),
        ):
            key = (str(transition.query_id), candidate_id)
            if key not in indexed.index:
                raise RuntimeError(f"missing {role} candidate evidence for {key}")
            evidence = indexed.loc[key]
            record[f"{role}_candidate_id"] = candidate_id
            for feature in FEATURES:
                record[f"{role}_{feature}"] = float(evidence[feature])
            for metadata in ("best_library_row", "reference_spectra"):
                if metadata in evidence.index:
                    record[f"{role}_{metadata}"] = int(evidence[metadata])
        for feature in FEATURES:
            record[f"proposal_minus_baseline__{feature}"] = (
                record[f"proposal_{feature}"] - record[f"baseline_{feature}"]
            )
            record[f"proposal_minus_truth__{feature}"] = (
                record[f"proposal_{feature}"] - record[f"truth_{feature}"]
            )
        # These are descriptive comparisons, not causal mechanism labels.
        record["proposal_has_more_path_coverage"] = bool(
            record["proposal_minus_baseline__known_path_fraction"] > 0
        )
        record["proposal_has_more_seed_support"] = bool(
            record["proposal_minus_baseline__known_log_seed_support_mean"] > 0
        )
        record["proposal_has_higher_degree"] = bool(
            record["proposal_minus_baseline__known_log_degree"] > 0
        )
        record["proposal_has_more_raw_edge_coverage"] = bool(
            record["proposal_minus_baseline__edge0_complete_fraction"] > 0
        )
        record["proposal_has_stronger_raw_bottleneck"] = bool(
            record["proposal_minus_baseline__edge0_bottleneck_mean"] > 0
        )
        record["proposal_supported_while_baseline_has_no_path"] = bool(
            record["proposal_known_path_fraction"] > 0
            and record["baseline_known_path_fraction"] == 0
        )
        record["proposal_raw_edge_while_baseline_has_none"] = bool(
            record["proposal_edge0_complete_fraction"] > 0
            and record["baseline_edge0_complete_fraction"] == 0
        )
        rows.append(record)
    result = pd.DataFrame(rows)
    expected = int(transitions["corrected"].sum() + transitions["introduced"].sum())
    if len(result) != expected:
        raise RuntimeError(f"expected {expected} changed queries, got {len(result)}")
    return result


def transition_summary(frame: pd.DataFrame) -> dict:
    summaries: dict[str, dict] = {}
    flag_columns = [
        "proposal_has_more_path_coverage",
        "proposal_has_more_seed_support",
        "proposal_has_higher_degree",
        "proposal_has_more_raw_edge_coverage",
        "proposal_has_stronger_raw_bottleneck",
        "proposal_supported_while_baseline_has_no_path",
        "proposal_raw_edge_while_baseline_has_none",
    ]
    for label, local in (
        ("corrected", frame[frame["corrected"].astype(bool)]),
        ("introduced", frame[frame["introduced"].astype(bool)]),
    ):
        feature_delta = {}
        for feature in NETWORK_FEATURES:
            values = local[f"proposal_minus_baseline__{feature}"].to_numpy(float)
            feature_delta[feature] = (
                {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "positive_fraction": float(np.mean(values > 0)),
                }
                if len(values)
                else {"mean": None, "median": None, "positive_fraction": None}
            )
        summaries[label] = {
            "queries": int(len(local)),
            "identities": int(local["truth_candidate_id"].nunique()),
            "formulas": int(local["truth_formula"].nunique()),
            "sources": sorted(local["unit_id"].astype(str).unique()),
            "descriptive_flag_counts": {
                column: int(local[column].astype(bool).sum()) for column in flag_columns
            },
            "proposal_minus_baseline_network_feature_deltas": feature_delta,
        }
    return summaries


def repeated_pairs(frame: pd.DataFrame) -> list[dict]:
    grouped = frame.groupby(
        ["truth_candidate_id", "baseline_candidate_id", "proposed_candidate_id", "corrected", "introduced"],
        dropna=False,
    ).agg(
        queries=("query_id", "size"),
        sources=("unit_id", lambda values: sorted(set(map(str, values)))),
        query_ids=("query_id", lambda values: sorted(map(str, values))),
    ).reset_index()
    grouped = grouped[grouped["queries"] > 1].sort_values(
        ["introduced", "corrected", "queries"], ascending=[False, False, False]
    )
    return grouped.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-features", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_loso_ranker_v3_identity_purged/candidate_features.csv.gz"),
    )
    parser.add_argument(
        "--transitions", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_source_loso_v1/source_identity_formula_purged__transitions.csv.gz"),
    )
    parser.add_argument(
        "--mona-manifest", type=Path,
        default=Path("data/models/mona_neg_dreams_emb/manifest.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_negative_transition_mechanisms_v1"),
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")

    candidates = pd.read_csv(args.candidate_features)
    transitions = pd.read_csv(args.transitions)
    required_features = {"query_id", "candidate_id", *FEATURES}
    if missing := required_features - set(candidates.columns):
        raise RuntimeError(f"candidate table missing columns: {sorted(missing)}")
    changed = role_rows(candidates, transitions)
    manifest = pd.read_csv(args.mona_manifest).reset_index(names="library_row")
    for role in ("truth", "baseline", "proposal"):
        row_column = f"{role}_best_library_row"
        if row_column in changed.columns:
            lookup = manifest[["library_row", "name", "smiles", "inchikey", "precursor_mz"]].rename(columns={
                "library_row": row_column,
                "name": f"{role}_name",
                "smiles": f"{role}_smiles",
                "inchikey": f"{role}_manifest_inchikey",
                "precursor_mz": f"{role}_library_precursor_mz",
            })
            changed = changed.merge(lookup, on=row_column, how="left", validate="many_to_one")
        else:  # pragma: no cover - retained for legacy feature tables
            molecules = molecule_lookup(args.mona_manifest).rename(columns={
                "candidate_id": f"{role}_candidate_id",
                "molecule_name": f"{role}_name",
                "molecule_smiles": f"{role}_smiles",
                "mona_reference_spectra": f"{role}_mona_reference_spectra",
            })
            changed = changed.merge(
                molecules, on=f"{role}_candidate_id", how="left", validate="many_to_one"
            )

    introduced_pairs = repeated_pairs(changed[changed["introduced"].astype(bool)])
    corrected_pairs = repeated_pairs(changed[changed["corrected"].astype(bool)])
    report = {
        "status": "bioaware_metdna3_negative_transition_mechanism_audit_complete",
        "formal": True,
        "protocol": "source-LOSO plus identity-and-formula-purged transitions; exact candidate evidence join; no threshold fitting",
        "changed_queries": int(len(changed)),
        "corrected": int(changed["corrected"].sum()),
        "introduced": int(changed["introduced"].sum()),
        "summary": transition_summary(changed),
        "repeated_introduced_candidate_pairs": introduced_pairs,
        "repeated_corrected_candidate_pairs": corrected_pairs,
        "contracts": {
            "new_gate_or_threshold_fitted": False,
            "mechanism_flags_are_causal_labels": False,
            "candidate_feature_assignment_changed": False,
            "P2b": "forbidden",
            "shared_embedding_changed": False,
        },
        "provenance": {
            "candidate_features_sha256": sha256(args.candidate_features),
            "transitions_sha256": sha256(args.transitions),
            "mona_manifest_sha256": sha256(args.mona_manifest),
        },
        "claim_limit": "Feature dominance is descriptive of changed rankings. It does not establish a biochemical causal mechanism and is not an independent performance result.",
    }
    args.output_dir.mkdir(parents=True)
    csv_path = args.output_dir / "changed_query_mechanisms.csv.gz"
    changed.to_csv(csv_path, index=False, compression="gzip")
    report["provenance"]["changed_query_mechanisms_sha256"] = sha256(csv_path)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
