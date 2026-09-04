"""Audit tolerance-rounded spectrum hashes that map to multiple identities.

The observability cohort hashes peaks after rounding m/z to 0.01 Da and
intensity to 1e-4.  A hash shared by different IK14 labels is therefore an
annotation/identifiability conflict under this cohort representation, not
proof that the original raw spectra are byte-identical.  This post-hoc audit
reports conflict prevalence and frozen-gate sensitivity after excluding query
groups containing a cross-identity same-hash negative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(ledger: pd.DataFrame, keep: np.ndarray) -> dict:
    frame = ledger.loc[keep].copy()
    old = frame["dreams_correct"].to_numpy(bool)
    new = old.copy()
    active = frame["route_activated"].to_numpy(bool)
    new[active] = frame.loc[active, "consensus_correct"].to_numpy(bool)
    corrected = (~old) & new
    introduced = old & (~new)
    return {
        "queries": int(len(frame)),
        "route_activated": int(active.sum()),
        "corrected": int(corrected.sum()),
        "introduced": int(introduced.sum()),
        "dreams_recall1": float(old.mean()),
        "routed_recall1": float(new.mean()),
        "delta_pp": float(100.0 * np.mean(new.astype(float) - old.astype(float))),
    }


def split_audit(
    pair_path: Path, manifest_path: Path, ledger_path: Path, split: str,
) -> tuple[dict, pd.DataFrame]:
    pairs = pd.read_csv(pair_path)
    manifest = pd.read_csv(manifest_path)
    ledger = pd.read_csv(ledger_path)
    hashes = manifest["spectrum_hash"].astype(str).to_numpy()
    identities = manifest["ik14"].astype(str).to_numpy()
    left = pairs["left"].to_numpy(np.int64)
    right = pairs["right"].to_numpy(np.int64)
    negative = pairs["label"].to_numpy(np.int8) == 0
    conflict_pair = negative & (hashes[left] == hashes[right]) & (identities[left] != identities[right])
    conflict_queries: set[int] = set()
    for a, b in zip(left[conflict_pair], right[conflict_pair]):
        conflict_queries.add(int(a))
        conflict_queries.add(int(b))
    affected = ledger["query_index"].astype(int).isin(conflict_queries).to_numpy()
    records = pairs.loc[conflict_pair, ["left", "right", "left_ik14", "right_ik14"]].copy()
    records.insert(0, "split", split)
    records["spectrum_hash"] = hashes[left[conflict_pair]]
    return ({
        "manifest_spectra": int(len(manifest)),
        "pair_edges": int(len(pairs)),
        "cross_identity_same_rounded_hash_negative_edges": int(conflict_pair.sum()),
        "affected_eligible_queries": int(affected.sum()),
        "affected_frozen_gate_activations": int(
            (affected & ledger["route_activated"].to_numpy(bool)).sum()
        ),
        "all_queries": evaluate(ledger, np.ones(len(ledger), dtype=bool)),
        "sensitivity_excluding_affected_queries": evaluate(ledger, ~affected),
    }, records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/chemaware_observability_label_conflicts_v2_20260903",
    )
    args = parser.parse_args()
    frozen = ROOT / "data/validation/chemaware_spectral_consensus_applicability_v4_frozen"
    residual = ROOT / "data/validation/large_observability_residual_audit"
    test_input = ROOT / "data/validation/chemaware_frozen_gate_test_inputs_20260902"
    test_output = ROOT / "data/validation/chemaware_frozen_spectral_gate_test_20260902"
    specifications = {
        "discovery": (
            residual / "discovery_pair_features.csv",
            ROOT / "data/validation/large_observability_embeddings_discovery/manifest.csv",
            frozen / "discovery_gate_ledger.csv.gz",
        ),
        "confirmation": (
            residual / "confirmation_pair_features.csv",
            ROOT / "data/validation/large_observability_embeddings_confirmation/manifest.csv",
            frozen / "confirmation_gate_ledger.csv.gz",
        ),
        "test": (
            test_input / "test_pair_features.csv.gz",
            test_input / "test_manifest.csv",
            test_output / "test_gate_ledger.csv.gz",
        ),
    }
    required = [path for values in specifications.values() for path in values]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")

    manifests = []
    for split, (_, manifest_path, _) in specifications.items():
        frame = pd.read_csv(manifest_path)
        frame.insert(0, "split", split)
        manifests.append(frame)
    union = pd.concat(manifests, ignore_index=True)
    grouped = union.groupby("spectrum_hash", sort=False).agg(
        rows=("ik14", "size"), identities=("ik14", "nunique"),
        formulas=("formula", "nunique"), splits=("split", "nunique"),
    )
    conflict_hashes = set(grouped.index[grouped["identities"] > 1].astype(str))
    cross_split_hashes = set(grouped.index[grouped["splits"] > 1].astype(str))
    conflict_rows = union.loc[union["spectrum_hash"].astype(str).isin(conflict_hashes)].copy()

    results = {}
    pair_records = []
    for split, paths in specifications.items():
        results[split], records = split_audit(*paths, split)
        pair_records.append(records)
    pairs_out = pd.concat(pair_records, ignore_index=True)
    args.output_dir.mkdir(parents=True)
    hashes_path = args.output_dir / "cross_identity_rounded_hash_rows.csv.gz"
    pairs_path = args.output_dir / "cross_identity_same_hash_negative_edges.csv.gz"
    conflict_rows.to_csv(hashes_path, index=False)
    pairs_out.to_csv(pairs_path, index=False)
    report = {
        "status": "chemaware_observability_label_conflicts_audited",
        "post_hoc_sensitivity_only": True,
        "hash_contract": {"mz_round_decimals": 2, "intensity_round_decimals": 4},
        "union": {
            "spectra": int(len(union)),
            "rounded_hashes": int(union["spectrum_hash"].nunique()),
            "cross_identity_rounded_hashes": int(len(conflict_hashes)),
            "rows_on_cross_identity_hashes": int(len(conflict_rows)),
            "cross_split_rounded_hashes": int(len(cross_split_hashes)),
            "cross_split_cross_identity_hashes": int(
                ((grouped["splits"] > 1) & (grouped["identities"] > 1)).sum()
            ),
            "cross_split_cross_formula_hashes": int(
                ((grouped["splits"] > 1) & (grouped["formulas"] > 1)).sum()
            ),
            "test_hash_overlap_with_discovery_or_confirmation": int(len(
                set(manifests[2]["spectrum_hash"].astype(str))
                & set(pd.concat(manifests[:2])["spectrum_hash"].astype(str))
            )),
            "all_cross_identity_hashes_single_formula": bool(
                (grouped.loc[list(conflict_hashes), "formulas"] == 1).all()
            ),
        },
        "splits": results,
        "interpretation_contract": (
            "Same rounded hash across identities is an ambiguity under the cohort peak "
            "representation, not proof of byte-identical raw spectra or a wrong database label. "
            "Such pairs must be excluded from future train/evaluation negatives."
        ),
        "provenance": {
            "inputs": {
                split: {
                    "pair_sha256": sha256(paths[0]),
                    "manifest_sha256": sha256(paths[1]),
                    "ledger_sha256": sha256(paths[2]),
                }
                for split, paths in specifications.items()
            },
            "conflict_rows_sha256": sha256(hashes_path),
            "conflict_pairs_sha256": sha256(pairs_path),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
