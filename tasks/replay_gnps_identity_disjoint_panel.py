#!/usr/bin/env python
"""Replay the sealed GNPS identity-disjoint panel from the frozen manifest.

The panel builder is deterministic given the manifest frame, its arguments and
the seed.  To prove the replay is faithful, the same code path is first rerun
for the formula-disjoint panel and compared array-by-array against the sealed
local artifact.  Only if that control matches exactly is the missing
identity-disjoint panel accepted and written to a new directory; the sealed
benchmark directory itself is never modified.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from build_gnps_gold_silver_10ppm_benchmark import (  # noqa: E402
    build_panel,
    expand_pair_ledger,
)

MANIFEST_DTYPES = {
    "row": np.int64,
    "inchikey": str,
    "ik14": str,
    "smiles": str,
    "formula": str,
    "exact_mass": float,
    "precursor_mz": float,
    "precursor_structure_ppm": float,
    "library_quality": np.int16,
    "quality_label": str,
    "n_peaks": np.int32,
    "spectrum_hash": str,
    "formula_disjoint": bool,
    "stereo_cross_field_consistent": bool,
    "filename": str,
    "spectrum_id": str,
    "usi": str,
    "instrument": str,
}


def replay_args(seed: int) -> argparse.Namespace:
    namespace = argparse.Namespace()
    namespace.ppm = 10.0
    namespace.queries_per_identity = 2
    namespace.references_per_identity = 3
    namespace.min_negative_identities = 1
    namespace.max_negative_identities = 200
    namespace.require_independent_positive = True
    namespace.seed = seed
    return namespace


def panels_equal(replayed: dict[str, np.ndarray], sealed: dict[str, np.ndarray]) -> list[str]:
    differences: list[str] = []
    keys = set(replayed) | set(sealed)
    if set(replayed) != set(sealed):
        differences.append(f"key sets differ: extra={sorted(set(replayed)-set(sealed))} missing={sorted(set(sealed)-set(replayed))}")
    for key in sorted(keys & set(replayed) & set(sealed)):
        left, right = replayed[key], sealed[key]
        if left.shape != right.shape:
            differences.append(f"{key}: shape {left.shape} != {right.shape}")
        elif left.dtype != right.dtype:
            if np.array_equal(left.astype(str), right.astype(str)):
                continue
            differences.append(f"{key}: dtype {left.dtype} != {right.dtype} and values differ")
        elif not np.array_equal(left, right):
            mismatches = int(np.sum(left != right)) if left.dtype != float else -1
            differences.append(f"{key}: values differ ({mismatches} cells)")
    return differences


def panel_stats(panel: dict[str, np.ndarray], pairs: dict[str, np.ndarray]) -> dict[str, int]:
    return {
        "queries": int(len(panel["query_row"])),
        "query_identities": int(len(np.unique(panel["query_ik14"]))),
        "query_formulas": int(len(np.unique(panel["query_formula"]))),
        "candidate_identities": int(len(np.unique(panel["molecule_ik14"]))),
        "candidate_molecules": int(len(panel["molecule_ik14"])),
        "candidate_spectra": int(len(panel["candidate_row"])),
        # declared positive_pairs counts query-reference pairs whose molecule
        # is the truth, i.e. one entry per positive pair in the pair ledger.
        "positive_pairs": int(pairs["label"].sum()),
        "near_queries": int(panel["near_query"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    panel_dir = args.panel_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite replay output: {output}")
    output.mkdir(parents=True)

    frame = pd.read_csv(panel_dir / "manifest.csv.gz", dtype=MANIFEST_DTYPES)
    report = json.loads((panel_dir / "report.json").read_text(encoding="utf-8"))

    namespace = replay_args(args.seed)

    formula_replay, _ = build_panel(frame, namespace, "formula_disjoint")
    formula_pairs = expand_pair_ledger(formula_replay)
    sealed_formula = dict(np.load(panel_dir / "panel_formula_disjoint.npz", allow_pickle=False))
    control_differences = panels_equal(formula_replay, sealed_formula)
    control_stats = panel_stats(formula_replay, formula_pairs)
    declared_formula = report.get("formula_disjoint", {})
    control_matches_declared = all(
        control_stats.get(key) == declared_formula.get(key)
        for key in control_stats
        if key in declared_formula
    )

    identity_replay, _ = build_panel(frame, namespace, "identity_disjoint")
    identity_pairs = expand_pair_ledger(identity_replay)
    identity_stats = panel_stats(identity_replay, identity_pairs)
    declared_identity = report.get("identity_disjoint", {})
    identity_matches_declared = all(
        identity_stats.get(key) == declared_identity.get(key)
        for key in identity_stats
        if key in declared_identity
    )

    faithful = not control_differences and control_matches_declared
    if faithful:
        np.savez_compressed(output / "panel_identity_disjoint.npz", **identity_replay)
        np.savez_compressed(output / "pairs_identity_disjoint.npz", **identity_pairs)

    payload = {
        "status": (
            "gnps_identity_panel_replay_faithful"
            if faithful
            else "gnps_identity_panel_replay_control_mismatch"
        ),
        "seed": args.seed,
        "manifest_rows": int(len(frame)),
        "formula_control": {
            "array_differences": control_differences,
            "replayed_stats": control_stats,
            "declared_stats": declared_formula,
            "matches_declared": bool(control_matches_declared),
        },
        "identity_replay": {
            "replayed_stats": identity_stats,
            "declared_stats": declared_identity,
            "matches_declared": bool(identity_matches_declared),
        },
        "written": bool(faithful),
        "note": (
            "Replay accepted only because the deterministic builder reproduced "
            "the sealed formula-disjoint panel exactly from the same manifest."
        ),
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        temporary = Path(handle.name)
    temporary.replace(output / "report.json")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    if not faithful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
