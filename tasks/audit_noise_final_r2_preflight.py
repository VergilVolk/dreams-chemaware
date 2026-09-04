"""Fail-closed data and realizability audit before R2 GPU training."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import CandidateGraph, json_dump, sha256_file  # noqa: E402
from noise_v3_core import attenuate_sequence  # noqa: E402
from train_e1_identity import preprocess_spectrum  # noqa: E402
from train_noise_final_r2_shared_encoder import parse_controls, parse_path  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--r1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r1_privileged_teacher")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r2_preflight")
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def replay(frame: pd.DataFrame, handle: h5py.File, n_highest: int) -> dict:
    target_count = control_count = 0
    for row in frame.itertuples(index=False):
        clean = preprocess_spectrum(
            np.asarray(handle["spectrum"][int(row.query_row)]),
            float(handle["precursor_mz"][int(row.query_row)]), n_highest,
        )
        target = parse_path(row.target_path)
        if not target:
            raise RuntimeError(f"empty target path at query {row.query_index}")
        changed = attenuate_sequence(clean, target, float(row.attenuation))
        if np.allclose(changed.numpy(), clean.numpy()):
            raise RuntimeError(f"no-op target action at query {row.query_index}")
        target_count += 1
        for control in parse_controls(row.matched_control_paths):
            attenuate_sequence(clean, control, float(row.attenuation))
            control_count += 1
    return {"target_views_replayed": target_count, "control_views_replayed": control_count}


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite R2 preflight: {args.output_dir}")
    required = [
        args.graph, args.data, args.r1_dir / "report.json",
        args.r1_dir / "corrective_teacher_actions.csv.gz",
        args.r1_dir / "robustness_teacher_actions.csv.gz",
        args.r1_dir / "query_ledger.csv.gz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    graph = CandidateGraph(args.graph)
    report = json.loads((args.r1_dir / "report.json").read_text(encoding="utf-8"))
    corrective = pd.read_csv(args.r1_dir / "corrective_teacher_actions.csv.gz")
    robust = pd.read_csv(args.r1_dir / "robustness_teacher_actions.csv.gz")
    ledger = pd.read_csv(args.r1_dir / "query_ledger.csv.gz")
    if report.get("locally_materialised_union_recoverable") != 882 or len(corrective) != 882:
        raise RuntimeError("R2 preflight requires all 882 locally materialised corrective actions")
    if report["contracts"].get("P2b") != "forbidden" or len(ledger) != graph.n_queries:
        raise RuntimeError("R2 preflight crossed a frozen contract")
    if corrective["query_index"].duplicated().any():
        raise RuntimeError("corrective teacher is not one action per query")
    if not corrective["baseline_rank"].astype(int).gt(1).all():
        raise RuntimeError("corrective teacher contains an official-correct query")

    with h5py.File(args.data, "r") as handle:
        corrective_replay = replay(corrective, handle, args.n_highest_peaks)
        robustness_replay = replay(robust, handle, args.n_highest_peaks)

    folds = []
    for fold in range(5):
        held_ledger = ledger.loc[ledger["formula_fold"].astype(int).eq(fold)]
        held = corrective.loc[corrective["formula_fold"].astype(int).eq(fold)]
        train = corrective.loc[corrective["formula_fold"].astype(int).ne(fold)]
        if set(train["query_formula"].astype(str)) & set(held_ledger["query_formula"].astype(str)):
            raise RuntimeError(f"formula leakage in fold {fold}")
        folds.append({
            "fold": fold,
            "held_queries": int(len(held_ledger)),
            "held_formulas": int(held_ledger["query_formula"].nunique()),
            "held_corrective_queries": int(len(held)),
            "held_corrective_identities": int(held["query_ik14"].nunique()),
            "held_corrective_formulas": int(held["query_formula"].nunique()),
            "train_corrective_queries": int(len(train)),
            "train_corrective_identities": int(train["query_ik14"].nunique()),
            "train_corrective_formulas": int(train["query_formula"].nunique()),
            "held_materialised_upper_bound_delta": float(len(held) / len(held_ledger)),
        })
    gates = {
        "all_actions_replayed": corrective_replay["target_views_replayed"] == 882,
        "all_robustness_views_replayed": robustness_replay["target_views_replayed"] == len(robust),
        "every_fold_train_identities_ge_300": all(row["train_corrective_identities"] >= 300 for row in folds),
        "every_fold_train_formulas_ge_150": all(row["train_corrective_formulas"] >= 150 for row in folds),
        "every_fold_held_corrective_ge_100": all(row["held_corrective_queries"] >= 100 for row in folds),
        "pooled_materialised_headroom_ge_0_035": len(corrective) / len(ledger) >= 0.035,
    }
    if not all(gates.values()):
        raise RuntimeError(f"R2 preflight gates failed: {gates}")
    body = {
        "status": "noise_final_r2_preflight_passed", "formal": True,
        "queries": int(graph.n_queries), "corrective_actions": int(len(corrective)),
        "corrective_identities": int(corrective["query_ik14"].nunique()),
        "corrective_formulas": int(corrective["query_formula"].nunique()),
        "robustness_actions": int(len(robust)),
        "corrective_replay": corrective_replay, "robustness_replay": robustness_replay,
        "formula_folds": folds, "gates": gates, "pass": True,
        "contracts": {
            "P2b": "forbidden", "P3_consumed": False,
            "action_outcomes_are_training_only": True,
            "held_formula_isolation": True,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "r1_report_sha256": sha256_file(args.r1_dir / "report.json"),
            "hdf5_sha256": sha256_file(args.data),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": "data/action replay and headroom audit only; no embedding performance",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="noise_r2_preflight_", dir=args.output_dir.parent))
    try:
        json_dump(temporary / "report.json", body)
        temporary.replace(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(body, indent=2), flush=True)


if __name__ == "__main__":
    main()
