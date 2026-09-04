"""Fail-closed structural validator for the CPG0 residual teacher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from build_noise_final_cpg0_residual_teacher import STATUS
from noise_final_core import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-outer-fold", type=int, required=True)
    args = parser.parse_args()
    required = {
        "report": args.output_dir / "report.json",
        "actions": args.output_dir / "actions.csv.gz",
        "residuals": args.output_dir / "candidate_residuals.h5",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    report = json.loads(required["report"].read_text(encoding="utf-8"))
    if (
        report.get("status") != STATUS or not report.get("formal")
        or int(report.get("outer_formula_fold", -1)) != args.expected_outer_fold
        or any(value is not True for value in report.get("gates", {}).values())
    ):
        raise RuntimeError("CPG0 report violates the formal contract")
    actions = pd.read_csv(required["actions"], low_memory=False)
    if len(actions) != int(report["action_rows"]) or actions["action_index"].duplicated().any():
        raise RuntimeError("CPG0 action table count/key mismatch")
    if not np.array_equal(actions["action_index"].to_numpy(np.int64), np.arange(len(actions))):
        raise RuntimeError("CPG0 action indices are not contiguous")
    if np.any(actions["formula_fold"].to_numpy(np.int8) == args.expected_outer_fold):
        raise RuntimeError("held formula fold leaked into CPG0 teacher")
    if set(actions["source"].astype(str)) != {"N", "P_intensity", "P_transfer"}:
        raise RuntimeError("CPG0 lost a mature action source")
    if set(actions["advantage_label"].astype(str)) != {"positive", "neutral", "harmful"}:
        raise RuntimeError("CPG0 did not retain all signed action classes")
    with h5py.File(required["residuals"], "r") as handle:
        expected = {
            "action_ptr", "negative_molecule_local", "clean_margin", "target_margin",
            "control_mean_margin", "paired_residual",
        }
        if expected - set(handle.keys()):
            raise RuntimeError("CPG0 residual file is incomplete")
        ptr = np.asarray(handle["action_ptr"], dtype=np.int64)
        if len(ptr) != len(actions) + 1 or ptr[0] != 0 or np.any(np.diff(ptr) < 1):
            raise RuntimeError("CPG0 ragged pointer is invalid")
        elements = int(ptr[-1])
        if elements != int(report["candidate_residual_elements"]):
            raise RuntimeError("CPG0 residual element count differs from report")
        for name in expected - {"action_ptr"}:
            if len(handle[name]) != elements:
                raise RuntimeError(f"CPG0 ragged dataset length mismatch: {name}")
        target = np.asarray(handle["target_margin"])
        control = np.asarray(handle["control_mean_margin"])
        residual = np.asarray(handle["paired_residual"])
        if not np.all(np.isfinite(residual)) or not np.allclose(target - control, residual, atol=2e-6):
            raise RuntimeError("CPG0 signed residual arithmetic failed replay")
        for row, left, right in zip(actions.itertuples(index=False), ptr[:-1], ptr[1:]):
            if int(right - left) != int(row.negative_candidates):
                raise RuntimeError(f"CPG0 candidate count mismatch at action {row.action_index}")
    print(json.dumps({
        "status": "noise_final_cpg0_residual_teacher_validation_passed",
        "actions": len(actions), "candidate_residual_elements": int(report["candidate_residual_elements"]),
        "outer_formula_fold": args.expected_outer_fold,
        "artifact_sha256": {key: sha256_file(path) for key, path in required.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
