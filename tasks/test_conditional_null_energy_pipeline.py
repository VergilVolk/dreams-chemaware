#!/usr/bin/env python
"""Tiny CPU end-to-end test for all conditional-null training arms."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from noise_final_core import sha256_file, stable_fold


ROOT = Path(__file__).resolve().parent.parent


def formulas_covering_folds(per_fold: int = 3) -> np.ndarray:
    buckets: dict[int, list[str]] = {fold: [] for fold in range(5)}
    index = 0
    while any(len(values) < per_fold for values in buckets.values()):
        formula = f"C{index + 2}H{2 * index + 6}O{index % 7 + 1}"
        fold = stable_fold(formula, 5, 20261004)
        if len(buckets[fold]) < per_fold:
            buckets[fold].append(formula)
        index += 1
    return np.asarray([value for fold in range(5) for value in buckets[fold]])


def build_fixture(root: Path) -> tuple[Path, Path]:
    formulas = formulas_covering_folds()
    query_count = len(formulas)
    molecule_count = 2 * query_count
    query_ptr = np.arange(0, molecule_count + 1, 2, dtype=np.int64)
    molecule_ptr = np.arange(molecule_count + 1, dtype=np.int64)
    labels = np.zeros(molecule_count, dtype=np.int8)
    # Reproduce the real CandidateGraph hazard: the positive is always first.
    labels[2 * np.arange(query_count)] = 1
    rng = np.random.default_rng(20261004)
    v1 = rng.normal(0.5, 0.12, size=molecule_count)
    p2b = v1 + 0.15 * labels + rng.normal(0, 0.04, size=molecule_count)
    neutral = rng.uniform(0, 1, size=molecule_count) + 0.1 * labels
    evidence_dir = root / "evidence"
    evidence_dir.mkdir()
    evidence_file = evidence_dir / "evidence.npz"
    np.savez_compressed(
        evidence_file,
        v1_cosine=v1.astype(np.float32),
        p2b_fused=p2b.astype(np.float32),
        neutral_loss_sqrt_cosine=neutral.astype(np.float32),
        molecule_ptr=molecule_ptr,
        query_ptr=query_ptr,
        molecule_label=labels,
        query_formula=formulas,
    )
    (evidence_dir / "report.json").write_text(
        json.dumps({"status": "NOISE_MSG_PAIR_EVIDENCE_COMPLETE"}), encoding="utf-8",
    )
    chem_valid = np.ones(molecule_count, dtype=bool)
    chem_valid[::2] = False
    correct = rng.normal(0, 0.2, size=molecule_count) + 0.25 * labels
    nulls = rng.normal(0, 0.2, size=(3, molecule_count))
    correct[~chem_valid] = 0
    nulls[:, ~chem_valid] = 0
    chem_file = root / "chem.npz"
    np.savez_compressed(
        chem_file,
        candidate_utility_correct=correct.astype(np.float32),
        candidate_utility_zero=nulls[0].astype(np.float32),
        candidate_utility_reversed=nulls[1].astype(np.float32),
        candidate_utility_rotated=nulls[2].astype(np.float32),
        chem_valid=chem_valid,
        candidate_key=np.asarray([f"IK{index:04d}" for index in range(molecule_count)]),
        query_key=np.asarray([f"row{index:04d}" for index in range(query_count)]),
        candidate_reference_count=np.ones(molecule_count, dtype=np.int32),
        query_candidate_count=np.full(query_count, 2, dtype=np.int32),
        query_has_near=np.asarray([(index % 2) == 0 for index in range(query_count)]),
        evidence_sha256=np.asarray(sha256_file(evidence_file)),
    )
    return evidence_dir, chem_file


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="conditional_null_pipeline_") as raw:
        root = Path(raw)
        evidence, chem = build_fixture(root)
        arm_paths = {}
        for arm in ("joint", "no_interaction", "spectral_only", "chem_only"):
            output = root / arm
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tasks/train_conditional_null_energy.py"),
                    "--evidence", str(evidence),
                    "--chem-evidence", str(chem),
                    "--output", str(output),
                    "--arm", arm,
                    "--epochs", "2",
                    "--hidden", "4",
                    "--device", "cpu",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            assert report["status"] == "CONDITIONAL_NULL_CANDIDATE_ENERGY_FROZEN"
            assert report["arm"] == arm
            if arm == "no_interaction":
                assert report["model_kind"] == "additive_two_head"
            arm_paths[arm] = output
        adjudication = root / "adjudication"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tasks/adjudicate_conditional_null_energy.py"),
                "--joint", str(arm_paths["joint"]),
                "--no-interaction", str(arm_paths["no_interaction"]),
                "--spectral-only", str(arm_paths["spectral_only"]),
                "--chem-only", str(arm_paths["chem_only"]),
                "--output", str(adjudication),
                "--bootstrap-resamples", "1000",
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        report = json.loads((adjudication / "report.json").read_text(encoding="utf-8"))
        assert report["status"] == "CONDITIONAL_NULL_CANDIDATE_ENERGY_ADJUDICATED"
        assert report["queries"] == len(formulas_covering_folds())
        assert set(report["fold_deltas_pp"]) == {"0", "1", "2", "3", "4"}
    print("[test_conditional_null_energy_pipeline] PASS arms=4")


if __name__ == "__main__":
    main()
