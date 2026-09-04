"""Unit tests for E15-M1 calibration contracts."""
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import audit_noise_final_e15_replay_calibration as audit
from noise_final_core import CandidateGraph
from noise_final_e15_calibration import (
    calibrate_source_local, diverse_panel, inverse_source_weights,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic() -> pd.DataFrame:
    rows = []
    for source_index, source in enumerate(("R0_N", "A4_exact", "C1_support_disjoint", "E14_mature_P")):
        for kind in ("corrective", "harmful"):
            for index in range(24):
                sign = 1.0 if kind == "corrective" else -1.0
                rows.append({
                    "source": source, "supervision_kind": kind,
                    "action_family": "large" if index < 20 else f"small_{source_index}",
                    "margin_delta": sign * (0.01 + source_index + index / 1000.0),
                    "query_index": source_index * 1000 + index,
                    "query_ik14": f"IK{source_index}_{index}",
                    "query_formula": f"F{index % 7}", "action_id": f"{source}|{kind}|{index}",
                })
    return pd.DataFrame(rows)


def main() -> None:
    for path in (
        ROOT / "tasks/noise_final_e15_calibration.py",
        ROOT / "tasks/audit_noise_final_e15_replay_calibration.py",
        ROOT / "tasks/validate_noise_final_e15_replay_calibration.py",
    ):
        ast.parse(path.read_text(encoding="utf-8"))
    calibrated, table = calibrate_source_local(synthetic())
    assert len(calibrated) == 192 and len(table) == 16
    assert np.isfinite(calibrated["calibrated_strength"]).all()
    assert calibrated["source_kind_percentile"].between(0, 1).all()
    assert set(table["level"]) == {"source_kind_family", "source_kind_fallback"}
    panel = diverse_panel(calibrated, per_source_kind=16, seed=11)
    assert len(panel) == 128
    assert panel.groupby(["source", "supervision_kind"]).size().eq(16).all()
    weights = inverse_source_weights(calibrated)
    for kind in ("corrective", "harmful"):
        mass = {
            source: weights[f"{kind}|{source}"] * len(block)
            for source, block in calibrated.loc[calibrated["supervision_kind"].eq(kind)].groupby("source")
        }
        assert max(mass.values()) - min(mass.values()) < 1e-12
    bad = synthetic()
    bad.loc[0, "margin_delta"] *= -1
    try:
        calibrate_source_local(bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-directional corrective action was accepted")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        r0 = pd.DataFrame([{
            "query_index": 0, "selector": "candidate_gradient", "attenuation": 0.5,
            "step": 6, "baseline_rank": 2, "baseline_margin": -0.1,
            "target_rank": 1, "target_margin": 0.1, "target_path": "1,2",
        }])
        a4 = pd.DataFrame([{
            "query_index": 0, "policy_eligible": True, "gradient_rank": 1,
            "token": 3, "attenuation": 0.5, "baseline_rank": 2,
            "baseline_margin": -0.1, "result_rank": 1, "result_margin": 0.05,
        }])
        c1 = pd.DataFrame([{
            "query_index": 0, "evaluation_positive_row": 10, "teacher_rows": "11",
            "baseline_rank": 2, "baseline_margin": -0.1,
            "teacher_rank": 1, "teacher_margin": 0.02,
        }])
        r0_path, a4_path, c1_path = root / "r0.csv", root / "a4.csv", root / "c1.csv"
        r0.to_csv(r0_path, index=False); a4.to_csv(a4_path, index=False); c1.to_csv(c1_path, index=False)
        assert audit.r0_expected(r0_path).iloc[0]["action_id"].endswith("step=6")
        assert "token=3" in audit.a4_expected(a4_path).iloc[0]["action_id"]
        assert "teachers=11" in audit.c1_expected(c1_path).iloc[0]["action_id"]
        e14_path = root / "e14.npz"
        np.savez_compressed(
            e14_path, queries=np.asarray([4, 5]), action_ids=np.asarray(["a", "b"]),
            clean_rank=np.asarray([2, 1]), clean_margin=np.asarray([-0.1, 0.1]),
            result_rank=np.asarray([[1, 2], [1, 3]]),
            result_margin=np.asarray([[0.1, -0.2], [0.2, -0.1]]),
        )
        e14 = audit.e14_expected(e14_path)
        assert list(zip(e14["query_index"], e14["action_id"])) == [(4, "a"), (5, "b")]

        graph_path = root / "graph.npz"
        np.savez_compressed(
            graph_path,
            feature_names=np.asarray(["dreams_similarity"]),
            features=np.asarray([[0.8], [0.7], [0.6]], dtype=np.float32),
            pair_candidate_row=np.asarray([10, 11, 20], dtype=np.int64),
            query_ptr=np.asarray([0, 2], dtype=np.int64),
            molecule_ptr=np.asarray([0, 2, 3], dtype=np.int64),
            molecule_label=np.asarray([1, 0], dtype=np.int8),
            molecule_ik14=np.asarray(["POS", "NEG"]),
            molecule_formula=np.asarray(["F", "F"]),
            molecule_mces_grade=np.asarray([-1, 1], dtype=np.int8),
            query_row=np.asarray([9], dtype=np.int64),
            query_ik14=np.asarray(["POS"]), query_formula=np.asarray(["F"]),
            query_has_near=np.asarray([True]),
        )
        graph = CandidateGraph(graph_path)
        sample_definition = next(iter(audit.action_definitions()))
        payload_frame = pd.DataFrame([
            {"source": "R0_N", "query_index": 0, "action_id": "r0", "action_payload": "1,2"},
            {"source": "A4_exact", "query_index": 0, "action_id": "a4",
             "action_payload": json.dumps({"token": 3, "mz": 100.0, "role": "x"})},
            {"source": "C1_support_disjoint", "query_index": 0, "action_id": "c1",
             "action_payload": json.dumps({"evaluation_positive_row": 10, "teacher_rows": "11"})},
            {"source": "E14_mature_P", "query_index": 0, "action_id": sample_definition.action_id,
             "action_payload": json.dumps({"reference_policy": sample_definition.reference_policy})},
        ])
        counts = audit.validate_payloads(payload_frame, graph)
        assert set(counts) == set(audit.SOURCES)

    sbatch = (ROOT / "tasks/run_noise_final_e15_replay_calibration.sbatch").read_text(encoding="utf-8")
    required = (
        "#SBATCH --partition=gpu", "#SBATCH --gpus=1", "set -euo pipefail",
        'fold_${FOLD}_run_${SLURM_JOB_ID}',
        "E15-M1 output: $OUTPUT",
        "test_noise_final_e15_replay_calibration.py",
        "audit_noise_final_e15_replay_calibration.py",
        "validate_noise_final_e15_replay_calibration.py",
    )
    missing = [token for token in required if token not in sbatch]
    if missing:
        raise RuntimeError(f"E15-M1 sbatch contract is incomplete: {missing}")
    forbidden = ("--mem=", "train_noise", "--array=", "rm -", "rm --", "python - <<")
    found = [token for token in forbidden if token in sbatch]
    if found:
        raise RuntimeError(f"E15-M1 unexpectedly trains or requests forbidden resources: {found}")
    redundant = (
        "test_noise_final_e15_multi_action.py",
        "validate_noise_final_e15_multi_action_ledger.py",
    )
    found_redundant = [token for token in redundant if token in sbatch]
    if found_redundant:
        raise RuntimeError(f"E15-M1 repeats already-passed M0 audits: {found_redundant}")
    source = (ROOT / "tasks/audit_noise_final_e15_replay_calibration.py").read_text(
        encoding="utf-8"
    )
    if '"P3_consumed": False' in source.split("gates = {", 1)[1].split("}", 1)[0]:
        raise RuntimeError("a negative-state P3 value was placed inside all(gates.values())")
    if '"P3_not_consumed": True' not in source:
        raise RuntimeError("E15-M1 lacks a positive-form P3 isolation gate")
    print("[test_noise_final_e15_replay_calibration] PASS")


if __name__ == "__main__":
    main()
