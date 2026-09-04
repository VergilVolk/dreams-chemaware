"""Evaluate the frozen ChemAware spectral-consensus gate once on test.

The evaluator loads fixed scaler/logistic parameters and a fixed threshold. It
does not read discovery labels, fit a model, or select a threshold.  Before
opening the test pair table it reproduces the recorded confirmation result;
any mismatch fails closed.
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


def probability(frame: pd.DataFrame, gate: dict) -> np.ndarray:
    features = list(gate["features"])
    values = frame[features].to_numpy(np.float64)
    mean = np.asarray(gate["standard_scaler_mean"], dtype=np.float64)
    scale = np.asarray(gate["standard_scaler_scale"], dtype=np.float64)
    coefficient = np.asarray(gate["logistic_coefficient"], dtype=np.float64)
    intercept = float(gate["logistic_intercept"])
    if values.shape[1] != len(mean) or mean.shape != scale.shape or mean.shape != coefficient.shape:
        raise RuntimeError("frozen gate dimension mismatch")
    logit = ((values - mean) / scale) @ coefficient + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0)))


def metrics(frame: pd.DataFrame, prob: np.ndarray, threshold: float) -> tuple[dict, np.ndarray]:
    active = frame["route_candidate"].to_numpy(bool) & (prob >= threshold)
    old = frame["dreams_correct"].to_numpy(bool)
    new = old.copy()
    new[active] = frame.loc[active, "consensus_correct"].to_numpy(bool)
    corrected = (~old) & new
    introduced = old & (~new)
    return ({
        "queries": len(frame),
        "identities": int(frame["ik14"].nunique()),
        "formulas": int(frame["formula"].nunique()),
        "route_candidates": int(frame["route_candidate"].sum()),
        "route_activated": int(active.sum()),
        "corrected": int(corrected.sum()),
        "introduced": int(introduced.sum()),
        "wrong_to_different_wrong": int((active & ~old & ~new).sum()),
        "risk_utility_corrected_minus_2x_introduced": int(corrected.sum() - 2 * introduced.sum()),
        "dreams_recall1": float(old.mean()),
        "routed_recall1": float(new.mean()),
        "delta_pp": float(100.0 * np.mean(new.astype(float) - old.astype(float))),
    }, new)


def bootstrap(frame: pd.DataFrame, new: np.ndarray, iterations: int, seed: int) -> list[float]:
    work = pd.DataFrame({
        "formula": frame["formula"].astype(str),
        "n": 1,
        "delta": new.astype(np.int8) - frame["dreams_correct"].to_numpy(np.int8),
    }).groupby("formula", sort=False).sum()
    n = work["n"].to_numpy(float)
    delta = work["delta"].to_numpy(float)
    rng = np.random.default_rng(seed)
    out = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 500):
        stop = min(start + 500, iterations)
        draw = rng.integers(0, len(work), size=(stop - start, len(work)))
        out[start:stop] = delta[draw].sum(axis=1) / n[draw].sum(axis=1)
    return [float(value) for value in np.quantile(out, (0.025, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze-dir", type=Path,
        default=ROOT / "data/validation/chemaware_spectral_consensus_applicability_v4_frozen",
    )
    parser.add_argument(
        "--confirmation-pairs", type=Path,
        default=ROOT / "data/validation/large_observability_residual_audit/confirmation_pair_features.csv",
    )
    parser.add_argument(
        "--confirmation-manifest", type=Path,
        default=ROOT / "data/validation/large_observability_embeddings_confirmation/manifest.csv",
    )
    parser.add_argument(
        "--test-input-dir", type=Path,
        default=ROOT / "data/validation/chemaware_frozen_gate_test_inputs_20260902",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/chemaware_frozen_spectral_gate_test_20260902",
    )
    args = parser.parse_args()
    from audit_chemaware_spectral_consensus_applicability import FEATURES, build_query_table

    gate_path = args.freeze_dir / "frozen_gate.json"
    freeze_report_path = args.freeze_dir / "report.json"
    test_report_path = args.test_input_dir / "report.json"
    test_pair_path = args.test_input_dir / "test_pair_features.csv.gz"
    test_manifest_path = args.test_input_dir / "test_manifest.csv"
    required = [
        gate_path, freeze_report_path, test_report_path, test_pair_path, test_manifest_path,
        args.confirmation_pairs, args.confirmation_manifest,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite frozen test result: {args.output_dir}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    freeze_report = json.loads(freeze_report_path.read_text(encoding="utf-8"))
    test_input_report = json.loads(test_report_path.read_text(encoding="utf-8"))
    if gate.get("format") != "chemaware_spectral_consensus_gate_v1":
        raise RuntimeError("unknown frozen gate format")
    if gate.get("test_split_seen") is not False or gate.get("frozen_before_test") is not True:
        raise RuntimeError("gate does not assert pre-test freezing")
    if list(gate.get("features", [])) != list(FEATURES):
        raise RuntimeError("frozen gate feature order differs from evaluator")
    if test_input_report.get("test_split_consumed") is not True:
        raise RuntimeError("test input ledger is not sealed as consumed")
    if test_input_report["provenance"]["test_pair_features_sha256"] != sha256(test_pair_path):
        raise RuntimeError("test pair table hash mismatch")
    if test_input_report["provenance"]["test_manifest_copy_sha256"] != sha256(test_manifest_path):
        raise RuntimeError("test manifest hash mismatch")

    confirmation = build_query_table(args.confirmation_pairs, args.confirmation_manifest, "confirmation")
    confirmation_probability = probability(confirmation, gate)
    confirmation_metrics, _ = metrics(confirmation, confirmation_probability, float(gate["threshold"]))
    expected = freeze_report["evaluation"]["formula_disjoint_confirmation"]
    for key in ("route_activated", "corrected", "introduced", "risk_utility_corrected_minus_2x_introduced"):
        if confirmation_metrics[key] != expected[key]:
            raise RuntimeError(
                f"frozen confirmation reproduction failed for {key}: "
                f"{confirmation_metrics[key]} != {expected[key]}"
            )

    test = build_query_table(test_pair_path, test_manifest_path, "test")
    test_probability = probability(test, gate)
    test_metrics, test_new = metrics(test, test_probability, float(gate["threshold"]))
    test_metrics["formula_cluster_bootstrap_delta_ci95"] = bootstrap(
        test, test_new, args.bootstrap, args.seed
    )
    ledger = test.copy()
    ledger["gate_probability"] = test_probability
    ledger["route_activated"] = (
        ledger["route_candidate"] & (test_probability >= float(gate["threshold"]))
    )
    args.output_dir.mkdir(parents=True)
    ledger_path = args.output_dir / "test_gate_ledger.csv.gz"
    ledger.to_csv(ledger_path, index=False)
    report = {
        "status": "chemaware_frozen_spectral_gate_test_consumed",
        "test_was_consumed_once": True,
        "model_or_gate_fit_on_test": False,
        "threshold_selected_on_test": False,
        "confirmation_reproduction": confirmation_metrics,
        "test": test_metrics,
        "frozen_gate": {"path": str(gate_path), "sha256": sha256(gate_path)},
        "test_ledger": {"path": str(ledger_path), "sha256": sha256(ledger_path)},
        "provenance": {
            "freeze_report_sha256": sha256(freeze_report_path),
            "test_input_report_sha256": sha256(test_report_path),
            "test_pair_features_sha256": sha256(test_pair_path),
            "test_manifest_sha256": sha256(test_manifest_path),
            "evaluator_script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Internal formula-isolated held-out test of a candidate-reference teacher gate. "
            "It is not P3, not an external dataset, and not a shared-embedding improvement."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
