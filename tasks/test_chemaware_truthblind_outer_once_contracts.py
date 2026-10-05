from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data/validation/chemaware_truthblind_candidate_policy/run_2338337/policy"


def function_body(source: str, name: str, next_name: str) -> str:
    start = source.index(f"def {name}(")
    end = source.index(f"def {next_name}(", start)
    return source[start:end]


def main() -> None:
    evaluator_path = ROOT / "tasks/evaluate_chemaware_truthblind_outer_once.py"
    evaluator = evaluator_path.read_text(encoding="utf-8")
    sbatch = (ROOT / "tasks/run_chemaware_truthblind_outer_once.sbatch").read_text(
        encoding="utf-8"
    )
    audit = (ROOT / "tasks/audit_chemaware_orthogonal_rule_residual_policy.py").read_text(
        encoding="utf-8"
    )

    inference = function_body(evaluator, "inference_ledger", "labels_for_queries")
    assert "molecule_label" not in inference
    assert "query_formula" not in inference
    assert "baseline_rank" not in inference
    assert "predict_truthblind_policy" in inference
    assert "predict_truthblind_baselines" in inference

    preflight_gate = evaluator.index('if not inner_preflight["passed"]')
    seal = evaluator.index("create_outer_seal(args.seal", preflight_gate)
    outer_inference = evaluator.index("outer_ledger = inference_ledger(", seal)
    prediction_freeze = evaluator.index("np.savez_compressed(prediction_path", outer_inference)
    truth_open = evaluator.index("evaluate_ledger(", prediction_freeze)
    assert preflight_gate < seal < outer_inference < prediction_freeze < truth_open
    assert 'for key in ("query_ptr", "molecule_ptr", "pair_candidate_row", "query_row")' in evaluator
    assert '"inference_truth_fields": []' in evaluator
    assert '"outer_tuning": False' in evaluator
    assert '"best": "best_predicted_utility"' in evaluator
    assert 'expected[f"{prefix}_best_utility"]' not in evaluator
    assert '"baselines": {' in audit
    assert '"same_feature_direct"' in audit and '"nuisance_only"' in audit

    canonical_hashes = {
        "truthblind_policy.joblib": "8402f32d07287a47008556bf25ad241d25c9e56e6d30fd1d68a407de1a2f2dee",
        "report.json": "64a997a4fc05f442c04e2746a5ac6e1022272bb21417b237cde279ea66aa47bf",
        "inner_policy.npz": "70e25e6d2459222be3597748a851f206c5121792cf920e8e745ed2caa6dcace3",
    }
    for name, expected in canonical_hashes.items():
        assert hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest() == expected
        assert expected in evaluator and expected in sbatch
    with np.load(CANONICAL / "inner_policy.npz", allow_pickle=False) as inner:
        assert {
            "correct_rank", "correct_selected_candidate_slot", "best_predicted_utility",
            "same_feature_direct_rank", "same_feature_direct_selected_candidate_slot",
            "same_feature_direct_best_utility", "nuisance_only_rank",
            "nuisance_only_selected_candidate_slot", "nuisance_only_best_utility",
        }.issubset(inner.files)

    directives = [line.strip() for line in sbatch.splitlines() if line.startswith("#SBATCH")]
    assert directives.count("#SBATCH --gpus=1") == 1
    assert not any("--mem" in line for line in directives)
    assert "--array" not in sbatch
    assert sbatch.count("python -u tasks/evaluate_chemaware_truthblind_outer_once.py \\") == 1
    assert "OUTER_FOLD_4_OPENED.lock" in sbatch
    assert "run_2338356/release_policy" in sbatch
    assert "Reusing completed release policy from run 2338356" in sbatch
    assert "audit_chemaware_orthogonal_rule_residual_policy.py" in sbatch
    assert "CHEMAWARE_TRUTHBLIND_OUTER_ONCE_COMPLETE" in sbatch

    locks = re.findall(r"^verify_sha (\S+) ([0-9a-f]{64})$", sbatch, flags=re.MULTILINE)
    runtime_locks = [(path, digest) for path, digest in locks if path.startswith("tasks/")]
    assert len(runtime_locks) == 6
    for relative, expected in runtime_locks:
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected, (relative, observed, expected)
    print("PASS: ChemAware exact-replay and sealed-outer sbatch contracts")


if __name__ == "__main__":
    main()
