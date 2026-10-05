"""Mutation tests for the exact historical E4 source and Slurm contract."""
from __future__ import annotations

import argparse
import ast
import tempfile
from pathlib import Path

from audit_noise_e4_faithful_v1 import audit_static


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "tasks/train_noise_e4_faithful_v1.py"
VALIDATOR = ROOT / "tasks/validate_noise_e4_faithful_v1.py"
HISTORICAL_TEST = ROOT / "tasks/test_noise_e4_faithful_v1.py"
NOISE_V3_CORE = ROOT / "tasks/noise_e4_faithful_v1_noise_v3_core.py"
DREAMS_LAYERS = ROOT / "dreams/models/dreams/layers_e4_historical.py"
SBATCH = ROOT / "tasks/run_noise_e4_faithful_v1_2gpu.sbatch"
MANIFEST = ROOT / "tasks/noise_e4_faithful_v1_source_manifest.sha256"
EVALUATOR = ROOT / "tasks/evaluate_noise_e4_faithful_v1.py"
SUMMARY = ROOT / "tasks/summarize_noise_e4_faithful_v1.py"


def namespace(
    *, trainer: Path = TRAINER, noise_v3_core: Path = NOISE_V3_CORE,
    sbatch: Path = SBATCH,
) -> argparse.Namespace:
    return argparse.Namespace(
        trainer=trainer,
        validator=VALIDATOR,
        historical_test=HISTORICAL_TEST,
        noise_v3_core=noise_v3_core,
        dreams_layers=DREAMS_LAYERS,
        sbatch=sbatch,
        source_manifest=MANIFEST,
        r0_dir=None,
        graph=None,
        historical_cache=None,
        data=None,
        official_checkpoint=None,
        architecture_checkpoint=None,
        corrected_graph=None,
        corrected_cache=None,
        corrected_metadata=None,
        runtime_report=None,
    )


def must_fail(args: argparse.Namespace, label: str) -> None:
    try:
        audit_static(args)
    except RuntimeError:
        return
    raise AssertionError(f"historical E4 audit accepted mutation: {label}")


def main() -> None:
    required = (
        TRAINER, VALIDATOR, HISTORICAL_TEST, NOISE_V3_CORE, DREAMS_LAYERS,
        SBATCH, MANIFEST, EVALUATOR, SUMMARY,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (
        TRAINER, VALIDATOR, HISTORICAL_TEST, NOISE_V3_CORE,
        DREAMS_LAYERS, EVALUATOR, SUMMARY,
    ):
        ast.parse(path.read_text(encoding="utf-8"))
    audit = audit_static(namespace())
    assert audit["post_e4_injector_or_optimizer_boundary"] is False
    assert audit["replays"] == {"primary": 20260830, "replica": 20260829}

    with tempfile.TemporaryDirectory(prefix="noise_e4_faithful_test_") as temp:
        root = Path(temp)
        changed_trainer = root / "trainer.py"
        changed_trainer.write_bytes(TRAINER.read_bytes() + b"\n")
        must_fail(namespace(trainer=changed_trainer), "trainer byte")

        changed_core = root / "noise_v3_core.py"
        changed_core.write_bytes(NOISE_V3_CORE.read_bytes() + b"\n")
        must_fail(namespace(noise_v3_core=changed_core), "action primitive byte")

        source = SBATCH.read_text(encoding="utf-8")
        mutations = {
            "manual memory": source.replace(
                "#SBATCH --gpus=2", "#SBATCH --gpus=2\n#SBATCH --mem=64G",
            ),
            "one GPU": source.replace("#SBATCH --gpus=2", "#SBATCH --gpus=1"),
            "warm start": source.replace(
                "--policy curriculum",
                "--initial-student-checkpoint bad.pt\n  --policy curriculum",
            ),
            "injector": source.replace(
                "--policy curriculum",
                "--optimizer-boundary-mode action_injector_v1\n  --policy curriculum",
            ),
            "changed dose": source.replace(
                "--views-per-identity 4", "--views-per-identity 3",
            ),
            "changed primary seed": source.replace(
                'launch_replay "$GPU_ZERO" primary 20260830',
                'launch_replay "$GPU_ZERO" primary 20260831',
            ),
        }
        for label, body in mutations.items():
            path = root / f"{label.replace(' ', '_')}.sbatch"
            path.write_text(body, encoding="utf-8")
            must_fail(namespace(sbatch=path), label)

    evaluator_source = EVALUATOR.read_text(encoding="utf-8")
    for token in (
        "full_metrics", "formula_bootstrap_delta",
        "massspecgym_pairwise_not_nist20_replication",
        "evaluation_cannot_modify_training", "score_embedding_query_subset",
        "official_cache_graph_max_abs_error",
        "checkpoint_identity_drift", "registered train-side development graph",
        "reference_spectrum_formula_isolation_enforced",
        "runtime_cuda_stack_byte_exact",
        "28c3b375d270fc2030783938d9390710c2c5ab8f0926a0b4ff507d62afa26885",
    ):
        if token not in evaluator_source:
            raise AssertionError(f"corrected evaluator contract missing: {token}")
    sbatch_source = SBATCH.read_text(encoding="utf-8")
    for token in (
        "terminate_worker_tree", 'pgrep -P "$ROOT_PID"',
        '! kill -0 "$PID_PRIMARY"', '! kill -0 "$PID_REPLICA"',
        "primary.status.tmp", "replica.status.tmp",
        "set -e\n      run_replay", 'cd "$RUN_ROOT"',
        "! -name 'artifact_manifest.sha256.tmp'",
        "mv artifact_manifest.sha256.tmp artifact_manifest.sha256",
        "86b7e60194b059b839d266e9e6b52a84b185e5b7160aa934531d4c1c026a5486",
        "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f",
        "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2",
        "8a57bb3a9cccdf69a738f2b093bfad8f6fc4393b23f1342c7a035ac54ce58fa1",
        "18d7632adc67dcda5d650a5c0bc344d8db3878f9e1ff3f8d51a6d4918b96bbda",
        "504ce0a570ac1bc461e76cad81acb3ca3560e58b008ec2d8ff14036e944599f5",
    ):
        if token not in sbatch_source:
            raise AssertionError(f"SBATCH frozen closure missing: {token}")
    summary_source = SUMMARY.read_text(encoding="utf-8")
    if 'held["corrected"] - 2 * held["introduced"]' not in summary_source:
        raise AssertionError("historical E4 lambda-2 risk-net definition drifted")
    if '"risk_net": int(held["corrected"] - held["introduced"])' in summary_source:
        raise AssertionError("obsolete lambda-1 risk-net leaked into summary")
    for token in (
        '"formal": False', "registered train-side development graph",
        "reference_spectrum_formula_isolation_enforced",
        "runtime_cuda_stack_byte_exact",
        'int(held["risk_net"]) != recomputed_risk_net',
    ):
        if token not in summary_source:
            raise AssertionError(f"summary claim boundary missing: {token}")
    print("[test_noise_e4_faithful_v1_sbatch] PASS tests=11")


if __name__ == "__main__":
    main()
