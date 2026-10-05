"""Static fail-closed checks for the bounded V8 two-GPU canary."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SBATCH = ROOT / "tasks/run_noise_corrected_v8_query_local_canary_2gpu.sbatch"
MANIFEST = ROOT / "tasks/noise_best_v8_source_manifest.sha256"


def test_two_gpus_and_no_manual_memory() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    directives = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    assert directives.count("#SBATCH --gpus=2") == 1
    assert not any("--mem" in line for line in directives)
    assert "srun " not in text


def test_canary_is_bounded_and_has_four_causal_arms() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    common = text.split("COMMON=(", 1)[1].split("\n)", 1)[0]
    assert "--development" in common
    assert "--epochs 1" in common
    assert "--maximum-corrective-queries 512" in common
    assert "--outer-held-eval-queries 4096" in common
    assert "--corrective-admission strict_top1" in common
    assert "--materialize-optimizer-update-restoration" in common
    assert "--optimizer-restoration-scope corrective_only" in common
    for invocation in (
        "query_local_routed routed_direct query_action_only",
        "query_local_shuffled shuffled_action_control query_action_only",
        "shared_routed routed_direct shared",
        "clean_control clean_control shared",
    ):
        assert invocation in text
    assert "summarize_noise_corrected_v8_query_local_canary.py" in text


def test_source_snapshot_and_manifest_are_exact() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    lines = [line for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) >= 60
    paths = []
    for line in lines:
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative
        paths.append(relative)
    assert len(paths) == len(set(paths))
    required = {
        "tasks/noise_corrected_gradient_locality_v8.py",
        "tasks/test_noise_corrected_gradient_locality_v8.py",
        "tasks/summarize_noise_corrected_v8_query_local_canary.py",
        "tasks/test_summarize_noise_corrected_v8_query_local_canary.py",
        "tasks/test_noise_corrected_v8_query_local_sbatch.py",
        "tasks/train_noise_corrected_routed_direct.py",
    }
    assert required <= set(paths)
    manifest_digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert f'EXPECTED_SOURCE_MANIFEST_SHA256="{manifest_digest}"' in text


def main() -> None:
    tests = [
        test_two_gpus_and_no_manual_memory,
        test_canary_is_bounded_and_has_four_causal_arms,
        test_source_snapshot_and_manifest_are_exact,
    ]
    for test in tests:
        test()
    print(f"[test_noise_corrected_v8_query_local_sbatch] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
