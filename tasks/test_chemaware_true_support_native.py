"""CPU contracts for the ChemAware true-support native-triplet route."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_chemaware_true_support_native_triplets import (
    ADVANTAGE_METRICS,
    RULE_METRICS,
    SUPPORT_METRICS,
    select_endpoint_sentinel,
    true_candidate_support,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/validation/chemaware_high_coverage_native/run_2340524"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def test_real_three_null_true_support() -> None:
    evidence = load_npz(BASE / "evidence/train_triplet_evidence.npz")
    manifest = load_npz(
        ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
    )
    selected, audit = true_candidate_support(evidence, manifest, 2)
    assert audit["official_error_queries"] == 427
    assert audit["broad_true_support_queries"] == 347
    assert audit["strict_true_support_queries"] == 118
    assert audit["broad_true_support_identities"] == 347
    assert audit["broad_true_support_formulas"] == 230
    assert audit["strict_true_support_formulas"] == 103
    assert len(selected) == 347
    assert len(SUPPORT_METRICS) == 3
    assert len(ADVANTAGE_METRICS) == 2
    assert len(RULE_METRICS) == 4


def test_endpoint_guard_prefers_inactive_nearest_boundary() -> None:
    geometries = {
        1: {"query": 1, "error": False, "margin": 0.04},
        2: {"query": 2, "error": False, "margin": 0.14},
        3: {"query": 3, "error": False, "margin": 0.30},
        4: {"query": 4, "error": True, "margin": -0.01},
    }
    selected = select_endpoint_sentinel(
        "A", {"A": [1, 2, 3, 4]}, geometries.__getitem__, 0.1, set(),
    )
    assert selected is geometries[2]
    fallback = select_endpoint_sentinel(
        "A", {"A": [1, 4]}, geometries.__getitem__, 0.1, set(),
    )
    assert fallback is geometries[1]
    assert select_endpoint_sentinel(
        "A", {"A": [4]}, geometries.__getitem__, 0.1, set(),
    ) is None


def test_local_full_construction_if_present() -> None:
    output = (
        ROOT
        / "data/validation/chemaware_true_support_official_geometry_localcheck_20260926"
    )
    if not (output / "report.json").is_file():
        return
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "CHEMAWARE_TRUE_SUPPORT_NATIVE_TRIPLETS_COMPLETE"
    assert all(report["gates"].values())
    assert report["direct_support"]["broad_true_support_queries"] == 347
    assert report["direct_support"]["strict_true_support_queries"] == 118
    assert report["events"]["total"] > 500
    with np.load(output / "train_pool.npz", allow_pickle=False) as pool:
        weights = np.asarray(pool["sampling_weight"], dtype=np.float64)
        assert len(weights) == len(pool["anchor_idx"])
        assert np.all(np.isfinite(weights)) and np.all(weights > 0)
        assert abs(float(weights.sum()) - 1.0) <= 1e-12
        assert np.all(np.diff(pool["positive_ptr"]) == 1)
        assert np.all(np.diff(pool["negative_ptr"]) == 1)


def test_one_gpu_sbatch_and_isolated_sources() -> None:
    path = ROOT / "tasks/run_chemaware_true_support_native.sbatch"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --mem" not in text
    assert "srun --export=ALL --preserve-env python -u" in text
    assert "--official-checkpoint \"$PHASEA\"" in text
    assert "--lr 1e-6" in text
    assert "--max-steps 400" in text
    assert "--save-every-n-steps 50" in text
    assert "--paired-reference phaseA_base --formula-role 2" in text
    assert "--require-positive-formula-ci" in text
    assert "tasks/build_chemaware_true_support_native_triplets.py" in text
    assert "tasks/freeze_chemaware_true_support_artifact.py" in text
    assert "tasks/run_noise" not in text


def main() -> None:
    test_real_three_null_true_support()
    test_endpoint_guard_prefers_inactive_nearest_boundary()
    test_local_full_construction_if_present()
    test_one_gpu_sbatch_and_isolated_sources()
    print("PASS: ChemAware true-support native-triplet contracts")


if __name__ == "__main__":
    main()
