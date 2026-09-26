"""CPU contracts for dense true-support ChemAware native triplets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCAL = (
    ROOT
    / "data/validation/chemaware_dense_true_support_official_geometry_localcheck_20260926"
)


def test_executed_full_graph_construction() -> None:
    if not (LOCAL / "report.json").is_file():
        return
    report = json.loads((LOCAL / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "CHEMAWARE_DENSE_TRUE_SUPPORT_NATIVE_TRIPLETS_COMPLETE"
    assert report["cache_kind"] == "official_engineering_standin"
    assert all(report["gates"].values())
    assert report["direct_support"]["broad_true_support_queries"] == 347
    assert report["direct_support"]["strict_true_support_queries"] == 118
    assert report["coverage"]["correction_queries"] >= 200
    assert report["coverage"]["distinct_identity_false_boundaries"] >= 200
    assert report["events"]["unique_correction_triplets"] >= 1000
    assert report["events"]["unique_correction_triplets"] == 4425
    assert report["sampler"] == {
        "type": "native DataLoader shuffle",
        "replacement": False,
        "custom_sampling_weight_present": False,
    }
    with np.load(LOCAL / "train_pool.npz", allow_pickle=False) as pool:
        assert "sampling_weight" not in pool.files
        assert len(pool["anchor_idx"]) == report["events"]["total"]
        signatures = set()
        for event, anchor in enumerate(pool["anchor_idx"]):
            p0, p1 = map(int, pool["positive_ptr"][event:event + 2])
            n0, n1 = map(int, pool["negative_ptr"][event:event + 2])
            signature = (
                int(anchor), tuple(map(int, pool["positive_idx"][p0:p1])),
                tuple(map(int, pool["negative_idx"][n0:n1])),
            )
            assert signature not in signatures
            signatures.add(signature)
        assert len(signatures) == len(pool["anchor_idx"])


def test_formal_builder_and_one_gpu_entrypoint() -> None:
    builder = (
        ROOT / "tasks/build_chemaware_dense_true_support_native_triplets.py"
    ).read_text(encoding="utf-8")
    assert "minimum_unique_correction_triplets" in builder
    assert "minimum_independent_correction_queries" in builder
    assert "minimum_distinct_identity_false_boundaries" in builder
    assert '"sampling_weight" not in output' in builder

    path = ROOT / "tasks/run_chemaware_dense_true_support_native.sbatch"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --mem" not in text
    assert "srun --export=ALL --preserve-env python -u" in text
    assert "tasks/train_chemaware_dreams_native.py" in text
    assert "tasks/train_chemaware_weighted_native.py" not in text
    assert "--minimum-correction-triplets 1000" in text
    assert "--official-checkpoint \"$PHASEA\"" in text
    assert "--paired-reference phaseA_base --formula-role 2" in text
    assert "--require-positive-formula-ci" in text
    assert "tasks/run_noise" not in text


def main() -> None:
    test_executed_full_graph_construction()
    test_formal_builder_and_one_gpu_entrypoint()
    print("PASS: ChemAware dense true-support native-triplet contracts", flush=True)


if __name__ == "__main__":
    main()
