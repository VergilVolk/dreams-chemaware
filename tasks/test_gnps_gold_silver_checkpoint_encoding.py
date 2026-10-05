#!/usr/bin/env python
"""CPU-only contract tests for the GNPS checkpoint encoding submission."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from encode_gnps_gold_silver_10ppm_checkpoint import load_selected_spectra, required_rows


ROOT = Path(__file__).resolve().parents[1]


def test_selected_mgf_rows_remain_in_sealed_row_order() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.mgf"
        blocks = []
        for row in range(4):
            if row in {1, 3}:
                blocks.append(
                    "BEGIN IONS\n"
                    f"PEPMASS={100 + row}\n"
                    f"{10 + row} {20 + row}\n"
                    "END IONS\n"
                )
            else:
                blocks.append("BEGIN IONS\nPEPMASS=1\n1 1\nEND IONS\n")
        path.write_text("".join(blocks), encoding="utf-8")
        spectra = load_selected_spectra(
            path, np.asarray([3, 1], dtype=np.int64), expected_records=4,
        )
        assert float(spectra[0][1]) == 103.0
        assert float(spectra[1][1]) == 101.0


def test_sbatch_is_one_gpu_evaluation_without_training_or_memory_request() -> None:
    body = (ROOT / "tasks/run_gnps_gold_silver_stage1_evaluation_1gpu.sbatch").read_text()
    assert "#SBATCH --gpus=1" in body
    assert "#SBATCH --partition=gpu" in body
    assert "#SBATCH --mem" not in body
    assert "train_noise" not in body
    assert "run_2344820" in body
    assert 'STAGE1_NATIVE="$STAGE1/checkpoint/targeted_final.ckpt"' in body
    assert "evaluate_gnps_gold_silver_10ppm_embeddings.py" in body
    assert 'OFFICIAL_ENCODING_REPORT="$LOCAL_ROOT/official_embeddings.json"' in body
    assert 'STAGE1_ENCODING_REPORT="$LOCAL_ROOT/stage1_embeddings.json"' in body
    assert "EMBEDDINGS.json" not in body


def test_two_panel_union_is_complete() -> None:
    benchmark = ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
    assert len(required_rows(benchmark)) == 52871


def main() -> None:
    test_selected_mgf_rows_remain_in_sealed_row_order()
    test_sbatch_is_one_gpu_evaluation_without_training_or_memory_request()
    test_two_panel_union_is_complete()
    print("[test_gnps_gold_silver_checkpoint_encoding] PASS tests=3")


if __name__ == "__main__":
    main()
