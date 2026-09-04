#!/usr/bin/env python
"""Synthetic end-to-end test for the frozen MoNA transfer evaluator."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from noise_final_core import ZeroInitPeakAdapter, sha256_file  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        panel_dir, token_dir, model_root = root / "panel", root / "tokens", root / "models"
        panel_dir.mkdir(); token_dir.mkdir()
        np.savez_compressed(
            panel_dir / "panel.npz",
            query_row=np.array([0]), query_ik14=np.array(["AAAAAAAAAAAAAA"]),
            query_formula=np.array(["C2H4"]), query_ptr=np.array([0, 2]),
            molecule_ik14=np.array(["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"]),
            molecule_label=np.array([True, False]), molecule_ptr=np.array([0, 1, 3]),
            candidate_row=np.array([1, 2, 3]),
        )
        (panel_dir / "report.json").write_text(json.dumps({
            "formal": True, "development_identity_overlap": 0,
            "construction_uses_model_scores": False, "claim_limit": "synthetic",
        }))
        rows = np.arange(4)
        embeddings = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0], [0.9, 0.1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0], [0, 0.8, 0.2, 0, 0, 0, 0, 0],
        ], dtype=np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        np.save(token_dir / "rows.npy", rows)
        np.save(token_dir / "tokens_f16.npy", np.zeros((4, 2, 8), np.float16))
        np.save(token_dir / "mz_f32.npy", np.ones((4, 2), np.float32))
        np.save(token_dir / "intensity_f32.npy", np.ones((4, 2), np.float32))
        np.save(token_dir / "valid.npy", np.ones((4, 2), bool))
        np.savez_compressed(token_dir / "official_embeddings.npz", rows=rows, embeddings=embeddings)
        (token_dir / "report.json").write_text(json.dumps({
            "status": "mona_identity_disjoint_transfer_token_cache_complete", "formal": True,
            "provenance": {"panel_sha256": sha256_file(panel_dir / "panel.npz")},
        }))
        hyperparameters = {"synthetic": True}
        for seed in (1, 2, 3):
            directory = model_root / "final" / f"seed_{seed}"
            directory.mkdir(parents=True)
            adapter = ZeroInitPeakAdapter(8, 4, 0.05)
            torch.save({"adapter": adapter.state_dict(), "configuration": {"hidden_dim": 4, "delta_bound": 0.05}}, directory / "final.pt")
            (directory / "report.json").write_text(json.dumps({
                "status": "bioaware_embedding_adapter_final_refit_complete", "formal": True,
                "training": {"frozen_hyperparameters": hyperparameters},
            }))
        output = root / "result.json"
        command = [
            sys.executable, str(ROOT / "tasks/evaluate_mona_bioaware_embedding_transfer.py"),
            "--panel-dir", str(panel_dir), "--token-dir", str(token_dir),
            "--model-root", str(model_root), "--seeds", "1", "2", "3",
            "--output", str(output), "--device", "cpu", "--bootstrap", "20",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        report = json.loads(output.read_text())
        assert report["baseline"]["recall1"] == 1.0
        assert report["primary_three_seed_embedding_ensemble"]["delta_recall1"] == 0.0
        assert report["contracts"]["external_outcome_used_for_seed_selection"] is False
    print("[test_mona_bioaware_embedding_transfer] PASS")


if __name__ == "__main__":
    main()
