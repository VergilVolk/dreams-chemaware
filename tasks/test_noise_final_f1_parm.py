"""Small shared-encoder F1 invariants; no checkpoint or GPU required."""
from __future__ import annotations

import json
import gc
import tempfile
from pathlib import Path

import numpy as np
import torch

from noise_final_core import CandidateGraph, ZeroInitPeakAdapter
from train_noise_final_f1_parm import (
    TokenStore, encode_all, evaluate_challenge, evaluate_full, hard_negative_rows,
    top_negative_rows,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        graph_path = root / "graph.npz"
        np.savez_compressed(
            graph_path,
            feature_names=np.asarray(["dreams_similarity"], dtype=object),
            features=np.asarray([[0.9], [0.7], [0.6], [0.8], [0.2]], dtype=np.float32),
            pair_candidate_row=np.asarray([0, 1, 2, 1, 3], dtype=np.int64),
            query_ptr=np.asarray([0, 3, 5], dtype=np.int64),
            molecule_ptr=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
            molecule_label=np.asarray([1, 0, 0, 1, 0], dtype=np.int8),
            molecule_ik14=np.asarray(["A" * 14, "B" * 14, "C" * 14, "B" * 14, "D" * 14]),
            molecule_formula=np.asarray(["FA", "FB", "FC", "FB", "FD"]),
            molecule_mces_grade=np.asarray([-1, 0, 1, -1, 0], dtype=np.int8),
            query_row=np.asarray([0, 1], dtype=np.int64),
            query_ik14=np.asarray(["A" * 14, "B" * 14]),
            query_formula=np.asarray(["FA", "FB"]),
            query_has_near=np.asarray([True, True]),
        )
        graph = CandidateGraph(graph_path)
        allowed = np.asarray([True, False, True, True, True])
        hard = hard_negative_rows(graph, allowed)
        assert hard.tolist() == [2, 3]
        top = top_negative_rows(graph, allowed, count=2)
        assert top.tolist() == [[2, 2], [3, 3]]

        token_dir = root / "tokens"
        token_dir.mkdir()
        rows = np.arange(4, dtype=np.int64)
        dimension, peaks = 16, 5
        rng = np.random.default_rng(7)
        official = rng.normal(size=(4, dimension)).astype(np.float32)
        official /= np.linalg.norm(official, axis=1, keepdims=True)
        np.savez_compressed(root / "embeddings.npz", rows=rows, embeddings=official)
        np.save(token_dir / "rows.npy", rows)
        np.save(token_dir / "tokens_f16.npy", rng.normal(size=(4, peaks, dimension)).astype(np.float16))
        np.save(token_dir / "mz_f32.npy", rng.uniform(10, 500, size=(4, peaks)).astype(np.float32))
        np.save(token_dir / "intensity_f32.npy", rng.uniform(0, 1, size=(4, peaks)).astype(np.float32))
        np.save(token_dir / "valid.npy", np.ones((4, peaks), dtype=bool))
        (token_dir / "report.json").write_text(json.dumps({
            "status": "noise_final_f1_full_token_cache_complete",
        }), encoding="utf-8")
        store = TokenStore(token_dir, root / "embeddings.npz")
        adapter = ZeroInitPeakAdapter(dimension, hidden_dim=8, delta_bound=0.1)
        before, after = store.adapt(adapter, rows, torch.device("cpu"))
        assert torch.allclose(before, after, atol=1e-6)
        assert torch.allclose(after @ after.T, before @ before.T, atol=1e-6)
        encoded = encode_all(adapter, store, torch.device("cpu"), batch_size=2)
        symmetric_rank = np.asarray([
            1 + int(np.sum((encoded[0] @ encoded[[1, 2]].T) >= (encoded[0] @ encoded[0]))),
            1 + int(np.sum((encoded[1] @ encoded[[3]].T) >= (encoded[1] @ encoded[1]))),
        ], dtype=np.int16)
        full = evaluate_full(encoded, store, graph, np.arange(2), symmetric_rank)
        assert full["summary"]["corrected"] == 0
        assert full["summary"]["introduced"] == 0
        examples = {
            "query_index": np.asarray([0, 1]),
            "positive_row": np.asarray([0, 1]),
            "baseline_rank": symmetric_rank.copy(),
            "has_near": np.asarray([True, True]),
            "identity_weight": np.asarray([1.0, 1.0], dtype=np.float32),
        }
        challenge = evaluate_challenge(encoded, store, graph, examples, np.arange(2))
        assert challenge["summary"]["n_query_identities"] == 2
        assert challenge["summary"]["corrected"] == 0
        assert challenge["summary"]["introduced"] == 0
        del store
        gc.collect()
    print("[test_noise_final_f1_parm] PASS")


if __name__ == "__main__":
    main()
