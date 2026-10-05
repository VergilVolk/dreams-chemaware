from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from combine_noise_corrected_np_action_banks import combine
from noise_final_core import sha256_file


def _write_bank(
    root: Path, status: str, frame: pd.DataFrame, graph: Path, cache: Path,
    graph_report: Path, with_spectra: bool,
) -> None:
    root.mkdir()
    frame.to_csv(root / "training_actions.csv.gz", index=False, compression="gzip")
    provenance = {
        "candidate_graph_sha256": sha256_file(graph),
        "embedding_cache_sha256": sha256_file(cache),
        "graph_report_sha256": sha256_file(graph_report),
    }
    if with_spectra:
        np.savez_compressed(
            root / "action_spectra.npz",
            action_ids=np.asarray(frame.action_id.astype(str), dtype=str),
            action_spectra=np.zeros((len(frame), 101, 2), dtype=np.float32),
        )
        provenance["action_spectra_sha256"] = sha256_file(root / "action_spectra.npz")
    report = {
        "status": status,
        "formal": True,
        "formal_training_authorized": True,
        "outer_formula_fold": 0,
        "contracts": {"action_outcomes_computed": False, "P3_consumed": False},
        "model_provenance": {
            "initialization": "mature_e4_current_geometry",
            "initial_student_checkpoint_sha256": "e4",
            "official_checkpoint_sha256": "official",
        },
        "provenance": provenance,
    }
    (root / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_combination_preserves_rows_and_p_indices() -> None:
    with tempfile.TemporaryDirectory() as value:
        root = Path(value)
        graph_dir = root / "graph"
        graph_dir.mkdir()
        graph = graph_dir / "candidate_graph.npz"
        cache = graph_dir / "official_embeddings.npz"
        report = graph_dir / "report.json"
        graph.write_bytes(b"graph")
        cache.write_bytes(b"cache")
        report.write_text("{}", encoding="utf-8")
        common = {
            "query_row": 10, "query_ik14": "IK", "query_formula": "C",
            "formula_fold": 1, "attenuation": 0.5, "step": 1,
        }
        n = pd.DataFrame([
            common | {"action_id": "n1", "query_index": 0, "selector": "candidate_gradient", "target_path": "1"},
            common | {"action_id": "n2", "query_index": 0, "selector": "role_confounder", "target_path": "2"},
        ])
        p = pd.DataFrame([
            common | {"action_id": "p1", "query_index": 0, "selector": "p_top3_transport_then_union",
                      "target_path": "", "action_kind": "precomputed_spectrum", "action_tensor_index": 0},
        ])
        _write_bank(root / "n", "noise_corrected_full_action_bank_complete", n, graph, cache, report, False)
        _write_bank(root / "p", "noise_corrected_fixed_p_action_bank_complete", p, graph, cache, report, True)
        result = combine(graph_dir, root / "n", root / "p", root / "out")
        combined = pd.read_csv(root / "out" / "training_actions.csv.gz")
        assert result["formal_training_authorized"] is True
        assert len(combined) == 3
        assert set(combined.action_kind) == {"attenuation_path", "precomputed_spectrum"}
        assert result["contracts"]["query_family_action_equal_required"] is True
        with np.load(root / "out" / "action_spectra.npz", allow_pickle=False) as body:
            assert body["action_ids"].astype(str).tolist() == ["p1"]


def test_combination_rejects_initialization_mismatch() -> None:
    with tempfile.TemporaryDirectory() as value:
        root = Path(value)
        graph_dir = root / "graph"
        graph_dir.mkdir()
        graph = graph_dir / "candidate_graph.npz"
        cache = graph_dir / "official_embeddings.npz"
        report = graph_dir / "report.json"
        graph.write_bytes(b"graph"); cache.write_bytes(b"cache"); report.write_text("{}")
        common = {
            "query_row": 10, "query_ik14": "IK", "query_formula": "C", "formula_fold": 1,
            "attenuation": 0.5, "step": 1, "target_path": "1",
        }
        n = pd.DataFrame([common | {"action_id": "n", "query_index": 0, "selector": "candidate_gradient"}])
        p = pd.DataFrame([common | {"action_id": "p", "query_index": 0, "selector": "p_top3_transport_then_union",
                                     "action_kind": "precomputed_spectrum", "action_tensor_index": 0}])
        _write_bank(root / "n", "noise_corrected_full_action_bank_complete", n, graph, cache, report, False)
        _write_bank(root / "p", "noise_corrected_fixed_p_action_bank_complete", p, graph, cache, report, True)
        body = json.loads((root / "p" / "report.json").read_text())
        body["model_provenance"]["initial_student_checkpoint_sha256"] = "wrong"
        (root / "p" / "report.json").write_text(json.dumps(body))
        try:
            combine(graph_dir, root / "n", root / "p", root / "out")
        except RuntimeError as error:
            assert "initialization differs" in str(error)
        else:
            raise AssertionError("mismatched E4 initialization was accepted")


if __name__ == "__main__":
    test_combination_preserves_rows_and_p_indices()
    test_combination_rejects_initialization_mismatch()
    print("PASS=2")
