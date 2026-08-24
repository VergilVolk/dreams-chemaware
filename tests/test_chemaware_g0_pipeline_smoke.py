import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from tasks.audit_chemaware_g0_full_graph import build_pair_rule_features, main as audit_g0
from tasks.build_chemaware_g0_rule_cache import main as build_rule_cache
from tasks.build_g8r_real_error_atlas import Cache


ROOT = Path(__file__).resolve().parent.parent


def write_graph(path: Path) -> None:
    np.savez_compressed(
        path,
        feature_names=np.asarray([
            "dreams_similarity", "sqrt_cosine", "entropy_similarity",
            "neutral_loss_sqrt_cosine",
        ], dtype=object),
        features=np.asarray([
            [0.8, 0.7, 0.7, 0.8], [0.4, 0.3, 0.2, 0.3], [0.2, 0.2, 0.1, 0.2],
            [0.7, 0.6, 0.6, 0.7], [0.5, 0.4, 0.5, 0.4], [0.1, 0.1, 0.1, 0.1],
        ], dtype=np.float32),
        pair_candidate_row=np.arange(6, dtype=np.int64),
        query_ptr=np.asarray([0, 3, 6], dtype=np.int64),
        molecule_ptr=np.arange(7, dtype=np.int64),
        molecule_label=np.asarray([1, 0, 0, 1, 0, 0], dtype=np.int8),
        molecule_ik14=np.asarray(["A" * 14, "B" * 14, "C" * 14,
                                  "D" * 14, "E" * 14, "F" * 14], dtype=object),
        molecule_formula=np.asarray(["C1", "C1", "C1", "C2", "C2", "C2"], dtype=object),
        molecule_mces_grade=np.asarray([-2, 0, 1, -2, -1, 2], dtype=np.int8),
        query_row=np.asarray([0, 3], dtype=np.int64),
        query_ik14=np.asarray(["A" * 14, "D" * 14], dtype=object),
        query_formula=np.asarray(["C1", "C2"], dtype=object),
        query_has_near=np.asarray([True, False]),
    )


class TestChemAwareG0PipelineSmoke(unittest.TestCase):
    def test_rule_cache_and_pair_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            graph_path = root / "graph.npz"
            data_path = root / "tiny.hdf5"
            output = root / "rules.npz"
            write_graph(graph_path)
            with h5py.File(data_path, "w") as handle:
                spectra = np.zeros((6, 2, 8), dtype=np.float32)
                for row in range(6):
                    spectra[row, 0, :3] = [50.0, 81.9894, 100.0 + row]
                    spectra[row, 1, :3] = [1.0, 0.5, 0.2]
                handle.create_dataset("spectrum", data=spectra)
                handle.create_dataset("precursor_mz", data=np.full(6, 100.0))
                handle.create_dataset("PARENT_MASS", data=np.full(6, 98.9922))
                handle.create_dataset("INSTRUMENT_TYPE", data=np.asarray(["Orbitrap"] * 6, dtype="S16"))
                handle.create_dataset("COLLISION_ENERGY", data=np.full(6, 30.0))
                handle.create_dataset(
                    "INCHIKEY",
                    data=np.asarray([f"{'ABCDEF'[row] * 14}-REST" for row in range(6)], dtype="S32"),
                )
            argv = [
                "build_chemaware_g0_rule_cache.py",
                "--graph", str(graph_path),
                "--data", str(data_path),
                "--core-rules", str(ROOT / "dreams/models/chem_aware/chem_rules_data.json"),
                "--massbank-rules", str(ROOT / "dreams/models/chem_aware/chem_rules_massbank.json"),
                "--output", str(output),
                "--max-spectra", "6",
            ]
            with patch.object(sys, "argv", argv):
                build_rule_cache()
            report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "chemaware_g0_rule_cache_smoke")
            self.assertEqual(report["rules"], 3486)
            self.assertEqual(report["semantics_counts"]["precursor_exact_mass_offset"], 79)

            graph = Cache(graph_path)
            with np.load(output, allow_pickle=True) as body:
                pair, names = build_pair_rule_features(
                    graph,
                    np.asarray(body["hdf5_row"], dtype=np.int64),
                    np.asarray(body["packed_rule_hits"], dtype=np.uint8),
                    np.asarray(body["rule_library"], dtype=object).astype(str),
                    np.asarray(body["rule_category"], dtype=object).astype(str),
                    np.asarray(body["rule_semantics"], dtype=object).astype(str),
                    3486,
                    100,
                )
            self.assertEqual(pair.shape, (6, 10))
            self.assertIn("rule_fragment_neutral_loss", names)
            self.assertIn("rule_precursor_offset", names)

            artifact = root / "artifact.json"
            artifact.write_text(json.dumps({
                "configuration": {
                    "normalization": "absolute",
                    "weights": [0.1, 0.0, 0.1, 0.8],
                    "min_support": 1,
                    "min_advantage": 0.0,
                },
                "selected_features": [
                    "dreams_similarity", "sqrt_cosine", "entropy_similarity",
                    "neutral_loss_sqrt_cosine",
                ],
            }), encoding="utf-8")
            p3 = root / "p3"
            p3.mkdir()
            (p3 / "p3_smoke_manifest.json").write_text(
                json.dumps({"queries": [{"ik14": "Z" * 14}]}), encoding="utf-8",
            )
            audit_output = root / "audit.json"
            audit_argv = [
                "audit_chemaware_g0_full_graph.py",
                "--graph", str(graph_path),
                "--rule-cache", str(output),
                "--artifact", str(artifact),
                "--data", str(data_path),
                "--p3-dir", str(p3),
                "--output", str(audit_output),
                "--priority-output", str(root / "priority.csv.gz"),
                "--pair-rule-output", str(root / "pair_rules.npz"),
                "--rule-level-output", str(root / "rule_level.csv.gz"),
                "--allow-small-smoke",
            ]
            with patch.object(sys, "argv", audit_argv):
                audit_g0()
            audit_report = json.loads(audit_output.read_text(encoding="utf-8"))
            self.assertEqual(audit_report["status"], "chemaware_g0_full_graph_smoke")
            self.assertFalse(audit_report["formal"])
            self.assertEqual(audit_report["full_graph"]["queries"], 2)


if __name__ == "__main__":
    unittest.main()
