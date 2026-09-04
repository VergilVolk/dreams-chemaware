"""End-to-end CPU smoke test for BioAware shared-embedding training."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = root / "manifest"; manifest.mkdir()
        token = root / "tokens"; token.mkdir()
        d0 = root / "d0"; d0.mkdir()
        f0 = root / "f0"; f0.mkdir()
        identities = [chr(65 + index) * 14 for index in range(8)]
        rows = np.arange(16, dtype=np.int64)
        dimension, peaks = 16, 5
        rng = np.random.default_rng(19)
        official = rng.normal(size=(16, dimension)).astype(np.float32)
        # Replicates are deliberately close but not identical.
        for index in range(8):
            official[2 * index + 1] = official[2 * index] + 0.05 * official[2 * index + 1]
        official /= np.linalg.norm(official, axis=1, keepdims=True)
        np.savez_compressed(root / "embeddings.npz", rows=rows, embeddings=official)
        np.save(token / "rows.npy", rows)
        np.save(token / "tokens_f16.npy", rng.normal(size=(16, peaks, dimension)).astype(np.float16))
        np.save(token / "mz_f32.npy", rng.uniform(10, 500, size=(16, peaks)).astype(np.float32))
        np.save(token / "intensity_f32.npy", rng.uniform(0, 1, size=(16, peaks)).astype(np.float32))
        np.save(token / "valid.npy", np.ones((16, peaks), dtype=bool))
        (token / "report.json").write_text(json.dumps({
            "status": "noise_final_f1_full_token_cache_complete",
        }), encoding="utf-8")
        row_table = pd.DataFrame({
            "row": rows,
            "ik14": np.repeat(identities, 2),
            "formula": np.repeat([f"F{i}" for i in range(8)], 2),
            "adduct": "[M+H]+", "instrument": np.tile(["Orbitrap", "QTOF"], 8),
            "collision_energy": 30.0, "precursor_mz": np.repeat(np.arange(100, 108), 2),
        })
        row_table.to_csv(manifest / "rows.csv.gz", index=False)
        pair_rows = []
        for index, identity in enumerate(identities):
            pair_rows.extend([
                (identity, identity, "same_identity"),
                (identity, identities[(index + 1) % 8], "near_isomer"),
                (identity, identities[(index + 2) % 8], "reaction_direction_unknown"),
                (identity, identities[(index + 3) % 8], "mass_matched_control"),
            ])
        pair_table = pd.DataFrame(pair_rows, columns=["identity_a", "identity_b", "relation_type"])
        formula_map = {identity: f"F{i}" for i, identity in enumerate(identities)}
        pair_table["formula_a"] = pair_table.identity_a.map(formula_map)
        pair_table["formula_b"] = pair_table.identity_b.map(formula_map)
        pair_table["reaction_ids"] = ""
        pair_table["evidence_count"] = 1
        pair_table["mces_grade"] = -1
        pair_table["formula_fold_a"] = pair_table.formula_a.str[1:].astype(int) % 5
        pair_table["formula_fold_b"] = pair_table.formula_b.str[1:].astype(int) % 5
        pair_table["same_formula"] = pair_table.formula_a == pair_table.formula_b
        pair_table.to_csv(manifest / "identity_pairs.csv.gz", index=False)
        (manifest / "report.json").write_text(json.dumps({
            "formal": True, "contracts": {"reaction_neighbour_is_positive": False},
        }), encoding="utf-8")
        query_rows = np.asarray([0, 2, 4, 6, 8], dtype=np.int64)
        candidate_rows = []
        features = []
        molecule_ik = []
        molecule_formula = []
        for index, query_row in enumerate(query_rows):
            wrong_identity = identities[(index + 1) % 8]
            candidate_rows.extend([query_row, 2 * ((index + 1) % 8)])
            features.extend([[1.0], [float(official[query_row] @ official[2 * ((index + 1) % 8)])]])
            molecule_ik.extend([identities[index], wrong_identity])
            molecule_formula.extend([f"F{index}", formula_map[wrong_identity]])
        graph = root / "graph.npz"
        np.savez_compressed(
            graph,
            feature_names=np.asarray(["dreams_similarity"], dtype=object),
            features=np.asarray(features, dtype=np.float32),
            pair_candidate_row=np.asarray(candidate_rows, dtype=np.int64),
            query_ptr=np.arange(0, 11, 2, dtype=np.int64),
            molecule_ptr=np.arange(11, dtype=np.int64),
            molecule_label=np.tile([1, 0], 5).astype(np.int8),
            molecule_ik14=np.asarray(molecule_ik),
            molecule_formula=np.asarray(molecule_formula),
            molecule_mces_grade=np.tile([-1, 1], 5).astype(np.int8),
            query_row=query_rows,
            query_ik14=np.asarray(identities[:5]),
            query_formula=np.asarray([f"F{i}" for i in range(5)]),
            query_has_near=np.ones(5, dtype=bool),
        )
        fold = np.arange(5, dtype=np.int8)
        baseline = np.ones(5, dtype=np.int16)
        np.savez_compressed(d0 / "manifest.npz", formula_fold=fold, baseline_rank=baseline)
        np.save(f0 / "symmetric_zero_rank.npy", baseline)
        preflight = root / "preflight.json"
        preflight.write_text(json.dumps({"formal": True, "gates": {"pass": True}}), encoding="utf-8")
        output = root / "output"
        subprocess.run([
            sys.executable, str(ROOT / "tasks/train_bioaware_embedding_adapter.py"),
            "--manifest-dir", str(manifest), "--preflight", str(preflight),
            "--graph", str(graph), "--d0-dir", str(d0), "--f0-dir", str(f0),
            "--token-dir", str(token), "--embedding-cache", str(root / "embeddings.npz"),
            "--output-root", str(output), "--outer-fold", "0", "--seed", "9",
            "--epochs", "2", "--steps-per-epoch", "2", "--batch-size", "2",
            "--eval-batch-size", "4", "--device", "cpu", "--smoke",
        ], check=True)
        report = json.loads((output / "fold_0/seed_9/report.json").read_text(encoding="utf-8"))
        assert report["formal"] is False
        assert report["contracts"]["reaction_neighbour_is_positive"] is False
        assert len(report["training"]["history"]) == 2
    print("[test_bioaware_embedding_adapter] PASS")


if __name__ == "__main__":
    main()
