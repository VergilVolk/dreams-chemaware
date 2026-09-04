"""Deterministic F0 protocol smoke test; no DreaMS checkpoint or GPU needed."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        graph = root / "graph.npz"
        data = root / "data.hdf5"
        cache = root / "embeddings.npz"
        d0 = root / "d0"
        p3 = root / "p3"
        c1 = root / "c1"
        output = root / "f0"
        for directory in (d0, p3, c1):
            directory.mkdir()

        identities = ["A" * 14, "B" * 14, "C" * 14, "A" * 14, "A" * 14]
        embeddings = np.asarray([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        ], dtype=np.float32)
        np.savez_compressed(cache, rows=np.arange(5), embeddings=embeddings)
        np.savez_compressed(
            graph,
            feature_names=np.asarray(["dreams_similarity"], dtype=object),
            features=np.asarray([[1.0], [0.0], [1.0], [0.0]], dtype=np.float32),
            pair_candidate_row=np.asarray([3, 1, 1, 2], dtype=np.int64),
            query_ptr=np.asarray([0, 2, 4], dtype=np.int64),
            molecule_ptr=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
            molecule_label=np.asarray([1, 0, 1, 0], dtype=np.int8),
            molecule_ik14=np.asarray([identities[0], identities[1], identities[1], identities[2]]),
            molecule_formula=np.asarray(["FA", "FB", "FB", "FC"]),
            molecule_mces_grade=np.asarray([-1, 0, -1, 1], dtype=np.int8),
            query_row=np.asarray([0, 1], dtype=np.int64),
            query_ik14=np.asarray([identities[0], identities[1]]),
            query_formula=np.asarray(["FA", "FB"]),
            query_has_near=np.asarray([True, False]),
        )
        with h5py.File(data, "w") as handle:
            handle.create_dataset("INCHIKEY", data=np.asarray([(value + "-X" * 7).encode() for value in identities]))
        (d0 / "decision.json").write_text(json.dumps({
            "status": "noise_final_d0_manifest_complete", "contains_p2b_fields": False,
        }), encoding="utf-8")
        np.savez_compressed(d0 / "manifest.npz", baseline_rank=np.asarray([1, 1], dtype=np.int16))
        (p3 / "p3_p2_allowed_training_ik14.json").write_text(json.dumps({
            "real_train_primary": {"ik14": sorted(set(identities))}, "p3_query_overlap": 0,
        }), encoding="utf-8")
        pd.DataFrame([{
            "query_index": 0, "query_row": 0, "query_ik14": identities[0],
            "evaluation_positive_row": 3, "teacher_rows": "4",
        }]).to_csv(c1 / "crossfit_examples.csv.gz", index=False, compression="gzip")
        (c1 / "decision.json").write_text(json.dumps({"pass_to_candidate_aware_student": True}), encoding="utf-8")
        official = root / "official.pt"
        architecture = root / "architecture.pt"
        official.write_bytes(b"fixture")
        architecture.write_bytes(b"fixture")

        subprocess.run([
            sys.executable, str(ROOT / "tasks/audit_noise_final_f0_protocol.py"),
            "--graph", str(graph), "--d0-dir", str(d0), "--p3-dir", str(p3),
            "--c1-dir", str(c1), "--data", str(data), "--embedding-cache", str(cache),
            "--official-ckpt", str(official), "--architecture-ckpt", str(architecture),
            "--output-dir", str(output), "--no-formal",
        ], check=True)
        decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
        assert decision["pass"] is True
        assert decision["zero_initialization"]["symmetric_cache_self_reproduction_rank_mismatches"] == 0
        assert decision["p3_isolation"]["training_query_identity_overlap"] == 0
        assert np.load(output / "allowed_molecule_mask.npy").all()
    print("[test_noise_final_f0_protocol] PASS")


if __name__ == "__main__":
    main()
