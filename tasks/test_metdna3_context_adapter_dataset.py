from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source_manifest = ROOT / "data/validation/bioaware_metdna3_dreams_cache_v2/external_spectra.csv.gz"
    if not source_manifest.exists():
        print("[test_metdna3_context_adapter_dataset] SKIP: local MetDNA3 cache absent")
        return
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        seed_dir = base / "seeds"
        output_dir = base / "dataset"
        train_output = base / "train"
        seed_dir.mkdir()
        manifest = pd.read_csv(source_manifest)
        manifest[[
            "truth_ik14", "truth_formula", "adduct", "polarity", "source_file",
            "spectrum_id", "spectrum_key",
        ]].to_csv(seed_dir / "manifest.csv.gz", index=False)
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(len(manifest), 1024)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        np.save(seed_dir / "embeddings.npy", embeddings)
        (seed_dir / "report.json").write_text(json.dumps({
            "formal": True,
            "contracts": {"P2b": "forbidden"},
        }), encoding="utf-8")
        subprocess.run([
            sys.executable, str(ROOT / "tasks/build_metdna3_context_adapter_dataset.py"),
            "--seed-dir", str(seed_dir), "--output-dir", str(output_dir),
        ], cwd=ROOT, check=True, text=True)
        report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
        assert report["formal"] and report["rotation_instances"] == 819
        assert report["contracts"]["heldout_truth_absent_from_context_seeds"]
        with np.load(output_dir / "dataset.npz") as values:
            assert len(values["query_ids"]) == 819
            assert int(values["edge_masks"].sum()) > 0
        subprocess.run([
            sys.executable, str(ROOT / "tasks/train_metdna3_context_adapter_oof.py"),
            "--dataset", str(output_dir / "dataset.npz"),
            "--output-dir", str(train_output), "--device", "cpu",
            "--epochs", "1", "--seeds", "20260830", "--hidden-dim", "16",
            "--relation-dim", "4", "--instance-batch-size", "256",
            "--bootstrap-resamples", "100",
        ], cwd=ROOT, check=True, text=True)
        training_report = json.loads((train_output / "report.json").read_text(encoding="utf-8"))
        assert training_report["formal"] and training_report["queries"] == 117
    print("[test_metdna3_context_adapter_dataset] PASS")


if __name__ == "__main__":
    main()
