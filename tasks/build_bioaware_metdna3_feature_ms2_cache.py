#!/usr/bin/env python
"""Attach one outcome-blind representative MS2 spectrum to stable MS1 nodes.

The representative is the MS2 scan closest to the stable node in normalized
precursor-mass/retention-time distance.  Identities, candidate labels and
annotation outcomes are deliberately not loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from metdna3_mzml import iter_ms2_spectra
except ModuleNotFoundError:
    from tasks.metdna3_mzml import iter_ms2_spectra

try:
    from annotation._inference import preprocess_spectrum
except ModuleNotFoundError:  # direct execution from tasks/
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from annotation._inference import preprocess_spectrum


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


class NodeIndex:
    def __init__(self, nodes: pd.DataFrame, ppm: float, rt_sec: float) -> None:
        self.ppm = float(ppm)
        self.rt_sec = float(rt_sec)
        self.tables: dict[str, pd.DataFrame] = {}
        self.mz: dict[str, np.ndarray] = {}
        for polarity, group in nodes.groupby("polarity", sort=False):
            table = group.sort_values(["mz", "rt_sec", "feature_node"]).reset_index(drop=True)
            self.tables[str(polarity)] = table
            self.mz[str(polarity)] = table["mz"].to_numpy(float)

    def nearest(self, polarity: str, precursor_mz: float, rt_sec: float) -> tuple[int, float] | None:
        if polarity not in self.tables or not np.isfinite(precursor_mz) or not np.isfinite(rt_sec):
            return None
        masses = self.mz[polarity]
        tolerance = precursor_mz * self.ppm * 1e-6
        lo = int(np.searchsorted(masses, precursor_mz - tolerance, side="left"))
        hi = int(np.searchsorted(masses, precursor_mz + tolerance, side="right"))
        if lo == hi:
            return None
        candidates = self.tables[polarity].iloc[lo:hi].copy()
        candidates["ppm"] = np.abs(candidates["mz"] - precursor_mz) / precursor_mz * 1e6
        candidates["rt_delta"] = np.abs(candidates["rt_sec"] - rt_sec)
        candidates = candidates[candidates["rt_delta"].le(self.rt_sec)]
        if candidates.empty:
            return None
        candidates["cost"] = (candidates["ppm"] / self.ppm) ** 2 + (
            candidates["rt_delta"] / self.rt_sec
        ) ** 2
        best = candidates.sort_values(["cost", "feature_node"]).iloc[0]
        return int(best["feature_node"]), float(best["cost"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nodes", type=Path,
        default=Path("data/validation/bioaware_metdna3_recursive_headroom_v1/stable_ms1_feature_nodes.csv.gz"),
    )
    parser.add_argument(
        "--mzml-dir", type=Path,
        default=Path("data/external/metdna3_2025/mzml/development"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_feature_ms2_cache_v1"),
    )
    parser.add_argument("--ppm", type=float, default=15.0)
    parser.add_argument("--rt-sec", type=float, default=25.0)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--expected-files", type=int, default=16)
    parser.add_argument("--scope", choices=("development", "internal_rplc", "external"), default="development")
    args = parser.parse_args()
    if not args.nodes.exists():
        raise FileNotFoundError(args.nodes)
    mzml = sorted(args.mzml_dir.glob("*.mzML"))
    if len(mzml) != args.expected_files:
        raise RuntimeError(f"expected {args.expected_files} frozen mzML files, got {len(mzml)}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    nodes = pd.read_csv(args.nodes)
    index = NodeIndex(nodes, args.ppm, args.rt_sec)
    best: dict[int, tuple[float, dict, np.ndarray]] = {}
    eligible_scans = 0
    matched_scans = 0
    for position, path in enumerate(mzml, start=1):
        polarity = "positive" if "_pos_" in path.name else "negative"
        for spectrum in iter_ms2_spectra(path):
            if len(spectrum["mz"]) < 2:
                continue
            eligible_scans += 1
            match = index.nearest(
                polarity, float(spectrum["precursor_mz"]), float(spectrum["rt_sec"])
            )
            if match is None:
                continue
            node, cost = match
            matched_scans += 1
            key = f"{path.name}|{spectrum['spectrum_id']}"
            if node in best and (cost, key) >= (best[node][0], best[node][1]["spectrum_key"]):
                continue
            raw = np.vstack([spectrum["mz"], spectrum["intensity"]])
            tensor = preprocess_spectrum(
                raw, float(spectrum["precursor_mz"]), args.n_highest_peaks
            ).numpy().astype(np.float32)
            best[node] = (cost, {
                "feature_node": node,
                "polarity": polarity,
                "source_file": path.name,
                "spectrum_id": str(spectrum["spectrum_id"]),
                "spectrum_key": key,
                "precursor_mz": float(spectrum["precursor_mz"]),
                "rt_sec": float(spectrum["rt_sec"]),
                "match_cost": cost,
            }, tensor)
        print(f"[feature MS2 {position}/{len(mzml)}] nodes={len(best):,}", flush=True)
    ordered = [best[node] for node in sorted(best)]
    metadata = pd.DataFrame([item[1] for item in ordered])
    tensors = np.stack([item[2] for item in ordered]) if ordered else np.empty((0, 101, 2), np.float32)
    metadata_path = output / "feature_ms2.csv.gz"
    tensor_path = output / "feature_ms2_tensors.npz"
    metadata.to_csv(metadata_path, index=False, compression="gzip")
    np.savez_compressed(tensor_path, feature_ms2_tensor=tensors)
    report = {
        "status": "bioaware_metdna3_feature_ms2_cache_complete",
        "formal": True,
        "scope": args.scope,
        "eligible_ms2_scans": eligible_scans,
        "matched_ms2_scans": matched_scans,
        "stable_nodes": int(len(nodes)),
        "nodes_with_representative_ms2": int(len(metadata)),
        "selection": "minimum normalized precursor-mass/RT distance; spectrum-key tie break",
        "contracts": {
            "identity_labels_loaded": False,
            "candidate_labels_loaded": False,
            "outcomes_loaded": False,
            "one_representative_spectrum_per_node": True,
        },
        "provenance": {
            "nodes_sha256": sha256(args.nodes),
            "mzml_sha256": {path.name: sha256(path) for path in mzml},
            "metadata_sha256": sha256(metadata_path),
            "tensors_sha256": sha256(tensor_path),
        },
        "parameters": {"ppm": args.ppm, "rt_sec": args.rt_sec},
        "claim_limit": "Outcome-blind execution cache; no path or annotation result.",
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
