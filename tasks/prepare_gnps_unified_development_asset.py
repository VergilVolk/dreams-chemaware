#!/usr/bin/env python
"""Register and verify the cleaned GNPS benchmark as a development asset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def npz_shape(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as body:
        return {key: list(body[key].shape) for key in body.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path,
        default=Path("data/validation/gnps_gold_silver_10ppm_benchmark_v1"),
    )
    parser.add_argument(
        "--identity-reconstruction", type=Path,
        default=Path("data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz"),
    )
    parser.add_argument(
        "--identity-reconstruction-report", type=Path,
        default=Path("data/validation/GLM_gnps_identity_panel_reconstruction/report.json"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/validation/unified_benchmark_assets/gnps_development_v1.json"),
    )
    args = parser.parse_args()

    report_path = args.benchmark / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "gnps_gold_silver_10ppm_benchmark_v1_sealed":
        raise RuntimeError("GNPS benchmark is not the frozen v1 asset")
    if report.get("protocol", {}).get("construction_model_blind") is not True:
        raise RuntimeError("GNPS construction_model_blind gate is absent")
    reconstructed = json.loads(args.identity_reconstruction_report.read_text(encoding="utf-8"))
    if reconstructed.get("status") != "GLM_GNPS_IDENTITY_PANEL_RECONSTRUCTED_CERTIFIED":
        raise RuntimeError("identity reconstruction is not certified")

    files = {
        "spectra": args.benchmark / "spectra.mgf",
        "manifest": args.benchmark / "manifest.csv.gz",
        "identity_panel": args.identity_reconstruction,
        "formula_panel": args.benchmark / "panel_formula_disjoint.npz",
        "identity_pairs": args.benchmark / "pairs_identity_disjoint.npz",
        "formula_pairs": args.benchmark / "pairs_formula_disjoint.npz",
        "source_report": report_path,
    }
    for name, path in files.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"GNPS asset missing: {name}={path}")

    formula_expected = report["provenance"]["panel_formula_sha256"]
    if sha256(files["formula_panel"]) != formula_expected:
        raise RuntimeError("frozen formula panel checksum changed")
    registry = {
        "schema": "unified_benchmark_asset_registry_v1",
        "dataset_id": "gnps_gold_silver_10ppm_clean_v1",
        "evaluation_role": "development",
        "truth_status": "consumed",
        "allow_final_claim": False,
        "claim_boundary": "large labelled development, ablation and ladder resource only",
        "counts": {
            "accepted_spectra": report["scan"]["accepted_spectra"],
            "identity_queries": report["identity_disjoint"]["queries"],
            "formula_queries": report["formula_disjoint"]["queries"],
        },
        "files": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in files.items()
        },
        "panel_shapes": {
            "identity": npz_shape(files["identity_panel"]),
            "formula": npz_shape(files["formula_panel"]),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
