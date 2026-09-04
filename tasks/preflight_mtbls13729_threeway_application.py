"""Fail-closed provenance preflight for the MTBLS13729 three-way application."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from e1_checkpoint_io import torch_load_compat  # noqa: E402
from shared_dreams_inference import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e6-checkpoint", type=Path, required=True)
    parser.add_argument("--p2b-artifact", type=Path, default=Path("data/validation/g8r_p2b_rank_fusion.json"))
    parser.add_argument("--p2b-p3", type=Path, default=Path("data/validation/g8r_p2b_p3_final.json"))
    args = parser.parse_args()
    decision_path = args.e6_checkpoint.parent / "decision.json"
    for path in (args.e6_checkpoint, decision_path, args.p2b_artifact, args.p2b_p3):
        if not path.is_file():
            raise FileNotFoundError(path)

    package = torch_load_compat(args.e6_checkpoint, map_location="cpu")
    if package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder":
        raise RuntimeError("wrong E6 shared-encoder status")
    expected_package = {
        "policy": "curriculum", "action_scope": "errors", "seed": 20260828,
        "outer_fold": 0, "inference_clean_only": True, "P2b_used": False,
    }
    for key, expected in expected_package.items():
        if package.get(key) != expected:
            raise RuntimeError(f"E6 checkpoint contract mismatch: {key}={package.get(key)!r}")

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    configuration = decision.get("configuration", {})
    expected_configuration = {
        "action_selection": "fixed", "views_per_identity": 2,
        "safety_stream_weight": 2.0, "unfreeze_blocks": 1,
        "outer_fold": 0, "seed": 20260828,
    }
    for key, expected in expected_configuration.items():
        if configuration.get(key) != expected:
            raise RuntimeError(f"E6 decision configuration mismatch: {key}={configuration.get(key)!r}")
    required_gates = {
        "clean_recall_positive", "formula_ci_positive", "corrected_gt_introduced",
        "risk_net_positive", "near_nonnegative", "mrr_nonnegative",
        "preservation_ge_0_995",
    }
    gates = decision.get("gates", {})
    failed = sorted(key for key in required_gates if gates.get(key) is not True)
    if failed or decision.get("pass_to_multifold") is not True:
        raise RuntimeError(f"E6 fixed-v2-sw2 failed application preflight gates: {failed}")

    p2b = json.loads(args.p2b_artifact.read_text(encoding="utf-8"))
    p3 = json.loads(args.p2b_p3.read_text(encoding="utf-8"))
    if p2b.get("status") != "g8r_p2b_rank_fusion_frozen":
        raise RuntimeError("P2b artifact is not frozen")
    if p2b.get("p3_used_for_training_or_selection") is not False:
        raise RuntimeError("P2b artifact used P3")
    if p3.get("status") != "g8r_p2b_p3_failed":
        raise RuntimeError("unexpected P2b P3 artifact status; audit before changing protocol")

    panel_counts = {}
    for panel in ("neg_rp", "pos_rp"):
        manifest_path = Path(f"data/mtbls13729/embeddings/{panel}/manifest.csv")
        hdf5 = sorted(Path(f"data/mtbls13729/mzml/{panel}").glob("*.hdf5"))
        if not manifest_path.is_file() or not hdf5:
            raise RuntimeError(f"{panel}: expected an official manifest and at least one HDF5 file")
        manifest = pd.read_csv(manifest_path, usecols=["file_name"])
        official_order = manifest.file_name.astype(str).drop_duplicates().tolist()
        hdf5_order = [path.stem for path in hdf5]
        if len(official_order) != len(hdf5_order) or set(official_order) != set(hdf5_order):
            missing = sorted(set(official_order) - set(hdf5_order))
            extra = sorted(set(hdf5_order) - set(official_order))
            raise RuntimeError(
                f"{panel}: HDF5 set differs from official manifest; missing={missing}, extra={extra}"
            )
        panel_counts[panel] = {
            "spectra": int(len(manifest)),
            "manifest_files": len(official_order),
            "hdf5_files": len(hdf5),
        }

    report = {
        "status": "mtbls13729_threeway_application_preflight_passed",
        "e6": {
            "checkpoint": str(args.e6_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.e6_checkpoint),
            "held_formula_fold_delta_recall1": decision["held_clean"]["delta_recall1"],
            "held_formula_fold_corrected": decision["held_clean"]["corrected"],
            "held_formula_fold_introduced": decision["held_clean"]["introduced"],
            "claim": "experimental one-fold shared embedding; not final external superiority",
        },
        "p2b": {
            "artifact_sha256": sha256_file(args.p2b_artifact),
            "boundary": "downstream frozen candidate expert; not embedding fine-tuning",
        },
        "panels": panel_counts,
        "comparison": "official DreaMS vs E6 shared embedding vs official DreaMS plus P2b",
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
