"""Validate target peak observability and create formula-isolated pilot splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from pilot_paired_layer_cka import preprocess_spectrum
from run_large_targeted_peak_occlusion import parse_values, target_tokens


def fold(formula: str, folds: int) -> int:
    return int.from_bytes(hashlib.blake2b(formula.encode(), digest_size=8).digest(), "little") % folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/e1/counterfactual_peak_finetune/discovery_oof_peak_evidence_manifest.csv"))
    parser.add_argument("--data", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--manifest", type=Path, default=Path("data/validation/large_observability_embeddings_discovery/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/e1/counterfactual_peak_finetune"))
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validation-fold", type=int, default=0)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    manifest = pd.read_csv(args.manifest).set_index("hdf5_row")
    observed_identity, observed_confounder = [], []
    with h5py.File(args.data, "r") as handle:
        for row in frame.itertuples(index=False):
            raw = np.asarray(handle["spectrum"][int(row.query_hdf5_row)])
            precursor = float(manifest.at[int(row.query_hdf5_row), "precursor_mz"])
            clean = preprocess_spectrum(raw, precursor, args.n_highest_peaks)
            identity = target_tokens(clean, parse_values(row.identity_peak_mz), args.tolerance)
            confounder = target_tokens(clean, parse_values(row.confounder_peak_mz), args.tolerance)
            observed_identity.append(len(identity))
            observed_confounder.append(len(confounder))
    frame["identity_observed_count"] = observed_identity
    frame["confounder_observed_count"] = observed_confounder
    frame["has_identity_intervention"] = frame["identity_observed_count"] > 0
    frame["has_confounder_intervention"] = frame["confounder_observed_count"] > 0
    frame["formula_fold"] = frame["formula"].map(lambda value: fold(str(value), args.folds))
    frame["pilot_split"] = np.where(frame["formula_fold"] == args.validation_fold, "validation", "train")
    frame.to_csv(args.output_dir / "counterfactual_peak_finetune_split.csv", index=False)
    train_formulas = set(frame.loc[frame["pilot_split"] == "train", "formula"])
    validation_formulas = set(frame.loc[frame["pilot_split"] == "validation", "formula"])
    report = {
        "status": "counterfactual_finetune_preflight",
        "examples": len(frame),
        "formula_overlap": len(train_formulas & validation_formulas),
        "splits": {},
        "observability": {
            "identity_examples_observed": int(frame["has_identity_intervention"].sum()),
            "confounder_examples_observed": int(frame["has_confounder_intervention"].sum()),
            "both_interventions_observed": int((frame["has_identity_intervention"] & frame["has_confounder_intervention"]).sum()),
            "identity_target_survival_fraction": float(frame["has_identity_intervention"].mean()),
            "confounder_target_survival_fraction": float(frame["has_confounder_intervention"].mean()),
        },
        "loss_definition": {
            "clean_margin": "cos(query, identity)-cos(query, confounder)",
            "identity_occlusion": "hinge(delta + masked_identity_margin - clean_margin)",
            "confounder_occlusion": "hinge(delta + clean_margin - masked_confounder_margin)",
            "preservation": "1-cos(new_clean_embedding, frozen_official_embedding)",
        },
    }
    for split, group in frame.groupby("pilot_split"):
        report["splits"][split] = {
            "examples": len(group), "molecules": int(group["ik14"].nunique()),
            "formulas": int(group["formula"].nunique()),
            "transition_counts": group["transition"].value_counts().to_dict(),
        }
    (args.output_dir / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
