"""Build deployable query-candidate features from the frozen 8-item panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from attribute_large_failure_peaks import load_rules
from audit_e0_observability_residual import greedy_matches, peaks


def read_spectra(handle: h5py.File, rows: np.ndarray) -> np.ndarray:
    order = np.argsort(rows)
    inverse = np.argsort(order)
    return np.asarray(handle["spectrum"][rows[order]])[inverse]


def peak_masks(
    mz: np.ndarray, intensity: np.ndarray, precursor: float,
    panel_ids: list[str], neutral_loss_values: np.ndarray, tolerance: float,
) -> dict[str, np.ndarray]:
    relative = intensity / max(float(intensity.max()), 1e-12)
    loss = precursor - mz
    output = {}
    for feature_id in panel_ids:
        if feature_id == "BIN::intensity::5-20%":
            mask = (relative >= 0.05) & (relative < 0.20)
        elif feature_id == "BIN::neutral_loss::0-20":
            mask = (loss > 0) & (loss < 20)
        elif feature_id == "CATEGORY::NL":
            mask = np.zeros(len(mz), dtype=bool)
            valid = loss > 0
            if valid.any():
                mask[valid] = np.min(
                    np.abs(loss[valid, None] - neutral_loss_values[None, :]), axis=1
                ) <= tolerance
        elif feature_id.startswith("CONCEPT::CF::"):
            target = float(feature_id.rsplit("::", 1)[1])
            mask = np.abs(mz - target) <= tolerance
        else:
            raise ValueError(f"Unsupported frozen feature: {feature_id}")
        output[feature_id] = mask
    return output


def pair_panel_features(
    spectrum_a: np.ndarray, precursor_a: float,
    spectrum_b: np.ndarray, precursor_b: float,
    panel_ids: list[str], neutral_loss_values: np.ndarray, tolerance: float,
) -> dict[str, float]:
    mz_a, intensity_a = peaks(spectrum_a)
    mz_b, intensity_b = peaks(spectrum_b)
    matches = greedy_matches(mz_a, mz_b, tolerance)
    matched_a = {i for i, _ in matches}
    matched_b = {j for _, j in matches}
    ia = intensity_a / max(float(intensity_a.sum()), 1e-12)
    ib = intensity_b / max(float(intensity_b.sum()), 1e-12)
    masks_a = peak_masks(mz_a, intensity_a, precursor_a, panel_ids, neutral_loss_values, tolerance)
    masks_b = peak_masks(mz_b, intensity_b, precursor_b, panel_ids, neutral_loss_values, tolerance)
    output = {}
    for feature_id in panel_ids:
        safe = feature_id.replace("::", "__").replace("%", "pct").replace("-", "_").replace(".", "p")
        a, b = masks_a[feature_id], masks_b[feature_id]
        a_indices, b_indices = np.flatnonzero(a), np.flatnonzero(b)
        a_matched = np.asarray([i in matched_a for i in a_indices], bool)
        b_matched = np.asarray([j in matched_b for j in b_indices], bool)
        both = sum(bool(a[i] and b[j]) for i, j in matches)
        either = sum(bool(a[i] or b[j]) for i, j in matches)
        a_matched_intensity = float(ia[a_indices[a_matched]].sum()) if len(a_indices) else 0.0
        b_matched_intensity = float(ib[b_indices[b_matched]].sum()) if len(b_indices) else 0.0
        a_unmatched_intensity = float(ia[a_indices[~a_matched]].sum()) if len(a_indices) else 0.0
        b_unmatched_intensity = float(ib[b_indices[~b_matched]].sum()) if len(b_indices) else 0.0
        a_fraction = float(a_matched.mean()) if len(a_matched) else 0.0
        b_fraction = float(b_matched.mean()) if len(b_matched) else 0.0
        output.update({
            f"panel_{safe}_matched_both_fraction": both / max(1, len(matches)),
            f"panel_{safe}_matched_either_fraction": either / max(1, len(matches)),
            f"panel_{safe}_matched_intensity_min": min(a_matched_intensity, b_matched_intensity),
            f"panel_{safe}_matched_intensity_mean": 0.5 * (a_matched_intensity + b_matched_intensity),
            f"panel_{safe}_unmatched_intensity_mean": 0.5 * (a_unmatched_intensity + b_unmatched_intensity),
            f"panel_{safe}_hit_match_fraction_min": min(a_fraction, b_fraction),
            f"panel_{safe}_hit_match_fraction_mean": 0.5 * (a_fraction + b_fraction),
        })
    return output


def process_split(
    split: str, pair_dir: Path, embedding_root: Path, data: Path,
    panel_ids: list[str], neutral_loss_values: np.ndarray, tolerance: float,
) -> pd.DataFrame:
    pairs = pd.read_csv(pair_dir / f"{split}_pair_features.csv")
    manifest = pd.read_csv(embedding_root / f"large_observability_embeddings_{split}" / "manifest.csv")
    with h5py.File(data, "r") as handle:
        spectra = read_spectra(handle, manifest["hdf5_row"].to_numpy(np.int64))
    rows = []
    for position, row in enumerate(pairs.itertuples(index=False), start=1):
        left, right = int(row.left), int(row.right)
        rows.append({"left": left, "right": right} | pair_panel_features(
            spectra[left], float(manifest.at[left, "precursor_mz"]),
            spectra[right], float(manifest.at[right, "precursor_mz"]),
            panel_ids, neutral_loss_values, tolerance,
        ))
        if position % 10000 == 0:
            print(f"  {split}: {position:,}/{len(pairs):,} pairs", flush=True)
    output = pd.DataFrame(rows)
    if not np.array_equal(output[["left", "right"]].to_numpy(), pairs[["left", "right"]].to_numpy()):
        raise RuntimeError("Pair alignment failure")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["discovery", "confirmation"])
    parser.add_argument("--pair-dir", type=Path, default=Path("data/validation/large_observability_residual_audit"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/validation"))
    parser.add_argument("--panel", type=Path, default=Path("data/validation/large_failure_peak_evidence_strata/frozen_test_panel.csv"))
    parser.add_argument("--rules", type=Path, default=Path("dreams/models/chem_aware/chem_rules_data.json"))
    parser.add_argument("--data", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/frozen_panel_pair_features"))
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.panel)
    panel_ids = panel["feature_id"].tolist()
    rules = load_rules(args.rules)
    nl_values = sorted({float(rule["value"]) for rule in rules if rule["category"] == "NL"})
    neutral_loss_values = np.asarray(nl_values, float)
    report = {
        "status": "frozen_panel_pair_features", "panel_ids": panel_ids,
        "feature_definition": "symmetric query-candidate matched and unmatched evidence; no identity label used",
        "splits": {},
    }
    for split in args.splits:
        output = process_split(
            split, args.pair_dir, args.embedding_root, args.data,
            panel_ids, neutral_loss_values, args.tolerance,
        )
        output.to_csv(args.output_dir / f"{split}_panel_pair_features.csv", index=False)
        report["splits"][split] = {"pairs": len(output), "panel_feature_columns": len(output.columns) - 2}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
