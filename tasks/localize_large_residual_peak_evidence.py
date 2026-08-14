"""Localize auditable peak evidence in large DreaMS residual failures.

For each high-consensus residual query, compare the query with its best
same-molecule spectrum and its highest-scoring DreaMS same-formula error.
Query peaks are partitioned into identity-supporting, confounder-supporting,
shared, and unassigned evidence.  The same partition is repeated in neutral-
loss space.  This is a localization audit, not yet a causal masking claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from audit_e0_observability_residual import greedy_matches, peaks


def query_match_sets(query: np.ndarray, candidate: np.ndarray, tolerance: float) -> set[int]:
    q_mz, _ = peaks(query)
    c_mz, _ = peaks(candidate)
    return {i for i, _ in greedy_matches(q_mz, c_mz, tolerance)}


def neutral_loss_match_sets(
    query: np.ndarray, query_precursor: float,
    candidate: np.ndarray, candidate_precursor: float,
    tolerance: float,
) -> set[int]:
    q_mz, _ = peaks(query)
    c_mz, _ = peaks(candidate)
    q_keep = np.flatnonzero(query_precursor - q_mz > 0)
    c_keep = np.flatnonzero(candidate_precursor - c_mz > 0)
    if not len(q_keep) or not len(c_keep):
        return set()
    q_order = np.argsort(query_precursor - q_mz[q_keep])
    c_order = np.argsort(candidate_precursor - c_mz[c_keep])
    q_loss = (query_precursor - q_mz[q_keep])[q_order]
    c_loss = (candidate_precursor - c_mz[c_keep])[c_order]
    matched_sorted = {i for i, _ in greedy_matches(q_loss, c_loss, tolerance)}
    return {int(q_keep[q_order[i]]) for i in matched_sorted}


def describe_partition(
    spectrum: np.ndarray, positive: set[int], negative: set[int], precursor: float,
    prefix: str,
) -> dict[str, object]:
    mz, intensity = peaks(spectrum)
    total = max(float(intensity.sum()), 1e-12)
    categories = {
        "identity_support": positive - negative,
        "confounder_support": negative - positive,
        "shared": positive & negative,
        "unassigned": set(range(len(mz))) - positive - negative,
    }
    output: dict[str, object] = {}
    for name, indices in categories.items():
        ordered = sorted(indices)
        output[f"{prefix}_{name}_count"] = len(ordered)
        output[f"{prefix}_{name}_intensity_fraction"] = float(intensity[ordered].sum() / total) if ordered else 0.0
        output[f"{prefix}_{name}_indices"] = ";".join(map(str, ordered))
        output[f"{prefix}_{name}_mz"] = ";".join(f"{mz[i]:.5f}" for i in ordered)
        output[f"{prefix}_{name}_neutral_loss"] = ";".join(
            f"{precursor - mz[i]:.5f}" for i in ordered if precursor > mz[i]
        )
        output[f"{prefix}_{name}_intensity"] = ";".join(f"{intensity[i]:.6g}" for i in ordered)
    return output


def read_spectra(handle: h5py.File, hdf5_rows: np.ndarray) -> np.ndarray:
    order = np.argsort(hdf5_rows)
    inverse = np.argsort(order)
    return np.asarray(handle["spectrum"][hdf5_rows[order]])[inverse]


def process_split(
    split: str, embedding_root: Path, audit_root: Path,
    data: Path, tolerance: float, selection: str,
) -> pd.DataFrame:
    manifest = pd.read_csv(embedding_root / f"large_observability_embeddings_{split}" / "manifest.csv")
    audit = pd.read_csv(audit_root / f"{split}_query_audit.csv")
    if selection == "robust_residual":
        audit = audit.loc[audit["robust_model_residual_candidate"].astype(bool)].copy()
    elif selection == "all_dreams_failures":
        audit = audit.loc[~audit["dreams_top1_correct"].astype(bool)].copy()
    elif selection != "all_queries":
        raise ValueError(selection)
    required_indices = np.unique(np.concatenate([
        audit["query_index"].to_numpy(np.int64),
        audit["dreams_best_positive_index"].to_numpy(np.int64),
        audit["dreams_best_negative_index"].to_numpy(np.int64),
    ]))
    hdf5_rows = manifest.loc[required_indices, "hdf5_row"].to_numpy(np.int64)
    with h5py.File(data, "r") as handle:
        loaded = read_spectra(handle, hdf5_rows)
    spectrum_by_index = {int(index): loaded[pos] for pos, index in enumerate(required_indices)}

    output = []
    for row in audit.itertuples(index=False):
        q_idx = int(row.query_index)
        p_idx = int(row.dreams_best_positive_index)
        n_idx = int(row.dreams_best_negative_index)
        query = spectrum_by_index[q_idx]
        positive = spectrum_by_index[p_idx]
        negative = spectrum_by_index[n_idx]
        q_precursor = float(manifest.at[q_idx, "precursor_mz"])
        p_precursor = float(manifest.at[p_idx, "precursor_mz"])
        n_precursor = float(manifest.at[n_idx, "precursor_mz"])

        fragment_positive = query_match_sets(query, positive, tolerance)
        fragment_negative = query_match_sets(query, negative, tolerance)
        loss_positive = neutral_loss_match_sets(query, q_precursor, positive, p_precursor, tolerance)
        loss_negative = neutral_loss_match_sets(query, q_precursor, negative, n_precursor, tolerance)
        q_mz, _ = peaks(query)
        output.append({
            "split": split,
            "query_index": q_idx,
            "positive_index": p_idx,
            "negative_index": n_idx,
            "query_hdf5_row": int(manifest.at[q_idx, "hdf5_row"]),
            "positive_hdf5_row": int(manifest.at[p_idx, "hdf5_row"]),
            "negative_hdf5_row": int(manifest.at[n_idx, "hdf5_row"]),
            "ik14": row.ik14,
            "formula": row.formula,
            "ring_class": row.ring_class,
            "query_smiles": row.smiles,
            "negative_ik14": row.dreams_best_negative_ik14,
            "negative_smiles": row.dreams_best_negative_smiles,
            "dreams_margin": float(row.dreams_margin),
            "raw_consensus_votes": int(row.raw_metric_consensus_votes),
            "audit_quadrant": row.audit_quadrant,
            "robust_model_residual_candidate": bool(row.robust_model_residual_candidate),
            "query_precursor_mz": q_precursor,
            "query_peak_count": len(q_mz),
        } | describe_partition(query, fragment_positive, fragment_negative, q_precursor, "fragment")
          | describe_partition(query, loss_positive, loss_negative, q_precursor, "neutral_loss"))
    return pd.DataFrame(output)


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    def coverage(prefix: str) -> dict[str, object]:
        identity = frame[f"{prefix}_identity_support_count"] > 0
        confounder = frame[f"{prefix}_confounder_support_count"] > 0
        both = identity & confounder
        return {
            "query_spectra": len(frame),
            "molecules": int(frame["ik14"].nunique()),
            "formulas": int(frame["formula"].nunique()),
            "queries_with_identity_support": int(identity.sum()),
            "identity_support_fraction": float(identity.mean()),
            "queries_with_confounder_support": int(confounder.sum()),
            "confounder_support_fraction": float(confounder.mean()),
            "queries_with_both": int(both.sum()),
            "both_fraction": float(both.mean()),
            "median_identity_peak_count": float(frame[f"{prefix}_identity_support_count"].median()),
            "median_confounder_peak_count": float(frame[f"{prefix}_confounder_support_count"].median()),
            "median_identity_intensity_fraction": float(frame[f"{prefix}_identity_support_intensity_fraction"].median()),
            "median_confounder_intensity_fraction": float(frame[f"{prefix}_confounder_support_intensity_fraction"].median()),
        }
    return {"fragment": coverage("fragment"), "neutral_loss": coverage("neutral_loss")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-root", type=Path, default=Path("data/validation"))
    parser.add_argument("--audit-root", type=Path, default=Path("data/validation/large_observability_residual_audit"))
    parser.add_argument("--data", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/large_residual_peak_localization"))
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--splits", nargs="+", default=["discovery", "confirmation"])
    parser.add_argument(
        "--selection", choices=("robust_residual", "all_dreams_failures", "all_queries"),
        default="robust_residual",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_frames = []
    report: dict[str, object] = {
        "status": "large_residual_peak_evidence_localization",
        "definition": "query peak matches DreaMS best identity positive only / best same-formula negative only / both / neither",
        "tolerance_da": args.tolerance,
        "selection": args.selection,
        "claim_limit": "descriptive localization only; causal attribution requires paired targeted masking",
        "splits": {},
    }
    for split in args.splits:
        frame = process_split(split, args.embedding_root, args.audit_root, args.data, args.tolerance, args.selection)
        frame.to_csv(args.output_dir / f"{split}_peak_evidence.csv", index=False)
        report["splits"][split] = summarize(frame)
        split_frames.append(frame)
    combined = pd.concat(split_frames, ignore_index=True)
    combined.to_csv(args.output_dir / "all_peak_evidence.csv", index=False)
    report["combined"] = summarize(combined)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
