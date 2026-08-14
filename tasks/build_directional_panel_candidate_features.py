"""Build directional query-to-candidate frozen-panel features.

Unlike the first symmetric pilot, these features explicitly ask what fraction
of the query's frozen-panel evidence is explained by a candidate spectrum.
Every undirected candidate pair is expanded into both retrieval directions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from attribute_large_failure_peaks import load_rules
from audit_e0_observability_residual import greedy_matches, peaks
from build_frozen_panel_pair_features import peak_masks, read_spectra


def directional_features(
    query_spectrum: np.ndarray, query_precursor: float,
    candidate_spectrum: np.ndarray, candidate_precursor: float,
    panel_ids: list[str], nl_values: np.ndarray, tolerance: float,
) -> dict[str, float]:
    q_mz, q_intensity = peaks(query_spectrum)
    c_mz, c_intensity = peaks(candidate_spectrum)
    matches = greedy_matches(q_mz, c_mz, tolerance)
    q_to_c = {q: c for q, c in matches}
    c_to_q = {c: q for q, c in matches}
    q_total = max(float(q_intensity.sum()), 1e-12)
    c_total = max(float(c_intensity.sum()), 1e-12)
    q_norm = q_intensity / q_total
    c_norm = c_intensity / c_total
    q_masks = peak_masks(q_mz, q_intensity, query_precursor, panel_ids, nl_values, tolerance)
    c_masks = peak_masks(c_mz, c_intensity, candidate_precursor, panel_ids, nl_values, tolerance)
    output = {}
    for feature_id in panel_ids:
        safe = feature_id.replace("::", "__").replace("%", "pct").replace("-", "_").replace(".", "p")
        q_mask, c_mask = q_masks[feature_id], c_masks[feature_id]
        q_indices, c_indices = np.flatnonzero(q_mask), np.flatnonzero(c_mask)
        q_matched = np.asarray([index in q_to_c for index in q_indices], bool)
        c_matched = np.asarray([index in c_to_q for index in c_indices], bool)
        q_match_indices = q_indices[q_matched]
        c_match_indices = c_indices[c_matched]
        q_feature_intensity = float(q_norm[q_indices].sum()) if len(q_indices) else 0.0
        c_feature_intensity = float(c_norm[c_indices].sum()) if len(c_indices) else 0.0
        q_matched_intensity = float(q_norm[q_match_indices].sum()) if len(q_match_indices) else 0.0
        c_matched_intensity = float(c_norm[c_match_indices].sum()) if len(c_match_indices) else 0.0
        q_matched_to_same_feature = sum(
            bool(c_mask[q_to_c[q]]) for q in q_match_indices
        )
        q_matched_to_nonfeature = len(q_match_indices) - q_matched_to_same_feature
        output.update({
            # Candidate-independent query prevalence acts only through interactions
            # below and is included for auditability.
            f"dir_{safe}_query_intensity": q_feature_intensity,
            f"dir_{safe}_candidate_intensity": c_feature_intensity,
            f"dir_{safe}_query_matched_intensity": q_matched_intensity,
            f"dir_{safe}_query_unmatched_intensity": q_feature_intensity - q_matched_intensity,
            f"dir_{safe}_candidate_matched_intensity": c_matched_intensity,
            f"dir_{safe}_query_hit_match_fraction": float(q_matched.mean()) if len(q_matched) else 0.0,
            f"dir_{safe}_candidate_hit_match_fraction": float(c_matched.mean()) if len(c_matched) else 0.0,
            f"dir_{safe}_same_feature_pair_fraction": q_matched_to_same_feature / max(1, len(matches)),
            f"dir_{safe}_query_feature_to_other_fraction": q_matched_to_nonfeature / max(1, len(matches)),
            # These are the deployable candidate-specific burden variables.
            f"dir_{safe}_matched_burden_of_query": q_matched_intensity,
            f"dir_{safe}_matched_burden_over_all_query": q_matched_intensity / max(float(q_norm[[q for q, _ in matches]].sum()) if matches else 0.0, 1e-12),
        })
    return output


def process_split(
    split: str, pair_dir: Path, embedding_root: Path, data: Path,
    panel_ids: list[str], nl_values: np.ndarray, tolerance: float,
) -> pd.DataFrame:
    pairs = pd.read_csv(pair_dir / f"{split}_pair_features.csv")
    manifest = pd.read_csv(embedding_root / f"large_observability_embeddings_{split}" / "manifest.csv")
    with h5py.File(data, "r") as handle:
        spectra = read_spectra(handle, manifest["hdf5_row"].to_numpy(np.int64))
    rows = []
    for position, pair in enumerate(pairs.itertuples(index=False), start=1):
        left, right = int(pair.left), int(pair.right)
        common = {
            "formula": pair.formula, "label": int(pair.label),
            "dreams_similarity": float(pair.dreams_similarity),
            "precursor_delta_ppm": float(pair.precursor_delta_ppm),
        }
        for query, candidate in ((left, right), (right, left)):
            rows.append(common | {
                "query": query, "candidate": candidate,
                "query_ik14": manifest.at[query, "ik14"],
                "candidate_ik14": manifest.at[candidate, "ik14"],
            } | directional_features(
                spectra[query], float(manifest.at[query, "precursor_mz"]),
                spectra[candidate], float(manifest.at[candidate, "precursor_mz"]),
                panel_ids, nl_values, tolerance,
            ))
        if position % 10000 == 0:
            print(f"  {split}: {position:,}/{len(pairs):,} undirected pairs", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["discovery", "confirmation"])
    parser.add_argument("--pair-dir", type=Path, default=Path("data/validation/large_observability_residual_audit"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/validation"))
    parser.add_argument("--panel", type=Path, default=Path("data/validation/large_failure_peak_evidence_strata/frozen_test_panel.csv"))
    parser.add_argument("--rules", type=Path, default=Path("dreams/models/chem_aware/chem_rules_data.json"))
    parser.add_argument("--data", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/directional_panel_candidate_features"))
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_ids = pd.read_csv(args.panel)["feature_id"].tolist()
    rules = load_rules(args.rules)
    nl_values = np.asarray(sorted({float(rule["value"]) for rule in rules if rule["category"] == "NL"}), float)
    report = {"status": "directional_frozen_panel_features", "panel_ids": panel_ids, "splits": {}}
    for split in args.splits:
        frame = process_split(split, args.pair_dir, args.embedding_root, args.data, panel_ids, nl_values, args.tolerance)
        frame.to_csv(args.output_dir / f"{split}_directional_features.csv", index=False)
        report["splits"][split] = {
            "directed_pairs": len(frame),
            "queries": int(frame["query"].nunique()),
            "directional_feature_columns": len([column for column in frame if column.startswith("dir_")]),
        }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
