#!/usr/bin/env python
"""Compute classical spectral-similarity baselines on a sealed GNPS panel.

Methods (frozen definitions):

- modified_cosine: sqrt-intensity dot-product with greedy one-to-one peak
  matching inside a fixed 0.01 Da fragment tolerance.
- spectral_entropy: entropy similarity (Li et al., Nat Commun 2021):
  S = 1 - (H(merged) - weighted_mean_entropy) / ln(2), matched peaks merged
  by intensity sum, weights equal to each spectrum's total intensity.

Reference spectra of one candidate molecule are aggregated by max, matching
the deployment molecule-max rule.  Query/candidate denominators come from the
frozen panel; no model is trained and no truth beyond the frozen panel labels
is opened.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from build_gnps_gold_silver_10ppm_benchmark import iter_mgf

FRAGMENT_TOLERANCE_DA = 0.01


def parse_needed_spectra(mgf_path: Path, needed: set[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read only the required spectra with the frozen MGF reader.

    The benchmark's ``spectra.mgf`` contains exactly the accepted records, so
    the enumeration index equals the manifest ``row`` used by both panels.
    """
    spectra: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, (_fields, peaks) in enumerate(iter_mgf(mgf_path)):
        if index in needed and peaks:
            spectra[index] = (
                np.asarray([peak[0] for peak in peaks], dtype=np.float64),
                np.asarray([peak[1] for peak in peaks], dtype=np.float64),
            )
        if len(spectra) == len(needed):
            break
    missing = needed - set(spectra)
    if missing:
        raise RuntimeError(f"MGF is missing {len(missing)} required spectra, e.g. {sorted(missing)[:5]}")
    return spectra


def normalized_peaks(mz: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(mz, kind="stable")
    return mz[order], intensity[order] / max(float(intensity.sum()), 1e-12)


def greedy_match(target_mz: np.ndarray, query_mz: np.ndarray) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used_target = np.zeros(len(target_mz), dtype=bool)
    order = np.argsort(-query_mz, kind="stable")
    for qi in order:
        if used_target.all():
            break
        distance = np.abs(target_mz - query_mz[qi])
        distance[used_target] = np.inf
        candidate = int(np.argmin(distance))
        if distance[candidate] <= FRAGMENT_TOLERANCE_DA:
            used_target[candidate] = True
            pairs.append((candidate, int(qi)))
    return pairs


def modified_cosine(
    query: tuple[np.ndarray, np.ndarray], reference: tuple[np.ndarray, np.ndarray]
) -> float:
    q_mz, q_int = normalized_peaks(*query)
    r_mz, r_int = normalized_peaks(*reference)
    pairs = greedy_match(r_mz, q_mz)
    if not pairs:
        return 0.0
    q_sqrt = np.sqrt(q_int)
    r_sqrt = np.sqrt(r_int)
    numerator = float(sum(q_sqrt[qi] * r_sqrt[ri] for ri, qi in pairs))
    denominator = float(np.linalg.norm(q_sqrt) * np.linalg.norm(r_sqrt))
    return numerator / denominator if denominator > 0 else 0.0


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    return float(-(positive * np.log(positive)).sum())


def spectral_entropy_similarity(
    query: tuple[np.ndarray, np.ndarray], reference: tuple[np.ndarray, np.ndarray]
) -> float:
    q_mz, q_int = normalized_peaks(*query)
    r_mz, r_int = normalized_peaks(*reference)
    pairs = greedy_match(r_mz, q_mz)
    if not pairs:
        return 0.0
    q_total = float(q_int.sum())
    r_total = float(r_int.sum())
    weighted_entropy = (q_total * _entropy(q_int) + r_total * _entropy(r_int)) / (q_total + r_total)
    merged_intensities: list[float] = []
    matched_q = {qi for _, qi in pairs}
    matched_r = {ri for ri, _ in pairs}
    for ri, qi in pairs:
        merged_intensities.append(float(q_int[qi] + r_int[ri]))
    merged_intensities.extend(float(q_int[qi]) for qi in range(len(q_int)) if qi not in matched_q)
    merged_intensities.extend(float(r_int[ri]) for ri in range(len(r_int)) if ri not in matched_r)
    merged = np.asarray(merged_intensities)
    merged = merged / merged.sum()
    similarity = 1.0 - (_entropy(merged) - weighted_entropy) / math.log(2.0)
    return float(min(1.0, max(0.0, similarity)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--panel-name", default="panel_formula_disjoint.npz")
    parser.add_argument("--panel-file", type=Path, default=None,
                        help="Explicit panel npz path; overrides --panel-dir/--panel-name.")
    parser.add_argument("--spectra-mgf", type=Path, default=None,
                        help="Explicit spectra.mgf path; defaults to --panel-dir/spectra.mgf.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-label", default="gnps_gold_silver_10ppm")
    args = parser.parse_args()
    panel_dir = args.panel_dir.resolve()
    panel_path = (args.panel_file.resolve() if args.panel_file
                  else panel_dir / args.panel_name)
    spectra_path = (args.spectra_mgf.resolve() if args.spectra_mgf
                    else panel_dir / "spectra.mgf")
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite classical baseline output: {output}")
    output.mkdir(parents=True)

    panel = np.load(panel_path, allow_pickle=False)
    n_queries = len(panel["query_row"])
    query_ptr = panel["query_ptr"]
    molecule_ptr = panel["molecule_ptr"]
    candidate_row = panel["candidate_row"]
    needed = set(int(value) for value in panel["query_row"]) | set(int(value) for value in candidate_row)
    print(f"[classical] parsing {len(needed)} needed spectra from MGF", flush=True)
    spectra = parse_needed_spectra(spectra_path, needed)
    print(f"[classical] parsed {len(spectra)} spectra", flush=True)

    # Each reference spectrum is reused by many queries; normalise once.
    normalized: dict[int, tuple[np.ndarray, np.ndarray]] = {
        index: normalized_peaks(*peaks) for index, peaks in spectra.items()
    }

    query_slot = np.repeat(np.arange(n_queries, dtype=np.int64), np.diff(query_ptr))
    rows: list[dict[str, object]] = []
    for molecule_slot in range(len(panel["molecule_ik14"])):
        left, right = int(molecule_ptr[molecule_slot]), int(molecule_ptr[molecule_slot + 1])
        if right <= left:
            continue
        query_index = int(query_slot[molecule_slot])
        query_spectrum = normalized[int(panel["query_row"][query_index])]
        references = [normalized[int(candidate_row[i])] for i in range(left, right)]
        rows.append({
            "query_id": str(query_index),
            "candidate_id": str(panel["molecule_ik14"][molecule_slot]),
            "is_truth": int(bool(panel["molecule_label"][molecule_slot])),
            "formula_cluster": str(panel["query_formula"][query_index]),
            "source": args.source_label,
            "polarity": "positive",
            "near_query": bool(panel["near_query"][query_index]),
            "modified_cosine": max(modified_cosine(query_spectrum, ref) for ref in references),
            "spectral_entropy": max(
                spectral_entropy_similarity(query_spectrum, ref) for ref in references
            ),
        })
    frame = pd.DataFrame(rows)
    if frame.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("panel exposes duplicate query/identity candidate slots")
    manifest = frame[[
        "query_id", "candidate_id", "is_truth", "formula_cluster",
        "source", "polarity", "near_query",
    ]]
    manifest.to_csv(output / "candidate_manifest.csv.gz", index=False, compression="gzip")
    for method in ("modified_cosine", "spectral_entropy"):
        frame[["query_id", "candidate_id", method]].rename(
            columns={method: "score"}
        ).to_csv(
            output / f"predictions_{method}.csv.gz", index=False, compression="gzip"
        )

    report = {
        "status": "bioaware_tracka_gnps_classical_baseline_inputs",
        "panel": str(panel_path),
        "spectra": str(spectra_path),
        "queries": int(frame["query_id"].nunique()),
        "candidate_rows": int(len(frame)),
        "fragment_tolerance_da": FRAGMENT_TOLERANCE_DA,
        "aggregation": "molecule_max",
        "methods": ["modified_cosine", "spectral_entropy"],
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    temporary.replace(output / "report.json")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
