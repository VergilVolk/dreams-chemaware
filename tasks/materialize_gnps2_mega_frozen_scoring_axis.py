#!/usr/bin/env python3
"""Materialize the score-blind GNPS2 seven-source resource for frozen scoring.

The preflight inventory is the row registry.  This program never changes its
``global_row`` values: the same integer indexes the combined MGF, manifest,
natural 10-ppm Top-k panels, and the labelled pairwise-verification axis.

No model, embedding, prediction, threshold, or outcome-dependent parameter is
read.  ``available_exclusion`` is deliberately labelled incomplete until the
preflight report says that every required consumed source was excluded.
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

from preflight_gnps2_standard_mega_benchmark import (
    _split_query_reference,
    load_source_manifest,
    sha256_file,
    strict_iter_mgf,
)


SCHEMA = "gnps2_mega_frozen_scoring_axis_v1"
PANELS = ("identity_disjoint", "formula_disjoint")


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _panel(frame: pd.DataFrame, *, ppm: float, seed: int) -> tuple[dict[str, np.ndarray], dict]:
    queries, references, split = _split_query_reference(frame, seed)
    domains: dict[tuple[str, str], tuple[pd.DataFrame, np.ndarray]] = {}
    for domain, group in references.groupby(["polarity", "adduct"], sort=False):
        ordered = group.sort_values(["precursor_mz", "global_row"], kind="stable")
        domains[domain] = (ordered, ordered.precursor_mz.to_numpy(float))

    query_rows: list[int] = []
    query_ik14: list[str] = []
    query_formula: list[str] = []
    query_mz: list[float] = []
    query_adduct: list[str] = []
    near: list[bool] = []
    candidate_ik14: list[str] = []
    candidate_formula: list[str] = []
    candidate_label: list[int] = []
    candidate_same_formula: list[bool] = []
    candidate_rows: list[int] = []
    query_ptr = [0]
    molecule_ptr = [0]

    for query in queries.itertuples(index=False):
        pool = domains.get((query.polarity, query.adduct))
        if pool is None:
            continue
        refs, masses = pool
        delta = abs(float(query.precursor_mz)) * ppm * 1e-6
        left = int(np.searchsorted(masses, float(query.precursor_mz) - delta, side="left"))
        right = int(np.searchsorted(masses, float(query.precursor_mz) + delta, side="right"))
        window = refs.iloc[left:right]
        if window.empty or not window.ik14.eq(query.ik14).any():
            continue
        groups = list(window.groupby("ik14", sort=True))
        if len(groups) < 2:
            continue
        query_rows.append(int(query.global_row))
        query_ik14.append(str(query.ik14))
        query_formula.append(str(query.formula))
        query_mz.append(float(query.precursor_mz))
        query_adduct.append(str(query.adduct))
        has_near = False
        for identity, block in groups:
            formula = str(block.formula.iloc[0])
            label = int(str(identity) == str(query.ik14))
            same_formula = formula == str(query.formula)
            has_near |= bool(not label and same_formula)
            candidate_ik14.append(str(identity))
            candidate_formula.append(formula)
            candidate_label.append(label)
            candidate_same_formula.append(same_formula)
            rows = block.sort_values("global_row", kind="stable").global_row.to_numpy(np.int64)
            candidate_rows.extend(rows.tolist())
            molecule_ptr.append(len(candidate_rows))
        near.append(has_near)
        query_ptr.append(len(candidate_ik14))

    arrays = {
        "query_row": np.asarray(query_rows, dtype=np.int64),
        "query_ik14": np.asarray(query_ik14, dtype="U14"),
        "query_formula": np.asarray(query_formula, dtype="U64"),
        "query_precursor_mz": np.asarray(query_mz, dtype=np.float64),
        "query_adduct": np.asarray(query_adduct, dtype="U32"),
        "near_query": np.asarray(near, dtype=bool),
        "independent_positive": np.ones(len(query_rows), dtype=bool),
        "query_ptr": np.asarray(query_ptr, dtype=np.int64),
        "molecule_ptr": np.asarray(molecule_ptr, dtype=np.int64),
        "molecule_label": np.asarray(candidate_label, dtype=np.int8),
        "molecule_ik14": np.asarray(candidate_ik14, dtype="U14"),
        "molecule_formula": np.asarray(candidate_formula, dtype="U64"),
        "molecule_same_formula": np.asarray(candidate_same_formula, dtype=bool),
        "candidate_row": np.asarray(candidate_rows, dtype=np.int64),
    }
    if not np.all(np.add.reduceat(arrays["molecule_label"], arrays["query_ptr"][:-1]) == 1):
        raise RuntimeError("each natural Top-k query must have exactly one positive molecule")
    report = {
        "N_Q": len(query_rows), "N_I": len(set(query_ik14)),
        "N_F": len(set(query_formula)), "N_E": len(candidate_ik14),
        "N_P": len(candidate_rows), "near_queries": int(np.sum(near)), **split,
    }
    return arrays, report


def _write_combined_mgf(
    path: Path, frame: pd.DataFrame, source_manifest: Path,
) -> dict[str, dict[str, object]]:
    _, sources = load_source_manifest(source_manifest)
    by_source = {str(key): body.copy() for key, body in frame.groupby("source_id", sort=False)}
    first_rows = {key: int(body.global_row.min()) for key, body in by_source.items()}
    ordered_sources = sorted(sources, key=lambda item: first_rows.get(item.source_id, 10**18))
    expected_next = 0
    source_report: dict[str, dict[str, object]] = {}
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for source in ordered_sources:
            wanted_frame = by_source.get(source.source_id)
            if wanted_frame is None:
                continue
            wanted = dict(zip(
                wanted_frame.source_row.astype(int), wanted_frame.global_row.astype(int), strict=True,
            ))
            seen: set[int] = set()
            raw_records = 0
            for source_row, fields, peaks in strict_iter_mgf(source.mgf):
                raw_records = source_row + 1
                global_row = wanted.get(int(source_row))
                if global_row is None:
                    continue
                if global_row != expected_next:
                    raise RuntimeError(
                        f"global_row order drifted at {source.source_id}/{source_row}: "
                        f"{global_row} != {expected_next}"
                    )
                seen.add(int(source_row))
                fields = dict(fields)
                fields["TITLE"] = f"GNPS2_MEGA_GLOBAL_ROW_{global_row}"
                fields["SCANS"] = str(global_row)
                fields["GLOBAL_ROW"] = str(global_row)
                fields["SOURCE_ID"] = source.source_id
                fields["SOURCE_ROW"] = str(source_row)
                output.write("BEGIN IONS\n")
                for key, value in fields.items():
                    output.write(f"{key}={value}\n")
                for mz, intensity in peaks:
                    output.write(f"{mz:.8f} {intensity:.8g}\n")
                output.write("END IONS\n\n")
                expected_next += 1
            missing = set(wanted) - seen
            if missing:
                raise RuntimeError(f"{source.source_id} misses {len(missing)} selected source rows")
            source_report[source.source_id] = {
                "source_mgf": str(source.mgf), "source_mgf_sha256": sha256_file(source.mgf),
                "raw_records": raw_records, "selected_records": len(wanted),
            }
    if expected_next != len(frame):
        raise RuntimeError(f"combined MGF wrote {expected_next} != {len(frame)} rows")
    return source_report


def _scope(frame: pd.DataFrame, panel: str, mode: str) -> pd.DataFrame:
    result = frame[frame.library_quality.isin([1, 2, 3])].copy()
    if mode == "available_exclusion":
        overlap = "consumed_identity_overlap" if panel == "identity_disjoint" else "consumed_formula_overlap"
        result = result[~result[overlap] & ~result.consumed_spectrum_overlap].copy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("pre_exclusion", "available_exclusion"), default="available_exclusion")
    parser.add_argument("--candidate-ppm", type=float, default=10.0)
    parser.add_argument("--split-seed", type=int, default=20261102)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    report_path = args.preflight_dir / "report.json"
    inventory_path = args.preflight_dir / "eligible_inventory.csv.gz"
    pair_path = args.preflight_dir / "pre_exclusion_q123_pairwise_verification_indices.npz"
    for path in (report_path, inventory_path, pair_path, args.source_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(inventory_path)
    rows = frame.global_row.to_numpy(np.int64)
    if not np.array_equal(rows, np.arange(len(frame), dtype=np.int64)):
        raise RuntimeError("inventory global_row is not contiguous and zero-based")
    with np.load(pair_path, allow_pickle=False) as body:
        q = np.asarray(body["query_row"], dtype=np.int64)
        r = np.asarray(body["reference_row"], dtype=np.int64)
        y = np.asarray(body["label"], dtype=np.uint8)
        t = np.asarray(body["pair_type"], dtype=np.uint8)
    if not (q.shape == r.shape == y.shape == t.shape) or np.any(q == r):
        raise RuntimeError("pairwise axis is malformed")
    if np.any(q < 0) or np.any(r < 0) or np.any(q >= len(frame)) or np.any(r >= len(frame)):
        raise RuntimeError("pairwise axis references outside global_row")
    if not np.array_equal(y, (t == 1).astype(np.uint8)):
        raise RuntimeError("pairwise label/type contract drifted")
    _, sources = load_source_manifest(args.source_manifest)
    for source in sources:
        if source.required and not source.mgf.is_file():
            raise FileNotFoundError(source.mgf)
    summary = {
        "status": "GNPS2_MEGA_FROZEN_SCORING_PREFLIGHT_PASS",
        "inventory_rows": len(frame), "pairwise_rows": len(q),
        "scope": args.scope,
        "overlap_exclusion_complete": bool(report["overlap_exclusion"]["complete"]),
        "allow_external_claim": bool(report["overlap_exclusion"]["complete"]),
    }
    if args.preflight_only:
        print(json.dumps(summary, indent=2))
        return
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent) as temp:
        staging = Path(temp) / args.output_dir.name
        staging.mkdir()
        manifest = frame.copy()
        manifest.insert(0, "row", manifest.global_row.to_numpy(np.int64))
        manifest.to_csv(staging / "manifest.csv.gz", index=False, compression="gzip")
        sources_report = _write_combined_mgf(staging / "spectra.mgf", frame, args.source_manifest)
        panel_reports = {}
        for panel in PANELS:
            arrays, panel_report = _panel(
                _scope(frame, panel, args.scope), ppm=args.candidate_ppm, seed=args.split_seed,
            )
            panel_path = staging / f"panel_{panel}.npz"
            _atomic_npz(panel_path, **arrays)
            panel_report.update({"path": panel_path.name, "sha256": sha256_file(panel_path)})
            panel_reports[panel] = panel_report
        # The 14.95M artifact stays byte-identical and uses the same global_row registry.
        pair_output = staging / "pairwise_verification_indices.npz"
        pair_output.write_bytes(pair_path.read_bytes())
        # Minimal panel-shaped row registry consumed only by the unchanged
        # DreaMS checkpoint encoder.  It does not define candidate blocks.
        _atomic_npz(
            staging / "panel_pairwise_verification.npz",
            query_row=np.unique(q).astype(np.int64),
            candidate_row=np.unique(r).astype(np.int64),
        )
        output_report = {
            "schema": SCHEMA,
            "status": "GNPS2_MEGA_FROZEN_SCORING_AXIS_COMPLETE",
            "dataset_id": report["dataset_id"], "scope": args.scope,
            "evaluation_role": "development" if not report["overlap_exclusion"]["complete"] else "unopened_external_candidate",
            "allow_final_claim": bool(report["overlap_exclusion"]["complete"]),
            "global_row_contract": "row == global_row == combined MGF zero-based record index",
            "scores_accessed": False, "parameters_selected_after_scores": False,
            "missing_required_exclusions": report["overlap_exclusion"]["missing_required_sources"],
            "inventory_rows": len(frame), "pairwise_rows_pre_exclusion": len(q),
            "panels": panel_reports, "sources": sources_report,
            "inputs": {
                "preflight_report": {"path": str(report_path), "sha256": sha256_file(report_path)},
                "inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
                "pairwise": {"path": str(pair_path), "sha256": sha256_file(pair_path)},
                "source_manifest": {"path": str(args.source_manifest), "sha256": sha256_file(args.source_manifest)},
            },
            "outputs": {
                "manifest.csv.gz": sha256_file(staging / "manifest.csv.gz"),
                "spectra.mgf": sha256_file(staging / "spectra.mgf"),
                "pairwise_verification_indices.npz": sha256_file(pair_output),
                "panel_pairwise_verification.npz": sha256_file(
                    staging / "panel_pairwise_verification.npz"
                ),
            },
            "claim_boundary": (
                "Top-k uses natural 10-ppm candidate panels. Pairwise verification is a separate "
                "ROC/PR axis and its pair count is never a query count. Incomplete consumed-source "
                "exclusions forbid a final external claim."
            ),
        }
        (staging / "report.json").write_text(json.dumps(output_report, indent=2) + "\n", encoding="utf-8")
        staging.replace(args.output_dir)
    print(json.dumps({**summary, "status": "GNPS2_MEGA_FROZEN_SCORING_AXIS_COMPLETE", "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
