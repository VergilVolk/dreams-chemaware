"""Classical MS2 cosine baseline on the fixed MTBLS1905 blind panel.

This is the non-learned comparator for official DreaMS retrieval.  Candidate
mass windows, direct QC-MS2 queries and connectivity truth are identical to
``eval_mtbls1905_official_dreams_blind.py``.  Fragment matching uses a fixed
0.02 Da tolerance and greedy one-to-one maximum-product matching.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyteomics import mzml


def ik14(value: object) -> str:
    return str(value).split("-", 1)[0]


def parse_mgf(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    records: list[tuple[np.ndarray, np.ndarray]] = []
    peaks: list[tuple[float, float]] = []
    in_block = False
    for raw in path.open(encoding="utf-8", errors="replace"):
        line = raw.strip()
        if line == "BEGIN IONS":
            in_block, peaks = True, []
        elif line == "END IONS":
            if in_block and peaks:
                arr = np.asarray(peaks, dtype=float)
                records.append((arr[:, 0], arr[:, 1]))
            in_block = False
        elif in_block and "=" not in line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
    return records


def unit_intensity(values: np.ndarray) -> np.ndarray:
    values = np.sqrt(np.clip(np.asarray(values, dtype=float), 0, None))
    norm = np.linalg.norm(values)
    return values / norm if norm else values


def greedy_cosine(qmz: np.ndarray, qi: np.ndarray, rmz: np.ndarray, ri: np.ndarray, tolerance: float) -> float:
    pairs: list[tuple[float, int, int]] = []
    qi, ri = unit_intensity(qi), unit_intensity(ri)
    for i, mass in enumerate(qmz):
        possible = np.flatnonzero(np.abs(rmz - mass) <= tolerance)
        pairs.extend((float(qi[i] * ri[j]), i, int(j)) for j in possible)
    used_q, used_r, score = set(), set(), 0.0
    for product, i, j in sorted(pairs, reverse=True):
        if i not in used_q and j not in used_r:
            used_q.add(i); used_r.add(j); score += product
    return score


def load_query_spectra(input_dir: Path, requested: set[tuple[str, str]]) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    found: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for path in input_dir.glob("QC*_MSMS_*.mzML"):
        if path.name.endswith("270_1050.mzML"):
            continue
        for spec in mzml.read(str(path)):
            key = (path.name, str(spec.get("id", "")))
            if key in requested:
                found[key] = (np.asarray(spec["m/z array"], dtype=float), np.asarray(spec["intensity array"], dtype=float))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=Path("data/external/MTBLS1905/reference/blind_connectivity_panel.tsv"))
    parser.add_argument("--target-matches", type=Path, default=Path("data/external/MTBLS1905/qc_ms2/audit/published_target_qc_ms2_matches.tsv"))
    parser.add_argument("--candidate-map", type=Path, default=Path("data/external/MTBLS1905/reference/hnscc_positive_mass_candidate_map.tsv"))
    parser.add_argument("--candidate-mgf", type=Path, default=Path("data/external/MTBLS1905/reference/hnscc_positive_mass_candidates.mgf"))
    parser.add_argument("--candidate-manifest", type=Path, default=Path("data/external/MTBLS1905/reference/hnscc_positive_reference_dreams/manifest.csv"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/external/MTBLS1905/qc_ms2"))
    parser.add_argument("--out", type=Path, default=Path("data/external/MTBLS1905/evaluation/classical_cosine_blind_retrieval.tsv"))
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    panel = pd.read_csv(args.panel, sep="\t"); panel = panel[panel.panel_status.eq("evaluable")].set_index("metabolite")
    hits = pd.read_csv(args.target_matches, sep="\t"); hits = hits[hits.metabolite.isin(panel.index)].copy()
    requested = set(zip(hits.source_file, hits.spectrum_id))
    queries = load_query_spectra(args.input_dir, requested)
    if len(queries) != len(requested):
        raise RuntimeError(f"Loaded {len(queries)} / {len(requested)} requested query spectra")
    refs = parse_mgf(args.candidate_mgf)
    ref_manifest = pd.read_csv(args.candidate_manifest)
    candidate_map = pd.read_csv(args.candidate_map, sep="\t")
    if len(refs) != len(ref_manifest):
        raise RuntimeError("Reference MGF/manifest records mismatch")
    ref_keys = ref_manifest.inchikey.map(ik14).to_numpy()
    rows: list[dict] = []
    for hit in hits.itertuples(index=False):
        truth = panel.loc[hit.metabolite, "truth_ik14"]
        candidates = np.asarray(sorted(candidate_map[candidate_map.target_metabolite.eq(hit.metabolite)].library_record_index.unique()), dtype=int)
        truth_pos = np.where(ref_keys[candidates] == truth)[0]
        if not len(truth_pos):
            raise RuntimeError(f"Truth absent: {hit.metabolite}")
        qmz, qi = queries[(hit.source_file, hit.spectrum_id)]
        scores = np.asarray([greedy_cosine(qmz, qi, refs[i][0], refs[i][1], args.fragment_tolerance) for i in candidates])
        order = np.argsort(-scores, kind="stable"); ranks = np.empty(len(candidates), dtype=int); ranks[order] = np.arange(1, len(candidates) + 1)
        rank = int(ranks[truth_pos].min()); best = int(order[0]); top_row = int(candidates[best])
        rows.append({"metabolite":hit.metabolite,"query_source_file":hit.source_file,"query_spectrum_id":hit.spectrum_id,"candidate_count":int(len(candidates)),"truth_rank":rank,"top1_correct_connectivity":bool(rank==1),"top5_correct_connectivity":bool(rank<=5),"top1_score":float(scores[best]),"truth_best_score":float(scores[truth_pos].max()),"top1_name":ref_manifest.iloc[top_row].get("name",""),"top1_ik14":ref_keys[top_row],"truth_ik14":truth})
    out = pd.DataFrame(rows); args.out.parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.out,sep="\t",index=False)
    by_target = out.groupby("metabolite",as_index=False).agg(queries=("truth_rank","size"),best_rank=("truth_rank","min"),mean_rank=("truth_rank","mean"),top1_any=("top1_correct_connectivity","max"),top5_any=("top5_correct_connectivity","max")); by_target.to_csv(args.out.with_name(args.out.stem+"_by_target.tsv"),sep="\t",index=False)
    report={"scope":"same fixed external HNSCC blind panel; classical greedy MS2 cosine","fragment_tolerance_da":args.fragment_tolerance,"n_query_spectra":int(len(out)),"n_targets":int(out.metabolite.nunique()),"spectrum_top1_accuracy":float(out.top1_correct_connectivity.mean()),"spectrum_top5_accuracy":float(out.top5_correct_connectivity.mean()),"target_top1_any_accuracy":float(by_target.top1_any.mean()),"target_top5_any_accuracy":float(by_target.top5_any.mean())}
    args.out.with_suffix(".json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
