"""
A step: apply the peak-overlap scorer (top_overlap) to the OFFICIAL full-retrieval
protocol and measure the real recall@1 vs DreaMS cosine.

Replicates eval_e0_baseline.evaluate_query_level EXACTLY:
  - 10-ppm precursor window,
  - peak-hash self-exclusion (hashes != hash_q),
  - adduct filter ([M+H]+ primary, [M+Na]+),
  - molecule (IK14) aggregation = max score per IK,
  - eligibility requires BOTH a positive and a negative candidate.
Then adds a second scorer: top-10 peak overlap (top_overlap), the peak signal that
B showed beats cosine (0.9482 vs 0.9020 on the eligible set, transductive — no
fitting, no CV, no train/test split, so it is directly comparable to the frozen
DreaMS cosine).

Reports, per adduct:
  - recall@1/5/10 + MRR on ELIGIBLE queries (the official metric),
  - overall annotation rate (includes trivially-solvable 'no-neg' queries at
    recall@1=1 and unanswerable 'no-pos'/'no-candidate' queries at recall@1=0).

This is NOT the cos+peak LR from eval_e0_peak_correction.py: B showed that mixing
cos into the peak signal only HURTS (0.9391 -> 0.9224). Here the peak scorer stands
alone.

Usage:
  python tasks/eval_e0_peak_full_retrieval.py --dry-run   # first 2000 queries smoke
  python tasks/eval_e0_peak_full_retrieval.py              # full
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from train_causal_chemmask_head import greedy_peak_matches
from eval_e0_peak_correction import load_raw_spectra, precompute_peaks

REPO_ROOT = Path(__file__).resolve().parent.parent
E0_DIR = REPO_ROOT / "data" / "validation" / "e0_baseline"
OUT_DIR = REPO_ROOT / "data" / "validation" / "e0_peak_correction"
DATA_PATH = REPO_ROOT / "data" / "models" / "MassSpecGym_MurckoHist_split.hdf5"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="Evaluate only the first 2000 queries")
    p.add_argument("--fragment-tolerance", type=float, default=0.02)
    p.add_argument("--ppm-tol", type=float, default=10.0)
    p.add_argument("--output-dir", type=str, default=str(OUT_DIR))
    return p.parse_args()


def load_manifest():
    import json
    with open(E0_DIR / "e0_manifest.json") as f:
        manifest = json.load(f)
    pmzs = np.array([float(m["precursor_mz"]) for m in manifest], dtype=np.float64)
    iks = np.array([m["inchikey_14"] for m in manifest])
    hashes = np.array([m["peak_hash"] for m in manifest])
    adducts = np.array([m["adduct"] for m in manifest])
    spectrum_ids = [m["spectrum_id"] for m in manifest]
    return pmzs, iks, hashes, adducts, spectrum_ids


def top_overlap(mzs, top_sets, i, j, tol):
    """top-10 peak overlap fraction; == pair_peak_features(...)[4] in the B script."""
    mz_a, mz_b = mzs[i], mzs[j]
    n_a, n_b = len(mz_a), len(mz_b)
    if n_a == 0 or n_b == 0:
        return 0.0
    matches = greedy_peak_matches(mz_a, mz_b, tol)
    if not matches:
        return 0.0
    top_a, top_b = top_sets[i], top_sets[j]
    top_matches = sum(a in top_a and b in top_b for a, b in matches)
    return top_matches / max(min(10, n_a, n_b), 1)


def evaluate_retrieval(pmzs, iks, hashes, adducts, emb, mzs, top_sets,
                       adduct_filter, ppm_tol, tol, dry_run):
    n_valid = len(pmzs)
    q_indices = np.where(adducts == adduct_filter)[0] if adduct_filter else np.arange(n_valid)
    if dry_run:
        q_indices = q_indices[:2000]

    stats = {
        "cos": {"recall": {1: [], 5: [], 10: []}, "mrr": []},
        "topov": {"recall": {1: [], 5: [], 10: []}, "mrr": []},
    }
    n_eligible = 0
    n_skipped = {"no_candidate": 0, "no_pos": 0, "no_neg": 0}

    for qi in q_indices:
        mz_q = pmzs[qi]
        ik_q = iks[qi]
        hash_q = hashes[qi]
        ppm_da = ppm_tol * 1e-6 * mz_q
        candidate_mask = (
            (np.abs(pmzs - mz_q) <= ppm_da)
            & (np.arange(n_valid) != qi)
            & (hashes != hash_q)
        )
        if adduct_filter:
            candidate_mask &= (adducts == adduct_filter)
        candidates = np.where(candidate_mask)[0]

        if len(candidates) == 0:
            n_skipped["no_candidate"] += 1
            continue

        cand_iks = iks[candidates]
        pos_mask = cand_iks == ik_q
        neg_mask = cand_iks != ik_q
        if not pos_mask.any():
            n_skipped["no_pos"] += 1
            continue
        if not neg_mask.any():
            n_skipped["no_neg"] += 1
            continue

        cand_cos = (emb[qi:qi + 1] * emb[candidates]).sum(axis=-1)
        cand_topov = np.array([top_overlap(mzs, top_sets, qi, c, tol) for c in candidates],
                              dtype=np.float64)

        for name, cand_sims in (("cos", cand_cos), ("topov", cand_topov)):
            ik_to_max = {}
            for c_idx in range(len(candidates)):
                ik = cand_iks[c_idx]
                s = float(cand_sims[c_idx])
                if ik not in ik_to_max or s > ik_to_max[ik]:
                    ik_to_max[ik] = s
            sorted_iks = [m[0] for m in sorted(ik_to_max.items(), key=lambda x: x[1], reverse=True)]
            try:
                rank = sorted_iks.index(ik_q) + 1
            except ValueError:
                rank = len(sorted_iks) + 1
            stats[name]["mrr"].append(1.0 / rank)
            for k in (1, 5, 10):
                stats[name]["recall"][k].append(1.0 if rank <= k else 0.0)
        n_eligible += 1

    out = {}
    total = len(q_indices)
    for name in ("cos", "topov"):
        rec = {k: float(np.mean(stats[name]["recall"][k])) for k in (1, 5, 10)}
        mrr = float(np.mean(stats[name]["mrr"]))
        # overall annotation rate: no-neg -> trivially correct (recall@1=1),
        # no-pos / no-candidate -> unanswerable (recall@1=0).
        overall_r1 = (float(np.sum(stats[name]["recall"][1])) + n_skipped["no_neg"]) / total
        out[name] = {
            "recall": rec,
            "mrr": mrr,
            "overall_recall_at_1": overall_r1,
            "n_eligible": n_eligible,
        }
    out["_meta"] = {
        "adduct_filter": adduct_filter,
        "total_queries": total,
        "n_eligible": n_eligible,
        "n_skipped": n_skipped,
        "dry_run": dry_run,
    }
    return out


def main():
    args = parse_args()
    pmzs, iks, hashes, adducts, spectrum_ids = load_manifest()
    emb = np.load(E0_DIR / "e0_embeddings.npy")

    print("Loading raw spectra + precomputing peaks ...", flush=True)
    spec = load_raw_spectra(spectrum_ids, DATA_PATH)
    mzs, ints, top_sets, _ = precompute_peaks(spec)

    report = {}
    for adduct in ("[M+H]+", "[M+Na]+"):
        print(f"\nEvaluating {adduct} ...", flush=True)
        res = evaluate_retrieval(pmzs, iks, hashes, adducts, emb, mzs, top_sets,
                                 adduct, args.ppm_tol, args.fragment_tolerance, args.dry_run)
        report[adduct] = res
        meta = res["_meta"]
        print(f"  total={meta['total_queries']} eligible={meta['n_eligible']} skipped={meta['n_skipped']}")
        for name in ("cos", "topov"):
            r = res[name]
            print(f"  {name:6s} recall@1={r['recall'][1]:.4f} recall@5={r['recall'][5]:.4f} "
                  f"recall@10={r['recall'][10]:.4f} MRR={r['mrr']:.4f} "
                  f"overall_R@1={r['overall_recall_at_1']:.4f}")
        d = res["topov"]["recall"][1] - res["cos"]["recall"][1]
        do = res["topov"]["overall_recall_at_1"] - res["cos"]["overall_recall_at_1"]
        print(f"  DELTA: eligible recall@1 {d:+.4f}  |  overall R@1 {do:+.4f}")

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / "peak_full_retrieval.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
