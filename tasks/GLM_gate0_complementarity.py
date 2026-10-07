"""Gate 0: Noise x Chem x WSE complementarity ledger on the frozen panels.

Consumes a score bundle that already contains noise_v1,
weighted_spectral_entropy and a ChemAware encoder row (any method name
given by --chem-method). Reports, per panel:

  G1  2x2 error intersection (noise x chem) + error-set Jaccard
  G2  conditional oracle headroom: among noise-wrong queries, the fraction
      chem alone fixes, WSE alone fixes, either fixes
  G3  CHEM INDEPENDENT VALUE (the Gate-0 verdict quantity): noise-wrong
      queries fixed by chem AND BY NONE OF {WSE, p2b_noise_v1_frozen,
      entropy_raw_public, spec2vec_gnps_public, denoising_search_public}
      -> count + formula-cluster bootstrap CI
  G4  damage: noise-right queries chem breaks (and whether WSE survives)
  G5  stratification: near flag x chem correction
  G6  reference-count confounder guard: corrections vs non-corrections
      among noise-wrong, by the positive's reference-spectrum count

Fail-closed: missing methods abort; CI is formula-cluster bootstrap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel  # noqa: E402

PANEL_DIR = ROOT / "data/validation/GLM_gnps_identity_panel_reconstruction"
DEFAULT_BUNDLE = ROOT / ("data/validation/GLM_gnps_article_benchmark_"
                         "s2v26/run15/bundle/method_scores.npz")
OTHERS = ("weighted_spectral_entropy", "p2b_noise_v1_frozen",
          "entropy_raw_public", "spec2vec_gnps_public",
          "denoising_search_public")
BOOT_N = 2000
RNG = np.random.default_rng(20261007)


def cluster_ci(mask: np.ndarray, clusters: np.ndarray) -> list[float]:
    uniq = np.unique(clusters)
    index = {c: np.flatnonzero(clusters == c) for c in uniq}
    stats = np.empty(BOOT_N)
    for b in range(BOOT_N):
        picked = RNG.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([index[c] for c in picked])
        stats[b] = mask[rows].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [round(float(lo) * 100, 2), round(float(hi) * 100, 2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--chem-method", required=True)
    args = ap.parse_args()

    report = {"status": "GLM_GATE0_COMPLEMENTARITY_LEDGER",
              "chem_method": args.chem_method, "panels": {}}
    with np.load(args.bundle) as b:
        methods = [str(v) for v in b["method_names"]]
        pair_scores = {p: np.asarray(b[f"scores_{p}"], dtype=np.float32)
                       for p in ("identity_disjoint", "formula_disjoint")}
    for panel in ("identity_disjoint", "formula_disjoint"):
        g = graph_from_panel(PANEL_DIR / f"panel_{panel}.npz")
        mol = np.maximum.reduceat(pair_scores[panel], g.molecule_ptr[:-1],
                                  axis=1)
        M, N = len(methods), g.n_queries
        rank = np.zeros((M, N), dtype=np.int32)
        for q in range(N):
            lo, hi = int(g.query_ptr[q]), int(g.query_ptr[q + 1])
            if hi <= lo:
                continue
            block = mol[:, lo:hi]
            # frozen strict tie policy: ties count AGAINST the positive
            rank[:, q] = 1 + (block[:, 1:] >= block[:, :1]).sum(axis=1)
        correct = rank == 1
        if "noise_v1" not in methods or args.chem_method not in methods:
            raise RuntimeError(f"{panel}: bundle lacks noise_v1 or "
                               f"{args.chem_method}")
        mi = {m: methods.index(m) for m in
              ["noise_v1", args.chem_method] + [o for o in OTHERS
                                                if o in methods]}
        n_ok = correct[mi["noise_v1"]]
        c_ok = correct[mi[args.chem_method]]

        noise_wrong = ~n_ok
        both_wrong = noise_wrong & ~c_ok
        chem_fixes = noise_wrong & c_ok
        chem_breaks = n_ok & ~c_ok

        others_fix = np.zeros_like(noise_wrong)
        for o in OTHERS:
            if o in mi:
                others_fix |= correct[mi[o]]
        chem_independent = chem_fixes & ~others_fix

        with np.load(Path("data/validation/GLM_gnps_identity_panel_"
                          "reconstruction") / f"panel_{panel}.npz") as z:
            near = np.asarray(z["near_query"], bool)
            q_formula = z["query_formula"].astype(str)

        # reference counts of the positive molecule (G6 guard)
        mptr = g.molecule_ptr
        pos = g.query_ptr[:-1]
        pos_ref_count = np.diff(mptr)[pos]

        entry = {
            "G1_intersection": {
                "queries": int(len(n_ok)),
                "noise_wrong": int(noise_wrong.sum()),
                "chem_wrong": int((~c_ok).sum()),
                "both_wrong": int(both_wrong.sum()),
                "error_jaccard": round(float(
                    both_wrong.sum() /
                    max(1, (noise_wrong | (~c_ok)).sum())), 4)},
            "G2_conditional_headroom": {
                "noise_wrong_fixed_by_chem_pct": round(float(
                    chem_fixes.sum() / max(1, noise_wrong.sum())) * 100, 2),
                "noise_wrong_fixed_by_wse_pct": None,
                "noise_wrong_fixed_by_either_pct": None},
            "G3_chem_independent": {
                "count": int(chem_independent.sum()),
                "pct_of_noise_wrong": round(float(
                    chem_independent.sum() / max(1, noise_wrong.sum())) * 100,
                                            2)},
            "G4_damage": {
                "chem_breaks_noise_right": int(chem_breaks.sum()),
                "net_correct_minus_break": int(chem_fixes.sum()
                                               - chem_breaks.sum())},
            "G5_near_strata": {
                "near_noise_wrong": int((noise_wrong & near).sum()),
                "near_chem_fixes": int((chem_fixes & near).sum())},
            "G6_refcount_guard": {
                "mean_pos_refs_when_chem_fixes": round(float(
                    pos_ref_count[chem_fixes].mean()), 3)
                if chem_fixes.any() else None,
                "mean_pos_refs_when_still_wrong": round(float(
                    pos_ref_count[noise_wrong & ~chem_fixes].mean()), 3)
                if (noise_wrong & ~chem_fixes).any() else None},
        }
        if "weighted_spectral_entropy" in mi:
            w_ok = correct[mi["weighted_spectral_entropy"]]
            entry["G2_conditional_headroom"][
                "noise_wrong_fixed_by_wse_pct"] = round(float(
                    (noise_wrong & w_ok).sum()
                    / max(1, noise_wrong.sum())) * 100, 2)
            entry["G2_conditional_headroom"][
                "noise_wrong_fixed_by_either_pct"] = round(float(
                    (noise_wrong & (w_ok | c_ok)).sum()
                    / max(1, noise_wrong.sum())) * 100, 2)
        if chem_independent.any():
            entry["G3_chem_independent"]["pct_of_queries_ci95"] = \
                cluster_ci(chem_independent, q_formula)
        report["panels"][panel] = entry
        print(f"=== {panel} ===")
        print(f"  noise wrong {noise_wrong.sum():,}; chem fixes "
              f"{chem_fixes.sum():,} (independent {chem_independent.sum():,});"
              f" chem breaks {chem_breaks.sum():,}")
        if chem_independent.any():
            ci = entry["G3_chem_independent"]["pct_of_queries_ci95"]
            print(f"  G3 CI: {ci}")

    out = Path("data/validation/GLM_candidate_differential_ledger/"
               "gate0_complementarity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("written:", out)


if __name__ == "__main__":
    main()
