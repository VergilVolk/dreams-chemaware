"""Gate 1 FAST: reference-opportunity-balanced exclusive evidence.

Same statistical content as GLM_gate1_opportunity_balanced.py, made fast:
  - vectorized consensus (numpy searchsorted over bin centers, no Python
    per-bin loop);
  - per-(query, molecule) null cache (the matched reference null does not
    depend on the pair partner);
  - pair filter: only pairs containing the positive OR noise's top-1 are
    evaluated (the gate statistics use nothing else);
  - N_DRAWS=4 (report parameter).

Gate criteria (no trained head):
  rescue-set sign rate > 50% (bootstrap CI) and corrected - 2*introduced > 0
  under the pre-registered flip rule.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GLM_RUN_DIR",
                      "data/validation/GLM_gnps_article_benchmark_s2v26/run15")
from GLM_score_challenger_models_on_gnps import parse_mgf_used  # noqa: E402
from GLM_truthblind_fusion_analysis import load_panel  # noqa: E402

BENCH = ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
OUT = ROOT / "data/validation/GLM_candidate_differential_ledger"
M_TOP, TOL = 5, 0.02
N_DRAWS = 4
MIN_REFS = 2
BOOT_N = 10000
RNG = np.random.default_rng(20261007)


def consensus_fast(mz_arrays):
    """Vectorized stable consensus of sorted mz arrays (>=50% of spectra)."""
    if not mz_arrays:
        return np.array([])
    allp = np.concatenate(mz_arrays)
    allp.sort()
    # bin boundaries where gap > TOL
    newbin = np.ones(len(allp), bool)
    newbin[1:] = np.diff(allp) > TOL
    bin_id = np.cumsum(newbin) - 1
    n_bins = bin_id[-1] + 1
    centers = np.zeros(n_bins)
    np.add.at(centers, bin_id, allp)
    counts = np.zeros(n_bins)
    np.add.at(counts, bin_id, 1)
    centers /= counts
    thr = max(1, int(np.ceil(STABILITY_FRAC * len(mz_arrays))))
    present = np.zeros(n_bins, np.int32)
    for mz in mz_arrays:
        lo = np.searchsorted(mz, centers - TOL)
        hi = np.searchsorted(mz, centers + TOL)
        present += (hi > lo).astype(np.int32)
    return centers[present >= thr]


STABILITY_FRAC = 0.5


def match_any(hay, needle, tol=TOL):
    if len(hay) == 0:
        return False
    return (np.searchsorted(hay, needle + tol)
            > np.searchsorted(hay, needle - tol))


def support_count(cons_self, cons_other, q_axes):
    """Query-supported exclusive count of self vs other (frag or loss)."""
    if len(cons_self) == 0:
        return 0
    other = cons_other
    ex_mask = np.array([not match_any(other, x) for x in cons_self])
    if not ex_mask.any():
        return 0
    sup = sum(1 for x in cons_self[ex_mask] if match_any(q_axes, x))
    return sup


def main() -> None:
    g, methods, mol = load_panel("identity_disjoint")
    qptr, mptr = g.query_ptr, g.molecule_ptr
    mol_noise = mol[methods.index("noise_v1")]
    n_q = g.n_queries

    needed = set(map(int, g.query_row))
    pool_mols = {}
    for q in range(n_q):
        lo, hi = int(qptr[q]), int(qptr[q + 1])
        if hi - lo < 2:
            continue
        order = lo + np.argsort(-mol_noise[lo:hi], kind="stable")[:M_TOP]
        for m in order:
            pool_mols.setdefault(int(m), []).extend(
                map(int, g.pair_candidate_row[mptr[m]:mptr[m + 1]]))
    needed.update(r for v in pool_mols.values() for r in v)
    spectra = parse_mgf_used(BENCH / "spectra.mgf", needed)
    print(f"spectra: {len(needed):,}", flush=True)

    prep = {}
    for m, rows in pool_mols.items():
        lst = []
        for r in rows:
            mz, inten, prec = spectra[r]
            mz = np.asarray(mz, float)
            lst.append((np.sort(mz), np.sort(float(prec) - mz)))
        prep[m] = lst

    stats = {"pairs_total": 0, "pairs_evaluable": 0, "rescue_total": 0,
             "rescue_signs": [], "nonsign_rescue": 0, "flip_events": []}

    for q in range(n_q):
        if q % 1000 == 0:
            print(f"  query {q:,}/{n_q:,}", flush=True)
        lo, hi = int(qptr[q]), int(qptr[q + 1])
        if hi - lo < 2:
            continue
        order = lo + np.argsort(-mol_noise[lo:hi], kind="stable")[:M_TOP]
        pos, top1 = lo, int(order[0])
        qmz, qint, qprec = spectra[int(g.query_row[q])]
        qmz = np.sort(np.asarray(qmz, float))
        q_losses = np.sort(float(qprec) - qmz)
        null_cache: dict[int, float] = {}

        def mol_null(m):
            if m not in null_cache:
                lst = prep[m]
                vals = []
                for _ in range(N_DRAWS):
                    perm = RNG.permutation(len(lst))
                    half = len(lst) // 2
                    h1, h2 = perm[:half], perm[half:]
                    if len(h1) == 0 or len(h2) == 0:
                        h1 = h2 = perm
                    f1 = consensus_fast([lst[i][0] for i in h1])
                    f2 = consensus_fast([lst[i][0] for i in h2])
                    l1 = consensus_fast([lst[i][1] for i in h1])
                    l2 = consensus_fast([lst[i][1] for i in h2])
                    nul = (support_count(f1, f2, qmz)
                           + support_count(l1, l2, q_losses))
                    vals.append(nul)
                null_cache[m] = float(np.mean(vals))
            return null_cache[m]

        def balanced(m_self, m_other):
            s_self, s_other = prep.get(m_self), prep.get(m_other)
            if not s_self or not s_other:
                return None
            if len(s_self) < MIN_REFS or len(s_other) < MIN_REFS:
                return None
            n_eq = min(len(s_self), len(s_other))
            obs = []
            for _ in range(N_DRAWS):
                i_s = RNG.choice(len(s_self), n_eq, replace=False)
                i_o = RNG.choice(len(s_other), n_eq, replace=False)
                fs = consensus_fast([s_self[i][0] for i in i_s])
                fo = consensus_fast([s_other[i][0] for i in i_o])
                ls = consensus_fast([s_self[i][1] for i in i_s])
                lo_ = consensus_fast([s_other[i][1] for i in i_o])
                obs.append(support_count(fs, fo, qmz)
                           + support_count(ls, lo_, q_losses))
            return float(np.mean(obs)) - mol_null(m_self)

        for i, a in enumerate(order):
            for b in order[i + 1:]:
                a, b = int(a), int(b)
                if not (a == pos or b == pos or a == top1 or b == top1):
                    continue  # not used by any gate statistic
                stats["pairs_total"] += 1
                bal_a = balanced(a, b)
                if bal_a is None:
                    continue
                bal_b = balanced(b, a)
                if bal_b is None:
                    continue
                stats["pairs_evaluable"] += 1
                l_bal = bal_a - bal_b
                noise_right = top1 == pos
                if a == pos or b == pos:
                    sign_ok = (l_bal > 0) if a == pos else (l_bal < 0)
                    is_sign = l_bal != 0
                    if not noise_right and (a == top1 or b == top1):
                        stats["rescue_total"] += 1
                        if is_sign:
                            stats["rescue_signs"].append(
                                1 if sign_ok else 0)
                        else:
                            stats["nonsign_rescue"] += 1
                if a == top1 or b == top1:
                    supports_flip = ((a == top1 and l_bal < -0.5)
                                     or (b == top1 and l_bal > 0.5))
                    stats["flip_events"].append((bool(noise_right),
                                                 bool(supports_flip)))

    signs = np.asarray(stats["rescue_signs"], int)
    sign_rate = float(signs.mean()) if len(signs) else float("nan")
    if len(signs):
        idx = RNG.integers(0, len(signs), size=(BOOT_N, len(signs)))
        boot = signs[idx].mean(axis=1)
        ci = [round(float(np.percentile(boot, 2.5)) * 100, 2),
              round(float(np.percentile(boot, 97.5)) * 100, 2)]
    else:
        ci = None
    flips = np.asarray(stats["flip_events"], bool)
    corrected = int((~flips[:, 0] & flips[:, 1]).sum())
    introduced = int((flips[:, 0] & flips[:, 1]).sum())
    report = {
        "status": "GLM_GATE1_OPPORTUNITY_BALANCED_FAST",
        "parameters": {"n_draws": N_DRAWS, "min_refs": MIN_REFS,
                       "m_top": M_TOP,
                       "pair_filter": "positive-or-top1 containing only"},
        "pairs_total": stats["pairs_total"],
        "pairs_evaluable": stats["pairs_evaluable"],
        "rescue_pairs": stats["rescue_total"],
        "rescue_signed": int(len(signs)),
        "rescue_nonsign": stats["nonsign_rescue"],
        "rescue_sign_rate_pct": round(sign_rate * 100, 2) if len(signs) else None,
        "rescue_sign_ci95_pct": ci,
        "flip_rule": "balanced L favors competitor by >0.5 on a top-1 pair",
        "corrected": corrected, "introduced": introduced,
        "risk_net_c2i": corrected - 2 * introduced,
        "gate_verdict": {
            "sign_rate_above_50_ci": bool(len(signs) and sign_rate > 0.5
                                          and ci and ci[0] > 50.0),
            "risk_net_positive": bool(corrected - 2 * introduced > 0),
        },
    }
    (OUT / "gate1_balanced.json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
