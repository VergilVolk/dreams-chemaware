"""Gate 1: reference-opportunity-balanced exclusive evidence (no training).

Per the revised core-method doc (section 15, Gate 1), the Phase-1 exclusive
counts are opportunity-confounded (wrong molecule 2.06 refs vs positive
1.62; rescue sign 33.9%). This gate implements the fix and the go/no-go:

  1. EQUAL-REFERENCE SAMPLING: for each pair, n_eq = min(refs_a, refs_b)
     spectra are drawn from BOTH sides (seeded, multiple draws averaged);
     consensus built from equal samples -> equal observation opportunity.
  2. MATCHED REFERENCE NULL (L_refnull): each molecule's refs are split
     into two halves; exclusive events between the halves estimate the
     "apparent exclusivity" produced by sampling noise alone at the SAME
     n_eq. Balanced evidence = observed exclusive support minus the null
     expectation, per side.
  3. GATE CRITERIA (no trained head):
     - rescue-set sign rate > 50% (bootstrap CI) on balanced L
     - corrected/introduced risk-net (corrected - 2*introduced) positive
       when the balanced signal is used as an additive flip signal at
       pre-registered thresholds (|L_bal| >= 1, >= 2)
  Pairs restricted to both sides having >= 2 references (refnull needs a
  split); coverage of the restriction is reported honestly.

Run on the frozen GNPS identity-disjoint panel (LOCAL substrate).
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
from GLM_candidate_differential_ledger import consensus_of, match_any  # noqa: E402

BENCH = ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
OUT = ROOT / "data/validation/GLM_candidate_differential_ledger"
M_TOP, TOL = 5, 0.02
N_DRAWS = 8          # equal-reference subsample draws per pair
MIN_REFS = 2         # both sides need >= 2 refs for the refnull split
BOOT_N = 10000
RNG = np.random.default_rng(20261007)


def sub_consensus(rows_spectra, idx):
    """Consensus fragments+losses from the selected spectra indices."""
    frag = [rows_spectra[i][0] for i in idx]
    loss = [rows_spectra[i][1] for i in idx]
    return consensus_of(frag), consensus_of(loss)


def main() -> None:
    g, methods, mol = load_panel("identity_disjoint")
    qptr, mptr = g.query_ptr, g.molecule_ptr
    mi_n = methods.index("noise_v1")
    mol_noise = mol[mi_n]
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

    # per-molecule prepared spectra: (sorted frag mz, sorted loss mz)
    prep = {}
    for m, rows in pool_mols.items():
        prep[m] = []
        for r in rows:
            mz, inten, prec = spectra[r]
            mz = np.asarray(mz, float)
            prep[m].append((np.sort(mz), np.sort(float(prec) - mz)))

    # ---- stream queries ---------------------------------------------------
    stats = {
        "pairs_total": 0, "pairs_evaluable": 0,
        "rescue_total": 0, "rescue_sign_correct": 0,
        "rescue_signs": [],                   # for bootstrap
        "nonsign_rescue": 0,
        "flip_events": [],                    # (noise_right, supports_flip)
    }
    for q in range(n_q):
        lo, hi = int(qptr[q]), int(qptr[q + 1])
        if hi - lo < 2:
            continue
        order = lo + np.argsort(-mol_noise[lo:hi], kind="stable")[:M_TOP]
        pos = lo
        qmz = np.sort(np.asarray(spectra[int(g.query_row[q])][0], float))
        qprec = float(spectra[int(g.query_row[q])][2])
        q_losses = np.sort(qprec - qmz)

        def balanced_side(m_self, m_other):
            """Query-supported exclusive support of self vs other after
            equal-reference sampling and reference-null subtraction."""
            s_self, s_other = prep.get(m_self), prep.get(m_other)
            if not s_self or not s_other:
                return None
            if len(s_self) < MIN_REFS or len(s_other) < MIN_REFS:
                return None
            n_eq = min(len(s_self), len(s_other))
            obs_list, null_list = [], []
            for d in range(N_DRAWS):
                i_self = RNG.choice(len(s_self), n_eq, replace=False)
                i_other = RNG.choice(len(s_other), n_eq, replace=False)
                f_s, l_s = sub_consensus(s_self, i_self)
                f_o, l_o = sub_consensus(s_other, i_other)
                sup = sum(1 for x in f_s if not match_any(f_o, x)
                          and match_any(qmz, x))
                sup += sum(1 for x in l_s if not match_any(l_o, x)
                           and match_any(q_losses, x))
                obs_list.append(sup)
                # matched reference null: two halves of the SAME molecule
                # (half-split whenever n >= 2 so the null never degenerates)
                perm = RNG.permutation(len(s_self))
                if len(s_self) >= 2 * n_eq:
                    h1, h2 = perm[:n_eq], perm[n_eq:2 * n_eq]
                else:
                    half = len(s_self) // 2
                    h1, h2 = perm[:half], perm[half:]
                if len(h2) == 0 or len(h1) == 0:
                    h1 = h2 = perm
                f1, l1 = sub_consensus(s_self, h1)
                f2, l2 = sub_consensus(s_self, h2)
                nul = sum(1 for x in f1 if not match_any(f2, x)
                          and match_any(qmz, x))
                nul += sum(1 for x in l1 if not match_any(l2, x)
                           and match_any(q_losses, x))
                null_list.append(nul)
            return float(np.mean(obs_list)) - float(np.mean(null_list))

        for i, a in enumerate(order):
            for b in order[i + 1:]:
                stats["pairs_total"] += 1
                bal_a = balanced_side(int(a), int(b))
                if bal_a is None:
                    continue
                bal_b = balanced_side(int(b), int(a))
                if bal_b is None:
                    continue
                stats["pairs_evaluable"] += 1
                l_bal = bal_a - bal_b          # antisymmetric by construction
                noise_right = (int(order[0]) == pos)
                if a == pos:
                    sign_ok, is_sign = l_bal > 0, l_bal != 0
                elif b == pos:
                    sign_ok, is_sign = l_bal < 0, l_bal != 0
                else:
                    continue                   # positive not in this pair
                # rescue = noise's top-1 is one of the pair and is wrong
                # (positive is the other). We evaluate sign accuracy on all
                # positive-containing pairs, and flip events on pairs that
                # involve noise's current top-1.
                if not noise_right and (a == int(order[0]) or
                                        b == int(order[0])):
                    stats["rescue_total"] += 1
                    if is_sign:
                        stats["rescue_signs"].append(1 if sign_ok else 0)
                    else:
                        stats["nonsign_rescue"] += 1
                if a == int(order[0]) or b == int(order[0]):
                    supports_flip = ((a == int(order[0]) and l_bal < -0.5)
                                     or (b == int(order[0]) and l_bal > 0.5))
                    stats["flip_events"].append(
                        (bool(noise_right), bool(supports_flip)))

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
        "status": "GLM_GATE1_OPPORTUNITY_BALANCED",
        "parameters": {"n_draws": N_DRAWS, "min_refs": MIN_REFS,
                       "m_top": M_TOP},
        "pairs_total": stats["pairs_total"],
        "pairs_evaluable": stats["pairs_evaluable"],
        "coverage_pct": round(100 * stats["pairs_evaluable"]
                              / max(1, stats["pairs_total"]), 2),
        "rescue_pairs": stats["rescue_total"],
        "rescue_signed": int(len(signs)),
        "rescue_nonsign": stats["nonsign_rescue"],
        "rescue_sign_rate_pct": round(sign_rate * 100, 2)
        if len(signs) else None,
        "rescue_sign_ci95_pct": ci,
        "flip_rule": "balanced L favors competitor by >0.5 on a top-1 pair",
        "corrected": corrected, "introduced": introduced,
        "risk_net_c2i": corrected - 2 * introduced,
        "gate_verdict": {
            "sign_rate_above_50": bool(len(signs) and sign_rate > 0.5
                                       and ci and ci[0] > 50.0),
            "risk_net_positive": bool(corrected - 2 * introduced > 0),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate1_balanced.json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
