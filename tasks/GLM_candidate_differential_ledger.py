"""Candidate differential evidence ledger - Phase 1, steps 1-2 + property tests.

Implements the unified algorithm's evidence substrate per
docs/CORE_METHOD_REDESIGN_CANDIDATE_DIFFERENTIAL_EVIDENCE_20261005.md:

  Step 1  top-M candidate PAIRS per query from the frozen candidate graph,
          ordered by the Noise V1 molecule scores (G_ab recorded, plus
          per-method score differentials as auxiliary channels);
  Step 2  candidate-EXCLUSIVE spectral evidence with shared evidence
          contributing exactly zero BY CONSTRUCTION:
            - exclusive stable fragments (consensus per molecule, absent
              from the competitor's consensus within tolerance)
            - exclusive stable neutral losses
            - exclusive peak-mass differences (top peaks)
          Query support: the query spectrum must exhibit the exclusive
          observation. L_ab = support(a-exclusive) - support(b-exclusive).

Property tests (directive step 5, subset):
  P1 antisymmetry    L_ba == -L_ab exactly (channel-wise)
  P2 shared-zero     injecting an extra peak into BOTH molecules' consensus
                     leaves every channel unchanged
  P3 coverage        fraction of pairs with >=1 query-supported exclusive
                     observation, per channel

First information probe (no learning): among pairs where the positive
molecule is one of the two, does sign(L_ab) point to the positive more
often than 50%? Reported separately for pairs where the noise ranking is
wrong (the candidate-differential rescue set) vs right.

Substrate note: the frozen GNPS identity-disjoint panel is the LOCAL
development substrate. The registered MassSpecGym formula-isolated dev
folds are server-side; this ledger's mechanics and property proofs transfer
unchanged, the registered run happens after local verification.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))
from GLM_score_challenger_models_on_gnps import parse_mgf_used  # noqa: E402

PANEL = ROOT / "data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz"
BENCH = ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
BUNDLE = ROOT / "data/validation/GLM_gnps_article_benchmark_s2v26/run15/bundle/method_scores.npz"
OUT = ROOT / "data/validation/GLM_candidate_differential_ledger"

M_TOP = 5           # top-M molecules per query
TOL = 0.02          # fragment match tolerance (Da), project convention
STABILITY = 0.5     # consensus: present in >= this fraction of the molecule's spectra
TOPK_DIFF = 10      # peak-mass differences computed among top-K consensus peaks
SEED = 20261007

rng = np.random.default_rng(SEED)


def consensus_of(mz_arrays):
    """Stable consensus of sorted mz arrays (fragments or losses)."""
    if not mz_arrays:
        return np.array([])
    all_peaks = np.concatenate(mz_arrays)
    all_peaks.sort()
    kept, i = [], 0
    n = len(mz_arrays)
    while i < len(all_peaks):
        j = i
        while j < len(all_peaks) and all_peaks[j] - all_peaks[i] <= TOL:
            j += 1
        distinct = 0
        for mz in mz_arrays:
            lo = np.searchsorted(mz, all_peaks[i] - TOL)
            hi = np.searchsorted(mz, all_peaks[i] + TOL)
            if hi > lo:
                distinct += 1
        if distinct >= max(1, int(np.ceil(STABILITY * n))):
            kept.append(float(np.mean(all_peaks[i:j])))
        i = j
    return np.asarray(kept)


def match_any(haystack: np.ndarray, needle: float, tol: float = TOL) -> bool:
    if len(haystack) == 0:
        return False
    lo = np.searchsorted(haystack, needle - tol)
    hi = np.searchsorted(haystack, needle + tol)
    return hi > lo


def main() -> None:
    with np.load(PANEL, allow_pickle=False) as z:
        panel = {k: np.asarray(z[k]) for k in z.files}
    with np.load(BUNDLE, allow_pickle=False) as b:
        methods = [str(v) for v in b["method_names"]]
        noise_scores = np.asarray(b["scores_identity_disjoint"],
                                  dtype=np.float32)[methods.index("noise_v1")]
        aux = {m: np.asarray(b["scores_identity_disjoint"],
                             dtype=np.float32)[i]
               for i, m in enumerate(methods)}

    n_queries = len(panel["query_row"])
    qptr, mptr = panel["query_ptr"], panel["molecule_ptr"]
    cand_rows = panel["candidate_row"]
    mol_ik = panel["molecule_ik14"].astype(str)

    # molecule-level noise scores (frozen max-pool semantics)
    mol_noise = np.maximum.reduceat(noise_scores, mptr[:-1])

    # spectra needed: all candidate rows of the top-M molecules + query rows
    needed = set(map(int, panel["query_row"]))
    top_pairs = []          # per query: list of (a_glob, b_glob) molecule indices
    mol_local_rows = {}     # global molecule idx -> list of candidate rows
    for q in range(n_queries):
        lo, hi = int(qptr[q]), int(qptr[q + 1])
        if hi - lo < 2:
            top_pairs.append([])
            continue
        order = lo + np.argsort(-mol_noise[lo:hi], kind="stable")[:M_TOP]
        pairs = [(int(a), int(b)) for i, a in enumerate(order)
                 for b in order[i + 1:]]
        top_pairs.append(pairs)
        for m in order:
            mol_local_rows.setdefault(int(m), []).extend(
                map(int, cand_rows[mptr[m]:mptr[m + 1]]))
    needed.update(r for rows in mol_local_rows.values() for r in rows)
    print(f"queries {n_queries:,}; molecules touched {len(mol_local_rows):,}; "
          f"spectra needed {len(needed):,}", flush=True)

    spectra = parse_mgf_used(BENCH / "spectra.mgf", needed)
    print("spectra parsed", flush=True)

    # per-molecule consensus (fragments + losses) cached globally
    cons_cache: dict[int, tuple] = {}

    def consensus(mol_i):
        if mol_i not in cons_cache:
            rows = mol_local_rows[mol_i]
            frag_sorted, loss_sorted, inten_longest = [], [], []
            for r in rows:
                mz, inten, prec = spectra[r]
                mz = np.asarray(mz, float)
                order = np.argsort(mz)
                frag_sorted.append(mz[order])
                inten_longest.append((mz[order], np.asarray(inten, float)[order]))
                loss_sorted.append(np.sort(float(prec) - mz))
            f = consensus_of(frag_sorted)
            l = consensus_of(loss_sorted)
            top_peaks = None
            best = max(inten_longest, key=lambda t: len(t[0])) if inten_longest \
                else (np.array([]), np.array([]))
            if len(best[0]):
                top_idx = np.argsort(-best[1])[:TOPK_DIFF]
                top_peaks = np.sort(best[0][top_idx])
            diffs = (np.abs(top_peaks[:, None] - top_peaks[None, :])
                     if top_peaks is not None and len(top_peaks) > 1
                     else np.array([]))
            diffs = np.unique(np.round(diffs[diffs > TOL], 4))
            cons_cache[mol_i] = (f, l, diffs)
        return cons_cache[mol_i]

    # ---- stream queries, build the ledger -------------------------------
    ledger = {
        "n_queries": n_queries,
        "pairs_total": 0,
        "chan_support": defaultdict(int),     # pairs with >=1 supported exclusive
        "sign_correct": defaultdict(int),     # L_ab sign points at positive
        "sign_total_pos": defaultdict(int),   # pairs where positive is a or b
        "rescue_sign_correct": defaultdict(int),   # noise ranking wrong pairs
        "rescue_total": defaultdict(int),
        # confounder diagnostic: reference-spectrum counts in rescue pairs
        "rescue_wrong_refs": [],              # n spectra of the wrong molecule
        "rescue_pos_refs": [],                # n spectra of the positive
    }
    antisym_ok = True
    shared_zero_ok = True
    checked = 0

    for q in range(n_queries):
        qmz, qint, qprec = spectra[int(panel["query_row"][q])]
        qmz = np.asarray(qmz, float)
        qsorted = np.sort(qmz)
        q_losses = np.sort(float(qprec) - qmz)
        pos_global = int(qptr[q])  # positive molecule is block-first
        for (a, b) in top_pairs[q]:
            if a not in mol_local_rows or b not in mol_local_rows:
                continue
            fa, la, da = consensus(a)
            fb, lb, db = consensus(b)

            def channels(f_self, f_other, l_self, l_other, d_self, d_other):
                """Exclusive evidence of self vs other with query support."""
                s = {}
                ex_self = np.array([x for x in f_self if not match_any(f_other, x)])
                ex_other = np.array([x for x in f_other if not match_any(f_self, x)])
                sup_self = sum(1 for x in ex_self if match_any(qsorted, x))
                sup_other = sum(1 for x in ex_other if match_any(qsorted, x))
                s["frag"] = sup_self - sup_other
                ex_self = np.array([x for x in l_self if not match_any(l_other, x)])
                ex_other = np.array([x for x in l_other if not match_any(l_self, x)])
                sup_self = sum(1 for x in ex_self if match_any(q_losses, x))
                sup_other = sum(1 for x in ex_other if match_any(q_losses, x))
                s["nl"] = sup_self - sup_other
                ex_self = np.array([x for x in d_self
                                    if not match_any(d_other, x, 0.03)])
                ex_other = np.array([x for x in d_other
                                     if not match_any(d_self, x, 0.03)])
                sup_self = sum(1 for x in ex_self if match_any(q_losses, x, 0.03))
                sup_other = sum(1 for x in ex_other if match_any(q_losses, x, 0.03))
                s["diff"] = sup_self - sup_other
                return s

            lab = channels(fa, fb, la, lb, da, db)
            lba = channels(fb, fa, lb, la, db, da)
            # P1 antisymmetry
            for ch in lab:
                if lab[ch] != -lba[ch]:
                    antisym_ok = False
            ledger["pairs_total"] += 1
            for ch, v in lab.items():
                if v != 0:
                    ledger["chan_support"][ch] += 1
                if a == pos_global or b == pos_global:
                    ledger["sign_total_pos"][ch] += 1
                    points_pos = (v > 0) if a == pos_global else (v < 0)
                    if v != 0 and points_pos:
                        ledger["sign_correct"][ch] += 1
                    # noise-wrong pairs = the positive is the LOWER-ranked of
                    # the two (rescue set: differential must beat noise order)
                    if a != pos_global and b == pos_global:
                        ledger["rescue_total"][ch] += 1
                        if v != 0 and v < 0:
                            ledger["rescue_sign_correct"][ch] += 1
                        if ch == "frag":
                            ledger["rescue_wrong_refs"].append(
                                len(mol_local_rows[a]))
                            ledger["rescue_pos_refs"].append(
                                len(mol_local_rows[b]))
            # P2 shared-zero on a random subset: inject a peak into BOTH
            # consensus at a position verified EMPTY in both (so it cannot
            # mask any real exclusive peak)
            checked += 1
            if checked % 97 == 0 and len(fa) and len(fb):
                inj = float(max(fa.max(), fb.max())) + 1.0
                while match_any(fa, inj) or match_any(fb, inj):
                    inj += 1.0
                fa2 = np.sort(np.append(fa, inj))
                fb2 = np.sort(np.append(fb, inj))
                lab2 = channels(fa2, fb2, la, lb, da, db)
                for ch in lab2:
                    if lab2[ch] != lab[ch]:
                        shared_zero_ok = False

    report = {
        "status": "GLM_CANDIDATE_DIFFERENTIAL_LEDGER_P1",
        "substrate": ("frozen GNPS identity-disjoint panel (LOCAL development "
                      "substrate); registered MassSpecGym dev folds are "
                      "server-side"),
        "parameters": {"M_top": M_TOP, "tol": TOL, "stability": STABILITY,
                       "topk_diff": TOPK_DIFF},
        "property_tests": {
            "P1_antisymmetry_exact": antisym_ok,
            "P2_shared_evidence_zero": shared_zero_ok,
            "pairs_checked": checked,
        },
        "pairs_total": ledger["pairs_total"],
        "channels": {},
    }
    for ch in ("frag", "nl", "diff"):
        sup = ledger["chan_support"][ch]
        tot = ledger["pairs_total"]
        tot_pos = ledger["sign_total_pos"][ch]
        cor = ledger["sign_correct"][ch]
        r_tot = ledger["rescue_total"][ch]
        r_cor = ledger["rescue_sign_correct"][ch]
        report["channels"][ch] = {
            "pairs_with_supported_exclusive": sup,
            "coverage_pct": round(100 * sup / max(1, tot), 2),
            "sign_accuracy_pos_pairs_pct": round(100 * cor / max(1, tot_pos), 2)
            if tot_pos else None,
            "n_pos_pairs": tot_pos,
            "rescue_pairs": r_tot,
            "rescue_sign_correct": r_cor,
            "rescue_sign_pct": round(100 * r_cor / max(1, r_tot), 2)
            if r_tot else None,
        }
    rw = np.asarray(ledger["rescue_wrong_refs"], dtype=float)
    rp = np.asarray(ledger["rescue_pos_refs"], dtype=float)
    if len(rw):
        report["confounder_diagnostic"] = {
            "note": ("raw L is size-confounded: molecules with more reference "
                     "spectra accumulate more exclusive peaks; the double "
                     "null of D_ab is designed to remove exactly this"),
            "rescue_wrong_molecule_mean_refs": round(float(rw.mean()), 2),
            "rescue_positive_mean_refs": round(float(rp.mean()), 2),
            "wrong_gt_positive_pct": round(float((rw > rp).mean() * 100), 1),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ledger_p1_identity.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["property_tests"], indent=1))
    print(json.dumps(report["channels"], indent=1))
    print("written:", OUT / "ledger_p1_identity.json")


if __name__ == "__main__":
    main()
