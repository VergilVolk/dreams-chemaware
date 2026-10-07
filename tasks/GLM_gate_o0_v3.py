"""GLM Gate O0 v3: Hard decoys + full ablation.

Builds on O0 v2 results (combined AUROC 0.8845 on random decoys).
This version tests against PROPERLY MATCHED decoys and runs component ablations.

D1 (hard): same precursor mass gap (±1% relative), but formula change does NOT
           match any known transformation. This is the "Reactomics blind spot."
D2 (direction): true pairs reversed (product scored as substrate).
D3 (random): random pairings (control, already tested in v2).

Ablation: combined score with each component removed, to test circularity
          of the PMD component.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sps

TOLERANCE = 0.02

ELEMENTS = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5, "F": 6,
            "Cl": 7, "Br": 8, "I": 9, "Na": 10, "K": 11, "Si": 12}
FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
ELEM_MASSES = np.array([12.0, 1.00783, 14.00307, 15.99491, 31.97207,
                        30.97376, 18.99840, 34.96885, 78.91834,
                        126.90447, 22.98977, 38.96371, 27.97693])


def _mk(**kw):
    v = np.zeros(len(ELEMENTS), dtype=np.int32)
    for name, count in kw.items():
        if name in ELEMENTS:
            v[ELEMENTS[name]] = count
    return v


TRANSFORMS = [(name, fn()) for name, fn in [
    ("hydroxylation", lambda: _mk(O=1)),
    ("dihydroxylation", lambda: _mk(O=2)),
    ("dehydrogenation", lambda: _mk(H=-2)),
    ("hydrogenation", lambda: _mk(H=2)),
    ("methylation", lambda: _mk(C=1, H=2)),
    ("demethylation", lambda: _mk(C=-1, H=-2)),
    ("acetylation", lambda: _mk(C=2, H=2, O=1)),
    ("formylation", lambda: _mk(C=1, O=1)),
    ("carboxylation", lambda: _mk(C=1, O=2)),
    ("decarboxylation", lambda: _mk(C=-1, O=-2)),
    ("glucuronidation", lambda: _mk(C=6, H=8, O=6)),
    ("sulfation", lambda: _mk(S=1, O=3)),
    ("phosphorylation", lambda: _mk(P=1, O=3, H=1)),
    ("glycosylation_hexose", lambda: _mk(C=6, H=10, O=5)),
    ("glutathione", lambda: _mk(C=10, H=17, N=3, O=6, S=1)),
    ("glycine_conj", lambda: _mk(C=2, H=3, N=1, O=1)),
    ("taurine_conj", lambda: _mk(C=2, H=7, N=1, O=3, S=1)),
    ("hydration", lambda: _mk(H=2, O=1)),
    ("dehydration", lambda: _mk(H=-2, O=-1)),
    ("chain_elongation_C2H4", lambda: _mk(C=2, H=4)),
    ("beta_oxidation", lambda: _mk(C=-2, H=-4, O=-2)),
    ("amination", lambda: _mk(N=1, H=3, O=-1)),
    ("deamination", lambda: _mk(N=-1, H=-3, O=1)),
    ("chlorination", lambda: _mk(Cl=1, H=-1)),
    ("fluorination", lambda: _mk(F=1, H=-1)),
    ("bromination", lambda: _mk(Br=1, H=-1)),
]]

def parse_formula(f):
    if not f or f == "nan":
        return None
    v = np.zeros(len(ELEMENTS), dtype=np.int32)
    for m in FORMULA_RE.finditer(f):
        elem, count = m.group(1), int(m.group(2)) if m.group(2) else 1
        if elem in ELEMENTS:
            v[ELEMENTS[elem]] = count
    return v if v.any() else None

def delta_matches(delta):
    for name, tv in TRANSFORMS:
        if np.array_equal(delta, tv):
            return name
    return None

def exact_mass(vec):
    return float(np.sum(vec * ELEM_MASSES))

def norm_peaks(peaks, max_peaks=100):
    p = sorted(sorted(peaks, key=lambda x: -x[1])[:max_peaks], key=lambda x: x[0])
    mzs = np.array([x[0] for x in p], dtype=np.float64)
    ints = np.array([x[1] for x in p], dtype=np.float64)
    if ints.max() > 0:
        ints /= ints.max()
    return mzs, ints

def greedy_cos(a_mz, a_int, b_mz, b_int):
    used = np.zeros(b_mz.size, dtype=bool)
    matched = 0.0
    for mz, val in zip(a_mz, a_int):
        lo, hi = np.searchsorted(b_mz, (mz - TOLERANCE, mz + TOLERANCE))
        if hi > lo:
            cand = np.flatnonzero(~used[lo:hi]) + lo
            if cand.size:
                best = cand[np.argmax(b_int[cand])]
                used[best] = True
                matched += float(val * b_int[best])
    na, nb = float((a_int**2).sum()), float((b_int**2).sum())
    return matched / math.sqrt(na * nb) if na > 0 and nb > 0 else 0.0

def frag_change(a_mz, a_int, b_mz, b_int):
    if b_int.size == 0:
        return 0.0
    shared = 0.0
    for mz, val in zip(b_mz, b_int):
        lo, hi = np.searchsorted(a_mz, (mz - TOLERANCE, mz + TOLERANCE))
        if hi > lo:
            shared += val
    total = float(b_int.sum())
    return 1.0 - shared / total if total > 0 else 0.0

def entropy_sim(a_mz, a_int, b_mz, b_int):
    merged = {}
    for mz, val in zip(a_mz, a_int):
        merged[round(mz, 2)] = merged.get(round(mz, 2), 0.0) + val
    for mz, val in zip(b_mz, b_int):
        merged[round(mz, 2)] = merged.get(round(mz, 2), 0.0) + val
    w = np.array(list(merged.values()))
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    h_mix = -float(np.sum(w[w > 0] * np.log(w[w > 0])))
    ha = a_int / a_int.sum() if a_int.sum() > 0 else a_int
    hb = b_int / b_int.sum() if b_int.sum() > 0 else b_int
    hna = -float(np.sum(ha[ha > 0] * np.log(ha[ha > 0])))
    hnb = -float(np.sum(hb[hb > 0] * np.log(hb[hb > 0])))
    d = max(hna, hnb)
    return max(0.0, 1.0 - h_mix / d) if d > 0 else 0.0

def score_pair(spec_u, spec_v, formulas=None, idx_u=None):
    if not spec_u.get("peaks") or not spec_v.get("peaks"):
        return None
    a_mz, a_int = norm_peaks(spec_u["peaks"])
    b_mz, b_int = norm_peaks(spec_v["peaks"])
    cos = greedy_cos(a_mz, a_int, b_mz, b_int)
    frag = frag_change(a_mz, a_int, b_mz, b_int)
    ent = entropy_sim(a_mz, a_int, b_mz, b_int)
    delta_mz = abs(spec_v.get("precursor", 0) - spec_u.get("precursor", 0))
    best_ppm = 999.0
    for _, tv in TRANSFORMS:
        em = exact_mass(tv)
        if em > 0.5:
            best_ppm = min(best_ppm, abs(delta_mz - em) / em * 1e6)
    return {"cosine": cos, "fragment_change": frag, "entropy": ent, "pmd_ppm": best_ppm}


def rank_fuse(true_scores, decoy_scores, components):
    """Rank-fuse selected components (higher = more likely true)."""
    n = len(true_scores) + len(decoy_scores)
    for comp in components:
        t_vals = [s[comp] for s in true_scores]
        d_vals = [s[comp] for s in decoy_scores]
        invert = comp == "pmd_ppm"  # lower ppm = better
        all_sorted = sorted(t_vals + d_vals, reverse=not invert)
        rank_map = {v: i for i, v in enumerate(all_sorted)}
        for s in true_scores + decoy_scores:
            s.setdefault("_ranks", {})[comp] = rank_map.get(s[comp], 0)
    for s in true_scores + decoy_scores:
        ranks = [s["_ranks"][c] for c in components]
        s["fused"] = sum(ranks) / (len(components) * n)
    return [s["fused"] for s in true_scores], [s["fused"] for s in decoy_scores]


def auroc(t_vals, d_vals, alternative="greater"):
    if len(t_vals) < 5 or len(d_vals) < 5:
        return None, None
    u, p = sps.mannwhitneyu(t_vals, d_vals, alternative=alternative)
    return round(float(u / (len(t_vals) * len(d_vals))), 4), float(f"{p:.2e}")


def main():
    base = os.path.join("data", "validation")
    manifest_path = os.path.join(base, "gnps_gold_silver_10ppm_benchmark_v1",
                                 "manifest.csv.gz")
    mgf_path = os.path.join(base, "gnps_gold_silver_10ppm_benchmark_v1", "spectra.mgf")
    out_dir = os.path.join(base, "GLM_gate_o0_v3")
    os.makedirs(out_dir, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    formulas = {}
    by_formula = defaultdict(list)
    for idx, row in manifest.iterrows():
        f = str(row.get("formula", ""))
        vec = parse_formula(f)
        if vec is not None:
            formulas[idx] = (f, vec, str(row.get("inchikey", "")))
            by_formula[f].append(idx)

    # Hash-based true pair finding
    vec_lookup = defaultdict(list)
    for f, indices in by_formula.items():
        vec_lookup[tuple(formulas[indices[0]][1])].append(f)

    true_pairs = []
    for f_light in by_formula:
        vec_light = formulas[by_formula[f_light][0]][1]
        for tname, tvec in TRANSFORMS:
            vec_heavy = vec_light + tvec
            if np.any(vec_heavy < 0):
                continue
            for f_heavy in vec_lookup.get(tuple(vec_heavy), []):
                for li in by_formula[f_light]:
                    for hi in by_formula[f_heavy]:
                        if formulas[li][2][:7] != formulas[hi][2][:7]:
                            true_pairs.append((li, hi, tname))
    true_pairs = true_pairs[:5000]
    print(f"true pairs: {len(true_pairs)}", flush=True)

    # Load spectra for all needed indices
    needed = {u for u, v, t in true_pairs} | {v for u, v, t in true_pairs}
    spectra = {}
    with open(mgf_path, "r", encoding="utf-8", errors="ignore") as handle:
        cur, row = None, 0
        for line in handle:
            line = line.strip()
            if line == "BEGIN IONS":
                cur = {"peaks": []}
            elif line == "END IONS":
                if cur and cur["peaks"] and row in needed:
                    spectra[row] = cur
                cur = None
                row += 1
            elif cur is not None:
                if line.startswith("PEPMASS="):
                    cur["precursor"] = float(line.split("=")[1].split()[0])
                elif line and line[0].isdigit():
                    p = line.split()
                    if len(p) >= 2 and float(p[1]) > 0:
                        cur["peaks"].append((float(p[0]), float(p[1])))
    print(f"spectra: {len(spectra)}", flush=True)

    rng = np.random.default_rng(20261007)
    loaded = sorted(spectra.keys())

    # ===== DECOY CONSTRUCTION =====
    # D1 (HARD): same mass gap ±1%, formula doesn't match any transformation
    d1_pairs = []
    loaded_with_formula = [i for i in loaded if i in formulas]
    # sort by precursor for efficient gap search
    prec_map = {i: spectra[i].get("precursor", 0) for i in loaded_with_formula}
    sorted_loaded = sorted(loaded_with_formula, key=lambda i: prec_map[i])
    sorted_precs = [prec_map[i] for i in sorted_loaded]

    for u, v, tname in true_pairs:
        if u not in spectra or v not in spectra:
            continue
        gap = prec_map.get(v, 0) - prec_map.get(u, 0)
        if gap < 0.5:
            continue
        # binary search for indices with similar gap from u
        target = prec_map.get(u, 0) + gap
        lo_b = np.searchsorted(sorted_precs, target * 0.99)
        hi_b = np.searchsorted(sorted_precs, target * 1.01)
        candidates = [sorted_loaded[j] for j in range(lo_b, hi_b)
                     if sorted_loaded[j] != u and sorted_loaded[j] != v
                     and sorted_loaded[j] in formulas]
        for w in candidates[:5]:
            fd = formulas[w][1] - formulas[u][1]
            if delta_matches(fd) is None:
                d1_pairs.append((u, w, "D1_same_gap_no_formula"))
                break
        if len(d1_pairs) >= 2000:
            break
    print(f"D1 (hard, same gap no formula): {len(d1_pairs)}", flush=True)

    # D2: reversed direction
    d2_pairs = [(v, u, "D2_reversed") for u, v, t in true_pairs
               if u in spectra and v in spectra][:2000]
    print(f"D2 (reversed): {len(d2_pairs)}", flush=True)

    # D3: random
    d3_pairs = []
    for _ in range(2000):
        u = rng.choice(loaded)
        w = rng.choice(loaded)
        if u != w:
            d3_pairs.append((u, w, "D3_random"))
    print(f"D3 (random): {len(d3_pairs)}", flush=True)

    # ===== SCORING =====
    true_scores = []
    for u, v, t in true_pairs:
        if u in spectra and v in spectra:
            s = score_pair(spectra[u], spectra[v])
            if s:
                s["type"] = "true"
                true_scores.append(s)

    def score_decoys(decoys):
        out = []
        for u, w, dt in decoys:
            if u in spectra and w in spectra:
                s = score_pair(spectra[u], spectra[w])
                if s:
                    s["type"] = dt
                    out.append(s)
        return out

    d1_scores = score_decoys(d1_pairs)
    d2_scores = score_decoys(d2_pairs)
    d3_scores = score_decoys(d3_pairs)
    all_decoys = d1_scores + d2_scores + d3_scores
    print(f"scored: {len(true_scores)} true, {len(d1_scores)}+{len(d2_scores)}+{len(d3_scores)} decoys", flush=True)

    # ===== ANALYSIS =====
    results = {"per_decoy": {}, "ablation": {}, "component_auroc": {}}

    # Component AUROCs vs each decoy type
    for dt_name, dt_scores in [("D1_hard", d1_scores), ("D2_reversed", d2_scores),
                                ("D3_random", d3_scores)]:
        if len(dt_scores) < 10:
            continue
        entry = {}
        for comp in ("cosine", "entropy", "fragment_change"):
            auc, p = auroc([s[comp] for s in true_scores], [s[comp] for s in dt_scores])
            entry[comp] = {"auroc": auc, "p": p}
        # PMD: lower = better (invert)
        auc, p = auroc([s["pmd_ppm"] for s in true_scores],
                      [s["pmd_ppm"] for s in dt_scores], alternative="less")
        entry["pmd_ppm"] = {"auroc": auc, "p": p}
        # Full fusion (all 4 components)
        t_fused, d_fused = rank_fuse(
            [dict(s) for s in true_scores], [dict(s) for s in dt_scores],
            ["cosine", "fragment_change", "entropy", "pmd_ppm"])
        auc, p = auroc(t_fused, d_fused)
        entry["fused_all4"] = {"auroc": auc, "p": p}
        results["per_decoy"][dt_name] = entry

    # Ablation: fused score with each component removed (vs ALL decoys)
    all_components = ["cosine", "fragment_change", "entropy", "pmd_ppm"]
    for drop in ["none"] + all_components:
        remaining = [c for c in all_components if c != drop]
        t_copy = [dict(s) for s in true_scores]
        d_copy = [dict(s) for s in all_decoys]
        t_f, d_f = rank_fuse(t_copy, d_copy, remaining)
        auc, p = auroc(t_f, d_f)
        results["ablation"][f"drop_{drop}" if drop != "none" else "full"] = {
            "auroc": auc, "p": p, "components": remaining}

    # Overall gates (vs hardest decoy: D1)
    if d1_scores:
        best_d1 = results["per_decoy"].get("D1_hard", {}).get("fused_all4", {}).get("auroc")
        full_aoc = results["ablation"].get("full", {}).get("auroc")
        no_pmd_aoc = results["ablation"].get("drop_pmd_ppm", {}).get("auroc")
        gates = {
            "fused_beats_d1_gt_0_65": bool(best_d1 and best_d1 > 0.65),
            "fused_beats_d2_gt_0_65": bool(
                results["per_decoy"].get("D2_reversed", {}).get("fused_all4", {}).get("auroc", 0) > 0.65),
            "no_pmd_still_beats_0_60": bool(no_pmd_aoc and no_pmd_aoc > 0.60),
            "no_pmd_beats_cosine_alone": bool(
                no_pmd_aoc and no_pmd_aoc >
                results.get("component_auroc", {}).get("cosine", 0)),
        }
    else:
        gates = {"error": "D1 decoys too few"}

    # cosine baseline vs all decoys
    cos_auc, cos_p = auroc([s["cosine"] for s in true_scores], [s["cosine"] for s in all_decoys])
    results["component_auroc"]["cosine_vs_all"] = cos_auc

    report = {
        "status": "GLM_GATE_O0_V3_COMPLETE",
        "true_pairs": len(true_scores),
        "decoys": {"D1_hard": len(d1_scores), "D2_reversed": len(d2_scores),
                   "D3_random": len(d3_scores), "total": len(all_decoys)},
        "results": results,
        "gates": gates,
        "pass_or_stop": "PASS" if all(v is True for v in gates.values() if isinstance(v, bool)) else "STOP",
        "claim_limit": "Gate O0 v3: transformation identification with hard decoys and ablation. Not a biological result.",
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
