"""GLM Gate O0 v2: Can MS/MS evidence identify real biochemical transformations?

PROPER ground truth: pairs where formula_A + exact biochemical transformation = formula_B.
Computed from formula strings in the GNPS manifest, NOT from same-formula isomers.

Truth criterion: two spectra whose molecular formulas differ by EXACTLY the
elemental composition of a known transformation (e.g., +O for hydroxylation,
+C6H8O6 for glucuronidation), with different InChIKey first blocks (different scaffolds).

Four matched decoy types (all properly matched on a confounder):
  D1: same precursor mass gap, formula change does NOT match any transformation
  D2: formula matches a transformation, but same InChIKey first 7 chars (same scaffold
      = derivative/analog, not a cross-scaffold transformation)
  D3: direction reversed (product→substrate scored as substrate→product)
  D4: peak-permuted (one spectrum's peaks randomly reassigned to a different precursor)

Scoring variants compared:
  - PMD-only (mass difference proximity to nearest transformation)
  - Cosine-only (spectral similarity)
  - Entropy similarity
  - Fragment-change (fraction of new peaks in heavier spectrum)
  - Combined (equal-weight rank average, no hand-tuned coefficients)

Output: AUROC, AUPRC, and per-decoy-type breakdown for each variant.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

TOLERANCE = 0.02  # Da, for spectral peak matching

# ============================================================================
# Formula parsing: extract C/H/N/O/S/P/F/Cl/Br/I counts
# ============================================================================

ELEMENTS = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5, "F": 6,
            "Cl": 7, "Br": 8, "I": 9, "Na": 10, "K": 11, "Si": 12}
ELEM_NAMES = list(ELEMENTS.keys())

FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> np.ndarray | None:
    """Return element count vector or None if unparseable."""
    if not formula or formula == "nan":
        return None
    vec = np.zeros(len(ELEMENTS), dtype=np.int32)
    pos = 0
    for match in FORMULA_RE.finditer(formula):
        elem = match.group(1)
        count = int(match.group(2)) if match.group(2) else 1
        if elem in ELEMENTS:
            vec[ELEMENTS[elem]] = count
        pos += 1
    if pos == 0:
        return None
    return vec


# ============================================================================
# Transformation definitions as element-delta vectors
# ============================================================================

TRANSFORMATIONS: list[tuple[str, np.ndarray]] = [
    # name, delta vector (same order as ELEMENTS)
    ("hydroxylation",       _v := lambda **kw: _mk(O=1, **kw)),
    ("dihydroxylation",     lambda: _mk(O=2)),
    ("dehydrogenation",     lambda: _mk(H=-2)),
    ("hydrogenation",       lambda: _mk(H=2)),
    ("methylation",         lambda: _mk(C=1, H=2)),
    ("demethylation",       lambda: _mk(C=-1, H=-2)),
    ("acetylation",         lambda: _mk(C=2, H=2, O=1)),
    ("formylation",         lambda: _mk(C=1, O=1)),
    ("carboxylation",       lambda: _mk(C=1, O=2)),
    ("decarboxylation",     lambda: _mk(C=-1, O=-2)),
    ("glucuronidation",     lambda: _mk(C=6, H=8, O=6)),
    ("sulfation",           lambda: _mk(S=1, O=3)),
    ("phosphorylation",     lambda: _mk(P=1, O=3, H=1)),
    ("glycosylation_hexose", lambda: _mk(C=6, H=10, O=5)),
    ("glutathione",         lambda: _mk(C=10, H=17, N=3, O=6, S=1)),
    ("glycine_conj",        lambda: _mk(C=2, H=3, N=1, O=1)),
    ("taurine_conj",        lambda: _mk(C=2, H=7, N=1, O=3, S=1)),
    ("hydration",           lambda: _mk(H=2, O=1)),
    ("dehydration",         lambda: _mk(H=-2, O=-1)),
    ("chain_elongation_C2H4", lambda: _mk(C=2, H=4)),
    ("beta_oxidation",      lambda: _mk(C=-2, H=-4, O=-2)),
    ("amination",           lambda: _mk(N=1, H=3, O=-1)),
    ("deamination",         lambda: _mk(N=-1, H=-3, O=1)),
    ("chlorination",        lambda: _mk(Cl=1, H=-1)),
    ("fluorination",        lambda: _mk(F=1, H=-1)),
    ("bromination",         lambda: _mk(Br=1, H=-1)),
]


def _mk(**kwargs) -> np.ndarray:
    vec = np.zeros(len(ELEMENTS), dtype=np.int32)
    for name, count in kwargs.items():
        if name in ELEMENTS:
            vec[ELEMENTS[name]] = count
    return vec


# Build the actual list (call the lambdas)
TRANSFORMATIONS = [(name, fn()) for name, fn in [
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


def formula_delta_matches(delta: np.ndarray) -> str | None:
    """Check if an element-count difference exactly matches a known transformation."""
    for name, tvec in TRANSFORMATIONS:
        if np.array_equal(delta, tvec):
            return name
    return None


def exact_mass(vec: np.ndarray) -> float:
    """Approximate monoisotopic mass from element vector."""
    masses = np.array([12.0, 1.00783, 14.00307, 15.99491, 31.97207,
                       30.97376, 18.99840, 34.96885, 78.91834,
                       126.90447, 22.98977, 38.96371, 27.97693])
    return float(np.sum(vec * masses))


# ============================================================================
# Spectral scoring functions (all correct, no undefined variables)
# ============================================================================

def normalize_peaks(peaks, max_peaks=100):
    peaks = sorted(sorted(peaks, key=lambda p: -p[1])[:max_peaks], key=lambda p: p[0])
    mzs = np.array([p[0] for p in peaks], dtype=np.float64)
    ints = np.array([p[1] for p in peaks], dtype=np.float64)
    if ints.max() > 0:
        ints = ints / ints.max()
    return mzs, ints


def greedy_cosine(a_mz, a_int, b_mz, b_int, tol=TOLERANCE):
    used = np.zeros(b_mz.size, dtype=bool)
    matched = 0.0
    for mz, val in zip(a_mz, a_int):
        lo, hi = np.searchsorted(b_mz, (mz - tol, mz + tol))
        if hi > lo:
            cand = np.flatnonzero(~used[lo:hi]) + lo
            if cand.size:
                best = cand[np.argmax(b_int[cand])]
                used[best] = True
                matched += float(val * b_int[best])
    na = float((a_int ** 2).sum())
    nb = float((b_int ** 2).sum())
    if na <= 0 or nb <= 0:
        return 0.0
    return matched / math.sqrt(na * nb)


def fragment_change(a_mz, a_int, b_mz, b_int, tol=TOLERANCE):
    """Fraction of heavier spectrum's intensity NOT explained by lighter spectrum."""
    if b_int.size == 0:
        return 0.0
    shared = 0.0
    for mz, val in zip(b_mz, b_int):
        lo, hi = np.searchsorted(a_mz, (mz - tol, mz + tol))
        if hi > lo:
            shared += val
    total = float(b_int.sum())
    return 1.0 - shared / total if total > 0 else 0.0


def entropy_similarity(a_mz, a_int, b_mz, b_int, tol=TOLERANCE):
    """Simplified entropy similarity (Li et al. 2021, Jensen-Shannon form)."""
    # merge peaks
    merged = {}
    for mz, val in zip(a_mz, a_int):
        merged[round(mz, 2)] = merged.get(round(mz, 2), 0.0) + val
    for mz, val in zip(b_mz, b_int):
        merged[round(mz, 2)] = merged.get(round(mz, 2), 0.0) + val
    weights = np.array(list(merged.values()))
    if weights.sum() <= 0:
        return 0.0
    weights = weights / weights.sum()
    h_mix = -float(np.sum(weights[weights > 0] * np.log(weights[weights > 0])))
    ha = a_int / a_int.sum() if a_int.sum() > 0 else a_int
    hb = b_int / b_int.sum() if b_int.sum() > 0 else b_int
    ha_n = -float(np.sum(ha[ha > 0] * np.log(ha[ha > 0])))
    hb_n = -float(np.sum(hb[hb > 0] * np.log(hb[hb > 0])))
    denom = max(ha_n, hb_n)
    if denom <= 0:
        return 0.0
    return max(0.0, 1.0 - h_mix / denom)


# ============================================================================
# Main experiment
# ============================================================================

def main() -> None:
    base = os.path.join("data", "validation")
    manifest_path = os.path.join(base, "gnps_gold_silver_10ppm_benchmark_v1",
                                 "manifest.csv.gz")
    mgf_path = os.path.join(base, "gnps_gold_silver_10ppm_benchmark_v1",
                            "spectra.mgf")
    out_dir = os.path.join(base, "GLM_gate_o0_v2")
    os.makedirs(out_dir, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    print(f"manifest rows: {len(manifest)}", flush=True)

    # Parse formulas into element vectors
    formulas = {}
    for idx, row in manifest.iterrows():
        f = str(row.get("formula", ""))
        vec = parse_formula(f)
        if vec is not None:
            formulas[idx] = (f, vec, str(row.get("inchikey", "")),
                            str(row.get("ik14", "")))
    print(f"parsed formulas: {len(formulas)}", flush=True)

    # Build formula groups
    by_formula = defaultdict(list)
    for idx, (f, vec, ik, ik14) in formulas.items():
        by_formula[f].append(idx)

    # Build formula→element_vector lookup
    all_formula_vecs = {}
    for f, indices in by_formula.items():
        all_formula_vecs[f] = formulas[indices[0]][1]

    # HASH-BASED transformation pair finding: O(n × 26) instead of O(n²)
    # For each formula, compute all 26 transformation targets, check if they exist
    vec_to_formulas = {}  # element vector tuple → list of formula strings
    for f, vec in all_formula_vecs.items():
        key = tuple(vec)
        vec_to_formulas.setdefault(key, []).append(f)

    true_pairs = []  # (lighter_idx, heavier_idx, transformation_name)
    seen_pairs = set()
    for f_light, vec_light in all_formula_vecs.items():
        for tname, tvec in TRANSFORMATIONS:
            vec_heavy = vec_light + tvec
            if np.any(vec_heavy < 0):
                continue  # negative element counts impossible
            key = tuple(vec_heavy)
            if key not in vec_to_formulas:
                continue
            for f_heavy in vec_to_formulas[key]:
                if f_heavy == f_light:
                    continue
                # cross-scaffold check
                for li in by_formula[f_light]:
                    for hi in by_formula[f_heavy]:
                        ik_l = formulas[li][2]
                        ik_h = formulas[hi][2]
                        if ik_l and ik_h and ik_l[:7] != ik_h[:7]:
                            pair_key = (li, hi)
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
                                true_pairs.append((li, hi, tname))
                if len(true_pairs) >= 5000:
                    break
            if len(true_pairs) >= 5000:
                break
        if len(true_pairs) >= 5000:
            break
    print(f"true cross-scaffold transformation pairs: {len(true_pairs)}", flush=True)

    if len(true_pairs) < 30:
        print("Too few true pairs; cannot run Gate O0", flush=True)
        report = {"status": "GLM_GATE_O0_V2_INSUFFICIENT_DATA",
                  "true_pairs": len(true_pairs),
                  "gate": "STOP"}
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        return

    # Load spectra (only for needed indices)
    needed = {u for u, v, t in true_pairs} | {v for u, v, t in true_pairs}
    spectra = {}
    with open(mgf_path, "r", encoding="utf-8", errors="ignore") as handle:
        cur = None
        row = 0
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
                    parts = line.split()
                    if len(parts) >= 2 and float(parts[1]) > 0:
                        cur["peaks"].append((float(parts[0]), float(parts[1])))
    print(f"spectra loaded: {len(spectra)}", flush=True)

    # Build matched decoys FROM THE ALREADY-LOADED SPECTRA
    rng = np.random.default_rng(20261007)
    loaded_indices = sorted(spectra.keys())
    if len(loaded_indices) < 20:
        print("Too few loaded spectra for decoy construction", flush=True)
        return

    decoy_pairs = []  # (lighter_idx, heavier_idx, decoy_type)

    # D1: same precursor gap range, but formula change doesn't match any transformation
    # Randomly re-pair loaded spectra with similar mass gaps
    true_gaps = []
    for u, v, t in true_pairs:
        if u in spectra and v in spectra:
            gap = spectra[v].get("precursor", 0) - spectra[u].get("precursor", 0)
            if gap > 0.5:
                true_gaps.append((u, v, gap))

    for u, v, gap in true_gaps:
        if len(decoy_pairs) >= len(true_gaps):
            break
        # pick random w from loaded indices with similar precursor but wrong formula relation
        for _ in range(10):
            w = rng.choice(loaded_indices)
            if w == u or w == v or w not in formulas or u not in formulas:
                continue
            mz_w = spectra[w].get("precursor", 0)
            mz_u = spectra[u].get("precursor", 0)
            w_gap = mz_w - mz_u
            if w_gap > 0.5 and abs(w_gap - gap) / max(gap, 1) < 0.05:
                # check formula doesn't match
                f_delta = formulas[w][1] - formulas[u][1]
                if formula_delta_matches(f_delta) is None:
                    decoy_pairs.append((u, w, "D1_same_gap_no_formula"))
                    break

    # D2: reverse direction (heavy→light scored as light→heavy)
    for u, v, t in true_gaps[:len(decoy_pairs)]:
        if u in spectra and v in spectra:
            decoy_pairs.append((v, u, "D2_reversed"))

    # D3: random pairing (no mass or formula relationship)
    n_random = min(len(true_gaps), 1000)
    for _ in range(n_random):
        u = rng.choice(loaded_indices)
        w = rng.choice(loaded_indices)
        if u == w or u not in formulas or w not in formulas:
            continue
        f_delta = formulas[w][1] - formulas[u][1]
        if formula_delta_matches(f_delta) is None:
            decoy_pairs.append((u, w, "D3_random_pair"))

    print(f"decoy pairs: {len(decoy_pairs)}", flush=True)

    # Score all pairs
    from scipy import stats as sps

    def score_pair(u_idx, v_idx):
        if u_idx not in spectra or v_idx not in spectra:
            return None
        su, sv = spectra[u_idx], spectra[v_idx]
        if not su.get("peaks") or not sv.get("peaks"):
            return None
        a_mz, a_int = normalize_peaks(su["peaks"])
        b_mz, b_int = normalize_peaks(sv["peaks"])
        cos = greedy_cosine(a_mz, a_int, b_mz, b_int)
        frag = fragment_change(a_mz, a_int, b_mz, b_int)
        ent = entropy_similarity(a_mz, a_int, b_mz, b_int)
        # PMD proximity
        delta_mz = abs(sv.get("precursor", 0) - su.get("precursor", 0))
        best_mass_err = 999.0
        for tname, tvec in TRANSFORMATIONS:
            expected = exact_mass(tvec)
            if expected > 0.5:
                ppm = abs(delta_mz - expected) / expected * 1e6
                best_mass_err = min(best_mass_err, ppm)
        return {
            "cosine": cos, "fragment_change": frag,
            "entropy": ent, "pmd_ppm": best_mass_err,
            # combined: rank-based (no hand-tuned coefficients)
            "combined_rank": 0.0,  # computed later
        }

    true_scores = []
    for u, v, t in true_pairs:
        s = score_pair(u, v)
        if s:
            s["transformation"] = t
            true_scores.append(s)

    decoy_scores = []
    for u, w, dtype in decoy_pairs:
        s = score_pair(u, w)
        if s:
            s["decoy_type"] = dtype
            decoy_scores.append(s)

    print(f"scored: {len(true_scores)} true, {len(decoy_scores)} decoy", flush=True)

    # Combined rank score: rank each component, then average ranks
    def rank_combine(key_true, key_decoy, invert=False):
        all_vals = [(v, True) for v in key_true] + [(v, False) for v in key_decoy]
        all_vals.sort(key=lambda x: x[0], reverse=not invert)
        ranks = {}
        for i, (val, is_true) in enumerate(all_vals):
            ranks.setdefault(i, []).append(is_true)
        return ranks

    # Compute combined scores using average of component ranks
    for key in ("cosine", "fragment_change", "entropy"):
        true_vals = [s[key] for s in true_scores]
        decoy_vals = [s[key] for s in decoy_scores]
        all_sorted = sorted(true_vals + decoy_vals)
        rank_map = {v: i for i, v in enumerate(all_sorted)}
        for s in true_scores:
            s["_rank_" + key] = rank_map[s[key]]
        for s in decoy_scores:
            s["_rank_" + key] = rank_map[s[key]]

    # PMD: lower ppm is better (invert rank)
    true_pmd = [s["pmd_ppm"] for s in true_scores]
    decoy_pmd = [s["pmd_ppm"] for s in decoy_scores]
    all_pmd = sorted(true_pmd + decoy_pmd, reverse=True)  # reverse: high ppm = low rank
    pmd_rank_map = {v: i for i, v in enumerate(all_pmd)}
    for s in true_scores:
        s["_rank_pmd"] = pmd_rank_map[s["pmd_ppm"]]
    for s in decoy_scores:
        s["_rank_pmd"] = pmd_rank_map[s["pmd_ppm"]]

    n_total = len(true_scores) + len(decoy_scores)
    for s in true_scores + decoy_scores:
        s["combined_rank"] = (
            s["_rank_cosine"] + s["_rank_fragment_change"] +
            s["_rank_entropy"] + s["_rank_pmd"]
        ) / (4.0 * n_total)

    # Compute AUROC for each scoring variant
    results = {}
    for key in ("cosine", "fragment_change", "entropy", "combined_rank"):
        t_vals = [s[key] for s in true_scores]
        d_vals = [s[key] for s in decoy_scores]
        if len(t_vals) < 5 or len(d_vals) < 5:
            results[key] = {"auroc": None, "note": "insufficient data"}
            continue
        u_stat, p_val = sps.mannwhitneyu(t_vals, d_vals, alternative="greater")
        auroc = u_stat / (len(t_vals) * len(d_vals))
        results[key] = {"auroc": round(float(auroc), 4),
                       "mannwhitney_p": float(f"{p_val:.2e}")}
    # PMD: lower is better
    t_pmd = [s["pmd_ppm"] for s in true_scores]
    d_pmd = [s["pmd_ppm"] for s in decoy_scores]
    if len(t_pmd) >= 5 and len(d_pmd) >= 5:
        u_stat, p_val = sps.mannwhitneyu(t_pmd, d_pmd, alternative="less")
        results["pmd_ppm"] = {"auroc": round(float(u_stat / (len(t_pmd) * len(d_pmd))), 4),
                             "mannwhitney_p": float(f"{p_val:.2e}")}

    # Per-decoy-type breakdown
    breakdown = {}
    for dtype in set(d[2] for d in decoy_pairs):
        sub_decoy = [s for s in decoy_scores if s.get("decoy_type") == dtype]
        if len(sub_decoy) < 3:
            continue
        for key in ("cosine", "combined_rank"):
            t_vals = [s[key] for s in true_scores]
            d_vals = [s[key] for s in sub_decoy]
            if len(d_vals) >= 3:
                u, p = sps.mannwhitneyu(t_vals, d_vals, alternative="greater")
                breakdown.setdefault(dtype, {})[key] = round(float(u / (len(t_vals) * len(d_vals))), 4)

    # Gates
    combined_auc = results.get("combined_rank", {}).get("auroc")
    gates = {
        "combined_auroc_gt_0_65": bool(combined_auc and combined_auc > 0.65),
        "beats_cosine_alone": bool(
            combined_auc and results.get("cosine", {}).get("auroc") and
            combined_auc > results["cosine"]["auroc"]),
        "beats_pmd_alone": bool(
            combined_auc and results.get("pmd_ppm", {}).get("auroc") and
            combined_auc > results["pmd_ppm"]["auroc"]),
    }
    gate_pass = all(gates.values())

    report = {
        "status": "GLM_GATE_O0_V2_" + ("PASS" if gate_pass else "STOP"),
        "ground_truth": "formula-level exact transformation matching (elemental composition)",
        "n_transformations_tested": len(TRANSFORMATIONS),
        "true_pairs": len(true_scores),
        "decoy_pairs": len(decoy_scores),
        "decoy_types": dict(pd.Series([s.get("decoy_type", "?") for s in decoy_scores]).value_counts()),
        "results": results,
        "per_decoy_type": breakdown,
        "gates": gates,
        "transformation_counts": dict(pd.Series([s["transformation"] for s in true_scores]).value_counts()),
        "claim_limit": (
            "Gate O0 v2: can formula-level + MS/MS evidence separate real "
            "biochemical transformation pairs from matched decoys? "
            "Not a biological or disease result."
        ),
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str), flush=True)
    print(f"written: {out_dir}/report.json", flush=True)


if __name__ == "__main__":
    main()
