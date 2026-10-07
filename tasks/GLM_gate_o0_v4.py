"""GLM Gate O0 v4: Transformation-SPECIFIC fragment evidence.

Key insight from v3: generic fragment_change (fraction of new peaks) achieves
AUROC 0.82 against hard decoys, while cosine is anti-predictive.

v4 refines this: instead of asking "are there new peaks?", ask
"are the new peaks chemically consistent with THIS specific transformation?"

Three evidence types:
  E1  Fragment-shift: peaks in u shifted by exactly the transformation mass
      appear in v (e.g., u_fragment + 15.9949 for hydroxylation).
  E2  Diagnostic fragments: known modification-specific product ions
      (e.g., m/z 175.0248 for glucuronide, m/z 79.9568 for sulfate loss).
  E3  New neutral losses: losses present in v but not u, matching the
      transformation mass (e.g., product loses SO3 but substrate doesn't).

Score = weighted fraction of new-peak intensity explained by E1+E2+E3.
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

# ============================================================================
# Transformation definitions
# ============================================================================

ELEMENTS = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5, "F": 6,
            "Cl": 7, "Br": 8, "I": 9}
ELEM_MASSES = np.array([12.0, 1.00783, 14.00307, 15.99491, 31.97207,
                        30.97376, 18.99840, 34.96885, 78.91834, 126.90447])
FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _mk(**kw):
    v = np.zeros(len(ELEMENTS), dtype=np.int32)
    for n, c in kw.items():
        if n in ELEMENTS:
            v[ELEMENTS[n]] = c
    return v


# Each transformation: (name, element_delta, exact_mass, diagnostic_fragments, diagnostic_neutral_losses)
TRANSFORMS = [
    ("hydroxylation", _mk(O=1), 15.99491,
     [], [15.99491]),
    ("dihydroxylation", _mk(O=2), 31.98983,
     [], [31.98983]),
    ("dehydrogenation", _mk(H=-2), -2.01565,
     [], []),
    ("hydrogenation", _mk(H=2), 2.01565,
     [], []),
    ("methylation", _mk(C=1, H=2), 14.01565,
     [], []),
    ("demethylation", _mk(C=-1, H=-2), -14.01565,
     [], [14.01565]),
    ("acetylation", _mk(C=2, H=2, O=1), 42.01057,
     [43.01839], [42.01057]),  # CH3CO+
    ("formylation", _mk(C=1, O=1), 27.99491,
     [29.00274], []),
    ("carboxylation", _mk(C=1, O=2), 43.98983,
     [44.99765], []),
    ("decarboxylation", _mk(C=-1, O=-2), -43.98983,
     [], [43.98983]),
    ("glucuronidation", _mk(C=6, H=8, O=6), 176.03209,
     [175.02484, 113.02444, 85.02951], [176.03209, 79.95682]),
    ("sulfation", _mk(S=1, O=3), 79.95682,
     [79.95682, 96.96010], [79.95682]),
    ("phosphorylation", _mk(P=1, O=3, H=1), 79.96633,
     [79.96633, 96.96962], [79.96633]),
    ("glycosylation_hexose", _mk(C=6, H=10, O=5), 162.05282,
     [145.05021, 127.03949, 85.02951], [162.05282]),
    ("glycine_conjugation", _mk(C=2, H=3, N=1, O=1), 57.02146,
     [58.02927], []),
    ("taurine_conjugation", _mk(C=2, H=7, N=1, O=3, S=1), 125.01966,
     [126.02749, 80.96464], []),
    ("hydration", _mk(H=2, O=1), 18.01056,
     [], [18.01056]),
    ("dehydration", _mk(H=-2, O=-1), -18.01056,
     [], [18.01056]),
    ("chain_elongation_C2H4", _mk(C=2, H=4), 28.03130,
     [], []),
    ("beta_oxidation", _mk(C=-2, H=-4, O=-2), -60.02113,
     [], []),
    ("amination", _mk(N=1, H=3, O=-1), 0.98402,
     [], []),
    ("deamination", _mk(N=-1, H=-3, O=1), -0.98402,
     [], []),
    ("chlorination", _mk(Cl=1, H=-1), 33.96103,
     [], []),
    ("fluorination", _mk(F=1, H=-1), 17.99054,
     [], []),
    ("bromination", _mk(Br=1, H=-1), 77.91051,
     [], []),
    ("glutathione_conjugation", _mk(C=10, H=17, N=3, O=6, S=1), 307.08380,
     [308.09108, 179.06709, 162.04591], []),
]

# Build lookup
TRANSFORM_LOOKUP = {name: (vec, mass, diags, nls) for name, vec, mass, diags, nls in TRANSFORMS}


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
    for name, vec, _, _, _ in TRANSFORMS:
        if np.array_equal(delta, vec):
            return name
    return None


# ============================================================================
# Spectral functions
# ============================================================================

def norm_peaks(peaks, max_peaks=100):
    p = sorted(sorted(peaks, key=lambda x: -x[1])[:max_peaks], key=lambda x: x[0])
    mzs = np.array([x[0] for x in p], dtype=np.float64)
    ints = np.array([x[1] for x in p], dtype=np.float64)
    if ints.max() > 0:
        ints /= ints.max()
    return mzs, ints


def get_neutral_losses(mz: np.ndarray, precursor: float, tol=TOLERANCE):
    """Neutral losses = precursor - fragment m/z."""
    return precursor - mz[mz < precursor - 10]


def transformation_specific_score(
    spec_u, spec_v, transform_name: str
) -> dict[str, float]:
    """Score whether v's new peaks are chemically consistent with the transformation."""
    _, mass, diag_frags, diag_losses = TRANSFORM_LOOKUP[transform_name]
    a_mz, a_int = norm_peaks(spec_u["peaks"])
    b_mz, b_int = norm_peaks(spec_v["peaks"])
    prec_u = spec_u.get("precursor", 0)
    prec_v = spec_v.get("precursor", 0)
    t_mass = abs(mass)

    # --- Identify "new" peaks in v (not present in u) ---
    new_ints = []
    new_mzs = []
    for mz, val in zip(b_mz, b_int):
        lo, hi = np.searchsorted(a_mz, (mz - TOLERANCE, mz + TOLERANCE))
        if hi == lo:  # not found in u
            new_mzs.append(mz)
            new_ints.append(val)
    new_mzs = np.array(new_mzs) if new_mzs else np.array([])
    new_ints = np.array(new_ints) if new_ints else np.array([])
    total_new_intensity = float(new_ints.sum()) if new_ints.size else 0.0
    total_v_intensity = float(b_int.sum())
    if total_new_intensity <= 0 or total_v_intensity <= 0:
        return {"specificity": 0.0, "e1_shift": 0.0, "e2_diagnostic": 0.0,
                "e3_new_loss": 0.0, "fraction_new": 0.0}

    # --- E1: Fragment-shift evidence ---
    # Peaks in v that match (peak in u + transformation mass)
    e1_intensity = 0.0
    if t_mass > 0.5:
        for nm, ni in zip(new_mzs, new_ints):
            expected = nm - t_mass  # where this peak would be in u
            lo, hi = np.searchsorted(a_mz, (expected - TOLERANCE, expected + TOLERANCE))
            if hi > lo:
                e1_intensity += ni

    # --- E2: Diagnostic fragment evidence ---
    # New peaks in v matching known diagnostic fragments of the transformation
    e2_intensity = 0.0
    for diag_mz in diag_frags:
        for nm, ni in zip(new_mzs, new_ints):
            if abs(nm - diag_mz) <= TOLERANCE * 2:
                e2_intensity += ni
                break

    # --- E3: New neutral loss evidence ---
    # Neutral losses in v but not in u, matching transformation mass
    e3_intensity = 0.0
    if t_mass > 0.5 and prec_v > 0 and prec_u > 0:
        losses_v = get_neutral_losses(b_mz, prec_v)
        losses_u = get_neutral_losses(a_mz, prec_u)
        for lv in losses_v:
            if not any(abs(lv - lu) <= TOLERANCE for lu in losses_u):
                # this loss is new in v; does it match the transformation?
                if abs(lv - t_mass) <= TOLERANCE * 3:
                    # find the corresponding peak in v
                    peak_mz = prec_v - lv
                    lo, hi = np.searchsorted(b_mz, (peak_mz - TOLERANCE, peak_mz + TOLERANCE))
                    if hi > lo:
                        e3_intensity += float(b_int[lo:hi].max())
        # also check diagnostic neutral losses
        for diag_nl in diag_losses:
            for lv in losses_v:
                if abs(lv - diag_nl) <= TOLERANCE * 2:
                    peak_mz = prec_v - lv
                    lo, hi = np.searchsorted(b_mz, (peak_mz - TOLERANCE, peak_mz + TOLERANCE))
                    if hi > lo:
                        e3_intensity += float(b_int[lo:hi].max())

    explained = e1_intensity + e2_intensity + e3_intensity
    specificity = min(1.0, explained / max(total_new_intensity, 1e-12))
    fraction_new = total_new_intensity / total_v_intensity

    return {
        "specificity": round(specificity, 4),
        "e1_shift": round(e1_intensity / max(total_new_intensity, 1e-12), 4),
        "e2_diagnostic": round(e2_intensity / max(total_new_intensity, 1e-12), 4),
        "e3_new_loss": round(e3_intensity / max(total_new_intensity, 1e-12), 4),
        "fraction_new": round(fraction_new, 4),
    }


# ============================================================================
# Main experiment
# ============================================================================

def main():
    base = os.path.join("data", "validation")
    manifest_path = os.path.join(base, "gnps_gold_silver_10ppm_benchmark_v1",
                                 "manifest.csv.gz")
    mgf_path = os.path.join(base, "gnps_gold_silver_10ppm_benchmark_v1", "spectra.mgf")
    out_dir = os.path.join(base, "GLM_gate_o0_v4")
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
        for tname, tvec, _, _, _ in TRANSFORMS:
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

    # Load spectra
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

    # Build D1 hard decoys (same mass gap, wrong formula)
    rng = np.random.default_rng(20261007)
    loaded = sorted(spectra.keys())
    loaded_with_f = [i for i in loaded if i in formulas]
    prec_map = {i: spectra[i].get("precursor", 0) for i in loaded_with_f}
    sorted_lf = sorted(loaded_with_f, key=lambda i: prec_map[i])
    sorted_pr = [prec_map[i] for i in sorted_lf]

    d1_pairs = []
    for u, v, tname in true_pairs:
        if u not in spectra or v not in spectra:
            continue
        gap = prec_map.get(v, 0) - prec_map.get(u, 0)
        if gap < 0.5:
            continue
        target = prec_map.get(u, 0) + gap
        lo_b = np.searchsorted(sorted_pr, target * 0.99)
        hi_b = np.searchsorted(sorted_pr, target * 1.01)
        for j in range(lo_b, min(hi_b, len(sorted_lf))):
            w = sorted_lf[j]
            if w != u and w != v and w in formulas and w in spectra:
                fd = formulas[w][1] - formulas[u][1]
                if delta_matches(fd) is None:
                    # assign the closest transformation for scoring
                    d1_pairs.append((u, w, tname))  # pretend it's the same transformation
                    break
        if len(d1_pairs) >= 2000:
            break
    print(f"D1 hard decoys: {len(d1_pairs)}", flush=True)

    # Score all pairs
    true_scores = []
    for u, v, tname in true_pairs:
        if u in spectra and v in spectra:
            s = transformation_specific_score(spectra[u], spectra[v], tname)
            if s:
                true_scores.append(s)

    decoy_scores = []
    for u, w, tname in d1_pairs:
        if u in spectra and w in spectra:
            s = transformation_specific_score(spectra[u], spectra[w], tname)
            if s:
                decoy_scores.append(s)

    print(f"scored: {len(true_scores)} true, {len(decoy_scores)} D1 decoys", flush=True)

    # AUROC for each sub-score
    results = {}
    for key in ("specificity", "e1_shift", "e2_diagnostic", "e3_new_loss", "fraction_new"):
        t_vals = [s[key] for s in true_scores]
        d_vals = [s[key] for s in decoy_scores]
        if len(t_vals) >= 10 and len(d_vals) >= 10:
            u_stat, p_val = sps.mannwhitneyu(t_vals, d_vals, alternative="greater")
            auc = float(u_stat) / (len(t_vals) * len(d_vals))
            results[key] = {"auroc": round(auc, 4), "p": float(f"{p_val:.2e}")}
        else:
            results[key] = {"auroc": None}

    # Combined: specificity × fraction_new (high specificity on many new peaks)
    t_combined = [s["specificity"] * s["fraction_new"] for s in true_scores]
    d_combined = [s["specificity"] * s["fraction_new"] for s in decoy_scores]
    if len(t_combined) >= 10 and len(d_combined) >= 10:
        u_stat, p_val = sps.mannwhitneyu(t_combined, d_combined, alternative="greater")
        results["specificity_x_fraction"] = {
            "auroc": round(float(u_stat) / (len(t_combined) * len(d_combined)), 4),
            "p": float(f"{p_val:.2e}")}

    best_auc = max((r.get("auroc") or 0) for r in results.values())
    gates = {
        "any_component_beats_0_70": best_auc > 0.70,
        "specificity_beats_0_60": (results.get("specificity", {}).get("auroc") or 0) > 0.60,
        "e1_shift_beats_0_60": (results.get("e1_shift", {}).get("auroc") or 0) > 0.60,
    }

    report = {
        "status": "GLM_GATE_O0_V4_" + ("PASS" if all(gates.values()) else "STOP"),
        "true_pairs": len(true_scores),
        "d1_hard_decoys": len(decoy_scores),
        "results": results,
        "gates": gates,
        "vs_v3": {
            "v3_fragment_change_auroc": 0.8175,
            "v4_best": best_auc,
            "improvement": round(best_auc - 0.8175, 4),
        },
        "claim_limit": "Gate O0 v4: transformation-SPECIFIC fragment evidence vs hard decoys.",
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
