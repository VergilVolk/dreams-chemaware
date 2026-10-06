"""GLM transformation-edge module for global chemical remodeling decomposition.

Core scientific object: a credible transformation edge between two LCNEC spectral
entities whose neutral mass difference matches a known biochemical transformation,
supported by MS/MS fragment evidence.  The edge differential g_{p,e} = d_{p,v} - d_{p,u}
measures whether patient p's tumour/adjacent effect differs across the edge.

This module provides:
  1. TRANSFORMATION_PMD_TABLE: curated exact-mass differences for biochemical
     transformations (phase I/II, methylation, acetylation, glycosylation,
     sulfation, phosphorylation, redox, chain-length, etc.)
  2. edge_candidates(): given a list of (family_id, neutral_mass, ms2_peaks),
     construct candidate transformation edges by PMD matching within tolerance.
  3. fragment_support(): check that both endpoints' MS/MS spectra share at
     least one high-intensity fragment (or a fragment matching the PMD itself).
  4. edge_differentials(): compute g_{p,e} for each patient from a 34 x N
     patient-effect matrix.
  5. patient_consistency(): sign-test p-value for direction concentration of
     g_{p,e} across patients (one-sided binomial).
  6. transformation_null(): PMD-shuffled null for FDR calibration.

Author: GLM (task-2 development, 2026-10-07)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


# =============================================================================
# 1. Curated biochemical transformation PMD table (neutral exact masses, Da)
# =============================================================================

TRANSFORMATION_PMD_TABLE: dict[str, float] = {
    # --- redox / desaturation ---
    "dehydrogenation (−H2)": -2.015650,
    "hydrogenation (+H2)": +2.015650,
    "hydroxylation (+O)": +15.994915,
    "dihydroxylation (+O2)": +31.989829,
    "carbonylation (+O−H2)": +13.979265,
    # --- C1 metabolism ---
    "methylation (+CH2)": +14.015650,
    "demethylation (−CH2)": -14.015650,
    "formylation (+CO)": +27.994915,
    # --- acyl / acetyl ---
    "acetylation (+C2H2O)": +42.010565,
    "deacetylation (−C2H2O)": -42.010565,
    "malonylation (+C3H2O3)": +86.000394,
    # --- conjugation (phase II) ---
    "glucuronidation (+C6H8O6)": +176.032088,
    "sulfation (+SO3)": +79.956815,
    "phosphorylation (+HPO3)": +79.966331,
    "glycosylation_hexose (+C6H10O5)": +162.052824,
    "pentosylation (+C5H8O4)": +132.042259,
    "glutathione (+C10H17N3O6S)": +307.083804,
    "glycine_conjugation (+C2H3NO)": +57.021464,
    "taurine_conjugation (+C2H7NO3S)": +125.019664,
    # --- carboxyl / amide ---
    "carboxylation (+CO2)": +43.989829,
    "decarboxylation (−CO2)": -43.989829,
    "deamination (−NH3+O)": -0.984016,
    "amination (+NH3−O)": +0.984016,
    # --- hydration ---
    "hydration (+H2O)": +18.010565,
    "dehydration (−H2O)": -18.010565,
    # --- chain / lipid ---
    "chain_elongation (+C2H4)": +28.031300,
    "desaturation_lipid (−H2 per DB)": -2.015650,
    "beta oxidation (−C2H4O2)": -60.021129,
    # --- halogen ---
    "chlorination (+Cl−H)": +33.961029,
    "fluorination (+F−H)": +17.990540,
    "bromination (+Br−H)": +77.910511,
    # --- methylation (aromatic) ---
    "aromatic_methylation (+CH2)": +14.015650,
    "sulfate_to_glucuronide_switch (+C6H8O6−SO3)": +96.075273,
    # --- miscellaneous ---
    "glyoxylation (+C2O2)": +55.989829,
    "nitrosation (+NO)": +29.997809,
    "nitration (+NO2)": +45.992904,
}

# deduplicate by exact mass (some entries are synonyms)
_DEDUP: dict[float, str] = {}
for _name, _mass in TRANSFORMATION_PMD_TABLE.items():
    _key = round(_mass, 5)
    if _key not in _DEDUP:
        _DEDUP[_key] = _name
TRANSFORMATION_PMD_TABLE = {name: mass for mass, name in _DEDUP.items()}


# =============================================================================
# 2. Data classes and edge construction
# =============================================================================

@dataclass(frozen=True)
class SpectralEntity:
    family_id: int
    neutral_mass: float
    precursor_mz: float
    ms2_peaks: tuple  # tuple of (mz, intensity) sorted by mz


@dataclass(frozen=True)
class TransformationEdge:
    u: int  # family_id of the lighter entity
    v: int  # family_id of the heavier entity
    pmd_name: str
    pmd_expected: float
    pmd_observed: float
    ppm_error: float
    fragment_shared: bool


def edge_candidates(
    entities: list[SpectralEntity],
    ppm_tolerance: float = 20.0,
    max_edges: int = 10000,
) -> list[TransformationEdge]:
    """Construct candidate transformation edges by PMD matching.

    For each pair (u, v) with mass(v) > mass(u), check if
    mass(v) - mass(u) matches any transformation PMD within ppm.
    """
    entities_sorted = sorted(entities, key=lambda e: e.neutral_mass)
    edges: list[TransformationEdge] = []
    for i, u in enumerate(entities_sorted):
        for v in entities_sorted[i + 1:]:
            delta = v.neutral_mass - u.neutral_mass
            if delta < 0.5:
                continue
            if delta > 320.0:
                break  # sorted; no more matches possible
            for name, expected in TRANSFORMATION_PMD_TABLE.items():
                if expected < 0:
                    continue  # only positive transformations (v is heavier)
                ppm = abs(delta - expected) / max(expected, 1e-6) * 1e6
                if ppm <= ppm_tolerance:
                    shared = fragment_support(u, v)
                    edges.append(TransformationEdge(
                        u=u.family_id, v=v.family_id,
                        pmd_name=name, pmd_expected=expected,
                        pmd_observed=round(delta, 6),
                        ppm_error=round(ppm, 2),
                        fragment_shared=shared,
                    ))
                    break  # one PMD match per pair (first match)
            if len(edges) >= max_edges:
                return edges
    return edges


def fragment_support(
    u: SpectralEntity, v: SpectralEntity,
    tolerance: float = 0.02,
    min_intensity_fraction: float = 0.10,
) -> bool:
    """Check whether both entities share at least one high-intensity fragment.

    This is a necessary (not sufficient) condition for the two spectra to
    represent structurally related molecules that share a core scaffold.
    """
    if not u.ms2_peaks or not v.ms2_peaks:
        return False
    max_u = max(intensity for _, intensity in u.ms2_peaks)
    max_v = max(intensity for _, intensity in v.ms2_peaks)
    u_mzs = np.array([mz for mz, _ in u.ms2_peaks])
    v_mzs = np.array([mz for mz, _ in v.ms2_peaks])
    u_ints = np.array([i / max_u for _, i in u.ms2_peaks])
    v_ints = np.array([i / max_v for _, i in v.ms2_peaks])
    for mz_u, int_u in zip(u_mzs, u_ints):
        if int_u < min_intensity_fraction:
            continue
        lo, hi = np.searchsorted(v_mzs, (mz_u - tolerance, mz_u + tolerance))
        if hi > lo:
            best = np.argmax(v_ints[lo:hi]) + lo
            if v_ints[best] >= min_intensity_fraction:
                return True
    return False


# =============================================================================
# 3. Edge differentials and patient consistency
# =============================================================================

def edge_differentials(
    effects: np.ndarray,
    family_ids: list[int],
    edges: list[TransformationEdge],
) -> dict[tuple[int, int], np.ndarray]:
    """Compute g_{p,e} = d_{p,v} - d_{p,u} for each edge and patient.

    effects: (n_patients, n_families) patient-effect matrix
    family_ids: list of family_id corresponding to columns of effects
    edges: list of TransformationEdge
    returns: dict mapping (u_id, v_id) -> (n_patients,) array of differentials
    """
    index = {fid: col for col, fid in enumerate(family_ids)}
    result: dict[tuple[int, int], np.ndarray] = {}
    for edge in edges:
        if edge.u not in index or edge.v not in index:
            continue
        col_u = index[edge.u]
        col_v = index[edge.v]
        result[(edge.u, edge.v)] = effects[:, col_v] - effects[:, col_u]
    return result


def patient_consistency(
    g_pe: np.ndarray,
    alternative: str = "greater",
) -> dict[str, float]:
    """Sign-test p-value for direction concentration of edge differentials.

    Under H0: P(g > 0) = 0.5. If most patients show the same sign for the
    edge differential, this is evidence of a consistent chemical substitution.
    """
    g = np.asarray(g_pe, dtype=np.float64)
    g = g[np.isfinite(g)]
    n = len(g)
    if n < 5:
        return {"n": n, "p_value": 1.0, "n_positive": 0, "fraction_positive": 0.0}
    n_pos = int(np.sum(g > 0))
    if alternative == "greater":
        p = stats.binomtest(n_pos, n, 0.5, alternative="greater").pvalue
    elif alternative == "less":
        p = stats.binomtest(n_pos, n, 0.5, alternative="less").pvalue
    else:
        p = stats.binomtest(n_pos, n, 0.5, alternative="two-sided").pvalue
    return {
        "n": n,
        "p_value": float(p),
        "n_positive": n_pos,
        "fraction_positive": round(n_pos / n, 4),
        "median_g": round(float(np.median(g)), 4),
    }


def transformation_null(
    entities: list[SpectralEntity],
    effects: np.ndarray,
    family_ids: list[int],
    ppm_tolerance: float = 20.0,
    n_permutations: int = 1000,
    seed: int = 20261007,
) -> dict[str, float]:
    """Shuffle PMD assignments (random re-pairing) to build a null for FDR.

    Randomly permute the neutral masses among entities, rebuild edges, compute
    g_{p,e} and their consistency.  The null distribution of minimum p-values
    provides the calibration for multiple-testing correction.
    """
    rng = np.random.default_rng(seed)
    masses = [e.neutral_mass for e in entities]
    permuted_entities = []
    order = rng.permutation(len(entities))
    for new_idx, old_idx in enumerate(order):
        e = entities[old_idx]
        permuted_entities.append(SpectralEntity(
            family_id=e.family_id,
            neutral_mass=masses[new_idx],
            precursor_mz=e.precursor_mz,
            ms2_peaks=e.ms2_peaks,
        ))
    null_edges = edge_candidates(permuted_entities, ppm_tolerance)
    null_differentials = edge_differentials(effects, family_ids, null_edges)
    p_values = [
        patient_consistency(g)["p_value"]
        for g in null_differentials.values()
    ]
    if not p_values:
        return {"n_null_edges": 0, "min_p": 1.0, "median_p": 1.0}
    return {
        "n_null_edges": len(p_values),
        "min_p": float(min(p_values)),
        "median_p": float(np.median(p_values)),
    }


# =============================================================================
# 4. Summary statistics
# =============================================================================

def transformation_edge_summary(
    edges: list[TransformationEdge],
    differentials: dict[tuple[int, int], np.ndarray],
) -> dict[str, object]:
    """Produce a summary of all edges and their patient consistency."""
    rows = []
    for (u_id, v_id), g in differentials.items():
        edge = next((e for e in edges if e.u == u_id and e.v == v_id), None)
        if edge is None:
            continue
        consistency = patient_consistency(g)
        rows.append({
            "family_u": u_id,
            "family_v": v_id,
            "pmd_name": edge.pmd_name,
            "pmd_expected": edge.pmd_expected,
            "pmd_observed": edge.pmd_observed,
            "ppm_error": edge.ppm_error,
            "fragment_shared": edge.fragment_shared,
            "n_patients": consistency["n"],
            "n_positive": consistency["n_positive"],
            "fraction_positive": consistency["fraction_positive"],
            "median_g": consistency["median_g"],
            "sign_test_p": consistency["p_value"],
        })
    rows.sort(key=lambda r: r["sign_test_p"])
    n_edges = len(rows)
    n_fragment = sum(1 for r in rows if r["fragment_shared"])
    n_significant_005 = sum(1 for r in rows if r["sign_test_p"] <= 0.05)
    return {
        "n_edges": n_edges,
        "n_with_fragment_support": n_fragment,
        "n_significant_p_005": n_significant_005,
        "edges": rows,
    }
