"""Phenotype-blind global MS1 peak-network primitives for BioAware.

The graph encodes observable ion relationships.  It is deliberately separate
from molecular identity ranking: an isotope/adduct/neutral-loss edge is not a
metabolite label.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


EPS = 1e-12


@dataclass(frozen=True)
class MassRelation:
    name: str
    delta_mz: float
    family: str
    merge_ion_family: bool


COMMON_LOSSES = (
    ("loss_H2O", 18.010565),
    ("loss_NH3", 17.026549),
    ("loss_CO", 27.994915),
    ("loss_CO2", 43.989829),
    ("loss_HCOOH", 46.005480),
    ("loss_H3PO4", 97.976896),
    ("loss_hexose", 162.052824),
)


def panel_relations(panel: str) -> tuple[MassRelation, ...]:
    """Return a fixed, outcome-independent relation catalogue."""
    shared = [
        MassRelation("isotope_z1", 1.003355, "isotope", True),
        MassRelation("isotope_z2", 0.501677, "isotope", True),
        *(MassRelation(name, mass, "in_source_loss", False) for name, mass in COMMON_LOSSES),
    ]
    if panel.startswith("pos"):
        adducts = (
            MassRelation("adduct_NH4_vs_H", 17.026549, "adduct", True),
            MassRelation("adduct_Na_vs_H", 21.981942, "adduct", True),
            MassRelation("adduct_K_vs_H", 37.955882, "adduct", True),
        )
    elif panel.startswith("neg"):
        adducts = (
            MassRelation("adduct_Na2H_vs_H", 21.981942, "adduct", True),
            MassRelation("adduct_Cl_vs_H", 35.976678, "adduct", True),
            MassRelation("adduct_formate_vs_H", 46.005477, "adduct", True),
            MassRelation("adduct_acetate_vs_H", 60.021127, "adduct", True),
        )
    else:
        raise ValueError(f"panel must begin with pos or neg, observed {panel!r}")
    return tuple(shared) + adducts


def normalize_intensity_matrix(values: np.ndarray) -> np.ndarray:
    """Sample-median normalize and log transform without phenotype labels."""
    matrix = np.asarray(values, dtype=float).copy()
    matrix[~np.isfinite(matrix) | (matrix < 0)] = 0.0
    positive = np.where(matrix > 0, matrix, np.nan)
    medians = np.nanmedian(positive, axis=0)
    if np.any(~np.isfinite(medians) | (medians <= 0)):
        bad = np.flatnonzero(~np.isfinite(medians) | (medians <= 0)).tolist()
        raise ValueError(f"samples without positive intensity: {bad}")
    return np.log1p(matrix / medians[None, :])


def pair_evidence(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    """Compute all-sample and deterministic-half abundance diagnostics."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("pair evidence expects equal one-dimensional vectors")
    present_x = x > 0
    present_y = y > 0
    joint = present_x & present_y
    union = present_x | present_y

    def pearson(left: np.ndarray, right: np.ndarray) -> float:
        if left.size < 3 or np.std(left) <= EPS or np.std(right) <= EPS:
            return np.nan
        return float(np.corrcoef(left, right)[0, 1])

    def spearman(left: np.ndarray, right: np.ndarray) -> float:
        if left.size < 3:
            return np.nan
        return pearson(rankdata(left, method="average"), rankdata(right, method="average"))

    even = np.arange(x.size) % 2 == 0
    odd = ~even
    return {
        "n_samples": int(x.size),
        "n_joint_detected": int(joint.sum()),
        "co_detection_jaccard": float(joint.sum() / union.sum()) if union.any() else np.nan,
        "abundance_pearson": pearson(x, y),
        "abundance_spearman": spearman(x, y),
        "pearson_half_even": pearson(x[even], y[even]),
        "pearson_half_odd": pearson(x[odd], y[odd]),
    }


def mass_relation_pairs(
    mz: np.ndarray,
    rt_sec: np.ndarray,
    relations: tuple[MassRelation, ...],
    ppm: float,
    absolute_floor_da: float,
    rt_tolerance_sec: float,
) -> list[tuple[int, int, MassRelation, float, float]]:
    """Enumerate mass/RT-compatible pairs without quadratic all-pairs search."""
    masses = np.asarray(mz, dtype=float)
    retention = np.asarray(rt_sec, dtype=float)
    if masses.shape != retention.shape or masses.ndim != 1:
        raise ValueError("m/z and RT must be equal one-dimensional arrays")
    if np.any(~np.isfinite(masses) | (masses <= 0) | ~np.isfinite(retention)):
        raise ValueError("m/z and RT must be finite and m/z positive")
    order = np.argsort(masses, kind="mergesort")
    sorted_mz = masses[order]
    rows: list[tuple[int, int, MassRelation, float, float]] = []
    for left in range(len(masses)):
        for relation in relations:
            target = masses[left] + relation.delta_mz
            tolerance = max(absolute_floor_da, target * ppm * 1e-6)
            lo = int(np.searchsorted(sorted_mz, target - tolerance, side="left"))
            hi = int(np.searchsorted(sorted_mz, target + tolerance, side="right"))
            for position in range(lo, hi):
                right = int(order[position])
                if right == left:
                    continue
                drt = abs(float(retention[right] - retention[left]))
                if drt > rt_tolerance_sec:
                    continue
                error = float((masses[right] - masses[left]) - relation.delta_mz)
                rows.append((left, right, relation, error, drt))
    return rows


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def labels(self) -> np.ndarray:
        roots = [self.find(index) for index in range(len(self.parent))]
        mapping = {root: label for label, root in enumerate(sorted(set(roots)))}
        return np.asarray([mapping[root] for root in roots], dtype=np.int64)
