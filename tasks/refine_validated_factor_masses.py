"""Refine coarse spectral-factor masses and enumerate auditable formulas.

The validated factor catalog was discovered with 0.02 Da windows.  This audit
uses the original retained peak m/z values, separates reproducible exact-mass
clusters, and enumerates CHNOPS elemental compositions.  Formula candidates
are annotations, not structural or mechanistic assignments.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ATOMIC_MASS = {
    "C": 12.0,
    "H": 1.00782503223,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "P": 30.97376199842,
    "S": 31.9720711744,
}
ELECTRON_MASS = 0.000548579909065
FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(text: str | None) -> dict[str, int]:
    if not text or not isinstance(text, str):
        return {}
    parsed: dict[str, int] = {}
    for element, count in FORMULA_RE.findall(text):
        parsed[element] = parsed.get(element, 0) + int(count or 1)
    return parsed


def format_formula(comp: dict[str, int]) -> str:
    return "".join(f"{element}{comp[element] if comp[element] != 1 else ''}"
                   for element in ("C", "H", "N", "O", "P", "S")
                   if comp.get(element, 0))


def composition_is_subset(candidate: dict[str, int], precursor: dict[str, int]) -> bool:
    return all(candidate.get(element, 0) <= precursor.get(element, 0)
               for element in candidate)


def dbe(comp: dict[str, int]) -> float:
    return (2 * comp.get("C", 0) + 2 + comp.get("N", 0) - comp.get("H", 0)) / 2


def enumerate_chnops(target: float, ppm: float, positive_ion: bool) -> list[dict]:
    tolerance = target * ppm * 1e-6
    neutral_target = target + ELECTRON_MASS if positive_ion else target
    results: list[dict] = []
    max_c = int((neutral_target + tolerance) // ATOMIC_MASS["C"])
    max_n = int((neutral_target + tolerance) // ATOMIC_MASS["N"])
    max_o = int((neutral_target + tolerance) // ATOMIC_MASS["O"])
    max_p = int((neutral_target + tolerance) // ATOMIC_MASS["P"])
    max_s = int((neutral_target + tolerance) // ATOMIC_MASS["S"])
    for c in range(max_c + 1):
        for n in range(max_n + 1):
            for o in range(max_o + 1):
                for p in range(max_p + 1):
                    for s in range(max_s + 1):
                        base = (c * ATOMIC_MASS["C"] + n * ATOMIC_MASS["N"]
                                + o * ATOMIC_MASS["O"] + p * ATOMIC_MASS["P"]
                                + s * ATOMIC_MASS["S"])
                        remaining = neutral_target - base
                        h = int(round(remaining / ATOMIC_MASS["H"]))
                        if h < 0:
                            continue
                        comp = {"C": c, "H": h, "N": n, "O": o, "P": p, "S": s}
                        if sum(comp.values()) == 0 or dbe(comp) < -1e-9:
                            continue
                        exact = base + h * ATOMIC_MASS["H"]
                        observed_mass = exact - ELECTRON_MASS if positive_ion else exact
                        error_da = observed_mass - target
                        if abs(error_da) <= tolerance:
                            results.append({
                                "formula": format_formula(comp),
                                "composition": comp,
                                "theoretical_mass": observed_mass,
                                "mass_error_da": error_da,
                                "mass_error_ppm": error_da / target * 1e6,
                                "dbe": dbe(comp),
                            })
    return sorted(results, key=lambda item: abs(item["mass_error_ppm"]))


def exact_mass_clusters(values: np.ndarray, target: float, ppm: float) -> list[np.ndarray]:
    """Greedy compact 1-D clusters; instrument-shifted clusters remain separate."""
    if len(values) == 0:
        return []
    values = np.sort(values.astype(float))
    tolerance = max(target * ppm * 1e-6, 5e-5)
    clusters: list[list[float]] = [[float(values[0])]]
    for value in values[1:]:
        candidate = clusters[-1] + [float(value)]
        center = float(np.median(candidate))
        if max(abs(np.asarray(candidate) - center)) <= tolerance:
            clusters[-1].append(float(value))
        else:
            clusters.append([float(value)])
    return [np.asarray(cluster) for cluster in clusters]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/validation/spectral_first_fragmentation_factor_pilot/validated_factor_catalog.csv"))
    parser.add_argument("--activation-dir", type=Path, default=Path("data/validation/mass_dense_all_peak_confirmation"))
    parser.add_argument("--codes", type=Path, default=Path("data/validation/peak_token_centered_sae_seed201/confirmation_codes.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/validated_factor_exact_mass_refinement"))
    parser.add_argument("--coarse-tolerance", type=float, default=0.02)
    parser.add_argument("--cluster-ppm", type=float, default=10.0)
    parser.add_argument("--formula-ppm", type=float, default=10.0)
    parser.add_argument("--min-cluster-molecules", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(args.catalog)
    mask = np.load(args.activation_dir / "peak_mask.npy")
    values = np.load(args.activation_dir / "peak_values.npy")
    codes = np.load(args.codes, mmap_mode="r")
    spectra = json.loads((args.activation_dir / "spectra.json").read_text(encoding="utf-8"))
    counts = mask.sum(axis=1)
    spectrum_index = np.repeat(np.arange(len(spectra)), counts)
    mz = values[:, :, 0][mask].astype(np.float64)
    if len(mz) != len(codes):
        raise RuntimeError("Peak codes and flattened peak metadata are misaligned")

    cluster_rows: list[dict] = []
    formula_rows: list[dict] = []
    consensus_rows: list[dict] = []
    factor_summaries: list[dict] = []
    for row in catalog.itertuples(index=False):
        factor = int(row.factor)
        kind = str(row.spectral_kind)
        coarse_mass = float(row.mass_da)
        score = np.asarray(codes[:, factor], dtype=np.float32)
        if kind == "fragment_mz":
            observed = mz
        elif kind == "neutral_loss":
            precursor = np.asarray([record["precursor_mz"] for record in spectra], dtype=float)
            observed = precursor[spectrum_index] - mz
        else:
            continue
        # Reproduce the discovery audit's exact bin membership.  A nominal
        # 0.02-Da bin is not a +/-0.02-Da window.
        coarse_bin = int(round(coarse_mass / args.coarse_tolerance))
        coarse = np.rint(observed / args.coarse_tolerance).astype(np.int64) == coarse_bin
        active = coarse & (score > 0)
        # Main formula audit is restricted to protonated, singly charged spectra.
        protonated = np.asarray([record.get("adduct") == "[M+H]+" for record in spectra])[spectrum_index]
        selected = active & protonated
        selected_values = observed[selected]
        selected_spectra = spectrum_index[selected]
        clusters = exact_mass_clusters(selected_values, coarse_mass, args.cluster_ppm)
        ranked = []
        for cluster_id, cluster in enumerate(clusters):
            lo = cluster.min() - 1e-12
            hi = cluster.max() + 1e-12
            cluster_members = selected & (observed >= lo) & (observed <= hi)
            cluster_spectra = np.unique(spectrum_index[cluster_members])
            cluster_molecules = {spectra[i]["ik14"] for i in cluster_spectra}
            center = float(np.median(cluster))
            mad = float(np.median(np.abs(cluster - center)))
            cluster_row = {
                "factor": factor,
                "spectral_kind": kind,
                "coarse_mass_da": coarse_mass,
                "cluster_id": cluster_id,
                "cluster_center_da": center,
                "cluster_mad_da": mad,
                "active_peaks": int(len(cluster)),
                "active_spectra": int(len(cluster_spectra)),
                "active_molecules": int(len(cluster_molecules)),
                "min_mz": float(cluster.min()),
                "max_mz": float(cluster.max()),
            }
            cluster_rows.append(cluster_row)
            ranked.append(cluster_row)
            if len(cluster_molecules) < args.min_cluster_molecules:
                continue
            candidates = enumerate_chnops(center, args.formula_ppm, kind == "fragment_mz")
            precursor_compositions = [parse_formula(spectra[i].get("PRECURSOR_FORMULA")) for i in cluster_spectra]
            molecule_by_spectrum = [spectra[i]["ik14"] for i in cluster_spectra]
            for candidate_rank, candidate in enumerate(candidates, start=1):
                supported_molecules = {
                    molecule for molecule, precursor_comp in zip(molecule_by_spectrum, precursor_compositions)
                    if composition_is_subset(candidate["composition"], precursor_comp)
                }
                formula_rows.append({
                    "factor": factor,
                    "spectral_kind": kind,
                    "coarse_mass_da": coarse_mass,
                    "cluster_id": cluster_id,
                    "cluster_center_da": center,
                    "cluster_molecules": len(cluster_molecules),
                    "candidate_rank_by_mass": candidate_rank,
                    "candidate_formula": candidate["formula"],
                    "theoretical_mass": candidate["theoretical_mass"],
                    "mass_error_da": candidate["mass_error_da"],
                    "mass_error_ppm": candidate["mass_error_ppm"],
                    "dbe": candidate["dbe"],
                    "precursor_subset_molecules": len(supported_molecules),
                    "precursor_subset_fraction": len(supported_molecules) / max(len(cluster_molecules), 1),
                })
        ranked.sort(key=lambda item: (item["active_molecules"], item["active_peaks"]), reverse=True)
        # Merge instrument-shifted microclusters through a shared elemental
        # formula.  Every individual peak must still be within formula_ppm of
        # the theoretical exact mass, and precursor composition support is
        # counted at the molecule level.
        factor_formula_rows = [item for item in formula_rows if item["factor"] == factor]
        seen_formulas: dict[str, float] = {}
        for item in factor_formula_rows:
            seen_formulas.setdefault(item["candidate_formula"], item["theoretical_mass"])
        formula_consensus = []
        for formula, theoretical_mass in seen_formulas.items():
            tolerance = theoretical_mass * args.formula_ppm * 1e-6
            matched = selected & (np.abs(observed - theoretical_mass) <= tolerance)
            matched_spectra = np.unique(spectrum_index[matched])
            comp = parse_formula(formula)
            supported_spectra = [i for i in matched_spectra
                                 if composition_is_subset(comp, parse_formula(spectra[i].get("PRECURSOR_FORMULA")))]
            supported_molecules = {spectra[i]["ik14"] for i in supported_spectra}
            matched_molecules = {spectra[i]["ik14"] for i in matched_spectra}
            formula_consensus.append({
                "factor": factor,
                "spectral_kind": kind,
                "coarse_mass_da": coarse_mass,
                "candidate_formula": formula,
                "theoretical_mass": theoretical_mass,
                "matched_active_peaks": int(matched.sum()),
                "matched_active_spectra": int(len(matched_spectra)),
                "matched_active_molecules": int(len(matched_molecules)),
                "precursor_subset_molecules": int(len(supported_molecules)),
                "precursor_subset_fraction": len(supported_molecules) / max(len(matched_molecules), 1),
            })
        formula_consensus.sort(key=lambda item: (
            item["precursor_subset_molecules"], item["matched_active_molecules"], item["matched_active_peaks"]
        ), reverse=True)
        for formula_rank, item in enumerate(formula_consensus, start=1):
            item["consensus_rank"] = formula_rank
            consensus_rows.append(item)
        factor_summaries.append({
            "factor": factor,
            "spectral_kind": kind,
            "coarse_mass_da": coarse_mass,
            "protonated_active_peaks": int(selected.sum()),
            "protonated_active_molecules": len({spectra[i]["ik14"] for i in np.unique(selected_spectra)}),
            "exact_clusters": len(clusters),
            "dominant_cluster": ranked[0] if ranked else None,
            "dominant_formula_consensus": formula_consensus[0] if formula_consensus else None,
        })

    cluster_frame = pd.DataFrame(cluster_rows)
    formula_frame = pd.DataFrame(formula_rows)
    consensus_frame = pd.DataFrame(consensus_rows)
    cluster_frame.to_csv(args.output_dir / "exact_mass_clusters.csv", index=False)
    formula_frame.to_csv(args.output_dir / "candidate_formulas.csv", index=False)
    consensus_frame.to_csv(args.output_dir / "formula_consensus.csv", index=False)
    summary = {
        "status": "validated_factor_exact_mass_refinement",
        "claim_limit": "Candidate elemental compositions only; no unique substructure or mechanism is assigned by exact mass.",
        "ion_convention": "Fragment formulas are singly-positive ion elemental compositions (neutral atomic masses minus one electron mass). Neutral-loss formulas use neutral monoisotopic masses.",
        "main_audit_restriction": "[M+H]+ spectra only",
        "coarse_window_da": args.coarse_tolerance,
        "cluster_ppm": args.cluster_ppm,
        "formula_ppm": args.formula_ppm,
        "factors": factor_summaries,
    }
    (args.output_dir / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
