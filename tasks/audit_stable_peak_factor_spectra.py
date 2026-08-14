"""Test stable peak-token factors against exact spectral evidence.

Candidate fragment m/z or neutral-loss bins are selected on discovery peaks
only.  The exact candidate bins are then tested without reselection on the
molecule-disjoint confirmation split.  No curated fragmentation rule is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--stability", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--minimum-active-peaks", type=int, default=8)
    parser.add_argument("--minimum-active-molecules", type=int, default=5)
    return parser.parse_args()


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def load_peak_table(directory: Path, codes_path: Path) -> dict:
    spectra = json.loads((directory / "spectra.json").read_text(encoding="utf-8"))
    mask = np.load(directory / "peak_mask.npy")
    values = np.load(directory / "peak_values.npy")
    codes = np.load(codes_path, mmap_mode="r").astype(np.float32)
    counts = mask.sum(axis=1)
    spectrum_index = np.repeat(np.arange(len(spectra)), counts)
    molecule = np.asarray([record["ik14"] for record in spectra], dtype=str)[spectrum_index]
    mz = values[:, :, 0][mask].astype(np.float64)
    intensity = values[:, :, 1][mask].astype(np.float64)
    precursor_by_spectrum = np.asarray([record["precursor_mz"] for record in spectra], dtype=float)
    collision_by_spectrum = np.asarray(
        [record.get("COLLISION_ENERGY") for record in spectra], dtype=float
    )
    precursor = precursor_by_spectrum[spectrum_index]
    collision = collision_by_spectrum[spectrum_index]
    if len(codes) != len(mz):
        raise RuntimeError(f"Code/peak mismatch: {len(codes)} vs {len(mz)}")
    return {
        "spectra": spectra,
        "spectrum_index": spectrum_index,
        "molecule": molecule,
        "mz": mz,
        "intensity": intensity,
        "precursor": precursor,
        "neutral_loss": precursor - mz,
        "collision_energy": collision,
        "codes": codes,
    }


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
        return float("nan")
    return float(spearmanr(x[finite], y[finite]).statistic)


def bin_ids(values: np.ndarray, width: float) -> np.ndarray:
    return np.rint(values / width).astype(np.int64)


def contingency(active: np.ndarray, in_bin: np.ndarray) -> tuple[np.ndarray, float, float]:
    a = int(np.sum(active & in_bin))
    b = int(np.sum(active & ~in_bin))
    c = int(np.sum(~active & in_bin))
    d = int(np.sum(~active & ~in_bin))
    table = np.asarray([[a, b], [c, d]], dtype=np.int64)
    odds, p = fisher_exact(table, alternative="greater")
    active_rate = (a + 0.5) / (a + b + 1)
    background_rate = (c + 0.5) / (c + d + 1)
    return table, float(np.log2(active_rate / background_rate)), float(p)


def choose_candidate(table: dict, active: np.ndarray, values: np.ndarray, args: argparse.Namespace) -> dict:
    ids = bin_ids(values, args.bin_width)
    candidates = []
    for value in np.unique(ids[active]):
        in_bin = ids == value
        active_in_bin = active & in_bin
        active_count = int(active_in_bin.sum())
        molecule_count = len(np.unique(table["molecule"][active_in_bin]))
        if active_count < args.minimum_active_peaks or molecule_count < args.minimum_active_molecules:
            continue
        matrix, enrichment, p = contingency(active, in_bin)
        candidates.append({
            "bin_id": int(value),
            "mass_da": float(value * args.bin_width),
            "active_peaks": active_count,
            "active_molecules": molecule_count,
            "log2_enrichment": enrichment,
            "p": p,
            "contingency": matrix.tolist(),
        })
    if not candidates:
        return {"found": False}
    # Enrichment first, then cross-molecule support; discovery p is descriptive
    # because this bin was selected after screening many bins.
    best = max(candidates, key=lambda item: (item["log2_enrichment"], item["active_molecules"], item["active_peaks"]))
    best["found"] = True
    best["screened_bins"] = len(candidates)
    return best


def confirm_candidate(table: dict, active: np.ndarray, candidate: dict, values: np.ndarray, width: float) -> dict:
    if not candidate.get("found"):
        return {"tested": False}
    in_bin = bin_ids(values, width) == candidate["bin_id"]
    matrix, enrichment, p = contingency(active, in_bin)
    active_in_bin = active & in_bin
    return {
        "tested": True,
        "mass_da": candidate["mass_da"],
        "active_peaks": int(active_in_bin.sum()),
        "active_spectra": int(len(np.unique(table["spectrum_index"][active_in_bin]))),
        "active_molecules": int(len(np.unique(table["molecule"][active_in_bin]))),
        "log2_enrichment": enrichment,
        "p": p,
        "contingency": matrix.tolist(),
    }


def condition_recurrence(table: dict, active: np.ndarray, width: float) -> dict:
    """Measure exact active-bin recurrence across the two spectra per pair."""
    ids = bin_ids(table["mz"], width)
    molecule_to_spectra: dict[str, list[int]] = {}
    for i, record in enumerate(table["spectra"]):
        molecule_to_spectra.setdefault(record["ik14"], []).append(i)
    eligible = 0
    recurrent = 0
    jaccards = []
    for spectra in molecule_to_spectra.values():
        if len(spectra) != 2:
            continue
        sets = []
        for spectrum in spectra:
            mask = active & (table["spectrum_index"] == spectrum)
            sets.append(set(ids[mask].tolist()))
        if not sets[0] or not sets[1]:
            continue
        eligible += 1
        intersection = sets[0] & sets[1]
        union = sets[0] | sets[1]
        recurrent += bool(intersection)
        jaccards.append(len(intersection) / len(union))
    return {
        "pairs_active_in_both_spectra": eligible,
        "pairs_with_exact_mz_recurrence": recurrent,
        "recurrence_fraction": recurrent / eligible if eligible else float("nan"),
        "median_active_mz_jaccard": float(np.median(jaccards)) if jaccards else float("nan"),
    }


def main() -> None:
    args = parse_args()
    stability = json.loads(args.stability.read_text(encoding="utf-8"))
    factors = [int(value) for value in stability["stable_feature_ids"]]
    if not factors:
        raise RuntimeError("No stable factors passed the preregistered stability screen")
    discovery = load_peak_table(args.discovery, args.run / "discovery_codes.npy")
    confirmation = load_peak_table(args.confirmation, args.run / "confirmation_codes.npy")
    rows = []
    confirmation_tests = []
    for factor in factors:
        discovery_score = discovery["codes"][:, factor]
        confirmation_score = confirmation["codes"][:, factor]
        discovery_active = discovery_score > 0
        confirmation_active = confirmation_score > 0
        fragment = choose_candidate(discovery, discovery_active, discovery["mz"], args)
        loss = choose_candidate(discovery, discovery_active, discovery["neutral_loss"], args)
        fragment_confirmation = confirm_candidate(confirmation, confirmation_active, fragment, confirmation["mz"], args.bin_width)
        loss_confirmation = confirm_candidate(confirmation, confirmation_active, loss, confirmation["neutral_loss"], args.bin_width)
        for kind, result in (("fragment_mz", fragment_confirmation), ("neutral_loss", loss_confirmation)):
            if result.get("tested"):
                confirmation_tests.append((factor, kind, result))
        rows.append({
            "factor": factor,
            "discovery": {
                "active_peaks": int(discovery_active.sum()),
                "active_spectra": int(len(np.unique(discovery["spectrum_index"][discovery_active]))),
                "active_molecules": int(len(np.unique(discovery["molecule"][discovery_active]))),
                "activation_fraction": float(discovery_active.mean()),
                "score_spearman": {
                    "fragment_mz": safe_spearman(discovery_score, discovery["mz"]),
                    "neutral_loss": safe_spearman(discovery_score, discovery["neutral_loss"]),
                    "log_intensity": safe_spearman(discovery_score, np.log1p(discovery["intensity"])),
                    "precursor_mz": safe_spearman(discovery_score, discovery["precursor"]),
                    "collision_energy": safe_spearman(discovery_score, discovery["collision_energy"]),
                },
                "fragment_candidate": fragment,
                "neutral_loss_candidate": loss,
                "condition_recurrence": condition_recurrence(discovery, discovery_active, args.bin_width),
            },
            "confirmation": {
                "active_peaks": int(confirmation_active.sum()),
                "active_spectra": int(len(np.unique(confirmation["spectrum_index"][confirmation_active]))),
                "active_molecules": int(len(np.unique(confirmation["molecule"][confirmation_active]))),
                "activation_fraction": float(confirmation_active.mean()),
                "score_spearman": {
                    "fragment_mz": safe_spearman(confirmation_score, confirmation["mz"]),
                    "neutral_loss": safe_spearman(confirmation_score, confirmation["neutral_loss"]),
                    "log_intensity": safe_spearman(confirmation_score, np.log1p(confirmation["intensity"])),
                    "precursor_mz": safe_spearman(confirmation_score, confirmation["precursor"]),
                    "collision_energy": safe_spearman(confirmation_score, confirmation["collision_energy"]),
                },
                "fragment_test": fragment_confirmation,
                "neutral_loss_test": loss_confirmation,
                "condition_recurrence": condition_recurrence(confirmation, confirmation_active, args.bin_width),
            },
        })
    if confirmation_tests:
        p = np.asarray([item[2]["p"] for item in confirmation_tests], dtype=float)
        q = benjamini_hochberg(p)
        for (_, _, result), value in zip(confirmation_tests, q):
            result["bh_q_across_fixed_confirmation_tests"] = float(value)
    report = {
        "status": "stable_peak_factor_spectral_audit",
        "bin_width_da": args.bin_width,
        "stable_factors_tested": factors,
        "rules_read": False,
        "selection_protocol": "Candidate exact masses selected on discovery only and tested unchanged on confirmation.",
        "factors": rows,
        "claim_limit": "Exact-mass recurrence supports spectral localization, not a molecular structure or fragmentation mechanism assignment.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
