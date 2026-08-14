"""Test whether a factor localizes exact masses beyond spectrum-level context.

For each fixed candidate mass, the null preserves the number and magnitude of
factor activations in every spectrum and permutes them only among that
spectrum's peaks.  This controls the strong within-spectrum dependence of
contextual DreaMS peak tokens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-audit", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--factor-ids", type=int, nargs="+", default=None,
        help="Optional subset of preregistered stable factor ids to test.",
    )
    return parser.parse_args()


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def load_table(directory: Path, codes_path: Path) -> dict:
    spectra = json.loads((directory / "spectra.json").read_text(encoding="utf-8"))
    mask = np.load(directory / "peak_mask.npy")
    values = np.load(directory / "peak_values.npy")
    codes = np.load(codes_path, mmap_mode="r").astype(np.float32)
    counts = mask.sum(axis=1)
    spectrum_index = np.repeat(np.arange(len(spectra)), counts)
    mz = values[:, :, 0][mask].astype(np.float64)
    precursor = np.asarray([item["precursor_mz"] for item in spectra], dtype=float)[spectrum_index]
    if len(codes) != len(mz):
        raise RuntimeError("Code/peak alignment mismatch")
    return {
        "spectra": spectra,
        "spectrum_index": spectrum_index,
        "mz": mz,
        "neutral_loss": precursor - mz,
        "codes": codes,
    }


def target_mask(values: np.ndarray, mass: float, width: float) -> np.ndarray:
    return np.rint(values / width).astype(np.int64) == int(round(mass / width))


def within_spectrum_test(
    table: dict,
    scores: np.ndarray,
    target: np.ndarray,
    permutations: int,
    seed: int,
) -> dict:
    """Permutation tests for binary hits and activation-magnitude localization."""
    rng = np.random.default_rng(seed)
    spectra = np.unique(table["spectrum_index"][target])
    observed_binary = 0
    observed_score = 0.0
    eligible_spectra = 0
    target_peaks = 0
    active_target_peaks = 0
    spectrum_blocks = []
    for spectrum in spectra:
        indices = np.flatnonzero(table["spectrum_index"] == spectrum)
        local_target = target[indices]
        if not np.any(local_target) or np.all(local_target):
            continue
        local_scores = scores[indices]
        local_active = local_scores > 0
        eligible_spectra += 1
        target_peaks += int(local_target.sum())
        active_target_peaks += int(np.sum(local_active & local_target))
        observed_binary += int(np.sum(local_active & local_target))
        observed_score += float(local_scores[local_target].sum())
        spectrum_blocks.append((local_active.astype(np.int8), local_scores, local_target))
    null_binary = np.zeros(permutations, dtype=np.float64)
    null_score = np.zeros(permutations, dtype=np.float64)
    for active, local_scores, local_target in spectrum_blocks:
        n = len(active)
        # Independent permutations per spectrum preserve its activation count
        # and full score distribution while breaking peak-position assignment.
        for permutation in range(permutations):
            order = rng.permutation(n)
            null_binary[permutation] += active[order][local_target].sum()
            null_score[permutation] += local_scores[order][local_target].sum()
    binary_p = (1 + int(np.sum(null_binary >= observed_binary))) / (permutations + 1)
    score_p = (1 + int(np.sum(null_score >= observed_score))) / (permutations + 1)
    return {
        "eligible_spectra_containing_target": eligible_spectra,
        "target_peaks": target_peaks,
        "active_target_peaks": active_target_peaks,
        "observed_active_target_hits": int(observed_binary),
        "null_active_target_hits_mean": float(null_binary.mean()),
        "null_active_target_hits_p95": float(np.quantile(null_binary, 0.95)),
        "binary_within_spectrum_permutation_p": binary_p,
        "observed_target_activation_sum": observed_score,
        "null_target_activation_sum_mean": float(null_score.mean()),
        "null_target_activation_sum_p95": float(np.quantile(null_score, 0.95)),
        "score_within_spectrum_permutation_p": score_p,
        "localization_pass": bool(binary_p <= 0.05 and score_p <= 0.05),
    }


def main() -> None:
    args = parse_args()
    audit = json.loads(args.spectral_audit.read_text(encoding="utf-8"))
    width = float(audit["bin_width_da"])
    discovery = load_table(args.discovery, args.run / "discovery_codes.npy")
    confirmation = load_table(args.confirmation, args.run / "confirmation_codes.npy")
    rows = []
    test_index = 0
    confirmation_results = []
    for factor_item in audit["factors"]:
        factor = int(factor_item["factor"])
        if args.factor_ids is not None and factor not in set(args.factor_ids):
            continue
        item = {"factor": factor, "discovery": {}, "confirmation": {}}
        for kind, discovery_key, confirmation_key, value_key in (
            ("fragment_mz", "fragment_candidate", "fragment_test", "mz"),
            ("neutral_loss", "neutral_loss_candidate", "neutral_loss_test", "neutral_loss"),
        ):
            candidate = factor_item["discovery"][discovery_key]
            confirmed = factor_item["confirmation"][confirmation_key]
            if not candidate.get("found") or not confirmed.get("tested"):
                continue
            mass = float(candidate["mass_da"])
            for split_name, table in (("discovery", discovery), ("confirmation", confirmation)):
                target = target_mask(table[value_key], mass, width)
                scores = table["codes"][:, factor]
                item[split_name][kind] = within_spectrum_test(
                    table, scores, target, args.permutations, args.seed + test_index
                )
                item[split_name][kind]["fixed_mass_da"] = mass
                if split_name == "confirmation":
                    confirmation_results.append(item[split_name][kind])
                test_index += 1
        rows.append(item)
    if confirmation_results:
        # Both localization tests must pass, so max(p_binary, p_score) is the
        # conservative per-candidate p-value.  Correct across all fixed masses
        # tested on confirmation.
        joint_p = np.asarray([
            max(
                item["binary_within_spectrum_permutation_p"],
                item["score_within_spectrum_permutation_p"],
            )
            for item in confirmation_results
        ])
        joint_q = benjamini_hochberg(joint_p)
        for item, p_value, q_value in zip(
            confirmation_results, joint_p, joint_q
        ):
            item["conservative_joint_p"] = float(p_value)
            item["bh_q_across_confirmation_candidates"] = float(q_value)
            item["localization_pass_bh"] = bool(q_value <= 0.05)
    report = {
        "status": "peak_factor_within_spectrum_localization_audit",
        "permutations": args.permutations,
        "null": "Factor activation locations are exchangeable only within each spectrum; activation counts and score values per spectrum are fixed.",
        "factors": rows,
        "claim_rule": "A fixed mass is peak-localized only when both binary-hit and score-sum tests pass on confirmation.",
        "multiple_testing": (
            "For each candidate use max(binary p, score p), then apply "
            "Benjamini-Hochberg across all fixed confirmation candidates."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
