"""Summarize DreaMS peak-token geometry for every same-formula pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_e0_observability_residual import greedy_matches
from build_frozen_panel_pair_features import peak_masks
from attribute_large_failure_peaks import load_rules


def summaries(
    left: int, right: int, tokens: np.ndarray, mz: np.ndarray, intensity: np.ndarray,
    valid: np.ndarray, precursor: np.ndarray, panel_ids: list[str], nl_values: np.ndarray,
    tolerance: float,
) -> dict[str, float]:
    l_valid, r_valid = valid[left], valid[right]
    l_mz, r_mz = mz[left, l_valid], mz[right, r_valid]
    l_int, r_int = intensity[left, l_valid], intensity[right, r_valid]
    l_tok = tokens[left, l_valid].astype(np.float32)
    r_tok = tokens[right, r_valid].astype(np.float32)
    matches = greedy_matches(l_mz, r_mz, tolerance)
    if matches:
        cosine = np.asarray([float(l_tok[i] @ r_tok[j]) for i, j in matches])
        weights = np.asarray([np.sqrt(float(l_int[i] * r_int[j])) for i, j in matches])
        weights /= max(float(weights.sum()), 1e-12)
    else:
        cosine, weights = np.empty(0), np.empty(0)
    output = {
        "token_match_count": len(matches),
        "token_cosine_mean": float(cosine.mean()) if len(cosine) else 0.0,
        "token_cosine_min": float(cosine.min()) if len(cosine) else 0.0,
        "token_cosine_p25": float(np.quantile(cosine, 0.25)) if len(cosine) else 0.0,
        "token_cosine_p75": float(np.quantile(cosine, 0.75)) if len(cosine) else 0.0,
        "token_cosine_max": float(cosine.max()) if len(cosine) else 0.0,
        "token_cosine_weighted": float(np.sum(weights * cosine)) if len(cosine) else 0.0,
        "token_low_similarity_fraction": float(np.mean(cosine < 0.5)) if len(cosine) else 0.0,
        "token_high_similarity_fraction": float(np.mean(cosine > 0.8)) if len(cosine) else 0.0,
    }
    l_masks = peak_masks(l_mz, l_int, precursor[left], panel_ids, nl_values, tolerance)
    r_masks = peak_masks(r_mz, r_int, precursor[right], panel_ids, nl_values, tolerance)
    for feature_id in panel_ids:
        safe = feature_id.replace("::", "__").replace("%", "pct").replace("-", "_").replace(".", "p")
        selected = np.asarray([
            l_masks[feature_id][i] or r_masks[feature_id][j] for i, j in matches
        ], bool)
        selected_both = np.asarray([
            l_masks[feature_id][i] and r_masks[feature_id][j] for i, j in matches
        ], bool)
        values = cosine[selected]
        both_values = cosine[selected_both]
        output.update({
            f"token_panel_{safe}_pair_count": int(selected.sum()),
            f"token_panel_{safe}_cosine_mean": float(values.mean()) if len(values) else 0.0,
            f"token_panel_{safe}_cosine_min": float(values.min()) if len(values) else 0.0,
            f"token_panel_{safe}_low_similarity_fraction": float(np.mean(values < 0.5)) if len(values) else 0.0,
            f"token_panel_{safe}_both_cosine_mean": float(both_values.mean()) if len(both_values) else 0.0,
        })
    return output


def process(split: str, pair_dir: Path, token_root: Path, panel_ids: list[str], nl_values: np.ndarray, tolerance: float) -> pd.DataFrame:
    pairs = pd.read_csv(pair_dir / f"{split}_pair_features.csv")
    directory = token_root / split
    tokens = np.load(directory / "peak_tokens_f16.npy", mmap_mode="r")
    mz = np.load(directory / "peak_mz.npy", mmap_mode="r")
    intensity = np.load(directory / "peak_intensity.npy", mmap_mode="r")
    valid = np.load(directory / "peak_valid.npy", mmap_mode="r")
    manifest = pd.read_csv(directory / "manifest.csv")
    precursor = manifest["precursor_mz"].to_numpy(float)
    rows = []
    for position, pair in enumerate(pairs.itertuples(index=False), start=1):
        rows.append({"left": int(pair.left), "right": int(pair.right)} | summaries(
            int(pair.left), int(pair.right), tokens, mz, intensity, valid, precursor,
            panel_ids, nl_values, tolerance,
        ))
        if position % 10000 == 0:
            print(f"  {split}: {position:,}/{len(pairs):,}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["discovery", "confirmation"])
    parser.add_argument("--pair-dir", type=Path, default=Path("data/validation/large_observability_residual_audit"))
    parser.add_argument("--token-root", type=Path, default=Path("data/validation/official_peak_tokens"))
    parser.add_argument("--panel", type=Path, default=Path("data/validation/large_failure_peak_evidence_strata/frozen_test_panel.csv"))
    parser.add_argument("--rules", type=Path, default=Path("dreams/models/chem_aware/chem_rules_data.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/peak_token_pair_features"))
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_ids = pd.read_csv(args.panel)["feature_id"].tolist()
    rules = load_rules(args.rules)
    nl_values = np.asarray(sorted({float(rule["value"]) for rule in rules if rule["category"] == "NL"}), float)
    report = {"status": "peak_token_pair_features", "splits": {}}
    for split in args.splits:
        frame = process(split, args.pair_dir, args.token_root, panel_ids, nl_values, args.tolerance)
        frame.to_csv(args.output_dir / f"{split}_token_pair_features.csv", index=False)
        report["splits"][split] = {"pairs": len(frame), "feature_columns": len(frame.columns) - 2}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
