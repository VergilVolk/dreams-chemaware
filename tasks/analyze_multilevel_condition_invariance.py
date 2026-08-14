"""Audit whether DreaMS model differences reflect identity or acquisition conditions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data/validation/multilevel_factor_pilot1000_qc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--chunk-size", type=int, default=64)
    return parser.parse_args()


def ce_bin(value) -> str:
    if value is None or not np.isfinite(value):
        return "unknown"
    if value < 20:
        return "low"
    if value < 45:
        return "medium"
    return "high"


def row_cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    numerator = np.sum(x * y, axis=-1)
    denominator = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return numerator / np.clip(denominator, 1e-12, None)


def pooled_peaks(path: Path, mask: np.ndarray, chunk_size: int) -> np.ndarray:
    peaks = np.load(path, mmap_mode="r")
    output = np.empty((peaks.shape[0], peaks.shape[1], peaks.shape[-1]), dtype=np.float32)
    for start in range(0, len(peaks), chunk_size):
        end = min(start + chunk_size, len(peaks))
        values = np.asarray(peaks[start:end], dtype=np.float32)
        weights = mask[start:end, None, :, None].astype(np.float32)
        output[start:end] = (
            (values * weights).sum(axis=2)
            / weights.sum(axis=2).clip(min=1)
        )
    return output


def build_row_metadata(pairs: list[dict]) -> list[dict]:
    rows = []
    for pair in pairs:
        for view in (0, 1):
            rows.append({
                "pair_id": pair["pair_id"],
                "ik14": pair["ik14"],
                "instrument": pair["instrument"][view],
                "adduct": pair["adduct"][view],
                "collision_energy": pair["collision_energy"][view],
                "ce_bin": ce_bin(pair["collision_energy"][view]),
                "precursor_mz": pair["precursor_mz"][view],
            })
    return rows


def select_condition_matched_negatives(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    """For each spectrum, choose another molecule with matched conditions.

    Matching relaxes in a declared order when the full condition tuple has no
    other molecule. Within the best available group, precursor m/z is nearest
    to the anchor, producing a harder and less condition-confounded negative.
    """
    group_specs = [
        lambda item: (item["instrument"], item["adduct"], item["ce_bin"]),
        lambda item: (item["instrument"], item["adduct"]),
        lambda item: (item["adduct"], item["ce_bin"]),
        lambda item: (item["adduct"],),
        lambda item: ("all",),
    ]
    labels = ["instrument+adduct+CE", "instrument+adduct", "adduct+CE", "adduct", "all"]
    grouped_levels = []
    for spec in group_specs:
        groups: dict[tuple, list[int]] = defaultdict(list)
        for index, item in enumerate(rows):
            groups[spec(item)].append(index)
        grouped_levels.append((spec, groups))

    negatives, match_levels = [], []
    for anchor_index, anchor in enumerate(rows):
        chosen = None
        chosen_level = None
        for level, ((spec, groups), label) in enumerate(zip(grouped_levels, labels)):
            candidates = [
                index for index in groups[spec(anchor)]
                if rows[index]["pair_id"] != anchor["pair_id"]
            ]
            if candidates:
                chosen = min(
                    candidates,
                    key=lambda index: (
                        abs(rows[index]["precursor_mz"] - anchor["precursor_mz"]),
                        index,
                    ),
                )
                chosen_level = label
                break
        if chosen is None:
            raise RuntimeError(f"No negative candidate for row {anchor_index}")
        negatives.append(chosen)
        match_levels.append(chosen_level)
    return np.asarray(negatives, dtype=np.int64), match_levels


def identity_metrics(
    representations: np.ndarray,
    negative_indices: np.ndarray,
    layers: list[int],
) -> list[dict]:
    positive_indices = np.arange(len(representations)) ^ 1
    results = []
    for layer_index, layer in enumerate(layers):
        anchor = representations[:, layer_index]
        positive = row_cosine(anchor, representations[positive_indices, layer_index])
        negative = row_cosine(anchor, representations[negative_indices, layer_index])
        labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
        scores = np.concatenate([positive, negative])
        results.append({
            "layer": layer,
            "positive_mean": float(positive.mean()),
            "negative_mean": float(negative.mean()),
            "separation": float((positive - negative).mean()),
            "triplet_accuracy": float(np.mean(positive > negative)),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
        })
    return results


def stratified_pair_invariance(
    raw: np.ndarray,
    official: np.ndarray,
    pairs: list[dict],
    layers: list[int],
    seed: int = 42,
    bootstrap_repeats: int = 2000,
) -> dict:
    raw_pair = row_cosine(raw[0::2], raw[1::2])
    official_pair = row_cosine(official[0::2], official[1::2])
    categories: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        diff = pair["condition_difference"]
        label = (
            f"instrument={diff['instrument']},"
            f"adduct={diff['adduct']},"
            f"ce={diff['collision_energy_ge_10']}"
        )
        categories[label].append(index)

    output = {}
    rng = np.random.RandomState(seed)
    for label, indices in sorted(categories.items()):
        idx = np.asarray(indices, dtype=np.int64)
        output[label] = []
        for layer_index, layer in enumerate(layers):
            raw_values = raw_pair[idx, layer_index]
            official_values = official_pair[idx, layer_index]
            paired_delta = official_values - raw_values
            bootstrap_indices = rng.randint(
                0, len(paired_delta), size=(bootstrap_repeats, len(paired_delta))
            )
            bootstrap_means = paired_delta[bootstrap_indices].mean(axis=1)
            output[label].append({
                "layer": layer,
                "n_pairs": len(idx),
                "raw_mean": float(raw_values.mean()),
                "official_mean": float(official_values.mean()),
                "official_minus_raw": float(paired_delta.mean()),
                "official_minus_raw_ci95": [
                    float(np.quantile(bootstrap_means, 0.025)),
                    float(np.quantile(bootstrap_means, 0.975)),
                ],
            })
    return output


def main() -> None:
    args = parse_args()
    report = json.loads((args.input_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "multilevel_activation_pilot":
        raise RuntimeError(
            f"Input run status is {report.get('status')!r}; refusing analysis"
        )
    layers = report["config"]["layers"]
    pairs = json.loads((args.input_dir / "pairs.json").read_text(encoding="utf-8"))
    row_metadata = build_row_metadata(pairs)
    negative_indices, match_levels = select_condition_matched_negatives(row_metadata)

    raw_precursor = np.load(args.input_dir / "raw_precursor.npy")
    official_precursor = np.load(args.input_dir / "official_precursor.npy")
    peak_mask = np.load(args.input_dir / "peak_mask.npy")
    raw_peak = pooled_peaks(args.input_dir / "raw_peak.npy", peak_mask, args.chunk_size)
    official_peak = pooled_peaks(
        args.input_dir / "official_peak.npy", peak_mask, args.chunk_size
    )

    negative_mz_delta = np.asarray([
        abs(row_metadata[i]["precursor_mz"] - row_metadata[j]["precursor_mz"])
        for i, j in enumerate(negative_indices)
    ])
    output = {
        "status": "condition_invariance_audit",
        "input": str(args.input_dir.resolve()),
        "n_spectra": len(row_metadata),
        "n_molecules": len(pairs),
        "negative_matching": {
            "level_counts": dict(sorted({
                key: match_levels.count(key) for key in set(match_levels)
            }.items())),
            "precursor_mz_delta_median": float(np.median(negative_mz_delta)),
            "precursor_mz_delta_p90": float(np.quantile(negative_mz_delta, 0.9)),
            "note": "Different-molecule negatives match conditions first, then nearest precursor m/z.",
        },
        "precursor_identity": {
            "raw": identity_metrics(raw_precursor, negative_indices, layers),
            "official": identity_metrics(official_precursor, negative_indices, layers),
        },
        "pooled_peak_identity": {
            "raw": identity_metrics(raw_peak, negative_indices, layers),
            "official": identity_metrics(official_peak, negative_indices, layers),
        },
        "precursor_pair_invariance_by_condition": stratified_pair_invariance(
            raw_precursor, official_precursor, pairs, layers
        ),
        "pooled_peak_pair_invariance_by_condition": stratified_pair_invariance(
            raw_peak, official_peak, pairs, layers
        ),
        "limits": [
            "This audit evaluates invariance and identity separation, not factor semantics.",
            "Peak representations are mean-pooled over 24 sampled peak tokens.",
            "Negatives are condition-matched but not guaranteed to fall within 10 ppm.",
        ],
    }
    out_path = args.input_dir / "condition_invariance_report.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Condition-invariance audit", flush=True)
    print("Negative matching:", output["negative_matching"], flush=True)
    for representation in ("precursor_identity", "pooled_peak_identity"):
        print(f"\n{representation}")
        raw_by_layer = {item["layer"]: item for item in output[representation]["raw"]}
        official_by_layer = {item["layer"]: item for item in output[representation]["official"]}
        for layer in layers:
            raw_item = raw_by_layer[layer]
            official_item = official_by_layer[layer]
            print(
                f"  L{layer}: raw AUC={raw_item['roc_auc']:.4f}, "
                f"sep={raw_item['separation']:.4f}; "
                f"official AUC={official_item['roc_auc']:.4f}, "
                f"sep={official_item['separation']:.4f}",
                flush=True,
            )
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
