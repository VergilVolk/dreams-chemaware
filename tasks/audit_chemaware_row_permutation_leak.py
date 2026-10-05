"""Detect library-row identity leakage in the legacy rule permutation control."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from audit_chemaware_counterfactual_rule_kernel import sha256


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--rule-library", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_row_permutation_leak_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    payload = json.loads(args.rule_library.read_text(encoding="utf-8"))
    records = payload.get("channels", payload.get("rules"))
    channels = sum(
        1 for item in records
        if (
            item.get("category") == "NL" and item.get("match_type") == "mass_diff"
        ) or (
            item.get("category") == "CF" and item.get("match_type") == "peak_mz"
        )
    )
    modulus = max(1, channels - 1)
    positive_equal: list[float] = []
    negative_equal: list[float] = []
    positive_distance: list[float] = []
    negative_distance: list[float] = []
    row_only_rank = []
    for query in range(len(body["query_row"])):
        query_offset = 1 + int(body["query_row"][query]) % modulus
        left, right = map(int, body["query_ptr"][query:query + 2])
        candidate_scores = []
        candidate_labels = []
        for molecule in range(left, right):
            ref_left, ref_right = map(int, body["molecule_ptr"][molecule:molecule + 2])
            offsets = 1 + body["pair_candidate_row"][ref_left:ref_right].astype(np.int64) % modulus
            equal = offsets == query_offset
            distance = np.abs(offsets - query_offset)
            distance = np.minimum(distance, modulus - distance)
            label = bool(body["molecule_label"][molecule])
            (positive_equal if label else negative_equal).append(float(np.mean(equal)))
            (positive_distance if label else negative_distance).append(float(np.min(distance)))
            candidate_scores.append(float(np.max(equal)))
            candidate_labels.append(label)
        scores = np.asarray(candidate_scores)
        labels = np.asarray(candidate_labels, dtype=bool)
        positive = float(scores[np.flatnonzero(labels)[0]])
        row_only_rank.append(1 + int(np.sum(scores[~labels] >= positive)))
    positive_equal_array = np.asarray(positive_equal)
    negative_equal_array = np.asarray(negative_equal)
    positive_distance_array = np.asarray(positive_distance)
    negative_distance_array = np.asarray(negative_distance)
    rank = np.asarray(row_only_rank)
    report = {
        "status": "CHEMAWARE_ROW_PERMUTATION_CONTROL_LEAK_CONFIRMED",
        "control_invalid": True,
        "reason": (
            "The legacy control rotates rule channels by spectrum_row modulo channel count. "
            "Positive references are strongly adjacent to their query in library-row order, so "
            "the control exposes dataset ordering and is not chemistry-null."
        ),
        "rule_channels": int(channels),
        "rotation_modulus": int(modulus),
        "molecules": {
            "positive": int(len(positive_equal_array)),
            "negative": int(len(negative_equal_array)),
        },
        "same_rotation": {
            "positive_mean": float(np.mean(positive_equal_array)),
            "negative_mean": float(np.mean(negative_equal_array)),
            "positive_any_fraction": float(np.mean(positive_equal_array > 0)),
            "negative_any_fraction": float(np.mean(negative_equal_array > 0)),
        },
        "minimum_circular_rotation_distance": {
            "quantiles": [0.0, 0.25, 0.5, 0.75, 0.9, 0.99],
            "positive": np.quantile(positive_distance_array, [0, .25, .5, .75, .9, .99]).tolist(),
            "negative": np.quantile(negative_distance_array, [0, .25, .5, .75, .9, .99]).tolist(),
        },
        "row_rotation_only_candidate_retrieval": {
            "strict_recall1": float(np.mean(rank == 1)),
            "strict_recall5": float(np.mean(rank <= 5)),
            "mean_rank": float(np.mean(rank)),
            "claim_limit": "Leak diagnostic only; not a usable model or performance result.",
        },
        "replacement_control": (
            "rule_response_content_permuted uses an avalanche hash of mz, intensity and precursor; "
            "it never reads the dataset row or source ordering."
        ),
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "rule_library_sha256": sha256(args.rule_library),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_row_leak_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    except Exception:
        for path in temporary.glob("*"):
            path.unlink(missing_ok=True)
        temporary.rmdir()
        raise
    print(json.dumps({
        "status": report["status"],
        "minimum_distance": report["minimum_circular_rotation_distance"],
        "row_rotation_only": report["row_rotation_only_candidate_retrieval"],
        "output": str((args.output / "report.json").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
