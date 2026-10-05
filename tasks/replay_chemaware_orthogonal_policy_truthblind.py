"""Replay a frozen ChemAware policy with truth absent from the inference path."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries_truthblind, strict_rank
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_truthblind_candidate_core import predict_truthblind_policy
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--token-dir", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--rule-library", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_ranks(
    predictions: dict[str, np.ndarray], scored: dict[str, np.ndarray],
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.empty(len(labels), dtype=np.int16)
    proposed = np.empty(len(labels), dtype=np.int16)
    for index in range(len(labels)):
        pointer = np.asarray(scored["reference_ptr"][index], dtype=np.int64)
        official = np.maximum.reduceat(
            np.asarray(scored["global"][index], dtype=np.float32), pointer[:-1],
        ).astype(np.float64)
        truth = np.asarray(labels[index], dtype=bool)
        baseline[index] = strict_rank(official, truth)
        if predictions["abstained"][index]:
            proposed[index] = baseline[index]
            continue
        selected = int(predictions["selected_candidate"][index])
        promoted = official.copy()
        promoted[selected] = np.nextafter(float(np.max(official)), np.inf)
        proposed[index] = strict_rank(promoted, truth)
    return baseline, proposed


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    policy_path = args.policy_dir / "truthblind_policy.joblib"
    expected_path = args.policy_dir / "inner_policy.npz"
    report_path = args.policy_dir / "report.json"
    for path in (policy_path, expected_path, report_path, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    bundle = joblib.load(policy_path)
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    replay_args = source_report["replay_contract"]["arguments"]
    with np.load(args.manifest, allow_pickle=False) as loaded:
        split_body = {
            "query_formula": np.asarray(loaded["query_formula"]),
            "query_ik14": np.asarray(loaded["query_ik14"]),
        }
        # This is the complete body visible to inference.  The truth label
        # array is deliberately not copied into it.
        truthblind_body = {
            key: np.asarray(loaded[key])
            for key in ("query_ptr", "molecule_ptr", "pair_candidate_row", "query_row")
        }
    fold = stable_formula_folds(
        split_body["query_formula"], int(replay_args["folds"]), int(replay_args["fold_seed"]),
    )
    pool = np.flatnonzero(fold == 3)
    queries = identity_balanced_queries(
        pool, split_body["query_ik14"],
        np.random.default_rng(int(replay_args["sampling_seed"]) + 19),
        int(replay_args["max_inner_identities"]),
    )
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    kernel_args = SimpleNamespace(
        token_dir=args.token_dir,
        rule_library=args.rule_library,
        top_peaks=int(replay_args["top_peaks"]),
        kernel_dim=int(replay_args["kernel_dim"]),
        bin_width=float(replay_args["bin_width"]),
        grid_offsets=int(replay_args["grid_offsets"]),
        intensity_power=float(replay_args["intensity_power"]),
        mass_shift_da=float(replay_args["mass_shift_da"]),
        pair_weight=float(replay_args["pair_weight"]),
        multi_bin_widths=tuple(replay_args["multi_bin_widths"]),
        uniform_channel_weight=float(replay_args["uniform_channel_weight"]),
        rule_tolerance=float(replay_args["rule_tolerance"]),
        rule_channel_weight=float(replay_args["rule_channel_weight"]),
    )
    variants = tuple(dict.fromkeys((
        "mass", str(bundle["rule_key"]), *map(str, bundle["control_rule_keys"]),
    )))
    cache = KernelCache(kernel_args, row_position, variants=variants)
    scored = score_queries_truthblind(
        queries, truthblind_body, official, row_position, cache, variants,
    )
    predictions = predict_truthblind_policy(
        bundle, scored, [scored for _ in bundle["control_rule_keys"]],
    )

    # Freeze predictions before opening the truth array for evaluation.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_truthblind_", dir=args.output.parent))
    try:
        prediction_path = temporary / "predictions.npz"
        np.savez_compressed(
            prediction_path,
            query=queries,
            selected_candidate=predictions["selected_candidate"],
            selected_candidate_slot=predictions["selected_candidate_slot"],
            abstained=predictions["abstained"],
            utility=predictions["utility"],
            benefit=predictions["benefit"],
            harm=predictions["harm"],
            proposed_candidate=predictions["proposed_candidate"],
            valid_candidate=predictions["valid"],
        )
        with np.load(args.manifest, allow_pickle=False) as loaded:
            labels = np.empty(len(queries), dtype=object)
            for out_index, query in enumerate(map(int, queries)):
                left, right = map(int, loaded["query_ptr"][query:query + 2])
                labels[out_index] = np.asarray(loaded["molecule_label"][left:right], dtype=bool)
        baseline_rank, proposed_rank = evaluation_ranks(predictions, scored, labels)
        with np.load(expected_path, allow_pickle=False) as expected:
            query_equal = np.array_equal(queries, expected["query"])
            slot_equal = np.array_equal(
                predictions["selected_candidate_slot"], expected["correct_selected_candidate_slot"],
            )
            rank_equal = np.array_equal(proposed_rank, expected["correct_rank"])
            baseline_equal = np.array_equal(baseline_rank, expected["baseline_rank"])
            expected_utility = np.asarray(expected["correct_candidate_utility"], dtype=np.float64)
            finite = np.isfinite(expected_utility) & np.isfinite(predictions["utility"])
            utility_max_abs = float(np.max(np.abs(
                predictions["utility"][finite] - expected_utility[finite]
            ))) if finite.any() else 0.0
            utility_finite_equal = np.array_equal(
                np.isfinite(predictions["utility"]), np.isfinite(expected_utility),
            )
        passed = bool(
            query_equal and slot_equal and rank_equal and baseline_equal
            and utility_finite_equal and utility_max_abs <= 1e-12
        )
        report = {
            "status": (
                "CHEMAWARE_TRUTHBLIND_POLICY_REPLAY_PASS"
                if passed else "CHEMAWARE_TRUTHBLIND_POLICY_REPLAY_FAIL"
            ),
            "inference_truth_fields": [],
            "queries": int(len(queries)),
            "selected": int(np.sum(~predictions["abstained"])),
            "query_equal": query_equal,
            "selected_slot_equal": slot_equal,
            "rank_equal": rank_equal,
            "baseline_equal": baseline_equal,
            "utility_finite_mask_equal": utility_finite_equal,
            "utility_max_abs_error": utility_max_abs,
            "policy_sha256": sha256(policy_path),
            "prediction_sha256": sha256(prediction_path),
            "outer_fold_untouched": True,
        }
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not passed:
            raise RuntimeError(json.dumps(report, sort_keys=True))
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
