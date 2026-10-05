"""Open the sealed ChemAware formula outer fold exactly once.

The script first replays the release candidate on the already-consumed inner
fold and requires exact agreement with run 2338337.  It then creates an atomic
one-shot seal, computes outer predictions without labels/formulas in the
inference body, freezes those predictions to disk, and only afterwards opens
the truth arrays for aggregate evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries_truthblind
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_outer_evaluation_core import (
    auc_metrics,
    evaluate_selections,
    paired_comparison,
    retrieval_metrics,
)
from chemaware_truthblind_candidate_core import (
    predict_truthblind_baselines,
    predict_truthblind_policy,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_POLICY_SHA256 = "8402f32d07287a47008556bf25ad241d25c9e56e6d30fd1d68a407de1a2f2dee"
CANONICAL_REPORT_SHA256 = "64a997a4fc05f442c04e2746a5ac6e1022272bb21417b237cde279ea66aa47bf"
CANONICAL_INNER_SHA256 = "70e25e6d2459222be3597748a851f206c5121792cf920e8e745ed2caa6dcace3"
MODEL_NAMES = ("chemaware", "same_feature_direct", "nuisance_only")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument(
        "--canonical-policy-dir", type=Path,
        default=ROOT / "data/validation/chemaware_truthblind_candidate_policy/run_2338337/policy",
    )
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
    parser.add_argument(
        "--seal", type=Path,
        default=ROOT / "data/validation/chemaware_truthblind_candidate_policy/OUTER_FOLD_4_OPENED.lock",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kernel_arguments(source: dict[str, object], token_dir: Path, rule_library: Path) -> SimpleNamespace:
    return SimpleNamespace(
        token_dir=token_dir,
        rule_library=rule_library,
        top_peaks=int(source["top_peaks"]),
        kernel_dim=int(source["kernel_dim"]),
        bin_width=float(source["bin_width"]),
        grid_offsets=int(source["grid_offsets"]),
        intensity_power=float(source["intensity_power"]),
        mass_shift_da=float(source["mass_shift_da"]),
        pair_weight=float(source["pair_weight"]),
        multi_bin_widths=tuple(source["multi_bin_widths"]),
        uniform_channel_weight=float(source["uniform_channel_weight"]),
        rule_tolerance=float(source["rule_tolerance"]),
        rule_channel_weight=float(source["rule_channel_weight"]),
    )


def inference_ledger(
    queries: np.ndarray, truthblind_body: dict[str, np.ndarray], bundle: dict[str, object],
    official: np.ndarray, row_position: dict[int, int], cache: KernelCache,
    variants: tuple[str, ...], batch_size: int,
) -> dict[str, np.ndarray]:
    """Return only truth-free predictions and official candidate scores."""
    query_parts = []
    candidate_score_parts = []
    candidate_count_parts = []
    selected = {name: [] for name in MODEL_NAMES}
    selected_slot = {name: [] for name in MODEL_NAMES}
    abstained = {name: [] for name in MODEL_NAMES}
    best_utility = {name: [] for name in MODEL_NAMES}
    for start in range(0, len(queries), int(batch_size)):
        stop = min(start + int(batch_size), len(queries))
        batch_query = np.asarray(queries[start:stop], dtype=np.int64)
        scored = score_queries_truthblind(
            batch_query, truthblind_body, official, row_position, cache, variants,
        )
        primary = predict_truthblind_policy(
            bundle, scored, [scored for _ in bundle["control_rule_keys"]],
        )
        controls = predict_truthblind_baselines(
            bundle, scored, [scored for _ in bundle["control_rule_keys"]],
        )
        predictions = {"chemaware": primary, **controls}
        for index in range(len(batch_query)):
            pointer = np.asarray(scored["reference_ptr"][index], dtype=np.int64)
            molecule_score = np.maximum.reduceat(
                np.asarray(scored["global"][index], dtype=np.float32), pointer[:-1],
            )
            candidate_score_parts.append(molecule_score.astype(np.float32))
            candidate_count_parts.append(len(molecule_score))
        for name, prediction in predictions.items():
            active_selection = np.asarray(prediction["selected_candidate"], dtype=np.int16).copy()
            active_selection[np.asarray(prediction["abstained"], dtype=bool)] = -1
            selected[name].append(active_selection)
            selected_slot[name].append(
                np.asarray(prediction["selected_candidate_slot"], dtype=np.int16)
            )
            abstained[name].append(np.asarray(prediction["abstained"], dtype=bool))
            best_utility[name].append(
                np.max(np.asarray(prediction["utility"], dtype=np.float64), axis=1)
            )
        query_parts.append(batch_query)
        print(f"truth-blind inference {stop}/{len(queries)}", flush=True)
    counts = np.asarray(candidate_count_parts, dtype=np.int32)
    candidate_ptr = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    output: dict[str, np.ndarray] = {
        "query": np.concatenate(query_parts).astype(np.int64),
        "candidate_ptr": candidate_ptr,
        "official_candidate_score": np.concatenate(candidate_score_parts).astype(np.float32),
    }
    for name in MODEL_NAMES:
        output[f"{name}_selected_candidate"] = np.concatenate(selected[name])
        output[f"{name}_selected_candidate_slot"] = np.concatenate(selected_slot[name])
        output[f"{name}_abstained"] = np.concatenate(abstained[name])
        output[f"{name}_best_utility"] = np.concatenate(best_utility[name])
    return output


def labels_for_queries(manifest: Path, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(manifest, allow_pickle=False) as loaded:
        query_ptr = np.asarray(loaded["query_ptr"], dtype=np.int64)
        counts = np.diff(query_ptr)[queries]
        candidate_ptr = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        labels = np.empty(int(candidate_ptr[-1]), dtype=bool)
        for out_index, query in enumerate(map(int, queries)):
            source_left, source_right = map(int, query_ptr[query:query + 2])
            target_left, target_right = map(int, candidate_ptr[out_index:out_index + 2])
            labels[target_left:target_right] = np.asarray(
                loaded["molecule_label"][source_left:source_right], dtype=bool,
            )
        formula = np.asarray(loaded["query_formula"][queries]).astype(str)
    return candidate_ptr, labels, formula


def evaluate_ledger(
    ledger: dict[str, np.ndarray], manifest: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    ptr, labels, formula = labels_for_queries(manifest, ledger["query"])
    if not np.array_equal(ptr, ledger["candidate_ptr"]):
        raise RuntimeError("truth-free candidate layout disagrees with evaluation labels")
    selections = {
        name: np.asarray(ledger[f"{name}_selected_candidate"], dtype=np.int16)
        for name in MODEL_NAMES
    }
    baseline, ranks, scores = evaluate_selections(
        ptr, ledger["official_candidate_score"], labels, selections,
    )
    return baseline, ranks, scores, labels, formula


def exact_inner_preflight(
    ledger: dict[str, np.ndarray], canonical_inner: Path, manifest: Path,
) -> dict[str, object]:
    baseline, ranks, _scores, _labels, _formula = evaluate_ledger(ledger, manifest)
    with np.load(canonical_inner, allow_pickle=False) as expected:
        checks: dict[str, object] = {
            "query_equal": bool(np.array_equal(ledger["query"], expected["query"])),
            "baseline_rank_equal": bool(np.array_equal(baseline, expected["baseline_rank"])),
        }
        expected_fields = {
            "chemaware": {
                "rank": "correct_rank",
                "slot": "correct_selected_candidate_slot",
                "best": "best_predicted_utility",
            },
            "same_feature_direct": {
                "rank": "same_feature_direct_rank",
                "slot": "same_feature_direct_selected_candidate_slot",
                "best": "same_feature_direct_best_utility",
            },
            "nuisance_only": {
                "rank": "nuisance_only_rank",
                "slot": "nuisance_only_selected_candidate_slot",
                "best": "nuisance_only_best_utility",
            },
        }
        for name, fields in expected_fields.items():
            absent = [field for field in fields.values() if field not in expected.files]
            if absent:
                raise RuntimeError(f"canonical inner fields absent for {name}: {absent}")
            checks[f"{name}_rank_equal"] = bool(np.array_equal(
                ranks[name], expected[fields["rank"]],
            ))
            checks[f"{name}_slot_equal"] = bool(np.array_equal(
                ledger[f"{name}_selected_candidate_slot"],
                expected[fields["slot"]],
            ))
            expected_best = np.asarray(expected[fields["best"]], dtype=np.float64)
            observed_best = np.asarray(ledger[f"{name}_best_utility"], dtype=np.float64)
            checks[f"{name}_best_utility_max_abs_error"] = float(
                np.max(np.abs(observed_best - expected_best))
            )
    exact_flags = [value for key, value in checks.items() if key.endswith("_equal")]
    errors = [
        value for key, value in checks.items() if key.endswith("_max_abs_error")
    ]
    passed = bool(all(exact_flags) and max(errors, default=0.0) <= 1e-12)
    return {
        "status": "CHEMAWARE_RELEASE_INNER_REPLAY_PASS" if passed else "CHEMAWARE_RELEASE_INNER_REPLAY_FAIL",
        "passed": passed,
        "truthblind_before_evaluation": True,
        **checks,
    }


def validate_release(
    args: argparse.Namespace, report: dict[str, object], bundle: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    canonical_policy = args.canonical_policy_dir / "truthblind_policy.joblib"
    canonical_report = args.canonical_policy_dir / "report.json"
    canonical_inner = args.canonical_policy_dir / "inner_policy.npz"
    expected = {
        canonical_policy: CANONICAL_POLICY_SHA256,
        canonical_report: CANONICAL_REPORT_SHA256,
        canonical_inner: CANONICAL_INNER_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"canonical run 2338337 artifact drifted: {path}")
    provenance = report["provenance"]
    observed_provenance = {
        "manifest_sha256": sha256(args.manifest),
        "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
        "rule_library_sha256": sha256(args.rule_library),
    }
    if observed_provenance != provenance:
        raise RuntimeError("release candidate input provenance drifted")
    if int(bundle.get("outer_formula_role", -1)) != 4 or "baselines" not in bundle:
        raise RuntimeError("release candidate is not outer-ready")
    canonical_metadata = json.loads(
        (args.canonical_policy_dir / "truthblind_policy.json").read_text(encoding="utf-8")
    )
    release_metadata = json.loads(
        (args.policy_dir / "truthblind_policy.json").read_text(encoding="utf-8")
    )
    frozen_keys = (
        "schema", "feature_names", "actions", "global_action", "dose", "threshold",
        "risk_penalty", "rule_key", "control_rule_keys", "contrast_representation",
        "training_formula_roles", "selection_formula_role", "development_formula_role",
        "outer_formula_role", "channel_contract",
    )
    if any(release_metadata.get(key) != canonical_metadata.get(key) for key in frozen_keys):
        raise RuntimeError("release candidate changed a frozen primary-policy field")
    return observed_provenance, {
        "policy": CANONICAL_POLICY_SHA256,
        "report": CANONICAL_REPORT_SHA256,
        "inner": CANONICAL_INNER_SHA256,
    }


def create_outer_seal(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    args = arguments()
    if args.batch_size <= 0 or args.bootstrap_draws <= 0:
        raise ValueError("batch size and bootstrap draws must be positive")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    required = (
        args.policy_dir / "truthblind_policy.joblib",
        args.policy_dir / "truthblind_policy.json",
        args.policy_dir / "report.json",
        args.policy_dir / "inner_policy.npz",
        args.manifest,
        args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy",
        args.rule_library,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    policy_path = args.policy_dir / "truthblind_policy.joblib"
    report = json.loads((args.policy_dir / "report.json").read_text(encoding="utf-8"))
    bundle = joblib.load(policy_path)
    observed_provenance, canonical_hashes = validate_release(args, report, bundle)
    replay_args = report["replay_contract"]["arguments"]

    with np.load(args.manifest, allow_pickle=False) as loaded:
        split_formula = np.asarray(loaded["query_formula"])
        truthblind_body = {
            key: np.asarray(loaded[key])
            for key in ("query_ptr", "molecule_ptr", "pair_candidate_row", "query_row")
        }
    fold = stable_formula_folds(
        split_formula, int(replay_args["folds"]), int(replay_args["fold_seed"]),
    )
    outer_queries = np.flatnonzero(fold == int(bundle["outer_formula_role"])).astype(np.int64)
    if len(outer_queries) != int(report["data"]["outer_queries_untouched"]):
        raise RuntimeError("sealed outer query count drifted")
    # Formula is used only to instantiate the pre-registered split.  Remove it
    # before either the inner replay or outer inference is invoked.
    del split_formula, fold

    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    variants = tuple(dict.fromkeys((
        "mass", str(bundle["rule_key"]), *map(str, bundle["control_rule_keys"]),
    )))
    cache = KernelCache(
        kernel_arguments(replay_args, args.token_dir, args.rule_library),
        row_position, variants=variants,
    )

    canonical_inner = args.canonical_policy_dir / "inner_policy.npz"
    with np.load(canonical_inner, allow_pickle=False) as expected:
        inner_queries = np.asarray(expected["query"], dtype=np.int64)
    inner_ledger = inference_ledger(
        inner_queries, truthblind_body, bundle, official, row_position, cache, variants,
        batch_size=len(inner_queries),
    )
    inner_preflight = exact_inner_preflight(inner_ledger, canonical_inner, args.manifest)
    print(json.dumps(inner_preflight, indent=2), flush=True)
    if not inner_preflight["passed"]:
        raise RuntimeError("release candidate failed exact inner replay; outer remains sealed")

    seal_payload = {
        "status": "CHEMAWARE_OUTER_FOLD_4_OPENED",
        "outer_role": int(bundle["outer_formula_role"]),
        "queries": int(len(outer_queries)),
        "policy_sha256": sha256(policy_path),
        "canonical_run": 2338337,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "output": str(args.output),
        "no_outer_tuning": True,
    }
    create_outer_seal(args.seal, seal_payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_outer_once_", dir=args.output.parent))
    try:
        # This call cannot access labels, formula identities, correctness, or ranks.
        outer_ledger = inference_ledger(
            outer_queries, truthblind_body, bundle, official, row_position, cache, variants,
            batch_size=args.batch_size,
        )
        prediction_path = temporary / "predictions_truthblind.npz"
        np.savez_compressed(prediction_path, **outer_ledger)
        prediction_hash = sha256(prediction_path)

        # Truth is opened only after the immutable prediction artifact exists.
        baseline_rank, ranks, proposed_scores, labels, formula = evaluate_ledger(
            outer_ledger, args.manifest,
        )
        official_auc = auc_metrics(
            outer_ledger["candidate_ptr"], outer_ledger["official_candidate_score"], labels,
        )
        models: dict[str, object] = {}
        for name in MODEL_NAMES:
            models[name] = {
                "selected": int(np.sum(~outer_ledger[f"{name}_abstained"])),
                "selected_formulas": int(len(np.unique(
                    formula[~outer_ledger[f"{name}_abstained"]]
                ))),
                "retrieval": retrieval_metrics(baseline_rank, ranks[name]),
                "auc": auc_metrics(outer_ledger["candidate_ptr"], proposed_scores[name], labels),
            }
        comparisons = {
            "chemaware_minus_official": paired_comparison(
                ranks["chemaware"], baseline_rank, formula,
                draws=args.bootstrap_draws, seed=args.seed + 1,
            ),
            "same_feature_direct_minus_official": paired_comparison(
                ranks["same_feature_direct"], baseline_rank, formula,
                draws=args.bootstrap_draws, seed=args.seed + 2,
            ),
            "nuisance_only_minus_official": paired_comparison(
                ranks["nuisance_only"], baseline_rank, formula,
                draws=args.bootstrap_draws, seed=args.seed + 3,
            ),
            "chemaware_minus_same_feature_direct": paired_comparison(
                ranks["chemaware"], ranks["same_feature_direct"], formula,
                draws=args.bootstrap_draws, seed=args.seed + 4,
            ),
            "chemaware_minus_nuisance_only": paired_comparison(
                ranks["chemaware"], ranks["nuisance_only"], formula,
                draws=args.bootstrap_draws, seed=args.seed + 5,
            ),
        }
        primary = models["chemaware"]["retrieval"]
        total_ci = comparisons["chemaware_minus_official"][
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ]
        chemical_ci = comparisons["chemaware_minus_nuisance_only"][
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ]
        direct_ci = comparisons["chemaware_minus_same_feature_direct"][
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ]
        gates = {
            "total_recall1_gain_at_least_3pp": float(primary["delta_recall1"]) >= 0.03,
            "total_formula_ci_strictly_positive": float(total_ci[0]) > 0.0,
            "corrected_strictly_more_than_twice_introduced": (
                int(primary["corrected_at_1"]) > 2 * int(primary["introduced_at_1"])
            ),
            "chemistry_increment_over_nuisance_ci_strictly_positive": float(chemical_ci[0]) > 0.0,
            "residualization_increment_over_direct_ci_strictly_positive": float(direct_ci[0]) > 0.0,
            "no_recall_drop_at_registered_k": all(
                float(primary[f"delta_recall{k}"]) >= 0.0 for k in (1, 3, 5, 10, 20, 50)
            ),
        }
        evaluation_path = temporary / "evaluation_with_truth.npz"
        np.savez_compressed(
            evaluation_path,
            query=outer_ledger["query"], formula=formula,
            baseline_rank=baseline_rank,
            **{f"{name}_rank": ranks[name] for name in MODEL_NAMES},
        )
        final_report = {
            "status": "CHEMAWARE_TRUTHBLIND_OUTER_ONCE_COMPLETE",
            "scientific_pass": bool(all(gates.values())),
            "candidate_conditioned": True,
            "shared_embedding_result": False,
            "outer_role": int(bundle["outer_formula_role"]),
            "queries": int(len(outer_queries)),
            "formulas": int(len(np.unique(formula))),
            "inference_truth_fields": [],
            "predictions_frozen_before_truth_open": True,
            "outer_tuning": False,
            "inner_release_preflight": inner_preflight,
            "official_auc": official_auc,
            "models": models,
            "paired_formula_cluster_comparisons": comparisons,
            "registered_gates": gates,
            "provenance": {
                **observed_provenance,
                "release_policy_sha256": sha256(policy_path),
                "canonical_run_2338337": canonical_hashes,
                "truthblind_predictions_sha256": prediction_hash,
                "evaluation_sha256": sha256(evaluation_path),
                "outer_seal": str(args.seal),
            },
        }
        (temporary / "report.json").write_text(
            json.dumps(final_report, indent=2), encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(final_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
