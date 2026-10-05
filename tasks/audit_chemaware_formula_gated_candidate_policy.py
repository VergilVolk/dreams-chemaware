"""Audit formula-gated, polarity-aware ChemAware candidate intervention."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sklearn

from audit_chemaware_candidate_evidence_policy import (
    FEATURE_NAMES,
    array_sha256,
    build_candidate_table,
    evaluate_arm,
    ranks_at_threshold,
    train_arm,
)
from audit_chemaware_counterfactual_rule_kernel import retrieval, sha256
from audit_chemaware_counterfactual_rule_kernel_natural import paired_rank_comparison
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries
from chemaware_formula_rule_core import (
    append_formula_rule_scores,
    inverse_document_frequency,
    positive_mode_mask,
    registry,
    rows_for_queries,
    shuffled_formula_assignment,
)
from chemaware_iceberg_direct_core import stable_formula_folds
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--rule-library", type=Path, default=ROOT / "dreams/models/chem_aware/chem_rules_data.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_formula_gated_candidate_policy_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--sampling-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--train-identities", type=int, default=4096)
    parser.add_argument("--validation-identities", type=int, default=2048)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--beta", type=float, nargs="+", default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60))
    parser.add_argument("--global-mass-beta", type=float, default=0.1)
    parser.add_argument("--global-rule-beta", type=float, default=0.2)
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--min-selected-formulas", type=int, default=50)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--multi-bin-widths", type=float, nargs="+", default=(0.01, 0.02, 0.05))
    parser.add_argument("--uniform-channel-weight", type=float, default=1.0)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--rule-channel-weight", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    actions = [
        (float(mass), float(rule)) for mass in args.beta for rule in args.beta
        if float(mass) > 0.0 or float(rule) > 0.0
    ]
    global_action = actions.index((args.global_mass_beta, args.global_rule_beta))
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    if not np.all(np.char.endswith(body["query_adduct"].astype(str), "+")):
        raise RuntimeError("positive-mode audit received a non-positive query")
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    pools = [np.flatnonzero((fold == 0) | (fold == 1)), np.flatnonzero(fold == 2), np.flatnonzero(fold == 3)]
    queries = [
        identity_balanced_queries(pools[0], body["query_ik14"], np.random.default_rng(args.sampling_seed + 1), args.train_identities),
        identity_balanced_queries(pools[1], body["query_ik14"], np.random.default_rng(args.sampling_seed + 2), args.validation_identities),
        identity_balanced_queries(pools[2], body["query_ik14"], np.random.default_rng(args.sampling_seed + 19), args.max_inner_identities),
    ]
    formulas = [body["query_formula"][query].astype(str) for query in queries]
    if any(set(formulas[i]) & set(formulas[j]) for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("formula split leaked")
    kernel_args = SimpleNamespace(**vars(args))
    cache = KernelCache(kernel_args, row_position, variants=("mass", "rule_response"))
    scored = []
    for name, query in zip(("train", "validation", "inner"), queries, strict=True):
        scored.append(score_queries(query, body, official, row_position, cache, ("mass", "rule_response")))
        print(f"completed {name} raw score geometry rows={len(cache.cache)}", flush=True)

    payload = json.loads(args.rule_library.read_text(encoding="utf-8"))
    channels = registry(payload.get("channels", payload.get("rules")))
    if len(channels) != len(cache.nl_rules) + len(cache.cf_rules):
        raise RuntimeError("formula rule registry does not align with cached response channels")
    training_rows = rows_for_queries(queries[0], body)
    training_response = np.stack([
        cache.get(int(row))["rule_response"] for row in training_rows
    ]).astype(np.float32)
    allowed = positive_mode_mask(channels)
    idf = inverse_document_frequency(training_response, allowed)
    shuffled = shuffled_formula_assignment(channels, args.seed + 1701)
    mode_only = [None] * len(channels)
    for split, (name, query, score) in enumerate(zip(("train", "validation", "inner"), queries, scored, strict=True)):
        append_formula_rule_scores(score, query, body, cache, idf, channels, None, "rule_formula_idf")
        append_formula_rule_scores(score, query, body, cache, idf, channels, shuffled, "rule_formula_shuffled_idf")
        append_formula_rule_scores(score, query, body, cache, idf, channels, mode_only, "rule_mode_idf")
        print(f"completed {name} formula rule scores", flush=True)

    keys = {
        "formula_idf": "rule_formula_idf",
        "formula_shuffled": "rule_formula_shuffled_idf",
        "mode_idf": "rule_mode_idf",
        "raw_rule": "rule_response",
    }
    tables: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in keys}
    for split, score in enumerate(scored):
        for name, key in keys.items():
            tables[name].append(build_candidate_table(score, actions, global_action, key))
        print(f"completed candidate tables split={split}", flush=True)

    feature_index = np.arange(len(FEATURE_NAMES), dtype=np.int64)
    arms = {
        name: train_arm(table[0], table[1], formulas[0], formulas[1], feature_index, args, permute=False)
        for name, table in tables.items()
    }
    arms["label_permuted"] = train_arm(
        tables["formula_idf"][0], tables["formula_idf"][1], formulas[0], formulas[1],
        feature_index, args, permute=True,
    )
    evaluated = {
        name: evaluate_arm(arm, tables[name][2], formulas[2], args)
        for name, arm in arms.items() if name != "label_permuted"
    }
    evaluated["label_permuted"] = evaluate_arm(
        arms["label_permuted"], tables["formula_idf"][2], formulas[2], args,
    )
    primary = evaluated["formula_idf"]
    validation_rank, validation_selected, validation_best = ranks_at_threshold(
        tables["formula_idf"][1], arms["formula_idf"]["validation_utility"],
        arms["formula_idf"]["threshold"],
    )

    def selected_candidate_index(table: dict[str, np.ndarray], selected: np.ndarray) -> np.ndarray:
        output = np.full(len(selected), -1, dtype=np.int16)
        active = selected >= 0
        output[active] = table["proposed_candidate"][
            np.arange(len(selected))[active], selected[active]
        ]
        return output

    validation_candidate = selected_candidate_index(
        tables["formula_idf"][1], validation_selected,
    )
    inner_candidate = selected_candidate_index(
        tables["formula_idf"][2], primary["selected"],
    )
    comparisons = {
        f"formula_idf_minus_{name}": paired_rank_comparison(
            primary["rank"], result["rank"], formulas[2], draws=args.bootstrap_draws,
            seed=args.seed + 700 + index,
        )
        for index, (name, result) in enumerate(evaluated.items()) if name != "formula_idf"
    }
    valid_rank = np.where(tables["formula_idf"][2]["valid"], tables["formula_idf"][2]["rank"], 32767)
    baseline_rank = tables["formula_idf"][2]["baseline_rank"]
    oracle_rank = np.minimum(baseline_rank, np.min(valid_rank, axis=1))
    gate_density = []
    shuffled_density = []
    for formula in formulas[2]:
        from chemaware_formula_rule_core import formula_gate
        gate_density.append(float(np.mean(formula_gate(channels, str(formula)))))
        shuffled_density.append(float(np.mean(formula_gate(channels, str(formula), shuffled))))
    report = {
        "status": "CHEMAWARE_FORMULA_GATED_CANDIDATE_POLICY_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "candidate_conditioned": True,
        "shared_embedding_result": False,
        "scope": "folds 0-1 fit and IDF; fold 2 threshold; used inner fold 3; fold 4 sealed",
        "claim_limit": "Development candidate policy, not shared embedding or external confirmation.",
        "rule_registry": {
            "response_channels": len(channels),
            "positive_mode_channels": int(np.sum(allowed)),
            "excluded_negative_mode_channels": int(np.sum(~allowed)),
            "formula_annotated_channels": int(sum(channel.formula is not None for channel in channels)),
            "training_rows_for_idf": int(len(training_rows)),
            "inner_correct_gate_fraction_quantiles": np.quantile(gate_density, [0, .25, .5, .75, 1]).tolist(),
            "inner_shuffled_gate_fraction_quantiles": np.quantile(shuffled_density, [0, .25, .5, .75, 1]).tolist(),
            "idf_fit_uses_truth": False,
        },
        "method": {
            "formula_rule": "positive-mode filter, elemental subformula gate, train-row IDF, per-spectrum L2 normalization",
            "shuffled_control": "rule formula assignments permuted within NL/CF while preserving formula inventory and mode exclusions",
            "candidate_utility": "P(correct official error)-2*P(destroy official correct)",
            "feature_names": FEATURE_NAMES,
            "explicit_no_op": True,
            "candidate_formula_feature": "used only to gate physically impossible rule channels; raw formula string/counts are not classifier features",
        },
        "data": {
            "train_queries": int(len(queries[0])), "validation_queries": int(len(queries[1])),
            "inner_queries": int(len(queries[2])), "outer_queries_untouched": int(np.sum(fold == 4)),
            "formula_overlap": 0,
        },
        "validation": {name: arm["threshold_selection"] for name, arm in arms.items()},
        "held_inner": {name: result["metric"] for name, result in evaluated.items()},
        "held_inner_absolute_formula_bootstrap_ci95": {name: result["ci"] for name, result in evaluated.items()},
        "held_inner_candidate_oracle": {
            **retrieval(baseline_rank, oracle_rank),
            "claim_limit": "best candidate chosen after observing truth; headroom only",
        },
        "paired_inner": comparisons,
        "gates": {
            "absolute_ci_positive": primary["ci"][0] > 0,
            "corrected_exceeds_twice_introduced": primary["metric"]["corrected_at_1"] > 2 * primary["metric"]["introduced_at_1"],
            "beats_formula_shuffle_ci": comparisons["formula_idf_minus_formula_shuffled"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "beats_mode_idf_ci": comparisons["formula_idf_minus_mode_idf"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "beats_raw_rule_ci": comparisons["formula_idf_minus_raw_rule"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "beats_label_permutation_ci": comparisons["formula_idf_minus_label_permuted"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "outer_fold_untouched": True,
        },
        "replay_contract": {
            "arguments": {key: (str(value.resolve()) if isinstance(value, Path) else value) for key, value in vars(args).items()},
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
            "query_sha256": {name: array_sha256(query) for name, query in zip(("train", "validation", "inner"), queries, strict=True)},
            "idf_sha256": array_sha256(idf),
            "formula_candidate_rank_sha256": {name: array_sha256(table["rank"]) for name, table in zip(("train", "validation", "inner"), tables["formula_idf"], strict=True)},
        },
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
            "rule_library_sha256": sha256(args.rule_library),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_formula_policy_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_policy.npz", query=queries[2], formula=formulas[2], baseline_rank=baseline_rank,
            formula_idf_rank=primary["rank"], formula_shuffled_rank=evaluated["formula_shuffled"]["rank"],
            mode_idf_rank=evaluated["mode_idf"]["rank"], raw_rule_rank=evaluated["raw_rule"]["rank"],
            label_permuted_rank=evaluated["label_permuted"]["rank"], oracle_rank=oracle_rank,
            selected_candidate_slot=primary["selected"], best_predicted_utility=primary["best"],
            selected_candidate_index=inner_candidate,
            baseline_candidate_index=tables["formula_idf"][2]["baseline_candidate"],
        )
        np.savez_compressed(
            temporary / "validation_policy.npz", query=queries[1], formula=formulas[1],
            baseline_rank=tables["formula_idf"][1]["baseline_rank"], policy_rank=validation_rank,
            selected_candidate_slot=validation_selected,
            selected_candidate_index=validation_candidate,
            baseline_candidate_index=tables["formula_idf"][1]["baseline_candidate"],
            best_predicted_utility=validation_best,
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "rule_registry": report["rule_registry"],
        "validation": report["validation"], "held_inner": report["held_inner"],
        "paired_inner": comparisons, "gates": report["gates"],
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
