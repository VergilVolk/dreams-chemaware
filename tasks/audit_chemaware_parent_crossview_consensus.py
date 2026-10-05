"""Test the validated cross-view teacher on top of the frozen shared parent map."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval
from audit_chemaware_counterfactual_rule_kernel_natural import paired_rank_comparison
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries, strict_rank
from audit_chemaware_whitened_centered_rule_kernel import FourCenterCache, TransformCache
from chemaware_crossview_consensus_core import crossview_policy
from chemaware_formula_rule_core import rows_for_queries
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_shrinkage_whitening_core import fit_shrinkage_whitener
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("correct", "zero_contrast", "reversed_contrast", "alignment_permuted")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--rule-library", type=Path, default=ROOT / "dreams/models/chem_aware/chem_rules_data.json")
    parser.add_argument("--parent-report", type=Path, default=ROOT / "data/validation/chemaware_whitened_centered_rule_kernel_v1/report.json")
    parser.add_argument("--repeat-ledger", type=Path, default=ROOT / "data/validation/chemaware_orthogonal_teacher_repeat_consistency_v3/repeat_ledger.npz")
    parser.add_argument("--consensus-report", type=Path, default=ROOT / "data/validation/chemaware_crossview_consensus_teacher_v2/report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_parent_crossview_consensus_v1")
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--split-seed", type=int, default=20260913)
    parser.add_argument("--final-fit-identities", type=int, default=8192)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def promote_candidate_rank(scores: np.ndarray, labels: np.ndarray, candidate: int | None) -> int:
    values = np.asarray(scores, dtype=np.float64).copy()
    if candidate is not None:
        if not 0 <= candidate < len(values):
            raise ValueError("candidate index outside molecule score array")
        values[candidate] = np.nextafter(float(np.max(values)), np.inf)
    return strict_rank(values, np.asarray(labels, dtype=bool))


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    parent_report = json.loads(args.parent_report.read_text(encoding="utf-8"))
    consensus_report = json.loads(args.consensus_report.read_text(encoding="utf-8"))
    parent_setting = parent_report["fold2_selection"]["whitened"]["true"]
    consensus_setting = consensus_report["selection"]
    if consensus_report.get("status") != "CHEMAWARE_CROSSVIEW_CONSENSUS_TEACHER_PASS":
        raise RuntimeError("cross-view teacher did not pass its frozen confirmation")
    if parent_setting["variant"] != "whitened_true_s0p5":
        raise RuntimeError("unexpected frozen shared parent variant")
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.repeat_ledger, allow_pickle=False) as loaded:
        ledger = {key: np.asarray(loaded[key]) for key in loaded.files}
    panel = ledger["query"].astype(np.int64)
    formula = ledger["formula"].astype(str); identity = ledger["identity"].astype(str)
    if not np.array_equal(formula, body["query_formula"][panel].astype(str)):
        raise RuntimeError("repeat ledger and manifest are not aligned")

    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], 5, args.fold_seed)
    if not np.all(fold[panel] == 3):
        raise RuntimeError("repeat ledger is not confined to inner fold 3")
    final_fit = identity_balanced_queries(
        np.flatnonzero(np.isin(fold, (0, 1, 2))), body["query_ik14"],
        np.random.default_rng(args.seed + 23), args.final_fit_identities,
    )
    if set(body["query_formula"][final_fit].astype(str)) & set(formula):
        raise RuntimeError("parent fit and panel formulas overlap")
    kernel_args = SimpleNamespace(
        token_dir=args.token_dir, rule_library=args.rule_library,
        top_peaks=args.top_peaks, kernel_dim=args.kernel_dim,
        bin_width=args.bin_width, grid_offsets=args.grid_offsets,
        intensity_power=args.intensity_power, mass_shift_da=args.mass_shift_da,
        pair_weight=0.25, multi_bin_widths=(0.01, 0.02, 0.05),
        uniform_channel_weight=1.0, rule_tolerance=args.rule_tolerance,
        rule_channel_weight=1.0,
    )
    base = KernelCache(
        kernel_args, row_position,
        variants=(
            "mass", "rule_response", "rule_response_local_background_a",
            "rule_response_local_background_b", "rule_response_local_background_c",
        ),
    )
    centers = FourCenterCache(base)
    fit_rows = rows_for_queries(final_fit, body)
    fit_feature = np.stack([
        np.asarray(centers.get(int(row))["true"], dtype=np.float32) for row in fit_rows
    ])
    mean, transform, transform_report = fit_shrinkage_whitener(fit_feature, 0.5)
    del fit_feature
    gc.collect()
    variant = str(parent_setting["variant"])
    cache = TransformCache(centers, {variant: (mean, transform)}, {variant: "true"})
    scored = score_queries(
        panel, body, official, row_position, cache, ("mass", variant),
    )

    baseline = ledger["baseline_rank"].astype(np.int64)
    parent_rank = np.empty(len(panel), dtype=np.int16)
    parent_scores: list[np.ndarray] = []
    molecule_labels: list[np.ndarray] = []
    for index in range(len(panel)):
        pointer = np.asarray(scored["reference_ptr"][index], dtype=np.int64)
        pair = (
            np.asarray(scored["global"][index], dtype=np.float64)
            + float(parent_setting["mass_beta"]) * np.asarray(scored["mass"][index], dtype=np.float64)
            + float(parent_setting["rule_beta"]) * np.asarray(scored[variant][index], dtype=np.float64)
        )
        molecule = np.maximum.reduceat(pair, pointer[:-1])
        labels = np.asarray(scored["labels"][index], dtype=bool)
        parent_scores.append(molecule); molecule_labels.append(labels)
        parent_rank[index] = strict_rank(molecule, labels)
    if not np.array_equal(np.asarray(scored["old_rank"], dtype=np.int64), baseline):
        raise RuntimeError("official baseline did not replay repeat ledger")

    arm_selected = {}; consensus_rank = {}
    for arm in ARMS:
        rank, selected, _score, _support = crossview_policy(
            baseline, ledger["candidate_identity"], ledger["candidate_valid"],
            ledger["proposal_rank"], ledger[f"{arm}_candidate_utility"], identity,
            threshold=float(consensus_setting["threshold"]),
            aggregation=str(consensus_setting["aggregation"]),
            min_context=int(consensus_setting["min_context"]),
        )
        consensus_rank[arm] = rank; arm_selected[arm] = selected

    hybrid_rank = {arm: parent_rank.copy() for arm in ARMS}
    for arm in ARMS:
        for index, slot in enumerate(arm_selected[arm]):
            if slot < 0:
                continue
            candidate = str(ledger["candidate_identity"][index, int(slot)])
            query = int(panel[index])
            left, right = map(int, body["query_ptr"][query : query + 2])
            local_identity = body["molecule_ik14"][left:right].astype(str)
            matches = np.flatnonzero(local_identity == candidate)
            if len(matches) != 1:
                raise RuntimeError("consensus candidate does not map uniquely into parent candidate set")
            hybrid_rank[arm][index] = promote_candidate_rank(
                parent_scores[index], molecule_labels[index], int(matches[0]),
            )

    role = stable_formula_folds(formula, 3, args.split_seed)
    confirmation = role == 2
    official_confirmation = baseline[confirmation]
    parent_confirmation = parent_rank[confirmation]
    held = {
        "official": retrieval(official_confirmation, official_confirmation),
        "parent": retrieval(official_confirmation, parent_confirmation),
        **{
            f"parent_plus_{arm}": retrieval(official_confirmation, rank[confirmation])
            for arm, rank in hybrid_rank.items()
        },
    }
    incremental = {
        arm: {
            "retrieval": retrieval(parent_confirmation, hybrid_rank[arm][confirmation]),
            "formula_cluster_bootstrap_ci95": bootstrap(
                formula[confirmation], parent_confirmation, hybrid_rank[arm][confirmation],
                args.bootstrap_draws, args.seed + 100 + index,
            ),
        }
        for index, arm in enumerate(ARMS)
    }
    paired = {
        f"correct_minus_{arm}": paired_rank_comparison(
            hybrid_rank["correct"][confirmation], hybrid_rank[arm][confirmation],
            formula[confirmation], draws=args.bootstrap_draws,
            seed=args.seed + 200 + index,
        )
        for index, arm in enumerate(ARMS[1:])
    }
    oracle = np.where(
        (parent_confirmation == 1) | (consensus_rank["correct"][confirmation] == 1),
        1,
        np.minimum(parent_confirmation, consensus_rank["correct"][confirmation]),
    )
    oracle[(parent_confirmation == 1) | (consensus_rank["correct"][confirmation] == 1)] = 1
    gates = {
        "parent_absolute_positive": held["parent"]["delta_recall1"] > 0,
        "hybrid_increment_parent_ci_positive": incremental["correct"]["formula_cluster_bootstrap_ci95"][0] > 0,
        "hybrid_increment_corrected_exceeds_twice_introduced": (
            incremental["correct"]["retrieval"]["corrected_at_1"]
            > 2 * incremental["correct"]["retrieval"]["introduced_at_1"]
        ),
        **{
            f"hybrid_beats_{arm}_ci": paired[f"correct_minus_{arm}"][
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0
            for arm in ARMS[1:]
        },
        "outer_fold_untouched": True,
    }
    report = {
        "status": (
            "CHEMAWARE_PARENT_CROSSVIEW_CONSENSUS_PASS"
            if all(gates.values()) else "CHEMAWARE_PARENT_CROSSVIEW_CONSENSUS_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": False,
        "candidate_conditioned": True,
        "shared_embedding_result": False,
        "scope": "Frozen shared parent plus validated cross-view teacher on formula-held inner-fold confirmation; outer fold 4 untouched.",
        "claim_limit": "Hybrid and oracle quantify transfer headroom; neither is a fine-tuned shared encoder.",
        "method": {
            "parent": parent_report["method"]["map"],
            "consensus": consensus_report["method"],
            "hybrid": "consensus-selected candidate is promoted in the independently fitted parent score list",
            "same_consensus_setting_all_arms": True,
        },
        "data": {
            "parent_fit_queries": int(len(final_fit)), "parent_fit_spectrum_rows": int(len(fit_rows)),
            "panel_queries": int(len(panel)), "confirmation_queries": int(np.sum(confirmation)),
            "confirmation_formulas": int(len(np.unique(formula[confirmation]))),
            "outer_queries_untouched": int(np.sum(fold == 4)),
        },
        "parent_transform": transform_report,
        "held_absolute_from_official": held,
        "increment_over_parent": incremental,
        "paired_hybrid_controls": paired,
        "oracle_union_parent_or_consensus": retrieval(official_confirmation, oracle),
        "gates": gates,
        "provenance": {
            "manifest_sha256": sha256(args.manifest), "parent_report_sha256": sha256(args.parent_report),
            "repeat_ledger_sha256": sha256(args.repeat_ledger),
            "consensus_report_sha256": sha256(args.consensus_report),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
        },
    }
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_parent_consensus_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "confirmation_ranks.npz", query=panel[confirmation],
            formula=formula[confirmation], identity=identity[confirmation],
            official_rank=official_confirmation, parent_rank=parent_confirmation,
            **{f"parent_plus_{arm}_rank": rank[confirmation] for arm, rank in hybrid_rank.items()},
            oracle_union_rank=oracle,
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "held": held, "increment": incremental,
        "paired": paired, "oracle": report["oracle_union_parent_or_consensus"],
        "gates": gates, "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
