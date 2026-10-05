"""Audit a pair-first local peak/neutral-loss teacher for shared embeddings.

The teacher is deliberately non-separable: it compares local fragment masses
and neutral losses before candidate-molecule pooling.  It is allowed only as a
training-time teacher for a clean-spectrum shared embedding.  Fusion weight is
selected on natural-prevalence training formulas; the locked inner fold is
used once, and the outer fold remains untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_chemaware_peak_late_interaction import strict_rank, summary  # noqa: E402
from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
from train_chemaware_full_candidate_alignment import (  # noqa: E402
    formula_bootstrap, identity_balanced_queries, official_outcomes,
)


VARIANTS = (
    "direct_mz", "neutral_loss", "direct_or_loss",
    "mass_shifted", "intensity_permuted",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--rule-rank-file", type=Path, default=ROOT / "data/validation/chemaware_rule_mass_pairfirst_full_inner_v1/inner_per_query.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--discovery-natural-identities", type=int, default=2048)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--abs-tolerance", type=float, default=0.01)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--fusion-grid", type=float, nargs="+", default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60))
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


class PeakMassCache:
    def __init__(self, args: argparse.Namespace, row_position: dict[int, int]):
        if args.top_peaks < 1 or args.intensity_power < 0:
            raise ValueError("invalid peak-overlap configuration")
        self.top = args.top_peaks
        self.power = args.intensity_power
        self.row_position = row_position
        self.mz = np.load(args.token_dir / "mz_f32.npy", mmap_mode="r")
        self.intensity = np.load(args.token_dir / "intensity_f32.npy", mmap_mode="r")
        self.valid = np.load(args.token_dir / "valid.npy", mmap_mode="r")
        self.precursor = np.load(args.token_dir / "precursor_mz_f32.npy", mmap_mode="r")
        self.cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}

    def get(self, row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        row = int(row)
        if row in self.cache:
            return self.cache[row]
        pos = self.row_position[row]
        available = np.flatnonzero(self.valid[pos])
        order = available[
            np.argsort(-np.asarray(self.intensity[pos, available]), kind="stable")[:self.top]
        ]
        mz = np.asarray(self.mz[pos, order], dtype=np.float32)
        raw = np.maximum(np.asarray(self.intensity[pos, order], dtype=np.float32), 0)
        weight = np.power(raw, self.power).astype(np.float32)
        weight /= max(float(np.sum(weight)), 1e-8)
        permuted = weight.copy()
        if len(permuted) > 1:
            permuted = np.roll(permuted, 1 + row % (len(permuted) - 1))
        result = (mz, weight, permuted, float(self.precursor[pos]))
        self.cache[row] = result
        return result


def tolerance(left: np.ndarray, right: np.ndarray, ppm: float,
              absolute: float) -> np.ndarray:
    return np.maximum(
        absolute,
        ppm * np.maximum(np.abs(left), np.abs(right)) / 1e6,
    )


def weighted_bidirectional_coverage(mask: np.ndarray, query_weight: np.ndarray,
                                    reference_weight: np.ndarray) -> np.ndarray:
    if mask.ndim != 3:
        raise ValueError("overlap mask must be reference x query-peak x reference-peak")
    query_seen = np.any(mask, axis=2)
    reference_seen = np.any(mask, axis=1)
    return 0.5 * (
        np.sum(query_seen * query_weight[None, :], axis=1)
        + np.sum(reference_seen * reference_weight, axis=1)
    )


def reference_scores(cache: PeakMassCache, query_row: int,
                     reference_rows: np.ndarray,
                     args: argparse.Namespace) -> dict[str, np.ndarray]:
    qmz, qw, qwp, qprecursor = cache.get(query_row)
    refs = [cache.get(int(row)) for row in reference_rows]
    output = {variant: np.zeros(len(refs), dtype=np.float32) for variant in VARIANTS}
    if not len(qmz):
        return output
    qloss = qprecursor - qmz
    for index, (rmz, rw, rwp, rprecursor) in enumerate(refs):
        if not len(rmz):
            continue
        direct_delta = np.abs(qmz[:, None] - rmz[None, :])
        direct = direct_delta <= tolerance(
            qmz[:, None], rmz[None, :], args.ppm, args.abs_tolerance,
        )
        rloss = rprecursor - rmz
        loss_delta = np.abs(qloss[:, None] - rloss[None, :])
        neutral = loss_delta <= tolerance(
            qloss[:, None], rloss[None, :], args.ppm, args.abs_tolerance,
        )
        shifted_mz = rmz + args.mass_shift_da
        shifted_direct = np.abs(qmz[:, None] - shifted_mz[None, :]) <= tolerance(
            qmz[:, None], shifted_mz[None, :], args.ppm, args.abs_tolerance,
        )
        shifted_loss = rprecursor - shifted_mz
        shifted_neutral = np.abs(qloss[:, None] - shifted_loss[None, :]) <= tolerance(
            qloss[:, None], shifted_loss[None, :], args.ppm, args.abs_tolerance,
        )
        masks = {
            "direct_mz": direct,
            "neutral_loss": neutral,
            "direct_or_loss": direct | neutral,
            "mass_shifted": shifted_direct | shifted_neutral,
        }
        for variant, mask in masks.items():
            output[variant][index] = weighted_bidirectional_coverage(
                mask[None, :, :], qw, rw[None, :],
            )[0]
        output["intensity_permuted"][index] = weighted_bidirectional_coverage(
            (direct | neutral)[None, :, :], qwp, rwp[None, :],
        )[0]
    return output


def score_queries(queries: np.ndarray, body: dict[str, np.ndarray],
                  official: np.ndarray, row_position: dict[int, int],
                  cache: PeakMassCache, args: argparse.Namespace) -> dict:
    count = len(queries)
    output: dict[str, np.ndarray] = {
        "query": np.asarray(queries, dtype=np.int64),
        "formula": body["query_formula"][queries],
        "old_rank": np.empty(count, dtype=np.int16),
        "labels": np.empty(count, dtype=object),
        "global": np.empty(count, dtype=object),
        "reference_ptr": np.empty(count, dtype=object),
    }
    for variant in VARIANTS:
        output[variant] = np.empty(count, dtype=object)
    for out_index, query in enumerate(map(int, queries)):
        left, right = map(int, body["query_ptr"][query:query + 2])
        ref_left = int(body["molecule_ptr"][left])
        ref_right = int(body["molecule_ptr"][right])
        reference_rows = body["pair_candidate_row"][ref_left:ref_right]
        labels = body["molecule_label"][left:right].astype(bool)
        qpos = row_position[int(body["query_row"][query])]
        rpos = np.asarray([row_position[int(row)] for row in reference_rows])
        global_pair = np.asarray(official[rpos] @ official[qpos], dtype=np.float32)
        ptr = body["molecule_ptr"][left:right + 1].astype(np.int64) - ref_left
        molecule_global = np.asarray([
            np.max(global_pair[a:b]) for a, b in zip(ptr[:-1], ptr[1:])
        ])
        output["old_rank"][out_index] = strict_rank(molecule_global, labels)
        output["labels"][out_index] = labels
        output["global"][out_index] = global_pair
        output["reference_ptr"][out_index] = ptr
        local = reference_scores(
            cache, int(body["query_row"][query]), reference_rows, args,
        )
        for variant in VARIANTS:
            output[variant][out_index] = local[variant]
        if (out_index + 1) % 64 == 0:
            print(f"scored {out_index + 1}/{count}", flush=True)
    return output


def fused_ranks(scored: dict, variant: str, beta: float) -> np.ndarray:
    ranks = np.empty(len(scored["query"]), dtype=np.int16)
    for index in range(len(ranks)):
        pair_score = (
            np.asarray(scored["global"][index], dtype=np.float32)
            + beta * np.asarray(scored[variant][index], dtype=np.float32)
        )
        ptr = np.asarray(scored["reference_ptr"][index], dtype=np.int64)
        molecule = np.asarray([
            np.max(pair_score[a:b]) for a, b in zip(ptr[:-1], ptr[1:])
        ])
        ranks[index] = strict_rank(
            molecule, np.asarray(scored["labels"][index], dtype=bool),
        )
    return ranks


def main() -> None:
    args = arguments(); started = time.time()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    required = (
        args.manifest, args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy",
        args.token_dir / "mz_f32.npy", args.token_dir / "intensity_f32.npy",
        args.token_dir / "valid.npy", args.token_dir / "precursor_mz_f32.npy",
        args.rule_rank_file,
    )
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    with np.load(args.manifest) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    train = np.flatnonzero((fold != args.inner_fold) & (fold != args.outer_fold))
    inner_pool = np.flatnonzero(fold == args.inner_fold)
    outer = np.flatnonzero(fold == args.outer_fold)
    train_error, _ = official_outcomes(body, train, official, row_position, None)
    discovery = identity_balanced_queries(
        train, body["query_ik14"], np.random.default_rng(args.seed + 72),
        args.discovery_natural_identities,
    )
    inner = identity_balanced_queries(
        inner_pool, body["query_ik14"], np.random.default_rng(args.seed + 19),
        args.max_inner_identities,
    )
    cache = PeakMassCache(args, row_position)
    print(
        f"natural_discovery={len(discovery)} inner={len(inner)} "
        f"outer_untouched={len(outer)}", flush=True,
    )
    discovery_scored = score_queries(
        discovery, body, official, row_position, cache, args,
    )
    inner_scored = score_queries(inner, body, official, row_position, cache, args)
    selection = []
    for beta in map(float, args.fusion_grid):
        rank = fused_ranks(discovery_scored, "direct_or_loss", beta)
        item = summary(discovery_scored["old_rank"], rank)
        item.update({
            "beta": beta,
            "risk_utility": int(item["corrected"] - 2 * item["introduced"]),
        })
        selection.append(item)
    selected = max(
        selection,
        key=lambda item: (
            item["risk_utility"], item["delta_recall1"], item["delta_mrr"],
            -item["beta"],
        ),
    )
    beta = float(selected["beta"])
    held_rank = {
        variant: fused_ranks(inner_scored, variant, beta)
        for variant in VARIANTS
    }
    held = {
        variant: summary(inner_scored["old_rank"], rank)
        for variant, rank in held_rank.items()
    }
    primary = "direct_or_loss"
    old_hit = inner_scored["old_rank"] == 1
    primary_hit = held_rank[primary] == 1
    absolute = formula_bootstrap(
        primary_hit.astype(float) - old_hit.astype(float),
        inner_scored["formula"], args.seed + 901, args.bootstrap_draws,
    )
    comparisons = {}
    for offset, control in enumerate(
        ("direct_mz", "neutral_loss", "mass_shifted", "intensity_permuted")
    ):
        delta = primary_hit.astype(float) - (held_rank[control] == 1).astype(float)
        comparisons[f"{primary}_minus_{control}"] = {
            "recall1_advantage": float(np.mean(delta)),
            **formula_bootstrap(
                delta, inner_scored["formula"],
                args.seed + 1201 + 41 * offset, args.bootstrap_draws,
            ),
        }
    with np.load(args.rule_rank_file) as loaded:
        if not np.array_equal(inner, loaded["query"].astype(np.int64)):
            raise RuntimeError("rule-mass and overlap audits use different inner queries")
        rule_rank = loaded["rule_mass_rank"].astype(np.int64)
    overlap_minus_rule = primary_hit.astype(float) - (rule_rank == 1).astype(float)
    comparisons["direct_or_loss_minus_rule_mass"] = {
        "recall1_advantage": float(np.mean(overlap_minus_rule)),
        **formula_bootstrap(
            overlap_minus_rule, inner_scored["formula"],
            args.seed + 1501, args.bootstrap_draws,
        ),
    }
    gates = {
        "absolute_formula_ci_positive": absolute["formula_cluster_bootstrap_95ci"][0] > 0,
        "corrected_exceeds_introduced": held[primary]["corrected"] > held[primary]["introduced"],
        "beats_mass_shifted": comparisons["direct_or_loss_minus_mass_shifted"]["formula_cluster_bootstrap_95ci"][0] > 0,
        "beats_current_rule_mass": comparisons["direct_or_loss_minus_rule_mass"]["formula_cluster_bootstrap_95ci"][0] > 0,
    }
    report = {
        "status": "DEVELOPMENT_OVERLAP_PASS" if all(gates.values()) else "DEVELOPMENT_OVERLAP_FAIL",
        "scope": "non-separable training-teacher audit; no weights updated; outer untouched",
        "protocol": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "data": {
            "natural_discovery_queries": int(len(discovery)),
            "natural_discovery_baseline_error_fraction": float(np.mean(train_error[discovery])),
            "inner_queries": int(len(inner)),
            "outer_queries_untouched": int(len(outer)),
            "cached_spectra": int(len(cache.cache)),
        },
        "selection_on_training_formulas": selection,
        "selected_beta": beta,
        "held_inner": held,
        "primary_absolute_formula_bootstrap": absolute,
        "paired_formula_cluster_comparisons": comparisons,
        "gates": gates,
        "contracts": {
            "pair_fusion_before_molecule_max": True,
            "natural_error_rate_used_for_selection": True,
            "local_teacher_nonseparable": True,
            "teacher_training_only": True,
            "student_inference_clean_spectrum_only": True,
            "outer_fold_evaluated": False,
            "training_was_run": False,
        },
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "rule_rank_file_sha256": sha256_file(args.rule_rank_file),
        },
        "runtime_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    np.savez_compressed(
        args.output / "inner_per_query.npz", query=inner,
        formula=inner_scored["formula"], old_rank=inner_scored["old_rank"],
        **{f"{variant}_rank": rank for variant, rank in held_rank.items()},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
