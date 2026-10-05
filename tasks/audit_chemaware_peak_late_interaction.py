"""Audit a spectrum-observable, peak-level late-interaction teacher.

The audit never updates DreaMS.  It asks whether local contextual peak-token
matching can improve the frozen official complete-candidate ranking and whether
the gain depends on chemically valid peak/mass associations.  Fusion weight is
chosen on training formulas and applied once to the held inner formula fold.
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

from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
from train_chemaware_full_candidate_alignment import (  # noqa: E402
    formula_bootstrap, identity_balanced_queries, official_outcomes,
)


METHODS = ("mass_token", "rule_mass_token", "unmasked_token", "peak_permuted", "mass_overlap")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--rules", type=Path, default=ROOT / "dreams/models/chem_aware/chem_rules_data.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--discovery-errors", type=int, default=128)
    parser.add_argument("--discovery-correct", type=int, default=128)
    parser.add_argument("--discovery-natural-identities", type=int, default=2048)
    parser.add_argument("--max-inner-identities", type=int, default=256)
    parser.add_argument("--top-peaks", type=int, default=16)
    parser.add_argument("--sketch-dim", type=int, default=128)
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--abs-tolerance", type=float, default=0.01)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--fusion-grid", type=float, nargs="+", default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80))
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def zscore(values: np.ndarray) -> np.ndarray:
    sd = float(np.std(values))
    return (values - float(np.mean(values))) / max(sd, 1e-6)


def strict_rank(scores: np.ndarray, labels: np.ndarray) -> int:
    positive = float(scores[np.flatnonzero(labels)[0]])
    return 1 + int(np.sum(scores[~labels] >= positive))


def summary(old_rank: np.ndarray, new_rank: np.ndarray) -> dict:
    old_hit = old_rank == 1; new_hit = new_rank == 1
    return {
        "queries": int(len(old_rank)),
        "baseline_recall1": float(np.mean(old_hit)),
        "recall1": float(np.mean(new_hit)),
        "delta_recall1": float(np.mean(new_hit) - np.mean(old_hit)),
        "baseline_mrr": float(np.mean(1 / old_rank)),
        "mrr": float(np.mean(1 / new_rank)),
        "delta_mrr": float(np.mean(1 / new_rank) - np.mean(1 / old_rank)),
        "corrected": int(np.sum(~old_hit & new_hit)),
        "introduced": int(np.sum(old_hit & ~new_hit)),
    }


class PeakCache:
    def __init__(self, args: argparse.Namespace, row_position: dict[int, int]):
        self.top = args.top_peaks; self.dim = args.sketch_dim; self.row_position = row_position
        if 1024 % self.dim:
            raise ValueError("sketch-dim must divide the 1024-dimensional token width")
        self.tokens = np.load(args.token_dir / "tokens_f16.npy", mmap_mode="r")
        self.mz = np.load(args.token_dir / "mz_f32.npy", mmap_mode="r")
        self.intensity = np.load(args.token_dir / "intensity_f32.npy", mmap_mode="r")
        self.valid = np.load(args.token_dir / "valid.npy", mmap_mode="r")
        self.precursor = np.load(args.token_dir / "precursor_mz_f32.npy", mmap_mode="r")
        rng = np.random.default_rng(args.seed + 411)
        self.permutation = rng.permutation(1024)
        self.sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=1024)
        self.group = 1024 // self.dim
        self.cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = {}

    def get(self, row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        row = int(row)
        if row in self.cache:
            return self.cache[row]
        pos = self.row_position[row]
        available = np.flatnonzero(self.valid[pos])
        order = available[np.argsort(-np.asarray(self.intensity[pos, available]), kind="stable")[:self.top]]
        count = len(order)
        z = np.zeros((self.top, self.dim), dtype=np.float32)
        mz = np.zeros(self.top, dtype=np.float32)
        weight = np.zeros(self.top, dtype=np.float32)
        mask = np.zeros(self.top, dtype=bool)
        if count:
            raw = np.asarray(self.tokens[pos, order], dtype=np.float32)
            raw = raw[:, self.permutation] * self.sign
            raw = raw.reshape(count, self.dim, self.group).sum(axis=2) / np.sqrt(self.group)
            raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-8)
            z[:count] = raw; mz[:count] = self.mz[pos, order]
            values = np.maximum(np.asarray(self.intensity[pos, order], dtype=np.float32), 0)
            weight[:count] = values / max(float(np.sum(values)), 1e-8); mask[:count] = True
        result = (z, mz, weight, mask, float(self.precursor[pos]))
        self.cache[row] = result
        return result


def load_rule_masses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    body = json.loads(path.read_text(encoding="utf-8"))
    neutral = [float(rule["value"]) for rule in body["rules"] if rule.get("category") == "NL" and rule.get("match_type") == "mass_diff"]
    fragment = [float(rule["value"]) for rule in body["rules"] if rule.get("category") == "CF" and rule.get("match_type") == "peak_mz"]
    return np.asarray(neutral, dtype=np.float32), np.asarray(fragment, dtype=np.float32)


def rule_hits(mz: np.ndarray, precursor: float, valid: np.ndarray, neutral: np.ndarray, fragment: np.ndarray, tolerance: float) -> np.ndarray:
    output = np.zeros((len(mz), len(neutral) + len(fragment)), dtype=bool)
    if np.any(valid):
        losses = precursor - mz[valid]
        output[valid, :len(neutral)] = np.abs(losses[:, None] - neutral[None, :]) <= tolerance
        output[valid, len(neutral):] = np.abs(mz[valid, None] - fragment[None, :]) <= tolerance
    return output


def symmetric(mask: np.ndarray, similarity: np.ndarray, qweight: np.ndarray, rweight: np.ndarray) -> np.ndarray:
    values = np.where(mask, np.maximum(similarity, 0), 0)
    qside = np.sum(np.max(values, axis=2) * qweight[None, :], axis=1)
    rside = np.sum(np.max(values, axis=1) * rweight, axis=1)
    return 0.5 * (qside + rside)


def reference_scores(
    cache: PeakCache, query_row: int, reference_rows: np.ndarray,
    args: argparse.Namespace, neutral_rules: np.ndarray, fragment_rules: np.ndarray,
) -> dict[str, np.ndarray]:
    qz, qmz, qw, qvalid, qprecursor = cache.get(query_row)
    n = len(reference_rows)
    rz = np.empty((n, args.top_peaks, args.sketch_dim), dtype=np.float32)
    rmz = np.empty((n, args.top_peaks), dtype=np.float32)
    rw = np.empty((n, args.top_peaks), dtype=np.float32)
    rvalid = np.empty((n, args.top_peaks), dtype=bool)
    rprecursor = np.empty(n, dtype=np.float32)
    rz_permuted = np.empty_like(rz)
    for index, row in enumerate(map(int, reference_rows)):
        z, mz, weight, valid, precursor = cache.get(row)
        rz[index] = z; rmz[index] = mz; rw[index] = weight
        rvalid[index] = valid; rprecursor[index] = precursor
        rz_permuted[index] = z
        count = int(np.sum(valid))
        if count > 1:
            shift = 1 + (row % (count - 1))
            rz_permuted[index, :count] = np.roll(z[:count], shift, axis=0)
    sim = np.einsum("kd,njd->nkj", qz, rz, optimize=True)
    sim_permuted = np.einsum("kd,njd->nkj", qz, rz_permuted, optimize=True)
    valid_pair = qvalid[None, :, None] & rvalid[:, None, :]
    delta = np.abs(qmz[None, :, None] - rmz[:, None, :])
    tolerance = np.maximum(args.abs_tolerance, args.ppm * np.maximum(np.abs(qmz[None, :, None]), np.abs(rmz[:, None, :])) / 1e6)
    direct = valid_pair & (delta <= tolerance)
    qloss = qprecursor - qmz
    rloss = rprecursor[:, None] - rmz
    loss_delta = np.abs(qloss[None, :, None] - rloss[:, None, :])
    loss_tol = np.maximum(args.abs_tolerance, args.ppm * np.maximum(np.abs(qloss[None, :, None]), np.abs(rloss[:, None, :])) / 1e6)
    mass_mask = direct | (valid_pair & (loss_delta <= loss_tol))
    qrule = rule_hits(qmz, qprecursor, qvalid, neutral_rules, fragment_rules, args.rule_tolerance)
    rule_mask = np.empty_like(valid_pair)
    for index in range(n):
        rrule = rule_hits(rmz[index], float(rprecursor[index]), rvalid[index], neutral_rules, fragment_rules, args.rule_tolerance)
        rule_mask[index] = (qrule.astype(np.uint8) @ rrule.astype(np.uint8).T) > 0
    rule_mass_mask = mass_mask | (valid_pair & rule_mask)
    return {
        "mass_token": symmetric(mass_mask, sim, qw, rw),
        "rule_mass_token": symmetric(rule_mass_mask, sim, qw, rw),
        "unmasked_token": symmetric(valid_pair, sim, qw, rw),
        "peak_permuted": symmetric(rule_mass_mask, sim_permuted, qw, rw),
        "mass_overlap": symmetric(mass_mask, np.ones_like(sim), qw, rw),
    }


def score_queries(
    queries: np.ndarray, body: dict[str, np.ndarray], official: np.ndarray,
    row_position: dict[int, int], cache: PeakCache, args: argparse.Namespace,
    neutral_rules: np.ndarray, fragment_rules: np.ndarray,
) -> dict[str, np.ndarray]:
    output = {"query": queries.astype(np.int64), "formula": body["query_formula"][queries]}
    old_rank = np.empty(len(queries), dtype=np.int16)
    local_rank = {method: np.empty(len(queries), dtype=np.int16) for method in METHODS}
    official_scores = np.empty(len(queries), dtype=object)
    local_scores = {method: np.empty(len(queries), dtype=object) for method in METHODS}
    reference_ptr = np.empty(len(queries), dtype=object)
    labels_all = np.empty(len(queries), dtype=object)
    for out_index, query in enumerate(map(int, queries)):
        left, right = map(int, body["query_ptr"][query:query + 2])
        labels = body["molecule_label"][left:right].astype(bool)
        ref_left, ref_right = map(int, body["molecule_ptr"][left:right + 1][[0, -1]])
        reference_rows = body["pair_candidate_row"][ref_left:ref_right]
        per_ref = reference_scores(cache, int(body["query_row"][query]), reference_rows, args, neutral_rules, fragment_rules)
        query_embedding = official[row_position[int(body["query_row"][query])]]
        reference_embedding = official[[row_position[int(row)] for row in reference_rows]]
        global_ref = reference_embedding @ query_embedding
        molecule_global = np.empty(right - left, dtype=np.float32)
        molecule_local = {method: np.empty(right - left, dtype=np.float32) for method in METHODS}
        for offset, molecule in enumerate(range(left, right)):
            mleft, mright = map(int, body["molecule_ptr"][molecule:molecule + 2])
            relative_left = mleft - ref_left; relative_right = mright - ref_left
            molecule_global[offset] = float(np.max(global_ref[relative_left:relative_right]))
            for method in METHODS:
                molecule_local[method][offset] = float(np.max(per_ref[method][relative_left:relative_right]))
        old_rank[out_index] = strict_rank(molecule_global, labels)
        # Preserve reference-level scores.  Global and local evidence must be
        # fused on the same query-reference pair before a candidate molecule
        # takes its best reference; otherwise two different spectra can
        # contribute the two maxima and create an impossible retrieval score.
        official_scores[out_index] = global_ref.astype(np.float32)
        reference_ptr[out_index] = (
            body["molecule_ptr"][left:right + 1].astype(np.int64) - ref_left
        )
        labels_all[out_index] = labels
        for method in METHODS:
            local_scores[method][out_index] = per_ref[method].astype(np.float32)
            local_rank[method][out_index] = strict_rank(molecule_local[method], labels)
        if (out_index + 1) % 64 == 0:
            print(f"scored {out_index + 1}/{len(queries)}", flush=True)
    output["old_rank"] = old_rank; output["labels"] = labels_all
    output["official_scores"] = official_scores
    output["reference_ptr"] = reference_ptr
    for method in METHODS:
        output[f"{method}_rank"] = local_rank[method]; output[f"{method}_scores"] = local_scores[method]
    return output


def fused_ranks(
    scored: dict[str, np.ndarray], method: str, alpha: float, normalize_within_query: bool = True,
) -> np.ndarray:
    ranks = np.empty(len(scored["query"]), dtype=np.int16)
    for index in range(len(ranks)):
        global_score = np.asarray(scored["official_scores"][index], dtype=np.float32)
        local_score = np.asarray(scored[f"{method}_scores"][index], dtype=np.float32)
        if normalize_within_query:
            pair_score = zscore(global_score) + alpha * zscore(local_score)
        else:
            pair_score = global_score + alpha * local_score
        ptr = np.asarray(scored["reference_ptr"][index], dtype=np.int64)
        molecule_score = np.asarray([
            np.max(pair_score[left:right])
            for left, right in zip(ptr[:-1], ptr[1:])
        ], dtype=np.float32)
        ranks[index] = strict_rank(
            molecule_score, np.asarray(scored["labels"][index], dtype=bool),
        )
    return ranks


def main() -> None:
    args = arguments(); started = time.time()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    with np.load(args.manifest) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    train_pool = np.flatnonzero((fold != args.inner_fold) & (fold != args.outer_fold))
    inner_pool = np.flatnonzero(fold == args.inner_fold)
    outer_pool = np.flatnonzero(fold == args.outer_fold)
    train_error, _ = official_outcomes(body, train_pool, official, row_position, None)
    rng = np.random.default_rng(args.seed + 71)
    discovery_error = identity_balanced_queries(
        train_pool[train_error[train_pool]], body["query_ik14"], rng, args.discovery_errors,
    )
    discovery_correct = identity_balanced_queries(
        train_pool[~train_error[train_pool]], body["query_ik14"], rng, args.discovery_correct,
    )
    balanced_discovery = np.concatenate((discovery_error, discovery_correct))
    rng.shuffle(balanced_discovery)
    # Select the fusion weight at the natural training-formula error rate.
    # The 50/50 cohort remains a headroom diagnostic only; using it for
    # selection previously over-weighted rare failures.
    discovery = identity_balanced_queries(
        train_pool, body["query_ik14"], np.random.default_rng(args.seed + 72),
        args.discovery_natural_identities,
    )
    inner = identity_balanced_queries(inner_pool, body["query_ik14"], np.random.default_rng(args.seed + 19), args.max_inner_identities)
    neutral_rules, fragment_rules = load_rule_masses(args.rules)
    cache = PeakCache(args, row_position)
    print(
        f"natural_discovery={len(discovery)} "
        f"balanced_headroom={len(balanced_discovery)} "
        f"inner={len(inner)} outer_untouched={len(outer_pool)}", flush=True,
    )
    discovery_scores = score_queries(discovery, body, official, row_position, cache, args, neutral_rules, fragment_rules)
    inner_scores = score_queries(inner, body, official, row_position, cache, args, neutral_rules, fragment_rules)
    selection = []
    for alpha in args.fusion_grid:
        ranks = fused_ranks(discovery_scores, "rule_mass_token", float(alpha))
        item = summary(discovery_scores["old_rank"], ranks)
        item["alpha"] = float(alpha); item["risk_utility"] = item["corrected"] - 2 * item["introduced"]
        selection.append(item)
    selected = max(selection, key=lambda item: (item["risk_utility"], item["delta_recall1"], item["delta_mrr"], -item["alpha"]))
    alpha = float(selected["alpha"])
    raw_selection = []
    for raw_alpha in args.fusion_grid:
        ranks = fused_ranks(
            discovery_scores, "mass_overlap", float(raw_alpha), normalize_within_query=False,
        )
        item = summary(discovery_scores["old_rank"], ranks)
        item["alpha"] = float(raw_alpha); item["risk_utility"] = item["corrected"] - 2 * item["introduced"]
        raw_selection.append(item)
    raw_selected = max(
        raw_selection,
        key=lambda item: (item["risk_utility"], item["delta_recall1"], item["delta_mrr"], -item["alpha"]),
    )
    raw_alpha = float(raw_selected["alpha"])
    raw_inner_rank = fused_ranks(
        inner_scores, "mass_overlap", raw_alpha, normalize_within_query=False,
    )
    held = {}
    held_rank = {}
    for method in METHODS:
        rank = fused_ranks(inner_scores, method, alpha); held_rank[method] = rank
        held[method] = summary(inner_scores["old_rank"], rank)
    comparisons = {}
    correct_hit = held_rank["rule_mass_token"] == 1
    for offset, control in enumerate(("mass_token", "unmasked_token", "peak_permuted", "mass_overlap")):
        delta = correct_hit.astype(float) - (held_rank[control] == 1).astype(float)
        comparisons[f"rule_mass_token_minus_{control}"] = {
            "recall1_advantage": float(np.mean(delta)),
            **formula_bootstrap(delta, inner_scores["formula"], args.seed + 1201 + offset * 37, args.bootstrap_draws),
        }
    absolute_delta = (held_rank["rule_mass_token"] == 1).astype(float) - (inner_scores["old_rank"] == 1).astype(float)
    absolute_ci = formula_bootstrap(absolute_delta, inner_scores["formula"], args.seed + 1601, args.bootstrap_draws)
    report = {
        "status": "DEVELOPMENT_HEADROOM_PASS" if (
            absolute_ci["formula_cluster_bootstrap_95ci"][0] > 0
            and held["rule_mass_token"]["corrected"] > held["rule_mass_token"]["introduced"]
            and comparisons["rule_mass_token_minus_mass_token"]["formula_cluster_bootstrap_95ci"][0] > 0
            and comparisons["rule_mass_token_minus_peak_permuted"]["formula_cluster_bootstrap_95ci"][0] > 0
        ) else "DEVELOPMENT_HEADROOM_FAIL",
        "scope": "frozen teacher audit only; DreaMS parameters unchanged; outer fold untouched",
        "protocol": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data": {"natural_discovery_queries": int(len(discovery)),
                 "natural_discovery_baseline_error_fraction": float(np.mean(train_error[discovery])),
                 "balanced_headroom_queries_not_used_for_selection": int(len(balanced_discovery)),
                 "inner_queries": int(len(inner)),
                 "outer_queries_untouched": int(len(outer_pool)), "cached_rows_touched": int(len(cache.cache))},
        "rule_library": {"neutral_loss_rules": int(len(neutral_rules)), "fragment_mass_rules": int(len(fragment_rules)),
                         "sha256": sha256_file(args.rules)},
        "selection_on_training_formulas": selection, "selected_alpha": alpha,
        "raw_mass_overlap_selection_on_training_formulas": raw_selection,
        "raw_mass_overlap_selected_alpha": raw_alpha,
        "raw_mass_overlap_held_inner": summary(inner_scores["old_rank"], raw_inner_rank),
        "held_inner": held, "held_absolute_formula_bootstrap": absolute_ci,
        "paired_formula_cluster_comparisons": comparisons,
        "contracts": {"clean_spectrum_only": True, "candidate_structure_used": False,
                      "same_query_reference_features": True, "formula_disjoint": True,
                      "pair_fusion_before_molecule_max": True,
                      "natural_error_rate_used_for_selection": True,
                      "outer_fold_evaluated": False, "training_was_run": False},
        "runtime_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output / "inner_per_query.npz", query=inner_scores["query"], formula=inner_scores["formula"],
        old_rank=inner_scores["old_rank"], raw_mass_overlap_rank=raw_inner_rank,
        **{f"{method}_rank": held_rank[method] for method in METHODS},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
