"""Learn and test a PSD observation subspace appended to frozen DreaMS.

The learned object is a nonnegative diagonal metric over strict positive-ion
observation channels. Each spectrum is independently transformed and unit
normalized, so concatenation with frozen official DreaMS is an exact shared
product-space embedding. Formula folds 0/1 fit the metric; fold 2 is a frozen
confirmation. Historical inner fold 3 and outer fold 4 are not read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]
from audit_chemaware_mass_kernel_embedding import KernelCache, strict_rank  # noqa: E402
from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
from train_chemaware_full_candidate_alignment import formula_bootstrap, identity_balanced_queries  # noqa: E402

VARIANTS = ("mass", "rule_response", "rule_response_shifted", "rule_response_row_permuted")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--registry", type=Path, default=ROOT / "dreams/models/chem_aware/chem_observation_channels_v2.json")
    parser.add_argument("--registry-report", type=Path, default=ROOT / "data/validation/chemaware_observation_channel_registry_v2/report.json")
    parser.add_argument("--data-semantics-report", type=Path, default=ROOT / "data/validation/chemaware_data_semantics_gate_v1/report.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--fit-folds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--confirmation-fold", type=int, default=2)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--max-fit-identities", type=int, default=512)
    parser.add_argument("--max-confirmation-identities", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.20)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--uniform-regularization", type=float, default=0.02)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--screen-ledgers", type=Path, nargs="*", default=(),
        help="Prior screen NPZ files; their confirmation formulae are excluded from locked confirmation.",
    )
    return parser.parse_args()


def transform(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    transformed = values * torch.sqrt(torch.clamp(weight, min=1e-12))
    return F.normalize(transformed, dim=-1)


def fit_metric(
    query: np.ndarray, positive: np.ndarray, negative: np.ndarray,
    official_margin: np.ndarray, formula: np.ndarray, args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict]]:
    q = torch.from_numpy(query.astype(np.float32)); p = torch.from_numpy(positive.astype(np.float32))
    n = torch.from_numpy(negative.astype(np.float32)); margin = torch.from_numpy(official_margin.astype(np.float32))
    _, inverse = np.unique(formula.astype(str), return_inverse=True)
    counts = np.bincount(inverse); sample_weight = torch.from_numpy((1.0 / counts[inverse]).astype(np.float32))
    sample_weight *= len(sample_weight) / sample_weight.sum()
    theta = torch.zeros(q.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=args.learning_rate)
    history = []
    for step in range(1, args.steps + 1):
        metric_weight = q.shape[1] * torch.softmax(theta, dim=0)
        chemical_margin = torch.sum(transform(q, metric_weight) * transform(p, metric_weight), dim=1)
        chemical_margin -= torch.sum(transform(q, metric_weight) * transform(n, metric_weight), dim=1)
        total_margin = margin + args.beta * chemical_margin
        ranking = torch.sum(F.softplus(-total_margin / args.temperature) * sample_weight) / sample_weight.sum()
        regularization = args.uniform_regularization * torch.mean((metric_weight - 1.0) ** 2)
        loss = ranking + regularization
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            history.append({
                "step": step, "loss": float(loss.detach()), "ranking": float(ranking.detach()),
                "regularization": float(regularization.detach()),
                "weight_min": float(metric_weight.min().detach()),
                "weight_max": float(metric_weight.max().detach()),
            })
    weight = (q.shape[1] * torch.softmax(theta.detach(), dim=0)).numpy().astype(np.float32)
    return weight, history


def select_pairs(
    queries: np.ndarray, body: dict[str, np.ndarray], official: np.ndarray,
    row_position: dict[int, int], cache: KernelCache, variant: str,
) -> dict[str, np.ndarray]:
    qvec = []; pvec = []; nvec = []; margin = []; kept = []
    for index, query in enumerate(map(int, queries)):
        left, right = map(int, body["query_ptr"][query:query + 2])
        qrow = int(body["query_row"][query]); qo = official[row_position[qrow]]
        qfeature = cache.get(qrow)[variant].astype(np.float32)
        best = []
        for molecule in range(left, right):
            rleft, rright = map(int, body["molecule_ptr"][molecule:molecule + 2])
            references = body["pair_candidate_row"][rleft:rright]
            scores = official[[row_position[int(row)] for row in references]] @ qo
            chosen = int(references[int(np.argmax(scores))])
            best.append((float(np.max(scores)), chosen))
        if len(best) < 2:
            continue
        negative_index = 1 + int(np.argmax([item[0] for item in best[1:]]))
        qvec.append(qfeature); pvec.append(cache.get(best[0][1])[variant].astype(np.float32))
        nvec.append(cache.get(best[negative_index][1])[variant].astype(np.float32))
        margin.append(best[0][0] - best[negative_index][0]); kept.append(query)
        if (index + 1) % 512 == 0:
            print(f"{variant} pairs {index + 1}/{len(queries)}", flush=True)
    return {
        "query": np.asarray(kept, dtype=np.int64), "query_feature": np.stack(qvec),
        "positive_feature": np.stack(pvec), "negative_feature": np.stack(nvec),
        "official_margin": np.asarray(margin, dtype=np.float32),
        "formula": body["query_formula"][np.asarray(kept, dtype=np.int64)].astype(str),
    }


def evaluate(
    queries: np.ndarray, body: dict[str, np.ndarray], official: np.ndarray,
    row_position: dict[int, int], cache: KernelCache, variant: str,
    weight: np.ndarray, beta: float,
) -> np.ndarray:
    rank = np.empty(len(queries), dtype=np.int16)
    root_weight = np.sqrt(np.maximum(weight, 1e-12)).astype(np.float32)
    for index, query in enumerate(map(int, queries)):
        left, right = map(int, body["query_ptr"][query:query + 2])
        labels = body["molecule_label"][left:right].astype(bool)
        qrow = int(body["query_row"][query]); qo = official[row_position[qrow]]
        qchem = cache.get(qrow)[variant].astype(np.float32) * root_weight
        qchem /= max(float(np.linalg.norm(qchem)), 1e-12)
        molecule_scores = []
        for molecule in range(left, right):
            rleft, rright = map(int, body["molecule_ptr"][molecule:molecule + 2])
            references = body["pair_candidate_row"][rleft:rright]
            ro = official[[row_position[int(row)] for row in references]] @ qo
            rchem = np.stack([cache.get(int(row))[variant] for row in references]).astype(np.float32)
            rchem *= root_weight[None, :]
            rchem /= np.maximum(np.linalg.norm(rchem, axis=1, keepdims=True), 1e-12)
            molecule_scores.append(float(np.max(ro + beta * (rchem @ qchem))))
        rank[index] = strict_rank(np.asarray(molecule_scores), labels)
        if (index + 1) % 256 == 0:
            print(f"evaluate {variant} {index + 1}/{len(queries)}", flush=True)
    return rank


def compare(left: np.ndarray, right: np.ndarray, formula: np.ndarray, seed: int, draws: int) -> dict:
    delta = (left == 1).astype(float) - (right == 1).astype(float)
    return {"mean": float(np.mean(delta)), **formula_bootstrap(delta, formula, seed, draws)}


def main() -> None:
    args = arguments(); started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(set((*args.fit_folds, args.confirmation_fold, args.inner_fold, args.outer_fold))) != len(args.fit_folds) + 3:
        raise ValueError("fit/confirmation/inner/outer folds overlap")
    required = [args.manifest, args.token_dir / "rows.npy", args.token_dir / "official_embeddings_f32.npy",
                args.token_dir / "report.json", args.registry, args.registry_report, args.data_semantics_report]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    registry_report = json.loads(args.registry_report.read_text(encoding="utf-8"))
    semantics = json.loads(args.data_semantics_report.read_text(encoding="utf-8"))
    if (registry.get("schema") != "chemaware_observation_channel_registry_v2"
            or registry_report.get("status") != "CHEMAWARE_OBSERVATION_REGISTRY_SCHEMA_PASS"
            or registry_report.get("artifacts", {}).get("registry_sha256") != sha256_file(args.registry)
            or semantics.get("status") != "CHEMAWARE_DATA_SEMANTICS_PASS"):
        raise RuntimeError("registry or data semantics did not pass")
    if args.formal and (args.max_fit_identities != 0 or args.max_confirmation_identities != 0
                        or args.bootstrap_draws < 10_000 or not args.screen_ledgers):
        raise RuntimeError("formal audit requires every identity and at least 10,000 bootstrap draws")
    missing_ledgers = [str(path) for path in args.screen_ledgers if not path.is_file()]
    if missing_ledgers:
        raise FileNotFoundError(missing_ledgers)
    with np.load(args.manifest) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    folds = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    fit_pool = np.flatnonzero(np.isin(folds, args.fit_folds)); confirmation_pool = np.flatnonzero(folds == args.confirmation_fold)
    excluded_confirmation_formulas: set[str] = set()
    for ledger_path in args.screen_ledgers:
        with np.load(ledger_path) as ledger:
            if "confirmation_formula" not in ledger.files:
                raise RuntimeError(f"screen ledger lacks confirmation_formula: {ledger_path}")
            excluded_confirmation_formulas.update(ledger["confirmation_formula"].astype(str))
    if excluded_confirmation_formulas:
        confirmation_pool = confirmation_pool[
            ~np.isin(body["query_formula"][confirmation_pool].astype(str), list(excluded_confirmation_formulas))
        ]
    if args.formal and not len(confirmation_pool):
        raise RuntimeError("screen exclusions consumed every confirmation formula")
    fit_queries = identity_balanced_queries(fit_pool, body["query_ik14"], np.random.default_rng(args.seed + 1), args.max_fit_identities)
    confirmation = identity_balanced_queries(confirmation_pool, body["query_ik14"], np.random.default_rng(args.seed + 2), args.max_confirmation_identities)
    if set(body["query_formula"][fit_queries].astype(str)) & set(body["query_formula"][confirmation].astype(str)):
        raise RuntimeError("fit/confirmation formula overlap")
    kernel_args = SimpleNamespace(
        token_dir=args.token_dir, rule_library=args.registry, top_peaks=args.top_peaks,
        kernel_dim=2048, bin_width=0.02, grid_offsets=4, intensity_power=0.5,
        mass_shift_da=args.mass_shift_da, pair_weight=0.25, multi_bin_widths=(0.01, 0.02, 0.05),
        uniform_channel_weight=1.0, rule_tolerance=args.rule_tolerance, rule_channel_weight=1.0,
    )
    cache = KernelCache(kernel_args, row_position, variants=VARIANTS)
    fit_payloads = {}; weights = {}; histories = {}
    for offset, variant in enumerate(VARIANTS[1:]):
        payload = select_pairs(fit_queries, body, official, row_position, cache, variant)
        weight, history = fit_metric(
            payload["query_feature"], payload["positive_feature"], payload["negative_feature"],
            payload["official_margin"], payload["formula"], args,
        )
        fit_payloads[variant] = payload; weights[variant] = weight; histories[variant] = history
    dimension = len(registry["channels"]); uniform = np.ones(dimension, dtype=np.float32)
    mass_uniform = np.ones(2048, dtype=np.float32)
    official_rank = evaluate(confirmation, body, official, row_position, cache, "rule_response", uniform, 0.0)
    mass_rank = evaluate(confirmation, body, official, row_position, cache, "mass", mass_uniform, 0.10)
    uniform_rank = evaluate(confirmation, body, official, row_position, cache, "rule_response", uniform, args.beta)
    learned_rank = evaluate(confirmation, body, official, row_position, cache, "rule_response", weights["rule_response"], args.beta)
    shifted_rank = evaluate(confirmation, body, official, row_position, cache, "rule_response_shifted", weights["rule_response_shifted"], args.beta)
    permuted_rank = evaluate(confirmation, body, official, row_position, cache, "rule_response_row_permuted", weights["rule_response_row_permuted"], args.beta)
    formula = body["query_formula"][confirmation].astype(str)
    ranks = {"official": official_rank, "mass": mass_rank, "uniform": uniform_rank, "learned": learned_rank,
             "shifted": shifted_rank, "permuted": permuted_rank}
    comparisons = {f"learned_minus_{name}": compare(learned_rank, value, formula, args.seed + 100 + i, args.bootstrap_draws)
                   for i, (name, value) in enumerate(ranks.items()) if name != "learned"}
    gates = {f"beats_{name}_point": item["mean"] > 0 for name, item in
             ((key.removeprefix("learned_minus_"), value) for key, value in comparisons.items())}
    gates.update({f"beats_{name}_formula_ci": item["formula_cluster_bootstrap_95ci"][0] > 0 for name, item in
                  ((key.removeprefix("learned_minus_"), value) for key, value in comparisons.items())})
    passed = bool(all(gates.values()))
    formal_pass = args.formal and passed
    report = {
        "status": "CHEMAWARE_PSD_OBSERVATION_FORMAL_PASS" if formal_pass else
                  "CHEMAWARE_PSD_OBSERVATION_SCREEN_PASS" if passed else "CHEMAWARE_PSD_OBSERVATION_FAIL",
        "formal": args.formal, "formal_training_authorized": formal_pass,
        "shared_embedding": {
            "definition": "normalize(concat(official, sqrt(beta)*normalize(sqrt(w)*observation)))",
            "dimension": int(official.shape[1] + dimension), "official_frozen": True,
            "one_spectrum_one_embedding": True, "positive_semidefinite_metric": True,
        },
        "data": {"fit_queries": len(fit_queries), "confirmation_queries": len(confirmation),
                 "fit_formulas": len(np.unique(body["query_formula"][fit_queries])),
                 "confirmation_formulas": len(np.unique(formula)), "formula_overlap": 0,
                 "screen_excluded_confirmation_formulas": len(excluded_confirmation_formulas),
                 "inner_fold_read": False, "outer_fold_read": False},
        "optimization": {"beta": args.beta, "temperature": args.temperature, "steps": args.steps,
                         "learning_rate": args.learning_rate, "uniform_regularization": args.uniform_regularization,
                         "mean_weight_fixed_to_one": True, "histories": histories},
        "confirmation": {name: {"recall1": float(np.mean(rank == 1)),
                                 "corrected_vs_official": int(np.sum((official_rank > 1) & (rank == 1))),
                                 "introduced_vs_official": int(np.sum((official_rank == 1) & (rank > 1)))}
                         for name, rank in ranks.items()},
        "comparisons": comparisons, "gates": gates,
        "controls": {"uniform_same_channels": True, "mass_shift_same_capacity": True,
                     "row_permuted_same_capacity": True},
        "provenance": {"manifest_sha256": sha256_file(args.manifest), "registry_sha256": sha256_file(args.registry),
                       "registry_report_sha256": sha256_file(args.registry_report),
                       "data_semantics_report_sha256": sha256_file(args.data_semantics_report),
                       "screen_ledger_sha256": {str(path): sha256_file(path) for path in args.screen_ledgers},
                       "script_sha256": sha256_file(Path(__file__))},
        "runtime_seconds": time.time() - started,
        "claim_limit": "Formula-disjoint product-space screen; no DreaMS weights changed and outer remains untouched.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "metric_and_ranks.npz", confirmation_query=confirmation,
                        confirmation_formula=formula, **{f"{key}_rank": value for key, value in ranks.items()},
                        **{f"{key}_weight": value for key, value in weights.items()})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
