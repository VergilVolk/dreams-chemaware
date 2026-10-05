"""Formula-OOF direct boundary transfer from the full S3A action ledger.

Unlike E4, this audit never ranks a successful action view as an easier second
positive.  Training-fold action outcomes only decide which *clean* molecular
candidate boundaries receive extra dose.  The label remains true identity and
the same bounded map is applied to clean queries and candidate references.

The full S3A ledger supplies substantially more formulas than the A4 scan.
Every evaluation formula is excluded from the S3A training pool in its outer
fold.  This remains a frozen-embedding reachability proxy, not an encoder
checkpoint or a replacement for raw-spectrum fine-tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_noise_direct_shared_metric_reachability import (
    evaluate_fold,
    transform_numpy,
)
from audit_noise_peak_gate_candidate_injection import (
    build_candidate_rows,
    formula_ci,
    stable_fold,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger", type=Path,
        default=Path("data/validation/g8r_noise_final_e0_unified_matrix_local_audit/unified_query_action_ledger.csv.gz"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"),
    )
    parser.add_argument(
        "--a4-dir", type=Path,
        default=Path("data/validation/g8r_noise_v3_a4_exact_peak_scan"),
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path("data/validation/chemaware_corrected_manifest_tokens_v1"),
    )
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/noise_s3a_direct_boundary_transfer_20260905"),
    )
    parser.add_argument(
        "--route", choices=(
            "clean_uniform", "action_correctable", "action_recurrence", "mature_n",
        ),
        default="action_correctable",
    )
    parser.add_argument(
        "--include-a4-training", action=argparse.BooleanOptionalAction, default=False,
        help="Merge A4 action rows with S3A before outer-formula exclusion.",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--top-negatives", type=int, default=8)
    parser.add_argument(
        "--spectra-per-molecule", type=int, default=1,
        help=(
            "Training-time candidate spectra retained per molecule. Values >1 "
            "use a differentiable molecule-max boundary instead of a frozen "
            "single representative."
        ),
    )
    parser.add_argument("--adapter-type", choices=("lowrank", "diagonal"), default="lowrank")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--residual-strength", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument(
        "--loss-kind", choices=("cross_entropy", "molecule_margin"),
        default="cross_entropy",
    )
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--correctable-weight", type=float, default=6.0)
    parser.add_argument(
        "--recurrence-saturation", type=float, default=4.0,
        help="Correcting-action count at which action_recurrence reaches full dose.",
    )
    parser.add_argument("--other-error-weight", type=float, default=0.75)
    parser.add_argument("--correct-control-weight", type=float, default=0.75)
    parser.add_argument(
        "--introduced-control-weight", type=float, default=None,
        help=(
            "Training dose for baseline-correct queries known in the training "
            "ledger to be broken by at least one action. None preserves the "
            "ordinary correct-control weight."
        ),
    )
    parser.add_argument("--safety-weight", type=float, default=10.0)
    parser.add_argument("--safety-slack", type=float, default=0.005)
    parser.add_argument("--preserve-weight", type=float, default=0.5)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cpu")
    # Reuse the trainer contract without branching its implementation.
    parser.set_defaults(map_side="shared", evaluation_mode="formula_oof")
    return parser.parse_args()


def restore_a4_query_rows(ledger: pd.DataFrame, scan_queries: pd.DataFrame) -> pd.DataFrame:
    """Restore the stable HDF5 key omitted from historical A4 action rows."""
    required = {"query_index", "query_row", "query_ik14", "query_formula"}
    if required - set(scan_queries.columns):
        raise RuntimeError("A4 scan query table cannot restore query_row")
    if scan_queries.query_index.duplicated().any():
        raise RuntimeError("A4 query_index mapping is not unique")
    mapping = scan_queries[list(required)].copy()
    row_by_query = mapping.set_index("query_index").query_row
    output = ledger.copy()
    a4 = output.stage.eq("A4")
    output.loc[a4, "query_row"] = output.loc[a4, "query_index"].map(row_by_query)
    if output.loc[a4, "query_row"].isna().any():
        missing = output.loc[a4 & output.query_row.isna(), "query_index"].unique()[:20]
        raise RuntimeError(f"A4 action rows lack scan-query mapping: {missing.tolist()}")
    identity = mapping.set_index("query_index").query_ik14.astype(str)
    formula = mapping.set_index("query_index").query_formula.astype(str)
    if not (
        output.loc[a4, "query_ik14"].astype(str).to_numpy()
        == output.loc[a4, "query_index"].map(identity).to_numpy()
    ).all():
        raise RuntimeError("A4 identity changed while restoring query_row")
    if not (
        output.loc[a4, "query_formula"].astype(str).to_numpy()
        == output.loc[a4, "query_index"].map(formula).to_numpy()
    ).all():
        raise RuntimeError("A4 formula changed while restoring query_row")
    return output


def load_s3a_queries(
    path: Path, include_a4: bool, a4_scan_queries: Path | None = None,
) -> pd.DataFrame:
    use = [
        "stage", "query_index", "query_row", "query_ik14", "query_formula",
        "baseline_rank", "baseline_margin", "corrected", "introduced",
        "specific_margin_excess", "hard_negative_row", "selector", "attenuation", "step",
    ]
    ledger = pd.read_csv(path, usecols=use, low_memory=False)
    stages = {"S3A", "A4"} if include_a4 else {"S3A"}
    ledger = ledger.loc[ledger.stage.isin(stages)].copy()
    if include_a4:
        if a4_scan_queries is None or not a4_scan_queries.is_file():
            raise FileNotFoundError("include_a4 requires the exact A4 scan_queries.csv.gz")
        ledger = restore_a4_query_rows(
            ledger, pd.read_csv(a4_scan_queries),
        )
    ledger["corrected"] = ledger.corrected.fillna(False).astype(bool)
    ledger["introduced"] = ledger.introduced.fillna(False).astype(bool)
    ledger["specific_margin_excess"] = pd.to_numeric(
        ledger.specific_margin_excess, errors="coerce"
    ).fillna(0.0)
    mature_n = (
        ledger.selector.eq("candidate_gradient")
        & ledger.attenuation.eq(0.50)
        & ledger.step.isin((3, 4, 5, 6))
    ) | (
        ledger.selector.eq("role_confounder")
        & ledger.attenuation.eq(1.00)
        & ledger.step.isin((1, 2, 3, 4, 5))
    )
    ledger["mature_n_corrected"] = mature_n & ledger.corrected
    ledger["mature_n_introduced"] = mature_n & ledger.introduced
    # HDF5 row is the stable spectrum key across source-specific query-index
    # conventions.  Grouping on query_index would silently split or merge rows
    # when S3A and A4 are combined.
    grouped = ledger.groupby("query_row", sort=True)
    query = grouped.agg(
        query_index=("query_index", "first"),
        query_ik14=("query_ik14", "first"),
        query_formula=("query_formula", "first"),
        baseline_rank=("baseline_rank", "first"),
        baseline_margin=("baseline_margin", "first"),
        action_correctable=("corrected", "max"),
        action_introduced=("introduced", "max"),
        positive_action_count=("corrected", "sum"),
        mature_n_correctable=("mature_n_corrected", "max"),
        mature_n_action_count=("mature_n_corrected", "sum"),
        mature_n_introduced=("mature_n_introduced", "max"),
        max_specific_margin_excess=("specific_margin_excess", "max"),
    ).reset_index()
    query["action_positive_advantage"] = np.maximum(
        0.0, query.max_specific_margin_excess.to_numpy(np.float32)
    )
    query["baseline_rank"] = query.baseline_rank.astype(np.int16)
    return query


def route_weights(query: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    """Allocate clean-boundary dose with an explicit harmful-action control."""
    if args.route == "clean_uniform":
        return np.ones(len(query), dtype=np.float32)
    is_error = query.baseline_rank.to_numpy(np.int64) > 1
    correctable = (
        query.mature_n_correctable.to_numpy(bool)
        if args.route == "mature_n" else query.action_correctable.to_numpy(bool)
    )
    introduced = query.action_introduced.to_numpy(bool)
    introduced_weight = (
        args.correct_control_weight
        if args.introduced_control_weight is None
        else args.introduced_control_weight
    )
    base = np.where(
        is_error, args.other_error_weight,
        np.where(introduced, introduced_weight, args.correct_control_weight),
    ).astype(np.float32)
    if args.route == "action_recurrence":
        if args.recurrence_saturation <= 0:
            raise ValueError("recurrence_saturation must be positive")
        recurrence = np.clip(
            query.positive_action_count.to_numpy(np.float32) / args.recurrence_saturation,
            0.0, 1.0,
        )
        return np.where(
            correctable,
            base + recurrence * (args.correctable_weight - base),
            base,
        ).astype(np.float32)
    return np.where(
        correctable,
        args.correctable_weight,
        base,
    ).astype(np.float32)


def manifest_representatives(
    query: pd.DataFrame, manifest_path: Path, embeddings: np.ndarray,
    cache_rows: np.ndarray, top_negatives: int, spectra_per_molecule: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if spectra_per_molecule < 1:
        raise ValueError("spectra_per_molecule must be positive")
    graph = np.load(manifest_path)
    graph_rows = graph["query_row"]
    query_ptr = graph["query_ptr"]
    graph_query_ik14 = graph["query_ik14"]
    graph_query_formula = graph["query_formula"]
    molecule_ptr = graph["molecule_ptr"]
    molecule_label = graph["molecule_label"]
    pair_candidate_row = graph["pair_candidate_row"]
    graph_query_by_row = {int(row): index for index, row in enumerate(graph_rows)}
    missing_query_rows = sorted(set(map(int, query.query_row)) - set(graph_query_by_row))
    if missing_query_rows:
        raise RuntimeError(
            f"candidate manifest misses {len(missing_query_rows)} S3A query rows"
        )
    cache_position = {int(row): index for index, row in enumerate(cache_rows)}
    positive = np.empty((len(query), spectra_per_molecule), dtype=np.int64)
    negatives = np.empty(
        (len(query), top_negatives, spectra_per_molecule), dtype=np.int64,
    )
    keep = np.ones(len(query), dtype=bool)
    for output, item in enumerate(query.itertuples(index=False)):
        q = int(graph_query_by_row[int(item.query_row)])
        if str(graph_query_ik14[q]) != str(item.query_ik14):
            raise RuntimeError(f"identity mismatch for S3A query row {item.query_row}")
        if str(graph_query_formula[q]) != str(item.query_formula):
            raise RuntimeError(f"formula mismatch for S3A query row {item.query_row}")
        molecule_left = int(query_ptr[q])
        molecule_right = int(query_ptr[q + 1])
        q_position = cache_position[int(item.query_row)]
        q_embedding = np.asarray(embeddings[q_position], dtype=np.float32)
        reps: list[np.ndarray] = []
        scores: list[float] = []
        labels: list[int] = []
        for molecule in range(molecule_left, molecule_right):
            pair_left = int(molecule_ptr[molecule])
            pair_right = int(molecule_ptr[molecule + 1])
            rows = pair_candidate_row[pair_left:pair_right]
            try:
                positions = np.asarray([cache_position[int(row)] for row in rows], dtype=np.int64)
            except KeyError:
                keep[output] = False
                break
            local_scores = np.asarray(embeddings[positions], dtype=np.float32) @ q_embedding
            spectrum_order = np.argsort(-local_scores, kind="stable")[:spectra_per_molecule]
            selected = positions[spectrum_order]
            if len(selected) < spectra_per_molecule:
                selected = np.pad(
                    selected, (0, spectra_per_molecule - len(selected)), mode="edge",
                )
            reps.append(np.asarray(selected, dtype=np.int64))
            scores.append(float(np.max(local_scores)))
            labels.append(int(molecule_label[molecule]))
        if not keep[output]:
            continue
        reps_array = np.stack(reps).astype(np.int64, copy=False)
        scores_array = np.asarray(scores, dtype=np.float32)
        labels_array = np.asarray(labels, dtype=np.int8)
        pos = np.flatnonzero(labels_array == 1)
        neg = np.flatnonzero(labels_array == 0)
        if len(pos) != 1 or len(neg) == 0:
            keep[output] = False
            continue
        order = neg[np.argsort(-scores_array[neg], kind="stable")]
        chosen = reps_array[order[:top_negatives]]
        if len(chosen) < top_negatives:
            chosen = np.pad(
                chosen, ((0, top_negatives - len(chosen)), (0, 0)), mode="edge",
            )
        positive[output] = reps_array[pos[0]]
        negatives[output] = chosen
    return query.loc[keep].reset_index(drop=True), positive[keep], negatives[keep]


def train_molecule_max_fold(
    train: np.ndarray,
    base_embeddings: np.ndarray,
    query_positions: np.ndarray,
    positive_positions: np.ndarray,
    negative_positions: np.ndarray,
    query: pd.DataFrame,
    weights: np.ndarray,
    args: argparse.Namespace,
    fold: int,
):
    """Shared residual metric with evaluator-aligned molecule maxima."""
    from audit_noise_direct_shared_metric_reachability import (
        SharedDiagonalMetric, SharedResidualMetric,
    )
    import torch.nn.functional as F

    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    device = torch.device(args.device)
    if args.adapter_type == "lowrank":
        model = SharedResidualMetric(
            base_embeddings.shape[1], args.hidden_dim, args.residual_strength,
        ).to(device)
    else:
        model = SharedDiagonalMetric(
            base_embeddings.shape[1], args.residual_strength,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed + fold)
    baseline_correct = query.baseline_rank.to_numpy(np.int64) == 1
    action_introduced = query.action_introduced.to_numpy(bool)
    introduced_weight = (
        1.0 if args.introduced_control_weight is None
        else float(args.introduced_control_weight)
    )
    last = None
    for epoch in range(args.epochs):
        order = rng.permutation(train)
        total = np.zeros(6, dtype=np.float64)
        for left in range(0, len(order), args.batch_size):
            index = order[left:left + args.batch_size]
            q0 = torch.as_tensor(
                np.asarray(base_embeddings[query_positions[index]], dtype=np.float32),
                device=device,
            )
            p0 = torch.as_tensor(
                np.asarray(base_embeddings[positive_positions[index]], dtype=np.float32),
                device=device,
            )
            n0 = torch.as_tensor(
                np.asarray(base_embeddings[negative_positions[index]], dtype=np.float32),
                device=device,
            )
            q = model(q0)
            if args.map_side == "shared":
                p = model(p0.reshape(-1, p0.shape[-1])).reshape_as(p0)
                n = model(n0.reshape(-1, n0.shape[-1])).reshape_as(n0)
            else:
                p, n = p0, n0
            positive_score = torch.einsum("bd,bpd->bp", q, p).max(dim=1).values
            negative_score = torch.einsum("bd,bnrd->bnr", q, n).max(dim=2).values
            if args.loss_kind == "cross_entropy":
                logits = torch.cat((positive_score[:, None], negative_score), dim=1) / args.temperature
                per_query = F.cross_entropy(
                    logits, torch.zeros(len(index), dtype=torch.long, device=device),
                    reduction="none",
                )
            else:
                current_hard_margin = positive_score - negative_score.max(dim=1).values
                per_query = F.softplus(
                    (float(args.rank_margin) - current_hard_margin) / args.temperature
                )
            sample_weight = torch.as_tensor(weights[index], device=device)
            rank_loss = torch.sum(sample_weight * per_query) / sample_weight.sum().clamp_min(1e-8)

            initial_positive = torch.einsum("bd,bpd->bp", q0, p0).max(dim=1).values
            initial_negative = torch.einsum("bd,bnrd->bnr", q0, n0).max(dim=2).values
            initial_margin = initial_positive - initial_negative.max(dim=1).values
            current_margin = positive_score - negative_score.max(dim=1).values
            safe = torch.as_tensor(baseline_correct[index], device=device)
            if bool(safe.any()):
                safety_deficit = F.relu(
                    initial_margin[safe].detach() - args.safety_slack - current_margin[safe]
                )
                local_introduced = torch.as_tensor(
                    action_introduced[index][baseline_correct[index]], device=device,
                )
                safety_weight = torch.where(
                    local_introduced,
                    torch.full_like(safety_deficit, introduced_weight),
                    torch.ones_like(safety_deficit),
                )
                safety = torch.sum(safety_weight * safety_deficit) / safety_weight.sum()
            else:
                safety = rank_loss * 0.0
            all_current = torch.cat((q, p.reshape(-1, p.shape[-1]), n.reshape(-1, n.shape[-1])))
            all_initial = torch.cat((q0, p0.reshape(-1, p0.shape[-1]), n0.reshape(-1, n0.shape[-1])))
            preserve = (1.0 - torch.sum(all_current * all_initial, dim=1)).mean()
            loss = rank_loss + args.safety_weight * safety + args.preserve_weight * preserve
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += np.asarray([
                float(loss.detach()), float(rank_loss.detach()), float(safety.detach()),
                float(preserve.detach()), float(grad_norm), len(index),
            ]) * len(index)
        last = total[:5] / total[5]
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 5 == 0:
            print(
                f"[molecule-max route={args.route} fold={fold} epoch={epoch + 1}] "
                f"loss={last[0]:.5f} rank={last[1]:.5f} safety={last[2]:.5f} "
                f"preserve={last[3]:.5f} grad={last[4]:.3f}", flush=True,
            )
    assert last is not None
    return model, {
        "final_loss": float(last[0]), "final_rank_loss": float(last[1]),
        "final_safety_loss": float(last[2]), "final_preserve_loss": float(last[3]),
        "final_gradient_norm": float(last[4]),
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.introduced_control_weight is not None and args.introduced_control_weight < 0:
        raise ValueError("introduced_control_weight must be non-negative")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    cache_rows = np.load(args.cache_dir / "rows.npy")
    embeddings = np.load(args.cache_dir / "official_embeddings_f32.npy", mmap_mode="r")
    cache_position = {int(row): index for index, row in enumerate(cache_rows)}

    train_query = load_s3a_queries(
        args.ledger, args.include_a4_training, args.a4_dir / "scan_queries.csv.gz",
    )
    train_query = train_query.loc[train_query.query_row.isin(cache_position)].reset_index(drop=True)
    original_s3a_queries = int(len(train_query))
    with np.load(args.manifest) as manifest_header:
        manifest_query_rows = set(map(int, manifest_header["query_row"]))
    train_query = train_query.loc[
        train_query.query_row.astype(int).isin(manifest_query_rows)
    ].reset_index(drop=True)
    missing_s3a_queries = original_s3a_queries - int(len(train_query))
    train_query, train_positive, train_negatives = manifest_representatives(
        train_query, args.manifest, embeddings, cache_rows, args.top_negatives,
        args.spectra_per_molecule,
    )
    train_query["formula_fold"] = train_query.query_formula.astype(str).map(
        lambda value: stable_fold(value, args.folds, args.formula_fold_seed)
    ).astype(np.int8)
    train_positions = np.asarray(
        [cache_position[int(row)] for row in train_query.query_row], dtype=np.int64
    )
    weights = route_weights(train_query, args)

    test_query = pd.read_csv(args.a4_dir / "scan_queries.csv.gz")
    test_query = test_query.loc[test_query.query_row.isin(cache_position)].reset_index(drop=True)
    test_query, _, _, candidates, candidate_identities = build_candidate_rows(
        test_query, cache_rows, embeddings, args.data, args.top_negatives,
    )
    exact = test_query.rebuilt_rank.eq(test_query.baseline_rank) & test_query.baseline_margin_error.le(1e-5)
    test_query = test_query.loc[exact].reset_index(drop=True)
    candidates = [value for value, flag in zip(candidates, exact) if flag]
    candidate_identities = [value for value, flag in zip(candidate_identities, exact) if flag]
    # The evaluation ledger does not enter training routing.  This column is
    # present only because the shared evaluator emits a stratification field.
    test_query["action_correctable"] = False
    test_query["action_positive_advantage"] = 0.0
    test_query["formula_fold"] = test_query.query_formula.astype(str).map(
        lambda value: stable_fold(value, args.folds, args.formula_fold_seed)
    ).astype(np.int8)
    test_positions = np.asarray(
        [cache_position[int(row)] for row in test_query.query_row], dtype=np.int64
    )

    records: list[dict[str, object]] = []
    logs: list[dict[str, object]] = []
    for fold in range(args.folds):
        held_formulas = set(test_query.loc[test_query.formula_fold.eq(fold), "query_formula"])
        train = np.flatnonzero(~train_query.query_formula.isin(held_formulas).to_numpy())
        test = np.flatnonzero(test_query.formula_fold.to_numpy() == fold)
        if set(train_query.iloc[train].query_formula) & held_formulas:
            raise RuntimeError(f"formula leakage in fold {fold}")
        model, log = train_molecule_max_fold(
            train, embeddings, train_positions, train_positive, train_negatives,
            train_query, weights, args, fold,
        )
        transformed = transform_numpy(model, embeddings, torch.device(args.device), 512)
        records.extend(evaluate_fold(
            test, test_query, transformed[test_positions], transformed, embeddings,
            test_positions, candidates, candidate_identities,
        ))
        logs.append({
            "fold": fold, "train_queries": int(len(train)),
            "held_queries": int(len(test)), "held_formulas": int(len(held_formulas)), **log,
        })
    result = pd.DataFrame(records).sort_values("query_index").reset_index(drop=True)
    metrics = formula_ci(result, args.bootstrap_resamples, args.seed)
    metrics.update({
        "queries": int(len(result)), "formulas": int(result.query_formula.nunique()),
        "baseline_errors": int(result.baseline_rank.gt(1).sum()),
        "corrected": int(result.corrected.sum()), "introduced": int(result.introduced.sum()),
        "preservation_mean": float(result.embedding_cosine.mean()),
        "preservation_p01": float(result.embedding_cosine.quantile(0.01)),
    })
    report = {
        "status": "noise_s3a_direct_boundary_transfer_complete",
        "metrics": metrics,
        "folds": logs,
        "training_pool": {
            "ledger_queries_before_manifest_intersection": original_s3a_queries,
            "queries_missing_from_candidate_manifest": missing_s3a_queries,
            "queries": int(len(train_query)),
            "formulas": int(train_query.query_formula.nunique()),
            "action_correctable_queries": int(train_query.action_correctable.sum()),
            "mature_n_correctable_queries": int(train_query.mature_n_correctable.sum()),
            "mature_n_introduced_queries": int(train_query.mature_n_introduced.sum()),
            "introduced_queries": int(train_query.action_introduced.sum()),
            "a4_training_requested": bool(args.include_a4_training),
            "a4_query_rows_restored": bool(args.include_a4_training),
        },
        "pass_4pp_local_reachability": bool(metrics["delta_pp"] >= 4.0),
        "pass_positive_formula_ci": bool(metrics["ci_low_pp"] > 0),
        "contracts": {
            "outer_formula_exclusion": True,
            "shared_query_and_candidate_map": True,
            "teacher_embedding_or_margin_target_used": False,
            "action_outcome_used_only_for_training_dose": bool(args.route != "clean_uniform"),
            "mature_n_route_excludes_role_shared": bool(args.route == "mature_n"),
            "identity_is_only_ranking_label": True,
            "evaluation_action_outcomes_enter_training": False,
            "a4_missing_query_row_restored_from_frozen_scan_table": bool(
                args.include_a4_training
            ),
            "bounded_per_spectrum_residual": True,
            "training_molecule_score_is_spectrum_max": True,
            "training_spectra_per_molecule": int(args.spectra_per_molecule),
            "clean_objective_is_current_molecule_boundary": bool(
                args.loss_kind == "molecule_margin"
            ),
            "action_introduced_controls_receive_distinct_dose": bool(
                args.introduced_control_weight is not None
            ),
        },
        "claim_limit": "Formula-OOF frozen-embedding shared-map proxy; not a raw-spectrum encoder checkpoint or official promotion result.",
        "configuration": vars(args),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.output_dir / "query_results.csv.gz", index=False, compression="gzip")
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
