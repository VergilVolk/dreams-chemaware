"""Formula-OOF local reachability test for residual-supervised peak injection.

The audit uses a frozen official token/embedding cache and reconstructs the A4
10-ppm candidate graph from raw HDF5 metadata.  Only queries whose official
rank and margin reproduce the immutable A4 baseline are retained.  On each
outer formula fold, an identity-initialized clean-spectrum peak gate is taught
the *training-fold* margin residual of the best exact A4 intervention.  The
gate never receives candidate features, peak roles, gradients, query labels or
held outcomes.  Candidate embeddings stay frozen so the result isolates the
query-side injection capacity; it is not a final shared-encoder result.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a4-dir", type=Path,
        default=Path("data/validation/g8r_noise_v3_a4_exact_peak_scan"),
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path("data/validation/chemaware_corrected_manifest_tokens_v1"),
        help="Frozen official DreaMS tokens/embeddings; no ChemAware labels are read.",
    )
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/noise_peak_gate_candidate_injection_20260905"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--sparse-projection-width", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--top-negatives", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-strength", type=float, default=0.50)
    parser.add_argument("--teacher-residual-cap", type=float, default=0.10)
    parser.add_argument("--rank-weight", type=float, default=0.10)
    parser.add_argument("--preserve-weight", type=float, default=0.25)
    parser.add_argument("--gate-weight", type=float, default=1e-4)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--adapter-type", choices=("peak_gate", "embedding_residual"),
        default="peak_gate",
    )
    parser.add_argument(
        "--evaluation-mode", choices=("formula_oof", "in_sample_capacity"),
        default="formula_oof",
    )
    return parser.parse_args()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=object)


def stable_fold(value: str, folds: int, seed: int) -> int:
    import hashlib
    payload = f"{seed}|{value}".encode("utf-8", "replace")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % folds


def take_h5(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    order = np.argsort(rows, kind="stable")
    result = np.asarray(dataset[rows[order]])
    return result[np.argsort(order, kind="stable")]


def build_candidate_rows(
    query: pd.DataFrame, rows: np.ndarray, embeddings: np.ndarray, data: Path,
    top_negatives: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    position = {int(row): index for index, row in enumerate(rows)}
    with h5py.File(data, "r") as handle:
        precursor = take_h5(handle["precursor_mz"], rows).astype(np.float64)
        adduct = decode(take_h5(handle["adduct"], rows))
        identity = np.asarray([value[:14] for value in decode(
            take_h5(handle["INCHIKEY"], rows)
        )], dtype=object)
    mass_order = np.argsort(precursor, kind="stable")
    sorted_mass = precursor[mass_order]
    positive = np.empty(len(query), dtype=np.int64)
    negatives = np.empty((len(query), top_negatives), dtype=np.int64)
    candidate_positions: list[np.ndarray] = []
    candidate_identities: list[np.ndarray] = []
    rebuilt_rank = np.empty(len(query), dtype=np.int16)
    rebuilt_margin = np.empty(len(query), dtype=np.float32)
    for output, item in enumerate(query.itertuples(index=False)):
        query_position = position[int(item.query_row)]
        tolerance = 10e-6 * precursor[query_position]
        left = np.searchsorted(sorted_mass, precursor[query_position] - tolerance)
        right = np.searchsorted(sorted_mass, precursor[query_position] + tolerance, side="right")
        candidates = mass_order[left:right]
        candidates = candidates[
            (candidates != query_position) & (adduct[candidates] == adduct[query_position])
        ]
        identities = np.unique(identity[candidates])
        if identity[query_position] not in identities or len(identities) < 2:
            raise RuntimeError(f"reconstructed graph lacks both classes for query {item.query_index}")
        query_embedding = np.asarray(embeddings[query_position], dtype=np.float32)
        molecule_scores = []
        representatives = []
        for molecule in identities:
            local = candidates[identity[candidates] == molecule]
            scores = np.asarray(embeddings[local], dtype=np.float32) @ query_embedding
            best = int(local[int(np.argmax(scores))])
            representatives.append(best)
            molecule_scores.append(float(np.max(scores)))
        molecule_scores = np.asarray(molecule_scores)
        representatives = np.asarray(representatives, dtype=np.int64)
        positive_local = int(np.flatnonzero(identities == identity[query_position])[0])
        negative_local = np.flatnonzero(identities != identity[query_position])
        negative_order = negative_local[np.argsort(-molecule_scores[negative_local], kind="stable")]
        chosen = representatives[negative_order[:top_negatives]]
        if len(chosen) < top_negatives:
            chosen = np.pad(chosen, (0, top_negatives - len(chosen)), mode="edge")
        positive[output] = representatives[positive_local]
        negatives[output] = chosen
        rebuilt_rank[output] = 1 + int(np.sum(
            molecule_scores[negative_local] >= molecule_scores[positive_local]
        ))
        rebuilt_margin[output] = float(
            molecule_scores[positive_local] - np.max(molecule_scores[negative_local])
        )
        candidate_positions.append(candidates)
        candidate_identities.append(identity[candidates])
    query = query.copy()
    query["rebuilt_rank"] = rebuilt_rank
    query["rebuilt_margin"] = rebuilt_margin
    query["baseline_margin_error"] = np.abs(query.baseline_margin - rebuilt_margin)
    return query, positive, negatives, candidate_positions, candidate_identities


def teacher_residual(a4_dir: Path, query: pd.DataFrame, cap: float) -> pd.DataFrame:
    with h5py.File(a4_dir / "exact_peak_scan.h5", "r") as handle:
        action_query = handle["action_query"][:].astype(np.int64)
        doses = json.loads(handle.attrs["attenuations_json"])
        margin = handle["result_margin"][:].reshape(-1, len(doses))
        rank = handle["result_rank"][:].reshape(-1, len(doses))
    best_margin = np.full(len(query), np.nan, dtype=np.float32)
    best_rank = np.full(len(query), -1, dtype=np.int16)
    by_scan = {int(value): np.flatnonzero(action_query == int(value)) for value in query.scan_position}
    for output, item in enumerate(query.itertuples(index=False)):
        action = by_scan[int(item.scan_position)]
        flat = int(np.argmax(margin[action].reshape(-1)))
        action_local, dose_local = divmod(flat, len(doses))
        best_margin[output] = margin[action[action_local], dose_local]
        best_rank[output] = rank[action[action_local], dose_local]
    query = query.copy()
    query["teacher_best_margin"] = best_margin
    query["teacher_best_rank"] = best_rank
    raw = best_margin - query.baseline_margin.to_numpy(np.float32)
    # Correct controls define an exact no-op/safety target.  Error queries keep
    # only positive residual capacity; harmful teacher moves have zero weight.
    is_error = query.baseline_rank.to_numpy(np.int64) > 1
    query["teacher_residual"] = np.where(
        is_error, np.clip(raw, 0.0, cap), 0.0,
    ).astype(np.float32)
    query["teacher_correctable"] = is_error & (best_rank == 1)
    return query


def sparse_token_projection(
    token_memmap: np.ndarray, cache_positions: np.ndarray, feature_dim: int,
    width: int, seed: int,
) -> tuple[np.ndarray, dict[str, list]]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, token_memmap.shape[2], size=(feature_dim, width))
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(feature_dim, width))
    output = np.empty((len(cache_positions), token_memmap.shape[1], feature_dim), dtype=np.float32)
    for left in range(0, len(cache_positions), 128):
        right = min(left + 128, len(cache_positions))
        block = np.asarray(token_memmap[cache_positions[left:right]], dtype=np.float32)
        projected = block[:, :, indices] * signs[None, None, :, :]
        output[left:right] = projected.sum(axis=-1) / math.sqrt(width)
    return output, {"indices": indices.tolist(), "signs": signs.tolist()}


class PeakGateAdapter(torch.nn.Module):
    def __init__(self, feature_dim: int, embedding_dim: int, hidden: int, strength: float):
        super().__init__()
        self.strength = float(strength)
        self.norm = torch.nn.LayerNorm(feature_dim)
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + 2, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
        )
        torch.nn.init.zeros_(self.scorer[-1].weight)
        torch.nn.init.zeros_(self.scorer[-1].bias)
        self.project = torch.nn.Linear(feature_dim, embedding_dim, bias=False)
        torch.nn.init.normal_(self.project.weight, std=0.02 / math.sqrt(feature_dim))

    def forward(
        self, base: torch.Tensor, token: torch.Tensor, mz: torch.Tensor,
        intensity: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        auxiliary = torch.stack((mz / 1000.0, intensity), dim=-1)
        logits = self.scorer(torch.cat((self.norm(token), auxiliary), dim=-1)).squeeze(-1)
        gate = 2.0 * torch.sigmoid(logits)
        gate = gate * valid.to(gate.dtype)
        weight = intensity * valid.to(intensity.dtype)
        base_pool = torch.sum(weight.unsqueeze(-1) * token, dim=1) / (
            weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        )
        gated_weight = weight * gate
        gated_pool = torch.sum(gated_weight.unsqueeze(-1) * token, dim=1) / (
            gated_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        )
        delta = self.project(gated_pool - base_pool)
        return F.normalize(base + self.strength * delta, dim=-1), gate


class EmbeddingResidualAdapter(torch.nn.Module):
    """Zero-initialized low-rank diagnostic for the clean embedding interface."""

    def __init__(self, embedding_dim: int, hidden: int, strength: float):
        super().__init__()
        self.strength = float(strength)
        self.norm = torch.nn.LayerNorm(embedding_dim)
        self.down = torch.nn.Linear(embedding_dim, hidden)
        self.up = torch.nn.Linear(hidden, embedding_dim)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        delta = self.up(F.gelu(self.down(self.norm(base))))
        return F.normalize(base + self.strength * delta, dim=-1)


def train_fold(
    train: np.ndarray, test: np.ndarray, arrays: dict[str, np.ndarray], args: argparse.Namespace,
    fold: int,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    device = torch.device(args.device)
    if args.adapter_type == "peak_gate":
        model: torch.nn.Module = PeakGateAdapter(
            args.feature_dim, arrays["base"].shape[1], args.hidden_dim, args.gate_strength,
        ).to(device)
    else:
        model = EmbeddingResidualAdapter(
            arrays["base"].shape[1], args.hidden_dim, args.gate_strength,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed + fold)
    history = []
    for epoch in range(args.epochs):
        order = rng.permutation(train)
        totals = np.zeros(5, dtype=np.float64)
        for left in range(0, len(order), args.batch_size):
            index = order[left:left + args.batch_size]
            base = torch.as_tensor(arrays["base"][index], device=device)
            token = torch.as_tensor(arrays["token"][index], device=device)
            mz = torch.as_tensor(arrays["mz"][index], device=device)
            intensity = torch.as_tensor(arrays["intensity"][index], device=device)
            valid = torch.as_tensor(arrays["valid"][index], device=device)
            positive = torch.as_tensor(arrays["positive"][index], device=device)
            negative = torch.as_tensor(arrays["negative"][index], device=device)
            teacher = torch.as_tensor(arrays["teacher"][index], device=device)
            if args.adapter_type == "peak_gate":
                output, gate = model(base, token, mz, intensity, valid)
            else:
                output = model(base)
                gate = torch.ones_like(intensity)
            current = torch.sum(output * positive, dim=1, keepdim=True) - torch.einsum(
                "bd,bkd->bk", output, negative,
            )
            initial = torch.sum(base * positive, dim=1, keepdim=True) - torch.einsum(
                "bd,bkd->bk", base, negative,
            )
            edge_weight = torch.softmax(-initial.detach() / 0.05, dim=1)
            residual = torch.sum(edge_weight * F.smooth_l1_loss(
                current - initial, teacher[:, None].expand_as(current),
                reduction="none", beta=0.005,
            ), dim=1).mean()
            active = teacher > 1e-6
            rank = (
                torch.sum(edge_weight[active] * F.softplus((0.01 - current[active]) / 0.05), dim=1).mean()
                if bool(active.any()) else residual * 0.0
            )
            preserve = (1.0 - torch.sum(output * base, dim=1)).mean()
            gate_regularizer = torch.sum(
                ((gate - 1.0) ** 2) * valid.to(gate.dtype)
            ) / valid.sum().clamp_min(1)
            loss = (
                residual + args.rank_weight * rank
                + args.preserve_weight * preserve + args.gate_weight * gate_regularizer
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += np.asarray([
                float(loss.detach()), float(residual.detach()), float(rank.detach()),
                float(preserve.detach()), float(gate_regularizer.detach()),
            ]) * len(index)
        history.append(totals / len(train))
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 12 == 0:
            print(
                f"[peak-gate fold={fold} epoch={epoch + 1}] "
                f"loss={history[-1][0]:.6f} residual={history[-1][1]:.6f} "
                f"preserve={history[-1][3]:.6f}", flush=True,
            )
    model.eval()
    pieces = []
    gate_deviation = []
    with torch.inference_mode():
        for left in range(0, len(test), args.batch_size):
            index = test[left:left + args.batch_size]
            base = torch.as_tensor(arrays["base"][index], device=device)
            if args.adapter_type == "peak_gate":
                output, gate = model(
                    base,
                    torch.as_tensor(arrays["token"][index], device=device),
                    torch.as_tensor(arrays["mz"][index], device=device),
                    torch.as_tensor(arrays["intensity"][index], device=device),
                    torch.as_tensor(arrays["valid"][index], device=device),
                )
            else:
                output = model(base)
                gate = torch.ones_like(torch.as_tensor(arrays["intensity"][index], device=device))
            pieces.append(output.cpu().numpy())
            valid = arrays["valid"][index]
            gate_deviation.append(np.abs(gate.cpu().numpy()[valid] - 1.0))
    return np.concatenate(pieces), {
        "final_loss": float(history[-1][0]),
        "final_residual_loss": float(history[-1][1]),
        "final_preservation_loss": float(history[-1][3]),
        "held_gate_abs_deviation": float(np.mean(np.concatenate(gate_deviation))),
    }


def evaluate_full(
    query: pd.DataFrame, output: np.ndarray, embeddings: np.ndarray,
    candidate_positions: list[np.ndarray], candidate_identities: list[np.ndarray],
    cache_rows: np.ndarray,
) -> pd.DataFrame:
    row_position = {int(row): index for index, row in enumerate(cache_rows)}
    records = []
    for local, (item, vector, candidates, identities) in enumerate(zip(
        query.itertuples(index=False), output, candidate_positions, candidate_identities,
    )):
        scores = np.asarray(embeddings[candidates], dtype=np.float32) @ vector
        unique = np.unique(identities)
        molecule = np.asarray([np.max(scores[identities == value]) for value in unique])
        q_identity = str(item.query_ik14)
        positive = molecule[unique == q_identity]
        wrong = molecule[unique != q_identity]
        rank = 1 + int(np.sum(wrong >= positive[0]))
        records.append({
            "query_index": int(item.query_index), "query_formula": str(item.query_formula),
            "baseline_rank": int(item.baseline_rank), "final_rank": rank,
            "teacher_residual": float(item.teacher_residual),
            "teacher_correctable": bool(item.teacher_correctable),
            "corrected": bool(item.baseline_rank > 1 and rank == 1),
            "introduced": bool(item.baseline_rank == 1 and rank > 1),
            "embedding_cosine": float(np.dot(vector, np.asarray(
                embeddings[row_position[int(item.query_row)]], dtype=np.float32
            ))),
        })
    return pd.DataFrame(records)


def formula_ci(frame: pd.DataFrame, repeats: int, seed: int) -> dict[str, float]:
    value = frame.corrected.to_numpy(float) - frame.introduced.to_numpy(float)
    grouped = pd.DataFrame({"formula": frame.query_formula, "value": value}).groupby(
        "formula"
    ).value.agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats)
    for index in range(repeats):
        take = rng.integers(0, len(grouped), len(grouped))
        boot[index] = 100.0 * sums[take].sum() / counts[take].sum()
    return {
        "delta_pp": float(100.0 * value.sum() / len(frame)),
        "ci_low_pp": float(np.quantile(boot, 0.025)),
        "ci_high_pp": float(np.quantile(boot, 0.975)),
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    cache_rows = np.load(args.cache_dir / "rows.npy")
    embeddings = np.load(args.cache_dir / "official_embeddings_f32.npy", mmap_mode="r")
    tokens = np.load(args.cache_dir / "tokens_f16.npy", mmap_mode="r")
    cache_position = {int(row): index for index, row in enumerate(cache_rows)}
    query = pd.read_csv(args.a4_dir / "scan_queries.csv.gz")
    query = query.loc[query.query_row.isin(cache_position)].reset_index(drop=True)
    query, positive_position, negative_position, candidates, candidate_identities = (
        build_candidate_rows(query, cache_rows, embeddings, args.data, args.top_negatives)
    )
    exact = (
        query.rebuilt_rank.eq(query.baseline_rank)
        & query.baseline_margin_error.le(1e-5)
    )
    query = query.loc[exact].reset_index(drop=True)
    positive_position = positive_position[exact]
    negative_position = negative_position[exact]
    candidates = [value for value, keep in zip(candidates, exact) if keep]
    candidate_identities = [value for value, keep in zip(candidate_identities, exact) if keep]
    query = teacher_residual(args.a4_dir, query, args.teacher_residual_cap)
    query["formula_fold"] = query.query_formula.astype(str).map(
        lambda value: stable_fold(value, args.folds, args.formula_fold_seed)
    ).astype(np.int8)
    query_cache = np.asarray([cache_position[int(row)] for row in query.query_row], dtype=np.int64)
    token, projection = sparse_token_projection(
        tokens, query_cache, args.feature_dim, args.sparse_projection_width, args.seed,
    )
    mz_memmap = np.load(args.cache_dir / "mz_f32.npy", mmap_mode="r")
    intensity_memmap = np.load(args.cache_dir / "intensity_f32.npy", mmap_mode="r")
    valid_memmap = np.load(args.cache_dir / "valid.npy", mmap_mode="r")
    arrays = {
        "base": np.asarray(embeddings[query_cache], dtype=np.float32),
        "token": token,
        "mz": np.asarray(mz_memmap[query_cache], dtype=np.float32),
        "intensity": np.asarray(intensity_memmap[query_cache], dtype=np.float32),
        "valid": np.asarray(valid_memmap[query_cache], dtype=bool),
        "positive": np.asarray(embeddings[positive_position], dtype=np.float32),
        "negative": np.asarray(embeddings[negative_position], dtype=np.float32),
        "teacher": query.teacher_residual.to_numpy(np.float32),
    }
    oof = np.full_like(arrays["base"], np.nan)
    logs = []
    if args.evaluation_mode == "formula_oof":
        for fold in range(args.folds):
            test = np.flatnonzero(query.formula_fold.to_numpy() == fold)
            train = np.flatnonzero(query.formula_fold.to_numpy() != fold)
            if set(query.iloc[train].query_formula) & set(query.iloc[test].query_formula):
                raise RuntimeError(f"formula leakage in fold {fold}")
            predicted, log = train_fold(train, test, arrays, args, fold)
            oof[test] = predicted
            logs.append({"fold": fold, "train_queries": len(train), "held_queries": len(test), **log})
        if not np.isfinite(oof).all():
            raise RuntimeError("OOF peak-gate embeddings are incomplete")
    else:
        all_queries = np.arange(len(query), dtype=np.int64)
        predicted, log = train_fold(all_queries, all_queries, arrays, args, 0)
        oof[:] = predicted
        logs.append({
            "fold": "in_sample_capacity", "train_queries": len(query),
            "held_queries": 0, **log,
        })
    result = evaluate_full(
        query, oof, embeddings, candidates, candidate_identities, cache_rows,
    )
    metrics = formula_ci(result, args.bootstrap_resamples, args.seed)
    metrics.update({
        "queries": int(len(result)),
        "formulas": int(result.query_formula.nunique()),
        "baseline_errors": int(result.baseline_rank.gt(1).sum()),
        "teacher_correctable": int(result.teacher_correctable.sum()),
        "corrected": int(result.corrected.sum()),
        "introduced": int(result.introduced.sum()),
        "preservation_mean": float(result.embedding_cosine.mean()),
        "preservation_p01": float(result.embedding_cosine.quantile(0.01)),
    })
    report = {
        "status": "noise_peak_gate_candidate_injection_audit_complete",
        "baseline_reproduction": {
            "A4_queries_cached": 4993,
            "rank_and_margin_exact_queries": int(len(query)),
            "margin_tolerance": 1e-5,
        },
        "metrics": metrics,
        "folds": logs,
        "pass_4pp_local_reachability": bool(metrics["delta_pp"] >= 4.0),
        "pass_positive_formula_ci": bool(metrics["ci_low_pp"] > 0),
        "contracts": {
            "formula_oof": bool(args.evaluation_mode == "formula_oof"),
            "in_sample_capacity_only": bool(args.evaluation_mode == "in_sample_capacity"),
            "identity_initialized": True,
            "clean_spectrum_tokens_only_at_inference": True,
            "candidate_features_enter_gate": False,
            "gradient_or_peak_role_enter_gate": False,
            "held_action_outcomes_enter_training": bool(
                args.evaluation_mode == "in_sample_capacity"
            ),
            "training_teacher_is_exact_A4_margin_residual": True,
            "candidate_embeddings_frozen": True,
        },
        "claim_limit": (
            "Local query-side in-sample capacity only; not generalization, a two-sided "
            "shared-encoder checkpoint, or an official full-graph result."
            if args.evaluation_mode == "in_sample_capacity" else
            "Local formula-OOF query-side reachability on the baseline-exact A4 cohort; "
            "not a two-sided shared-encoder checkpoint or official full-graph result."
        ),
        "configuration": vars(args),
        "sparse_projection": projection,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.output_dir / "oof_query_results.csv.gz", index=False, compression="gzip")
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(json.dumps({key: value for key, value in report.items() if key != "sparse_projection"}, indent=2, default=str))


if __name__ == "__main__":
    main()
