"""R2: transfer the faithful noise teacher into one shared DreaMS encoder.

This is the first stage in the final noise programme that changes the DreaMS
embedding itself.  It is deliberately candidate-independent at inference:
both query and reference spectra are encoded by the same model and inference
uses the clean spectrum only.

The training-only privileged actions from R1 are used in four ways:

* clean and action views must rank the true molecule above current hard
  negatives;
* the clean view is pulled toward the (better ranked) action view;
* S3A action margins must exceed two intensity/mz/role-matched controls;
* protected-correct queries retain their official margin and embedding.

P2b and every downstream reranker are forbidden.  Formula folds are frozen
before training.  The held fold is evaluated once after the fixed epoch count;
there is no held-fold checkpoint selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import (  # noqa: E402
    CandidateGraph, json_dump, load_embedding_cache, seed_everything,
    sha256_file, strict_rank,
)
from noise_v3_core import attenuate_sequence  # noqa: E402
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--r1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r1_privileged_teacher")
    parser.add_argument("--preflight-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r2_preflight")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/validation/g8r_noise_final_r2_shared_encoder")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-identities", type=int, default=4)
    parser.add_argument("--positive-spectra", type=int, default=4)
    parser.add_argument("--negative-molecules", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=5e-6)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lambda-action-rank", type=float, default=0.75)
    parser.add_argument("--lambda-transfer", type=float, default=0.50)
    parser.add_argument("--lambda-specificity", type=float, default=0.25)
    parser.add_argument("--specificity-margin", type=float, default=0.01)
    parser.add_argument("--lambda-robust", type=float, default=0.50)
    parser.add_argument("--lambda-protected", type=float, default=1.0)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--protected-margin-slack", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_path(value: object) -> tuple[int, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    tokens = tuple(int(part) for part in text.split(",") if part != "")
    if len(tokens) != len(set(tokens)):
        raise RuntimeError("an R1 peak path reuses a token")
    return tokens


def parse_controls(value: object) -> tuple[tuple[int, ...], ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    paths = tuple(parse_path(part) for part in text.split(";"))
    if len(paths) != 2 or any(not path for path in paths):
        raise RuntimeError("S3A actions require exactly two non-empty matched controls")
    return paths


def unfreeze_last_block(model) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    encoder = model.backbone.transformer_encoder
    layer = int(encoder.n_layers) - 1
    count = 0
    for module in (encoder.atts[layer], encoder.ffs[layer]):
        for parameter in module.parameters():
            parameter.requires_grad = True
            count += parameter.numel()
    for module in (encoder.scales[2 * layer], encoder.scales[2 * layer + 1]):
        for parameter in module.parameters():
            parameter.requires_grad = True
            count += parameter.numel()
    if getattr(encoder, "pre_norm", False):
        for parameter in encoder.scales[-1].parameters():
            parameter.requires_grad = True
            count += parameter.numel()
    return {
        "transformer_layers": int(encoder.n_layers),
        "unfrozen_layer": layer,
        "unfrozen_backbone_parameters": int(count),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
    }


class SpectrumStore:
    """All graph-reachable clean spectra in a small CPU tensor cache."""

    def __init__(self, data: Path, rows: np.ndarray, n_highest_peaks: int):
        self.rows = np.asarray(sorted(set(map(int, rows))), dtype=np.int64)
        self.position = {int(row): index for index, row in enumerate(self.rows)}
        tensors: list[torch.Tensor] = []
        with h5py.File(data, "r") as handle:
            for row in self.rows:
                tensors.append(preprocess_spectrum(
                    np.asarray(handle["spectrum"][int(row)]),
                    float(handle["precursor_mz"][int(row)]), n_highest_peaks,
                ))
        self.tensor = torch.stack(tensors)

    def get(self, rows: list[int] | tuple[int, ...] | np.ndarray) -> torch.Tensor:
        positions = [self.position[int(row)] for row in rows]
        return self.tensor[positions]

    def one(self, row: int) -> torch.Tensor:
        return self.tensor[self.position[int(row)]]


@dataclass(frozen=True)
class Example:
    query_index: int
    query_row: int
    identity: str
    formula: str
    positive_rows: tuple[int, ...]
    negative_rows: tuple[int, ...]
    target_path: tuple[int, ...] = ()
    control_paths: tuple[tuple[int, ...], ...] = ()
    attenuation: float = 0.0
    official_margin: float = 0.0


def representatives(graph: CandidateGraph, query: int, positives: int, negatives: int,
                    forced_negative: int | None = None) -> tuple[tuple[int, ...], tuple[int, ...]]:
    left, right = map(int, graph.query_ptr[query:query + 2])
    score_col = graph.dreams_column
    pos_left, pos_right = map(int, graph.molecule_ptr[left:left + 2])
    pos_order = np.argsort(-graph.features[pos_left:pos_right, score_col], kind="stable")
    positive_rows = tuple(map(int, graph.pair_candidate_row[pos_left:pos_right][pos_order[:positives]]))
    choices: list[tuple[float, int]] = []
    for molecule in range(left + 1, right):
        pair_left, pair_right = map(int, graph.molecule_ptr[molecule:molecule + 2])
        local = int(np.argmax(graph.features[pair_left:pair_right, score_col]))
        pair = pair_left + local
        choices.append((float(graph.features[pair, score_col]), int(graph.pair_candidate_row[pair])))
    choices.sort(key=lambda item: (-item[0], item[1]))
    negative_rows = [row for _, row in choices[:negatives]]
    if forced_negative is not None and int(forced_negative) >= 0 and int(forced_negative) not in negative_rows:
        negative_rows.append(int(forced_negative))
    if not positive_rows or not negative_rows:
        raise RuntimeError(f"query {query} lacks a positive or negative reference")
    return positive_rows, tuple(negative_rows)


def build_examples(graph: CandidateGraph, frame: pd.DataFrame, positives: int,
                   negatives: int, action: bool) -> list[Example]:
    output = []
    for row in frame.itertuples(index=False):
        forced = int(row.teacher_hard_negative_row) if hasattr(row, "teacher_hard_negative_row") else None
        pos, neg = representatives(graph, int(row.query_index), positives, negatives, forced)
        output.append(Example(
            query_index=int(row.query_index), query_row=int(row.query_row),
            identity=str(row.query_ik14), formula=str(row.query_formula),
            positive_rows=pos, negative_rows=neg,
            target_path=parse_path(row.target_path) if action else (),
            control_paths=parse_controls(row.matched_control_paths) if action else (),
            attenuation=float(row.attenuation) if action else 0.0,
            official_margin=float(row.baseline_margin) if hasattr(row, "baseline_margin") else 0.0,
        ))
    return output


def identity_epoch_sample(examples: list[Example], rng: np.random.Generator) -> list[Example]:
    by_identity: dict[str, list[Example]] = {}
    for example in examples:
        by_identity.setdefault(example.identity, []).append(example)
    sampled = [values[int(rng.integers(0, len(values)))] for values in by_identity.values()]
    rng.shuffle(sampled)
    return sampled


def forward_embeddings(model, spectra: torch.Tensor, amp: bool) -> torch.Tensor:
    with torch.autocast(device_type=spectra.device.type, dtype=torch.float16,
                        enabled=amp and spectra.device.type == "cuda"):
        return model(spectra)


def flatten_batch(store: SpectrumStore, examples: list[Example], include_actions: bool,
                  include_controls: bool) -> tuple[torch.Tensor, list[dict], list[int]]:
    tensors: list[torch.Tensor] = []
    clean_rows: list[int] = []
    layout: list[dict] = []
    for example in examples:
        item: dict[str, object] = {}
        item["clean"] = len(tensors)
        tensors.append(store.one(example.query_row))
        clean_rows.append(example.query_row)
        if include_actions:
            item["action"] = len(tensors)
            tensors.append(attenuate_sequence(store.one(example.query_row), example.target_path, example.attenuation))
            item["controls"] = []
            if include_controls:
                for path in example.control_paths:
                    item["controls"].append(len(tensors))
                    tensors.append(attenuate_sequence(store.one(example.query_row), path, example.attenuation))
        item["positive"] = list(range(len(tensors), len(tensors) + len(example.positive_rows)))
        tensors.extend(store.get(example.positive_rows))
        clean_rows.extend(example.positive_rows)
        item["negative"] = list(range(len(tensors), len(tensors) + len(example.negative_rows)))
        tensors.extend(store.get(example.negative_rows))
        clean_rows.extend(example.negative_rows)
        layout.append(item)
    return torch.stack(tensors), layout, clean_rows


def margins(encoded: torch.Tensor, layout: list[dict], view: str) -> torch.Tensor:
    values = []
    for item in layout:
        query = encoded[int(item[view])]
        positive = torch.max(encoded[item["positive"]] @ query)
        negative = torch.max(encoded[item["negative"]] @ query)
        values.append(positive - negative)
    return torch.stack(values)


def correction_loss(model, store: SpectrumStore, examples: list[Example], device: torch.device,
                    official_by_row: dict[int, np.ndarray], args) -> tuple[torch.Tensor, dict]:
    spectra, layout, clean_rows = flatten_batch(store, examples, True, True)
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    clean_margin = margins(encoded, layout, "clean")
    action_margin = margins(encoded, layout, "action")
    rank_clean = F.softplus((args.rank_margin - clean_margin) / args.temperature).mean()
    rank_action = F.softplus((args.rank_margin - action_margin) / args.temperature).mean()
    clean_z = torch.stack([encoded[int(item["clean"])] for item in layout])
    action_z = torch.stack([encoded[int(item["action"])] for item in layout])
    transfer = (1.0 - torch.sum(clean_z * action_z.detach(), dim=1)).mean()
    specificity_terms = []
    for index, item in enumerate(layout):
        controls = item["controls"]
        if controls:
            control_margin = []
            for control_index in controls:
                temp = dict(item)
                temp["control"] = int(control_index)
                control_margin.append(margins(encoded, [temp], "control")[0])
            specificity_terms.append(F.relu(
                torch.stack(control_margin).mean() + args.specificity_margin - action_margin[index]
            ))
    specificity = torch.stack(specificity_terms).mean() if specificity_terms else encoded.sum() * 0.0
    official = torch.from_numpy(np.stack([official_by_row[row] for row in clean_rows])).to(device)
    # clean_rows correspond to every clean/reference tensor but not action/control tensors.
    clean_indices = []
    for item in layout:
        clean_indices.append(int(item["clean"]))
        clean_indices.extend(item["positive"])
        clean_indices.extend(item["negative"])
    preserve = (1.0 - torch.sum(encoded[clean_indices] * official, dim=1)).mean()
    loss = (rank_clean + args.lambda_action_rank * rank_action
            + args.lambda_transfer * transfer
            + args.lambda_specificity * specificity
            + args.lambda_preserve * preserve)
    return loss, {
        "corr_clean_rank": float(rank_clean.detach()),
        "corr_action_rank": float(rank_action.detach()),
        "corr_transfer": float(transfer.detach()),
        "corr_specificity": float(specificity.detach()),
        "corr_preserve": float(preserve.detach()),
        "corr_clean_margin": float(clean_margin.mean().detach()),
        "corr_action_margin": float(action_margin.mean().detach()),
    }


def safety_loss(model, store: SpectrumStore, examples: list[Example], device: torch.device,
                official_by_row: dict[int, np.ndarray], args, robustness: bool) -> tuple[torch.Tensor, dict]:
    spectra, layout, clean_rows = flatten_batch(store, examples, robustness, False)
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    clean_margin = margins(encoded, layout, "clean")
    floors = torch.tensor(
        [example.official_margin - args.protected_margin_slack for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    floor_loss = F.relu(floors - clean_margin).mean()
    consistency = encoded.sum() * 0.0
    if robustness:
        clean_z = torch.stack([encoded[int(item["clean"])] for item in layout])
        action_z = torch.stack([encoded[int(item["action"])] for item in layout])
        consistency = (1.0 - torch.sum(clean_z * action_z, dim=1)).mean()
    official = torch.from_numpy(np.stack([official_by_row[row] for row in clean_rows])).to(device)
    clean_indices = []
    for item in layout:
        clean_indices.append(int(item["clean"]))
        clean_indices.extend(item["positive"])
        clean_indices.extend(item["negative"])
    preserve = (1.0 - torch.sum(encoded[clean_indices] * official, dim=1)).mean()
    weight = args.lambda_robust if robustness else args.lambda_protected
    loss = weight * (floor_loss + consistency) + args.lambda_preserve * preserve
    return loss, {
        ("robust_floor" if robustness else "protected_floor"): float(floor_loss.detach()),
        ("robust_consistency" if robustness else "protected_consistency"): float(consistency.detach()),
        ("robust_preserve" if robustness else "protected_preserve"): float(preserve.detach()),
    }


@torch.no_grad()
def encode_rows(model, store: SpectrumStore, rows: np.ndarray, device: torch.device,
                batch_size: int, amp: bool, label: str) -> np.ndarray:
    model.eval()
    output = np.empty((len(rows), model.head.out_features), dtype=np.float32)
    started = time.time()
    for left in range(0, len(rows), batch_size):
        right = min(left + batch_size, len(rows))
        batch_rows = rows[left:right]
        batch = forward_embeddings(
            model, store.get(batch_rows).to(device), amp,
        ).float().cpu().numpy()
        # DreaMS uses Fourier/mz features.  A zero-change audit must never
        # silently accept an fp16 overflow as a retrieval result.  Retry only
        # the affected batch in fp32, then fail closed with the exact HDF5 rows
        # if the model is still non-finite.
        invalid = ~np.all(np.isfinite(batch), axis=1)
        if np.any(invalid) and amp:
            bad_rows = batch_rows[invalid]
            print(
                f"[{label}] retrying {len(bad_rows)} non-finite rows in fp32: "
                f"{bad_rows[:20].tolist()}", flush=True,
            )
            batch[invalid] = forward_embeddings(
                model, store.get(bad_rows).to(device), False,
            ).float().cpu().numpy()
            invalid = ~np.all(np.isfinite(batch), axis=1)
        if np.any(invalid):
            raise RuntimeError(
                f"{label} produced non-finite embeddings for HDF5 rows "
                f"{batch_rows[invalid][:50].tolist()}"
            )
        norms = np.linalg.norm(batch, axis=1)
        if not np.all(np.isfinite(norms)) or np.any(np.abs(norms - 1.0) > 2e-3):
            bad = np.flatnonzero(~np.isfinite(norms) | (np.abs(norms - 1.0) > 2e-3))
            raise RuntimeError(
                f"{label} produced non-unit embeddings for HDF5 rows "
                f"{batch_rows[bad][:50].tolist()}; norms={norms[bad][:50].tolist()}"
            )
        output[left:right] = batch
        if right == len(rows) or right % (batch_size * 20) == 0:
            print(f"[{label}] {right:,}/{len(rows):,} rows; {time.time() - started:.0f}s", flush=True)
    return output


def evaluate_embeddings(graph: CandidateGraph, rows: np.ndarray, encoded: np.ndarray,
                        queries: np.ndarray) -> tuple[np.ndarray, dict]:
    bad_encoded = np.flatnonzero(~np.all(np.isfinite(encoded), axis=1))
    if len(bad_encoded):
        raise RuntimeError(
            "evaluation received non-finite embeddings for HDF5 rows "
            f"{rows[bad_encoded[:50]].tolist()}"
        )
    position = {int(row): index for index, row in enumerate(rows)}
    qpos = np.asarray([position[int(row)] for row in graph.query_row], dtype=np.int64)
    cpos = np.asarray([position[int(row)] for row in graph.pair_candidate_row], dtype=np.int64)
    molecule_query = np.repeat(np.arange(graph.n_queries), np.diff(graph.query_ptr))
    pair_query = np.repeat(molecule_query, np.diff(graph.molecule_ptr))
    pair_score = np.einsum("ij,ij->i", encoded[qpos[pair_query]], encoded[cpos])
    bad_pairs = np.flatnonzero(~np.isfinite(pair_score))
    if len(bad_pairs):
        raise RuntimeError(
            "evaluation produced non-finite pair scores at pair indices "
            f"{bad_pairs[:50].tolist()}"
        )
    molecule_score = np.maximum.reduceat(pair_score, graph.molecule_ptr[:-1])
    ranks = []
    for query in queries:
        left, right = map(int, graph.query_ptr[int(query):int(query) + 2])
        scores = molecule_score[left:right]
        if len(scores) < 2 or not np.all(np.isfinite(scores)):
            raise RuntimeError(
                f"invalid molecule scores for query={int(query)} "
                f"query_row={int(graph.query_row[int(query)])}: {scores.tolist()}"
            )
        ranks.append(strict_rank(scores))
    rank = np.asarray(ranks, dtype=np.int16)
    near = graph.query_has_near[queries]
    return rank, {
        "n_queries": int(len(rank)), "recall1": float(np.mean(rank == 1)),
        "mrr": float(np.mean(1.0 / rank)), "errors": int(np.sum(rank != 1)),
        "near_n": int(np.sum(near)),
        "near_recall1": float(np.mean(rank[near] == 1)) if np.any(near) else float("nan"),
    }


def formula_bootstrap_delta(old: np.ndarray, new: np.ndarray, formulas: np.ndarray,
                            resamples: int, seed: int) -> dict[str, float]:
    unique, inverse = np.unique(formulas.astype(str), return_inverse=True)
    effect = (new == 1).astype(float) - (old == 1).astype(float)
    sums = np.bincount(inverse, weights=effect)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(unique), len(unique))
        draws[index] = sums[sampled].sum() / counts[sampled].sum()
    return {
        "mean": float(np.mean(effect)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def formula_bootstrap_mean(values: np.ndarray, formulas: np.ndarray,
                           resamples: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    unique, inverse = np.unique(np.asarray(formulas, dtype=str), return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(unique), len(unique))
        draws[index] = sums[sampled].sum() / counts[sampled].sum()
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


@torch.no_grad()
def evaluate_action_views(model, store: SpectrumStore, examples: list[Example],
                          device: torch.device, args) -> dict:
    """Held-formula diagnostic; never used for checkpoint or recipe selection."""
    model.eval()
    action_minus_clean: list[float] = []
    action_minus_control: list[float] = []
    action_formulas: list[str] = []
    control_formulas: list[str] = []
    clean_correct = action_correct = 0
    batch_size = max(1, args.batch_identities * 2)
    for left in range(0, len(examples), batch_size):
        block = examples[left:left + batch_size]
        spectra, layout, _ = flatten_batch(store, block, True, True)
        encoded = forward_embeddings(model, spectra.to(device), args.amp)
        clean_margin = margins(encoded, layout, "clean")
        action_margin = margins(encoded, layout, "action")
        clean_correct += int(torch.sum(clean_margin > 0).cpu())
        action_correct += int(torch.sum(action_margin > 0).cpu())
        action_minus_clean.extend((action_margin - clean_margin).float().cpu().numpy().tolist())
        action_formulas.extend(example.formula for example in block)
        for index, item in enumerate(layout):
            controls = item["controls"]
            if not controls:
                continue
            values = []
            for control_index in controls:
                temp = dict(item)
                temp["control"] = int(control_index)
                values.append(margins(encoded, [temp], "control")[0])
            action_minus_control.append(float(
                action_margin[index] - torch.stack(values).mean()
            ))
            control_formulas.append(block[index].formula)
    output = {
        "queries": int(len(examples)),
        "clean_margin_positive": int(clean_correct),
        "action_margin_positive": int(action_correct),
        "action_minus_clean": formula_bootstrap_mean(
            np.asarray(action_minus_clean), np.asarray(action_formulas),
            args.bootstrap_resamples, args.seed + 17,
        ) if action_minus_clean else None,
        "matched_control_queries": int(len(action_minus_control)),
        "action_minus_matched_control": formula_bootstrap_mean(
            np.asarray(action_minus_control), np.asarray(control_formulas),
            args.bootstrap_resamples, args.seed + 23,
        ) if action_minus_control else None,
    }
    return output


def main() -> None:
    args = arguments()
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be 0..4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("R2 requires a CUDA device")
    if args.backbone_lr <= 0 or args.head_lr <= 0 or args.head_lr < args.backbone_lr:
        raise ValueError("R2 requires positive learning rates and head-lr >= backbone-lr")
    seed_everything(args.seed)
    device = torch.device(args.device)
    required = [
        args.graph, args.data, args.embedding_cache, args.official_checkpoint,
        args.architecture_checkpoint, args.r1_dir / "report.json",
        args.preflight_dir / "report.json",
        args.r1_dir / "corrective_teacher_actions.csv.gz",
        args.r1_dir / "robustness_teacher_actions.csv.gz",
        args.r1_dir / "query_ledger.csv.gz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    r1 = json.loads((args.r1_dir / "report.json").read_text(encoding="utf-8"))
    preflight = json.loads((args.preflight_dir / "report.json").read_text(encoding="utf-8"))
    if not r1.get("formal") or r1["contracts"].get("P2b") != "forbidden" or r1.get("locally_materialised_union_recoverable") != 882:
        raise RuntimeError("R2 requires the formal P2b-free 882-query R1 teacher")
    if preflight.get("status") != "noise_final_r2_preflight_passed" or not preflight.get("pass"):
        raise RuntimeError("R2 requires a passing formal preflight")
    if preflight["provenance"].get("graph_sha256") != sha256_file(args.graph):
        raise RuntimeError("R2 graph changed after preflight")
    if preflight["provenance"].get("r1_report_sha256") != sha256_file(args.r1_dir / "report.json"):
        raise RuntimeError("R1 changed after R2 preflight")

    tag = f"blr_{args.backbone_lr:.0e}_hlr_{args.head_lr:.0e}"
    output = args.output_root / tag / f"seed_{args.seed}" / f"fold_{args.outer_fold}"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite R2: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    graph = CandidateGraph(args.graph)
    corrective = pd.read_csv(args.r1_dir / "corrective_teacher_actions.csv.gz")
    robust = pd.read_csv(args.r1_dir / "robustness_teacher_actions.csv.gz")
    ledger = pd.read_csv(args.r1_dir / "query_ledger.csv.gz")
    if len(corrective) != 882 or len(ledger) != graph.n_queries:
        raise RuntimeError("R1 tables drifted")
    train_corrective = corrective.loc[corrective["formula_fold"].astype(int).ne(args.outer_fold)].copy()
    held_corrective = corrective.loc[corrective["formula_fold"].astype(int).eq(args.outer_fold)].copy()
    train_robust = robust.loc[robust["formula_fold"].astype(int).ne(args.outer_fold)].copy()
    if "baseline_margin" not in train_robust.columns:
        train_robust = train_robust.merge(
            ledger[["query_index", "baseline_margin"]], on="query_index", how="left",
            validate="one_to_one",
        )
    protected = ledger.loc[
        ledger["formula_fold"].astype(int).ne(args.outer_fold)
        & ledger["baseline_rank"].astype(int).eq(1)
    ].copy()
    held_queries = ledger.loc[ledger["formula_fold"].astype(int).eq(args.outer_fold), "query_index"].to_numpy(np.int64)
    if train_corrective["query_formula"].isin(set(ledger.loc[ledger["formula_fold"].astype(int).eq(args.outer_fold), "query_formula"])).any():
        raise RuntimeError("formula isolation failed")

    corr_examples = build_examples(graph, train_corrective, args.positive_spectra, args.negative_molecules, True)
    robust_examples = build_examples(graph, train_robust, args.positive_spectra, args.negative_molecules, True)
    protected_examples = build_examples(graph, protected, args.positive_spectra, args.negative_molecules, False)
    reachable_rows = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row))).astype(np.int64)
    spectrum_store = SpectrumStore(args.data, reachable_rows, args.n_highest_peaks)
    cache_rows, cache_embeddings, cache_index = load_embedding_cache(args.embedding_cache)
    if set(map(int, reachable_rows)) - set(cache_index):
        raise RuntimeError("official embedding cache does not cover the graph")
    official_by_row = {int(row): cache_embeddings[index] for row, index in cache_index.items() if int(row) in spectrum_store.position}

    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    capacity = unfreeze_last_block(model)
    # Crucial protocol: gradients remain enabled while dropout stays disabled.
    model.eval()
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": list(model.head.parameters()), "lr": args.head_lr, "weight_decay": args.weight_decay},
        {"params": backbone_parameters, "lr": args.backbone_lr, "weight_decay": 0.0},
    ])
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    # Zero-change gate: fresh official forward must reproduce cached official embeddings/ranks.
    # The zero-change gate is a numerical identity audit against an fp32
    # official cache.  It must not depend on the training AMP setting.
    initial_encoded = encode_rows(model, spectrum_store, reachable_rows, device,
                                  args.eval_batch_size, False, "R2-init-fp32")
    official_encoded = np.stack([official_by_row[int(row)] for row in reachable_rows])
    initial_preservation = np.einsum("ij,ij->i", initial_encoded, official_encoded)
    baseline_rank, baseline_summary = evaluate_embeddings(
        graph, reachable_rows, official_encoded, held_queries,
    )
    locked_baseline_rank = ledger.set_index("query_index").loc[
        held_queries, "baseline_rank"
    ].to_numpy(np.int16)
    locked_rank_mismatches = int(np.sum(locked_baseline_rank != baseline_rank))
    if locked_rank_mismatches / max(len(held_queries), 1) > 0.001:
        raise RuntimeError(
            f"symmetric official baseline drifted from R1 ledger: "
            f"{locked_rank_mismatches}/{len(held_queries)}"
        )
    initial_rank, _ = evaluate_embeddings(graph, reachable_rows, initial_encoded, held_queries)
    initial_rank_mismatches = int(np.sum(initial_rank != baseline_rank))
    if float(np.mean(initial_preservation)) < 0.9999 or initial_rank_mismatches / max(len(held_queries), 1) > 0.001:
        raise RuntimeError(
            f"official zero-change reproduction failed: cos={np.mean(initial_preservation):.6f} "
            f"rank mismatches={initial_rank_mismatches}/{len(held_queries)}"
        )
    del initial_encoded
    torch.cuda.empty_cache()

    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
        model.eval()
        correction_epoch = identity_epoch_sample(corr_examples, rng)
        robust_epoch = identity_epoch_sample(robust_examples, rng)
        protected_epoch = identity_epoch_sample(protected_examples, rng)
        if args.smoke:
            correction_epoch = correction_epoch[:8]
            robust_epoch = robust_epoch[:4]
            protected_epoch = protected_epoch[:8]
        steps = math.ceil(len(correction_epoch) / args.batch_identities)
        totals: dict[str, float] = {}
        started = time.time()
        for step in range(steps):
            left = step * args.batch_identities
            corr = correction_epoch[left:left + args.batch_identities]
            if not corr:
                continue
            safe_start = (step * args.batch_identities) % len(protected_epoch)
            safe = (protected_epoch + protected_epoch)[safe_start:safe_start + len(corr)]
            robust_batch: list[Example] = []
            if robust_epoch:
                rstart = (step * max(1, len(corr) // 2)) % len(robust_epoch)
                robust_batch = (robust_epoch + robust_epoch)[rstart:rstart + max(1, len(corr) // 2)]
            optimizer.zero_grad(set_to_none=True)
            corr_loss, corr_log = correction_loss(model, spectrum_store, corr, device, official_by_row, args)
            scaler.scale(corr_loss).backward()
            safe_loss, safe_log = safety_loss(model, spectrum_store, safe, device, official_by_row, args, False)
            scaler.scale(safe_loss).backward()
            total_value = float(corr_loss.detach()) + float(safe_loss.detach())
            robust_log = {}
            if robust_batch:
                rloss, robust_log = safety_loss(model, spectrum_store, robust_batch, device, official_by_row, args, True)
                scaler.scale(rloss).backward()
                total_value += float(rloss.detach())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            log = {"loss": total_value, **corr_log, **safe_log, **robust_log}
            for key, value in log.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            if (step + 1) % 20 == 0 or step + 1 == steps:
                print(f"[R2 epoch={epoch}] {step + 1}/{steps} loss={totals['loss']/(step+1):.5f}", flush=True)
        record = {key: value / steps for key, value in totals.items()}
        record.update({"epoch": epoch, "steps": steps, "seconds": time.time() - started})
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)

    # Formal retrieval metrics are always evaluated in fp32.  AMP is an
    # execution option for gradient steps, not part of the scoring protocol.
    final_encoded = encode_rows(model, spectrum_store, reachable_rows, device,
                                args.eval_batch_size, False, "R2-final-fp32")
    final_rank, final_summary = evaluate_embeddings(
        graph, reachable_rows, final_encoded, held_queries,
    )
    final_preservation = np.einsum("ij,ij->i", final_encoded, official_encoded)
    old_correct, new_correct = baseline_rank == 1, final_rank == 1
    near = graph.query_has_near[held_queries]
    delta_ci = formula_bootstrap_delta(
        baseline_rank, final_rank, graph.query_formula[held_queries],
        args.bootstrap_resamples, args.seed,
    )
    final_summary.update({
        "baseline_recall1": baseline_summary["recall1"],
        "delta_recall1": float(final_summary["recall1"] - baseline_summary["recall1"]),
        "baseline_mrr": baseline_summary["mrr"],
        "delta_mrr": float(final_summary["mrr"] - baseline_summary["mrr"]),
        "baseline_near_recall1": baseline_summary["near_recall1"],
        "delta_near_recall1": float(final_summary["near_recall1"] - baseline_summary["near_recall1"]),
        "corrected": int(np.sum(~old_correct & new_correct)),
        "introduced": int(np.sum(old_correct & ~new_correct)),
        "risk_net": int(np.sum(~old_correct & new_correct) - 2 * np.sum(old_correct & ~new_correct)),
        "preservation_mean": float(np.mean(final_preservation)),
        "preservation_p01": float(np.quantile(final_preservation, 0.01)),
        "formula_cluster_delta_recall1": delta_ci,
        "held_corrective_queries": int(len(held_corrective)),
    })
    held_corrective_mask = np.isin(
        held_queries, held_corrective["query_index"].to_numpy(np.int64),
    )
    if np.any(held_corrective_mask):
        final_summary.update({
            "held_corrective_baseline_accuracy": float(np.mean(baseline_rank[held_corrective_mask] == 1)),
            "held_corrective_student_accuracy": float(np.mean(final_rank[held_corrective_mask] == 1)),
            "held_corrective_corrected": int(np.sum(final_rank[held_corrective_mask] == 1)),
        })
    held_action_examples = build_examples(
        graph, held_corrective, args.positive_spectra, args.negative_molecules, True,
    )
    held_action_audit = evaluate_action_views(
        model, spectrum_store, held_action_examples, device, args,
    )
    gates = {
        "clean_recall_positive": bool(final_summary["delta_recall1"] > 0),
        "formula_ci_positive": bool(delta_ci["ci_low"] > 0),
        "corrected_gt_introduced": bool(final_summary["corrected"] > final_summary["introduced"]),
        "risk_net_positive": bool(final_summary["risk_net"] > 0),
        "near_nonnegative": bool(final_summary["delta_near_recall1"] >= 0),
        "mrr_nonnegative": bool(final_summary["delta_mrr"] >= 0),
        "preservation_mean_ge_0_995": bool(final_summary["preservation_mean"] >= 0.995),
    }
    checkpoint = {
        "status": "noise_final_r2_shared_dreams_encoder",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "initialization": initialization, "seed": args.seed, "outer_fold": args.outer_fold,
        "capacity": capacity, "P2b_used": False, "inference_clean_only": True,
    }
    output.mkdir(parents=False, exist_ok=False)
    torch.save(checkpoint, output / "final_shared_encoder.pt")
    decision = {
        "status": "noise_final_r2_shared_encoder_pilot_complete",
        "formal": not args.smoke,
        "configuration": {
            "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
            "epochs": 1 if args.smoke else args.epochs,
            "rank_margin": args.rank_margin, "temperature": args.temperature,
        },
        "capacity": capacity,
        "data": {
            "train_corrective_rows": int(len(train_corrective)),
            "train_corrective_identities": int(train_corrective["query_ik14"].nunique()),
            "train_corrective_formulas": int(train_corrective["query_formula"].nunique()),
            "held_corrective_rows": int(len(held_corrective)),
            "held_queries": int(len(held_queries)),
        },
        "zero_change_gate": {
            "preservation_mean": float(np.mean(initial_preservation)),
            "rank_mismatches": initial_rank_mismatches,
            "locked_ledger_rank_mismatches": locked_rank_mismatches,
        },
        "held_clean": final_summary, "held_action_diagnostic": held_action_audit,
        "gates": gates,
        "pass_to_multifold": bool(all(gates.values())), "history": history,
        "contracts": {
            "shared_query_reference_encoder": True,
            "last_transformer_block_and_official_head_trainable": True,
            "dropout_disabled_during_gradient_training": True,
            "dynamic_hard_negative_within_frozen_candidate_shortlist": True,
            "identity_equal_epoch_sampling": True,
            "formula_held_out": True,
            "inference_clean_spectrum_only": True,
            "P2b": "forbidden",
            "P3_consumed": False,
            "zero_change_and_final_evaluation_fp32": True,
            "training_amp_enabled": bool(args.amp and device.type == "cuda"),
        },
        "provenance": {
            "r1_report_sha256": sha256_file(args.r1_dir / "report.json"),
            "preflight_sha256": sha256_file(args.preflight_dir / "report.json"),
            "graph_sha256": sha256_file(args.graph),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "one held-formula pilot. The 3.853 pp value is a privileged action-space upper bound; "
            "only held clean retrieval in this report is student embedding performance."
        ),
    }
    json_dump(output / "decision.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
