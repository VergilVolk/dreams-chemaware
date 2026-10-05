"""Fine-tune the shared DreaMS embedding with qualified ICEBERG peak actions.

The deployable model remains one candidate-independent spectrum encoder.  A
structure-conditioned intensity action is used only during training as a
same-identity counterfactual *query view*.  Both clean and action views are
ranked against the complete real-spectrum candidate list with the same DreaMS
encoder.  The legacy ``direct_dual_listwise`` objective contains no teacher
score, teacher embedding, clean/action consistency, official-embedding
preservation, or official-margin loss.  The default projected-guarded variant
adds clean-spectrum safety anchors, removes action-gradient components that
oppose the primary clean objective, and caps the surviving action norm.  It
still contains no ICEBERG score or embedding target.  At evaluation and
inference only one clean experimental spectrum exists.

``direct_action_delta_transfer`` is the post-audit successor.  It keeps the
qualified top-3 peak action fixed, measures that action's score displacement
with frozen official DreaMS, and makes only the live *clean* query inherit the
candidate-centred displacement.  It never forwards an action view through the
trainable encoder and never lets the chemical branch move candidate references.
"""
from __future__ import annotations

import argparse
import json
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
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_chemaware_iceberg_synthetic_embedding import peak_permute, synthetic_tensor  # noqa: E402
from chemaware_iceberg_direct_core import (  # noqa: E402
    ARMS, action_prediction_indices, assert_formula_disjoint,
    identity_balanced_positions, molecule_scores_from_pairs, rescue_positions,
    stable_formula_folds,
)
from chemaware_iceberg_peak_action_core import (  # noqa: E402
    apply_peak_action, differential_evidence, hard_negative_indices,
)
from chemaware_direct_training_core import (  # noqa: E402
    assert_no_distillation_objective,
    projected_guarded_auxiliary,
    qualified_action_positions,
    validate_action_role_matrix,
)
from chemaware_direct_action_core import ROLE_CODE  # noqa: E402
from chemaware_action_delta_transfer_core import (  # noqa: E402
    clean_inherits_action_delta_loss,
    optimizer_descent_geometry,
    role_calibrated_dose,
)
from chemaware_retrieval_graph import RetrievalGraph  # noqa: E402
from chemaware_shared_v2_core import paired_evaluation  # noqa: E402
from noise_final_core import sha256_file, strict_rank  # noqa: E402
from train_e1_identity import load_base_model, torch_load_compat  # noqa: E402
from train_noise_final_e4a_direct_augmentation import unfreeze_last_blocks  # noqa: E402
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows, forward_embeddings  # noqa: E402
from chemaware_frozen_prefix_cache import FrozenPrefixSpectrumStore  # noqa: E402


@dataclass(frozen=True)
class Example:
    query: int
    query_row: int
    identity: str
    formula: str
    candidate_rows: tuple[int, ...]
    molecule_ptr: tuple[int, ...]
    official_rank: int
    official_margin: float
    teacher_position: int | None = None
    action_role: int | None = None
    action_dose: float = 1.0
    action_weight: float = 1.0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1/graph.npz",
    )
    parser.add_argument(
        "--evaluation-manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
        help="Broad held-formula graph used for registered retrieval metrics.",
    )
    parser.add_argument(
        "--teacher-dir", type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_teacher_700_v1",
    )
    parser.add_argument(
        "--token-dir", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--architecture-checkpoint", type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument(
        "--initial-shared-checkpoint", type=Path, default=None,
        help="Optional fold-matched direct DreaMS checkpoint; default starts from official DreaMS.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_direct_action_views_v1",
    )
    parser.add_argument(
        "--training-objective",
        choices=(
            "direct_dual_listwise", "direct_guarded_listwise",
            "direct_pcgrad_guarded", "direct_projected_guarded",
            "direct_action_delta_transfer", "legacy_consistency",
        ),
        default="direct_projected_guarded",
    )
    parser.add_argument(
        "--action-bank", type=Path,
        default=ROOT / "data/validation/chemaware_direct_action_bank_v1",
        help="Passed action qualification output; mandatory for direct listwise objectives.",
    )
    parser.add_argument("--arm", choices=ARMS, default="correct_synthetic")
    parser.add_argument(
        "--action-kind", choices=("synthetic", "differential"), default="differential",
        help="Synthetic full-spectrum view or candidate-differential intensity action.",
    )
    parser.add_argument(
        "--differential-mode",
        choices=("signed_exp", "conflict_attenuate", "support_boost"),
        default="conflict_attenuate",
    )
    parser.add_argument("--action-strength", type=float, default=0.75)
    parser.add_argument("--action-top-k", type=int, default=10)
    parser.add_argument("--lambda-peak-contrast", type=float, default=0.0)
    parser.add_argument("--peak-contrast-k", type=int, default=5)
    parser.add_argument("--peak-contrast-margin", type=float, default=0.02)
    parser.add_argument(
        "--transfer-target", choices=("symmetric", "frozen_action"), default="symmetric",
        help="For frozen_action, clean embeddings chase the official encoder's fixed action embedding.",
    )
    parser.add_argument(
        "--action-scope", choices=("bank_qualified", "rescues", "teacher_correct"),
        default="bank_qualified",
        help="Use the qualified bank (direct) or a legacy ICEBERG selection rule.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260935)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--action-generation-seed", type=int, default=20260935,
        help=(
            "Frozen seed used by the qualified action bank to construct its "
            "peak-permuted control; this must not follow the optimizer seed."
        ),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-action-identities", type=int, default=512)
    parser.add_argument("--max-safety-identities", type=int, default=512)
    parser.add_argument("--batch-queries", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lambda-clean-rank", type=float, default=1.0)
    parser.add_argument("--lambda-action-rank", type=float, default=1.0)
    parser.add_argument(
        "--action-delta-alpha", type=float, default=0.5,
        help="Fraction of the frozen qualified peak-action effect inherited by clean queries.",
    )
    parser.add_argument(
        "--action-delta-huber", type=float, default=0.02,
        help="Smooth-L1 transition in cosine-score units for action-effect transfer.",
    )
    parser.add_argument("--lambda-consistency", type=float, default=0.0)
    parser.add_argument("--lambda-margin-floor", type=float, default=0.0)
    parser.add_argument("--lambda-preserve", type=float, default=0.0)
    parser.add_argument("--margin-floor-slack", type=float, default=0.005)
    parser.add_argument("--safety-stream-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--maximum-action-gradient-ratio", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument(
        "--max-eval-identities", type=int, default=2000,
        help="Maximum identity-balanced held queries; zero evaluates every held identity.",
    )
    parser.add_argument(
        "--frozen-prefix-cache", action=argparse.BooleanOptionalAction, default=True,
        help="Cache the frozen backbone prefix; the trainable final block/head still run every step.",
    )
    parser.add_argument("--prefix-cache-batch-size", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def official_rank_margin(graph: RetrievalGraph) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.empty(graph.n_queries, dtype=np.int16)
    margins = np.empty(graph.n_queries, dtype=np.float32)
    for query in range(graph.n_queries):
        scores = graph.official_molecule_scores(query)
        ranks[query] = strict_rank(scores)
        margins[query] = float(scores[0] - np.max(scores[1:]))
    return ranks, margins


def make_example(
    graph: RetrievalGraph, query: int, ranks: np.ndarray, margins: np.ndarray,
    teacher_position: int | None, action_role: int | None = None,
    action_dose: float = 1.0, action_weight: float = 1.0,
) -> Example:
    _, candidate_rows, local_ptr, _ = graph.query_block(query)
    return Example(
        query=int(query), query_row=int(graph.query_row[query]),
        identity=str(graph.query_ik14[query]), formula=str(graph.query_formula[query]),
        candidate_rows=tuple(map(int, candidate_rows)),
        molecule_ptr=tuple(map(int, local_ptr)), official_rank=int(ranks[query]),
        official_margin=float(margins[query]), teacher_position=teacher_position,
        action_role=action_role, action_dose=float(action_dose),
        action_weight=float(action_weight),
    )


class ActionStore:
    def __init__(
        self, teacher_dir: Path, selected: np.ndarray, query_ptr: np.ndarray,
        clean_store: SpectrumStore, graph: RetrievalGraph, seed: int,
        action_kind: str = "synthetic", differential_mode: str = "conflict_attenuate",
        action_strength: float = 0.75, action_top_k: int = 10,
    ):
        predictions = np.load(teacher_dir / "iceberg_predictions_f16.npy").astype(np.float32)
        if len(predictions) != int(query_ptr[-1]):
            raise RuntimeError("teacher predictions do not align with query_ptr")
        self.prediction = predictions
        self.permuted = peak_permute(predictions, seed + 41)
        self.index = {
            arm: action_prediction_indices(query_ptr, arm)
            for arm in ARMS if arm != "clean_duplicate"
        }
        self.precursor = np.asarray([
            float(clean_store.one(int(graph.query_row[query]))[0, 0]) for query in selected
        ], dtype=np.float32)
        self.action_kind = action_kind
        self.differential_mode = differential_mode
        self.action_strength = float(action_strength)
        self.action_top_k = int(action_top_k)
        score_file = np.load(teacher_dir / "scores_and_ranks.npz", allow_pickle=True)
        swapped = np.concatenate([
            np.roll(predictions[int(left):int(right)], 1, axis=0)
            for left, right in zip(query_ptr[:-1], query_ptr[1:])
        ])
        self.arm_prediction = {
            "correct_synthetic": predictions,
            "candidate_swapped": swapped,
            "peak_permuted": self.permuted,
        }
        self.hard_negative = {
            "correct_synthetic": hard_negative_indices(query_ptr, score_file["correct_score"]),
            "candidate_swapped": hard_negative_indices(
                query_ptr, score_file["candidate_swapped_score"],
            ),
            "peak_permuted": hard_negative_indices(query_ptr, score_file["peak_permuted_score"]),
        }
        self.query_ptr = np.asarray(query_ptr, dtype=np.int64)
        self.frozen_embedding: np.ndarray | None = None
        self.frozen_qualified_embedding: dict[int, np.ndarray] = {}

    def tensor(self, position: int, arm: str, clean: torch.Tensor) -> torch.Tensor:
        if arm == "clean_duplicate":
            return clean.clone()
        if self.action_kind == "differential":
            prediction = self.arm_prediction[arm]
            evidence = differential_evidence(
                prediction[int(self.query_ptr[position])],
                prediction[int(self.hard_negative[arm][position])],
                clean[1:, 0].numpy(),
            )
            return apply_peak_action(
                clean, evidence, self.differential_mode,
                self.action_strength, self.action_top_k,
            )
        index = int(self.index[arm][position])
        prediction = self.permuted[index] if arm == "peak_permuted" else self.prediction[index]
        return synthetic_tensor(prediction, float(self.precursor[position]), 100, 1000.0)

    def evidence(self, position: int, arm: str, clean: torch.Tensor) -> np.ndarray:
        if arm == "clean_duplicate":
            return np.zeros(len(clean) - 1, dtype=np.float32)
        prediction = self.arm_prediction[arm]
        return differential_evidence(
            prediction[int(self.query_ptr[position])],
            prediction[int(self.hard_negative[arm][position])],
            clean[1:, 0].numpy(),
        )


class ActionTensorSource:
    """Raw-tensor source used once to build an exact frozen-prefix cache."""

    def __init__(
        self, actions: ActionStore, selected: np.ndarray, graph: RetrievalGraph,
        clean_store: SpectrumStore, arm: str,
    ):
        self.rows = np.arange(len(selected), dtype=np.int64)
        self.position = {int(row): int(row) for row in self.rows}
        self.tensor = torch.stack([
            actions.tensor(
                position, arm, clean_store.one(int(graph.query_row[int(query)])),
            )
            for position, query in enumerate(selected)
        ])

    def get(self, rows) -> torch.Tensor:
        return self.tensor[np.asarray(rows, dtype=np.int64)]


class QualifiedActionTensorSource:
    """Materialise actions only for training folds admitted by the frozen bank."""

    def __init__(
        self, actions: ActionStore, teacher_positions: np.ndarray,
        selected: np.ndarray, graph: RetrievalGraph, clean_store: SpectrumStore,
        arm: str,
    ):
        self.rows = np.asarray(teacher_positions, dtype=np.int64)
        if self.rows.ndim != 1 or not len(self.rows) or len(np.unique(self.rows)) != len(self.rows):
            raise ValueError("qualified action positions must be a non-empty unique vector")
        self.position = {int(row): index for index, row in enumerate(self.rows)}
        self.tensor = torch.stack([
            actions.tensor(
                int(position), arm,
                clean_store.one(int(graph.query_row[int(selected[int(position)])])),
            )
            for position in self.rows
        ])

    def get(self, rows) -> torch.Tensor:
        index = np.asarray([self.position[int(row)] for row in rows], dtype=np.int64)
        return self.tensor[index]


def flatten_action(
    examples: list[Example], store: SpectrumStore, actions: ActionStore, arm: str,
) -> tuple[torch.Tensor, list[dict[str, object]], list[int], list[int]]:
    tensors: list[torch.Tensor] = []
    layout: list[dict[str, object]] = []
    real_rows: list[int] = []
    real_indices: list[int] = []
    for example in examples:
        clean_index = len(tensors)
        clean = store.one(example.query_row)
        tensors.append(clean); real_rows.append(example.query_row); real_indices.append(clean_index)
        action_index = len(tensors)
        if example.teacher_position is None:
            raise RuntimeError("action example lacks teacher position")
        tensors.append(actions.tensor(example.teacher_position, arm, clean))
        candidate_index = list(range(len(tensors), len(tensors) + len(example.candidate_rows)))
        tensors.extend(store.get(example.candidate_rows))
        real_rows.extend(example.candidate_rows); real_indices.extend(candidate_index)
        layout.append({
            "clean": clean_index, "action": action_index,
            "candidate": candidate_index,
            "molecule_ptr": np.asarray(example.molecule_ptr, dtype=np.int64),
        })
    return torch.stack(tensors), layout, real_rows, real_indices


def flatten_clean(
    examples: list[Example], store: SpectrumStore,
) -> tuple[torch.Tensor, list[dict[str, object]], list[int]]:
    tensors: list[torch.Tensor] = []
    layout: list[dict[str, object]] = []
    rows: list[int] = []
    for example in examples:
        clean_index = len(tensors)
        tensors.append(store.one(example.query_row)); rows.append(example.query_row)
        candidate_index = list(range(len(tensors), len(tensors) + len(example.candidate_rows)))
        tensors.extend(store.get(example.candidate_rows)); rows.extend(example.candidate_rows)
        layout.append({
            "clean": clean_index, "candidate": candidate_index,
            "molecule_ptr": np.asarray(example.molecule_ptr, dtype=np.int64),
        })
    return torch.stack(tensors), layout, rows


def list_scores(encoded: torch.Tensor, item: dict[str, object], view: str) -> torch.Tensor:
    query = encoded[int(item[view])]
    pair = encoded[item["candidate"]] @ query
    return molecule_scores_from_pairs(pair, item["molecule_ptr"])


def cached_action_embeddings(
    model, examples: list[Example], real_cache: FrozenPrefixSpectrumStore,
    action_cache: FrozenPrefixSpectrumStore, device: torch.device, args,
) -> tuple[torch.Tensor, list[dict[str, object]], list[int], list[int]]:
    real_rows: list[int] = []
    for example in examples:
        real_rows.append(example.query_row); real_rows.extend(example.candidate_rows)
    unique, inverse = np.unique(np.asarray(real_rows, dtype=np.int64), return_inverse=True)
    real_encoded = real_cache.forward(model, unique, device, args.eval_batch_size, args.amp)
    action_positions = np.asarray([int(value.teacher_position) for value in examples], dtype=np.int64)
    action_encoded = action_cache.forward(
        model, action_positions, device, args.eval_batch_size, args.amp,
    )
    output: list[torch.Tensor] = []
    layout: list[dict[str, object]] = []
    cursor = 0
    real_indices: list[int] = []
    preserve_rows: list[int] = []
    for index, example in enumerate(examples):
        clean_index = len(output); output.append(real_encoded[int(inverse[cursor])])
        real_indices.append(clean_index); preserve_rows.append(example.query_row); cursor += 1
        action_index = len(output); output.append(action_encoded[index])
        candidate_index = []
        for row in example.candidate_rows:
            candidate_index.append(len(output)); output.append(real_encoded[int(inverse[cursor])])
            real_indices.append(candidate_index[-1]); preserve_rows.append(row); cursor += 1
        layout.append({
            "clean": clean_index, "action": action_index, "candidate": candidate_index,
            "molecule_ptr": np.asarray(example.molecule_ptr, dtype=np.int64),
        })
    return torch.stack(output), layout, preserve_rows, real_indices


def cached_clean_embeddings(
    model, examples: list[Example], real_cache: FrozenPrefixSpectrumStore,
    device: torch.device, args,
) -> tuple[torch.Tensor, list[dict[str, object]], list[int]]:
    rows: list[int] = []
    for example in examples:
        rows.append(example.query_row); rows.extend(example.candidate_rows)
    unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
    values = real_cache.forward(model, unique, device, args.eval_batch_size, args.amp)
    encoded: list[torch.Tensor] = []
    layout: list[dict[str, object]] = []
    cursor = 0
    for example in examples:
        clean_index = len(encoded); encoded.append(values[int(inverse[cursor])]); cursor += 1
        candidate_index = []
        for _ in example.candidate_rows:
            candidate_index.append(len(encoded)); encoded.append(values[int(inverse[cursor])]); cursor += 1
        layout.append({
            "clean": clean_index, "candidate": candidate_index,
            "molecule_ptr": np.asarray(example.molecule_ptr, dtype=np.int64),
        })
    return torch.stack(encoded), layout, rows


def cached_final_tokens(
    model, rows: np.ndarray, cache: FrozenPrefixSpectrumStore,
    device: torch.device, batch_size: int, amp: bool,
) -> torch.Tensor:
    """Run the trainable suffix and retain every final token for local contrast."""
    positions = np.asarray([cache.position[int(row)] for row in rows], dtype=np.int64)
    output = []
    encoder = model.backbone.transformer_encoder
    dtype = next(model.parameters()).dtype
    for left in range(0, len(positions), batch_size):
        index = positions[left:left + batch_size]
        x = cache.prefix[index].to(device=device, dtype=dtype)
        mask = cache.padding_mask[index].to(device=device)
        projected = cache.graphormer_projected[index].to(device=device, dtype=dtype)
        bias = projected.unsqueeze(2) - projected.unsqueeze(1)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=bool(amp and device.type == "cuda"),
        ):
            for layer in cache.adapted_layers:
                x = encoder._layer_forward(layer, x, mask, bias)
            output.append(encoder.scales[-1](x))
    return torch.cat(output, dim=0)


def peak_contrast_loss(
    model, examples: list[Example], store: SpectrumStore, actions: ActionStore,
    arm: str, cache: FrozenPrefixSpectrumStore, device: torch.device, args,
) -> torch.Tensor:
    rows = np.asarray([example.query_row for example in examples], dtype=np.int64)
    tokens = cached_final_tokens(
        model, rows, cache, device, args.eval_batch_size, args.amp,
    )
    losses = []
    for index, example in enumerate(examples):
        clean = store.one(example.query_row)
        evidence = actions.evidence(int(example.teacher_position), arm, clean)
        valid = np.flatnonzero((clean[1:, 0].numpy() > 0) & (clean[1:, 1].numpy() > 0))
        if len(valid) < 2:
            continue
        k = min(args.peak_contrast_k, len(valid) // 2)
        order = valid[np.argsort(evidence[valid], kind="stable")]
        conflict = torch.as_tensor(order[:k] + 1, device=device)
        support = torch.as_tensor(order[-k:] + 1, device=device)
        precursor = F.normalize(tokens[index, 0].float(), dim=-1)
        peak = F.normalize(tokens[index, 1:].float(), dim=-1)
        support_score = (peak[support - 1] @ precursor).mean()
        conflict_score = (peak[conflict - 1] @ precursor).mean()
        losses.append(F.relu(args.peak_contrast_margin - support_score + conflict_score))
    return torch.stack(losses).mean() if losses else tokens.sum() * 0.0


def action_delta_transfer_loss(
    model, examples: list[Example], store: SpectrumStore, actions: ActionStore,
    official_by_row: dict[int, np.ndarray], device: torch.device, args,
    real_cache: FrozenPrefixSpectrumStore | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    """Transfer a frozen qualified action effect into live clean queries.

    The ordinary clean listwise and preservation terms still use the shared
    live query/reference encoder.  Only the chemical term uses frozen official
    references, so the action effect cannot be absorbed by candidate movement.
    """
    if not actions.frozen_qualified_embedding:
        raise RuntimeError("qualified frozen action targets were not initialized")
    if real_cache is None:
        spectra, layout, real_rows = flatten_clean(examples, store)
        encoded = forward_embeddings(model, spectra.to(device), args.amp)
    else:
        encoded, layout, real_rows = cached_clean_embeddings(
            model, examples, real_cache, device, args,
        )
    clean_ce, chemical, floors, target_norm, student_norm = [], [], [], [], []
    weights = []
    for example, item in zip(examples, layout):
        if example.teacher_position is None or example.action_role not in (
            ROLE_CODE["corrective_rank"], ROLE_CODE["corrective_margin"],
        ):
            raise RuntimeError("action-delta transfer received an unqualified action")
        if not 0 < example.action_dose <= 1 or not 0 < example.action_weight:
            raise RuntimeError("qualified action has an invalid dose or formula weight")
        live_clean = encoded[int(item["clean"])]
        clean_scores = list_scores(encoded, item, "clean")
        clean_ce.append(-F.log_softmax(clean_scores / args.temperature, dim=0)[0])
        clean_margin = clean_scores[0] - torch.max(clean_scores[1:])
        floors.append(F.relu(
            example.official_margin - args.margin_floor_slack - clean_margin
        ))
        official_clean = torch.from_numpy(
            np.asarray(official_by_row[example.query_row], dtype=np.float32)
        ).to(device)
        official_action = torch.from_numpy(
            actions.frozen_qualified_embedding[int(example.teacher_position)]
        ).to(device)
        official_reference = torch.from_numpy(np.stack([
            official_by_row[int(row)] for row in example.candidate_rows
        ]).astype(np.float32, copy=False)).to(device)
        value, audit = clean_inherits_action_delta_loss(
            live_clean,
            official_clean,
            official_action,
            official_reference,
            np.asarray(example.molecule_ptr, dtype=np.int64),
            alpha=args.action_delta_alpha,
            dose=example.action_dose,
            huber_beta=args.action_delta_huber,
        )
        chemical.append(value)
        target_norm.append(torch.linalg.vector_norm(audit["dosed_target"]))
        student_norm.append(torch.linalg.vector_norm(audit["student_delta"]))
        weights.append(example.action_weight)
    weight = torch.as_tensor(weights, device=device, dtype=encoded.dtype)
    # Global formula weights have mean one.  Dividing by batch size, rather
    # than re-normalising inside each batch, preserves exact formula-equal mass
    # over a complete coverage-first epoch.
    denominator = int(args.batch_queries)
    if not 1 <= len(examples) <= denominator:
        raise RuntimeError("action batch exceeds its registered fixed denominator")
    clean_value = torch.sum(torch.stack(clean_ce) * weight) / denominator
    chemical_value = torch.sum(torch.stack(chemical) * weight) / denominator
    floor_value = torch.sum(torch.stack(floors) * weight) / denominator
    official = torch.from_numpy(np.stack([
        official_by_row[row] for row in real_rows
    ]).astype(np.float32, copy=False)).to(device)
    preserve_value = (1.0 - torch.sum(encoded * official, dim=1)).mean()
    zero = encoded.sum() * 0.0
    components = {
        "clean_full_list": clean_value,
        "action_full_list": chemical_value,
        "action_delta_transfer": chemical_value,
        "action_target_norm": torch.stack(target_norm).mean(),
        "action_student_delta_norm": torch.stack(student_norm).mean(),
        "symmetric_consistency": zero,
        "margin_floor": floor_value,
        "preserve": preserve_value,
        "peak_contrast": zero,
    }
    loss = (
        args.lambda_clean_rank * clean_value
        + args.lambda_action_rank * chemical_value
        + args.lambda_margin_floor * floor_value
        + args.lambda_preserve * preserve_value
    )
    return loss, components, {
        key: float(value.detach()) for key, value in components.items()
    }


def action_loss(
    model, examples: list[Example], store: SpectrumStore, actions: ActionStore,
    arm: str, official_by_row: dict[int, np.ndarray], device: torch.device, args,
    real_cache: FrozenPrefixSpectrumStore | None = None,
    action_cache: FrozenPrefixSpectrumStore | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    if args.training_objective == "direct_action_delta_transfer":
        return action_delta_transfer_loss(
            model, examples, store, actions, official_by_row, device, args,
            real_cache,
        )
    if real_cache is None:
        spectra, layout, real_rows, real_indices = flatten_action(examples, store, actions, arm)
        encoded = forward_embeddings(model, spectra.to(device), args.amp)
    else:
        if action_cache is None:
            raise RuntimeError("real prefix cache requires an action prefix cache")
        encoded, layout, real_rows, real_indices = cached_action_embeddings(
            model, examples, real_cache, action_cache, device, args,
        )
    direct = args.training_objective.startswith("direct_")
    clean_ce, action_ce, consistency, floors = [], [], [], []
    for example, item in zip(examples, layout):
        clean_scores = list_scores(encoded, item, "clean")
        action_scores = list_scores(encoded, item, "action")
        clean_ce.append(-F.log_softmax(clean_scores / args.temperature, dim=0)[0])
        action_ce.append(-F.log_softmax(action_scores / args.temperature, dim=0)[0])
        clean_z, action_z = encoded[int(item["clean"])], encoded[int(item["action"])]
        if not direct:
            if args.transfer_target == "frozen_action":
                if actions.frozen_embedding is None:
                    raise RuntimeError("frozen action targets were not initialized")
                fixed = torch.from_numpy(
                    actions.frozen_embedding[int(example.teacher_position)]
                ).to(device)
                consistency.append(1.0 - torch.sum(clean_z * fixed))
            else:
                consistency.append(1.0 - torch.sum(clean_z * action_z))
        clean_margin = clean_scores[0] - torch.max(clean_scores[1:])
        floors.append(F.relu(example.official_margin - args.margin_floor_slack - clean_margin))
    zero = encoded.sum() * 0.0
    target = torch.from_numpy(np.stack([official_by_row[row] for row in real_rows])).to(device)
    preserve = 1.0 - torch.sum(encoded[real_indices] * target, dim=1)
    consistency_value = zero if direct else torch.stack(consistency).mean()
    floor_value = torch.stack(floors).mean()
    preserve_value = preserve.mean()
    components = {
        "clean_full_list": torch.stack(clean_ce).mean(),
        "action_full_list": torch.stack(action_ce).mean(),
        "symmetric_consistency": consistency_value,
        "margin_floor": floor_value,
        "preserve": preserve_value,
    }
    if args.lambda_peak_contrast > 0:
        if real_cache is None:
            raise RuntimeError("peak contrast currently requires --frozen-prefix-cache")
        components["peak_contrast"] = peak_contrast_loss(
            model, examples, store, actions, arm, real_cache, device, args,
        )
    else:
        components["peak_contrast"] = encoded.sum() * 0.0
    loss = (
        args.lambda_clean_rank * components["clean_full_list"]
        + args.lambda_action_rank * components["action_full_list"]
        + args.lambda_consistency * components["symmetric_consistency"]
        + args.lambda_margin_floor * components["margin_floor"]
        + args.lambda_preserve * components["preserve"]
        + args.lambda_peak_contrast * components["peak_contrast"]
    )
    return loss, components, {key: float(value.detach()) for key, value in components.items()}


def evaluate_action_delta_fit(
    model, examples: list[Example], store: SpectrumStore, actions: ActionStore,
    arm: str, official_by_row: dict[int, np.ndarray], device: torch.device, args,
    real_cache: FrozenPrefixSpectrumStore | None,
    action_cache: FrozenPrefixSpectrumStore | None,
) -> dict[str, float]:
    """Evaluate complete qualified-action target fit without optimizer updates."""
    if args.training_objective != "direct_action_delta_transfer":
        raise ValueError("action-delta fit is defined only for the transfer objective")
    weighted_huber_numerator = 0.0
    formula_weight_mass = 0.0
    target_norm_sum = 0.0
    student_norm_sum = 0.0
    observations = 0
    with torch.no_grad():
        for left in range(0, len(examples), args.batch_queries):
            batch = examples[left:left + args.batch_queries]
            _loss, _tensors, components = action_loss(
                model, batch, store, actions, arm, official_by_row, device, args,
                real_cache, action_cache,
            )
            weighted_huber_numerator += (
                components["action_delta_transfer"] * args.batch_queries
            )
            formula_weight_mass += sum(example.action_weight for example in batch)
            target_norm_sum += components["action_target_norm"] * len(batch)
            student_norm_sum += components["action_student_delta_norm"] * len(batch)
            observations += len(batch)
    if observations != len(examples) or formula_weight_mass <= 0:
        raise RuntimeError("complete action-delta fit did not cover its registered panel")
    values = {
        "qualified_actions": int(observations),
        "formula_weighted_huber": float(
            weighted_huber_numerator / formula_weight_mass
        ),
        "mean_target_delta_l2": float(target_norm_sum / observations),
        "mean_student_delta_l2": float(student_norm_sum / observations),
    }
    if not all(np.isfinite(float(value)) for value in values.values()):
        raise RuntimeError("action-delta fit contains a non-finite value")
    return values


def safety_loss(
    model, examples: list[Example], store: SpectrumStore,
    official_by_row: dict[int, np.ndarray], device: torch.device, args,
    real_cache: FrozenPrefixSpectrumStore | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    if real_cache is None:
        spectra, layout, rows = flatten_clean(examples, store)
        encoded = forward_embeddings(model, spectra.to(device), args.amp)
    else:
        encoded, layout, rows = cached_clean_embeddings(model, examples, real_cache, device, args)
    direct = args.training_objective.startswith("direct_")
    ce, floors = [], []
    for example, item in zip(examples, layout):
        scores = list_scores(encoded, item, "clean")
        ce.append(-F.log_softmax(scores / args.temperature, dim=0)[0])
        margin = scores[0] - torch.max(scores[1:])
        floors.append(F.relu(example.official_margin - args.margin_floor_slack - margin))
    target = torch.from_numpy(np.stack([official_by_row[row] for row in rows])).to(device)
    preserve_value = (1.0 - torch.sum(encoded * target, dim=1)).mean()
    floor_value = torch.stack(floors).mean()
    components = {
        "safety_full_list": torch.stack(ce).mean(),
        "safety_margin_floor": floor_value,
        "safety_preserve": preserve_value,
    }
    loss = (
        components["safety_full_list"]
        + args.lambda_margin_floor * components["safety_margin_floor"]
        + args.lambda_preserve * components["safety_preserve"]
    )
    return loss, components, {key: float(value.detach()) for key, value in components.items()}


def formula_bootstrap(delta: np.ndarray, formulas: np.ndarray, seed: int, draws: int) -> dict:
    unique = np.unique(formulas)
    macro = np.asarray([np.mean(delta[formulas == value]) for value in unique], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.asarray([
        np.mean(macro[rng.integers(0, len(macro), len(macro))]) for _ in range(draws)
    ])
    return {
        "formula_macro_delta": float(np.mean(macro)),
        "formula_cluster_bootstrap_95ci": [
            float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975)),
        ],
        "formula_clusters": int(len(unique)), "draws": int(draws),
    }


def add_registered_rank_metrics(
    evaluation: dict, graph: RetrievalGraph, queries: np.ndarray,
) -> dict:
    """Attach the registered retrieval metrics to a paired evaluation."""
    old_rank = np.asarray(evaluation["old_rank"], dtype=np.int64)
    new_rank = np.asarray(evaluation["new_rank"], dtype=np.int64)
    queries = np.asarray(queries, dtype=np.int64)
    candidate_count = np.diff(graph.query_ptr)[queries].astype(np.int64)
    if not (len(old_rank) == len(new_rank) == len(candidate_count)):
        raise RuntimeError("paired ranks and candidate counts are not aligned")
    if np.any(candidate_count < 2) or np.any(old_rank > candidate_count) or np.any(new_rank > candidate_count):
        raise RuntimeError("retrieval ranks lie outside their candidate lists")
    summary = evaluation["summary"]
    for k in (5, 10, 20, 50):
        baseline = float(np.mean(old_rank <= k))
        current = float(np.mean(new_rank <= k))
        summary[f"baseline_recall{k}"] = baseline
        summary[f"recall{k}"] = current
        summary[f"delta_recall{k}"] = current - baseline
    denominator = candidate_count - 1
    old_auc = (candidate_count - old_rank) / denominator
    new_auc = (candidate_count - new_rank) / denominator
    summary.update({
        "baseline_macro_auc": float(np.mean(old_auc)),
        "macro_auc": float(np.mean(new_auc)),
        "delta_macro_auc": float(np.mean(new_auc - old_auc)),
        "baseline_micro_auc": float(np.sum(candidate_count - old_rank) / np.sum(denominator)),
        "micro_auc": float(np.sum(candidate_count - new_rank) / np.sum(denominator)),
        "delta_micro_auc": float(np.sum(old_rank - new_rank) / np.sum(denominator)),
    })
    evaluation["candidate_count"] = candidate_count
    return evaluation


def identity_balanced_manifest_queries(
    queries: np.ndarray, identities: np.ndarray, seed: int, limit: int,
) -> np.ndarray:
    grouped: dict[str, list[int]] = {}
    for query in np.asarray(queries, dtype=np.int64):
        grouped.setdefault(str(identities[int(query)]), []).append(int(query))
    keys = np.asarray(sorted(grouped), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    if limit > 0:
        keys = keys[:limit]
    selected = [
        grouped[str(key)][int(rng.integers(0, len(grouped[str(key)])))]
        for key in keys
    ]
    return np.asarray(selected, dtype=np.int64)


def manifest_evaluation_rows(body: dict[str, np.ndarray], queries: np.ndarray) -> np.ndarray:
    blocks = [np.asarray(body["query_row"])[np.asarray(queries, dtype=np.int64)]]
    for query in np.asarray(queries, dtype=np.int64):
        left, right = map(int, body["query_ptr"][query:query + 2])
        edge_left = int(body["molecule_ptr"][left])
        edge_right = int(body["molecule_ptr"][right])
        blocks.append(np.asarray(body["pair_candidate_row"])[edge_left:edge_right])
    return np.unique(np.concatenate(blocks)).astype(np.int64)


def evaluate_manifest_subset(
    body: dict[str, np.ndarray], queries: np.ndarray,
    official: np.ndarray, row_position: dict[int, int],
    encoded_rows: np.ndarray, encoded: np.ndarray,
) -> dict:
    local = {int(row): index for index, row in enumerate(encoded_rows)}
    old_rank, new_rank, old_margin, new_margin, candidate_count = [], [], [], [], []
    for query in np.asarray(queries, dtype=np.int64):
        query_row = int(body["query_row"][query])
        qold = np.asarray(official[row_position[query_row]])
        qnew = encoded[local[query_row]]
        left, right = map(int, body["query_ptr"][query:query + 2])
        old_score, new_score = [], []
        for molecule in range(left, right):
            rleft, rright = map(int, body["molecule_ptr"][molecule:molecule + 2])
            candidate_rows = np.asarray(body["pair_candidate_row"])[rleft:rright]
            old_reference = np.asarray([
                row_position[int(row)] for row in candidate_rows
            ], dtype=np.int64)
            new_reference = np.asarray([
                local[int(row)] for row in candidate_rows
            ], dtype=np.int64)
            old_score.append(float(np.max(np.asarray(official[old_reference]) @ qold)))
            new_score.append(float(np.max(encoded[new_reference] @ qnew)))
        old_score = np.asarray(old_score)
        new_score = np.asarray(new_score)
        old_rank.append(strict_rank(old_score)); new_rank.append(strict_rank(new_score))
        old_margin.append(float(old_score[0] - np.max(old_score[1:])))
        new_margin.append(float(new_score[0] - np.max(new_score[1:])))
        candidate_count.append(len(old_score))
    old_rank = np.asarray(old_rank, dtype=np.int64)
    new_rank = np.asarray(new_rank, dtype=np.int64)
    old_margin = np.asarray(old_margin, dtype=np.float32)
    new_margin = np.asarray(new_margin, dtype=np.float32)
    candidate_count = np.asarray(candidate_count, dtype=np.int64)
    denominator = candidate_count - 1
    old_ok, new_ok = old_rank == 1, new_rank == 1
    summary = {
        "n_queries": int(len(queries)),
        "baseline_recall1": float(np.mean(old_ok)),
        "recall1": float(np.mean(new_ok)),
        "delta_recall1": float(np.mean(new_ok) - np.mean(old_ok)),
        "baseline_mrr": float(np.mean(1.0 / old_rank)),
        "mrr": float(np.mean(1.0 / new_rank)),
        "delta_mrr": float(np.mean(1.0 / new_rank - 1.0 / old_rank)),
        "baseline_macro_auc": float(np.mean((candidate_count - old_rank) / denominator)),
        "macro_auc": float(np.mean((candidate_count - new_rank) / denominator)),
        "delta_macro_auc": float(np.mean((old_rank - new_rank) / denominator)),
        "baseline_micro_auc": float(np.sum(candidate_count - old_rank) / np.sum(denominator)),
        "micro_auc": float(np.sum(candidate_count - new_rank) / np.sum(denominator)),
        "delta_micro_auc": float(np.sum(old_rank - new_rank) / np.sum(denominator)),
        "corrected": int(np.sum(~old_ok & new_ok)),
        "introduced": int(np.sum(old_ok & ~new_ok)),
        "delta_mean_margin": float(np.mean(new_margin - old_margin)),
        "preservation_mean": float(np.mean(np.sum(
            encoded * np.asarray(official[[row_position[int(row)] for row in encoded_rows]]),
            axis=1,
        ))),
    }
    for k in (5, 10, 20, 50):
        summary[f"baseline_recall{k}"] = float(np.mean(old_rank <= k))
        summary[f"recall{k}"] = float(np.mean(new_rank <= k))
        summary[f"delta_recall{k}"] = float(np.mean(new_rank <= k) - np.mean(old_rank <= k))
    return {
        "query": np.asarray(queries, dtype=np.int64),
        "old_rank": old_rank, "new_rank": new_rank,
        "old_margin": old_margin, "new_margin": new_margin,
        "candidate_count": candidate_count, "summary": summary,
    }


def validate_cli_contract(args: argparse.Namespace) -> None:
    """Reject invalid pilot settings before any data or model allocation."""
    if args.folds < 3:
        raise ValueError("at least three formula folds are required")
    if not 0 <= args.inner_fold < args.folds or not 0 <= args.outer_fold < args.folds:
        raise ValueError("inner-fold and outer-fold must be in [0, folds)")
    if args.inner_fold == args.outer_fold:
        raise ValueError("inner and outer folds must differ")
    positive_integer = {
        "batch-queries": args.batch_queries,
        "eval-batch-size": args.eval_batch_size,
        "prefix-cache-batch-size": args.prefix_cache_batch_size,
        "n-highest-peaks": args.n_highest_peaks,
        "unfreeze-blocks": args.unfreeze_blocks,
        "bootstrap-draws": args.bootstrap_draws,
        "torch-threads": args.torch_threads,
    }
    invalid_integer = {name: value for name, value in positive_integer.items() if value < 1}
    if invalid_integer:
        raise ValueError(f"positive integer options required: {invalid_integer}")
    finite_scalar = {
        "backbone-lr": args.backbone_lr,
        "head-lr": args.head_lr,
        "weight-decay": args.weight_decay,
        "temperature": args.temperature,
        "grad-clip": args.grad_clip,
        "margin-floor-slack": args.margin_floor_slack,
    }
    invalid_scalar = {
        name: value for name, value in finite_scalar.items()
        if not np.isfinite(value)
    }
    if invalid_scalar:
        raise ValueError(f"finite scalar options required: {invalid_scalar}")
    if args.temperature <= 0 or args.grad_clip <= 0 or args.weight_decay < 0:
        raise ValueError("temperature and grad-clip must be positive; weight-decay must be non-negative")
    if args.head_lr < args.backbone_lr or args.backbone_lr <= 0:
        raise ValueError("require head-lr >= backbone-lr > 0")
    if args.max_action_identities < 1 or args.max_action_identities > 512:
        raise ValueError("direct pilot permits 1-512 action identities")
    if args.max_safety_identities < 1 or args.max_safety_identities > 2048:
        raise ValueError("direct pilot permits 1-2048 safety identities")
    if args.max_eval_identities < 0:
        raise ValueError("max-eval-identities must be non-negative; zero means all identities")


def validate_teacher_arrays(
    graph: RetrievalGraph,
    selected: np.ndarray,
    query_ptr: np.ndarray,
    prediction_path: Path,
    teacher_values,
) -> None:
    """Validate every teacher-array boundary before constructing DreaMS."""
    if selected.ndim != 1 or query_ptr.ndim != 1 or len(selected) == 0:
        raise RuntimeError("teacher selection and query_ptr must be non-empty vectors")
    if len(query_ptr) != len(selected) + 1 or query_ptr[0] != 0 or np.any(np.diff(query_ptr) < 2):
        raise RuntimeError("teacher query_ptr is not a valid >=2-candidate partition")
    if np.any(selected < 0) or np.any(selected >= graph.n_queries):
        raise RuntimeError("teacher selected_queries contains an out-of-range graph query")
    if len(np.unique(selected)) != len(selected):
        raise RuntimeError("teacher selected_queries contains duplicate graph queries")
    expected_width = np.diff(graph.query_ptr)[selected]
    if not np.array_equal(np.diff(query_ptr), expected_width):
        raise RuntimeError("teacher query_ptr candidate counts do not match the retrieval graph")
    prediction = np.load(prediction_path, mmap_mode="r")
    if prediction.ndim != 2 or prediction.shape[0] != int(query_ptr[-1]):
        raise RuntimeError("teacher predictions do not align with query_ptr")
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("teacher predictions contain non-finite values")
    per_query = ("official_rank", "official_margin", "correct_rank", "correct_margin")
    per_candidate = ("correct_score", "candidate_swapped_score", "peak_permuted_score")
    missing = set((*per_query, *per_candidate, "formula")) - set(teacher_values.files)
    if missing:
        raise RuntimeError(f"teacher score archive is missing arrays: {sorted(missing)}")
    if any(np.asarray(teacher_values[name]).shape != (len(selected),) for name in per_query):
        raise RuntimeError("teacher per-query score/rank arrays are not aligned")
    if any(np.asarray(teacher_values[name]).shape != (int(query_ptr[-1]),) for name in per_candidate):
        raise RuntimeError("teacher per-candidate score arrays are not aligned")
    if any(not np.all(np.isfinite(np.asarray(teacher_values[name]))) for name in (*per_query, *per_candidate)):
        raise RuntimeError("teacher score/rank arrays contain non-finite values")
    candidate_counts = np.diff(query_ptr)
    for name in ("official_rank", "correct_rank"):
        values = np.asarray(teacher_values[name])
        if np.any(values < 1) or np.any(values > candidate_counts):
            raise RuntimeError(f"teacher {name} lies outside its candidate list")
    if not np.array_equal(
        np.asarray(teacher_values["formula"]).astype(str), graph.query_formula[selected].astype(str),
    ):
        raise RuntimeError("teacher formulas do not match selected graph queries")


def validate_clean_data_and_embedding_cache(
    data_path: Path,
    all_rows: np.ndarray,
    cache_rows: np.ndarray,
    cache_embedding: np.ndarray,
) -> dict[int, np.ndarray]:
    """Prove HDF5/cache coverage and numerical validity before model construction."""
    if all_rows.ndim != 1 or len(all_rows) == 0 or len(np.unique(all_rows)) != len(all_rows):
        raise RuntimeError("required spectrum rows must be a non-empty unique vector")
    if (
        cache_rows.ndim != 1 or not np.issubdtype(cache_rows.dtype, np.integer)
        or len(cache_rows) != len(np.unique(cache_rows))
    ):
        raise RuntimeError("official embedding cache rows must be a unique vector")
    if (
        cache_embedding.ndim != 2 or len(cache_embedding) != len(cache_rows)
        or cache_embedding.shape[1] != 1024 or cache_embedding.dtype != np.float32
    ):
        raise RuntimeError("official embedding cache rows and matrix are not aligned")
    for left in range(0, len(cache_embedding), 8192):
        block = np.asarray(cache_embedding[left:left + 8192], dtype=np.float32)
        if not np.all(np.isfinite(block)):
            raise RuntimeError("official embedding cache contains non-finite values")
        norms = np.linalg.norm(block, axis=1)
        if np.any(np.abs(norms - 1.0) > 2e-3):
            raise RuntimeError("official embedding cache contains non-unit embeddings")
    cache_position = {int(row): index for index, row in enumerate(cache_rows)}
    missing = set(map(int, all_rows)) - set(cache_position)
    if missing:
        raise RuntimeError(f"official embedding cache misses graph-reachable rows: {sorted(missing)[:20]}")
    with h5py.File(data_path, "r") as handle:
        if "spectrum" not in handle or "precursor_mz" not in handle:
            raise RuntimeError("spectrum HDF5 lacks spectrum or precursor_mz")
        n_spectra = len(handle["spectrum"])
        if len(handle["precursor_mz"]) != n_spectra:
            raise RuntimeError("spectrum and precursor_mz HDF5 datasets are not aligned")
        if np.any(all_rows < 0) or int(np.max(all_rows)) >= n_spectra:
            raise RuntimeError("graph-reachable spectrum row is outside the HDF5 dataset")
    return {int(row): cache_embedding[cache_position[int(row)]] for row in all_rows}


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main() -> None:
    started = time.time()
    args = arguments()
    validate_cli_contract(args)
    direct = args.training_objective.startswith("direct_")
    delta_transfer = args.training_objective == "direct_action_delta_transfer"
    guarded_direct = args.training_objective in {
        "direct_guarded_listwise", "direct_pcgrad_guarded", "direct_projected_guarded",
        "direct_action_delta_transfer",
    }
    pcgrad_direct = args.training_objective in {
        "direct_pcgrad_guarded", "direct_projected_guarded",
    }
    if direct and not delta_transfer:
        assert_no_distillation_objective(
            action_kind=args.action_kind,
            transfer_target=args.transfer_target,
            lambda_consistency=args.lambda_consistency,
            lambda_margin_floor=args.lambda_margin_floor,
            lambda_preserve=args.lambda_preserve,
            lambda_peak_contrast=args.lambda_peak_contrast,
            allow_clean_regularizers=guarded_direct,
        )
    elif delta_transfer:
        if args.action_kind != "differential":
            raise ValueError("action-delta transfer requires observed-peak differential actions")
        if args.transfer_target != "symmetric":
            raise ValueError("action-delta transfer does not use a frozen embedding chase target")
        if args.lambda_consistency != 0 or args.lambda_peak_contrast != 0:
            raise ValueError("action-delta transfer forbids consistency and peak-token losses")
        if (
            not np.isfinite(args.action_delta_alpha)
            or not np.isfinite(args.action_delta_huber)
            or not 0 < args.action_delta_alpha <= 1
            or args.action_delta_huber <= 0
        ):
            raise ValueError("action-delta alpha and Huber transition are invalid")
        if args.initial_shared_checkpoint is not None:
            raise ValueError("action-delta Phase A must initialize from frozen official DreaMS")
    if direct:
        if args.action_scope != "bank_qualified":
            raise ValueError("direct listwise training requires --action-scope bank_qualified")
        if args.epochs < 1 or args.epochs > 2:
            raise ValueError("direct listwise training is capped at two pilot epochs")
        if min(args.lambda_clean_rank, args.lambda_action_rank, args.safety_stream_weight) <= 0:
            raise ValueError("all three direct listwise supervision streams must be active")
        if guarded_direct and min(args.lambda_margin_floor, args.lambda_preserve) <= 0:
            raise ValueError("guarded direct training requires positive margin-floor and preservation weights")
        if pcgrad_direct and not 0 < args.maximum_action_gradient_ratio <= 1:
            raise ValueError("PCGrad direct training requires maximum action gradient ratio in (0, 1]")
    elif args.action_scope == "bank_qualified":
        raise ValueError("legacy_consistency requires an explicit legacy action scope")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    required = [
        args.graph, args.evaluation_manifest, args.data,
        args.official_checkpoint, args.architecture_checkpoint,
        args.teacher_dir / "report.json", args.teacher_dir / "selected_queries.npy",
        args.teacher_dir / "query_ptr.npy", args.teacher_dir / "iceberg_predictions_f16.npy",
        args.teacher_dir / "scores_and_ranks.npz", args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy", args.token_dir / "report.json",
        args.evaluation_manifest.with_name("manifest.json"),
    ]
    if args.initial_shared_checkpoint is not None:
        required.append(args.initial_shared_checkpoint)
    if direct:
        required.extend((args.action_bank / "report.json", args.action_bank / "action_bank.npz"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    evaluation_manifest_sha256 = sha256_file(args.evaluation_manifest)
    official_checkpoint_sha256 = sha256_file(args.official_checkpoint)
    architecture_checkpoint_sha256 = sha256_file(args.architecture_checkpoint)
    manifest_report = json.loads(
        args.evaluation_manifest.with_name("manifest.json").read_text(encoding="utf-8")
    )
    token_report = json.loads((args.token_dir / "report.json").read_text(encoding="utf-8"))
    if (
        manifest_report.get("status") != "chemaware_corrected_candidate_manifest_built"
        or manifest_report.get("provenance", {}).get("manifest_sha256")
        != evaluation_manifest_sha256
        or token_report.get("status")
        != "chemaware_corrected_manifest_token_cache_complete"
        or token_report.get("provenance", {}).get("manifest_sha256")
        != evaluation_manifest_sha256
        or token_report.get("provenance", {}).get("official_checkpoint_sha256")
        != official_checkpoint_sha256
    ):
        raise RuntimeError(
            "evaluation manifest, official checkpoint, and embedding cache provenance drifted"
        )
    teacher_report = json.loads((args.teacher_dir / "report.json").read_text(encoding="utf-8"))
    if teacher_report.get("status") != "PASS" or not teacher_report.get("scope", {}).get("teacher_only"):
        raise RuntimeError("training requires a passed teacher-only ICEBERG audit")

    graph = RetrievalGraph(args.graph)
    with np.load(args.evaluation_manifest, allow_pickle=True) as loaded:
        evaluation_manifest = {key: loaded[key] for key in loaded.files}
    required_manifest = {
        "query_row", "query_ik14", "query_formula", "query_ptr",
        "molecule_ptr", "pair_candidate_row", "molecule_label",
    }
    if missing_manifest := required_manifest - set(evaluation_manifest):
        raise RuntimeError(f"evaluation manifest misses arrays: {sorted(missing_manifest)}")
    manifest_query_ptr = np.asarray(evaluation_manifest["query_ptr"], dtype=np.int64)
    manifest_molecule_ptr = np.asarray(evaluation_manifest["molecule_ptr"], dtype=np.int64)
    manifest_labels = np.asarray(evaluation_manifest["molecule_label"], dtype=np.int8)
    if (
        len(manifest_query_ptr) != len(evaluation_manifest["query_row"]) + 1
        or len(evaluation_manifest["query_ik14"]) != len(evaluation_manifest["query_row"])
        or len(evaluation_manifest["query_formula"]) != len(evaluation_manifest["query_row"])
        or manifest_query_ptr[0] != 0
        or manifest_query_ptr[-1] != len(manifest_labels)
        or manifest_molecule_ptr[0] != 0
        or manifest_molecule_ptr[-1] != len(evaluation_manifest["pair_candidate_row"])
        or np.any(np.diff(manifest_query_ptr) < 2)
        or np.any(np.diff(manifest_molecule_ptr) < 1)
    ):
        raise RuntimeError("evaluation manifest pointer topology is invalid")
    for left, right in zip(manifest_query_ptr[:-1], manifest_query_ptr[1:]):
        label = manifest_labels[int(left):int(right)]
        if label[0] != 1 or int(np.sum(label)) != 1:
            raise RuntimeError("evaluation manifest truth candidate is not unique and first")
    manifest_folds = stable_formula_folds(
        evaluation_manifest["query_formula"], args.folds, args.fold_seed,
    )
    broad_inner = identity_balanced_manifest_queries(
        np.flatnonzero(manifest_folds == args.inner_fold),
        evaluation_manifest["query_ik14"], args.seed + 817,
        args.max_eval_identities,
    )
    if not len(broad_inner):
        raise RuntimeError("broad held-formula evaluation contains zero queries")
    broad_eval_rows = manifest_evaluation_rows(evaluation_manifest, broad_inner)
    selected = np.load(args.teacher_dir / "selected_queries.npy").astype(np.int64)
    query_ptr = np.load(args.teacher_dir / "query_ptr.npy").astype(np.int64)
    teacher_values = np.load(args.teacher_dir / "scores_and_ranks.npz", allow_pickle=True)
    validate_teacher_arrays(
        graph, selected, query_ptr,
        args.teacher_dir / "iceberg_predictions_f16.npy", teacher_values,
    )
    selected_identity = graph.query_ik14[selected]
    if len(np.unique(selected_identity)) != len(selected):
        raise RuntimeError(
            "teacher panel identities are not globally unique; rebuild with the corrected selector"
        )
    official_teacher_rank = np.asarray(teacher_values["official_rank"], dtype=np.int16)
    correct_teacher_rank = np.asarray(teacher_values["correct_rank"], dtype=np.int16)
    teacher_values.close()
    selected_folds = stable_formula_folds(graph.query_formula[selected], args.folds, args.fold_seed)
    assert_formula_disjoint(
        graph.query_formula[selected], selected_folds, args.inner_fold, args.outer_fold,
    )
    rescue = rescue_positions(
        official_teacher_rank, correct_teacher_rank, selected_folds,
        args.inner_fold, args.outer_fold,
    )
    rescue = identity_balanced_positions(rescue, selected_identity, args.seed)
    if not direct and len(rescue) < 8:
        raise RuntimeError(f"insufficient train-fold ICEBERG rescues: {len(rescue)}")

    rank, margin = official_rank_margin(graph)
    graph_folds = stable_formula_folds(graph.query_formula, args.folds, args.fold_seed)
    assert_formula_disjoint(graph.query_formula, graph_folds, args.inner_fold, args.outer_fold)
    inner_selected = selected[selected_folds == args.inner_fold]
    inner_all = np.flatnonzero(graph_folds == args.inner_fold)
    outer_selected = selected[selected_folds == args.outer_fold]
    train_correct = np.flatnonzero(
        (graph_folds != args.inner_fold) & (graph_folds != args.outer_fold) & (rank == 1)
    )
    train_correct = identity_balanced_positions(train_correct, graph.query_ik14, args.seed + 71)
    train_correct = train_correct[:args.max_safety_identities]
    train_teacher = (selected_folds != args.inner_fold) & (selected_folds != args.outer_fold)
    action_bank_contract: dict[str, object] | None = None
    selected_action_role: np.ndarray | None = None
    selected_action_dose: np.ndarray | None = None
    selected_action_weight: np.ndarray | None = None
    if direct:
        bank_report = json.loads((args.action_bank / "report.json").read_text(encoding="utf-8"))
        if (
            bank_report.get("status") != "CHEMAWARE_DIRECT_ACTION_BANK_PASS"
            or not bank_report.get("pass_to_direct_action_pilot")
            or bank_report.get("formal_training_authorized")
        ):
            raise RuntimeError("direct training requires a passed pilot-only action bank")
        split_contract = bank_report.get("split", {})
        if (
            int(split_contract.get("folds", -1)) != args.folds
            or int(split_contract.get("fold_seed", -1)) != args.fold_seed
            or int(split_contract.get("embedding_evaluation_fold", -1)) != args.inner_fold
            or int(split_contract.get("reserve_fold", -1)) != args.outer_fold
        ):
            raise RuntimeError("action bank formula split does not match training evaluation roles")
        provenance = bank_report.get("provenance", {})
        if (
            provenance.get("graph_sha256") != sha256_file(args.graph)
            or provenance.get("teacher_report_sha256")
            != sha256_file(args.teacher_dir / "report.json")
            or provenance.get("action_bank_sha256")
            != sha256_file(args.action_bank / "action_bank.npz")
        ):
            raise RuntimeError("action bank provenance does not match graph, teacher, and bank payload")
        bank = np.load(args.action_bank / "action_bank.npz")
        required_bank_arrays = {
            "selected_query", "formula", "action_fold", "setting_mode",
            "setting_strength", "setting_top_k", "ranks", "margins",
            "role_code", "selected_setting", "direct_training_eligible",
        }
        missing_bank_arrays = required_bank_arrays - set(bank.files)
        if missing_bank_arrays:
            raise RuntimeError(f"action bank is missing arrays: {sorted(missing_bank_arrays)}")
        bank_selected = np.asarray(bank["selected_query"], dtype=np.int64)
        bank_fold = np.asarray(bank["action_fold"], dtype=np.int16)
        bank_formula = np.asarray(bank["formula"]).astype(str)
        eligible_mask = np.asarray(bank["direct_training_eligible"], dtype=bool)
        if (
            not np.array_equal(bank_selected, selected)
            or not np.array_equal(bank_fold, selected_folds)
            or not np.array_equal(bank_formula, graph.query_formula[selected].astype(str))
            or eligible_mask.shape != (len(selected),)
        ):
            raise RuntimeError("action bank formula identities or folds drifted")
        training_folds = tuple(split_contract.get("discovery_folds", [])) + (
            int(split_contract["confirmation_fold"]),
        )
        selected_setting = np.asarray(bank["selected_setting"])
        if selected_setting.shape != (1,):
            raise RuntimeError("action bank selected setting must be a scalar vector")
        chosen_for_integrity = int(selected_setting[0])
        role_code = np.asarray(bank["role_code"], dtype=np.int8)
        validate_action_role_matrix(
            role_code,
            chosen_for_integrity,
            bank_fold,
            tuple(map(int, split_contract.get("discovery_folds", []))),
            int(split_contract["confirmation_fold"]),
            tuple(ROLE_CODE.values()),
        )
        expected_eligible = np.isin(bank_fold, training_folds) & np.isin(
            role_code[chosen_for_integrity],
            (ROLE_CODE["corrective_rank"], ROLE_CODE["corrective_margin"]),
        )
        if not np.array_equal(eligible_mask, expected_eligible):
            raise RuntimeError("action bank eligibility mask is not derivable from its frozen roles")
        evaluation = np.isin(bank_fold, (args.inner_fold, args.outer_fold))
        ranks = np.asarray(bank["ranks"])
        margins = np.asarray(bank["margins"])
        if (
            ranks.shape != (len(role_code), 3, len(selected))
            or margins.shape != ranks.shape
        ):
            raise RuntimeError("action bank rank/margin cubes are not aligned")
        if np.any(ranks[:, :, evaluation] != -1) or np.any(np.isfinite(margins[:, :, evaluation])):
            raise RuntimeError("action bank contains forbidden evaluation-fold action outcomes")
        action_position = qualified_action_positions(
            selected,
            np.asarray(bank["selected_query"], dtype=np.int64),
            eligible_mask,
            selected_folds,
            args.inner_fold,
            args.outer_fold,
        )
        action_position = identity_balanced_positions(
            action_position, selected_identity, args.seed + 211,
        )[:args.max_action_identities]
        if len(action_position) < 8:
            raise RuntimeError(
                f"qualified action bank has too few training identities: {len(action_position)}"
            )
        chosen = chosen_for_integrity
        modes = np.asarray(bank["setting_mode"]).astype(str)
        strengths = np.asarray(bank["setting_strength"], dtype=np.float64)
        top_ks = np.asarray(bank["setting_top_k"], dtype=np.int64)
        if not (len(modes) == len(strengths) == len(top_ks)):
            raise RuntimeError("action bank setting arrays are not aligned")
        if chosen < 0 or chosen >= len(modes):
            raise RuntimeError("action bank selected setting is invalid")
        bank_mode = str(modes[chosen])
        bank_strength = float(strengths[chosen])
        bank_top_k = int(top_ks[chosen])
        bank_generation_seed = bank_report.get("action_space", {}).get(
            "action_generation_seed"
        )
        if not isinstance(bank_generation_seed, int):
            raise RuntimeError("action bank lacks its frozen action-generation seed")
        # The immutable qualified bank, not an independently typed CLI value,
        # is authoritative for the action operator, dose, k and control seed.
        args.differential_mode = bank_mode
        args.action_strength = bank_strength
        args.action_top_k = bank_top_k
        args.action_generation_seed = bank_generation_seed
        selected_action_role = role_code[chosen_for_integrity].astype(np.int8, copy=True)
        selected_action_dose = role_calibrated_dose(
            selected_action_role,
            np.asarray(margins[chosen_for_integrity, 0], dtype=np.float64)
            - margin[selected].astype(np.float64),
            ROLE_CODE["corrective_rank"],
            ROLE_CODE["corrective_margin"],
        )
        selected_action_weight = np.zeros(len(selected), dtype=np.float32)
        chosen_formula = graph.query_formula[selected[action_position]].astype(str)
        _, formula_inverse, formula_count = np.unique(
            chosen_formula, return_inverse=True, return_counts=True,
        )
        formula_weight = 1.0 / formula_count[formula_inverse].astype(np.float64)
        formula_weight *= len(formula_weight) / formula_weight.sum()
        selected_action_weight[action_position] = formula_weight.astype(np.float32)
        if np.any(selected_action_dose[action_position] <= 0):
            raise RuntimeError("qualified action received a zero calibrated transfer dose")
        action_bank_contract = {
            "path": str(args.action_bank.resolve()),
            "report_sha256": sha256_file(args.action_bank / "report.json"),
            "bank_sha256": sha256_file(args.action_bank / "action_bank.npz"),
            "selected_setting": chosen,
            "mode": bank_mode,
            "strength": bank_strength,
            "top_k": bank_top_k,
            "action_generation_seed": bank_generation_seed,
            "eligible_training_actions": int(len(action_position)),
            "corrective_rank_actions": int(np.sum(
                selected_action_role[action_position] == ROLE_CODE["corrective_rank"]
            )),
            "corrective_margin_actions": int(np.sum(
                selected_action_role[action_position] == ROLE_CODE["corrective_margin"]
            )),
            "margin_role_dose_calibration": "median positive corrective-rank margin gain",
            "formula_equal_training_mass": True,
            "embedding_evaluation_actions_generated": False,
            "outer_actions_generated": False,
            "training_arm": args.arm,
            "training_arm_is_matched_control": args.arm != "correct_synthetic",
        }
    elif args.action_scope == "teacher_correct":
        action_position = np.flatnonzero(train_teacher & (correct_teacher_rank == 1))
        action_position = identity_balanced_positions(action_position, selected_identity, args.seed)
    else:
        action_position = rescue
    if delta_transfer:
        if any(value is None for value in (
            selected_action_role, selected_action_dose, selected_action_weight,
        )):
            raise RuntimeError("action bank did not produce transfer metadata")
        action_examples = [
            make_example(
                graph, int(selected[pos]), rank, margin, int(pos),
                action_role=int(selected_action_role[pos]),
                action_dose=float(selected_action_dose[pos]),
                action_weight=float(selected_action_weight[pos]),
            )
            for pos in action_position
        ]
        # Formula folds are hash-derived, but prove the actual selected rows are
        # disjoint rather than trusting only the fold labels.
        overlap = set(graph.query_formula[selected[action_position]].astype(str)) & set(
            evaluation_manifest["query_formula"][broad_inner].astype(str)
        )
        if overlap:
            raise RuntimeError(
                f"qualified action formulas leak into broad held evaluation: {sorted(overlap)[:10]}"
            )
    else:
        action_examples = [
            make_example(graph, int(selected[pos]), rank, margin, int(pos))
            for pos in action_position
        ]
    safety_examples = [make_example(graph, int(query), rank, margin, None) for query in train_correct]
    if len(inner_selected) == 0:
        raise RuntimeError("inner selected evaluation fold contains zero queries")
    if len(inner_all) == 0:
        raise RuntimeError("inner all-query evaluation fold contains zero queries")
    if len(action_examples) == 0:
        raise RuntimeError("training contains zero action examples")
    if len(safety_examples) == 0:
        raise RuntimeError("training contains zero clean safety examples")

    inner_candidate_rows = [graph.query_block(int(query))[1] for query in inner_all]
    required_real_rows = [
        graph.query_row[selected],
        graph.query_row[inner_all],
        np.asarray([example.query_row for example in action_examples], dtype=np.int64),
        np.asarray([example.query_row for example in safety_examples], dtype=np.int64),
        *(np.asarray(example.candidate_rows, dtype=np.int64) for example in action_examples),
        *(np.asarray(example.candidate_rows, dtype=np.int64) for example in safety_examples),
        *inner_candidate_rows,
    ]
    all_rows = np.unique(np.concatenate(required_real_rows)).astype(np.int64)
    required_training_rows = {
        int(row)
        for example in (*action_examples, *safety_examples)
        for row in (example.query_row, *example.candidate_rows)
    }
    missing_training_rows = required_training_rows - set(map(int, all_rows))
    if missing_training_rows:
        raise RuntimeError(
            f"internal cache-closure error; missing training rows: "
            f"{sorted(missing_training_rows)[:20]}"
        )
    cache_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    cache_embedding = np.load(
        args.token_dir / "official_embeddings_f32.npy", mmap_mode="r",
    )
    cache_position = {int(row): index for index, row in enumerate(cache_rows)}
    if len(cache_position) != len(cache_rows):
        raise RuntimeError("official embedding cache contains duplicate rows")
    official_by_row = validate_clean_data_and_embedding_cache(
        args.data, all_rows, cache_rows, cache_embedding,
    )
    cache_row_set = set(map(int, cache_rows))
    missing_broad_rows = set(map(int, broad_eval_rows)) - cache_row_set
    if missing_broad_rows:
        raise RuntimeError(
            f"official cache misses broad evaluation rows: {sorted(missing_broad_rows)[:20]}"
        )

    preflight = {
        "status": "chemaware_iceberg_direct_preflight_passed",
        "training_objective": args.training_objective,
        "arm": args.arm, "action_kind": args.action_kind,
        "action_scope": args.action_scope, "transfer_target": args.transfer_target,
        "differential_action": {
            "mode": args.differential_mode,
            "strength": args.action_strength,
            "top_k": args.action_top_k,
            "action_generation_seed": args.action_generation_seed,
        },
        "train_rescues": len(action_examples),
        "train_rescue_identities": len({value.identity for value in action_examples}),
        "train_rescue_formulas": len({value.formula for value in action_examples}),
        "train_safety_queries": len(safety_examples),
        "inner_selected_queries": int(len(inner_selected)),
        "inner_all_queries": int(len(inner_all)),
        "broad_inner_queries": int(len(broad_inner)),
        "broad_inner_formulas": int(len(np.unique(
            evaluation_manifest["query_formula"][broad_inner]
        ))),
        "broad_inner_rows": int(len(broad_eval_rows)),
        "outer_selected_queries_not_evaluated": int(len(outer_selected)),
        "contracts": {
            "global_teacher_identity_uniqueness": True,
            "formula_disjoint": True, "full_real_candidate_lists": True,
            "teacher_training_only": True, "outer_fold_evaluated": False,
            "same_query_reference_encoder": True, "P2b": "forbidden",
            "teacher_score_loss": False,
            # This is not an embedding-chase loss, but it is deliberately
            # labelled as teacher-derived for causal summaries: the target
            # score displacement is constructed from frozen official action
            # and reference embeddings and must not be reported as a pure
            # no-distillation arm.
            "teacher_embedding_loss": delta_transfer,
            "direct_action_embedding_chase_loss": False,
            "qualified_action_effect_target": delta_transfer,
            "frozen_official_action_embedding_used_to_construct_effect": delta_transfer,
            "trainable_action_view_forward": False if delta_transfer else None,
            "chemical_reference_gradient": False if delta_transfer else None,
            "candidate_centred_action_effect": delta_transfer,
            "formula_equal_action_mass": delta_transfer,
            "unsupported_action_role_weight_zero": delta_transfer,
            "base_chemical_operator_split": delta_transfer,
            "separate_optimizer_moments": delta_transfer,
            "chemical_step_weight_decay": 0.0 if delta_transfer else None,
            "clean_action_embedding_consistency": (
                not direct and args.lambda_consistency > 0
            ),
            "official_embedding_preservation_loss": args.lambda_preserve > 0,
            "official_margin_floor_loss": args.lambda_margin_floor > 0,
            "action_gradient_nonconflicting_with_primary": pcgrad_direct,
            "action_gradient_norm_cap": (
                args.maximum_action_gradient_ratio if pcgrad_direct else None
            ),
        },
        "action_bank": action_bank_contract,
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "evaluation_manifest_sha256": evaluation_manifest_sha256,
            "official_checkpoint_sha256": official_checkpoint_sha256,
            "architecture_checkpoint_sha256": architecture_checkpoint_sha256,
            "teacher_report_sha256": sha256_file(args.teacher_dir / "report.json"),
            "teacher_predictions_sha256": sha256_file(args.teacher_dir / "iceberg_predictions_f16.npy"),
        },
    }
    args.output.mkdir(parents=True)
    atomic_write_text(
        args.output / "preflight.json",
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
    )
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("direct shared-encoder training requires an available CUDA device")

    torch.set_num_threads(args.torch_threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    store = SpectrumStore(args.data, all_rows, args.n_highest_peaks)
    actions = ActionStore(
        args.teacher_dir, selected, query_ptr, store, graph, args.action_generation_seed,
        args.action_kind, args.differential_mode, args.action_strength, args.action_top_k,
    )

    model, initialization_kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    initialization = f"official_dreams:{initialization_kind}"
    if args.initial_shared_checkpoint is not None:
        package = torch_load_compat(args.initial_shared_checkpoint, map_location="cpu")
        if "model_state" not in package:
            raise RuntimeError("initial shared checkpoint lacks model_state")
        checkpoint_fold = package.get("outer_fold")
        if checkpoint_fold is not None and int(checkpoint_fold) != args.outer_fold:
            raise RuntimeError("initial shared checkpoint outer fold does not match")
        model.load_state_dict(package["model_state"], strict=True)
        initialization = str(args.initial_shared_checkpoint.resolve())
    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    real_prefix_cache = None
    action_prefix_cache = None
    prefix_audit: dict[str, object] = {"enabled": False}
    if args.frozen_prefix_cache:
        real_prefix_cache = FrozenPrefixSpectrumStore(
            model, store, device, args.prefix_cache_batch_size, args.unfreeze_blocks,
        )
        if delta_transfer:
            action_source = QualifiedActionTensorSource(
                actions, action_position, selected, graph, store, args.arm,
            )
        else:
            action_source = ActionTensorSource(actions, selected, graph, store, args.arm)
        action_prefix_cache = FrozenPrefixSpectrumStore(
            model, action_source, device, args.prefix_cache_batch_size, args.unfreeze_blocks,
        )
        prefix_audit = {
            "enabled": True, "real": real_prefix_cache.audit,
            "action": action_prefix_cache.audit,
        }
        with torch.no_grad():
            if delta_transfer:
                frozen_action = action_prefix_cache.forward(
                    model, action_position, device, args.eval_batch_size, args.amp,
                ).float().cpu().numpy()
                actions.frozen_qualified_embedding = {
                    int(position): frozen_action[index]
                    for index, position in enumerate(action_position)
                }
            elif (
                args.training_objective == "legacy_consistency"
                and args.transfer_target == "frozen_action"
            ):
                actions.frozen_embedding = action_prefix_cache.forward(
                    model, np.arange(len(selected), dtype=np.int64), device,
                    args.eval_batch_size, args.amp,
                ).float().cpu().numpy()
            initial_encoded = real_prefix_cache.forward(
                model, all_rows, device, args.eval_batch_size, args.amp,
            ).float().cpu().numpy()
    else:
        if delta_transfer:
            action_source = QualifiedActionTensorSource(
                actions, action_position, selected, graph, store, args.arm,
            )
            frozen_action = encode_rows(
                model, action_source, action_position, device,
                args.eval_batch_size, args.amp, "qualified_action_effect_targets",
            )
            actions.frozen_qualified_embedding = {
                int(position): frozen_action[index]
                for index, position in enumerate(action_position)
            }
        else:
            action_source = ActionTensorSource(actions, selected, graph, store, args.arm)
        if (not delta_transfer and
            args.training_objective == "legacy_consistency"
            and args.transfer_target == "frozen_action"
        ):
            actions.frozen_embedding = encode_rows(
                model, action_source, action_source.rows, device,
                args.eval_batch_size, args.amp, "fixed_action_targets",
            )
        initial_encoded = encode_rows(
            model, store, all_rows, device, args.eval_batch_size, args.amp, "initial",
        )
    if delta_transfer and args.arm == "clean_duplicate":
        # The matched continuation control has a mathematically exact zero
        # chemical target.  Reuse the same cached official clean vector instead
        # of allowing harmless batch-order floating-point drift to masquerade
        # as a nonzero pseudo action.
        actions.frozen_qualified_embedding = {
            int(example.teacher_position): np.asarray(
                official_by_row[example.query_row], dtype=np.float32,
            ).copy()
            for example in action_examples
        }
    if args.initial_shared_checkpoint is None:
        cached = np.stack([official_by_row[int(row)] for row in all_rows])
        cosine = np.einsum("ij,ij->i", initial_encoded, cached)
        if float(np.min(cosine)) < 0.999:
            raise RuntimeError(f"official initialization replay drifted: min cosine={float(np.min(cosine))}")

    action_target_audit: dict[str, object] | None = None
    action_delta_fit_initial: dict[str, float] | None = None
    if delta_transfer:
        bank_arm = {
            "correct_synthetic": 0,
            "candidate_swapped": 1,
            "peak_permuted": 2,
        }.get(args.arm)
        replay_rank = []
        replay_margin = []
        target_norm = []
        for example in action_examples:
            position = int(example.teacher_position)
            query_official = np.asarray(official_by_row[example.query_row], dtype=np.float32)
            action_official = np.asarray(
                actions.frozen_qualified_embedding[position], dtype=np.float32,
            )
            reference = np.stack([
                official_by_row[int(row)] for row in example.candidate_rows
            ]).astype(np.float32, copy=False)
            ptr = np.asarray(example.molecule_ptr, dtype=np.int64)
            clean_pair = reference @ query_official
            action_pair = reference @ action_official
            clean_score = np.maximum.reduceat(clean_pair, ptr[:-1])
            action_score = np.maximum.reduceat(action_pair, ptr[:-1])
            replay_rank.append(strict_rank(action_score))
            replay_margin.append(float(action_score[0] - np.max(action_score[1:])))
            delta = action_score - clean_score
            delta -= np.mean(delta)
            target_norm.append(float(np.linalg.norm(delta)))
        replay_rank = np.asarray(replay_rank, dtype=np.int16)
        replay_margin = np.asarray(replay_margin, dtype=np.float32)
        baseline_rank = np.asarray(
            [example.official_rank for example in action_examples], dtype=np.int16,
        )
        baseline_margin = np.asarray(
            [example.official_margin for example in action_examples], dtype=np.float32,
        )
        if bank_arm is None:
            expected_rank = rank[selected[action_position]]
            expected_margin = margin[selected[action_position]]
        else:
            expected_rank = ranks[chosen_for_integrity, bank_arm, action_position]
            expected_margin = margins[chosen_for_integrity, bank_arm, action_position]
        maximum_margin_error = float(np.max(np.abs(replay_margin - expected_margin)))
        if not np.array_equal(replay_rank, expected_rank) or maximum_margin_error > 5e-4:
            raise RuntimeError(
                "frozen qualified action target does not replay its bank outcome: "
                f"rank_mismatches={int(np.sum(replay_rank != expected_rank))}, "
                f"max_margin_error={maximum_margin_error}"
            )
        if args.arm == "clean_duplicate" and np.any(np.asarray(target_norm) > 1e-7):
            raise RuntimeError("clean-duplicate action target is not an exact no-op")
        action_corrected = int(np.sum((baseline_rank > 1) & (replay_rank == 1)))
        action_introduced = int(np.sum((baseline_rank == 1) & (replay_rank > 1)))
        action_margin_gain = replay_margin - baseline_margin
        if args.arm == "correct_synthetic" and (
            action_corrected < 1 or float(np.mean(action_margin_gain)) <= 0
        ):
            raise RuntimeError(
                "qualified correct action lost its frozen decision-level effect"
            )
        action_target_audit = {
            "qualified_actions": int(len(action_examples)),
            "rank_replay_mismatches": int(np.sum(replay_rank != expected_rank)),
            "maximum_margin_replay_error": maximum_margin_error,
            "mean_candidate_delta_l2": float(np.mean(target_norm)),
            "zero_target_actions": int(np.sum(np.asarray(target_norm) == 0)),
            "frozen_action_corrected": action_corrected,
            "frozen_action_introduced": action_introduced,
            "frozen_action_mean_margin_gain": float(np.mean(action_margin_gain)),
            "frozen_action_positive_margin_fraction": float(np.mean(action_margin_gain > 0)),
            "action_view_trainable_forward": False,
            "chemical_reference_gradient": False,
        }
        atomic_write_text(
            args.output / "action_target_audit.json",
            json.dumps(action_target_audit, indent=2) + "\n",
        )
        action_delta_fit_initial = evaluate_action_delta_fit(
            model, action_examples, store, actions, args.arm,
            official_by_row, device, args, real_prefix_cache, action_prefix_cache,
        )

    head_parameters = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": args.backbone_lr},
        {"params": head_parameters, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    chemical_optimizer = None
    if delta_transfer:
        chemical_optimizer = torch.optim.AdamW([
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ], weight_decay=0.0)

    history = []
    chemical_step_audits: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed + 503)
    for epoch in range(1, args.epochs + 1):
        # E8's validated direct protocol keeps dropout disabled during gradient
        # training.  ``eval`` does not disable autograd; it only removes a
        # stochastic geometry shift that otherwise makes preservation nonzero
        # before the first optimizer update.
        model.eval()
        action_order = np.arange(len(action_examples), dtype=np.int64)
        rng.shuffle(action_order)
        safety_order = rng.permutation(len(safety_examples))
        if len(safety_order) < len(action_order):
            safety_order = np.resize(safety_order, len(action_order))
        epoch_batches = int(np.ceil(len(action_order) / args.batch_queries))
        registered_chemical_steps = {0, epoch_batches - 1}
        totals: dict[str, float] = {"loss": 0.0, "grad_norm": 0.0, "clip_fraction": 0.0}
        batches = 0
        for left in range(0, len(action_order), args.batch_queries):
            action_batch = [action_examples[index] for index in action_order[left:left + args.batch_queries]]
            safety_batch = [safety_examples[index] for index in safety_order[left:left + len(action_batch)]]
            optimizer.zero_grad(set_to_none=True)
            loss_action, action_tensors, action_components = action_loss(
                model, action_batch, store, actions, args.arm, official_by_row, device, args,
                real_prefix_cache, action_prefix_cache,
            )
            loss_safety, _safety_tensors, safety_components = safety_loss(
                model, safety_batch, store, official_by_row, device, args, real_prefix_cache,
            )
            if delta_transfer:
                primary_loss = (
                    args.lambda_clean_rank * action_tensors["clean_full_list"]
                    + args.lambda_margin_floor * action_tensors["margin_floor"]
                    + args.lambda_preserve * action_tensors["preserve"]
                    + args.safety_stream_weight * loss_safety
                )
                if not bool(torch.isfinite(primary_loss.detach())):
                    raise RuntimeError(
                        f"non-finite primary loss at epoch={epoch} batch={batches + 1}"
                    )
                primary_loss.backward()
                primary_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip, error_if_nonfinite=True,
                ))
                optimizer.step()

                if chemical_optimizer is None:
                    raise RuntimeError("action-delta transfer lacks its chemical optimizer")
                chemical_optimizer.zero_grad(set_to_none=True)
                _replayed_loss, replayed_tensors, replayed_components = action_loss(
                    model, action_batch, store, actions, args.arm,
                    official_by_row, device, args, real_prefix_cache, action_prefix_cache,
                )
                chemical_loss = (
                    args.lambda_action_rank * replayed_tensors["action_delta_transfer"]
                )
                if not bool(torch.isfinite(chemical_loss.detach())):
                    raise RuntimeError(
                        f"non-finite chemical loss at epoch={epoch} batch={batches + 1}"
                    )
                chemical_loss.backward()
                chemical_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip, error_if_nonfinite=True,
                ))
                audit_optimizer_step = batches in registered_chemical_steps
                parameter_snapshot = (
                    [
                        parameter.detach().clone()
                        for parameter in (*backbone_parameters, *head_parameters)
                    ]
                    if audit_optimizer_step else None
                )
                chemical_loss_before = float(chemical_loss.detach())
                chemical_optimizer.step()
                if audit_optimizer_step:
                    if parameter_snapshot is None:
                        raise RuntimeError("registered chemical optimizer step lacks a snapshot")
                    geometry = optimizer_descent_geometry(
                        parameter_snapshot, backbone_parameters, head_parameters,
                    )
                    del parameter_snapshot
                    with torch.no_grad():
                        _post_loss, post_tensors, _post_components = action_loss(
                            model, action_batch, store, actions, args.arm,
                            official_by_row, device, args,
                            real_prefix_cache, action_prefix_cache,
                        )
                    chemical_loss_after = float(
                        args.lambda_action_rank
                        * post_tensors["action_delta_transfer"].detach()
                    )
                    converged = chemical_loss_before <= 1e-10
                    loss_tolerance = max(1e-8, 1e-3 * chemical_loss_before)
                    loss_decreased = bool(
                        chemical_loss_after <= chemical_loss_before + loss_tolerance
                    )
                    geometry_valid = bool(converged or all(
                        values["postclip_gradient_norm"] > 0
                        and values["descent_update_norm"] > 0
                        and values["gradient_update_cosine"] > 0
                        and values["first_order_descent"] > 0
                        for values in geometry.values()
                    ))
                    chemical_step_audits.append({
                        "epoch": epoch,
                        "batch": batches + 1,
                        "registered_position": (
                            "first" if batches == 0 else "last"
                        ),
                        "chemical_loss_before": chemical_loss_before,
                        "chemical_loss_after": chemical_loss_after,
                        "relative_loss_change": float(
                            (chemical_loss_after - chemical_loss_before)
                            / max(chemical_loss_before, 1e-12)
                        ),
                        "already_converged": converged,
                        "loss_nonincreasing_with_tolerance": loss_decreased,
                        "geometry_valid": geometry_valid,
                        "groups": geometry,
                    })
                    if not loss_decreased or not geometry_valid:
                        raise RuntimeError(
                            "chemical AdamW step failed its realized-update audit: "
                            f"loss={chemical_loss_before:.8g}->{chemical_loss_after:.8g}, "
                            f"geometry={geometry}"
                        )
                loss = primary_loss.detach() + chemical_loss.detach()
                action_components = replayed_components
                gradient_audit = {
                    "primary_norm": primary_norm,
                    "chemical_norm": chemical_norm,
                    "primary_clipped": float(primary_norm > args.grad_clip),
                    "chemical_clipped": float(chemical_norm > args.grad_clip),
                }
                norm = 0.5 * (primary_norm + chemical_norm)
                clipped = 0.5 * (
                    float(primary_norm > args.grad_clip)
                    + float(chemical_norm > args.grad_clip)
                )
            elif pcgrad_direct:
                primary_loss = (
                    args.lambda_clean_rank * action_tensors["clean_full_list"]
                    + args.lambda_margin_floor * action_tensors["margin_floor"]
                    + args.lambda_preserve * action_tensors["preserve"]
                    + args.safety_stream_weight * loss_safety
                )
                auxiliary_loss = args.lambda_action_rank * action_tensors["action_full_list"]
                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                primary_gradient = torch.autograd.grad(
                    primary_loss, trainable, retain_graph=True, allow_unused=True,
                )
                auxiliary_gradient = torch.autograd.grad(
                    auxiliary_loss, trainable, allow_unused=True,
                )
                combined, gradient_audit = projected_guarded_auxiliary(
                    primary_gradient, auxiliary_gradient, trainable,
                    args.maximum_action_gradient_ratio,
                )
                for parameter, gradient in zip(trainable, combined):
                    parameter.grad = gradient
                loss = primary_loss + auxiliary_loss
            else:
                loss = loss_action + args.safety_stream_weight * loss_safety
                loss.backward()
                gradient_audit = {}
            if not bool(torch.isfinite(loss.detach())):
                raise RuntimeError(f"non-finite training loss at epoch={epoch} batch={batches + 1}")
            if not delta_transfer:
                norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip, error_if_nonfinite=True,
                ))
                clipped = float(norm > args.grad_clip)
                optimizer.step()
            totals["loss"] += float(loss.detach()); totals["grad_norm"] += norm
            totals["clip_fraction"] += clipped
            for key, value in action_components.items():
                totals[key] = totals.get(key, 0.0) + value
            for key, value in safety_components.items():
                totals[key] = totals.get(key, 0.0) + value
            for key, value in gradient_audit.items():
                totals[f"gradient_{key}"] = totals.get(f"gradient_{key}", 0.0) + float(value)
            batches += 1
        if real_prefix_cache is not None:
            with torch.no_grad():
                encoded = real_prefix_cache.forward(
                    model, all_rows, device, args.eval_batch_size, args.amp,
                ).float().cpu().numpy()
        else:
            encoded = encode_rows(
                model, store, all_rows, device, args.eval_batch_size, args.amp, f"epoch_{epoch}",
            )
        inner_selected_eval = paired_evaluation(
            encoded, initial_encoded, store, graph, inner_selected,
        )
        inner_all_eval = paired_evaluation(encoded, initial_encoded, store, graph, inner_all)
        entry = {
            "epoch": epoch,
            "train": {key: value / batches for key, value in totals.items()},
            "inner_selected": inner_selected_eval["summary"],
            "inner_all": inner_all_eval["summary"],
        }
        history.append(entry)
        print(
            f"arm={args.arm} epoch={epoch}/{args.epochs} "
            f"selected_delta={entry['inner_selected']['delta_recall1']:+.4f} "
            f"all_delta={entry['inner_all']['delta_recall1']:+.4f} "
            f"clip={entry['train']['clip_fraction']:.3f}", flush=True,
        )

    final_encoded = encoded
    action_delta_fit_final = (
        evaluate_action_delta_fit(
            model, action_examples, store, actions, args.arm,
            official_by_row, device, args, real_prefix_cache, action_prefix_cache,
        ) if delta_transfer else None
    )
    selected_eval = add_registered_rank_metrics(
        paired_evaluation(final_encoded, initial_encoded, store, graph, inner_selected),
        graph, inner_selected,
    )
    all_eval = add_registered_rank_metrics(
        paired_evaluation(final_encoded, initial_encoded, store, graph, inner_all),
        graph, inner_all,
    )
    selected_delta = (
        (selected_eval["new_rank"] == 1).astype(float)
        - (selected_eval["old_rank"] == 1).astype(float)
    )
    ci = formula_bootstrap(
        selected_delta, graph.query_formula[inner_selected], args.seed + 901, args.bootstrap_draws,
    )
    per_query = pd.DataFrame({
        "query_index": inner_all,
        "formula": graph.query_formula[inner_all],
        "identity": graph.query_ik14[inner_all],
        "initial_rank": all_eval["old_rank"], "final_rank": all_eval["new_rank"],
        "initial_margin": all_eval["old_margin"], "final_margin": all_eval["new_margin"],
        "candidate_count": all_eval["candidate_count"],
    })
    broad_store = SpectrumStore(args.data, broad_eval_rows, args.n_highest_peaks)
    broad_encoded = encode_rows(
        model, broad_store, broad_eval_rows, device,
        args.eval_batch_size, args.amp, "broad-held-formula-evaluation",
    )
    broad_eval = evaluate_manifest_subset(
        evaluation_manifest, broad_inner, cache_embedding, cache_position,
        broad_eval_rows, broad_encoded,
    )
    broad_delta = (
        (broad_eval["new_rank"] == 1).astype(float)
        - (broad_eval["old_rank"] == 1).astype(float)
    )
    broad_ci = formula_bootstrap(
        broad_delta,
        evaluation_manifest["query_formula"][broad_inner],
        args.seed + 1901,
        args.bootstrap_draws,
    )
    broad_per_query = pd.DataFrame({
        "query_index": broad_inner,
        "formula": evaluation_manifest["query_formula"][broad_inner],
        "identity": evaluation_manifest["query_ik14"][broad_inner],
        "initial_rank": broad_eval["old_rank"],
        "final_rank": broad_eval["new_rank"],
        "initial_margin": broad_eval["old_margin"],
        "final_margin": broad_eval["new_margin"],
        "candidate_count": broad_eval["candidate_count"],
    })
    near_delta = all_eval["summary"]["delta_near_recall1"]
    near_n = int(all_eval["summary"]["near_n"])
    if (near_n == 0) != (near_delta is None):
        raise RuntimeError("near-stratum count and metric applicability disagree")
    gate_eval = broad_eval if delta_transfer else all_eval
    gate_ci = broad_ci if delta_transfer else ci
    required_final_metrics = {
        "inner_selected_delta_recall1": selected_eval["summary"]["delta_recall1"],
        "inner_selected_formula_ci_low": ci["formula_cluster_bootstrap_95ci"][0],
        "gate_delta_recall1": gate_eval["summary"]["delta_recall1"],
        "gate_formula_ci_low": gate_ci["formula_cluster_bootstrap_95ci"][0],
        "gate_delta_mrr": gate_eval["summary"]["delta_mrr"],
        "gate_delta_macro_auc": gate_eval["summary"]["delta_macro_auc"],
        "gate_delta_micro_auc": gate_eval["summary"]["delta_micro_auc"],
        "gate_preservation_mean": gate_eval["summary"]["preservation_mean"],
        **{
            f"gate_delta_recall{k}": gate_eval["summary"][f"delta_recall{k}"]
            for k in (5, 10, 20, 50)
        },
        **({"inner_all_delta_near_recall1": near_delta} if near_delta is not None else {}),
    }
    invalid_final_metrics = {
        name: value for name, value in required_final_metrics.items()
        if not np.isfinite(float(value))
    }
    if invalid_final_metrics:
        raise RuntimeError(f"non-finite final evaluation metrics: {invalid_final_metrics}")
    clip_fractions = [float(row["train"]["clip_fraction"]) for row in history]
    if not clip_fractions or not np.all(np.isfinite(clip_fractions)):
        raise RuntimeError("training history lacks finite clip fractions")
    if delta_transfer:
        primary_clip_fraction = float(np.mean([
            row["train"]["gradient_primary_clipped"] for row in history
        ]))
        chemical_clip_fraction = float(np.mean([
            row["train"]["gradient_chemical_clipped"] for row in history
        ]))
        maximum_stream_clip_fraction = max(
            primary_clip_fraction, chemical_clip_fraction,
        )
        expected_optimizer_audits = args.epochs * min(
            2, int(np.ceil(len(action_examples) / args.batch_queries)),
        )
        if len(chemical_step_audits) != expected_optimizer_audits:
            raise RuntimeError(
                "chemical optimizer audit coverage drifted: "
                f"observed={len(chemical_step_audits)} expected={expected_optimizer_audits}"
            )
        chemical_optimizer_signal = {
            "registered_positions": "first_and_last_chemical_step_per_epoch",
            "observations": len(chemical_step_audits),
            "expected_observations": expected_optimizer_audits,
            "all_losses_nonincreasing": bool(all(
                row["loss_nonincreasing_with_tolerance"]
                for row in chemical_step_audits
            )),
            "all_group_geometries_valid": bool(all(
                row["geometry_valid"] for row in chemical_step_audits
            )),
            "minimum_gradient_update_cosine": {
                group: float(min(
                    (
                        row["groups"][group]["gradient_update_cosine"]
                        for row in chemical_step_audits
                        if not row["already_converged"]
                    ),
                    default=1.0,
                ))
                for group in ("all", "backbone", "head")
            },
            "gate_passed": bool(all(
                row["loss_nonincreasing_with_tolerance"] and row["geometry_valid"]
                for row in chemical_step_audits
            )),
            "steps": chemical_step_audits,
        }
    else:
        primary_clip_fraction = None
        chemical_clip_fraction = None
        maximum_stream_clip_fraction = float(np.mean(clip_fractions))
        chemical_optimizer_signal = None
    gates: dict[str, bool | None] = {
        "gate_recall1_positive": gate_eval["summary"]["delta_recall1"] > 0,
        "gate_recall1_formula_ci_positive": gate_ci["formula_cluster_bootstrap_95ci"][0] > 0,
        "inner_all_near_nonnegative": (
            None if delta_transfer or near_delta is None else near_delta >= 0
        ),
        "gate_mrr_nonnegative": gate_eval["summary"]["delta_mrr"] >= 0,
        "gate_macro_auc_nonnegative": gate_eval["summary"]["delta_macro_auc"] >= 0,
        "gate_micro_auc_nonnegative": gate_eval["summary"]["delta_micro_auc"] >= 0,
        **{
            f"gate_recall{k}_nonnegative": gate_eval["summary"][f"delta_recall{k}"] >= 0
            for k in (5, 10, 20, 50)
        },
        "gate_preservation": gate_eval["summary"]["preservation_mean"] >= 0.995,
        "clip_not_saturated": maximum_stream_clip_fraction < 0.9,
        "chemical_optimizer_signal_preserved": (
            chemical_optimizer_signal["gate_passed"] if delta_transfer else None
        ),
        "chemical_target_fit_improved": (
            None if not delta_transfer or args.arm == "clean_duplicate" else
            action_delta_fit_final["formula_weighted_huber"]
            < action_delta_fit_initial["formula_weighted_huber"]
        ),
    }
    applicable_gates = {name: bool(value) for name, value in gates.items() if value is not None}
    if not applicable_gates:
        raise RuntimeError("no applicable evaluation gates were produced")
    passed = all(applicable_gates.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "decision": (
            (
                "Qualified ChemAware action effects improved the clean shared embedding."
                if delta_transfer else
                "ChemAware action views improved the directly fine-tuned shared embedding."
            ) if passed else
            "This arm did not pass the fixed shared-embedding performance and safety gates."
        ),
        "preflight": preflight, "initialization": initialization,
        "model": capacity | {
            "same_query_reference_encoder": True,
            "trainable_scope": "projection head plus final Transformer block",
            "deployment_input": "one clean MS/MS spectrum plus precursor metadata",
            "deployment_output": "one normalized 1024-dimensional embedding",
        },
        "optimization": {
            "training_objective": args.training_objective,
            "seed": args.seed, "fold_seed": args.fold_seed,
            "inner_fold": args.inner_fold, "outer_fold": args.outer_fold,
            "epochs": args.epochs, "batch_queries": args.batch_queries,
            "max_action_identities": args.max_action_identities,
            "max_safety_identities": args.max_safety_identities,
            "max_eval_identities": args.max_eval_identities,
            "unfreeze_blocks": args.unfreeze_blocks,
            "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "maximum_action_gradient_ratio": args.maximum_action_gradient_ratio,
            "temperature": args.temperature, "arm": args.arm,
            "action_delta_alpha": args.action_delta_alpha,
            "action_delta_huber": args.action_delta_huber,
            "base_chemical_operator_split": delta_transfer,
            "separate_optimizer_moments": delta_transfer,
            "chemical_weight_decay": 0.0 if delta_transfer else None,
            "action_kind": args.action_kind,
            "differential_mode": args.differential_mode,
            "action_strength": args.action_strength,
            "action_top_k": args.action_top_k,
            "lambda_peak_contrast": args.lambda_peak_contrast,
            "peak_contrast_k": args.peak_contrast_k,
            "peak_contrast_margin": args.peak_contrast_margin,
            "action_scope": args.action_scope,
            "transfer_target": args.transfer_target,
            "lambda_margin_floor": args.lambda_margin_floor,
            "lambda_preserve": args.lambda_preserve,
            "margin_floor_slack": args.margin_floor_slack,
            "lambda_clean_rank": args.lambda_clean_rank,
            "lambda_action_rank": args.lambda_action_rank,
            "safety_stream_weight": args.safety_stream_weight,
            "losses": {
                "clean_full_candidate_listwise": args.lambda_clean_rank,
                "action_full_candidate_listwise": (
                    0.0 if delta_transfer else args.lambda_action_rank
                ),
                "qualified_action_effect_to_clean": (
                    args.lambda_action_rank if delta_transfer else 0.0
                ),
                "safety_full_candidate_listwise": args.safety_stream_weight,
                "teacher_score": 0.0,
                "teacher_embedding": 0.0,
                "teacher_derived_frozen_action_effect": (
                    args.lambda_action_rank if delta_transfer else 0.0
                ),
                "clean_action_embedding_consistency": args.lambda_consistency,
                "official_margin_floor": args.lambda_margin_floor,
                "official_embedding_preservation": args.lambda_preserve,
            },
        },
        "action_target_audit": action_target_audit,
        "chemical_target_fit": (
            {
                "initial": action_delta_fit_initial,
                "final": action_delta_fit_final,
                "formula_weighted_huber_reduction": (
                    action_delta_fit_initial["formula_weighted_huber"]
                    - action_delta_fit_final["formula_weighted_huber"]
                ),
            } if delta_transfer else None
        ),
        "gradient_clipping": {
            "mean_combined_step_fraction": float(np.mean(clip_fractions)),
            "primary_stream_fraction": primary_clip_fraction,
            "chemical_stream_fraction": chemical_clip_fraction,
            "maximum_stream_fraction": maximum_stream_clip_fraction,
        },
        "chemical_optimizer_signal": chemical_optimizer_signal,
        "prefix_cache": prefix_audit,
        "final": {
            "inner_selected": selected_eval["summary"],
            "inner_all": all_eval["summary"], "formula_bootstrap": ci,
            "broad_inner": broad_eval["summary"],
            "broad_formula_bootstrap": broad_ci,
            "primary_gate": "broad_inner" if delta_transfer else "inner_all",
        },
        "gates": gates,
        "gate_applicability": {
            "inner_all_near_nonnegative": near_delta is not None,
            "inner_all_near_reason": (
                None if near_delta is not None else "inner evaluation subset contains zero near queries"
            ),
        },
        "history": history,
        "scope": {
            "development_only": True, "outer_fold_evaluated": False,
            "massspecgym_checkpoint_overlap_warning": True,
            "candidate_or_structure_input_at_deployment": False,
            "dropout_disabled_during_gradient_training": True,
        },
        "runtime_seconds": float(time.time() - started),
    }
    # Serialize the complete report before writing the large checkpoint.  This
    # makes schema/type failures cheap and ensures a COMPLETE marker can only
    # describe a fully materialized arm.
    report_text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    checkpoint = {
        "status": "chemaware_iceberg_direct_shared_encoder",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "initialization": initialization, "arm": args.arm, "outer_fold": args.outer_fold,
        "inner_fold": args.inner_fold, "seed": args.seed,
        "inference_clean_only": True, "teacher_used_at_inference": False,
        "capacity": capacity,
    }
    per_query_path = args.output / "inner_per_query.csv.gz"
    per_query_temporary = per_query_path.with_name(per_query_path.name + ".tmp")
    per_query.to_csv(per_query_temporary, index=False, compression="gzip")
    per_query_temporary.replace(per_query_path)
    broad_per_query_path = args.output / "broad_inner_per_query.csv.gz"
    broad_per_query_temporary = broad_per_query_path.with_name(
        broad_per_query_path.name + ".tmp"
    )
    broad_per_query.to_csv(
        broad_per_query_temporary, index=False, compression="gzip",
    )
    broad_per_query_temporary.replace(broad_per_query_path)
    atomic_torch_save(checkpoint, args.output / "final_shared_encoder.pt")
    atomic_write_text(args.output / "report.json", report_text)
    atomic_write_text(
        args.output / "COMPLETE.json",
        json.dumps({
            "status": "CHEMAWARE_DIRECT_ARM_COMPLETE",
            "arm": args.arm,
            "report_status": report["status"],
            "required_files": ([
                "preflight.json", "inner_per_query.csv.gz",
                "broad_inner_per_query.csv.gz", "final_shared_encoder.pt", "report.json",
            ] + (["action_target_audit.json"] if delta_transfer else [])),
        }, indent=2) + "\n",
    )
    print(json.dumps(report["final"], indent=2, ensure_ascii=False), flush=True)
    print(f"decision={report['status']} report={args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
