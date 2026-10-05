#!/usr/bin/env python
"""Directly fine-tune official DreaMS using BioAware-routed hard errors.

This is deliberately a shared clean-spectrum encoder: the same trainable model
encodes query and reference spectra.  BioAware is a training-only router and is
absent at inference.  The held formula fold is evaluated once after a fixed
number of optimizer steps and is never used for checkpoint selection.

The optional B5 supervision mode keeps the exact same routed examples and
optimizer budget, but replaces the one-hot listwise target with the nested-OOF
candidate-level BioAware distribution.  This makes ``direct_onehot`` versus
``bioaware_soft`` a controlled comparison rather than a change of dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from noise_final_core import sha256_file  # noqa: E402
from annotation.bioaware_supervision import normalise_within_query_logits  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_e4a_direct_augmentation import unfreeze_last_blocks  # noqa: E402


ARMS = {
    "full_bioaware_safe": ("safe_corrected", "safe_introduced"),
    "full_no_edge_high_recall": ("recall_corrected", "recall_introduced"),
    # The graph-prior arm is routed by a separately frozen, nested formula-OOF
    # action ledger.  Its loss remains the ordinary one-hot molecular identity
    # ranking loss; no graph score or teacher probability enters the encoder.
    "graph_prior_direct": (None, None),
    # B8 matched controls use the same true-identity objective as B7.  The two
    # arms differ only in which baseline-wrong identity clusters are routed.
    "matched_graph_direct": (None, None),
    "matched_generic_direct": (None, None),
}

DIRECT_ROUTE_ARMS = {
    "graph_prior_direct": {
        "status": "bioaware_b7_graph_action_router_frozen",
        "selection_column": "corrected",
        "pass_column": "pass_to_direct_gradient_gate",
    },
    "matched_graph_direct": {
        "status": "bioaware_b8_matched_error_routers_frozen",
        "selection_column": "corrective_selected",
        "pass_column": None,
    },
    "matched_generic_direct": {
        "status": "bioaware_b8_matched_error_routers_frozen",
        "selection_column": "corrective_selected",
        "pass_column": None,
    },
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b4_direct_manifest_v1",
    )
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--architecture-checkpoint", type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument(
        "--supervision-mode",
        choices=("direct_onehot", "bioaware_soft"),
        default="direct_onehot",
    )
    parser.add_argument("--candidate-teacher-dir", type=Path)
    parser.add_argument(
        "--action-router-dir", type=Path,
        help="Frozen complete action ledger required by a direct-route arm.",
    )
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--batch-queries", type=int, default=4)
    parser.add_argument("--references-per-molecule", type=int, default=2)
    parser.add_argument("--corrective-fraction", type=float, default=0.5)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--lambda-listwise", type=float, default=0.5)
    parser.add_argument("--lambda-corrective", type=float, default=1.0)
    parser.add_argument("--lambda-safety-floor", type=float, default=2.0)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--harm-safety-weight", type=float, default=4.0)
    parser.add_argument("--margin-floor-slack", type=float, default=0.005)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    # Full-backbone B4 training is numerically unstable under fp16 autocast on
    # the target cluster (the first formal step produced NaNs while FP32 smoke
    # was finite).  FP32 is therefore the safe default; AMP requires an explicit
    # future requalification rather than an accidental omission of --no-amp.
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    required = {
        "query_id", "unit_id", "query_ik14", "query_formula", "query_tensor",
        "query_official_embedding",
        "formula_fold", "query_ptr", "molecule_id", "molecule_best_row",
        "molecule_formula", "molecule_official_score", "reference_ptr",
        "reference_rows", "reference_tensor_rows", "reference_tensor",
        "reference_official_embedding", "baseline_rank", "safe_corrected", "safe_introduced",
        "recall_corrected", "recall_introduced",
        "safe_corrected_by_outer", "safe_introduced_by_outer",
        "recall_corrected_by_outer", "recall_introduced_by_outer",
    }
    if missing := required - set(body):
        raise RuntimeError(f"manifest lacks {sorted(missing)}")
    n = len(body["query_id"])
    if len(body["query_ptr"]) != n + 1 or int(body["query_ptr"][-1]) != len(body["molecule_id"]):
        raise RuntimeError("invalid query pointers")
    if len(body["reference_ptr"]) != len(body["molecule_id"]) + 1:
        raise RuntimeError("invalid reference pointers")
    return body


class ManifestReferenceStore:
    """MoNA reference spectra frozen in the manifest with their library rows."""

    def __init__(self, rows: np.ndarray, tensor: np.ndarray):
        self.rows = np.asarray(rows, dtype=np.int64)
        if (self.rows.ndim != 1 or len(self.rows) != len(tensor)
                or np.any(np.diff(self.rows) <= 0)):
            raise RuntimeError("invalid manifest reference tensor index")
        self.position = {int(row): index for index, row in enumerate(self.rows)}
        self.tensor = torch.from_numpy(np.asarray(tensor, dtype=np.float32))

    def get(self, rows: list[int] | tuple[int, ...] | np.ndarray) -> torch.Tensor:
        try:
            positions = [self.position[int(row)] for row in rows]
        except KeyError as error:
            raise RuntimeError(
                f"MoNA reference row absent from manifest: {error.args[0]}"
            ) from error
        return self.tensor[positions]


def query_candidates(body: dict[str, np.ndarray], query: int,
                     forbidden_formulas: set[str] | None = None) -> list[int]:
    left, right = map(int, body["query_ptr"][query:query + 2])
    candidates = []
    for molecule in range(left, right):
        if (forbidden_formulas is not None
                and str(body["molecule_formula"][molecule]) in forbidden_formulas):
            continue
        candidates.append(molecule)
    if left not in candidates:
        raise RuntimeError(f"query {query}: truth candidate was filtered")
    candidates.remove(left)
    return [left, *candidates]


def official_margin(body: dict[str, np.ndarray], query: int,
                    forbidden_formulas: set[str] | None = None) -> float:
    molecules = query_candidates(body, query, forbidden_formulas)
    score = body["molecule_official_score"][molecules].astype(float)
    return float(score[0] - np.max(score[1:]))


def by_identity(indices: np.ndarray, identities: np.ndarray) -> dict[str, np.ndarray]:
    output: dict[str, list[int]] = {}
    for index in map(int, indices):
        output.setdefault(str(identities[index]), []).append(index)
    return {key: np.asarray(value, dtype=np.int64) for key, value in output.items()}


def sample_identity_balanced(pool: dict[str, np.ndarray], count: int,
                             rng: np.random.Generator) -> list[int]:
    identities = list(pool)
    if not identities:
        raise RuntimeError("empty identity-balanced pool")
    chosen = []
    for _ in range(count):
        identity = identities[int(rng.integers(0, len(identities)))]
        values = pool[identity]
        chosen.append(int(values[int(rng.integers(0, len(values)))]))
    return chosen


def load_candidate_teacher(
    directory: Path,
    arm: str,
    outer_fold: int,
    corrective_queries: np.ndarray,
    body: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Load and validate one nested-OOF teacher view for a training fold."""
    candidate_path = directory / "candidate_teacher_scores.csv.gz"
    action_path = directory / "teacher_actions.csv.gz"
    report_path = directory / "report.json"
    for path in (candidate_path, action_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b5_candidate_teacher_frozen":
        raise RuntimeError("wrong B5 candidate-teacher status")
    candidates = pd.read_csv(candidate_path)
    actions = pd.read_csv(action_path)
    candidates = candidates[
        candidates["arm"].eq(arm)
        & candidates["outer_fold"].eq(outer_fold)
    ].copy()
    actions = actions[
        actions["arm"].eq(arm)
        & actions["outer_fold"].eq(outer_fold)
        & actions["corrected"].astype(bool)
    ].copy()
    expected_query_ids = {
        str(body["query_id"][int(query)]) for query in corrective_queries
    }
    if actions["query_id"].astype(str).duplicated().any():
        raise RuntimeError("duplicate B5 corrected action in one outer fold")
    if set(actions["query_id"].astype(str)) != expected_query_ids:
        raise RuntimeError("B4 routed-correction set differs from B5 teacher actions")
    candidates = candidates[
        candidates["query_id"].astype(str).isin(expected_query_ids)
    ].copy()
    teacher: dict[str, dict[str, float]] = {}
    for query_id, group in candidates.groupby("query_id", sort=False):
        candidate_ids = group["candidate_id"].astype(str)
        if candidate_ids.duplicated().any() or len(group) < 2:
            raise RuntimeError(f"{query_id}: invalid candidate-teacher group")
        scores = group["teacher_model_score"].to_numpy(float)
        if not np.isfinite(scores).all():
            raise RuntimeError(f"{query_id}: non-finite candidate-teacher logits")
        teacher[str(query_id)] = dict(zip(candidate_ids, scores, strict=True))
    if set(teacher) != expected_query_ids:
        raise RuntimeError("candidate-teacher coverage differs from corrective query set")
    return teacher, {
        "candidate_teacher_report_sha256": sha256_file(report_path),
        "candidate_teacher_scores_sha256": sha256_file(candidate_path),
        "candidate_teacher_actions_sha256": sha256_file(action_path),
    }


def load_direct_action_routes(
    directory: Path,
    arm: str,
    outer_fold: int,
    train_queries: np.ndarray,
    body: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Load actions used only to select true-identity training queries.

    Every non-held query must have exactly one inner-OOF action for this outer
    fold.  The action score is never returned and therefore cannot enter the
    embedding loss accidentally.
    """
    action_path = directory / "action_routes.csv.gz"
    report_path = directory / "report.json"
    for path in (action_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if arm not in DIRECT_ROUTE_ARMS:
        raise ValueError(f"not a direct-route arm: {arm}")
    contract = DIRECT_ROUTE_ARMS[arm]
    if report.get("status") != contract["status"]:
        raise RuntimeError(f"wrong action-router status for {arm}")
    pass_column = contract["pass_column"]
    if pass_column is not None and not bool(report.get(pass_column, False)):
        raise RuntimeError(f"action-router gate is false for {arm}")
    if (report.get("status") == "bioaware_b8_matched_error_routers_frozen"
            and not bool(report.get("outer_folds", {}).get(str(outer_fold), {}).get(
                "pass_to_training", False
            ))):
        raise RuntimeError(f"B8 matched router did not pass outer fold {outer_fold}")
    actions = pd.read_csv(action_path)
    actions = actions[
        actions["arm"].eq(arm)
        & actions["outer_fold"].eq(outer_fold)
    ].copy()
    if actions["query_id"].astype(str).duplicated().any():
        raise RuntimeError("duplicate direct action for one query/outer fold")
    complete_expected = {
        str(body["query_id"][index])
        for index in range(len(body["query_id"]))
        if int(body["formula_fold"][index]) != outer_fold
    }
    observed = set(actions["query_id"].astype(str))
    if observed != complete_expected:
        missing = sorted(complete_expected - observed)[:10]
        extra = sorted(observed - complete_expected)[:10]
        raise RuntimeError(
            "direct action-router coverage mismatch: "
            f"missing={missing} extra={extra}"
        )
    expected = {str(body["query_id"][int(query)]) for query in train_queries}
    actions = actions[actions["query_id"].astype(str).isin(expected)].copy()
    if set(actions["query_id"].astype(str)) != expected:
        raise RuntimeError("direct routes do not cover the post-safety-filter training set")
    position = {str(value): int(index) for index, value in enumerate(body["query_id"])}
    selection_column = str(contract["selection_column"])
    if selection_column not in actions or "introduced" not in actions:
        raise RuntimeError(
            f"action routes lack {selection_column!r} or 'introduced'"
        )
    corrective = np.asarray([
        position[str(row.query_id)]
        for row in actions.itertuples(index=False)
        if bool(getattr(row, selection_column))
    ], dtype=np.int64)
    introduced = np.asarray([
        position[str(row.query_id)]
        for row in actions.itertuples(index=False)
        if bool(row.introduced)
    ], dtype=np.int64)
    if set(map(int, corrective)) & set(map(int, introduced)):
        raise RuntimeError("direct action cannot be both corrective and introduced")
    return corrective, introduced, {
        "action_router_report_sha256": sha256_file(report_path),
        "action_router_actions_sha256": sha256_file(action_path),
    }


def candidate_teacher_target(
    score_by_candidate: dict[str, float],
    candidate_ids: list[str],
    device: torch.device,
    dtype: torch.dtype,
    temperature: float,
) -> torch.Tensor:
    """Return a fold-calibrated within-query candidate distribution.

    Pairwise logistic decision scores are not calibrated categorical logits.
    Standardising within each query preserves their ordering and relative gaps
    while preventing an arbitrary inner-fold score scale from changing the
    strength of the distillation target.
    """
    if temperature <= 0:
        raise ValueError("teacher temperature must be positive")
    if len(candidate_ids) < 2 or set(candidate_ids) - set(score_by_candidate):
        raise RuntimeError("candidate-teacher universe mismatch")
    try:
        values = normalise_within_query_logits(np.asarray(
            [score_by_candidate[value] for value in candidate_ids], dtype=np.float64,
        ))
    except ValueError as error:
        raise RuntimeError("invalid candidate-teacher target") from error
    logits = torch.as_tensor(values, device=device, dtype=dtype)
    return F.softmax(logits / temperature, dim=0).detach()


@torch.no_grad()
def encode_tensor(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device,
                  batch_size: int, amp: bool, label: str) -> np.ndarray:
    result = []
    model.eval()
    for left in range(0, len(tensor), batch_size):
        batch = tensor[left:left + batch_size].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            embedding = model(batch)
        result.append(embedding.float().cpu().numpy())
    output = np.concatenate(result).astype(np.float32)
    print(f"[{label}] encoded={len(output):,}", flush=True)
    return output


def fresh_scores(body: dict[str, np.ndarray], query_indices: np.ndarray,
                 query_z: np.ndarray, reference_z: np.ndarray,
                 reference_position: dict[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranks, margins, candidate_scores = [], [], []
    for position, query in enumerate(map(int, query_indices)):
        local = []
        for molecule in query_candidates(body, query):
            left, right = map(int, body["reference_ptr"][molecule:molecule + 2])
            rows = body["reference_rows"][left:right]
            locations = [reference_position[int(row)] for row in rows]
            local.append(float(np.max(reference_z[locations] @ query_z[position])))
        local_array = np.asarray(local, dtype=np.float32)
        ranks.append(1 + int(np.sum(local_array[1:] >= local_array[0])))
        margins.append(float(local_array[0] - np.max(local_array[1:])))
        candidate_scores.append(local_array)
    return (np.asarray(ranks, dtype=np.int16), np.asarray(margins, dtype=np.float32),
            np.asarray(candidate_scores, dtype=object))


def make_batch(body: dict[str, np.ndarray], queries: list[int], roles: list[str],
               forbidden_formulas: set[str], store: ManifestReferenceStore,
               query_tensor: torch.Tensor, references_per_molecule: int,
               rng: np.random.Generator) -> tuple[torch.Tensor, list[dict]]:
    tensors: list[torch.Tensor] = []
    layout = []
    for query, role in zip(queries, roles, strict=True):
        item = {"query": len(tensors), "query_index": query, "role": role}
        tensors.append(query_tensor[query])
        item["reference_groups"] = []
        item["reference_rows"] = []
        item["molecules"] = query_candidates(body, query, forbidden_formulas)
        for molecule in item["molecules"]:
            left, right = map(int, body["reference_ptr"][molecule:molecule + 2])
            available = list(map(int, body["reference_rows"][left:right]))
            best = int(body["molecule_best_row"][molecule])
            selected = [best]
            alternatives = [row for row in available if row != best]
            if alternatives and references_per_molecule > 1:
                count = min(references_per_molecule - 1, len(alternatives))
                draw = rng.choice(len(alternatives), size=count, replace=False)
                selected.extend(alternatives[int(index)] for index in np.atleast_1d(draw))
            positions = list(range(len(tensors), len(tensors) + len(selected)))
            item["reference_groups"].append(positions)
            item["reference_rows"].extend(selected)
            tensors.extend(store.get(selected))
        layout.append(item)
    return torch.stack(tensors), layout


def main() -> None:
    args = arguments()
    started = time.time()
    if not (0 <= args.outer_fold < args.folds):
        raise ValueError("outer fold is outside the frozen range")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    if (args.head_lr < args.backbone_lr or args.epochs < 1
            or args.steps_per_epoch < 1 or args.batch_queries < 2
            or args.references_per_molecule < 1
            or not 0 < args.corrective_fraction < 1
            or args.teacher_temperature <= 0):
        raise ValueError("invalid optimization configuration")
    if args.supervision_mode == "bioaware_soft" and args.candidate_teacher_dir is None:
        raise ValueError("bioaware_soft requires --candidate-teacher-dir")
    if args.arm in DIRECT_ROUTE_ARMS:
        if args.action_router_dir is None:
            raise ValueError(f"{args.arm} requires --action-router-dir")
        if args.supervision_mode != "direct_onehot":
            raise ValueError(f"{args.arm} forbids candidate-score distillation")
    if args.arm in {"matched_graph_direct", "matched_generic_direct"}:
        if args.references_per_molecule != 1:
            raise ValueError(
                "B8 matched arms require exactly one reference per candidate"
            )
    manifest_path = args.manifest_dir / "manifest.npz"
    manifest_report_path = args.manifest_dir / "report.json"
    required = [manifest_path, manifest_report_path,
                args.official_checkpoint, args.architecture_checkpoint]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    manifest_report = json.loads(manifest_report_path.read_text(encoding="utf-8"))
    if manifest_report.get("status") != "bioaware_b4_direct_manifest_frozen":
        raise RuntimeError("wrong B4 manifest status")
    if manifest_report["provenance"].get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("B4 manifest hash mismatch")
    body = load_manifest(manifest_path)

    held = np.flatnonzero(body["formula_fold"] == args.outer_fold)
    train = np.flatnonzero(body["formula_fold"] != args.outer_fold)
    held_formulas = set(map(str, body["query_formula"][held]))
    if held_formulas & set(map(str, body["query_formula"][train])):
        raise RuntimeError("truth formula leakage across folds")
    # A held formula is never allowed to reach a training reference gradient.
    train = np.asarray([
        query for query in train
        if len(query_candidates(body, int(query), held_formulas)) >= 2
    ], dtype=np.int64)
    corrected_key, introduced_key = ARMS[args.arm]
    route_provenance: dict[str, str] = {}
    if args.arm in DIRECT_ROUTE_ARMS:
        corrective, harm, route_provenance = load_direct_action_routes(
            args.action_router_dir, args.arm, args.outer_fold, train, body,
        )
    else:
        routed_corrected_key = str(corrected_key) + "_by_outer"
        routed_introduced_key = str(introduced_key) + "_by_outer"
        corrective = train[body[routed_corrected_key][args.outer_fold, train].astype(bool)]
        harm = train[body[routed_introduced_key][args.outer_fold, train].astype(bool)]
    safety = train[body["baseline_rank"][train] == 1]
    safety = np.unique(np.r_[safety, harm]).astype(np.int64)
    minimum_corrective = 2 if args.smoke else 8
    if len(corrective) < minimum_corrective or not len(safety) or not len(held):
        raise RuntimeError(
            f"insufficient fold data: corrective={len(corrective)} safety={len(safety)} held={len(held)}"
        )
    corrective_pool = by_identity(corrective, body["query_ik14"])
    safety_pool = by_identity(safety, body["query_ik14"])
    harm_pool = by_identity(harm, body["query_ik14"]) if len(harm) else {}
    if (args.arm in {"matched_graph_direct", "matched_generic_direct"}
            and len(corrective_pool) != len(corrective)):
        raise RuntimeError(
            "B8 matched arms require exactly one selected query per identity"
        )
    candidate_teacher: dict[str, dict[str, float]] = {}
    teacher_provenance: dict[str, str] = {}
    if args.supervision_mode == "bioaware_soft":
        candidate_teacher, teacher_provenance = load_candidate_teacher(
            args.candidate_teacher_dir, args.arm, args.outer_fold,
            corrective, body,
        )

    torch.set_num_threads(args.torch_threads)
    seed_everything(args.seed + args.outer_fold)
    device = torch.device(args.device)
    if (not args.smoke and
            (not args.device.startswith("cuda") or not torch.cuda.is_available())):
        raise RuntimeError("formal B4 direct training requires CUDA")
    store = ManifestReferenceStore(
        body["reference_tensor_rows"], body["reference_tensor"],
    )
    # Replay is deliberately performed on *every* manifest query before any
    # fold training.  Its row index must therefore cover the complete manifest,
    # not merely the post-filtered train/held subset.  Some training queries are
    # removed above when held-formula exclusion leaves fewer than two candidates;
    # deriving the index from that subset made otherwise valid replay rows absent.
    reference_rows = store.rows
    reference_position = store.position
    manifest_reference_rows = np.unique(body["reference_rows"].astype(np.int64))
    if not np.array_equal(reference_rows, manifest_reference_rows):
        missing = np.setdiff1d(manifest_reference_rows, reference_rows)
        extra = np.setdiff1d(reference_rows, manifest_reference_rows)
        raise RuntimeError(
            "manifest tensor/reference-row universe mismatch: "
            f"missing={missing[:10].tolist()} extra={extra[:10].tolist()}"
        )
    query_tensor = torch.from_numpy(body["query_tensor"].astype(np.float32))
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        device, args.n_highest_peaks,
    )
    model.eval()
    official_query = encode_tensor(
        model, query_tensor, device, args.batch_size, False, "official-query-fp32",
    )
    frozen_query = body["query_official_embedding"].astype(np.float32)
    query_replay_error = float(np.max(np.abs(official_query - frozen_query)))
    query_replay_min_cosine = float(np.min(np.sum(
        official_query * frozen_query, axis=1,
    )))
    if query_replay_error > 5e-4 or query_replay_min_cosine < 0.99999:
        raise RuntimeError(
            "official external-query replay mismatch: "
            f"max_element_error={query_replay_error} "
            f"min_cosine={query_replay_min_cosine}"
        )
    official_reference = encode_tensor(
        model, store.tensor, device, args.batch_size, False, "official-reference-fp32",
    )
    frozen_reference = body["reference_official_embedding"].astype(np.float32)
    if frozen_reference.shape != official_reference.shape:
        raise RuntimeError("frozen/fresh MoNA reference embedding shape mismatch")
    reference_replay_error = float(np.max(np.abs(
        official_reference - frozen_reference
    )))
    reference_replay_cosine = np.sum(
        official_reference * frozen_reference, axis=1,
    )
    reference_replay_min_cosine = float(np.min(reference_replay_cosine))
    if reference_replay_error > 5e-4 or reference_replay_min_cosine < 0.99999:
        raise RuntimeError(
            "official MoNA reference replay mismatch: "
            f"max_element_error={reference_replay_error} "
            f"min_cosine={reference_replay_min_cosine}"
        )
    all_queries = np.arange(len(body["query_id"]), dtype=np.int64)
    replay_rank, replay_margin, _ = fresh_scores(
        body, all_queries, official_query, official_reference, reference_position,
    )
    rank_mismatches = int(np.sum(replay_rank != body["baseline_rank"]))
    score_errors = []
    for query in all_queries:
        qpos = int(query)
        left, right = map(int, body["query_ptr"][query:query + 2])
        for molecule in range(left, right):
            rleft, rright = map(int, body["reference_ptr"][molecule:molecule + 2])
            locations = [reference_position[int(row)] for row in body["reference_rows"][rleft:rright]]
            computed = float(np.max(official_reference[locations] @ official_query[qpos]))
            score_errors.append(abs(computed - float(body["molecule_official_score"][molecule])))
    maximum_score_error = float(max(score_errors, default=0.0))
    if rank_mismatches or maximum_score_error > 5e-4:
        raise RuntimeError(
            f"official forward mismatch: ranks={rank_mismatches} max_score_error={maximum_score_error}"
        )
    print(json.dumps({
        "status": "bioaware_b4_official_replay_passed",
        "query_max_element_error": query_replay_error,
        "query_min_cosine": query_replay_min_cosine,
        "reference_max_element_error": reference_replay_error,
        "reference_min_cosine": reference_replay_min_cosine,
        "candidate_max_score_error": maximum_score_error,
        "rank_mismatches": rank_mismatches,
    }, indent=2), flush=True)

    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    model.eval()  # dropout stays off while autograd remains enabled
    head = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    head_ids = {id(parameter) for parameter in head}
    backbone = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr, "weight_decay": 0.0},
        {"params": head, "lr": args.head_lr, "weight_decay": args.weight_decay},
    ])
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    # Query/safety scheduling and within-molecule reference draws must not
    # share an RNG stream.  Otherwise a different number of reference spectra
    # in one corrective arm changes all later sampled queries, invalidating a
    # paired router comparison despite an identical seed.
    routing_rng = np.random.default_rng(args.seed + 1009 * args.outer_fold)
    reference_rng = np.random.default_rng(args.seed + 1009 * args.outer_fold + 104729)
    schedule_digest = hashlib.sha256()
    history = []
    corrective_per_batch = max(1, min(args.batch_queries - 1,
        int(round(args.batch_queries * args.corrective_fraction))))
    harm_set = set(map(int, harm))

    for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
        totals = {key: 0.0 for key in (
            "loss", "listwise", "corrective", "safety", "preserve",
            "grad_norm", "clip_fraction",
        )}
        for step in range(1 if args.smoke else args.steps_per_epoch):
            correction_queries = sample_identity_balanced(
                corrective_pool, corrective_per_batch, routing_rng,
            )
            safety_count = args.batch_queries - corrective_per_batch
            safety_queries = []
            # The high-recall strategy has two known harmful interventions.
            # Force them into half of batches when available, rather than
            # relying on a very unlikely draw from hundreds of correct cases.
            if harm_pool and safety_count and float(routing_rng.random()) < 0.5:
                safety_queries.extend(sample_identity_balanced(harm_pool, 1, routing_rng))
            safety_queries.extend(sample_identity_balanced(
                safety_pool, safety_count - len(safety_queries), routing_rng,
            ))
            batch_queries = correction_queries + safety_queries
            roles = ["corrective"] * len(correction_queries) + ["safety"] * len(safety_queries)
            permutation = routing_rng.permutation(len(batch_queries))
            batch_queries = [batch_queries[int(index)] for index in permutation]
            roles = [roles[int(index)] for index in permutation]
            spectra, layout = make_batch(
                body, batch_queries, roles, held_formulas, store, query_tensor,
                args.references_per_molecule, reference_rng,
            )
            schedule_digest.update(json.dumps([
                {
                    "query_id": str(body["query_id"][int(item["query_index"])]),
                    "role": str(item["role"]),
                    "reference_rows": list(map(int, item["reference_rows"])),
                }
                for item in layout
            ], sort_keys=True, separators=(",", ":")).encode("utf-8"))
            spectra = spectra.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=args.amp and device.type == "cuda"):
                z = model(spectra)
                if not bool(torch.isfinite(z).all().item()):
                    raise RuntimeError(
                        f"non-finite embedding at epoch={epoch} step={step}; "
                        f"amp={args.amp}"
                    )
                listwise_losses, correction_losses, safety_losses = [], [], []
                official_batch_vectors = []
                for item in layout:
                    query = int(item["query_index"])
                    qz = z[int(item["query"])]
                    molecule_similarity = []
                    for group in item["reference_groups"]:
                        molecule_similarity.append(torch.max(z[list(map(int, group))] @ qz))
                    similarity = torch.stack(molecule_similarity)
                    margin = similarity[0] - torch.max(similarity[1:])
                    if item["role"] == "corrective":
                        if args.supervision_mode == "bioaware_soft":
                            query_id = str(body["query_id"][query])
                            candidate_ids = [
                                str(body["molecule_id"][molecule])
                                for molecule in item["molecules"]
                            ]
                            target = candidate_teacher_target(
                                candidate_teacher[query_id], candidate_ids,
                                device, similarity.dtype,
                                args.teacher_temperature,
                            )
                            listwise_losses.append(-torch.sum(
                                target * F.log_softmax(
                                    similarity / args.temperature, dim=0,
                                )
                            ))
                        else:
                            listwise_losses.append(-F.log_softmax(
                                similarity / args.temperature, dim=0,
                            )[0])
                        correction_losses.append(F.softplus(
                            (args.rank_margin - margin) / args.temperature,
                        ))
                    else:
                        floor = official_margin(body, query, held_formulas) - args.margin_floor_slack
                        weight = args.harm_safety_weight if query in harm_set else 1.0
                        safety_losses.append(weight * F.relu(
                            torch.as_tensor(floor, device=device, dtype=z.dtype) - margin
                        ))
                    official_batch_vectors.append(torch.from_numpy(official_query[query]))
                    for row in item["reference_rows"]:
                        official_batch_vectors.append(torch.from_numpy(
                            official_reference[reference_position[int(row)]]
                        ))
                official_z = torch.stack(official_batch_vectors).to(device=device, dtype=z.dtype)
                if len(official_z) != len(z):
                    raise RuntimeError("preservation layout mismatch")
                listwise = torch.stack(listwise_losses).mean()
                corrective_loss = torch.stack(correction_losses).mean()
                safety_loss = torch.stack(safety_losses).mean()
                preserve = (1 - torch.sum(z * official_z, dim=1)).mean()
                loss = (args.lambda_listwise * listwise
                        + args.lambda_corrective * corrective_loss
                        + args.lambda_safety_floor * safety_loss
                        + args.lambda_preserve * preserve)
            named_losses = {
                "loss": loss, "listwise": listwise,
                "corrective": corrective_loss, "safety": safety_loss,
                "preserve": preserve,
            }
            nonfinite_losses = [
                name for name, value in named_losses.items()
                if not bool(torch.isfinite(value).item())
            ]
            if nonfinite_losses:
                raise RuntimeError(
                    f"non-finite training objective at epoch={epoch} step={step}: "
                    f"{nonfinite_losses}; amp={args.amp}"
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nonfinite_gradients = [
                name for name, parameter in model.named_parameters()
                if (parameter.requires_grad and parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all().item()))
            ]
            if nonfinite_gradients:
                raise RuntimeError(
                    f"non-finite gradients at epoch={epoch} step={step}: "
                    f"{nonfinite_gradients[:8]}; amp={args.amp}"
                )
            norm = float(torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.grad_clip,
            ))
            if not np.isfinite(norm):
                raise RuntimeError(
                    f"non-finite gradient norm at epoch={epoch} step={step}; "
                    f"amp={args.amp}"
                )
            scaler.step(optimizer)
            scaler.update()
            for key, value in (
                ("loss", loss), ("listwise", listwise),
                ("corrective", corrective_loss), ("safety", safety_loss),
                ("preserve", preserve),
            ):
                totals[key] += float(value.detach())
            totals["grad_norm"] += norm
            totals["clip_fraction"] += float(norm > args.grad_clip)
        divisor = 1 if args.smoke else args.steps_per_epoch
        record = {key: value / divisor for key, value in totals.items()}
        record["epoch"] = epoch
        history.append(record)
        print(f"[B4 {args.arm} fold={args.outer_fold}] epoch={epoch} {record}", flush=True)

    # The held fold is touched exactly once, after the fixed training budget.
    adapted_query = encode_tensor(
        model, query_tensor[held], device, args.batch_size, False, "held-query-fp32",
    )
    held_reference_rows: set[int] = set()
    for query in held:
        for molecule in query_candidates(body, int(query)):
            left, right = map(int, body["reference_ptr"][molecule:molecule + 2])
            held_reference_rows.update(map(int, body["reference_rows"][left:right]))
    held_reference_rows_array = np.asarray(sorted(held_reference_rows), dtype=np.int64)
    held_reference_tensor = store.get(held_reference_rows_array)
    adapted_reference = encode_tensor(
        model, held_reference_tensor, device, args.batch_size, False, "held-reference-fp32",
    )
    held_reference_position = {
        int(row): position for position, row in enumerate(held_reference_rows_array)
    }
    new_rank, new_margin, _ = fresh_scores(
        body, held, adapted_query, adapted_reference, held_reference_position,
    )
    old_rank = body["baseline_rank"][held].astype(np.int16)
    corrected = (old_rank != 1) & (new_rank == 1)
    introduced = (old_rank == 1) & (new_rank != 1)
    query_preservation = np.sum(adapted_query * official_query[held], axis=1)
    official_ref_for_held = np.stack([
        official_reference[reference_position[int(row)]] for row in held_reference_rows_array
    ])
    reference_preservation = np.sum(adapted_reference * official_ref_for_held, axis=1)
    preservation = float(np.mean(np.r_[query_preservation, reference_preservation]))
    per_query = pd.DataFrame({
        "query_index": held,
        "query_id": body["query_id"][held],
        "unit_id": body["unit_id"][held],
        "truth_ik14": body["query_ik14"][held],
        "truth_formula": body["query_formula"][held],
        "outer_fold": args.outer_fold,
        "arm": args.arm,
        "supervision_mode": args.supervision_mode,
        "old_rank": old_rank,
        "new_rank": new_rank,
        "old_margin": replay_margin[held],
        "new_margin": new_margin,
        "corrected": corrected,
        "introduced": introduced,
    })
    args.output.mkdir(parents=True)
    per_query.to_csv(args.output / "held_per_query.csv.gz", index=False, compression="gzip")
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    trainable_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items() if name in trainable_names
    }
    checkpoint = {
        "status": "bioaware_b4_direct_shared_embedding_fold",
        "format": "official_plus_trainable_parameter_state_v1",
        "trainable_state": trainable_state,
        "trainable_parameter_names": sorted(trainable_names),
        "arm": args.arm,
        "supervision_mode": args.supervision_mode,
        "outer_fold": args.outer_fold,
        "seed": args.seed,
        "inference_clean_spectrum_only": True,
        "query_reference_encoder_shared": True,
        "candidate_inputs_at_inference": False,
        "bioaware_context_used_at_inference": False,
        "bioaware_context_role": "training_example_router_only",
        "P2b_used": False,
        "formal": False,
        "validation_pass": False,
        "provenance": {
            "manifest_sha256": sha256_file(manifest_path),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "raw_checkpoint_sha256": sha256_file(args.architecture_checkpoint),
        },
    }
    torch.save(checkpoint, args.output / "final_trainable_state.pt")
    report = {
        "status": "bioaware_b4_direct_shared_embedding_fold_complete",
        "formal": False,
        "arm": args.arm,
        "supervision_mode": args.supervision_mode,
        "outer_fold": args.outer_fold,
        "held_queries": int(len(held)),
        "held_formulas": int(len(held_formulas)),
        "train_queries": int(len(train)),
        "train_corrective_queries": int(len(corrective)),
        "train_corrective_identities": int(len(corrective_pool)),
        "train_safety_queries": int(len(safety)),
        "train_known_harm_queries": int(len(harm)),
        "baseline": {
            "recall1": float(np.mean(old_rank == 1)),
            "mrr": float(np.mean(1.0 / old_rank)),
        },
        "shared_embedding": {
            "recall1": float(np.mean(new_rank == 1)),
            "mrr": float(np.mean(1.0 / new_rank)),
            "delta_recall1": float(np.mean(new_rank == 1) - np.mean(old_rank == 1)),
            "delta_mrr": float(np.mean(1.0 / new_rank) - np.mean(1.0 / old_rank)),
            "corrected": int(corrected.sum()),
            "introduced": int(introduced.sum()),
            "risk_net_lambda2": int(corrected.sum() - 2 * introduced.sum()),
            "preservation": preservation,
        },
        "official_replay": {
            "rank_mismatches": rank_mismatches,
            "maximum_candidate_score_error": maximum_score_error,
            "reference_maximum_element_error": reference_replay_error,
            "reference_minimum_cosine": reference_replay_min_cosine,
            "query_maximum_element_error": query_replay_error,
            "query_minimum_cosine": query_replay_min_cosine,
        },
        "capacity": capacity,
        "optimization": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "history": history,
        "training_schedule_sha256": schedule_digest.hexdigest(),
        "numeric_integrity": {
            "all_finite": True,
            "amp_enabled": bool(args.amp and device.type == "cuda"),
        },
        "contracts": {
            "held_formula_disjoint": True,
            "held_formula_candidate_references_excluded_from_training": True,
            "outer_formula_fold_excluded_from_router_fit": True,
            "training_router_inner_formula_crossfit": True,
            "held_fold_used_for_selection": False,
            "same_raw_encoder_query_reference": True,
            "dropout_off": True,
            "reaction_neighbours_used_as_positives": False,
            "P2b_used": False,
        },
        "provenance": {
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_report_sha256": sha256_file(manifest_report_path),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "architecture_checkpoint_sha256": sha256_file(args.architecture_checkpoint),
            "script_sha256": sha256_file(Path(__file__)),
            **teacher_provenance,
            **route_provenance,
        },
        "runtime_seconds": time.time() - started,
        "claim_limit": "One opened-cohort formula fold; pooled OOF and independent validation are required.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "arm": args.arm, "fold": args.outer_fold,
        "supervision_mode": args.supervision_mode,
        "baseline": report["baseline"], "shared_embedding": report["shared_embedding"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
