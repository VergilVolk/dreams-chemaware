"""E4-A: direct peak-noise fine-tuning of the shared DreaMS embedding.

This stage deliberately has no P2b score or post-embedding reranker.  It has
two mutually explicit action-selection modes.  ``fixed`` uses two previously
frozen S3A policies exactly like image augmentations:

* candidate_gradient, attenuation 0.50, terminal step 6;
* role_confounder, attenuation 1.00, terminal step 5.

``outcome_mined`` uses one correction-producing raw-spectrum intervention per
query, selected only inside the non-held formula partition from the frozen
S3A+A4 action matrix.  Outcome fields select the training augmentation and are
then stripped; they never become a loss, sample weight, evaluation input or
inference input.

For a training query the clean and perturbed spectra are both encoded by the
same trainable DreaMS model.  Positive and negative reference spectra are also
encoded by that model.  The objective combines clean groupwise ranking,
perturbed-view groupwise ranking, clean/perturbed consistency, an official
margin floor and clean-embedding preservation.  At inference the saved model
receives only an ordinary clean spectrum and emits one new embedding.

Formula folds are fixed before training.  The held formula fold is evaluated
once after a fixed epoch count and is never used for checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import (  # noqa: E402
    CandidateGraph, json_dump, load_embedding_cache, seed_everything,
    sha256_file, stable_fold, strict_rank,
)
from train_e1_identity import load_base_model, torch_load_compat  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore, encode_rows, evaluate_embeddings, formula_bootstrap_delta,
    forward_embeddings, margins, parse_controls, parse_path, representatives,
)
from noise_v3_core import attenuate_sequence  # noqa: E402
from audit_noise_final_positive_guided_matrix import (  # noqa: E402
    apply_action as apply_positive_intensity_action,
    reference_profile,
)
from audit_noise_final_positive_peak_transfer import (  # noqa: E402
    apply_transfer as apply_positive_peak_transfer,
    recurrent_missing_peaks,
)


FIXED_POLICY = {
    "candidate": (("candidate_gradient", 0.50, 6),),
    "confounder": (("role_confounder", 1.00, 5),),
    "combined": (
        ("candidate_gradient", 0.50, 6),
        ("role_confounder", 1.00, 5),
    ),
    # Every cell below already exists in the frozen, outcome-free R0 table.
    # This is a dose curriculum, not per-query post-outcome action selection.
    "curriculum": (
        ("candidate_gradient", 0.50, 3),
        ("candidate_gradient", 0.50, 4),
        ("candidate_gradient", 0.50, 5),
        ("candidate_gradient", 0.50, 6),
        ("role_confounder", 1.00, 1),
        ("role_confounder", 1.00, 2),
        ("role_confounder", 1.00, 3),
        ("role_confounder", 1.00, 4),
        ("role_confounder", 1.00, 5),
    ),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--r0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a")
    parser.add_argument(
        "--positive-manifest-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_pn_positive_manifest",
        help="Frozen strict cross-condition P-arm manifest. Used only when positive-stream-weight > 0.",
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument(
        "--initial-student-checkpoint", type=Path, default=None,
        help=(
            "Optional mature E4-A checkpoint for residual continuation. Its outer fold "
            "must equal --outer-fold; preservation and incremental evaluation are then "
            "anchored to this checkpoint while official DreaMS remains the primary baseline."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/validation/g8r_noise_final_e4a_direct")
    parser.add_argument(
        "--causal-arm",
        choices=("legacy", "clean_duplicate", "matched_random", "targeted"),
        default="legacy",
        help=(
            "Strict E4-A attribution factor. clean_duplicate repeats the clean query as "
            "the action view; matched_random uses one of the two frozen R0 matched-control "
            "paths selected without outcomes; targeted uses the frozen R0 target path."
        ),
    )
    parser.add_argument("--policy", choices=tuple(FIXED_POLICY), default="candidate")
    parser.add_argument(
        "--action-selection", choices=("fixed", "outcome_mined"), default="fixed",
        help=(
            "fixed uses globally preregistered S3A cells; outcome_mined uses one "
            "train-fold correction-mined raw-spectrum action per query."
        ),
    )
    parser.add_argument(
        "--outcome-action-dir", type=Path, default=None,
        help="Directory containing report.json and corrective_teacher_actions.csv.gz for outcome_mined mode.",
    )
    parser.add_argument("--action-scope", choices=("errors", "all"), default="errors")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-actions", type=int, default=4)
    parser.add_argument("--views-per-identity", type=int, default=2)
    parser.add_argument(
        "--error-views-per-identity", type=int, default=0,
        help="Additional N-arm views per baseline-error identity; zero preserves the validated baseline sampler.",
    )
    parser.add_argument("--positive-spectra", type=int, default=4)
    parser.add_argument("--negative-molecules", type=int, default=8)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--head-lr", type=float, default=5e-6)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lambda-clean-rank", type=float, default=1.0)
    parser.add_argument("--lambda-aug-rank", type=float, default=1.0)
    parser.add_argument("--lambda-consistency", type=float, default=0.25)
    parser.add_argument(
        "--direct-transfer-mode",
        choices=("symmetric", "student_action_stopgrad", "official_action"),
        default="symmetric",
        help=(
            "How the clean spectrum receives the action-view signal. symmetric preserves "
            "the historical E4-A objective; student_action_stopgrad prevents the clean "
            "target from being erased by the consistency gradient; official_action uses "
            "the frozen official encoder's action embedding as a fixed training-only target."
        ),
    )
    parser.add_argument(
        "--rank-reference-mode", choices=("shared", "official"), default="shared",
        help=(
            "shared preserves historical end-to-end rank gradients. official anchors rank "
            "losses to frozen official reference embeddings so the optimizer cannot solve "
            "a corrective query merely by moving its candidate references."
        ),
    )
    parser.add_argument("--lambda-margin-floor", type=float, default=2.0)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--margin-floor-slack", type=float, default=0.005)
    parser.add_argument(
        "--safety-ratio", type=float, default=1.0,
        help="Number of safety examples per action example; changes coverage, not loss magnitude.",
    )
    parser.add_argument(
        "--safety-stream-weight", type=float, default=1.0,
        help="Explicit multiplier on the mean safety loss. Use this, not safety-ratio, to strengthen safety gradients.",
    )
    parser.add_argument("--positive-stream-weight", type=float, default=0.0)
    parser.add_argument("--positive-ratio", type=float, default=1.0)
    parser.add_argument("--positive-views-per-identity", type=int, default=2)
    parser.add_argument("--lambda-positive-rank", type=float, default=1.0)
    parser.add_argument("--lambda-positive-margin-floor", type=float, default=2.0)
    parser.add_argument(
        "--guided-noise-policy", choices=("none", "intensity", "transfer", "both", "selected"),
        default="none",
        help="Fixed real-positive-guided peak-noise stream; independent of the legacy P pair stream.",
    )
    parser.add_argument(
        "--guided-query-scope", choices=("positive_deficit_errors", "all"),
        default="positive_deficit_errors",
        help=(
            "Queries receiving the fixed guided action. all is an outcome-free noise "
            "augmentation and requires a formal action-matrix authorization."
        ),
    )
    parser.add_argument(
        "--guided-intensity-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_positive_guided_matrix",
    )
    parser.add_argument(
        "--guided-transfer-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_positive_peak_transfer",
    )
    parser.add_argument(
        "--guided-action-authorization-dir", type=Path, default=None,
        help=(
            "Formal action-matrix artifact authorizing a non-historical guided recipe. "
            "E13 uses the frozen E12-B top3 recurrence result here."
        ),
    )
    parser.add_argument(
        "--guided-crossfit-root", type=Path, default=None,
        help=(
            "Root containing fold_0..fold_4 E14 selected_actions.csv.gz and report.json. "
            "Required only for guided-noise-policy=selected. The current outer fold is "
            "excluded fail-closed."
        ),
    )
    parser.add_argument(
        "--guided-reference-checkpoint", type=Path, default=None,
        help=(
            "Frozen shared encoder used only to choose top3 real same-identity reference "
            "spectra. Required by the E12-B-authorized recipe."
        ),
    )
    parser.add_argument(
        "--error-signatures", type=Path,
        default=ROOT / "data/validation/g8r_real_error_analysis/query_error_signatures.csv.gz",
    )
    parser.add_argument("--guided-noise-weight", type=float, default=1.0)
    parser.add_argument(
        "--guided-auto-balance", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Scale the selected P branch once at initialization so its gradient norm "
            "does not exceed the combined mature N+safety gradient norm."
        ),
    )
    parser.add_argument("--guided-noise-ratio", type=float, default=1.0)
    parser.add_argument("--guided-noise-views-per-identity", type=int, default=2)
    parser.add_argument(
        "--guided-risk-control-ratio", type=float, default=0.0,
        help=(
            "Action-specific mature-clean-correct controls per selected corrective "
            "example. Available only in E14 selected mode."
        ),
    )
    parser.add_argument("--lambda-guided-transfer", type=float, default=0.50)
    parser.add_argument(
        "--lambda-guided-teacher-margin", type=float, default=0.0,
        help="Weight of the frozen crossfit teacher-margin floor in selected E14 mode.",
    )
    parser.add_argument("--guided-teacher-margin-cap", type=float, default=0.20)
    parser.add_argument(
        "--guided-teacher-target-mode", choices=("absolute", "delta"),
        default="absolute",
        help=(
            "absolute replays the action margin; delta transfers only a conservative "
            "fraction of action-minus-clean margin."
        ),
    )
    parser.add_argument("--guided-teacher-delta-fraction", type=float, default=0.50)
    parser.add_argument("--guided-teacher-delta-cap", type=float, default=0.20)
    parser.add_argument(
        "--guided-transfer-mode", choices=("stopgrad", "symmetric"), default="stopgrad",
        help=(
            "Gradient convention for clean/action consistency in the guided P stream. "
            "symmetric updates both branches of the one shared encoder; stopgrad is the "
            "historical E7 mechanism control."
        ),
    )
    parser.add_argument(
        "--guided-recurrence-prevalence", type=float, default=0.67,
        help="Minimum positive-reference prevalence for a recurrent missing peak.",
    )
    parser.add_argument(
        "--guided-recurrence-max-peaks", type=int, default=5,
        help="Maximum recurrent positive peaks transferred into one action view.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--run-suffix", default="",
        help="Optional filesystem-safe suffix for preregistered optimizer scans.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class DirectExample:
    query_index: int
    query_row: int
    identity: str
    formula: str
    positive_rows: tuple[int, ...]
    negative_rows: tuple[int, ...]
    official_margin: float
    official_rank: int
    sample_weight: float
    policy: str = "clean_safety"
    target_path: tuple[int, ...] = ()
    attenuation: float = 0.0


@dataclass(frozen=True)
class GuidedNoiseExample:
    """Outcome-free, real-positive-guided training augmentation."""

    query_index: int
    query_row: int
    identity: str
    formula: str
    positive_rows: tuple[int, ...]
    negative_rows: tuple[int, ...]
    action_reference_rows: tuple[int, ...]
    official_margin: float
    official_rank: int
    sample_weight: float
    policy: str
    family: str
    dose: float
    auxiliary_dose: float = 0.0
    recurrence_prevalence: float = 0.67
    recurrence_max_peaks: int = 5
    support_weighted: bool = False
    teacher_margin: float = float("nan")
    teacher_margin_delta: float = float("nan")
    supervision_kind: str = "corrective"


def strict_bool(series: pd.Series, name: str) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise RuntimeError(f"{name} is not a strict boolean column")
    return normalized.isin({"true", "1"}).to_numpy(bool)


def parse_reference_rows(value: object) -> tuple[int, ...]:
    rows = tuple(int(part) for part in str(value).split(";") if str(part).strip())
    if not rows or len(rows) != len(set(rows)):
        raise RuntimeError("guided action reference rows must be non-empty and unique")
    return rows


def official_rank_margin(graph: CandidateGraph) -> tuple[np.ndarray, np.ndarray]:
    score = graph.features[:, graph.dreams_column]
    molecule_score = np.maximum.reduceat(score, graph.molecule_ptr[:-1])
    rank = np.empty(graph.n_queries, dtype=np.int16)
    margin = np.empty(graph.n_queries, dtype=np.float32)
    for query in range(graph.n_queries):
        left, right = map(int, graph.query_ptr[query:query + 2])
        values = molecule_score[left:right]
        if len(values) < 2 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"invalid frozen candidate scores for query {query}")
        rank[query] = strict_rank(values)
        margin[query] = float(values[0] - np.max(values[1:]))
    return rank, margin


def _stable_control_index(row: pd.Series) -> int:
    """Choose one of two frozen matched controls without using any outcome."""
    key = (
        f"{int(row['query_index'])}|{str(row['selector'])}|"
        f"{float(row['attenuation']):.8f}|{int(row['step'])}|e4a-causal-v1"
    )
    return int(hashlib.sha256(key.encode("utf-8")).digest()[0] & 1)


def materialize_causal_arm(
    actions: pd.DataFrame, arm: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Materialize the sole experimental difference in the attribution trial.

    The returned table retains every row, query, policy label, hard negative and
    sampling key.  Only ``target_path`` changes.  This guarantees that the
    identity-balanced sampler and all candidate references are arm-invariant.
    """
    if arm == "legacy":
        return actions, {"arm": arm, "rows": int(len(actions))}
    required = {
        "query_index", "selector", "attenuation", "step", "target_path",
        "matched_control_paths", "hard_negative_row", "query_ik14", "query_formula",
    }
    missing = required - set(actions.columns)
    if missing:
        raise RuntimeError(f"causal attribution actions lack columns: {sorted(missing)}")
    output = actions.copy()
    control_indices: list[int] = []
    selected_paths: list[str] = []
    target_paths: list[tuple[int, ...]] = []
    for _, row in output.iterrows():
        target = parse_path(row["target_path"])
        controls = parse_controls(row["matched_control_paths"])
        step = int(row["step"])
        if len(target) != step or any(len(path) != step for path in controls):
            raise RuntimeError("target/control path length does not match the frozen step")
        if target in controls or controls[0] == controls[1]:
            raise RuntimeError("matched-control paths are not distinct from the target and each other")
        target_paths.append(target)
        if arm == "targeted":
            control_indices.append(-1)
            selected_paths.append(",".join(map(str, target)))
        elif arm == "matched_random":
            index = _stable_control_index(row)
            control_indices.append(index)
            selected_paths.append(",".join(map(str, controls[index])))
        elif arm == "clean_duplicate":
            control_indices.append(-1)
            selected_paths.append("")
        else:  # pragma: no cover - argparse and tests protect this branch
            raise ValueError(f"unknown causal arm: {arm}")
    output["target_path"] = selected_paths
    output["causal_control_index"] = np.asarray(control_indices, dtype=np.int8)
    canonical = "\n".join(
        f"{int(row.query_index)}|{row.selector}|{float(row.attenuation):.8f}|"
        f"{int(row.step)}|{row.target_path}|{int(row.causal_control_index)}"
        for row in output[[
            "query_index", "selector", "attenuation", "step", "target_path",
            "causal_control_index",
        ]].itertuples(index=False)
    )
    audit: dict[str, object] = {
        "arm": arm,
        "rows": int(len(output)),
        "queries": int(output["query_index"].nunique()),
        "identities": int(output["query_ik14"].astype(str).nunique()),
        "formulas": int(output["query_formula"].astype(str).nunique()),
        "materialized_path_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "target_rows_preserved": bool(len(target_paths) == len(output)),
    }
    if arm == "matched_random":
        counts = output["causal_control_index"].value_counts().sort_index()
        audit["matched_control_index_counts"] = {
            str(int(index)): int(value) for index, value in counts.items()
        }
        if set(map(int, counts.index)) != {0, 1}:
            raise RuntimeError("deterministic matched-control assignment did not use both controls")
        imbalance = abs(int(counts.loc[0]) - int(counts.loc[1])) / max(len(output), 1)
        audit["matched_control_assignment_imbalance"] = float(imbalance)
        # Small smoke panels have ordinary binomial variation; the tolerance
        # contracts to 2% for the full R0 ledger without outcome-dependent retry.
        allowed_imbalance = max(0.02, 2.5 / math.sqrt(max(len(output), 1)))
        audit["matched_control_assignment_max_imbalance"] = float(allowed_imbalance)
        if imbalance > allowed_imbalance:
            raise RuntimeError(
                "matched-control assignment is unexpectedly imbalanced: "
                f"{imbalance:.4f} > {allowed_imbalance:.4f}"
            )
    return output, audit


def full_graph_query_details(
    graph: CandidateGraph, rows: np.ndarray, encoded: np.ndarray, queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strict rank, local top molecule and positive-vs-best-negative margin."""
    position = {int(row): index for index, row in enumerate(rows)}
    qpos = np.asarray([position[int(row)] for row in graph.query_row], dtype=np.int64)
    cpos = np.asarray([position[int(row)] for row in graph.pair_candidate_row], dtype=np.int64)
    molecule_query = np.repeat(np.arange(graph.n_queries), np.diff(graph.query_ptr))
    pair_query = np.repeat(molecule_query, np.diff(graph.molecule_ptr))
    pair_score = np.einsum("ij,ij->i", encoded[qpos[pair_query]], encoded[cpos])
    molecule_score = np.maximum.reduceat(pair_score, graph.molecule_ptr[:-1])
    ranks: list[int] = []
    top_local: list[int] = []
    margins_out: list[float] = []
    for query in queries:
        left, right = map(int, graph.query_ptr[int(query):int(query) + 2])
        values = molecule_score[left:right]
        if len(values) < 2 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"invalid full-graph molecule scores for query {int(query)}")
        ranks.append(strict_rank(values))
        top_local.append(int(np.argmax(values)))
        margins_out.append(float(values[0] - np.max(values[1:])))
    return (
        np.asarray(ranks, dtype=np.int16),
        np.asarray(top_local, dtype=np.int32),
        np.asarray(margins_out, dtype=np.float32),
    )


def unfreeze_last_blocks(model, blocks: int) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    encoder = model.backbone.transformer_encoder
    if blocks < 1 or blocks > int(encoder.n_layers):
        raise ValueError(f"unfreeze-blocks must be in 1..{int(encoder.n_layers)}")
    layers = list(range(int(encoder.n_layers) - blocks, int(encoder.n_layers)))
    count = 0
    for layer in layers:
        for module in (encoder.atts[layer], encoder.ffs[layer],
                       encoder.scales[2 * layer], encoder.scales[2 * layer + 1]):
            for parameter in module.parameters():
                if not parameter.requires_grad:
                    parameter.requires_grad = True
                    count += parameter.numel()
    if getattr(encoder, "pre_norm", False):
        for parameter in encoder.scales[-1].parameters():
            if not parameter.requires_grad:
                parameter.requires_grad = True
                count += parameter.numel()
    return {
        "transformer_layers": int(encoder.n_layers),
        "unfrozen_layers": layers,
        "unfrozen_backbone_parameters": int(count),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
    }


def make_examples(graph: CandidateGraph, frame: pd.DataFrame, rank: np.ndarray,
                  margin: np.ndarray, positives: int, negatives: int,
                  action: bool) -> list[DirectExample]:
    if frame.empty:
        return []
    output: list[DirectExample] = []
    for row in frame.itertuples(index=False):
        query = int(row.query_index)
        forced = int(row.hard_negative_row) if action and hasattr(row, "hard_negative_row") else None
        positive_rows, negative_rows = representatives(
            graph, query, positives, negatives, forced,
        )
        identity = str(row.query_ik14)
        output.append(DirectExample(
            query_index=query,
            query_row=int(row.query_row),
            identity=identity,
            formula=str(row.query_formula),
            positive_rows=positive_rows,
            negative_rows=negative_rows,
            official_margin=float(margin[query]),
            official_rank=int(rank[query]),
            # Exact identity balance is imposed by identity_balanced_epoch.
            sample_weight=1.0,
            policy=(
                f"{row.selector}|step={int(row.step)}" if action
                else "clean_safety"
            ),
            target_path=parse_path(row.target_path) if action else (),
            attenuation=float(row.attenuation) if action else 0.0,
        ))
    return output


def make_positive_examples(graph: CandidateGraph, frame: pd.DataFrame,
                           rank: np.ndarray, margin: np.ndarray,
                           negatives: int) -> list[DirectExample]:
    """Build P-arm examples with one explicit real cross-condition positive."""
    output: list[DirectExample] = []
    for row in frame.itertuples(index=False):
        query = int(row.query_index)
        _, negative_rows = representatives(graph, query, 1, negatives, None)
        output.append(DirectExample(
            query_index=query,
            query_row=int(row.query_row),
            identity=str(row.query_ik14),
            formula=str(row.query_formula),
            positive_rows=(int(row.positive_row),),
            negative_rows=negative_rows,
            official_margin=float(margin[query]),
            official_rank=int(rank[query]),
            sample_weight=1.0,
            policy=f"positive|{row.relation}",
        ))
    return output


def make_guided_noise_examples(
    graph: CandidateGraph, frame: pd.DataFrame, rank: np.ndarray, margin: np.ndarray,
    positives: int, negatives: int,
) -> list[GuidedNoiseExample]:
    output: list[GuidedNoiseExample] = []
    for row in frame.itertuples(index=False):
        query = int(row.query_index)
        if hasattr(row, "teacher_positive_row") and hasattr(row, "teacher_hard_negative_row"):
            # E14 stores the exact spectra that defined its molecular margin.
            # Replaying a different candidate subset silently changes the target.
            positive_rows = (int(row.teacher_positive_row),)
            negative_rows = (int(row.teacher_hard_negative_row),)
        else:
            positive_rows, negative_rows = representatives(
                graph, query, positives, negatives, None,
            )
        baseline_margin = float(getattr(
            row, "teacher_pair_clean_margin",
            getattr(row, "crossfit_clean_margin", margin[query]),
        ))
        baseline_rank = int(getattr(row, "crossfit_clean_rank", rank[query]))
        output.append(GuidedNoiseExample(
            query_index=query,
            query_row=int(row.query_row),
            identity=str(row.query_ik14),
            formula=str(row.query_formula),
            positive_rows=positive_rows,
            negative_rows=negative_rows,
            action_reference_rows=parse_reference_rows(row.positive_reference_rows),
            official_margin=baseline_margin,
            official_rank=baseline_rank,
            sample_weight=1.0,
            policy=str(row.guided_policy),
            family=str(row.guided_family),
            dose=float(row.guided_dose),
            auxiliary_dose=float(getattr(row, "guided_auxiliary_dose", 0.0)),
            recurrence_prevalence=float(getattr(
                row, "guided_recurrence_prevalence", 0.67,
            )),
            recurrence_max_peaks=int(getattr(
                row, "guided_recurrence_max_peaks", 5,
            )),
            support_weighted=bool(getattr(row, "guided_support_weighted", False)),
            teacher_margin=float(getattr(row, "teacher_margin", float("nan"))),
            teacher_margin_delta=float(getattr(
                row, "teacher_margin_delta", float("nan"),
            )),
            supervision_kind=str(getattr(row, "control_kind", "corrective")),
        ))
    return output


def guided_variant(
    store: SpectrumStore, example: GuidedNoiseExample,
    recurrence_prevalence: float = 0.67, recurrence_max_peaks: int = 5,
) -> torch.Tensor:
    clean = store.one(example.query_row)
    references = [store.one(row) for row in example.action_reference_rows]
    prevalence, target = reference_profile(clean, references, 0.02)
    recurrence_prevalence = float(example.recurrence_prevalence)
    recurrence_max_peaks = int(example.recurrence_max_peaks)
    if not np.isfinite(recurrence_prevalence) or not 0 < recurrence_prevalence <= 1:
        raise RuntimeError("guided recurrence prevalence is invalid")
    if recurrence_max_peaks < 1:
        raise RuntimeError("guided recurrence maximum is invalid")
    if example.family in {"consensus_projection", "matched_intensity_transport"}:
        return apply_positive_intensity_action(
            clean, prevalence, target, example.family, example.dose,
        )
    if example.family in {
        "recurrent_union_mix", "recurrent_peak_graft", "balanced_peak_exchange",
    }:
        missing = recurrent_missing_peaks(
            clean, references, 0.02,
            recurrence_prevalence, recurrence_max_peaks,
        )
        if example.support_weighted and len(missing):
            missing = np.asarray(missing, dtype=np.float32).copy()
            missing[:, 1] *= missing[:, 2]
        variant, _ = apply_positive_peak_transfer(
            clean, missing, prevalence, example.family, example.dose,
        )
        return variant
    if example.family in {"transport_then_union", "consensus_then_union"}:
        first_family = (
            "matched_intensity_transport"
            if example.family == "transport_then_union"
            else "consensus_projection"
        )
        intensity = apply_positive_intensity_action(
            clean, prevalence, target, first_family, example.dose,
        )
        missing = recurrent_missing_peaks(
            clean, references, 0.02,
            recurrence_prevalence, recurrence_max_peaks,
        )
        variant, _ = apply_positive_peak_transfer(
            intensity, missing, prevalence, "recurrent_union_mix",
            example.auxiliary_dose,
        )
        return variant
    raise RuntimeError(f"unregistered guided action family: {example.family}")


def flatten_guided(
    store: SpectrumStore, examples: list[GuidedNoiseExample],
    recurrence_prevalence: float = 0.67, recurrence_max_peaks: int = 5,
):
    tensors: list[torch.Tensor] = []
    clean_rows: list[int] = []
    layout: list[dict] = []
    for example in examples:
        item: dict[str, object] = {"clean": len(tensors)}
        tensors.append(store.one(example.query_row))
        clean_rows.append(example.query_row)
        item["action"] = len(tensors)
        tensors.append(guided_variant(
            store, example, recurrence_prevalence, recurrence_max_peaks,
        ))
        item["positive"] = list(range(len(tensors), len(tensors) + len(example.positive_rows)))
        tensors.extend(store.get(example.positive_rows))
        clean_rows.extend(example.positive_rows)
        item["negative"] = list(range(len(tensors), len(tensors) + len(example.negative_rows)))
        tensors.extend(store.get(example.negative_rows))
        clean_rows.extend(example.negative_rows)
        layout.append(item)
    return torch.stack(tensors), layout, clean_rows


def guided_consistency_values(
    clean_z: torch.Tensor, action_z: torch.Tensor, mode: str,
) -> torch.Tensor:
    if mode not in {"stopgrad", "symmetric"}:
        raise ValueError(f"unknown guided transfer mode: {mode}")
    action_target = action_z if mode == "symmetric" else action_z.detach()
    return 1.0 - torch.sum(clean_z * action_target, dim=1)


def guided_noise_loss(
    model, store: SpectrumStore, examples: list[GuidedNoiseExample],
    official_by_row: dict[int, np.ndarray], device: torch.device, args,
) -> tuple[torch.Tensor, dict[str, float]]:
    spectra, layout, clean_rows = flatten_guided(
        store, examples,
        args.guided_recurrence_prevalence,
        args.guided_recurrence_max_peaks,
    )
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    clean_margin = margins(encoded, layout, "clean")
    action_margin = margins(encoded, layout, "action")
    clean_rank_each = F.softplus((args.rank_margin - clean_margin) / args.temperature)
    action_rank_each = F.softplus((args.rank_margin - action_margin) / args.temperature)
    clean_z = torch.stack([encoded[int(item["clean"])] for item in layout])
    action_z = torch.stack([encoded[int(item["action"])] for item in layout])
    # E8 established that symmetric clean/action training transfers ranking
    # information more effectively than a detached action branch.  E13 keeps
    # the old stop-gradient form only as an explicit mechanism control.
    consistency_each = guided_consistency_values(
        clean_z, action_z, args.guided_transfer_mode,
    )
    teacher_margin = torch.tensor(
        [example.teacher_margin for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    baseline_teacher_margin = torch.tensor(
        [example.official_margin for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    teacher_margin_delta = torch.tensor(
        [example.teacher_margin_delta for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    has_fixed_teacher = torch.isfinite(teacher_margin)
    if args.guided_teacher_target_mode == "absolute":
        teacher_target = torch.clamp(
            teacher_margin, min=0.0,
            max=args.guided_teacher_margin_cap,
        )
    elif args.guided_teacher_target_mode == "delta":
        observed_delta = torch.where(
            torch.isfinite(teacher_margin_delta),
            teacher_margin_delta,
            teacher_margin - baseline_teacher_margin,
        )
        conservative_delta = args.guided_teacher_delta_fraction * torch.clamp(
            observed_delta, min=0.0, max=args.guided_teacher_delta_cap,
        )
        teacher_target = baseline_teacher_margin + conservative_delta
    else:
        raise RuntimeError(f"unknown guided teacher target mode: {args.guided_teacher_target_mode}")
    teacher_target = torch.where(
        has_fixed_teacher, teacher_target, clean_margin.detach(),
    )
    self_transfer = F.relu(action_margin.detach() - clean_margin)
    fixed_transfer = F.relu(teacher_target - clean_margin)
    fixed_teacher_each = torch.where(
        has_fixed_teacher, fixed_transfer, torch.zeros_like(fixed_transfer),
    )
    floors = torch.tensor(
        [example.official_margin - args.margin_floor_slack for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    floor_each = F.relu(floors - clean_margin)
    clean_indices: list[int] = []
    for item in layout:
        clean_indices.append(int(item["clean"]))
        clean_indices.extend(item["positive"])
        clean_indices.extend(item["negative"])
    official = torch.from_numpy(np.stack([official_by_row[row] for row in clean_rows])).to(device)
    preserve = (1.0 - torch.sum(encoded[clean_indices] * official, dim=1)).mean()
    clean_rank = weighted_mean(clean_rank_each, examples)
    action_rank = weighted_mean(action_rank_each, examples)
    consistency = weighted_mean(consistency_each, examples)
    floor = weighted_mean(floor_each, examples)
    loss = (
        args.lambda_clean_rank * clean_rank
        + args.lambda_aug_rank * action_rank
        + args.lambda_consistency * consistency
        + args.lambda_guided_transfer * weighted_mean(self_transfer, examples)
        + args.lambda_guided_teacher_margin * weighted_mean(fixed_teacher_each, examples)
        + args.lambda_margin_floor * floor
        + args.lambda_preserve * preserve
    )
    return loss, {
        "guided_clean_rank": float(clean_rank.detach()),
        "guided_action_rank": float(action_rank.detach()),
        "guided_consistency": float(consistency.detach()),
        "guided_transfer": float(weighted_mean(self_transfer, examples).detach()),
        "guided_fixed_teacher_margin": float(weighted_mean(fixed_teacher_each, examples).detach()),
        "guided_fixed_teacher_fraction": float(has_fixed_teacher.float().mean().detach()),
        "guided_fixed_teacher_target": float(torch.where(
            has_fixed_teacher, teacher_target, torch.zeros_like(teacher_target)
        ).sum().detach() / torch.clamp(has_fixed_teacher.float().sum(), min=1.0)),
        "guided_risk_control_fraction": float(np.mean([
            example.supervision_kind != "corrective" for example in examples
        ])),
        "guided_margin_floor": float(floor.detach()),
        "guided_preserve": float(preserve.detach()),
        "guided_clean_margin": float(clean_margin.mean().detach()),
        "guided_action_margin": float(action_margin.mean().detach()),
        "guided_action_advantage": float((action_margin - clean_margin).mean().detach()),
    }


@torch.no_grad()
def audit_guided_teacher_replay(
    model,
    store: SpectrumStore,
    examples: list[GuidedNoiseExample],
    device: torch.device,
    args,
) -> dict[str, float]:
    """Fail closed unless action construction reproduces frozen teacher margins."""
    clean_error: list[np.ndarray] = []
    action_error: list[np.ndarray] = []
    for left in range(0, len(examples), args.eval_batch_size):
        batch = examples[left:left + args.eval_batch_size]
        spectra, layout, _ = flatten_guided(
            store, batch,
            args.guided_recurrence_prevalence,
            args.guided_recurrence_max_peaks,
        )
        encoded = forward_embeddings(model, spectra.to(device), False)
        clean = margins(encoded, layout, "clean").float().cpu().numpy()
        action = margins(encoded, layout, "action").float().cpu().numpy()
        expected_clean = np.asarray([item.official_margin for item in batch], dtype=np.float32)
        expected_action = np.asarray([item.teacher_margin for item in batch], dtype=np.float32)
        clean_error.append(np.abs(clean - expected_clean))
        action_error.append(np.abs(action - expected_action))
    clean_error_array = np.concatenate(clean_error)
    action_error_array = np.concatenate(action_error)
    report = {
        "queries": int(len(examples)),
        "clean_margin_max_abs_error": float(np.max(clean_error_array)),
        "action_margin_max_abs_error": float(np.max(action_error_array)),
        "clean_margin_p99_abs_error": float(np.quantile(clean_error_array, 0.99)),
        "action_margin_p99_abs_error": float(np.quantile(action_error_array, 0.99)),
    }
    if report["clean_margin_max_abs_error"] > 2e-4 or report["action_margin_max_abs_error"] > 2e-4:
        raise RuntimeError(f"E14 teacher replay failed: {report}")
    return report


def flatten_direct(store: SpectrumStore, examples: list[DirectExample], action: bool):
    tensors: list[torch.Tensor] = []
    clean_rows: list[int] = []
    layout: list[dict] = []
    for example in examples:
        item: dict[str, object] = {"clean": len(tensors)}
        tensors.append(store.one(example.query_row))
        clean_rows.append(example.query_row)
        if action:
            item["action"] = len(tensors)
            tensors.append(attenuate_sequence(
                store.one(example.query_row), example.target_path, example.attenuation,
            ))
        item["positive"] = list(range(len(tensors), len(tensors) + len(example.positive_rows)))
        tensors.extend(store.get(example.positive_rows))
        clean_rows.extend(example.positive_rows)
        item["negative"] = list(range(len(tensors), len(tensors) + len(example.negative_rows)))
        tensors.extend(store.get(example.negative_rows))
        clean_rows.extend(example.negative_rows)
        layout.append(item)
    return torch.stack(tensors), layout, clean_rows


def weighted_mean(values: torch.Tensor, examples: list[DirectExample]) -> torch.Tensor:
    weights = torch.tensor(
        [example.sample_weight for example in examples],
        device=values.device, dtype=values.dtype,
    )
    return torch.sum(values * weights) / torch.sum(weights)


def action_key(example: DirectExample) -> tuple[int, str, tuple[int, ...], float]:
    return (
        int(example.query_index), str(example.policy), tuple(example.target_path),
        float(example.attenuation),
    )


def frozen_reference_margins(
    query_vectors: torch.Tensor, examples: list[DirectExample],
    official_by_row: dict[int, np.ndarray],
) -> torch.Tensor:
    """Score trainable queries against fixed official candidate anchors.

    This keeps the retrieval geometry shared at inference while making the
    *training diagnostic* identifiable: a query-ranking improvement cannot be
    paid for by moving the few positive/negative reference spectra in its
    minibatch.  Reference spectra still receive the ordinary preservation
    loss and are encoded by the same saved model at evaluation.
    """
    output: list[torch.Tensor] = []
    for vector, example in zip(query_vectors, examples):
        positive = torch.as_tensor(
            np.stack([official_by_row[int(row)] for row in example.positive_rows]),
            device=vector.device, dtype=vector.dtype,
        )
        negative = torch.as_tensor(
            np.stack([official_by_row[int(row)] for row in example.negative_rows]),
            device=vector.device, dtype=vector.dtype,
        )
        output.append(torch.max(positive @ vector) - torch.max(negative @ vector))
    return torch.stack(output)


@torch.no_grad()
def encode_official_action_targets(
    model: torch.nn.Module, store: SpectrumStore, examples: list[DirectExample],
    device: torch.device, batch_size: int,
) -> dict[tuple[int, str, tuple[int, ...], float], np.ndarray]:
    """Freeze the official embedding of every distinct raw-spectrum action.

    Targets are generated once before the first optimizer step.  They are not
    labels, candidate scores, P2b outputs, or post-embedding modules: each is
    simply the official DreaMS encoding of the preregistered perturbed spectrum.
    """
    unique: dict[tuple[int, str, tuple[int, ...], float], DirectExample] = {}
    for example in examples:
        unique.setdefault(action_key(example), example)
    ordered = [unique[key] for key in sorted(unique)]
    targets: dict[tuple[int, str, tuple[int, ...], float], np.ndarray] = {}
    model.eval()
    for left in range(0, len(ordered), batch_size):
        block = ordered[left:left + batch_size]
        spectra = torch.stack([
            attenuate_sequence(
                store.one(example.query_row), example.target_path, example.attenuation,
            ) for example in block
        ]).to(device)
        vectors = forward_embeddings(model, spectra, False).float().cpu().numpy()
        if not np.all(np.isfinite(vectors)):
            raise RuntimeError("official action target encoding produced non-finite values")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(np.abs(norms - 1.0) > 2e-3):
            raise RuntimeError("official action target encoding produced non-unit values")
        for example, vector in zip(block, vectors):
            targets[action_key(example)] = np.asarray(vector, dtype=np.float32)
        right = left + len(block)
        if right == len(ordered) or right % (batch_size * 20) == 0:
            print(f"[official-action-targets] {right:,}/{len(ordered):,}", flush=True)
    if len(targets) != len(unique):
        raise RuntimeError("official action target cache is incomplete")
    return targets


def gradient_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    """Return the pre-clipping L2 norm without modifying gradients."""
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = parameter.grad.detach().float().norm(2).item()
            squared += value * value
    return math.sqrt(squared)


def detached_loss_gradients(
    loss: torch.Tensor, parameters: list[torch.nn.Parameter],
) -> tuple[float, list[torch.Tensor | None]]:
    """Return one branch gradient without touching optimizer ``.grad`` buffers."""
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=False, create_graph=False, allow_unused=True,
    )
    detached = [gradient.detach() if gradient is not None else None for gradient in gradients]
    squared = sum(
        float(torch.sum(gradient.float() * gradient.float()).detach())
        for gradient in detached if gradient is not None
    )
    return math.sqrt(squared), detached


def gradient_cosine(
    first: list[torch.Tensor | None], second: list[torch.Tensor | None],
) -> float:
    dot = 0.0
    first_squared = 0.0
    second_squared = 0.0
    for left, right in zip(first, second):
        if left is not None:
            first_squared += float(torch.sum(left.float() * left.float()).detach())
        if right is not None:
            second_squared += float(torch.sum(right.float() * right.float()).detach())
        if left is not None and right is not None:
            dot += float(torch.sum(left.float() * right.float()).detach())
    denominator = math.sqrt(first_squared * second_squared)
    return dot / denominator if denominator > 0 else float("nan")


def combined_gradient_norm(
    components: list[tuple[list[torch.Tensor | None], float]],
) -> float:
    squared = 0.0
    for parameter_index in range(len(components[0][0])):
        combined = None
        for gradients, weight in components:
            gradient = gradients[parameter_index]
            if gradient is None:
                continue
            contribution = gradient.float() * float(weight)
            combined = contribution if combined is None else combined + contribution
        if combined is not None:
            squared += float(torch.sum(combined * combined).detach())
    return math.sqrt(squared)


def direct_action_loss(model, store: SpectrumStore, examples: list[DirectExample],
                       official_by_row: dict[int, np.ndarray], device: torch.device,
                       args, official_action_targets=None) -> tuple[torch.Tensor, dict[str, float]]:
    spectra, layout, clean_rows = flatten_direct(store, examples, True)
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    clean_z = torch.stack([encoded[int(item["clean"])] for item in layout])
    aug_z = torch.stack([encoded[int(item["action"])] for item in layout])
    if args.rank_reference_mode == "official":
        clean_margin = frozen_reference_margins(clean_z, examples, official_by_row)
        aug_margin = frozen_reference_margins(aug_z, examples, official_by_row)
    else:
        clean_margin = margins(encoded, layout, "clean")
        aug_margin = margins(encoded, layout, "action")
    clean_rank_each = F.softplus((args.rank_margin - clean_margin) / args.temperature)
    aug_rank_each = F.softplus((args.rank_margin - aug_margin) / args.temperature)
    if args.direct_transfer_mode == "symmetric":
        transfer_target = aug_z
    elif args.direct_transfer_mode == "student_action_stopgrad":
        transfer_target = aug_z.detach()
    elif args.direct_transfer_mode == "official_action":
        if official_action_targets is None:
            raise RuntimeError("official_action transfer requires frozen action targets")
        transfer_target = torch.as_tensor(
            np.stack([official_action_targets[action_key(example)] for example in examples]),
            device=device, dtype=clean_z.dtype,
        )
    else:  # pragma: no cover - argparse protects this branch
        raise RuntimeError(f"unknown direct transfer mode: {args.direct_transfer_mode}")
    consistency_each = 1.0 - torch.sum(clean_z * transfer_target, dim=1)
    floors = torch.tensor(
        [example.official_margin - args.margin_floor_slack for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    floor_each = F.relu(floors - clean_margin)

    clean_indices: list[int] = []
    for item in layout:
        clean_indices.append(int(item["clean"]))
        clean_indices.extend(item["positive"])
        clean_indices.extend(item["negative"])
    official = torch.from_numpy(np.stack([official_by_row[row] for row in clean_rows])).to(device)
    preserve = (1.0 - torch.sum(encoded[clean_indices] * official, dim=1)).mean()
    clean_rank = weighted_mean(clean_rank_each, examples)
    aug_rank = weighted_mean(aug_rank_each, examples)
    consistency = weighted_mean(consistency_each, examples)
    floor = weighted_mean(floor_each, examples)
    loss = (
        args.lambda_clean_rank * clean_rank
        + args.lambda_aug_rank * aug_rank
        + args.lambda_consistency * consistency
        + args.lambda_margin_floor * floor
        + args.lambda_preserve * preserve
    )
    return loss, {
        "action_clean_rank": float(clean_rank.detach()),
        "action_aug_rank": float(aug_rank.detach()),
        "action_consistency": float(consistency.detach()),
        "action_margin_floor": float(floor.detach()),
        "action_preserve": float(preserve.detach()),
        "action_clean_margin": float(clean_margin.mean().detach()),
        "action_aug_margin": float(aug_margin.mean().detach()),
        "action_clean_margin_pass": float((clean_margin > 0).float().mean().detach()),
        "action_aug_margin_pass": float((aug_margin > 0).float().mean().detach()),
        "action_transfer_target_cosine": float(
            torch.sum(clean_z * transfer_target, dim=1).mean().detach()
        ),
    }


def safety_loss(model, store: SpectrumStore, examples: list[DirectExample],
                official_by_row: dict[int, np.ndarray], device: torch.device,
                args) -> tuple[torch.Tensor, dict[str, float]]:
    spectra, layout, clean_rows = flatten_direct(store, examples, False)
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    clean_z = torch.stack([encoded[int(item["clean"])] for item in layout])
    clean_margin = (
        frozen_reference_margins(clean_z, examples, official_by_row)
        if args.rank_reference_mode == "official"
        else margins(encoded, layout, "clean")
    )
    floors = torch.tensor(
        [example.official_margin - args.margin_floor_slack for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    floor = weighted_mean(F.relu(floors - clean_margin), examples)
    clean_indices: list[int] = []
    for item in layout:
        clean_indices.append(int(item["clean"]))
        clean_indices.extend(item["positive"])
        clean_indices.extend(item["negative"])
    official = torch.from_numpy(np.stack([official_by_row[row] for row in clean_rows])).to(device)
    preserve = (1.0 - torch.sum(encoded[clean_indices] * official, dim=1)).mean()
    loss = args.lambda_margin_floor * floor + args.lambda_preserve * preserve
    return loss, {
        "safety_margin_floor": float(floor.detach()),
        "safety_preserve": float(preserve.detach()),
        "safety_margin": float(clean_margin.mean().detach()),
    }


def positive_arm_loss(model, store: SpectrumStore, examples: list[DirectExample],
                      official_by_row: dict[int, np.ndarray], device: torch.device,
                      args) -> tuple[torch.Tensor, dict[str, float]]:
    """Rank an explicit cross-condition positive over the same query negatives.

    The official model supplies only a per-example safety floor and embedding
    preservation target.  It is not a teacher action and cannot choose pairs.
    """
    spectra, layout, clean_rows = flatten_direct(store, examples, False)
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    current_margin = margins(encoded, layout, "clean")
    rank_each = F.softplus((args.rank_margin - current_margin) / args.temperature)

    official = torch.from_numpy(np.stack([official_by_row[row] for row in clean_rows])).to(device)
    official_margins: list[torch.Tensor] = []
    for item in layout:
        q = official[int(item["clean"])]
        positive_score = torch.max(official[item["positive"]] @ q)
        negative_score = torch.max(official[item["negative"]] @ q)
        official_margins.append(positive_score - negative_score)
    official_margin = torch.stack(official_margins)
    floor_each = F.relu(official_margin - args.margin_floor_slack - current_margin)

    clean_indices: list[int] = []
    for item in layout:
        clean_indices.append(int(item["clean"]))
        clean_indices.extend(item["positive"])
        clean_indices.extend(item["negative"])
    preserve = (1.0 - torch.sum(encoded[clean_indices] * official, dim=1)).mean()
    rank = weighted_mean(rank_each, examples)
    floor = weighted_mean(floor_each, examples)
    loss = (
        args.lambda_positive_rank * rank
        + args.lambda_positive_margin_floor * floor
        + args.lambda_preserve * preserve
    )
    return loss, {
        "positive_rank": float(rank.detach()),
        "positive_margin_floor": float(floor.detach()),
        "positive_preserve": float(preserve.detach()),
        "positive_margin": float(current_margin.mean().detach()),
        "positive_margin_pass": float((current_margin > 0).float().mean().detach()),
        "positive_official_margin": float(official_margin.mean().detach()),
    }


def batched(values: list[DirectExample], size: int):
    for left in range(0, len(values), size):
        yield values[left:left + size]


def identity_balanced_epoch(examples: list[DirectExample], rng: np.random.Generator,
                            views_per_identity: int) -> list[DirectExample]:
    """Draw exactly K views per identity and round-robin available policies."""
    if views_per_identity < 1:
        raise ValueError("views-per-identity must be positive")
    groups: dict[str, list[DirectExample]] = {}
    for example in examples:
        groups.setdefault(example.identity, []).append(example)
    output: list[DirectExample] = []
    for identity in sorted(groups):
        values = groups[identity]
        by_policy: dict[str, list[DirectExample]] = {}
        for value in values:
            by_policy.setdefault(value.policy, []).append(value)
        policies = sorted(by_policy)
        policy_order = np.asarray(policies, dtype=object)[rng.permutation(len(policies))]
        local_orders = {
            policy: rng.permutation(len(by_policy[policy])) for policy in policies
        }
        for offset in range(views_per_identity):
            policy = str(policy_order[offset % len(policy_order)])
            order = local_orders[policy]
            local = int(order[(offset // len(policy_order)) % len(order)])
            output.append(by_policy[policy][local])
    rng.shuffle(output)
    return output


def sampling_schedule_sha256(examples: list[DirectExample]) -> str:
    """Hash arm-invariant sampler keys; deliberately exclude the action path."""
    canonical = "\n".join(
        f"{item.query_index}|{item.query_row}|{item.identity}|{item.formula}|"
        f"{item.policy}|{','.join(map(str, item.positive_rows))}|"
        f"{','.join(map(str, item.negative_rows))}"
        for item in examples
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_causal_configuration(args: argparse.Namespace) -> None:
    """Freeze every non-arm degree of freedom in the first attribution trial."""
    if args.causal_arm == "legacy":
        return
    exact = {
        "action_selection": "fixed",
        "policy": "curriculum",
        "action_scope": "all",
        "outer_fold": 0,
        "formula_fold_seed": 20260825,
        "epochs": 4,
        "batch_actions": 4,
        "views_per_identity": 4,
        "error_views_per_identity": 0,
        "positive_spectra": 4,
        "negative_molecules": 8,
        "unfreeze_blocks": 1,
        "direct_transfer_mode": "symmetric",
        "rank_reference_mode": "shared",
        "guided_noise_policy": "none",
    }
    for name, expected in exact.items():
        observed = getattr(args, name)
        if observed != expected:
            raise ValueError(
                f"causal attribution freezes --{name.replace('_', '-')}={expected!r}; "
                f"observed {observed!r}"
            )
    floats = {
        "backbone_lr": 2e-6,
        "head_lr": 1e-5,
        "weight_decay": 1e-4,
        "rank_margin": 0.05,
        "temperature": 0.10,
        "lambda_clean_rank": 1.0,
        "lambda_aug_rank": 1.0,
        "lambda_consistency": 0.25,
        "lambda_margin_floor": 2.0,
        "lambda_preserve": 5.0,
        "margin_floor_slack": 0.005,
        "safety_ratio": 1.0,
        "safety_stream_weight": 1.0,
        "positive_stream_weight": 0.0,
        "grad_clip": 1.0,
    }
    for name, expected in floats.items():
        observed = float(getattr(args, name))
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(
                f"causal attribution freezes --{name.replace('_', '-')}={expected}; "
                f"observed {observed}"
            )
    if args.initial_student_checkpoint is not None:
        raise ValueError("causal attribution must start from the official initialization")
    if args.amp:
        raise ValueError("causal attribution freezes full-fp32 training (--no-amp)")
    if not args.run_suffix:
        raise ValueError("causal attribution requires a unique --run-suffix")


def main() -> None:
    args = arguments()
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be 0..4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("E4-A direct augmentation requires CUDA")
    if args.head_lr < args.backbone_lr or args.backbone_lr <= 0:
        raise ValueError("require head-lr >= backbone-lr > 0")
    validate_causal_configuration(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    required = [
        args.graph, args.data, args.embedding_cache, args.official_checkpoint,
        args.architecture_checkpoint, args.r0_dir / "report.json",
        args.r0_dir / "training_actions.csv.gz",
    ]
    if args.action_selection == "outcome_mined":
        if args.outcome_action_dir is None:
            raise ValueError("outcome_mined action selection requires --outcome-action-dir")
        required.extend([
            args.outcome_action_dir / "report.json",
            args.outcome_action_dir / "corrective_teacher_actions.csv.gz",
        ])
    if args.positive_stream_weight > 0:
        required.extend([
            args.positive_manifest_dir / "report.json",
            args.positive_manifest_dir / "positive_pairs.csv.gz",
        ])
    initial_decision_path: Path | None = None
    initial_ledger_path: Path | None = None
    if args.initial_student_checkpoint is not None:
        initial_decision_path = args.initial_student_checkpoint.parent / "decision.json"
        initial_ledger_path = args.initial_student_checkpoint.parent / "held_per_query.csv.gz"
        required.extend([args.initial_student_checkpoint, initial_decision_path])
    if args.guided_noise_policy == "selected":
        if args.guided_crossfit_root is None:
            raise ValueError("selected guided noise requires --guided-crossfit-root")
        if args.initial_student_checkpoint is None:
            raise ValueError("selected guided noise requires --initial-student-checkpoint")
    elif args.guided_noise_policy != "none":
        required.extend([
            args.guided_intensity_dir / "report.json",
            args.guided_intensity_dir / "action_manifest.csv.gz",
            args.guided_transfer_dir / "report.json",
            args.guided_transfer_dir / "action_manifest.csv.gz",
            args.error_signatures,
        ])
        if args.guided_action_authorization_dir is not None:
            required.append(args.guided_action_authorization_dir / "report.json")
            if args.guided_noise_policy in {"transfer", "both"}:
                if args.guided_reference_checkpoint is None:
                    raise ValueError(
                        "E12-B-authorized transfer requires --guided-reference-checkpoint"
                    )
                required.append(args.guided_reference_checkpoint)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    r0 = json.loads((args.r0_dir / "report.json").read_text(encoding="utf-8"))
    if not r0.get("formal") or r0.get("contracts", {}).get("P2b") != "forbidden":
        raise RuntimeError("E4-A requires the formal, P2b-free R0 manifest")
    if not r0.get("contracts", {}).get("action_outcomes_absent_from_training_manifest"):
        raise RuntimeError("R0 does not certify outcome-free training actions")
    if args.positive_stream_weight < 0 or args.positive_ratio < 0:
        raise ValueError("positive stream weight/ratio must be nonnegative")
    if args.safety_ratio <= 0 or args.safety_stream_weight <= 0:
        raise ValueError("safety-ratio and safety-stream-weight must be positive")
    if (
        args.guided_noise_weight < 0
        or args.guided_noise_ratio < 0
        or args.guided_risk_control_ratio < 0
    ):
        raise ValueError("guided noise weight/ratio must be nonnegative")
    if args.guided_noise_policy != "none" and args.guided_noise_weight <= 0:
        raise ValueError("guided noise policy requires a positive guided-noise-weight")
    if args.lambda_guided_transfer < 0 or args.lambda_guided_teacher_margin < 0:
        raise ValueError("guided transfer weights must be nonnegative")
    if args.guided_teacher_margin_cap <= 0:
        raise ValueError("guided teacher margin cap must be positive")
    if not 0 < args.guided_teacher_delta_fraction <= 1:
        raise ValueError("guided teacher delta fraction must be in (0, 1]")
    if args.guided_teacher_delta_cap <= 0:
        raise ValueError("guided teacher delta cap must be positive")
    if args.guided_risk_control_ratio > 0 and args.guided_noise_policy != "selected":
        raise ValueError("guided risk controls are available only in selected E14 mode")
    if args.guided_auto_balance and args.guided_noise_policy != "selected":
        raise ValueError("guided auto-balance is available only in selected E14 mode")
    if args.guided_noise_views_per_identity < 1:
        raise ValueError("guided-noise-views-per-identity must be positive")
    if not 0 < args.guided_recurrence_prevalence <= 1:
        raise ValueError("guided-recurrence-prevalence must be in (0, 1]")
    if args.guided_recurrence_max_peaks < 1:
        raise ValueError("guided-recurrence-max-peaks must be positive")
    nonhistorical_guided_recipe = (
        not math.isclose(args.guided_recurrence_prevalence, 0.67, abs_tol=1e-12)
        or args.guided_recurrence_max_peaks != 5
        or args.guided_transfer_mode == "symmetric"
        or args.guided_query_scope == "all"
    )
    if (
        args.guided_noise_policy in {"transfer", "both"}
        and nonhistorical_guided_recipe
        and args.guided_action_authorization_dir is None
    ):
        raise ValueError(
            "non-historical guided transfer requires --guided-action-authorization-dir"
        )
    if args.error_views_per_identity < 0:
        raise ValueError("error-views-per-identity must be nonnegative")

    if args.run_suffix and not all(
        character.isalnum() or character in "-_" for character in args.run_suffix
    ):
        raise ValueError("run-suffix may contain only letters, digits, '-' and '_'")
    tag = (
        f"{args.policy}_{args.action_scope}_views{args.views_per_identity}_blocks{args.unfreeze_blocks}_"
        f"blr_{args.backbone_lr:.0e}_hlr_{args.head_lr:.0e}"
    )
    if args.action_selection != "fixed":
        tag += f"_as_{args.action_selection}"
    if args.initial_student_checkpoint is not None:
        tag += "_warm"
    if args.positive_stream_weight > 0:
        tag += f"_pnw_{args.positive_stream_weight:g}_pv{args.positive_views_per_identity}"
    if args.guided_noise_policy != "none":
        tag += (
            f"_gpn_{args.guided_noise_policy}_gw{args.guided_noise_weight:g}"
            f"_gv{args.guided_noise_views_per_identity}"
            f"_gtm_{args.guided_transfer_mode}"
            f"_grp{args.guided_recurrence_prevalence:g}"
            f"_gmax{args.guided_recurrence_max_peaks}"
            f"_gscope_{args.guided_query_scope}"
        )
        if args.guided_noise_policy == "selected":
            tag += (
                f"_gtmargin{args.lambda_guided_teacher_margin:g}"
                f"_gttarget{args.guided_teacher_target_mode}"
                f"_gtdf{args.guided_teacher_delta_fraction:g}"
                f"_grisk{args.guided_risk_control_ratio:g}"
                f"_gbal{int(args.guided_auto_balance)}"
            )
    if args.safety_stream_weight != 1.0:
        tag += f"_sw{args.safety_stream_weight:g}"
    if args.error_views_per_identity > 0:
        tag += f"_ev{args.error_views_per_identity}"
    if args.run_suffix:
        tag += f"_{args.run_suffix}"
    if args.causal_arm != "legacy":
        tag += f"_causal_{args.causal_arm}"
    output = args.output_root / tag / f"seed_{args.seed}" / f"fold_{args.outer_fold}"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite E4-A result: {output}")

    graph = CandidateGraph(args.graph)
    official_rank, official_margin = official_rank_margin(graph)
    outcome_report: dict = {}
    if args.action_selection == "fixed":
        actions = pd.read_csv(args.r0_dir / "training_actions.csv.gz")
        forbidden_columns = {
            "corrected", "introduced", "target_rank", "target_margin", "random_margin"
        }
        leaked = forbidden_columns.intersection(actions.columns)
        if leaked:
            raise RuntimeError(f"post-outcome columns leaked into fixed direct training: {sorted(leaked)}")
        selected = []
        for selector, attenuation, step in FIXED_POLICY[args.policy]:
            block = actions.loc[
                actions["selector"].astype(str).eq(selector)
                & np.isclose(actions["attenuation"].astype(float), attenuation)
                & actions["step"].astype(int).eq(step)
            ].copy()
            if block.empty:
                raise RuntimeError(f"missing fixed policy cell {selector}|{attenuation}|{step}")
            selected.append(block)
        actions = pd.concat(selected, ignore_index=True)
    else:
        report_path = args.outcome_action_dir / "report.json"
        action_path = args.outcome_action_dir / "corrective_teacher_actions.csv.gz"
        outcome_report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            outcome_report.get("status") != "noise_final_r1_privileged_teacher_complete"
            or not outcome_report.get("formal")
            or outcome_report.get("contracts", {}).get("P2b") != "forbidden"
            or int(outcome_report.get("locally_materialised_union_recoverable", -1)) != 882
        ):
            raise RuntimeError("outcome-mined action artifact is not formal and P2b-free")
        actions = pd.read_csv(action_path)
        if len(actions) != 882:
            raise RuntimeError(f"outcome-mined action count drifted: {len(actions)} != 882")
        required = {
            "query_index", "query_row", "query_ik14", "query_formula", "formula_fold",
            "baseline_rank", "teacher_rank", "teacher_margin", "selector", "attenuation",
            "step", "target_path", "teacher_hard_negative_row",
        }
        if required - set(actions.columns):
            raise RuntimeError(
                f"outcome-mined action table is missing columns: {sorted(required - set(actions.columns))}"
            )
        if actions["query_index"].duplicated().any():
            raise RuntimeError("outcome-mined action table must contain one selected action per query")
        if actions["baseline_rank"].astype(int).le(1).any() or actions["teacher_rank"].astype(int).ne(1).any():
            raise RuntimeError("outcome-mined action table contains a non-correction")
        for row in actions[["query_index", "teacher_hard_negative_row"]].itertuples(index=False):
            query = int(row.query_index)
            forced = int(row.teacher_hard_negative_row)
            _, candidate_rows, molecule_ptr, _ = graph.query_block(query)
            positive_rows = set(map(int, candidate_rows[: int(molecule_ptr[1])]))
            negative_rows = set(map(int, candidate_rows[int(molecule_ptr[1]):]))
            if forced in positive_rows or forced not in negative_rows:
                raise RuntimeError(
                    f"outcome-mined hard negative row {forced} is not a negative candidate for query {query}"
                )
        # Outcome fields are legal for train-fold action mining, but must never
        # enter a loss or sample weight.  Strip them before example creation.
        actions = actions.rename(columns={"teacher_hard_negative_row": "hard_negative_row"})
        actions = actions.drop(columns=[
            column for column in ("teacher_rank", "teacher_margin") if column in actions
        ])
    actions, causal_action_audit = materialize_causal_arm(actions, args.causal_arm)
    actions["baseline_rank"] = official_rank[actions["query_index"].to_numpy(np.int64)]
    if args.action_scope == "errors":
        actions = actions.loc[actions["baseline_rank"].astype(int).ne(1)].copy()
    train_actions = actions.loc[actions["formula_fold"].astype(int).ne(args.outer_fold)].copy()
    # Outcome-mined held actions are deliberately not used even for subgroup
    # evaluation.  The primary evaluation is the complete clean held fold.
    held_action = (
        actions.loc[actions["formula_fold"].astype(int).eq(args.outer_fold)].copy()
        if args.action_selection == "fixed" else actions.iloc[0:0].copy()
    )
    held_formulas = {
        str(formula) for formula in graph.query_formula
        if stable_fold(str(formula), 5, args.formula_fold_seed) == args.outer_fold
    }
    if train_actions["query_formula"].astype(str).isin(held_formulas).any():
        raise RuntimeError("formula isolation failed in action manifest")

    positive_pairs = pd.DataFrame()
    train_positive = pd.DataFrame()
    held_positive = pd.DataFrame()
    positive_report: dict = {}
    if args.positive_stream_weight > 0:
        positive_report = json.loads(
            (args.positive_manifest_dir / "report.json").read_text(encoding="utf-8")
        )
        contracts = positive_report.get("contracts", {})
        if not positive_report.get("formal") or not positive_report.get("pass_to_pn_training"):
            raise RuntimeError("P-arm manifest is not a passing formal artifact")
        if contracts.get("teacher") != "forbidden" or contracts.get("P2b") != "forbidden":
            raise RuntimeError("P-arm manifest violates teacher/P2b boundary")
        positive_pairs = pd.read_csv(args.positive_manifest_dir / "positive_pairs.csv.gz")
        forbidden_positive = {
            "corrected", "introduced", "target_rank", "target_margin",
            "teacher_score", "teacher_rows", "p2b_score",
        }
        leaked_positive = forbidden_positive.intersection(positive_pairs.columns)
        if leaked_positive:
            raise RuntimeError(f"outcome/teacher columns leaked into P-arm: {sorted(leaked_positive)}")
        required_positive = {
            "query_index", "query_row", "positive_row", "query_ik14",
            "query_formula", "formula_fold", "relation",
        }
        if required_positive - set(positive_pairs.columns):
            raise RuntimeError("P-arm manifest is missing required pair columns")
        positive_query = positive_pairs["query_index"].to_numpy(np.int64)
        if np.any((positive_query < 0) | (positive_query >= graph.n_queries)):
            raise RuntimeError("P-arm query index is out of graph range")
        if not np.array_equal(
            positive_pairs["query_row"].to_numpy(np.int64), graph.query_row[positive_query]
        ):
            raise RuntimeError("P-arm query rows do not reproduce frozen candidate graph")
        if not np.array_equal(
            positive_pairs["query_ik14"].astype(str).to_numpy(), graph.query_ik14[positive_query]
        ):
            raise RuntimeError("P-arm identity does not reproduce frozen candidate graph")
        if not np.array_equal(
            positive_pairs["query_formula"].astype(str).to_numpy(), graph.query_formula[positive_query]
        ):
            raise RuntimeError("P-arm formula does not reproduce frozen candidate graph")
        for query, positive_row in zip(
            positive_query, positive_pairs["positive_row"].to_numpy(np.int64)
        ):
            pair_slice, candidate_rows, local_ptr, _ = graph.query_block(int(query))
            del pair_slice
            if int(positive_row) not in set(map(int, candidate_rows[: int(local_ptr[1])])):
                raise RuntimeError(f"P-arm row {positive_row} is not a positive for query {query}")
        observed = positive_pairs["query_formula"].astype(str).map(
            lambda value: stable_fold(value, 5, args.formula_fold_seed)
        ).to_numpy(np.int8)
        if not np.array_equal(observed, positive_pairs["formula_fold"].to_numpy(np.int8)):
            raise RuntimeError("P-arm formula folds do not reproduce locally")
        train_positive = positive_pairs.loc[
            positive_pairs["formula_fold"].astype(int).ne(args.outer_fold)
        ].copy()
        held_positive = positive_pairs.loc[
            positive_pairs["formula_fold"].astype(int).eq(args.outer_fold)
        ].copy()
        if train_positive["query_formula"].astype(str).isin(
            set(held_positive["query_formula"].astype(str))
        ).any():
            raise RuntimeError("formula isolation failed in P-arm manifest")

    guided_frame = pd.DataFrame()
    train_guided = pd.DataFrame()
    held_guided = pd.DataFrame()
    train_guided_risk = pd.DataFrame()
    guided_intensity_report: dict = {}
    guided_transfer_report: dict = {}
    guided_authorization_report: dict = {}
    guided_crossfit_reports: dict[str, dict] = {}
    if args.guided_noise_policy == "selected":
        if args.guided_crossfit_root is None:
            raise RuntimeError("selected guided noise requires --guided-crossfit-root")
        graph_hash = sha256_file(args.graph)
        fold_dir = args.guided_crossfit_root / f"fold_{args.outer_fold}"
        report_path = fold_dir / "report.json"
        manifest_path = fold_dir / "selected_actions.csv.gz"
        risk_path = fold_dir / "risk_controls.csv.gz"
        outcome_path = fold_dir / "action_outcomes.npz"
        amendment_path = fold_dir / "capacity_amendment.json"
        if not report_path.is_file() or not manifest_path.is_file() or not risk_path.is_file():
            raise FileNotFoundError(f"E14 teacher is incomplete: {fold_dir}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        capacity_authorized = bool(report.get("pass_to_shared_encoder_transfer"))
        amendment: dict = {}
        if not capacity_authorized and amendment_path.is_file():
            amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
            capacity_authorized = bool(
                outcome_path.is_file()
                and amendment.get("status")
                == "noise_final_e14_capacity_amendment_complete"
                and amendment.get("formal")
                and amendment.get("posthoc_amendment")
                and amendment.get("original_report_unchanged")
                and amendment.get("pass_to_shared_encoder_transfer")
                and amendment.get("provenance", {}).get("report_sha256")
                == sha256_file(report_path)
                and amendment.get("provenance", {}).get("selected_actions_sha256")
                == sha256_file(manifest_path)
                and amendment.get("provenance", {}).get("action_outcomes_sha256")
                == sha256_file(outcome_path)
                and amendment.get("provenance", {}).get("graph_sha256") == graph_hash
            )
        if (
            report.get("status") != "noise_final_e14_crossfit_p_teacher_complete"
            or not report.get("formal")
            or not capacity_authorized
            or int(report.get("outer_formula_fold", -1)) != args.outer_fold
            or report.get("provenance", {}).get("graph_sha256") != graph_hash
            or args.initial_student_checkpoint is None
            or report.get("provenance", {}).get("student_checkpoint_sha256")
            != sha256_file(args.initial_student_checkpoint)
            or not report.get("contracts", {}).get(
                "teacher_checkpoint_excludes_student_outer_formula_fold"
            )
            or not report.get("contracts", {}).get(
                "all_selected_queries_exclude_student_outer_formula_fold"
            )
            or not report.get("contracts", {}).get(
                "prior_fixed_cell_safety_filter_applied"
            )
            or not report.get("contracts", {}).get(
                "outer_train_multifold_action_safety_filter_applied"
            )
            or not report.get("contracts", {}).get(
                "action_specific_risk_controls_materialized"
            )
        ):
            raise RuntimeError(f"invalid E14 teacher for outer fold {args.outer_fold}")
        guided_crossfit_reports[str(args.outer_fold)] = report | {
            "capacity_authorization": (
                "posthoc_clustered_amendment" if amendment else "original_gate"
            ),
            "capacity_amendment": amendment,
        }
        guided_frame = pd.read_csv(manifest_path)
        required_selected = {
            "query_index", "query_row", "query_ik14", "query_formula",
            "formula_fold", "official_rank", "crossfit_clean_rank",
            "teacher_rank", "teacher_margin", "teacher_margin_delta",
            "positive_reference_rows",
            "teacher_positive_row", "teacher_hard_negative_row",
            "teacher_pair_clean_margin",
            "action_id", "guided_family", "guided_dose",
            "guided_auxiliary_dose", "guided_recurrence_prevalence",
            "guided_recurrence_max_peaks", "guided_support_weighted",
        }
        if required_selected - set(guided_frame.columns):
            raise RuntimeError(
                "selected E14 manifest is missing columns: "
                f"{sorted(required_selected - set(guided_frame.columns))}"
            )
        if guided_frame["query_index"].duplicated().any():
            raise RuntimeError("selected E14 teacher repeats a query")
        query = guided_frame["query_index"].to_numpy(np.int64)
        if np.any((query < 0) | (query >= graph.n_queries)):
            raise RuntimeError("selected E14 query is outside the graph")
        if not np.array_equal(
            guided_frame["query_row"].to_numpy(np.int64), graph.query_row[query]
        ):
            raise RuntimeError("selected E14 query rows drifted")
        if not np.array_equal(
            guided_frame["query_ik14"].astype(str).to_numpy(), graph.query_ik14[query]
        ) or not np.array_equal(
            guided_frame["query_formula"].astype(str).to_numpy(), graph.query_formula[query]
        ):
            raise RuntimeError("selected E14 identity/formula drifted")
        observed_fold = guided_frame["query_formula"].astype(str).map(
            lambda value: stable_fold(value, 5, args.formula_fold_seed)
        ).to_numpy(np.int8)
        if not np.array_equal(
            observed_fold, guided_frame["formula_fold"].to_numpy(np.int8)
        ):
            raise RuntimeError("selected E14 formula fold does not reproduce")
        if np.any(observed_fold == args.outer_fold):
            raise RuntimeError("held formula actions leaked into selected E14 training")
        if not (
            (guided_frame["official_rank"].to_numpy(int) != 1).all()
            and (guided_frame["crossfit_clean_rank"].to_numpy(int) != 1).all()
            and (guided_frame["teacher_rank"].to_numpy(int) == 1).all()
        ):
            raise RuntimeError("selected E14 rows are not strict clean-wrong/action-correct cases")
        guided_frame["guided_policy"] = guided_frame["action_id"].astype(str)
        train_guided = guided_frame.copy()
        held_guided = guided_frame.iloc[0:0].copy()
        train_guided_risk = pd.read_csv(risk_path)
        required_risk = required_selected | {"control_kind"}
        if required_risk - set(train_guided_risk.columns):
            raise RuntimeError(
                "selected E14 risk controls are missing columns: "
                f"{sorted(required_risk - set(train_guided_risk.columns))}"
            )
        if train_guided_risk[["query_index", "action_id"]].duplicated().any():
            raise RuntimeError("selected E14 risk controls repeat a query/action pair")
        risk_query = train_guided_risk["query_index"].to_numpy(np.int64)
        if np.any((risk_query < 0) | (risk_query >= graph.n_queries)):
            raise RuntimeError("selected E14 risk-control query is outside the graph")
        risk_observed_fold = train_guided_risk["query_formula"].astype(str).map(
            lambda value: stable_fold(value, 5, args.formula_fold_seed)
        ).to_numpy(np.int8)
        if not np.array_equal(
            risk_observed_fold,
            train_guided_risk["formula_fold"].to_numpy(np.int8),
        ) or np.any(risk_observed_fold == args.outer_fold):
            raise RuntimeError("selected E14 risk controls violate formula isolation")
        if not (
            train_guided_risk["crossfit_clean_rank"].to_numpy(int) == 1
        ).all() or not train_guided_risk["control_kind"].astype(str).isin(
            {"introduced", "protected_boundary"}
        ).all():
            raise RuntimeError("selected E14 risk controls are not mature-clean-correct")
        train_guided_risk["guided_policy"] = train_guided_risk["action_id"].astype(str)
        held_formulas = {
            str(formula) for formula in graph.query_formula
            if stable_fold(str(formula), 5, args.formula_fold_seed) == args.outer_fold
        }
        if train_guided["query_formula"].astype(str).isin(held_formulas).any():
            raise RuntimeError("selected E14 teacher violates held-formula isolation")
        for row in train_guided[[
            "query_index", "positive_reference_rows", "teacher_positive_row",
            "teacher_hard_negative_row",
        ]].itertuples(index=False):
            _, candidate_rows, local_ptr, _ = graph.query_block(int(row.query_index))
            positive_set = set(map(int, candidate_rows[: int(local_ptr[1])]))
            negative_set = set(map(int, candidate_rows[int(local_ptr[1]):]))
            if not set(parse_reference_rows(row.positive_reference_rows)) <= positive_set:
                raise RuntimeError(
                    f"selected E14 reference is not positive for query {row.query_index}"
                )
            if int(row.teacher_positive_row) not in positive_set:
                raise RuntimeError("selected E14 teacher-positive row is not positive")
            if int(row.teacher_hard_negative_row) not in negative_set:
                raise RuntimeError("selected E14 teacher-negative row is not negative")
        for row in train_guided_risk[[
            "query_index", "positive_reference_rows", "teacher_positive_row",
            "teacher_hard_negative_row",
        ]].itertuples(index=False):
            _, candidate_rows, local_ptr, _ = graph.query_block(int(row.query_index))
            positive_set = set(map(int, candidate_rows[: int(local_ptr[1])]))
            negative_set = set(map(int, candidate_rows[int(local_ptr[1]):]))
            if not set(parse_reference_rows(row.positive_reference_rows)) <= positive_set:
                raise RuntimeError("selected E14 risk reference is not positive")
            if int(row.teacher_positive_row) not in positive_set:
                raise RuntimeError("selected E14 risk positive row is not positive")
            if int(row.teacher_hard_negative_row) not in negative_set:
                raise RuntimeError("selected E14 risk negative row is not negative")
    elif args.guided_noise_policy != "none":
        guided_intensity_report = json.loads(
            (args.guided_intensity_dir / "report.json").read_text(encoding="utf-8")
        )
        guided_transfer_report = json.loads(
            (args.guided_transfer_dir / "report.json").read_text(encoding="utf-8")
        )
        if guided_intensity_report.get("status") != "noise_final_positive_guided_matrix_complete":
            raise RuntimeError("guided intensity matrix is not a formal completed artifact")
        if guided_transfer_report.get("status") != "noise_final_positive_peak_transfer_complete":
            raise RuntimeError("guided transfer matrix is not a formal completed artifact")
        if not guided_intensity_report.get("formal") or not guided_transfer_report.get("formal"):
            raise RuntimeError("guided noise artifacts must be formal")
        graph_hash = sha256_file(args.graph)
        if guided_intensity_report.get("provenance", {}).get("graph_sha256") != graph_hash:
            raise RuntimeError("guided intensity matrix graph mismatch")
        if guided_transfer_report.get("provenance", {}).get("graph_sha256") != graph_hash:
            raise RuntimeError("guided transfer matrix graph mismatch")
        required_cells = set()
        if args.guided_noise_policy in {"intensity", "both"}:
            required_cells.add("consensus_projection|dose=0.75")
        if args.guided_noise_policy in {"transfer", "both"}:
            required_cells.add("recurrent_union_mix|dose=0.50")
        available_cells = set(guided_intensity_report.get("passing_cells", [])) | set(
            guided_transfer_report.get("passing_cells", [])
        )
        if not required_cells <= available_cells:
            raise RuntimeError(f"guided policy uses cells that did not pass: {sorted(required_cells - available_cells)}")

        if args.guided_action_authorization_dir is not None:
            guided_authorization_report = json.loads(
                (args.guided_action_authorization_dir / "report.json").read_text(encoding="utf-8")
            )
            if (
                guided_authorization_report.get("status")
                != "noise_final_e12b_relaxed_recurrence_complete"
                or not guided_authorization_report.get("formal")
            ):
                raise RuntimeError("guided action authorization is not the formal E12-B artifact")
            if guided_authorization_report.get("provenance", {}).get("graph_sha256") != graph_hash:
                raise RuntimeError("guided action authorization graph mismatch")
            if (
                args.guided_reference_checkpoint is not None
                and guided_authorization_report.get("provenance", {}).get("student_checkpoint_sha256")
                != sha256_file(args.guided_reference_checkpoint)
            ):
                raise RuntimeError("guided reference checkpoint differs from E12-B provenance")
            expected_cell = (
                "top3|standard|"
                f"max={args.guided_recurrence_max_peaks}|dose=0.50"
            )
            best_cell = guided_authorization_report.get("best_fixed_cell", {})
            if (
                args.guided_noise_policy in {"transfer", "both"}
                and (
                    not math.isclose(args.guided_recurrence_prevalence, 0.50, abs_tol=1e-12)
                    or best_cell.get("cell_id") != expected_cell
                    or expected_cell not in set(guided_authorization_report.get("passing_fixed_cells", []))
                )
            ):
                raise RuntimeError(
                    "E12-B does not authorize the requested relaxed recurrence recipe"
                )

        base = pd.read_csv(
            args.guided_intensity_dir / "action_manifest.csv.gz",
            usecols=[
                "query_index", "query_row", "query_ik14", "query_formula",
                "positive_reference_rows",
            ],
        )
        transfer = pd.read_csv(
            args.guided_transfer_dir / "action_manifest.csv.gz",
            usecols=["query_index", "positive_missing_peak_count"],
        )
        signature = pd.read_csv(
            args.error_signatures,
            usecols=["query_index", "positive_deficit"],
        )
        if any(frame["query_index"].duplicated().any() for frame in (base, transfer, signature)):
            raise RuntimeError("guided noise inputs must be one row per query")
        guided_frame = base.merge(transfer, on="query_index", validate="one_to_one").merge(
            signature, on="query_index", validate="one_to_one",
        ).sort_values("query_index", kind="stable").reset_index(drop=True)
        if len(guided_frame) != graph.n_queries or not np.array_equal(
            guided_frame["query_index"].to_numpy(np.int64), np.arange(graph.n_queries)
        ):
            raise RuntimeError("guided action ledgers do not cover the candidate graph one-to-one")
        query = guided_frame["query_index"].to_numpy(np.int64)
        if not np.array_equal(guided_frame["query_row"].to_numpy(np.int64), graph.query_row[query]):
            raise RuntimeError("guided query rows do not reproduce graph")
        if not np.array_equal(guided_frame["query_ik14"].astype(str).to_numpy(), graph.query_ik14[query]):
            raise RuntimeError("guided identities do not reproduce graph")
        if not np.array_equal(guided_frame["query_formula"].astype(str).to_numpy(), graph.query_formula[query]):
            raise RuntimeError("guided formulas do not reproduce graph")
        positive_deficit = strict_bool(guided_frame["positive_deficit"], "positive_deficit")
        if args.guided_query_scope == "all":
            eligible = np.ones(len(guided_frame), dtype=bool)
        else:
            eligible = positive_deficit & (official_rank[query] != 1)
        guided_frame = guided_frame.loc[eligible].copy()
        configured: list[pd.DataFrame] = []
        if args.guided_noise_policy in {"intensity", "both"}:
            block = guided_frame.copy()
            block["guided_policy"] = "positive_intensity_consensus"
            block["guided_family"] = "consensus_projection"
            block["guided_dose"] = 0.75
            configured.append(block)
        if args.guided_noise_policy in {"transfer", "both"}:
            # The historical count was computed with prevalence 0.67/max 5.
            # It must not filter the relaxed E12-B recipe (0.50/max 10), whose
            # expanded eligibility is precisely the intervention being tested.
            if guided_authorization_report:
                block = guided_frame.copy()
            else:
                block = guided_frame.loc[
                    guided_frame["positive_missing_peak_count"].astype(int) > 0
                ].copy()
            block["guided_policy"] = "positive_recurrent_peak_transfer"
            block["guided_family"] = "recurrent_union_mix"
            block["guided_dose"] = 0.50
            configured.append(block)
        guided_frame = pd.concat(configured, ignore_index=True)
        guided_frame["formula_fold"] = guided_frame["query_formula"].astype(str).map(
            lambda value: stable_fold(value, 5, args.formula_fold_seed)
        ).astype(np.int8)
        # Reference rows are checked against the positive molecule.  This is
        # the only place identity labels enter action construction.
        for row in guided_frame[["query_index", "positive_reference_rows"]].drop_duplicates().itertuples(index=False):
            _, candidate_rows, local_ptr, _ = graph.query_block(int(row.query_index))
            positive_set = set(map(int, candidate_rows[: int(local_ptr[1])]))
            if not set(parse_reference_rows(row.positive_reference_rows)) <= positive_set:
                raise RuntimeError(f"guided reference is not positive for query {row.query_index}")
        train_guided = guided_frame.loc[
            guided_frame["formula_fold"].astype(int).ne(args.outer_fold)
        ].copy()
        held_guided = guided_frame.loc[
            guided_frame["formula_fold"].astype(int).eq(args.outer_fold)
        ].copy()
        if train_guided["query_formula"].astype(str).isin(
            set(held_guided["query_formula"].astype(str))
        ).any():
            raise RuntimeError("formula isolation failed in guided noise stream")

    # Full clean ledger is reconstructed from frozen graph labels/scores.  It
    # is not a teacher: it only provides ground-truth ranking and safety replay.
    formula_fold_by_query = {
        query: stable_fold(str(formula), 5, args.formula_fold_seed)
        for query, formula in enumerate(graph.query_formula)
    }
    # The independently recomputed split must reproduce every R0 action row.
    observed_fold = actions["query_index"].astype(int).map(formula_fold_by_query).to_numpy(np.int8)
    if not np.array_equal(observed_fold, actions["formula_fold"].to_numpy(np.int8)):
        raise RuntimeError("formula-fold reconstruction does not reproduce frozen R0")
    held_queries = np.asarray(
        [query for query, fold in formula_fold_by_query.items() if fold == args.outer_fold],
        dtype=np.int64,
    )
    safety_queries = np.asarray([
        query for query, fold in formula_fold_by_query.items()
        if fold != args.outer_fold and official_rank[query] == 1
    ], dtype=np.int64)
    safety_frame = pd.DataFrame({
        "query_index": safety_queries,
        "query_row": graph.query_row[safety_queries],
        "query_ik14": graph.query_ik14[safety_queries],
        "query_formula": graph.query_formula[safety_queries],
    })

    action_examples = make_examples(
        graph, train_actions, official_rank, official_margin,
        args.positive_spectra, args.negative_molecules, True,
    )
    safety_examples = make_examples(
        graph, safety_frame, official_rank, official_margin,
        args.positive_spectra, args.negative_molecules, False,
    )
    positive_examples = make_positive_examples(
        graph, train_positive, official_rank, official_margin,
        args.negative_molecules,
    ) if args.positive_stream_weight > 0 else []
    guided_examples: list[GuidedNoiseExample] = []
    error_action_examples = [example for example in action_examples if example.official_rank != 1]
    if len(action_examples) < 100 or len(set(x.identity for x in action_examples)) < 100:
        raise RuntimeError("direct action training pool is unexpectedly small")
    if args.positive_stream_weight > 0 and (
        len(positive_examples) < 500
        or len(set(x.identity for x in positive_examples)) < 250
    ):
        raise RuntimeError("strict cross-condition P-arm training pool is unexpectedly small")
    reachable_rows = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row))).astype(np.int64)
    store = SpectrumStore(args.data, reachable_rows, args.n_highest_peaks)
    _, cache_embeddings, cache_index = load_embedding_cache(args.embedding_cache)
    if set(map(int, reachable_rows)) - set(cache_index):
        raise RuntimeError("official embedding cache does not cover candidate graph")
    official_by_row = {
        int(row): cache_embeddings[index]
        for row, index in cache_index.items() if int(row) in store.position
    }
    official_encoded = np.stack([official_by_row[int(row)] for row in reachable_rows])

    # E12-B selected top3 real same-identity references in the mature E8
    # geometry, not in the original official geometry. Recompute those rows
    # from the frozen, provenance-checked E8 checkpoint before any optimizer
    # exists. They construct training noise only and are never used at inference.
    if (
        args.guided_noise_policy in {"transfer", "both"}
        and args.guided_action_authorization_dir is not None
    ):
        reference_model, reference_initialization = load_base_model(
            args.official_checkpoint, args.architecture_checkpoint,
            device, args.n_highest_peaks,
        )
        if reference_initialization not in {"official_embedding", "official_embedding_slim"}:
            raise RuntimeError("guided reference encoder has unexpected initialization")
        reference_package = torch_load_compat(args.guided_reference_checkpoint, map_location="cpu")
        if (
            reference_package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
            or not reference_package.get("inference_clean_only")
            or reference_package.get("P2b_used")
        ):
            raise RuntimeError("guided reference checkpoint violates shared-embedding contract")
        reference_model.load_state_dict(reference_package["model_state"], strict=True)
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        reference_model.eval()
        reference_encoded = encode_rows(
            reference_model, store, reachable_rows, device,
            args.eval_batch_size, False, "E13-reference-fp32",
        )
        reference_index = {int(row): index for index, row in enumerate(reachable_rows)}
        selected_rows: dict[int, str] = {}
        for query in np.unique(train_guided["query_index"].to_numpy(np.int64)):
            _, candidate_rows, local_ptr, _ = graph.query_block(int(query))
            positive_rows = np.asarray(candidate_rows[: int(local_ptr[1])], dtype=np.int64)
            query_vector = reference_encoded[reference_index[int(graph.query_row[int(query)])]]
            positive_vectors = reference_encoded[
                [reference_index[int(row)] for row in positive_rows]
            ]
            order = np.argsort(-(positive_vectors @ query_vector), kind="stable")[:3]
            chosen = tuple(map(int, positive_rows[order]))
            if not chosen:
                raise RuntimeError(f"guided query {query} lacks a positive reference")
            selected_rows[int(query)] = ";".join(map(str, chosen))
        train_guided = train_guided.copy()
        train_guided["positive_reference_rows"] = train_guided["query_index"].astype(int).map(
            selected_rows
        )
        if train_guided["positive_reference_rows"].isna().any():
            raise RuntimeError("failed to assign mature-E8 top3 references")
        del reference_model, reference_encoded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    guided_examples = make_guided_noise_examples(
        graph, train_guided, official_rank, official_margin,
        args.positive_spectra, args.negative_molecules,
    ) if args.guided_noise_policy != "none" else []
    guided_risk_examples = make_guided_noise_examples(
        graph, train_guided_risk, official_rank, official_margin,
        args.positive_spectra, args.negative_molecules,
    ) if args.guided_noise_policy == "selected" else []
    if args.guided_noise_policy != "none" and (
        len(guided_examples) < 500
        or len(set(example.identity for example in guided_examples)) < 250
    ):
        raise RuntimeError("guided positive-noise training pool is unexpectedly small")
    if args.guided_risk_control_ratio > 0 and (
        len(guided_risk_examples) < 100
        or len(set(example.identity for example in guided_risk_examples)) < 50
    ):
        raise RuntimeError("guided action-specific risk-control pool is unexpectedly small")
    guided_teacher_replay: dict[str, float] = {}
    branch_gradient_audit: dict = {}
    effective_guided_noise_weight = float(args.guided_noise_weight)

    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    initial_package: dict = {}
    initial_decision: dict = {}
    if args.initial_student_checkpoint is not None:
        initial_package = torch_load_compat(args.initial_student_checkpoint, map_location="cpu")
        if initial_decision_path is None:
            raise RuntimeError("initial decision path was not resolved")
        initial_decision = json.loads(initial_decision_path.read_text(encoding="utf-8"))
        initial_configuration = initial_decision.get("configuration", {})
        if (
            initial_package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
            or int(initial_package.get("outer_fold", -1)) != args.outer_fold
            or not initial_package.get("inference_clean_only")
            or initial_package.get("P2b_used")
            or initial_decision.get("status")
            != "noise_final_e4a_direct_augmentation_complete"
            or not initial_decision.get("formal")
            or int(initial_configuration.get("outer_fold", -1)) != args.outer_fold
            or int(initial_configuration.get("formula_fold_seed", -1))
            != args.formula_fold_seed
            or int(initial_configuration.get("seed", -1))
            != int(initial_package.get("seed", -2))
        ):
            raise RuntimeError("initial student checkpoint violates the E14 outer-fold contract")
        model.load_state_dict(initial_package["model_state"], strict=True)
        initialization = "mature_e4a_continuation"
    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    # Gradients stay on, stochastic dropout stays off.
    model.eval()
    official_action_targets = (
        encode_official_action_targets(
            model, store, action_examples, device, args.eval_batch_size,
        )
        if args.direct_transfer_mode == "official_action" else None
    )
    head_parameters = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": head_parameters, "lr": args.head_lr, "weight_decay": args.weight_decay},
        {"params": backbone_parameters, "lr": args.backbone_lr, "weight_decay": 0.0},
    ])
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    initial_encoded = encode_rows(
        model, store, reachable_rows, device, args.eval_batch_size, False, "E4A-init-fp32",
    )
    initial_cosine = np.einsum("ij,ij->i", initial_encoded, official_encoded)
    baseline_rank, baseline_summary = evaluate_embeddings(
        graph, reachable_rows, official_encoded, held_queries,
    )
    initial_rank, initial_summary = evaluate_embeddings(
        graph, reachable_rows, initial_encoded, held_queries,
    )
    if args.initial_student_checkpoint is None:
        initial_mismatches = int(np.sum(initial_rank != baseline_rank))
        if float(np.mean(initial_cosine)) < 0.9999 or initial_mismatches / len(held_queries) > 0.001:
            raise RuntimeError(
                f"zero-change gate failed: cos={np.mean(initial_cosine):.7f}, "
                f"rank mismatches={initial_mismatches}/{len(held_queries)}"
            )
    else:
        if initial_ledger_path is None:
            raise RuntimeError("initial held-ledger path was not resolved")
        if initial_ledger_path.is_file():
            initial_ledger = pd.read_csv(initial_ledger_path).sort_values(
                "query_index", kind="stable"
            )
            if not np.array_equal(
                initial_ledger["query_index"].to_numpy(np.int64), held_queries
            ):
                raise RuntimeError("initial checkpoint held ledger does not match outer fold")
            initial_mismatches: int | None = int(np.sum(
                initial_rank != initial_ledger["final_rank"].to_numpy(np.int16)
            ))
            if initial_mismatches:
                raise RuntimeError(
                    f"mature initialization replay changed {initial_mismatches} held ranks"
                )
            initial_replay_verification = "checkpoint+decision+held_ledger+graph"
        else:
            held_summary = initial_decision.get("held_clean", {})
            reproduced = {
                "n_queries": int(len(initial_rank)),
                "errors": int(np.sum(initial_rank != 1)),
                "corrected": int(np.sum((baseline_rank != 1) & (initial_rank == 1))),
                "introduced": int(np.sum((baseline_rank == 1) & (initial_rank != 1))),
            }
            expected = {key: int(held_summary.get(key, -1)) for key in reproduced}
            if reproduced["n_queries"] != expected["n_queries"]:
                raise RuntimeError(
                    "mature initialization query count disagrees with decision: "
                    f"reproduced={reproduced} expected={expected}"
                )
            # A freshly loaded CUDA model can change a handful of strict ranks
            # at near-exact score ties.  The missing per-query ledger cannot be
            # reconstructed after the fact, so compare the immutable decision
            # aggregates under the same 0.1% numerical-replay tolerance used by
            # the official baseline gate.  This is not a performance tolerance:
            # larger drift still fails closed before optimization.
            replay_count_tolerance = max(1, int(math.ceil(0.001 * len(initial_rank))))
            aggregate_delta = {
                key: int(reproduced[key] - expected[key])
                for key in ("errors", "corrected", "introduced")
            }
            if any(abs(value) > replay_count_tolerance for value in aggregate_delta.values()):
                raise RuntimeError(
                    "mature initialization aggregate replay disagrees with decision: "
                    f"reproduced={reproduced} expected={expected} "
                    f"tolerance={replay_count_tolerance}"
                )
            expected_recall = float(held_summary.get("recall1", float("nan")))
            if not np.isfinite(expected_recall) or not np.isclose(
                float(np.mean(initial_rank == 1)), expected_recall,
                rtol=0.0,
                atol=replay_count_tolerance / max(len(initial_rank), 1) + 1e-12,
            ):
                raise RuntimeError(
                    "mature initialization recall replay disagrees with decision"
                )
            initial_mismatches = None
            initial_replay_verification = "checkpoint+decision+graph_aggregate"
    training_anchor_by_row = {
        int(row): initial_encoded[index] for index, row in enumerate(reachable_rows)
    }
    initial_reference_encoded = initial_encoded.copy()
    if args.guided_noise_policy == "selected":
        guided_teacher_replay = audit_guided_teacher_replay(
            model, store, guided_examples, device, args,
        )
        print(f"[E14 teacher replay] {json.dumps(guided_teacher_replay)}", flush=True)
        # Measure each branch at the identical mature initialization before any
        # update.  Historical E5 mixed branches before clipping, hiding whether
        # P was diluted or dominated the validated N/safety operating regime.
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        gradient_records: dict[str, list[torch.Tensor | None]] = {}
        branch_losses: dict[str, float] = {}
        branch_norms: dict[str, float] = {}
        audit_batches = {
            "n_action": action_examples[: min(4, len(action_examples))],
            "safety": safety_examples[: min(4, len(safety_examples))],
            "p_corrective": guided_examples[: min(4, len(guided_examples))],
            "p_risk": guided_risk_examples[: min(4, len(guided_risk_examples))],
        }
        for branch, batch in audit_batches.items():
            if not batch:
                continue
            if branch == "n_action":
                branch_loss, _ = direct_action_loss(
                    model, store, batch, training_anchor_by_row, device, args,
                    official_action_targets,
                )
            elif branch == "safety":
                branch_loss, _ = safety_loss(
                    model, store, batch, training_anchor_by_row, device, args,
                )
            else:
                branch_loss, _ = guided_noise_loss(
                    model, store, batch, training_anchor_by_row, device, args,
                )
            norm, gradients = detached_loss_gradients(
                branch_loss, trainable_parameters,
            )
            branch_losses[branch] = float(branch_loss.detach())
            branch_norms[branch] = float(norm)
            gradient_records[branch] = gradients
        pairwise_cosine: dict[str, float] = {}
        names = sorted(gradient_records)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                pairwise_cosine[f"{left}__{right}"] = gradient_cosine(
                    gradient_records[left], gradient_records[right]
                )
        branch_gradient_audit = {
            "examples_per_branch": {
                key: int(len(value)) for key, value in audit_batches.items()
            },
            "loss": branch_losses,
            "gradient_l2_norm": branch_norms,
            "gradient_cosine": pairwise_cosine,
            "p_corrective_to_n_norm_ratio": float(
                branch_norms.get("p_corrective", float("nan"))
                / max(branch_norms.get("n_action", 0.0), 1e-12)
            ),
            "p_corrective_to_safety_norm_ratio": float(
                branch_norms.get("p_corrective", float("nan"))
                / max(branch_norms.get("safety", 0.0), 1e-12)
            ),
        }
        if args.guided_auto_balance:
            base_components = [
                (gradient_records["n_action"], 1.0),
                (gradient_records["safety"], args.safety_stream_weight),
            ]
            risk_ratio = (
                args.guided_risk_control_ratio
                if "p_risk" in gradient_records else 0.0
            )
            normalizer = 1.0 + risk_ratio
            guided_components = [(
                gradient_records["p_corrective"], 1.0 / normalizer,
            )]
            if "p_risk" in gradient_records and args.guided_risk_control_ratio > 0:
                guided_components.append((
                    gradient_records["p_risk"], risk_ratio / normalizer,
                ))
            base_norm = combined_gradient_norm(base_components)
            guided_norm = combined_gradient_norm(guided_components)
            balance_scale = min(1.0, max(0.05, base_norm / max(guided_norm, 1e-12)))
            effective_guided_noise_weight = float(
                args.guided_noise_weight * balance_scale
            )
            branch_gradient_audit.update({
                "combined_n_safety_gradient_norm": float(base_norm),
                "combined_p_gradient_norm": float(guided_norm),
                "auto_balance_scale": float(balance_scale),
                "effective_guided_noise_weight": effective_guided_noise_weight,
            })
        print(f"[E14 branch gradients] {json.dumps(branch_gradient_audit)}", flush=True)
        del gradient_records
        model.zero_grad(set_to_none=True)
    del initial_encoded
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rng = np.random.default_rng(args.seed)
    # Keep the already-validated N/safety sampling stream bitwise independent
    # of whether P-arm is enabled.  This makes the P-weight scan a true paired
    # intervention rather than a hidden resampling experiment.
    positive_rng = np.random.default_rng(args.seed + 104729)
    guided_rng = np.random.default_rng(args.seed + 209759)
    history = []
    epochs = 1 if args.smoke else args.epochs
    for epoch in range(1, epochs + 1):
        model.eval()
        epoch_actions = identity_balanced_epoch(
            action_examples, rng, args.views_per_identity,
        )
        if args.error_views_per_identity > 0:
            extra_errors = identity_balanced_epoch(
                error_action_examples, rng, args.error_views_per_identity,
            )
            epoch_actions.extend(extra_errors)
            rng.shuffle(epoch_actions)
        safety_epoch = identity_balanced_epoch(safety_examples, rng, 1)
        positive_epoch = (
            identity_balanced_epoch(
                positive_examples, positive_rng, args.positive_views_per_identity,
            ) if positive_examples else []
        )
        guided_epoch = (
            identity_balanced_epoch(
                guided_examples, guided_rng, args.guided_noise_views_per_identity,
            ) if guided_examples else []
        )
        guided_risk_epoch = (
            identity_balanced_epoch(guided_risk_examples, guided_rng, 1)
            if guided_risk_examples else []
        )
        if args.smoke:
            epoch_actions = epoch_actions[:16]
            safety_epoch = safety_epoch[:32]
            positive_epoch = positive_epoch[:32]
            guided_epoch = guided_epoch[:32]
            guided_risk_epoch = guided_risk_epoch[:32]
        action_schedule_sha256 = sampling_schedule_sha256(epoch_actions)
        safety_schedule_sha256 = sampling_schedule_sha256(safety_epoch)
        steps = math.ceil(len(epoch_actions) / args.batch_actions)
        totals: dict[str, float] = {}
        started = time.time()
        safety_cursor = 0
        positive_cursor = 0
        guided_cursor = 0
        guided_risk_cursor = 0
        for step, action_batch in enumerate(batched(epoch_actions, args.batch_actions), start=1):
            safety_size = max(1, int(round(len(action_batch) * args.safety_ratio)))
            if safety_cursor + safety_size > len(safety_epoch):
                rng.shuffle(safety_epoch)
                safety_cursor = 0
            safe_batch = safety_epoch[safety_cursor:safety_cursor + safety_size]
            safety_cursor += safety_size
            optimizer.zero_grad(set_to_none=True)
            action_loss, action_log = direct_action_loss(
                model, store, action_batch, training_anchor_by_row, device, args,
                official_action_targets,
            )
            scaler.scale(action_loss).backward()
            positive_log: dict[str, float] = {}
            positive_loss_value = 0.0
            if positive_epoch and args.positive_stream_weight > 0:
                positive_size = max(1, int(round(len(action_batch) * args.positive_ratio)))
                if positive_cursor + positive_size > len(positive_epoch):
                    positive_rng.shuffle(positive_epoch)
                    positive_cursor = 0
                positive_batch = positive_epoch[positive_cursor:positive_cursor + positive_size]
                positive_cursor += positive_size
                positive_loss, positive_log = positive_arm_loss(
                    model, store, positive_batch, training_anchor_by_row, device, args,
                )
                positive_loss_value = float(positive_loss.detach())
                scaler.scale(args.positive_stream_weight * positive_loss).backward()
            guided_log: dict[str, float] = {}
            guided_loss_value = 0.0
            if guided_epoch and args.guided_noise_policy != "none":
                guided_size = max(1, int(round(len(action_batch) * args.guided_noise_ratio)))
                if guided_cursor + guided_size > len(guided_epoch):
                    guided_rng.shuffle(guided_epoch)
                    guided_cursor = 0
                guided_batch = guided_epoch[guided_cursor:guided_cursor + guided_size]
                guided_cursor += guided_size
                if guided_risk_epoch and args.guided_risk_control_ratio > 0:
                    risk_size = max(
                        1, int(round(guided_size * args.guided_risk_control_ratio))
                    )
                    if guided_risk_cursor + risk_size > len(guided_risk_epoch):
                        guided_rng.shuffle(guided_risk_epoch)
                        guided_risk_cursor = 0
                    guided_batch = guided_batch + guided_risk_epoch[
                        guided_risk_cursor:guided_risk_cursor + risk_size
                    ]
                    guided_risk_cursor += risk_size
                guided_loss, guided_log = guided_noise_loss(
                    model, store, guided_batch, training_anchor_by_row, device, args,
                )
                guided_loss_value = float(guided_loss.detach())
                scaler.scale(effective_guided_noise_weight * guided_loss).backward()
            safe_loss, safe_log = safety_loss(
                model, store, safe_batch, training_anchor_by_row, device, args,
            )
            scaler.scale(args.safety_stream_weight * safe_loss).backward()
            scaler.unscale_(optimizer)
            head_grad_norm = gradient_l2_norm(head_parameters)
            backbone_grad_norm = gradient_l2_norm(backbone_parameters)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.grad_clip,
            )
            grad_norm_value = float(grad_norm)
            clip_applied = float(grad_norm_value > args.grad_clip)
            clip_scale = min(1.0, args.grad_clip / max(grad_norm_value, 1e-12))
            scaler.step(optimizer)
            scaler.update()
            log = {
                "loss": (
                    float(action_loss.detach())
                    + args.safety_stream_weight * float(safe_loss.detach())
                    + args.positive_stream_weight * positive_loss_value
                    + effective_guided_noise_weight * guided_loss_value
                ),
                "gradient_norm": grad_norm_value,
                "head_gradient_norm": head_grad_norm,
                "backbone_gradient_norm": backbone_grad_norm,
                "gradient_clip_applied": clip_applied,
                "gradient_clip_scale": clip_scale,
                **action_log, **positive_log, **guided_log, **safe_log,
            }
            for key, value in log.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            if step % 25 == 0 or step == steps:
                print(
                    f"[E4A epoch={epoch}] {step}/{steps} "
                    f"loss={totals['loss']/step:.5f} grad={totals['gradient_norm']/step:.4f}",
                    flush=True,
                )
        record = {key: value / steps for key, value in totals.items()}
        record.update({
            "epoch": epoch,
            "steps": steps,
            "seconds": time.time() - started,
            "action_sampling_schedule_sha256": action_schedule_sha256,
            "safety_sampling_schedule_sha256": safety_schedule_sha256,
        })
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)

    final_encoded = encode_rows(
        model, store, reachable_rows, device, args.eval_batch_size, False, "E4A-final-fp32",
    )
    final_rank, final_summary = evaluate_embeddings(
        graph, reachable_rows, final_encoded, held_queries,
    )
    baseline_detail_rank, baseline_top_molecule, baseline_full_margin = full_graph_query_details(
        graph, reachable_rows, official_encoded, held_queries,
    )
    initial_detail_rank, initial_top_molecule, initial_full_margin = full_graph_query_details(
        graph, reachable_rows, initial_reference_encoded, held_queries,
    )
    final_detail_rank, final_top_molecule, final_full_margin = full_graph_query_details(
        graph, reachable_rows, final_encoded, held_queries,
    )
    if not np.array_equal(baseline_detail_rank, baseline_rank):
        raise RuntimeError("full-graph baseline detail ranks disagree with primary evaluation")
    if not np.array_equal(initial_detail_rank, initial_rank):
        raise RuntimeError("full-graph initialization detail ranks disagree with primary evaluation")
    if not np.array_equal(final_detail_rank, final_rank):
        raise RuntimeError("full-graph final detail ranks disagree with primary evaluation")
    final_preservation = np.einsum("ij,ij->i", final_encoded, initial_reference_encoded)
    final_official_cosine = np.einsum("ij,ij->i", final_encoded, official_encoded)
    old_correct, new_correct = baseline_rank == 1, final_rank == 1
    initial_correct = initial_rank == 1
    delta_ci = formula_bootstrap_delta(
        baseline_rank, final_rank, graph.query_formula[held_queries],
        args.bootstrap_resamples, args.seed,
    )
    incremental_ci = formula_bootstrap_delta(
        initial_rank, final_rank, graph.query_formula[held_queries],
        args.bootstrap_resamples, args.seed + 1,
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
        "initialization_recall1": initial_summary["recall1"],
        "incremental_delta_recall1": float(
            final_summary["recall1"] - initial_summary["recall1"]
        ),
        "initialization_mrr": initial_summary["mrr"],
        "incremental_delta_mrr": float(final_summary["mrr"] - initial_summary["mrr"]),
        "initialization_near_recall1": initial_summary["near_recall1"],
        "incremental_delta_near_recall1": float(
            final_summary["near_recall1"] - initial_summary["near_recall1"]
        ),
        "incremental_corrected": int(np.sum(~initial_correct & new_correct)),
        "incremental_introduced": int(np.sum(initial_correct & ~new_correct)),
        "incremental_risk_net": int(
            np.sum(~initial_correct & new_correct) - 2 * np.sum(initial_correct & ~new_correct)
        ),
        "initialization_formula_cluster_delta_recall1": incremental_ci,
        "preservation_vs_initialization_mean": float(np.mean(final_preservation)),
        "preservation_vs_initialization_p01": float(np.quantile(final_preservation, 0.01)),
        "cosine_vs_official_mean": float(np.mean(final_official_cosine)),
        "mean_full_margin_delta_vs_official": float(
            np.mean(final_full_margin - baseline_full_margin)
        ),
        "mean_full_margin_delta_vs_initialization": float(
            np.mean(final_full_margin - initial_full_margin)
        ),
        "top_molecule_changed_vs_official": int(np.sum(
            final_top_molecule != baseline_top_molecule
        )),
        "wrong_to_different_wrong": int(np.sum(
            (baseline_rank != 1) & (final_rank != 1)
            & (final_top_molecule != baseline_top_molecule)
        )),
    })
    if args.positive_stream_weight > 0 and not held_positive.empty:
        row_to_final = {int(row): final_encoded[index] for index, row in enumerate(reachable_rows)}
        row_to_official = {int(row): official_encoded[index] for index, row in enumerate(reachable_rows)}
        pair_official: list[float] = []
        pair_final: list[float] = []
        for row in held_positive.itertuples(index=False):
            q, p = int(row.query_row), int(row.positive_row)
            pair_official.append(float(np.dot(row_to_official[q], row_to_official[p])))
            pair_final.append(float(np.dot(row_to_final[q], row_to_final[p])))
        pair_official_array = np.asarray(pair_official, dtype=np.float64)
        pair_final_array = np.asarray(pair_final, dtype=np.float64)
        identity_delta = pd.DataFrame({
            "identity": held_positive["query_ik14"].astype(str).to_numpy(),
            "delta": pair_final_array - pair_official_array,
        }).groupby("identity", sort=True)["delta"].mean().to_numpy()
        final_summary["held_cross_condition_positive"] = {
            "pairs": int(len(held_positive)),
            "identities": int(held_positive["query_ik14"].nunique()),
            "baseline_cosine": float(np.mean(pair_official_array)),
            "student_cosine": float(np.mean(pair_final_array)),
            "delta_cosine": float(np.mean(pair_final_array - pair_official_array)),
            "identity_mean_delta_cosine": float(np.mean(identity_delta)),
            "fraction_pairs_improved": float(np.mean(pair_final_array > pair_official_array)),
        }
    held_action_query = np.unique(held_action["query_index"].to_numpy(np.int64))
    held_action_mask = np.isin(held_queries, held_action_query)
    if np.any(held_action_mask):
        final_summary["held_action_clean"] = {
            "queries": int(np.sum(held_action_mask)),
            "baseline_accuracy": float(np.mean(baseline_rank[held_action_mask] == 1)),
            "student_accuracy": float(np.mean(final_rank[held_action_mask] == 1)),
            "corrected": int(np.sum((baseline_rank[held_action_mask] != 1) & (final_rank[held_action_mask] == 1))),
            "introduced": int(np.sum((baseline_rank[held_action_mask] == 1) & (final_rank[held_action_mask] != 1))),
        }
    if args.positive_stream_weight > 0 and not held_positive.empty:
        held_positive_query = np.unique(held_positive["query_index"].to_numpy(np.int64))
        held_positive_mask = np.isin(held_queries, held_positive_query)
        final_summary["held_positive_clean"] = {
            "queries": int(np.sum(held_positive_mask)),
            "baseline_accuracy": float(np.mean(baseline_rank[held_positive_mask] == 1)),
            "student_accuracy": float(np.mean(final_rank[held_positive_mask] == 1)),
            "delta_accuracy": float(
                np.mean(final_rank[held_positive_mask] == 1)
                - np.mean(baseline_rank[held_positive_mask] == 1)
            ),
            "corrected": int(np.sum((baseline_rank[held_positive_mask] != 1) & (final_rank[held_positive_mask] == 1))),
            "introduced": int(np.sum((baseline_rank[held_positive_mask] == 1) & (final_rank[held_positive_mask] != 1))),
        }
    if args.guided_noise_policy != "none" and not held_guided.empty:
        held_guided_query = np.unique(held_guided["query_index"].to_numpy(np.int64))
        held_guided_mask = np.isin(held_queries, held_guided_query)
        guided_scope_summary = {
            "queries": int(np.sum(held_guided_mask)),
            "baseline_accuracy": float(np.mean(baseline_rank[held_guided_mask] == 1)),
            "student_accuracy": float(np.mean(final_rank[held_guided_mask] == 1)),
            "delta_accuracy": float(
                np.mean(final_rank[held_guided_mask] == 1)
                - np.mean(baseline_rank[held_guided_mask] == 1)
            ),
            "corrected": int(np.sum(
                (baseline_rank[held_guided_mask] != 1) & (final_rank[held_guided_mask] == 1)
            )),
            "introduced": int(np.sum(
                (baseline_rank[held_guided_mask] == 1) & (final_rank[held_guided_mask] != 1)
            )),
        }
        final_summary["held_guided_action_scope"] = guided_scope_summary
        if args.guided_query_scope == "positive_deficit_errors":
            final_summary["held_guided_positive_deficit"] = guided_scope_summary

    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({
        "query_index": held_queries,
        "query_row": graph.query_row[held_queries],
        "query_ik14": graph.query_ik14[held_queries],
        "query_formula": graph.query_formula[held_queries],
        "has_near": graph.query_has_near[held_queries],
        "baseline_rank": baseline_rank,
        "initialization_rank": initial_rank,
        "final_rank": final_rank,
        "baseline_top_molecule_local": baseline_top_molecule,
        "initialization_top_molecule_local": initial_top_molecule,
        "final_top_molecule_local": final_top_molecule,
        "baseline_full_margin": baseline_full_margin,
        "initialization_full_margin": initial_full_margin,
        "final_full_margin": final_full_margin,
        "corrected": (baseline_rank != 1) & (final_rank == 1),
        "introduced": (baseline_rank == 1) & (final_rank != 1),
    }).to_csv(output / "held_per_query.csv.gz", index=False, compression="gzip")
    checkpoint = {
        "status": "noise_final_e4a_direct_shared_dreams_encoder",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "initialization": initialization,
        "policy": args.policy,
        "action_scope": args.action_scope,
        "seed": args.seed,
        "outer_fold": args.outer_fold,
        "causal_arm": args.causal_arm,
        "capacity": capacity,
        "initial_student_checkpoint_sha256": (
            sha256_file(args.initial_student_checkpoint)
            if args.initial_student_checkpoint is not None else None
        ),
        "inference_clean_only": True,
        "P2b_used": False,
        "teacher_used": bool(
            args.direct_transfer_mode == "official_action"
            or args.guided_noise_policy == "selected"
        ),
        "teacher_kind": (
            "frozen_official_raw_action_embedding"
            if args.direct_transfer_mode == "official_action"
            else (
                "outer_fold_isolated_privileged_action_margin"
                if args.guided_noise_policy == "selected" else "none"
            )
        ),
        "positive_arm_used": bool(args.positive_stream_weight > 0),
        "guided_noise_policy": args.guided_noise_policy,
        "guided_query_scope": args.guided_query_scope,
        "guided_noise_used": bool(args.guided_noise_policy != "none"),
        "guided_transfer_mode": args.guided_transfer_mode,
        "guided_recurrence_prevalence": args.guided_recurrence_prevalence,
        "guided_recurrence_max_peaks": args.guided_recurrence_max_peaks,
        "guided_teacher_target_mode": args.guided_teacher_target_mode,
        "guided_teacher_delta_fraction": args.guided_teacher_delta_fraction,
        "guided_risk_control_ratio": args.guided_risk_control_ratio,
    }
    torch.save(checkpoint, output / "final_shared_encoder.pt")
    gates = {
        "clean_recall_positive": bool(final_summary["delta_recall1"] > 0),
        "formula_ci_positive": bool(delta_ci["ci_low"] > 0),
        "corrected_gt_introduced": bool(final_summary["corrected"] > final_summary["introduced"]),
        "risk_net_positive": bool(final_summary["risk_net"] > 0),
        "near_nonnegative": bool(final_summary["delta_near_recall1"] >= 0),
        "mrr_nonnegative": bool(final_summary["delta_mrr"] >= 0),
        "preservation_ge_0_995": bool(final_summary["preservation_mean"] >= 0.995),
    }
    if args.positive_stream_weight > 0:
        gates.update({
            "cross_condition_pair_cosine_positive": bool(
                final_summary["held_cross_condition_positive"]["delta_cosine"] > 0
            ),
            "cross_condition_query_recall_nonnegative": bool(
                final_summary["held_positive_clean"]["delta_accuracy"] >= 0
            ),
        })
    if args.guided_noise_policy not in {"none", "selected"}:
        gates.update({
            "guided_action_scope_corrections_positive": bool(
                final_summary["held_guided_action_scope"]["corrected"] > 0
            ),
        })
    elif args.guided_noise_policy == "selected":
        gates.update({
            "crossfit_teacher_outer_fold_excluded": True,
            "teacher_action_replay_exact": bool(
                guided_teacher_replay.get("action_margin_max_abs_error", 1.0) <= 2e-4
            ),
            "incremental_formula_ci_nonnegative": bool(
                incremental_ci["ci_low"] >= 0
            ),
            "incremental_corrected_gt_introduced": bool(
                final_summary["incremental_corrected"]
                > final_summary["incremental_introduced"]
            ),
        })
    decision = {
        "status": "noise_final_e4a_direct_augmentation_complete",
        "formal": not args.smoke,
        "configuration": vars(args) | {
            "r0_fixed_cells": FIXED_POLICY[args.policy] if args.action_selection == "fixed" else [],
        },
        "causal_action_audit": causal_action_audit,
        "capacity": capacity,
        "data": {
            "train_action_rows": len(train_actions),
            "train_action_identities": int(train_actions["query_ik14"].nunique()),
            "train_action_formulas": int(train_actions["query_formula"].nunique()),
            "train_action_baseline_errors": int(np.sum(train_actions["baseline_rank"].astype(int).ne(1))),
            "train_action_baseline_correct": int(np.sum(train_actions["baseline_rank"].astype(int).eq(1))),
            "train_action_cells": (
                train_actions.groupby(["selector", "attenuation", "step"])
                .size().rename("rows").reset_index().to_dict("records")
            ),
            "held_action_rows": len(held_action),
            "train_positive_pairs": len(train_positive),
            "train_positive_identities": int(train_positive["query_ik14"].nunique()) if not train_positive.empty else 0,
            "train_positive_formulas": int(train_positive["query_formula"].nunique()) if not train_positive.empty else 0,
            "held_positive_pairs": len(held_positive),
            "train_guided_rows": len(train_guided),
            "train_guided_identities": int(train_guided["query_ik14"].nunique()) if not train_guided.empty else 0,
            "train_guided_formulas": int(train_guided["query_formula"].nunique()) if not train_guided.empty else 0,
            "train_guided_policies": (
                train_guided.groupby(["guided_family", "guided_dose"])
                .size().rename("rows").reset_index().to_dict("records")
                if not train_guided.empty else []
            ),
            "train_guided_risk_rows": len(train_guided_risk),
            "train_guided_risk_identities": int(
                train_guided_risk["query_ik14"].nunique()
            ) if not train_guided_risk.empty else 0,
            "train_guided_risk_formulas": int(
                train_guided_risk["query_formula"].nunique()
            ) if not train_guided_risk.empty else 0,
            "train_guided_risk_kinds": (
                train_guided_risk["control_kind"].value_counts().astype(int).to_dict()
                if not train_guided_risk.empty else {}
            ),
            "held_guided_rows": len(held_guided),
            "held_queries": len(held_queries),
            "safety_queries": len(safety_examples),
        },
        "zero_change_gate": {
            "preservation_mean": float(np.mean(initial_cosine)),
            "rank_mismatches": initial_mismatches,
            "verification": (
                initial_replay_verification
                if args.initial_student_checkpoint is not None
                else "official_checkpoint_reproduction"
            ),
        },
        "guided_teacher_replay": guided_teacher_replay,
        "branch_gradient_audit": branch_gradient_audit,
        "effective_guided_noise_weight": effective_guided_noise_weight,
        "held_clean": final_summary,
        "gates": gates,
        "pass_to_multifold": bool(all(gates.values())),
        "history": history,
        "contracts": {
            "shared_query_reference_encoder": True,
            "model_weights_changed": True,
            "last_transformer_blocks_and_official_head_trainable": True,
            "clean_and_augmented_raw_spectra_train_same_encoder": True,
            "direct_transfer_mode": args.direct_transfer_mode,
            "rank_reference_mode": args.rank_reference_mode,
            "official_action_targets_frozen_before_optimizer": bool(
                args.direct_transfer_mode == "official_action"
            ),
            "official_reference_anchors_training_only": bool(
                args.rank_reference_mode == "official"
            ),
            "real_cross_condition_positive_pairs_train_same_encoder": bool(args.positive_stream_weight > 0),
            "real_positive_guided_peak_noise_trains_same_encoder": bool(args.guided_noise_policy != "none"),
            "guided_queries_are_positive_deficit_official_errors": bool(
                args.guided_noise_policy not in {"none", "selected"}
                and args.guided_query_scope == "positive_deficit_errors"
            ),
            "guided_query_scope": args.guided_query_scope,
            "guided_action_cells_fixed_globally_before_training": bool(
                args.guided_noise_policy not in {"none", "selected"}
            ),
            "guided_transfer_mode": args.guided_transfer_mode,
            "guided_recurrence_recipe": {
                "reference_policy": "top3",
                "minimum_reference_prevalence": args.guided_recurrence_prevalence,
                "maximum_transferred_peaks": args.guided_recurrence_max_peaks,
                "dose": 0.50,
            },
            "guided_nonhistorical_recipe_formally_authorized": bool(
                guided_authorization_report or guided_crossfit_reports
            ),
            "guided_reference_policy": (
                "formula_outer_fold_excluded_per_query_selected_references"
                if args.guided_noise_policy == "selected"
                else (
                    "top3_by_frozen_mature_e8_embedding"
                    if guided_authorization_report else "historical_manifest"
                )
            ),
            "guided_action_outcomes_used_for_per_query_selection": bool(
                args.guided_noise_policy == "selected"
            ),
            "guided_action_specific_risk_controls_used": bool(
                args.guided_noise_policy == "selected"
                and args.guided_risk_control_ratio > 0
            ),
            "guided_teacher_target_mode": args.guided_teacher_target_mode,
            "positive_pair_selection_uses_model_outcome": False,
            "action_recipe_fixed_before_this_training_run": True,
            "causal_attribution_arm": args.causal_arm,
            "causal_arm_changes_only_action_view": bool(args.causal_arm != "legacy"),
            "matched_control_selection_uses_outcome": False,
            "causal_sampler_keys_arm_invariant": bool(args.causal_arm != "legacy"),
            "causal_candidate_references_arm_invariant": bool(args.causal_arm != "legacy"),
            "action_selection": args.action_selection,
            "training_only_outcome_mined_actions": bool(args.action_selection == "outcome_mined"),
            "action_outcomes_used_for_training_action_selection": bool(
                args.action_selection == "outcome_mined"
            ),
            "action_outcomes_used_in_loss_or_sample_weight": bool(
                args.guided_noise_policy == "selected"
            ),
            "identity_equal_action_weighting": True,
            "action_views_per_identity_per_epoch": args.views_per_identity,
            "additional_error_views_per_identity_per_epoch": args.error_views_per_identity,
            "formula_held_out": True,
            "dropout_disabled_during_gradient_training": True,
            "inference_clean_spectrum_only": True,
            "teacher": (
                "frozen_official_raw_action_embedding"
                if args.direct_transfer_mode == "official_action"
                else (
                    "training_only_action_mining"
                    if args.action_selection == "outcome_mined"
                    else (
                        "outer_fold_isolated_privileged_action_margin"
                        if args.guided_noise_policy == "selected" else "forbidden"
                    )
                )
            ),
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "provenance": {
            "r0_report_sha256": sha256_file(args.r0_dir / "report.json"),
            "r0_actions_sha256": sha256_file(args.r0_dir / "training_actions.csv.gz"),
            "outcome_action_report_sha256": (
                sha256_file(args.outcome_action_dir / "report.json")
                if args.action_selection == "outcome_mined" else None
            ),
            "outcome_action_manifest_sha256": (
                sha256_file(args.outcome_action_dir / "corrective_teacher_actions.csv.gz")
                if args.action_selection == "outcome_mined" else None
            ),
            "positive_report_sha256": (
                sha256_file(args.positive_manifest_dir / "report.json")
                if args.positive_stream_weight > 0 else None
            ),
            "positive_pairs_sha256": (
                sha256_file(args.positive_manifest_dir / "positive_pairs.csv.gz")
                if args.positive_stream_weight > 0 else None
            ),
            "guided_intensity_report_sha256": (
                sha256_file(args.guided_intensity_dir / "report.json")
                if args.guided_noise_policy not in {"none", "selected"} else None
            ),
            "guided_intensity_manifest_sha256": (
                sha256_file(args.guided_intensity_dir / "action_manifest.csv.gz")
                if args.guided_noise_policy not in {"none", "selected"} else None
            ),
            "guided_transfer_report_sha256": (
                sha256_file(args.guided_transfer_dir / "report.json")
                if args.guided_noise_policy not in {"none", "selected"} else None
            ),
            "guided_transfer_manifest_sha256": (
                sha256_file(args.guided_transfer_dir / "action_manifest.csv.gz")
                if args.guided_noise_policy not in {"none", "selected"} else None
            ),
            "guided_action_authorization_report_sha256": (
                sha256_file(args.guided_action_authorization_dir / "report.json")
                if args.guided_action_authorization_dir is not None else None
            ),
            "guided_reference_checkpoint_sha256": (
                sha256_file(args.guided_reference_checkpoint)
                if args.guided_reference_checkpoint is not None else None
            ),
            "initial_student_checkpoint_sha256": (
                sha256_file(args.initial_student_checkpoint)
                if args.initial_student_checkpoint is not None else None
            ),
            "guided_crossfit_report_sha256": (
                sha256_file(
                    args.guided_crossfit_root / f"fold_{args.outer_fold}" / "report.json"
                )
                if args.guided_noise_policy == "selected" else None
            ),
            "guided_crossfit_manifest_sha256": (
                sha256_file(
                    args.guided_crossfit_root / f"fold_{args.outer_fold}" / "selected_actions.csv.gz"
                )
                if args.guided_noise_policy == "selected" else None
            ),
            "guided_crossfit_risk_controls_sha256": (
                sha256_file(
                    args.guided_crossfit_root / f"fold_{args.outer_fold}" / "risk_controls.csv.gz"
                )
                if args.guided_noise_policy == "selected" else None
            ),
            "guided_crossfit_capacity_amendment_sha256": (
                sha256_file(
                    args.guided_crossfit_root
                    / f"fold_{args.outer_fold}" / "capacity_amendment.json"
                )
                if (
                    args.guided_noise_policy == "selected"
                    and (
                        args.guided_crossfit_root
                        / f"fold_{args.outer_fold}" / "capacity_amendment.json"
                    ).is_file()
                ) else None
            ),
            "graph_sha256": sha256_file(args.graph),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "held-formula development result for a directly fine-tuned shared embedding. "
            "Historical 3.85 pp is action-oracle headroom, not a promised weight gain."
        ),
    }
    # Path objects are not JSON serialisable.
    decision["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in decision["configuration"].items()
    }
    json_dump(output / "decision.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
