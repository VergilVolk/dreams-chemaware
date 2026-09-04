"""Build the mature N/P full-candidate counterfactual residual teacher.

All action cells are fixed before this run.  The outer held formula fold is
excluded before action outcomes are computed.  The output retains one signed
target-minus-control residual for every negative candidate molecule instead of
collapsing each action to the current hardest negative.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_noise_final_l0_action_learnability_ledger import encode_action_variants  # noqa: E402
from audit_noise_final_positive_guided_matrix import (  # noqa: E402
    apply_action as apply_positive_intensity_action,
    reference_profile,
)
from audit_noise_final_positive_peak_transfer import (  # noqa: E402
    apply_transfer as apply_positive_peak_transfer,
    recurrent_missing_peaks,
)
from e1_checkpoint_io import torch_load_compat  # noqa: E402
from noise_final_core import CandidateGraph, json_dump, sha256_file, stable_fold, strict_rank  # noqa: E402
from noise_final_cpg_core import paired_candidate_residual  # noqa: E402
from noise_v3_core import attenuate_sequence  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows, parse_controls, parse_path  # noqa: E402


STATUS = "noise_final_cpg0_mature_np_residual_teacher_complete"
N_GRID = {
    ("candidate_gradient", 0.50, step) for step in range(3, 7)
} | {
    ("role_confounder", 1.00, step) for step in range(1, 6)
}
P_INTENSITY_FAMILIES = (
    "matched_intensity_transport", "prevalence_attenuation", "consensus_projection",
)
P_INTENSITY_DOSES = (0.25, 0.50, 0.75, 1.00)
P_TRANSFER_FAMILIES = (
    "recurrent_peak_graft", "balanced_peak_exchange", "recurrent_union_mix",
)
P_TRANSFER_DOSES = (0.10, 0.25, 0.50)

ACTION_FIELDS = (
    "action_index", "source", "mechanism", "cell_id", "family", "query_index",
    "query_row", "query_ik14", "query_formula", "formula_fold", "attenuation",
    "step", "dose", "target_path", "control_paths", "positive_reference_rows",
    "wrong_reference_rows", "control_count", "negative_candidates", "clean_rank",
    "target_rank", "control0_rank", "control1_rank", "clean_margin",
    "target_margin", "control_mean_margin", "paired_advantage", "residual_mean",
    "residual_q10", "residual_q50", "residual_q90", "residual_min", "residual_max",
    "transition", "advantage_label", "clean_top_negative_local",
    "target_top_negative_local", "control_top_negative_local", "near",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--r0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a")
    parser.add_argument(
        "--positive-guided-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_positive_guided_matrix",
    )
    parser.add_argument(
        "--positive-transfer-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_positive_peak_transfer",
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument(
        "--architecture-checkpoint", type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument(
        "--clean-checkpoint", type=Path,
        default=(ROOT / "data/validation/g8r_noise_final_e4a_causal_attribution/"
                 "curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_"
                 "e4a_causal_v1_20260901_causal_clean_duplicate/seed_20260828/"
                 "fold_0/final_shared_encoder.pt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fp32-retry-batch-size", type=int, default=8)
    parser.add_argument("--actions-per-chunk", type=int, default=128)
    parser.add_argument("--advantage-threshold", type=float, default=0.01)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--positive-references", type=int, default=3)
    parser.add_argument("--minimum-reference-prevalence", type=float, default=0.67)
    parser.add_argument("--maximum-transferred-peaks", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-queries", type=int, default=0, help="debug only; zero is formal")
    return parser.parse_args()


@dataclass
class PendingAction:
    source: str
    mechanism: str
    cell_id: str
    family: str
    query: int
    target: torch.Tensor
    controls: tuple[torch.Tensor, ...]
    attenuation: float = 0.0
    step: int = 0
    dose: float = 0.0
    target_path: str = ""
    control_paths: str = ""
    positive_reference_rows: str = ""
    wrong_reference_rows: str = ""


def parse_rows(value: object) -> tuple[int, ...]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ()
    return tuple(int(part) for part in text.split(";") if part.strip())


def load_report(path: Path, expected_status: str, graph_hash: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != expected_status or not report.get("formal"):
        raise RuntimeError(f"source report is not formal {expected_status}: {path}")
    observed = report.get("provenance", {}).get("graph_sha256")
    if observed is not None and observed != graph_hash:
        raise RuntimeError(f"source report candidate graph differs: {path}")
    return report


def load_mature_model(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict]:
    package = torch_load_compat(args.clean_checkpoint, map_location="cpu")
    if (
        package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
        or package.get("P2b_used")
        or not package.get("inference_clean_only")
    ):
        raise RuntimeError("clean checkpoint is not a candidate-independent mature E4-A encoder")
    configuration = package.get("configuration", {})
    checkpoint_fold = configuration.get("outer_formula_fold", package.get("outer_formula_fold"))
    if checkpoint_fold is not None and int(checkpoint_fold) != args.outer_fold:
        raise RuntimeError(
            f"clean checkpoint fold {checkpoint_fold} does not match outer fold {args.outer_fold}"
        )
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    model.load_state_dict(package["model_state"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, {
        "initialization_kind": initialization,
        "clean_checkpoint_sha256": sha256_file(args.clean_checkpoint),
    }


def molecule_scores(vector: np.ndarray, reference: np.ndarray, positions: np.ndarray,
                    ptr: np.ndarray) -> np.ndarray:
    pair = reference[positions] @ np.asarray(vector, dtype=np.float32)
    scores = np.maximum.reduceat(pair, ptr[:-1]).astype(np.float32, copy=False)
    if len(scores) < 2 or not np.all(np.isfinite(scores)):
        raise RuntimeError("action produced an invalid molecule score vector")
    return scores


def rank_margin(scores: np.ndarray) -> tuple[int, float, int]:
    rank = strict_rank(scores)
    negative = np.asarray(scores[1:], dtype=np.float32)
    top_negative = int(np.argmax(negative)) + 1
    return rank, float(scores[0] - negative[top_negative - 1]), top_negative


def transition(clean_rank: int, target_rank: int) -> str:
    if clean_rank == 1 and target_rank == 1:
        return "protected_correct"
    if clean_rank == 1:
        return "introduced"
    if target_rank == 1:
        return "corrected"
    return "persistent_wrong"


def advantage_label(value: float, threshold: float) -> str:
    if value >= threshold:
        return "positive"
    if value <= -threshold:
        return "harmful"
    return "neutral"


def require_positive_gates(gates: dict[str, bool]) -> None:
    """Every gate is phrased as a positive condition and must be literal True."""
    invalid = {key: value for key, value in gates.items() if value is not True}
    if invalid:
        raise RuntimeError(f"CPG0 teacher gates failed: {gates}")


class RaggedWriter:
    def __init__(self, output: Path):
        self.handle = h5py.File(output, "w")
        self.ptr = self.handle.create_dataset("action_ptr", shape=(1,), maxshape=(None,), dtype="i8")
        self.ptr[0] = 0
        self.negative_local = self.handle.create_dataset(
            "negative_molecule_local", shape=(0,), maxshape=(None,), dtype="i2",
            compression="gzip", compression_opts=4, shuffle=True, chunks=(262144,),
        )
        self.clean = self._float("clean_margin")
        self.target = self._float("target_margin")
        self.control = self._float("control_mean_margin")
        self.residual = self._float("paired_residual")
        self.actions = 0
        self.elements = 0

    def _float(self, name: str):
        return self.handle.create_dataset(
            name, shape=(0,), maxshape=(None,), dtype="f4", compression="gzip",
            compression_opts=4, shuffle=True, chunks=(262144,),
        )

    def append(self, clean: np.ndarray, target: np.ndarray, control: np.ndarray,
               residual: np.ndarray) -> None:
        arrays = [np.asarray(value, dtype=np.float32) for value in (clean, target, control, residual)]
        if not arrays[0].ndim == 1 or any(value.shape != arrays[0].shape for value in arrays):
            raise RuntimeError("ragged candidate vectors disagree")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise RuntimeError("ragged candidate vectors contain non-finite values")
        count = len(arrays[0])
        left, right = self.elements, self.elements + count
        for dataset, values in zip((self.clean, self.target, self.control, self.residual), arrays):
            dataset.resize((right,))
            dataset[left:right] = values
        self.negative_local.resize((right,))
        self.negative_local[left:right] = np.arange(1, count + 1, dtype=np.int16)
        self.ptr.resize((self.actions + 2,))
        self.ptr[self.actions + 1] = right
        self.actions += 1
        self.elements = right

    def close(self, attributes: dict[str, object]) -> None:
        for key, value in attributes.items():
            self.handle.attrs[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value)
        self.handle.close()


def main() -> None:
    args = arguments()
    formal = args.max_queries == 0
    if (
        args.outer_fold not in range(5) or args.batch_size < 1 or args.actions_per_chunk < 1
        or args.fp32_retry_batch_size < 1 or args.fp32_retry_batch_size > args.batch_size
        or args.advantage_threshold <= 0
    ):
        raise ValueError("invalid CPG0 arguments")
    required = [
        args.graph, args.data, args.official_checkpoint, args.architecture_checkpoint,
        args.clean_checkpoint, args.r0_dir / "report.json", args.r0_dir / "training_actions.csv.gz",
        args.positive_guided_dir / "report.json", args.positive_guided_dir / "action_manifest.csv.gz",
        args.positive_transfer_dir / "report.json", args.positive_transfer_dir / "action_manifest.csv.gz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        raise RuntimeError(f"completed CPG0 output already exists: {args.output_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    graph_hash = sha256_file(args.graph)
    graph = CandidateGraph(args.graph)
    if formal and graph.n_queries != 23876:
        raise RuntimeError(f"formal CPG0 expects 23,876 queries, observed {graph.n_queries}")
    r0_report = load_report(
        args.r0_dir / "report.json", "noise_final_r0_faithful_s3a_manifest_complete", graph_hash,
    )
    if int(r0_report.get("contracts", {}).get("matched_controls_preserved", -1)) != 2:
        raise RuntimeError("R0 does not preserve exactly two matched controls")
    positive_report = load_report(
        args.positive_guided_dir / "report.json", "noise_final_positive_guided_matrix_complete", graph_hash,
    )
    transfer_report = load_report(
        args.positive_transfer_dir / "report.json", "noise_final_positive_peak_transfer_complete", graph_hash,
    )
    if (
        positive_report.get("contracts", {}).get("P2b_forbidden") is not True
        or transfer_report.get("contracts", {}).get("P2b_forbidden") is not True
    ):
        raise RuntimeError("a positive source violates the no-reranker contract")
    positive_manifest_path = args.positive_guided_dir / "action_manifest.csv.gz"
    if transfer_report.get("provenance", {}).get("previous_manifest_sha256") != sha256_file(
        positive_manifest_path
    ):
        raise RuntimeError("positive-transfer source was not built from the supplied positive manifest")

    folds = np.asarray([
        stable_fold(str(formula), 5, args.formula_fold_seed) for formula in graph.query_formula
    ], dtype=np.int8)
    eligible_queries = np.flatnonzero(folds != args.outer_fold)
    if args.max_queries:
        eligible_queries = eligible_queries[:args.max_queries]
    eligible_set = set(map(int, eligible_queries))

    r0 = pd.read_csv(args.r0_dir / "training_actions.csv.gz", low_memory=False)
    required_n = {
        "query_index", "query_row", "query_ik14", "query_formula", "formula_fold",
        "selector", "attenuation", "step", "target_path", "matched_control_paths",
    }
    if missing_n := required_n - set(r0.columns):
        raise RuntimeError(f"R0 action table misses columns: {sorted(missing_n)}")
    observed_fold = r0["query_formula"].astype(str).map(
        lambda value: stable_fold(value, 5, args.formula_fold_seed)
    ).to_numpy(np.int8)
    if not np.array_equal(observed_fold, r0["formula_fold"].to_numpy(np.int8)):
        raise RuntimeError("R0 formula folds disagree with the frozen fold function")
    grid_mask = np.asarray([
        (str(row.selector), float(row.attenuation), int(row.step)) in N_GRID
        for row in r0.itertuples(index=False)
    ])
    r0 = r0.loc[grid_mask & r0["query_index"].astype(int).isin(eligible_set)].copy()
    observed_grid = set(zip(r0["selector"], r0["attenuation"].astype(float), r0["step"].astype(int)))
    if formal and observed_grid != N_GRID:
        raise RuntimeError(f"mature N grid is incomplete: {sorted(N_GRID - observed_grid)}")
    if r0.duplicated(["query_index", "selector", "attenuation", "step"]).any():
        raise RuntimeError("mature N grid contains duplicate action keys")

    positive_manifest = pd.read_csv(positive_manifest_path, low_memory=False)
    required_p = {
        "query_index", "query_row", "query_ik14", "query_formula",
        "positive_reference_rows", "hardest_wrong_reference_rows",
    }
    if missing_p := required_p - set(positive_manifest.columns):
        raise RuntimeError(f"positive action manifest misses columns: {sorted(missing_p)}")
    if positive_manifest["query_index"].duplicated().any():
        raise RuntimeError("positive action manifest is not one row per query")
    transfer_manifest = pd.read_csv(
        args.positive_transfer_dir / "action_manifest.csv.gz", low_memory=False,
        usecols=["query_index", "query_row", "query_ik14", "query_formula"],
    )
    if transfer_manifest["query_index"].duplicated().any():
        raise RuntimeError("positive-transfer manifest is not one row per query")
    source_keys = ["query_index", "query_row", "query_ik14", "query_formula"]
    if not positive_manifest[source_keys].reset_index(drop=True).equals(
        transfer_manifest[source_keys].reset_index(drop=True)
    ):
        raise RuntimeError("positive intensity and transfer manifests disagree on query metadata")
    positive_manifest = positive_manifest.set_index("query_index", drop=False)
    missing_queries = eligible_set - set(map(int, positive_manifest.index))
    if missing_queries:
        raise RuntimeError(f"positive manifest misses {len(missing_queries)} eligible queries")

    reachable = set(map(int, graph.query_row))
    reachable.update(map(int, graph.pair_candidate_row))
    rows = np.asarray(sorted(reachable), dtype=np.int64)
    store = SpectrumStore(args.data, rows, args.n_highest_peaks)
    model, model_provenance = load_mature_model(args, device)
    embeddings = encode_rows(model, store, rows, device, args.batch_size, args.amp, "CPG0-clean")
    row_position = {int(row): index for index, row in enumerate(rows)}
    blocks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    clean_scores: dict[int, np.ndarray] = {}
    for query in eligible_queries:
        _, candidate_rows, ptr, _ = graph.query_block(int(query))
        positions = np.asarray([row_position[int(row)] for row in candidate_rows], dtype=np.int64)
        blocks[int(query)] = (positions, np.asarray(ptr, dtype=np.int64))
        clean_vector = embeddings[row_position[int(graph.query_row[int(query)])]]
        clean_scores[int(query)] = molecule_scores(clean_vector, embeddings, positions, ptr)

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".cpg0_", dir=args.output_dir.parent))
    compute_complete = False
    action_csv = staging / "actions.csv"
    h5_path = staging / "candidate_residuals.h5"
    writer = RaggedWriter(h5_path)
    counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    identities: defaultdict[str, set[str]] = defaultdict(set)
    formulas: defaultdict[str, set[str]] = defaultdict(set)
    omitted_ineffective: Counter[str] = Counter()
    residual_signs = Counter()
    action_index = 0
    started = time.time()
    pending: list[PendingAction] = []

    def flush(csv_writer: csv.DictWriter) -> None:
        nonlocal pending, action_index
        if not pending:
            return
        views: list[torch.Tensor] = []
        offsets: list[tuple[int, int]] = []
        for item in pending:
            left = len(views)
            views.append(item.target)
            views.extend(item.controls)
            offsets.append((left, len(views)))
        vectors = encode_action_variants(
            model, views, device, args.batch_size, args.fp32_retry_batch_size, args.amp,
        )
        for item, (left, right) in zip(pending, offsets):
            query = item.query
            positions, ptr = blocks[query]
            target_scores = molecule_scores(vectors[left], embeddings, positions, ptr)
            control_scores = [
                molecule_scores(vectors[index], embeddings, positions, ptr)
                for index in range(left + 1, right)
            ]
            clean_score = clean_scores[query]
            # Recompute through the shared core math as an independent arithmetic path.
            candidate_reference = torch.from_numpy(embeddings[positions])
            residual_t, target_t, control_t = paired_candidate_residual(
                torch.from_numpy(vectors[left]),
                torch.from_numpy(vectors[left + 1:right]),
                candidate_reference,
                ptr,
            )
            clean_margin_vector = clean_score[0] - clean_score[1:]
            target_margin_vector = target_scores[0] - target_scores[1:]
            control_margin_vector = np.mean(
                [score[0] - score[1:] for score in control_scores], axis=0,
            )
            residual = target_margin_vector - control_margin_vector
            if not (
                np.allclose(target_margin_vector, target_t.numpy(), atol=2e-6)
                and np.allclose(control_margin_vector, control_t.numpy(), atol=2e-6)
                and np.allclose(residual, residual_t.numpy(), atol=2e-6)
            ):
                raise RuntimeError("independent full-candidate residual replay disagrees")
            clean_rank, clean_margin, clean_top = rank_margin(clean_score)
            target_rank, target_margin, target_top = rank_margin(target_scores)
            control_audits = [rank_margin(score) for score in control_scores]
            control_mean_margin = float(np.mean([value[1] for value in control_audits]))
            paired_advantage = target_margin - control_mean_margin
            label = advantage_label(paired_advantage, args.advantage_threshold)
            movement = transition(clean_rank, target_rank)
            writer.append(clean_margin_vector, target_margin_vector, control_margin_vector, residual)
            record = {
                "action_index": action_index, "source": item.source,
                "mechanism": item.mechanism, "cell_id": item.cell_id, "family": item.family,
                "query_index": query, "query_row": int(graph.query_row[query]),
                "query_ik14": str(graph.query_ik14[query]),
                "query_formula": str(graph.query_formula[query]), "formula_fold": int(folds[query]),
                "attenuation": item.attenuation, "step": item.step, "dose": item.dose,
                "target_path": item.target_path, "control_paths": item.control_paths,
                "positive_reference_rows": item.positive_reference_rows,
                "wrong_reference_rows": item.wrong_reference_rows,
                "control_count": len(item.controls), "negative_candidates": len(residual),
                "clean_rank": clean_rank, "target_rank": target_rank,
                "control0_rank": control_audits[0][0],
                "control1_rank": control_audits[1][0] if len(control_audits) > 1 else "",
                "clean_margin": clean_margin, "target_margin": target_margin,
                "control_mean_margin": control_mean_margin, "paired_advantage": paired_advantage,
                "residual_mean": float(np.mean(residual)), "residual_q10": float(np.quantile(residual, .1)),
                "residual_q50": float(np.quantile(residual, .5)), "residual_q90": float(np.quantile(residual, .9)),
                "residual_min": float(np.min(residual)), "residual_max": float(np.max(residual)),
                "transition": movement, "advantage_label": label,
                "clean_top_negative_local": clean_top, "target_top_negative_local": target_top,
                "control_top_negative_local": control_audits[0][2],
                "near": bool(graph.query_has_near[query]),
            }
            csv_writer.writerow(record)
            action_index += 1
            counts[item.source] += 1
            cell_counts[item.cell_id] += 1
            labels[label] += 1
            transitions[movement] += 1
            identities[item.source].add(str(graph.query_ik14[query]))
            formulas[item.source].add(str(graph.query_formula[query]))
            residual_signs["positive_elements"] += int(np.sum(residual > 0))
            residual_signs["negative_elements"] += int(np.sum(residual < 0))
            residual_signs["zero_elements"] += int(np.sum(residual == 0))
        pending = []
        if action_index % 5000 < args.actions_per_chunk:
            print(f"[CPG0 actions] {action_index:,}; {time.time() - started:.0f}s", flush=True)

    try:
        with action_csv.open("w", newline="", encoding="utf-8") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=ACTION_FIELDS)
            csv_writer.writeheader()

            for row in r0.itertuples(index=False):
                query = int(row.query_index)
                target_path = parse_path(row.target_path)
                controls = parse_controls(row.matched_control_paths)
                step = int(row.step)
                if len(target_path) != step or any(len(value) != step for value in controls):
                    raise RuntimeError("N action/control path length differs from the mature step")
                clean = store.one(int(row.query_row))
                attenuation = float(row.attenuation)
                pending.append(PendingAction(
                    source="N", mechanism=str(row.selector),
                    cell_id=f"N:{row.selector}|a={attenuation:.2f}|step={step}",
                    family=str(row.selector), query=query,
                    target=attenuate_sequence(clean, target_path, attenuation),
                    controls=tuple(attenuate_sequence(clean, path, attenuation) for path in controls),
                    attenuation=attenuation, step=step,
                    target_path=str(row.target_path), control_paths=str(row.matched_control_paths),
                ))
                if len(pending) >= args.actions_per_chunk:
                    flush(csv_writer)

            for query in eligible_queries:
                row = positive_manifest.loc[int(query)]
                query_row = int(graph.query_row[int(query)])
                if int(row["query_row"]) != query_row:
                    raise RuntimeError(f"positive manifest query row mismatch at query {query}")
                positive_rows = parse_rows(row["positive_reference_rows"])
                wrong_rows = parse_rows(row["hardest_wrong_reference_rows"])
                if not positive_rows or not wrong_rows:
                    raise RuntimeError(f"query {query} lacks positive or wrong direction references")
                if query_row in positive_rows:
                    raise RuntimeError(f"query {query} positive action reference contains the query itself")
                clean = store.one(query_row)
                references = (
                    [store.one(value) for value in positive_rows],
                    [store.one(value) for value in wrong_rows],
                )
                profiles = [reference_profile(clean, values, args.fragment_tolerance) for values in references]
                positive_text = ";".join(map(str, positive_rows))
                wrong_text = ";".join(map(str, wrong_rows))

                for family in P_INTENSITY_FAMILIES:
                    for dose in P_INTENSITY_DOSES:
                        target = apply_positive_intensity_action(clean, *profiles[0], family, dose)
                        control = apply_positive_intensity_action(clean, *profiles[1], family, dose)
                        if torch.equal(target, clean) and torch.equal(control, clean):
                            omitted_ineffective[f"P_intensity:{family}|dose={dose:.2f}"] += 1
                            continue
                        pending.append(PendingAction(
                            source="P_intensity", mechanism=family,
                            cell_id=f"P_intensity:{family}|dose={dose:.2f}", family=family,
                            query=int(query), target=target, controls=(control,), dose=dose,
                            positive_reference_rows=positive_text, wrong_reference_rows=wrong_text,
                        ))
                        if len(pending) >= args.actions_per_chunk:
                            flush(csv_writer)

                missing = [
                    recurrent_missing_peaks(
                        clean, values, args.fragment_tolerance,
                        args.minimum_reference_prevalence, args.maximum_transferred_peaks,
                    ) for values in references
                ]
                for family in P_TRANSFER_FAMILIES:
                    for dose in P_TRANSFER_DOSES:
                        target, target_count = apply_positive_peak_transfer(
                            clean, missing[0], profiles[0][0], family, dose,
                        )
                        control, control_count = apply_positive_peak_transfer(
                            clean, missing[1], profiles[1][0], family, dose,
                        )
                        if target_count == 0 and control_count == 0:
                            omitted_ineffective[f"P_transfer:{family}|dose={dose:.2f}"] += 1
                            continue
                        pending.append(PendingAction(
                            source="P_transfer", mechanism=family,
                            cell_id=f"P_transfer:{family}|dose={dose:.2f}", family=family,
                            query=int(query), target=target, controls=(control,), dose=dose,
                            positive_reference_rows=positive_text, wrong_reference_rows=wrong_text,
                        ))
                        if len(pending) >= args.actions_per_chunk:
                            flush(csv_writer)
            flush(csv_writer)

        writer.close({
            "status": STATUS, "outer_formula_fold": args.outer_fold,
            "formula_fold_seed": args.formula_fold_seed, "action_rows": action_index,
            "candidate_residual_elements": writer.elements,
        })
        # The expensive GPU pass is complete as soon as the ragged HDF5 writer
        # closes successfully.  Mark it *before* CSV compression/report assembly
        # so any subsequent engineering failure preserves the numerical output.
        json_dump(staging / "compute_complete.json", {
            "status": "noise_final_cpg0_compute_complete",
            "action_rows": action_index,
            "candidate_residual_elements": writer.elements,
        })
        compute_complete = True
        with action_csv.open("rb") as source, gzip.open(staging / "actions.csv.gz", "wb", compresslevel=6) as target:
            shutil.copyfileobj(source, target)
        action_csv.unlink()

        expected_cells = 9 + 12 + 9
        report = {
            "status": STATUS,
            "formal": formal,
            "outer_formula_fold": args.outer_fold,
            "eligible_queries": int(len(eligible_queries)),
            "action_rows": action_index,
            "candidate_residual_elements": writer.elements,
            "sources": dict(sorted(counts.items())),
            "cells_with_effective_actions": len(cell_counts),
            "cell_counts": dict(sorted(cell_counts.items())),
            "expected_fixed_cells": expected_cells,
            "omitted_ineffective_payloads": dict(sorted(omitted_ineffective.items())),
            "advantage_labels": dict(sorted(labels.items())),
            "transitions": dict(sorted(transitions.items())),
            "residual_signs": dict(sorted(residual_signs.items())),
            "source_identities": {key: len(value) for key, value in sorted(identities.items())},
            "source_formulas": {key: len(value) for key, value in sorted(formulas.items())},
            "gates": {
                "all_three_sources_present": set(counts) == {"N", "P_intensity", "P_transfer"},
                "all_fixed_cells_have_effective_actions": len(cell_counts) == expected_cells,
                "actions_ge_10000": action_index >= 10000,
                "candidate_residuals_ge_actions": writer.elements >= action_index,
                "signed_positive_and_negative_elements_present": (
                    residual_signs["positive_elements"] > 0 and residual_signs["negative_elements"] > 0
                ),
                "held_formula_fold_absent": True,
                "P2b_forbidden": True,
                "P3_not_consumed": True,
            },
            "contracts": {
                "full_candidate_margin_vector_stored": True,
                "target_minus_matched_direction_control": True,
                "all_fixed_cells_retained_before_outer_train_selection": True,
                "harmful_signed_residuals_retained": True,
                "ineffective_payloads_reported_not_substituted": True,
                "candidate_library_not_reduced_for_holdout": True,
                "shared_mature_query_reference_encoder": True,
                "optimizer_steps": 0,
                "P2b": "forbidden",
                "P3_not_consumed": True,
            },
            "provenance": {
                "graph_sha256": graph_hash,
                "hdf5_sha256": sha256_file(args.data),
                "r0_report_sha256": sha256_file(args.r0_dir / "report.json"),
                "r0_actions_sha256": sha256_file(args.r0_dir / "training_actions.csv.gz"),
                "positive_report_sha256": sha256_file(args.positive_guided_dir / "report.json"),
                "positive_manifest_sha256": sha256_file(args.positive_guided_dir / "action_manifest.csv.gz"),
                "transfer_report_sha256": sha256_file(args.positive_transfer_dir / "report.json"),
                "transfer_manifest_sha256": sha256_file(args.positive_transfer_dir / "action_manifest.csv.gz"),
                **model_provenance,
                "script_sha256": sha256_file(Path(__file__)),
            },
            "claim_limit": (
                "Outer-train full-candidate signed action teacher under one frozen mature encoder; "
                "not a trained embedding, held-formula gain, selector, or deployable result."
            ),
        }
        json_dump(staging / "report.json", report)
        require_positive_gates(report["gates"])
        staging.replace(args.output_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        try:
            writer.handle.close()
        except Exception:
            pass
        if staging.exists():
            suffix = ".failed_complete" if compute_complete else ".failed_partial"
            failed = args.output_dir.with_name(args.output_dir.name + suffix)
            if failed.exists():
                failed = args.output_dir.with_name(
                    args.output_dir.name + f"{suffix}_{int(time.time())}"
                )
            staging.replace(failed)
            print(
                f"[CPG0 recovery] artifacts preserved at {failed}; "
                f"compute_complete={compute_complete}",
                file=sys.stderr, flush=True,
            )
        raise
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
