"""E14-M0: build an outer-fold-isolated privileged positive-noise teacher.

The E12-B 4.93 pp number is a no-op-aware union of complementary actions, not
a single globally useful perturbation.  E13 therefore could not transfer that
capacity by applying one recurrence recipe to almost every query.  For student
outer fold k, this stage loads the mature E4-A encoder trained without fold k
and mines actions only on folds != k.  It records at most one correcting action
per official-and-mature error.  This avoids the subtle leakage caused by
combining label-fold teachers that may themselves have trained on student fold
k.  The manifest is training-only supervision and is unavailable at inference.

Only positive-reference actions are built here.  The already validated E4-A
candidate-gradient/role-confounder curriculum remains the N arm of the shared
encoder trainer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_noise_final_e10_positive_residual_matrix import (  # noqa: E402
    EXPANSION_CELLS,
    CORE_INTENSITY_CELLS,
    CORE_TRANSFER_CELLS,
    cell_id,
    cell_variant,
)
from audit_noise_final_e11_reference_diversity_matrix import (  # noqa: E402
    RECIPES as E11_RECIPES,
    REFERENCE_POLICIES,
    select_rows,
)
from audit_noise_final_e12b_relaxed_recurrence_matrix import (  # noqa: E402
    RECIPES as E12_RECIPES,
    cell_name as e12_cell_name,
    relaxed_variant,
)
from audit_noise_final_e9_action_staleness import load_student, rank_margin  # noqa: E402
from audit_noise_final_positive_guided_matrix import reference_profile  # noqa: E402
from audit_noise_final_positive_peak_transfer import recurrent_missing_peaks  # noqa: E402
from calibrate_noise_final_e1_empirical import clean_instrument, decode  # noqa: E402
from noise_final_core import (  # noqa: E402
    CandidateGraph,
    json_dump,
    sha256_file,
    stable_fold,
    strict_rank,
)
from train_e1_identity import torch_load_compat  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore,
    encode_rows,
    forward_embeddings,
)


@dataclass(frozen=True)
class ActionDefinition:
    source: str
    reference_policy: str
    family: str
    dose: float
    auxiliary_dose: float
    prevalence: float
    maximum_peaks: int
    support_weighted: bool

    @property
    def action_id(self) -> str:
        weighted = "weighted" if self.support_weighted else "standard"
        return (
            f"{self.source}|{self.reference_policy}|{self.family}|"
            f"dose={self.dose:.2f}|aux={self.auxiliary_dose:.2f}|"
            f"prev={self.prevalence:.2f}|max={self.maximum_peaks}|{weighted}"
        )


def action_definitions() -> tuple[ActionDefinition, ...]:
    output: list[ActionDefinition] = []
    for family, dose, auxiliary in (
        CORE_INTENSITY_CELLS + CORE_TRANSFER_CELLS + EXPANSION_CELLS
    ):
        output.append(ActionDefinition(
            "E10B", "top3", family, float(dose), float(auxiliary),
            0.67, 5, False,
        ))
    for policy in REFERENCE_POLICIES:
        for family, dose, auxiliary in E11_RECIPES:
            output.append(ActionDefinition(
                "E11", policy, family, float(dose), float(auxiliary),
                0.67, 5, False,
            ))
    for policy in ("top3",) + REFERENCE_POLICIES:
        for maximum, dose, weighted in E12_RECIPES:
            output.append(ActionDefinition(
                "E12B", policy, "recurrent_union_mix", float(dose), 0.0,
                0.50, int(maximum), bool(weighted),
            ))
    identifiers = [item.action_id for item in output]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("E14 action definitions are not unique")
    return tuple(output)


def prior_cell_id(definition: ActionDefinition) -> str:
    """Map an E14 action back to the fixed-cell id in its source report."""
    if definition.source == "E10B":
        return cell_id(definition.family, definition.dose, definition.auxiliary_dose)
    if definition.source == "E11":
        return (
            f"{definition.reference_policy}|"
            f"{cell_id(definition.family, definition.dose, definition.auxiliary_dose)}"
        )
    if definition.source == "E12B":
        return e12_cell_name(
            definition.reference_policy, definition.maximum_peaks,
            definition.dose, definition.support_weighted,
        )
    raise RuntimeError(f"unknown E14 action source: {definition.source}")


def load_prior_safe_definitions(args: argparse.Namespace) -> tuple[tuple[ActionDefinition, ...], dict]:
    """Admit only actions that passed their preregistered fixed-cell safety gate.

    E6 showed that selecting isolated successful queries from a globally harmful
    action family does not transfer to a shared encoder.  E14 therefore cannot
    search the full recipe catalogue and keep only post-outcome successes.
    """
    specifications = {
        "E10B": (
            args.e10b_report,
            "noise_final_e10b_positive_action_expansion_complete",
        ),
        "E11": (
            args.e11_report,
            "noise_final_e11_reference_diversity_complete",
        ),
        "E12B": (
            args.e12b_report,
            "noise_final_e12b_relaxed_recurrence_complete",
        ),
    }
    reports: dict[str, dict] = {}
    passing: dict[str, set[str]] = {}
    for source, (path, expected_status) in specifications.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("status") != expected_status
            or not report.get("formal")
            or report.get("contracts", {}).get("P2b") != "forbidden"
            or report.get("contracts", {}).get("P3_consumed") not in {False, None}
        ):
            raise RuntimeError(f"E14 prior action report is not formal/safe: {path}")
        cells = set(map(str, report.get("passing_fixed_cells", [])))
        if not cells:
            raise RuntimeError(f"E14 prior report has no passing fixed cells: {path}")
        reports[source] = report
        passing[source] = cells
    catalogue = action_definitions()
    selected = tuple(
        definition for definition in catalogue
        if prior_cell_id(definition) in passing[definition.source]
    )
    if not selected:
        raise RuntimeError("E14 prior fixed-cell filter removed every action")
    unmatched = {
        source: sorted(cells - {
            prior_cell_id(definition) for definition in catalogue
            if definition.source == source
        })
        for source, cells in passing.items()
    }
    if any(unmatched.values()):
        raise RuntimeError(f"E14 cannot map passing fixed cells to actions: {unmatched}")
    audit = {
        "catalogue_actions": int(len(catalogue)),
        "prior_safe_actions": int(len(selected)),
        "prior_passing_cells": {
            source: sorted(cells) for source, cells in passing.items()
        },
        "report_sha256": {
            source: sha256_file(path) for source, (path, _) in specifications.items()
        },
    }
    return selected, audit


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--e10b-report", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e10b_positive_action_expansion/report.json",
    )
    parser.add_argument(
        "--e11-report", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e11_reference_diversity/report.json",
    )
    parser.add_argument(
        "--e12b-report", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e12b_relaxed_recurrence/report.json",
    )
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-queries", type=int, default=0)
    return parser.parse_args()


def official_rank_margin(graph: CandidateGraph) -> tuple[np.ndarray, np.ndarray]:
    score = graph.features[:, graph.dreams_column]
    molecule_score = np.maximum.reduceat(score, graph.molecule_ptr[:-1])
    rank = np.empty(graph.n_queries, dtype=np.int16)
    margin = np.empty(graph.n_queries, dtype=np.float32)
    for query in range(graph.n_queries):
        left, right = map(int, graph.query_ptr[query:query + 2])
        values = molecule_score[left:right]
        rank[query] = strict_rank(values)
        margin[query] = float(values[0] - np.max(values[1:]))
    return rank, margin


def top_rows(rows: np.ndarray, scores: np.ndarray, count: int = 3) -> np.ndarray:
    order = np.argsort(-np.asarray(scores), kind="stable")[:count]
    return np.asarray(rows[order], dtype=np.int64)


def build_variant(
    clean: torch.Tensor,
    references: list[torch.Tensor],
    definition: ActionDefinition,
    tolerance: float,
) -> torch.Tensor:
    prevalence, target = reference_profile(clean, references, tolerance)
    missing = recurrent_missing_peaks(
        clean, references, tolerance,
        definition.prevalence, definition.maximum_peaks,
    )
    return build_variant_from_evidence(
        clean, prevalence, target, missing, definition,
    )


def build_variant_from_evidence(
    clean: torch.Tensor,
    prevalence: np.ndarray,
    target: np.ndarray,
    missing: np.ndarray,
    definition: ActionDefinition,
) -> torch.Tensor:
    """Apply one action to evidence cached for the same query/reference set."""
    if definition.source == "E12B":
        return relaxed_variant(
            clean, missing, prevalence, definition.maximum_peaks,
            definition.dose, definition.support_weighted,
        )
    return cell_variant(
        clean, (prevalence, target), missing, definition.family,
        definition.dose, definition.auxiliary_dose,
    )


def rank_margin_rows(
    graph: CandidateGraph,
    query: int,
    query_vector: np.ndarray,
    embeddings: np.ndarray,
    embedding_index: dict[int, int],
) -> tuple[int, float, int, int]:
    """Return rank, margin and the exact pair of spectra defining the margin."""
    _, rows, ptr, _ = graph.query_block(query)
    candidate = embeddings[[embedding_index[int(row)] for row in rows]]
    pair_scores = candidate @ np.asarray(query_vector, dtype=np.float32)
    molecule_scores = np.maximum.reduceat(pair_scores, ptr[:-1])
    positive_left, positive_right = map(int, ptr[:2])
    positive_pair = positive_left + int(np.argmax(pair_scores[positive_left:positive_right]))
    hard_molecule = int(np.argmax(molecule_scores[1:])) + 1
    hard_left, hard_right = map(int, ptr[hard_molecule:hard_molecule + 2])
    hard_pair = hard_left + int(np.argmax(pair_scores[hard_left:hard_right]))
    rank = 1 + int(np.sum(molecule_scores[1:] >= molecule_scores[0]))
    margin = float(molecule_scores[0] - molecule_scores[hard_molecule])
    return rank, margin, int(rows[positive_pair]), int(rows[hard_pair])


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E14 crossfit teacher: {args.output_dir}")
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be in 0..4")
    held_ledger_path = args.student_checkpoint.parent / "held_per_query.csv.gz"
    decision_path = args.student_checkpoint.parent / "decision.json"
    required = [
        args.graph, args.data, args.official_checkpoint,
        args.architecture_checkpoint, args.student_checkpoint,
        decision_path,
        args.e10b_report, args.e11_report, args.e12b_report,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("formal E14 teacher construction requires CUDA")

    graph = CandidateGraph(args.graph)
    fold = np.asarray([
        stable_fold(str(formula), 5, args.formula_fold_seed)
        for formula in graph.query_formula
    ], dtype=np.int8)
    held_queries = np.flatnonzero(fold == args.outer_fold).astype(np.int64)
    queries = np.flatnonzero(fold != args.outer_fold).astype(np.int64)
    if args.max_queries:
        queries = queries[: args.max_queries]
    formal = args.max_queries == 0
    package = torch_load_compat(args.student_checkpoint, map_location="cpu")
    student_checkpoint_sha256 = sha256_file(args.student_checkpoint)
    package_fold = int(package.get("outer_fold", -1))
    if package_fold != args.outer_fold:
        raise RuntimeError(
            f"checkpoint fold {package_fold} cannot generate fold {args.outer_fold} crossfit labels"
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    configuration = decision.get("configuration", {})
    if (
        decision.get("status") != "noise_final_e4a_direct_augmentation_complete"
        or not decision.get("formal")
        or int(configuration.get("outer_fold", -1)) != args.outer_fold
        or int(configuration.get("formula_fold_seed", -1)) != args.formula_fold_seed
        or int(configuration.get("seed", -1)) != int(package.get("seed", -2))
    ):
        raise RuntimeError(
            "mature E4-A checkpoint/decision does not reproduce the requested formula fold"
        )
    # Older completed E4-A runs can lack only this derived CSV sidecar even
    # though the checkpoint and formal decision are intact.  The ledger is not
    # an input to action scoring: E14 recomputes every clean embedding and rank.
    # Validate it when available; otherwise reconstruct its query membership
    # from the frozen graph, formula-fold seed and two independent fold records
    # (checkpoint plus decision).  Never infer or substitute model weights.
    held_ledger_present = held_ledger_path.is_file()
    if held_ledger_present:
        held = pd.read_csv(held_ledger_path)
        held = held.sort_values("query_index", kind="stable").reset_index(drop=True)
        expected_queries = held["query_index"].to_numpy(np.int64)
        if formal and not np.array_equal(expected_queries, held_queries):
            raise RuntimeError(
                "checkpoint held-query ledger does not reproduce requested formula fold"
            )
    held_fold_verification = (
        "checkpoint+decision+held_ledger+graph"
        if held_ledger_present else "checkpoint+decision+graph_reconstruction"
    )

    reachable = np.unique(np.concatenate([graph.query_row, graph.pair_candidate_row])).astype(np.int64)
    store = SpectrumStore(args.data, reachable, 100)
    with h5py.File(args.data, "r") as handle:
        instrument_values = decode(handle["INSTRUMENT_TYPE"][store.rows])
        collision_values = np.asarray(handle["COLLISION_ENERGY"][store.rows], dtype=float)
    instruments = {
        int(row): clean_instrument(str(value))
        for row, value in zip(store.rows, instrument_values)
    }
    collision = {int(row): float(value) for row, value in zip(store.rows, collision_values)}
    model = load_student(args, device)
    embeddings = encode_rows(
        model, store, store.rows, device, args.batch_size, args.amp, "E14-teacher",
    )
    embedding_index = {int(row): index for index, row in enumerate(store.rows)}
    official_rank, official_margin = official_rank_margin(graph)

    definitions, prior_action_audit = load_prior_safe_definitions(args)
    policy_names = ("top3",) + REFERENCE_POLICIES
    reference_rows: list[dict[str, np.ndarray]] = []
    clean_rank = np.empty(len(queries), dtype=np.int16)
    clean_margin = np.empty(len(queries), dtype=np.float32)
    for local, query in enumerate(queries):
        _, rows, ptr, _ = graph.query_block(int(query))
        qrow = int(graph.query_row[int(query)])
        qvector = embeddings[embedding_index[qrow]]
        left, right = map(int, ptr[:2])
        positive_rows = np.asarray(rows[left:right], dtype=np.int64)
        positive_vectors = embeddings[[embedding_index[int(row)] for row in positive_rows]]
        positive_scores = positive_vectors @ qvector
        candidate_vectors = embeddings[[embedding_index[int(row)] for row in rows]]
        pair_scores = candidate_vectors @ qvector
        molecule_scores = np.maximum.reduceat(pair_scores, ptr[:-1])
        clean_rank[local] = 1 + int(np.sum(molecule_scores[1:] >= molecule_scores[0]))
        clean_margin[local] = float(molecule_scores[0] - np.max(molecule_scores[1:]))
        selected: dict[str, np.ndarray] = {"top3": top_rows(positive_rows, positive_scores, 3)}
        for policy in REFERENCE_POLICIES:
            selected[policy] = select_rows(
                positive_rows, positive_scores, positive_vectors, policy,
                instruments[qrow], collision[qrow], instruments, collision,
            )
        if set(selected) != set(policy_names):
            raise RuntimeError("reference policy coverage is incomplete")
        reference_rows.append(selected)
        if (local + 1) % 1000 == 0 or local + 1 == len(queries):
            print(f"[E14 references] {local + 1:,}/{len(queries):,}", flush=True)

    if formal:
        held_rank = np.empty(len(held_queries), dtype=np.int16)
        for local, query in enumerate(held_queries):
            query_row = int(graph.query_row[int(query)])
            held_rank[local] = rank_margin(
                graph, int(query), embeddings[embedding_index[query_row]],
                embeddings, embedding_index,
            )[0]
        if held_ledger_present:
            expected_rank = held["final_rank"].to_numpy(np.int16)
            mismatches = int(np.sum(held_rank != expected_rank))
            if mismatches:
                raise RuntimeError(
                    f"E14 failed to reproduce {mismatches} mature checkpoint ranks"
                )
        else:
            # The exact per-query ledger is derived output.  When it is absent,
            # reproduce the independently persisted decision totals from the
            # checkpoint itself.  Counts are exact, so this remains fail-closed
            # without inventing any per-query label.
            held_summary = decision.get("held_clean", {})
            held_official_rank = official_rank[held_queries]
            reproduced = {
                "n_queries": int(len(held_rank)),
                "errors": int(np.sum(held_rank != 1)),
                "corrected": int(np.sum((held_official_rank != 1) & (held_rank == 1))),
                "introduced": int(np.sum((held_official_rank == 1) & (held_rank != 1))),
            }
            expected = {
                key: int(held_summary.get(key, -1)) for key in reproduced
            }
            if reproduced != expected:
                raise RuntimeError(
                    "E14 reconstructed held-fold ranks disagree with mature decision: "
                    f"reproduced={reproduced} expected={expected}"
                )
            reproduced_recall = float(np.mean(held_rank == 1))
            expected_recall = float(held_summary.get("recall1", float("nan")))
            if not np.isfinite(expected_recall) or not np.isclose(
                reproduced_recall, expected_recall, rtol=0.0, atol=1e-12,
            ):
                raise RuntimeError(
                    "E14 reconstructed held-fold recall disagrees with mature decision"
                )

    result_shape = (len(queries), len(definitions))
    result_rank = np.empty(result_shape, dtype=np.int16)
    result_margin = np.empty(result_shape, dtype=np.float32)
    result_positive_row = np.empty(result_shape, dtype=np.int64)
    result_hard_negative_row = np.empty(result_shape, dtype=np.int64)
    total = len(queries) * len(definitions)
    definition_ids = np.asarray([item.action_id for item in definitions], dtype=str)
    scan_script_sha256 = sha256_file(Path(__file__))
    partial_path = args.output_dir.parent / f".{args.output_dir.name}.action_scan_partial.npz"
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    scan_start = 0
    if partial_path.is_file():
        with np.load(partial_path, allow_pickle=False) as partial:
            if (
                str(partial["student_checkpoint_sha256"].item())
                != student_checkpoint_sha256
                or str(partial["script_sha256"].item()) != scan_script_sha256
                or not np.array_equal(partial["queries"], queries)
                or not np.array_equal(partial["action_ids"].astype(str), definition_ids)
            ):
                raise RuntimeError(f"stale E14 action-scan checkpoint: {partial_path}")
            scan_start = int(partial["completed"].item())
            if (
                scan_start < 0 or scan_start > total
                or (scan_start != total and scan_start % args.batch_size)
            ):
                raise RuntimeError("invalid E14 action-scan checkpoint boundary")
            for destination, key in (
                (result_rank, "result_rank"),
                (result_margin, "result_margin"),
                (result_positive_row, "result_positive_row"),
                (result_hard_negative_row, "result_hard_negative_row"),
            ):
                source = partial[key]
                if source.shape != result_shape:
                    raise RuntimeError("E14 action-scan checkpoint shape drifted")
                destination[...] = source
        print(
            f"[E14 actions] resuming at {scan_start:,}/{total:,} from {partial_path}",
            flush=True,
        )
    scan_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(scan_start, total, args.batch_size):
            stop = min(start + args.batch_size, total)
            linear = np.arange(start, stop, dtype=np.int64)
            local_queries = linear // len(definitions)
            local_actions = linear % len(definitions)
            # The original implementation recomputed the same reference peak
            # matching and recurrent-peak clustering once per action.  Those
            # operations depend only on query, reference policy, prevalence and
            # maximum peak count, not on dose/family.  Cache them inside each
            # batch; a query can straddle at most two batches, so this removes
            # nearly all redundant CPU work without changing any action.
            clean_cache: dict[int, torch.Tensor] = {}
            profile_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
            missing_cache: dict[tuple[int, str, float, int], np.ndarray] = {}
            for local_query in np.unique(local_queries):
                local_query_int = int(local_query)
                query = int(queries[local_query_int])
                clean = store.one(int(graph.query_row[query]))
                clean_cache[local_query_int] = clean
                action_indices = np.unique(local_actions[local_queries == local_query])
                for local_action in action_indices:
                    definition = definitions[int(local_action)]
                    profile_key = (local_query_int, definition.reference_policy)
                    rows = reference_rows[local_query_int][definition.reference_policy]
                    references = [store.one(int(row)) for row in rows]
                    if profile_key not in profile_cache:
                        profile_cache[profile_key] = reference_profile(
                            clean, references, args.fragment_tolerance,
                        )
                    missing_key = (
                        local_query_int, definition.reference_policy,
                        definition.prevalence, definition.maximum_peaks,
                    )
                    if missing_key not in missing_cache:
                        missing_cache[missing_key] = recurrent_missing_peaks(
                            clean, references, args.fragment_tolerance,
                            definition.prevalence, definition.maximum_peaks,
                        )
            variants: list[torch.Tensor] = []
            for local_query, local_action in zip(local_queries, local_actions):
                local_query_int = int(local_query)
                definition = definitions[int(local_action)]
                prevalence, target = profile_cache[
                    (local_query_int, definition.reference_policy)
                ]
                missing = missing_cache[(
                    local_query_int, definition.reference_policy,
                    definition.prevalence, definition.maximum_peaks,
                )]
                variants.append(build_variant_from_evidence(
                    clean_cache[local_query_int], prevalence, target, missing,
                    definition,
                ))
            vectors = forward_embeddings(
                model, torch.stack(variants).to(device), args.amp,
            ).float().cpu().numpy()
            for vector, local_query, local_action in zip(
                vectors, local_queries, local_actions,
            ):
                query = int(queries[int(local_query)])
                rank, margin, positive_row, hard_negative_row = rank_margin_rows(
                    graph, query, vector, embeddings, embedding_index,
                )
                result_rank[int(local_query), int(local_action)] = rank
                result_margin[int(local_query), int(local_action)] = margin
                result_positive_row[int(local_query), int(local_action)] = positive_row
                result_hard_negative_row[int(local_query), int(local_action)] = hard_negative_row
            should_log = stop % 5000 < args.batch_size or stop == total
            should_checkpoint = stop % 25000 < args.batch_size or stop == total
            if should_checkpoint:
                temporary_partial = Path(str(partial_path) + ".tmp.npz")
                np.savez_compressed(
                    temporary_partial,
                    completed=np.asarray(stop, dtype=np.int64),
                    student_checkpoint_sha256=np.asarray(student_checkpoint_sha256),
                    script_sha256=np.asarray(scan_script_sha256),
                    queries=queries,
                    action_ids=definition_ids,
                    result_rank=result_rank,
                    result_margin=result_margin,
                    result_positive_row=result_positive_row,
                    result_hard_negative_row=result_hard_negative_row,
                )
                temporary_partial.replace(partial_path)
            if should_log:
                elapsed = max(time.perf_counter() - scan_started, 1e-9)
                completed_this_run = stop - scan_start
                print(
                    f"[E14 actions] {stop:,}/{total:,}; "
                    f"{completed_this_run / elapsed:,.1f} actions/s; "
                    f"{elapsed / 60:.1f} min this run",
                    flush=True,
                )

    # Second gate: a cell that was safe on the original mature-E8 audit must
    # also replicate in the current outer-train geometry.  This gate is fitted
    # only on student-training formulas; the outer held fold remains untouched.
    clean_correct = clean_rank == 1
    action_safety_records: list[dict] = []
    eligible_action = np.zeros(len(definitions), dtype=bool)
    inner_folds = sorted(set(map(int, fold[queries])))
    for action_index, definition in enumerate(definitions):
        action_correct = result_rank[:, action_index] == 1
        corrected = int(np.sum(~clean_correct & action_correct))
        introduced = int(np.sum(clean_correct & ~action_correct))
        fold_risk: dict[str, int] = {}
        for inner_fold in inner_folds:
            mask = fold[queries] == inner_fold
            local_corrected = int(np.sum(mask & ~clean_correct & action_correct))
            local_introduced = int(np.sum(mask & clean_correct & ~action_correct))
            fold_risk[str(inner_fold)] = local_corrected - 2 * local_introduced
        replicated = bool(
            corrected >= 10
            and corrected > 2 * introduced
            and all(value >= 0 for value in fold_risk.values())
            and sum(value > 0 for value in fold_risk.values()) >= 2
        )
        eligible_action[action_index] = replicated
        action_safety_records.append({
            "action_id": definition.action_id,
            "source": definition.source,
            "prior_cell_id": prior_cell_id(definition),
            "corrected": corrected,
            "introduced": introduced,
            "risk_net_lambda2": corrected - 2 * introduced,
            "inner_fold_risk_net_lambda2": json.dumps(fold_risk, sort_keys=True),
            "replicated_safe": replicated,
        })
    if not np.any(eligible_action):
        raise RuntimeError("E14 has no action that replicates across outer-train formula folds")

    # Materialise action-specific hard safety controls from the same outer-train
    # geometry.  All action-introduced errors are kept.  For each action we also
    # retain the boundary-hardest protected queries, matched by action id.  These
    # controls teach the shared encoder where the same perturbation must *not*
    # move a clean-correct query across its local decision boundary.
    risk_records: list[dict] = []
    for action_index, definition in enumerate(definitions):
        if not eligible_action[action_index]:
            continue
        introduced_local = np.flatnonzero(
            clean_correct & (result_rank[:, action_index] != 1)
        )
        protected_local = np.flatnonzero(
            clean_correct & (result_rank[:, action_index] == 1)
        )
        protected_count = min(
            len(protected_local), max(25, min(200, 2 * len(introduced_local)))
        )
        if protected_count:
            protected_order = protected_local[
                np.argsort(clean_margin[protected_local], kind="stable")[:protected_count]
            ]
        else:
            protected_order = np.asarray([], dtype=np.int64)
        for control_kind, local_indices in (
            ("introduced", introduced_local),
            ("protected_boundary", protected_order),
        ):
            for local in local_indices:
                query = int(queries[int(local)])
                refs = reference_rows[int(local)][definition.reference_policy]
                positive_row = int(result_positive_row[int(local), action_index])
                hard_negative_row = int(
                    result_hard_negative_row[int(local), action_index]
                )
                query_vector = embeddings[
                    embedding_index[int(graph.query_row[query])]
                ]
                pair_clean_margin = float(
                    np.dot(query_vector, embeddings[embedding_index[positive_row]])
                    - np.dot(
                        query_vector,
                        embeddings[embedding_index[hard_negative_row]],
                    )
                )
                risk_records.append({
                    "query_index": query,
                    "query_row": int(graph.query_row[query]),
                    "query_ik14": str(graph.query_ik14[query]),
                    "query_formula": str(graph.query_formula[query]),
                    "formula_fold": int(fold[query]),
                    "official_rank": int(official_rank[query]),
                    "official_margin": float(official_margin[query]),
                    "crossfit_clean_rank": int(clean_rank[int(local)]),
                    "crossfit_clean_margin": float(clean_margin[int(local)]),
                    "teacher_rank": int(result_rank[int(local), action_index]),
                    "teacher_margin": float("nan"),
                    "teacher_margin_delta": float("nan"),
                    "teacher_positive_row": positive_row,
                    "teacher_hard_negative_row": hard_negative_row,
                    "teacher_pair_clean_margin": pair_clean_margin,
                    "teacher_checkpoint_sha256": student_checkpoint_sha256,
                    "action_id": definition.action_id,
                    "action_source": definition.source,
                    "reference_policy": definition.reference_policy,
                    "positive_reference_rows": ";".join(map(str, refs)),
                    "guided_family": definition.family,
                    "guided_dose": float(definition.dose),
                    "guided_auxiliary_dose": float(definition.auxiliary_dose),
                    "guided_recurrence_prevalence": float(definition.prevalence),
                    "guided_recurrence_max_peaks": int(definition.maximum_peaks),
                    "guided_support_weighted": bool(definition.support_weighted),
                    "control_kind": control_kind,
                })
    risk_controls = pd.DataFrame(risk_records)
    if risk_controls.empty:
        raise RuntimeError("E14 replicated-safe actions produced no risk controls")
    if risk_controls[["query_index", "action_id"]].duplicated().any():
        raise RuntimeError("E14 risk controls repeat a query/action pair")
    if not np.all(risk_controls["crossfit_clean_rank"].to_numpy(int) == 1):
        raise RuntimeError("E14 risk controls must start mature-clean correct")

    selected_records: list[dict] = []
    per_action_counts = {definition.action_id: 0 for definition in definitions}
    for local, query in enumerate(queries):
        # The teacher may mine outcomes only on the student's training formulas.
        # The checkpoint and every selected row exclude the student outer fold.
        if official_rank[int(query)] == 1 or clean_rank[local] == 1:
            continue
        correcting = np.flatnonzero((result_rank[local] == 1) & eligible_action)
        if not len(correcting):
            continue
        best = int(correcting[np.argmax(result_margin[local, correcting])])
        definition = definitions[best]
        refs = reference_rows[local][definition.reference_policy]
        positive_row = int(result_positive_row[local, best])
        hard_negative_row = int(result_hard_negative_row[local, best])
        query_vector = embeddings[embedding_index[int(graph.query_row[int(query)])]]
        pair_clean_margin = float(
            np.dot(query_vector, embeddings[embedding_index[positive_row]])
            - np.dot(query_vector, embeddings[embedding_index[hard_negative_row]])
        )
        per_action_counts[definition.action_id] += 1
        selected_records.append({
            "query_index": int(query),
            "query_row": int(graph.query_row[int(query)]),
            "query_ik14": str(graph.query_ik14[int(query)]),
            "query_formula": str(graph.query_formula[int(query)]),
            # This is the selected query's actual formula fold, not the outer
            # fold excluded from the teacher/student training partition.  The
            # trainer independently recomputes this value and fails closed.
            "formula_fold": int(fold[int(query)]),
            "official_rank": int(official_rank[int(query)]),
            "official_margin": float(official_margin[int(query)]),
            "crossfit_clean_rank": int(clean_rank[local]),
            "crossfit_clean_margin": float(clean_margin[local]),
            "teacher_rank": int(result_rank[local, best]),
            "teacher_margin": float(result_margin[local, best]),
            "teacher_margin_delta": float(result_margin[local, best] - clean_margin[local]),
            "teacher_positive_row": positive_row,
            "teacher_hard_negative_row": hard_negative_row,
            "teacher_pair_clean_margin": pair_clean_margin,
            "teacher_checkpoint_sha256": student_checkpoint_sha256,
            "action_id": definition.action_id,
            "action_source": definition.source,
            "reference_policy": definition.reference_policy,
            "positive_reference_rows": ";".join(map(str, refs)),
            "guided_family": definition.family,
            "guided_dose": float(definition.dose),
            "guided_auxiliary_dose": float(definition.auxiliary_dose),
            "guided_recurrence_prevalence": float(definition.prevalence),
            "guided_recurrence_max_peaks": int(definition.maximum_peaks),
            "guided_support_weighted": bool(definition.support_weighted),
        })
    selected = pd.DataFrame(selected_records)
    if selected.empty:
        raise RuntimeError("E14 crossfit teacher found no corrective P actions")
    if selected["query_index"].duplicated().any():
        raise RuntimeError("E14 teacher must contain at most one action per query")
    if not np.all(selected["teacher_rank"].to_numpy(int) == 1):
        raise RuntimeError("E14 selected a non-correcting teacher action")
    selected_formulas = set(selected["query_formula"].astype(str))
    held_formulas = set(map(str, graph.query_formula[fold == args.outer_fold]))
    if selected_formulas & held_formulas:
        raise RuntimeError("held formula leaked into E14 teacher actions")

    official_error_count = int(np.sum(official_rank[queries] != 1))
    recovered = int(len(selected))
    gates = {
        "selected_corrective_queries_ge_500": bool(recovered >= 500),
        "selected_corrective_identities_ge_250": bool(
            selected["query_ik14"].nunique() >= 250
        ),
        "selected_corrective_formulas_ge_200": bool(
            selected["query_formula"].nunique() >= 200
        ),
        "teacher_incremental_headroom_ge_2pp": bool(recovered / len(queries) >= 0.02),
    }
    report = {
        "status": "noise_final_e14_crossfit_p_teacher_complete",
        "formal": formal,
        "outer_formula_fold": int(args.outer_fold),
        "query_scope": "student-training formulas only; outer formula fold excluded",
        "queries": int(len(queries)),
        "identities": int(len(set(map(str, graph.query_ik14[queries])))),
        "formulas": int(len(set(map(str, graph.query_formula[queries])))),
        "official_errors": official_error_count,
        "mature_crossfit_errors": int(np.sum(clean_rank != 1)),
        "selected_corrective_queries": recovered,
        "selected_corrective_identities": int(selected["query_ik14"].nunique()),
        "selected_corrective_formulas": int(selected["query_formula"].nunique()),
        "selected_incremental_headroom_beyond_mature_crossfit": float(recovered / len(queries)),
        "fraction_official_errors_action_recoverable": float(
            recovered / max(official_error_count, 1)
        ),
        "action_definitions": int(len(definitions)),
        "selected_action_counts": {
            key: int(value) for key, value in per_action_counts.items() if value
        },
        "prior_action_filter": prior_action_audit,
        "outer_train_replicated_safe_actions": int(np.sum(eligible_action)),
        "risk_controls": int(len(risk_controls)),
        "risk_control_identities": int(risk_controls["query_ik14"].nunique()),
        "risk_control_formulas": int(risk_controls["query_formula"].nunique()),
        "risk_control_kind_counts": {
            str(key): int(value)
            for key, value in risk_controls["control_kind"].value_counts().items()
        },
        "gates": gates,
        "pass_to_shared_encoder_transfer": bool(all(gates.values())),
        "contracts": {
            "teacher_checkpoint_excludes_student_outer_formula_fold": True,
            "mature_checkpoint_and_decision_fold_verified": True,
            "held_fold_membership_reconstructed_fail_closed": True,
            "all_selected_queries_exclude_student_outer_formula_fold": True,
            "selected_query_formula_fold_is_materialized": True,
            "prior_fixed_cell_safety_filter_applied": True,
            "outer_train_multifold_action_safety_filter_applied": True,
            "action_specific_risk_controls_materialized": True,
            "one_selected_action_per_query": True,
            "only_official_errors_selected": True,
            "selected_action_rank_is_one": True,
            "identity_labels_training_only": True,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "hdf5_sha256": sha256_file(args.data),
            "student_checkpoint_sha256": student_checkpoint_sha256,
            "student_decision_sha256": sha256_file(decision_path),
            "student_held_ledger_sha256": (
                sha256_file(held_ledger_path) if held_ledger_present else None
            ),
            "held_fold_verification": held_fold_verification,
            "e10b_report_sha256": sha256_file(args.e10b_report),
            "e11_report_sha256": sha256_file(args.e11_report),
            "e12b_report_sha256": sha256_file(args.e12b_report),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Outer-fold-isolated privileged action teacher. Its action labels are "
            "mined on the student-training partition, so they are not per-query OOF "
            "predictions. This is supervision headroom, "
            "not a trained shared-embedding gain."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected.to_csv(
        args.output_dir / "selected_actions.csv.gz", index=False, compression="gzip",
    )
    pd.DataFrame(action_safety_records).to_csv(
        args.output_dir / "action_safety.csv.gz", index=False, compression="gzip",
    )
    risk_controls.to_csv(
        args.output_dir / "risk_controls.csv.gz", index=False, compression="gzip",
    )
    np.savez_compressed(
        args.output_dir / "action_outcomes.npz",
        queries=queries,
        action_ids=np.asarray([item.action_id for item in definitions], dtype=object),
        clean_rank=clean_rank,
        clean_margin=clean_margin,
        result_rank=result_rank,
        result_margin=result_margin,
        result_positive_row=result_positive_row,
        result_hard_negative_row=result_hard_negative_row,
    )
    json_dump(args.output_dir / "report.json", report)
    partial_path.unlink(missing_ok=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
