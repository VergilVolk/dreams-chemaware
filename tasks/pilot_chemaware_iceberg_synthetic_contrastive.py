"""Formula-held-out shared-embedding pilot with ICEBERG synthetic hard negatives.

Every arm retains the same real-query/real-reference identity anchor.  The
auxiliary view is either the same real references (identity-only), correctly
aligned ICEBERG synthetic candidate spectra, candidate-swapped synthetic
spectra, or peak-permuted synthetic spectra.  Query and all reference views use
one shared candidate-independent adapter.  Evaluation uses real spectra only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_chemaware_iceberg_synthetic_embedding import (  # noqa: E402
    peak_permute,
    synthetic_tensor,
)
from chemaware_shared_v2_core import formula_folds, paired_evaluation  # noqa: E402
from dreams.models.chem_aware.hierarchical_chemical_adapter import (  # noqa: E402
    HierarchicalChemicalResidualAdapter,
    deployable_parameter_count,
)
from noise_final_core import CandidateGraph  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path,
        default=Path("data/validation/chemaware_shared_v2_cached_real_diagnostic/graph.npz"),
    )
    parser.add_argument(
        "--token-dir", type=Path,
        default=Path("data/validation/chemaware_shared_v2_cached_real_diagnostic/tokens"),
    )
    parser.add_argument(
        "--teacher-dir", type=Path,
        default=Path("data/validation/chemaware_iceberg_teacher_ledger_inner_v1"),
    )
    parser.add_argument(
        "--synthetic-dir", type=Path,
        default=Path("data/validation/chemaware_iceberg_synthetic_embedding_150_inner_v1"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/validation/chemaware_iceberg_synthetic_contrastive_inner_v1"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-queries", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--synthetic-alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--lambda-protect", type=float, default=0.5)
    parser.add_argument("--lambda-preserve", type=float, default=0.10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--fold-seed", type=int, default=20260904)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SpectrumStore:
    def __init__(self, directory: Path):
        self.rows = np.load(directory / "rows.npy").astype(np.int64)
        self.position = {int(row): index for index, row in enumerate(self.rows)}
        self.official = np.load(directory / "official_embeddings_f32.npy").astype(np.float32)
        self.mz = np.load(directory / "mz_f32.npy").astype(np.float32)
        self.intensity = np.load(directory / "intensity_f32.npy").astype(np.float32)
        self.valid = np.load(directory / "valid.npy").astype(bool)
        self.precursor = np.load(directory / "precursor_mz_f32.npy").astype(np.float32)
        if not np.allclose(np.linalg.norm(self.official, axis=1), 1.0, atol=2e-4):
            raise RuntimeError("real official embeddings are not normalized")

    def take(self, rows: np.ndarray) -> tuple[torch.Tensor, ...]:
        try:
            index = np.asarray([self.position[int(row)] for row in rows], dtype=np.int64)
        except KeyError as error:
            raise RuntimeError(f"real spectrum absent from cache: {error}") from error
        return self.take_positions(index)

    def take_positions(self, index: np.ndarray) -> tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(self.official[index]),
            torch.from_numpy(self.mz[index]),
            torch.from_numpy(self.intensity[index]),
            torch.from_numpy(self.precursor[index]),
            torch.from_numpy(self.valid[index]),
        )


class SyntheticStore:
    def __init__(self, teacher_dir: Path, synthetic_dir: Path):
        self.query_ptr = np.load(teacher_dir / "query_ptr.npy").astype(np.int64)
        prediction = np.load(teacher_dir / "iceberg_predictions_f16.npy").astype(np.float32)
        teacher_report = json.loads((teacher_dir / "report.json").read_text(encoding="utf-8"))
        correct_embedding = np.load(
            synthetic_dir / "correct_synthetic_embeddings_f32.npy"
        ).astype(np.float32)
        permuted_embedding = np.load(
            synthetic_dir / "peak_permuted_synthetic_embeddings_f32.npy"
        ).astype(np.float32)
        if len(prediction) != len(correct_embedding) or len(prediction) != len(permuted_embedding):
            raise RuntimeError("synthetic spectra and embeddings are not aligned")
        query_precursor = np.load(teacher_dir / "true_binned_f16.npy")  # shape check only below
        if len(query_precursor) != len(self.query_ptr) - 1:
            raise RuntimeError("teacher query arrays are not aligned")
        # Precursor values are repeated from the real token cache later via set_precursors.
        self.prediction = prediction
        self.permuted_prediction = peak_permute(
            prediction, int(teacher_report["protocol"]["seed"]) + 41
        )
        self.embedding = {
            "correct_synthetic": correct_embedding,
            "candidate_swapped": correct_embedding,
            "peak_permuted": permuted_embedding,
        }
        self._arrays: dict[str, dict[str, np.ndarray]] = {}

    def build_arrays(self, flat_precursor: np.ndarray, max_mz: float = 1000.0) -> None:
        if len(flat_precursor) != len(self.prediction):
            raise RuntimeError("synthetic precursor array is misaligned")
        for arm, prediction in (
            ("correct_synthetic", self.prediction),
            ("peak_permuted", self.permuted_prediction),
        ):
            tensors = [
                synthetic_tensor(value, float(flat_precursor[index]), 100, max_mz).numpy()
                for index, value in enumerate(prediction)
            ]
            stacked = np.asarray(tensors, dtype=np.float32)
            self._arrays[arm] = {
                "official": self.embedding[arm],
                "mz": stacked[:, 1:, 0],
                "intensity": stacked[:, 1:, 1],
                "precursor": stacked[:, 0, 0],
                "valid": stacked[:, 1:, 0] > 0,
            }
        self._arrays["candidate_swapped"] = self._arrays["correct_synthetic"]

    def take(self, arm: str, index: np.ndarray) -> tuple[torch.Tensor, ...]:
        values = self._arrays[arm]
        return (
            torch.from_numpy(values["official"][index]),
            torch.from_numpy(values["mz"][index]),
            torch.from_numpy(values["intensity"][index]),
            torch.from_numpy(values["precursor"][index]),
            torch.from_numpy(values["valid"][index]),
        )


def adapt(model: HierarchicalChemicalResidualAdapter, tensors: tuple[torch.Tensor, ...]):
    return model(*tensors).embedding


def representative_rows(graph: CandidateGraph, query: int) -> np.ndarray:
    left, right = graph.query_ptr[query : query + 2]
    return np.asarray(
        [graph.pair_candidate_row[graph.molecule_ptr[molecule]] for molecule in range(left, right)],
        dtype=np.int64,
    )


def loss_for_batch(
    model: HierarchicalChemicalResidualAdapter,
    real: SpectrumStore,
    synthetic: SyntheticStore,
    graph: CandidateGraph,
    ledger_query: np.ndarray,
    ledger_ptr: np.ndarray,
    ledger_positions: np.ndarray,
    arm: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    real_rows = []
    payload = []
    synthetic_indices = []
    for position in ledger_positions:
        query = int(ledger_query[position])
        candidate_rows = representative_rows(graph, query)
        offset = len(real_rows)
        real_rows.append(int(graph.query_row[query]))
        real_rows.extend(map(int, candidate_rows))
        left, right = ledger_ptr[position : position + 2]
        index = np.arange(left, right, dtype=np.int64)
        if arm == "candidate_swapped":
            index = np.roll(index, 1)
        synthetic_offset = len(synthetic_indices)
        synthetic_indices.extend(index.tolist())
        payload.append((offset, len(candidate_rows), synthetic_offset))

    unique_rows, inverse = np.unique(np.asarray(real_rows, dtype=np.int64), return_inverse=True)
    real_unique_tensors = real.take(unique_rows)
    adapted_unique = adapt(model, real_unique_tensors)
    official_unique = real_unique_tensors[0]
    inverse_tensor = torch.from_numpy(inverse).long()
    real_adapted = adapted_unique[inverse_tensor]
    real_official = official_unique[inverse_tensor]
    if arm != "identity_only":
        synthetic_adapted = adapt(
            model, synthetic.take(arm, np.asarray(synthetic_indices, dtype=np.int64))
        )
    losses = []
    real_losses = []
    auxiliary_losses = []
    protect_losses = []
    for offset, count, synthetic_offset in payload:
        query_embedding = real_adapted[offset]
        real_scores = real_adapted[offset + 1 : offset + 1 + count] @ query_embedding
        real_ce = -F.log_softmax(real_scores / args.temperature, dim=0)[0]
        if arm == "identity_only":
            auxiliary_ce = real_ce
        else:
            auxiliary_scores = (
                synthetic_adapted[synthetic_offset : synthetic_offset + count] @ query_embedding
            )
            auxiliary_ce = -F.log_softmax(auxiliary_scores / args.temperature, dim=0)[0]
        old_scores = real_official[offset + 1 : offset + 1 + count] @ real_official[offset]
        old_margin = old_scores[0] - torch.max(old_scores[1:])
        new_margin = real_scores[0] - torch.max(real_scores[1:])
        protect = torch.relu(torch.clamp(old_margin.detach(), min=0.0, max=0.05) - new_margin)
        loss = (
            (1.0 - args.synthetic_alpha) * real_ce
            + args.synthetic_alpha * auxiliary_ce
            + args.lambda_protect * protect
        )
        losses.append(loss)
        real_losses.append(real_ce)
        auxiliary_losses.append(auxiliary_ce)
        protect_losses.append(protect)
    preserve = torch.mean(1.0 - torch.sum(adapted_unique * official_unique, dim=1))
    total = torch.stack(losses).mean() + args.lambda_preserve * preserve
    return total, {
        "real_ce": float(torch.stack(real_losses).mean().detach()),
        "auxiliary_ce": float(torch.stack(auxiliary_losses).mean().detach()),
        "protect": float(torch.stack(protect_losses).mean().detach()),
        "preserve": float(preserve.detach()),
    }


@torch.no_grad()
def encode_all(
    model: HierarchicalChemicalResidualAdapter, real: SpectrumStore, batch_size: int
) -> np.ndarray:
    model.eval()
    output = np.empty_like(real.official)
    for left in range(0, len(real.rows), batch_size):
        right = min(left + batch_size, len(real.rows))
        output[left:right] = adapt(
            model, real.take_positions(np.arange(left, right, dtype=np.int64))
        ).numpy()
    return output


def run_arm(
    arm: str,
    initial_state: dict[str, torch.Tensor],
    real: SpectrumStore,
    synthetic: SyntheticStore,
    graph: CandidateGraph,
    ledger_query: np.ndarray,
    ledger_ptr: np.ndarray,
    train_positions: np.ndarray,
    inner_selected: np.ndarray,
    inner_all: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, np.ndarray], dict[str, torch.Tensor]]:
    set_seed(args.seed)
    model = HierarchicalChemicalResidualAdapter(dropout=0.0, use_formula_moments=True)
    model.load_state_dict(copy.deepcopy(initial_state))
    initial = encode_all(model, real, args.eval_batch_size)
    if np.max(np.abs(initial - real.official)) != 0:
        raise RuntimeError("adapter initialization is not exactly official")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed + 577)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_positions)
        totals = {"loss": 0.0, "real_ce": 0.0, "auxiliary_ce": 0.0, "protect": 0.0, "preserve": 0.0}
        batches = 0
        for left in range(0, len(order), args.batch_queries):
            positions = order[left : left + args.batch_queries]
            optimizer.zero_grad(set_to_none=True)
            loss, components = loss_for_batch(
                model, real, synthetic, graph, ledger_query, ledger_ptr, positions, arm, args
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for key, value in components.items():
                totals[key] += value
            batches += 1
        encoded = encode_all(model, real, args.eval_batch_size)
        selected_eval = paired_evaluation(encoded, real.official, real, graph, inner_selected)
        all_eval = paired_evaluation(encoded, real.official, real, graph, inner_all)
        history.append(
            {
                "epoch": epoch,
                "train": {key: value / batches for key, value in totals.items()},
                "inner_selected": selected_eval["summary"],
                "inner_all_graph": all_eval["summary"],
            }
        )
        print(
            f"arm={arm} epoch={epoch}/{args.epochs} "
            f"selected={selected_eval['summary']['recall1']:.4f} "
            f"all={all_eval['summary']['recall1']:.4f}",
            flush=True,
        )
    encoded = encode_all(model, real, args.eval_batch_size)
    selected_eval = paired_evaluation(encoded, real.official, real, graph, inner_selected)
    all_eval = paired_evaluation(encoded, real.official, real, graph, inner_all)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return (
        {
            "arm": arm,
            "final": {
                "inner_selected": selected_eval["summary"],
                "inner_all_graph": all_eval["summary"],
            },
            "history": history,
        },
        {
            "selected_rank": selected_eval["new_rank"],
            "selected_margin": selected_eval["new_margin"],
            "all_rank": all_eval["new_rank"],
            "all_margin": all_eval["new_margin"],
        },
        state,
    )


def bootstrap(
    difference: np.ndarray, formulas: np.ndarray, seed: int, draws: int
) -> dict:
    unique = np.unique(formulas)
    values = np.asarray(
        [np.mean(difference[formulas == formula]) for formula in unique], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for draw in range(draws):
        estimates[draw] = np.mean(values[rng.integers(0, len(values), len(values))])
    return {
        "formula_macro_advantage": float(np.mean(values)),
        "formula_cluster_bootstrap_95ci": [
            float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))
        ],
        "formula_clusters": int(len(unique)),
        "draws": int(draws),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.synthetic_alpha <= 1:
        raise ValueError("synthetic-alpha must be in [0, 1]")
    if args.inner_fold == args.outer_fold:
        raise ValueError("inner and outer folds must differ")
    torch.set_num_threads(args.torch_threads)
    args.output.mkdir(parents=True, exist_ok=True)
    graph = CandidateGraph(args.graph)
    real = SpectrumStore(args.token_dir)
    selected = np.load(args.teacher_dir / "selected_queries.npy").astype(np.int64)
    ledger_ptr = np.load(args.teacher_dir / "query_ptr.npy").astype(np.int64)
    if ledger_ptr.shape != (len(selected) + 1,):
        raise RuntimeError("teacher query pointer shape mismatch")
    synthetic = SyntheticStore(args.teacher_dir, args.synthetic_dir)
    query_precursor = real.precursor[[real.position[int(graph.query_row[q])] for q in selected]]
    synthetic.build_arrays(np.repeat(query_precursor, np.diff(ledger_ptr)))

    selected_folds = formula_folds(graph.query_formula[selected], args.folds, args.fold_seed)
    train_positions = np.flatnonzero(
        (selected_folds != args.inner_fold) & (selected_folds != args.outer_fold)
    )
    inner_positions = np.flatnonzero(selected_folds == args.inner_fold)
    outer_positions = np.flatnonzero(selected_folds == args.outer_fold)
    inner_selected = selected[inner_positions]
    all_folds = formula_folds(graph.query_formula, args.folds, args.fold_seed)
    inner_all = np.flatnonzero(all_folds == args.inner_fold)
    train_formula = set(graph.query_formula[selected[train_positions]])
    inner_formula = set(graph.query_formula[inner_selected])
    outer_formula = set(graph.query_formula[selected[outer_positions]])
    if train_formula & inner_formula or train_formula & outer_formula or inner_formula & outer_formula:
        raise RuntimeError("formula leakage")

    set_seed(args.seed)
    template = HierarchicalChemicalResidualAdapter(dropout=0.0, use_formula_moments=True)
    initial_state = {key: value.detach().cpu() for key, value in template.state_dict().items()}
    arms = ["identity_only", "correct_synthetic", "candidate_swapped", "peak_permuted"]
    runs = []
    arrays = {}
    for arm in arms:
        summary, evaluation, state = run_arm(
            arm,
            initial_state,
            real,
            synthetic,
            graph,
            selected,
            ledger_ptr,
            train_positions,
            inner_selected,
            inner_all,
            args,
        )
        torch.save(state, args.output / f"{arm}.pt")
        np.savez_compressed(args.output / f"{arm}_evaluation.npz", **evaluation)
        runs.append(summary)
        arrays[arm] = evaluation
    comparisons = {}
    selected_formulas = graph.query_formula[inner_selected]
    all_formulas = graph.query_formula[inner_all]
    correct = arrays["correct_synthetic"]
    for control in ("identity_only", "candidate_swapped", "peak_permuted"):
        value = arrays[control]
        comparisons[f"correct_vs_{control}"] = {
            "inner_selected_hit1": bootstrap(
                (correct["selected_rank"] == 1).astype(float)
                - (value["selected_rank"] == 1).astype(float),
                selected_formulas, args.seed + 101, args.bootstrap_draws,
            ),
            "inner_selected_margin": bootstrap(
                correct["selected_margin"] - value["selected_margin"],
                selected_formulas, args.seed + 103, args.bootstrap_draws,
            ),
            "inner_all_graph_hit1": bootstrap(
                (correct["all_rank"] == 1).astype(float)
                - (value["all_rank"] == 1).astype(float),
                all_formulas, args.seed + 107, args.bootstrap_draws,
            ),
            "inner_all_graph_margin": bootstrap(
                correct["all_margin"] - value["all_margin"],
                all_formulas, args.seed + 109, args.bootstrap_draws,
            ),
        }
    required = []
    for control in ("identity_only", "candidate_swapped", "peak_permuted"):
        required.extend([
            comparisons[f"correct_vs_{control}"]["inner_selected_hit1"]["formula_cluster_bootstrap_95ci"][0] > 0,
            comparisons[f"correct_vs_{control}"]["inner_selected_margin"]["formula_cluster_bootstrap_95ci"][0] > 0,
        ])
    passed = bool(all(required))
    report = {
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "Correct synthetic-spectrum contrast transferred to real-only shared retrieval."
            if passed
            else "Synthetic contrast did not transfer beyond identity and chemistry controls."
        ),
        "scope": {
            "non_formal": True,
            "massspecgym_overlap_warning": True,
            "teacher_and_synthetic_spectra_training_only": True,
            "evaluation_real_spectra_only": True,
            "outer_fold_evaluated": False,
            "deployment_candidate_inputs": False,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "teacher_report_sha256": sha256_file(args.teacher_dir / "report.json"),
            "teacher_predictions_sha256": sha256_file(args.teacher_dir / "iceberg_predictions_f16.npy"),
            "synthetic_report_sha256": sha256_file(args.synthetic_dir / "report.json"),
            "correct_synthetic_embedding_sha256": sha256_file(args.synthetic_dir / "correct_synthetic_embeddings_f32.npy"),
            "peak_permuted_embedding_sha256": sha256_file(args.synthetic_dir / "peak_permuted_synthetic_embeddings_f32.npy"),
        },
        "split": {
            "train_queries": int(len(train_positions)),
            "inner_selected_queries": int(len(inner_positions)),
            "inner_all_graph_queries": int(len(inner_all)),
            "outer_queries_untouched": int(len(outer_positions)),
            "train_formulas": len(train_formula),
            "inner_formulas": len(inner_formula),
            "outer_formulas": len(outer_formula),
            "formula_disjoint": True,
        },
        "model": {
            "class": "HierarchicalChemicalResidualAdapter",
            "official_dreams_frozen": True,
            "all_adapter_parameters_trainable": True,
            "deployable_parameters": deployable_parameter_count(template),
            "delta_bound": template.delta_bound,
            "shared_for_real_query_real_reference_and_synthetic_training_views": True,
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_queries": args.batch_queries,
            "learning_rate": args.learning_rate,
            "synthetic_alpha": args.synthetic_alpha,
            "temperature": args.temperature,
            "lambda_protect": args.lambda_protect,
            "lambda_preserve": args.lambda_preserve,
            "seed": args.seed,
            "matched_initialization_capacity_queries_and_steps": True,
        },
        "runs": runs,
        "comparisons": comparisons,
        "gate": (
            "Correct synthetic arm must beat identity-only, candidate-swapped, and peak-permuted "
            "arms on formula-cluster-bootstrap Hit@1 and margin lower bounds in selected inner queries."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparisons, indent=2, ensure_ascii=False), flush=True)
    print(f"decision={report['status']} report={args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
