"""Distill an offline ICEBERG candidate distribution into a shared spectrum encoder.

The teacher sees structures only while the ledger is built.  The student sees
one spectrum at a time (official DreaMS embedding, raw peaks and precursor m/z)
and produces a normalized 1024-D embedding used symmetrically for query and
reference spectra.  Formula-held-out evaluation therefore exercises the exact
deployment contract without candidate structures, formula labels, or ICEBERG.

Four matched arms isolate the source of any improvement:

* identity_only: one-hot molecule identity supervision;
* correct_teacher: identity anchor plus the aligned ICEBERG soft distribution;
* candidate_swapped: identical training with structure predictions relabelled;
* peak_permuted: identical training with ICEBERG peak intensities permuted.

This is a non-formal mechanism pilot because both the local graph and public
teacher weights originate from MassSpecGym.
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

from chemaware_shared_v2_core import formula_folds, paired_evaluation  # noqa: E402
from dreams.models.chem_aware.hierarchical_chemical_adapter import (  # noqa: E402
    HierarchicalChemicalResidualAdapter,
    deployable_parameter_count,
)
from noise_final_core import CandidateGraph  # noqa: E402


ARM_SCORE_KEYS = {
    "correct_teacher": "correct_score",
    "candidate_swapped": "candidate_swapped_score",
    "peak_permuted": "peak_permuted_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/validation/chemaware_shared_v2_cached_real_diagnostic/graph.npz"),
    )
    parser.add_argument(
        "--token-dir",
        type=Path,
        default=Path("data/validation/chemaware_shared_v2_cached_real_diagnostic/tokens"),
    )
    parser.add_argument(
        "--teacher-dir",
        type=Path,
        default=Path("data/validation/chemaware_iceberg_teacher_ledger_inner_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/chemaware_iceberg_distillation_inner_v1"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-queries", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-alpha", type=float, default=0.35)
    parser.add_argument("--teacher-temperature", type=float, default=0.10)
    parser.add_argument("--student-temperature", type=float, default=0.05)
    parser.add_argument("--lambda-protect", type=float, default=0.5)
    parser.add_argument("--lambda-preserve", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--fold-seed", type=int, default=20260904)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260904])
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RawSpectrumStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.rows = np.load(directory / "rows.npy").astype(np.int64)
        self.position = {int(row): index for index, row in enumerate(self.rows)}
        self.official = np.load(directory / "official_embeddings_f32.npy").astype(np.float32)
        self.mz = np.load(directory / "mz_f32.npy").astype(np.float32)
        self.intensity = np.load(directory / "intensity_f32.npy").astype(np.float32)
        self.valid = np.load(directory / "valid.npy").astype(bool)
        self.precursor = np.load(directory / "precursor_mz_f32.npy").astype(np.float32)
        if not (
            len(self.rows)
            == len(self.official)
            == len(self.mz)
            == len(self.intensity)
            == len(self.valid)
            == len(self.precursor)
        ):
            raise RuntimeError("raw-spectrum cache arrays are not aligned")
        if len(np.unique(self.rows)) != len(self.rows):
            raise RuntimeError("raw-spectrum cache contains duplicate rows")
        norms = np.linalg.norm(self.official, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-4):
            raise RuntimeError("official cached embeddings are not normalized")

    def positions(self, rows: np.ndarray) -> np.ndarray:
        try:
            return np.asarray([self.position[int(row)] for row in rows], dtype=np.int64)
        except KeyError as error:
            raise RuntimeError(f"spectrum row absent from cache: {error}") from error

    def tensors(self, rows: np.ndarray) -> tuple[torch.Tensor, ...]:
        index = self.positions(rows)
        return (
            torch.from_numpy(self.official[index]),
            torch.from_numpy(self.mz[index]),
            torch.from_numpy(self.intensity[index]),
            torch.from_numpy(self.precursor[index]),
            torch.from_numpy(self.valid[index]),
        )


def load_teacher_ledger(
    directory: Path, graph: CandidateGraph, graph_path: Path
) -> dict[str, np.ndarray | dict]:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError("teacher ledger did not pass candidate-specificity gate")
    if report["inputs"].get("graph_sha256") != sha256_file(graph_path):
        raise RuntimeError("teacher ledger graph provenance mismatch")
    query = np.load(directory / "selected_queries.npy").astype(np.int64)
    query_ptr = np.load(directory / "query_ptr.npy").astype(np.int64)
    molecule = np.load(directory / "candidate_molecule_index.npy").astype(np.int64)
    scores_file = directory / "scores_and_ranks.npz"
    with np.load(scores_file, allow_pickle=True) as body:
        scores = {key: body[key].astype(np.float32) for key in ARM_SCORE_KEYS.values()}
    if query_ptr.shape != (len(query) + 1,) or query_ptr[-1] != len(molecule):
        raise RuntimeError("teacher ledger pointers are malformed")
    if any(len(value) != len(molecule) for value in scores.values()):
        raise RuntimeError("teacher score arrays are not aligned")
    for position, query_index in enumerate(query):
        left, right = query_ptr[position : position + 2]
        graph_left, graph_right = graph.query_ptr[query_index : query_index + 2]
        expected = np.arange(graph_left, graph_right, dtype=np.int64)
        if not np.array_equal(molecule[int(left) : int(right)], expected):
            raise RuntimeError(f"teacher candidates disagree with graph query {query_index}")
    return {
        "report": report,
        "query": query,
        "query_ptr": query_ptr,
        "molecule": molecule,
        "scores": scores,
        "scores_sha256": sha256_file(scores_file),
    }


def query_payload(graph: CandidateGraph, query: int) -> tuple[int, np.ndarray, np.ndarray]:
    _, candidate_rows, local_ptr, _ = graph.query_block(query)
    return int(graph.query_row[query]), candidate_rows, local_ptr


def encode_unique(
    model: HierarchicalChemicalResidualAdapter,
    store: RawSpectrumStore,
    rows: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    unique, inverse = np.unique(rows, return_inverse=True)
    official, mz, intensity, precursor, valid = store.tensors(unique)
    output = model(official, mz, intensity, precursor, valid)
    index = torch.from_numpy(inverse).long()
    return output.embedding[index], official[index]


def molecule_scores(
    query_embedding: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    local_ptr: np.ndarray,
) -> torch.Tensor:
    pair = candidate_embeddings @ query_embedding
    return torch.stack(
        [torch.max(pair[int(left) : int(right)]) for left, right in zip(local_ptr[:-1], local_ptr[1:])]
    )


def batch_loss(
    model: HierarchicalChemicalResidualAdapter,
    store: RawSpectrumStore,
    graph: CandidateGraph,
    ledger_positions: np.ndarray,
    ledger: dict,
    arm: str,
    teacher_alpha: float,
    teacher_temperature: float,
    student_temperature: float,
    lambda_protect: float,
    lambda_preserve: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    all_rows: list[int] = []
    payloads = []
    for ledger_position in ledger_positions:
        query = int(ledger["query"][ledger_position])
        query_row, candidate_rows, local_ptr = query_payload(graph, query)
        offset = len(all_rows)
        all_rows.append(query_row)
        all_rows.extend(map(int, candidate_rows))
        payloads.append((int(ledger_position), offset, len(candidate_rows), local_ptr))
    adapted, official = encode_unique(model, store, np.asarray(all_rows, dtype=np.int64))
    losses = []
    identity_values = []
    teacher_values = []
    protect_values = []
    for ledger_position, offset, n_candidate_rows, local_ptr in payloads:
        new_score = molecule_scores(
            adapted[offset], adapted[offset + 1 : offset + 1 + n_candidate_rows], local_ptr
        )
        old_score = molecule_scores(
            official[offset], official[offset + 1 : offset + 1 + n_candidate_rows], local_ptr
        )
        identity = -F.log_softmax(new_score / student_temperature, dim=0)[0]
        if arm == "identity_only":
            target = torch.zeros_like(new_score)
            target[0] = 1.0
            teacher_ce = torch.zeros((), dtype=new_score.dtype)
        else:
            key = ARM_SCORE_KEYS[arm]
            left, right = ledger["query_ptr"][ledger_position : ledger_position + 2]
            distance = torch.from_numpy(ledger["scores"][key][int(left) : int(right)])
            teacher_probability = torch.softmax(-distance / teacher_temperature, dim=0)
            target = teacher_alpha * teacher_probability
            target = target.clone()
            target[0] += 1.0 - teacher_alpha
            teacher_ce = -torch.sum(
                teacher_probability * F.log_softmax(new_score / student_temperature, dim=0)
            )
        soft_ce = -torch.sum(target * F.log_softmax(new_score / student_temperature, dim=0))
        old_margin = old_score[0] - torch.max(old_score[1:])
        new_margin = new_score[0] - torch.max(new_score[1:])
        protect = torch.relu(torch.clamp(old_margin.detach(), min=0.0, max=0.05) - new_margin)
        losses.append(soft_ce + lambda_protect * protect)
        identity_values.append(identity)
        teacher_values.append(teacher_ce)
        protect_values.append(protect)
    preserve = torch.mean(1.0 - torch.sum(adapted * official, dim=1))
    total = torch.stack(losses).mean() + lambda_preserve * preserve
    return total, {
        "identity_ce": float(torch.stack(identity_values).mean().detach()),
        "teacher_ce": float(torch.stack(teacher_values).mean().detach()),
        "protect": float(torch.stack(protect_values).mean().detach()),
        "preserve": float(preserve.detach()),
    }


@torch.no_grad()
def encode_all(
    model: HierarchicalChemicalResidualAdapter,
    store: RawSpectrumStore,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = np.empty_like(store.official)
    for left in range(0, len(store.rows), batch_size):
        right = min(left + batch_size, len(store.rows))
        official, mz, intensity, precursor, valid = store.tensors(store.rows[left:right])
        output[left:right] = model(official, mz, intensity, precursor, valid).embedding.numpy()
    return output


def arm_run(
    arm: str,
    seed: int,
    initial_state: dict[str, torch.Tensor],
    store: RawSpectrumStore,
    graph: CandidateGraph,
    ledger: dict,
    train_positions: np.ndarray,
    inner_selected_queries: np.ndarray,
    inner_all_queries: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, np.ndarray], dict[str, torch.Tensor]]:
    set_seed(seed)
    model = HierarchicalChemicalResidualAdapter(dropout=0.0, use_formula_moments=True)
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    initial = encode_all(model, store, args.eval_batch_size)
    if np.max(np.abs(initial - store.official)) != 0:
        raise RuntimeError(f"{arm} does not exactly reproduce official embeddings at initialization")
    history = []
    rng = np.random.default_rng(seed + 701)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_positions)
        totals = {"loss": 0.0, "identity_ce": 0.0, "teacher_ce": 0.0, "protect": 0.0, "preserve": 0.0}
        batches = 0
        for left in range(0, len(order), args.batch_queries):
            positions = order[left : left + args.batch_queries]
            optimizer.zero_grad(set_to_none=True)
            loss, components = batch_loss(
                model,
                store,
                graph,
                positions,
                ledger,
                arm,
                args.teacher_alpha,
                args.teacher_temperature,
                args.student_temperature,
                args.lambda_protect,
                args.lambda_preserve,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for key, value in components.items():
                totals[key] += value
            batches += 1
        encoded = encode_all(model, store, args.eval_batch_size)
        selected_eval = paired_evaluation(
            encoded, store.official, store, graph, inner_selected_queries
        )
        all_eval = paired_evaluation(encoded, store.official, store, graph, inner_all_queries)
        history.append(
            {
                "epoch": epoch,
                "train": {key: value / batches for key, value in totals.items()},
                "inner_selected": selected_eval["summary"],
                "inner_all_graph": all_eval["summary"],
            }
        )
        print(
            f"seed={seed} arm={arm} epoch={epoch}/{args.epochs} "
            f"selected={selected_eval['summary']['recall1']:.4f} "
            f"all={all_eval['summary']['recall1']:.4f}",
            flush=True,
        )
    final_encoded = encode_all(model, store, args.eval_batch_size)
    selected_final = paired_evaluation(
        final_encoded, store.official, store, graph, inner_selected_queries
    )
    all_final = paired_evaluation(final_encoded, store.official, store, graph, inner_all_queries)
    return (
        {
            "arm": arm,
            "seed": seed,
            "final": {
                "inner_selected": selected_final["summary"],
                "inner_all_graph": all_final["summary"],
            },
            "history": history,
        },
        {
            "selected_rank": selected_final["new_rank"],
            "selected_margin": selected_final["new_margin"],
            "all_rank": all_final["new_rank"],
            "all_margin": all_final["new_margin"],
        },
        {name: value.detach().cpu() for name, value in model.state_dict().items()},
    )


def formula_cluster_bootstrap(
    difference: np.ndarray,
    formulas: np.ndarray,
    seed: int,
    draws: int,
) -> dict:
    unique = np.unique(formulas)
    values = np.asarray(
        [np.mean(difference[formulas == formula]) for formula in unique], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        estimates[draw] = np.mean(values[rng.integers(0, len(values), len(values))])
    return {
        "formula_macro_advantage": float(np.mean(values)),
        "formula_cluster_bootstrap_95ci": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "formula_clusters": int(len(unique)),
        "draws": int(draws),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.teacher_alpha <= 1:
        raise ValueError("teacher-alpha must be in [0, 1]")
    if args.inner_fold == args.outer_fold:
        raise ValueError("inner and outer folds must differ")
    if min(args.epochs, args.batch_queries, args.eval_batch_size, args.bootstrap_draws) < 1:
        raise ValueError("epoch, batch and bootstrap arguments must be positive")
    torch.set_num_threads(args.torch_threads)
    args.output.mkdir(parents=True, exist_ok=True)
    graph = CandidateGraph(args.graph)
    store = RawSpectrumStore(args.token_dir)
    reachable = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row)))
    if not np.array_equal(np.sort(store.rows), reachable):
        raise RuntimeError("raw-spectrum cache does not exactly cover candidate graph")
    ledger = load_teacher_ledger(args.teacher_dir, graph, args.graph)
    ledger_query = ledger["query"]
    ledger_folds = formula_folds(graph.query_formula[ledger_query], args.folds, args.fold_seed)
    train_positions = np.flatnonzero(
        (ledger_folds != args.inner_fold) & (ledger_folds != args.outer_fold)
    )
    inner_positions = np.flatnonzero(ledger_folds == args.inner_fold)
    outer_positions = np.flatnonzero(ledger_folds == args.outer_fold)
    inner_selected_queries = ledger_query[inner_positions]
    all_query_folds = formula_folds(graph.query_formula, args.folds, args.fold_seed)
    inner_all_queries = np.flatnonzero(all_query_folds == args.inner_fold)
    if not len(train_positions) or not len(inner_positions) or not len(outer_positions):
        raise RuntimeError("formula split produced an empty partition")
    train_formulas = set(graph.query_formula[ledger_query[train_positions]])
    inner_formulas = set(graph.query_formula[inner_selected_queries])
    outer_formulas = set(graph.query_formula[ledger_query[outer_positions]])
    if train_formulas & inner_formulas or train_formulas & outer_formulas or inner_formulas & outer_formulas:
        raise RuntimeError("formula split leakage")

    set_seed(args.seeds[0])
    template = HierarchicalChemicalResidualAdapter(dropout=0.0, use_formula_moments=True)
    initial_state = {name: value.detach().cpu() for name, value in template.state_dict().items()}
    initial_hash = state_sha256(initial_state)
    arms = ["identity_only", "correct_teacher", "candidate_swapped", "peak_permuted"]
    runs = []
    run_arrays: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for seed in args.seeds:
        for arm in arms:
            summary, arrays, state = arm_run(
                arm,
                seed,
                initial_state,
                store,
                graph,
                ledger,
                train_positions,
                inner_selected_queries,
                inner_all_queries,
                args,
            )
            run_dir = args.output / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(state, run_dir / f"{arm}.pt")
            np.savez_compressed(run_dir / f"{arm}_evaluation.npz", **arrays)
            runs.append(summary)
            run_arrays[(seed, arm)] = arrays

    comparisons = []
    selected_formulas = graph.query_formula[inner_selected_queries]
    all_formulas = graph.query_formula[inner_all_queries]
    for seed in args.seeds:
        correct = run_arrays[(seed, "correct_teacher")]
        for control in ("identity_only", "candidate_swapped", "peak_permuted"):
            baseline = run_arrays[(seed, control)]
            comparisons.append(
                {
                    "seed": seed,
                    "correct_minus": control,
                    "inner_selected_hit1": formula_cluster_bootstrap(
                        (correct["selected_rank"] == 1).astype(float)
                        - (baseline["selected_rank"] == 1).astype(float),
                        selected_formulas,
                        seed + 1701,
                        args.bootstrap_draws,
                    ),
                    "inner_selected_margin": formula_cluster_bootstrap(
                        correct["selected_margin"] - baseline["selected_margin"],
                        selected_formulas,
                        seed + 1703,
                        args.bootstrap_draws,
                    ),
                    "inner_all_graph_hit1": formula_cluster_bootstrap(
                        (correct["all_rank"] == 1).astype(float)
                        - (baseline["all_rank"] == 1).astype(float),
                        all_formulas,
                        seed + 1709,
                        args.bootstrap_draws,
                    ),
                    "inner_all_graph_margin": formula_cluster_bootstrap(
                        correct["all_margin"] - baseline["all_margin"],
                        all_formulas,
                        seed + 1711,
                        args.bootstrap_draws,
                    ),
                }
            )

    correct_identity = [
        value for value in comparisons if value["correct_minus"] == "identity_only"
    ]
    correct_controls = [
        value for value in comparisons if value["correct_minus"] != "identity_only"
    ]
    pass_identity = all(
        value["inner_selected_hit1"]["formula_cluster_bootstrap_95ci"][0] > 0
        and value["inner_selected_margin"]["formula_cluster_bootstrap_95ci"][0] > 0
        for value in correct_identity
    )
    pass_controls = all(
        value["inner_selected_margin"]["formula_cluster_bootstrap_95ci"][0] > 0
        for value in correct_controls
    )
    passed = bool(pass_identity and pass_controls)
    report = {
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "Aligned teacher information transferred beyond identity-only and negative controls."
            if passed
            else "No reliable teacher-specific transfer; do not widen or formally train this adapter route."
        ),
        "scope": {
            "non_formal_mechanism_pilot": True,
            "teacher_training_only": True,
            "student_deployment_inputs": [
                "one raw MS/MS spectrum",
                "precursor m/z",
                "official DreaMS embedding of that same spectrum",
            ],
            "student_deployment_output": "normalized 1024-D shared embedding",
            "candidate_or_structure_input_at_deployment": False,
            "outer_fold_evaluated": False,
            "massspecgym_overlap_warning": True,
        },
        "provenance": {
            "graph": str(args.graph.resolve()),
            "graph_sha256": sha256_file(args.graph),
            "token_dir": str(args.token_dir.resolve()),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "teacher_dir": str(args.teacher_dir.resolve()),
            "teacher_report_sha256": sha256_file(args.teacher_dir / "report.json"),
            "teacher_scores_sha256": ledger["scores_sha256"],
            "initial_state_sha256": initial_hash,
        },
        "model": {
            "class": "HierarchicalChemicalResidualAdapter",
            "official_dreams_frozen": True,
            "use_formula_moments": True,
            "formula_labels_used": False,
            "all_adapter_parameters_trainable": True,
            "total_parameters": sum(value.numel() for value in template.parameters()),
            "deployable_parameters": deployable_parameter_count(template),
            "delta_bound": template.delta_bound,
        },
        "split": {
            "folds": args.folds,
            "fold_seed": args.fold_seed,
            "train_folds": sorted(set(range(args.folds)) - {args.inner_fold, args.outer_fold}),
            "inner_fold": args.inner_fold,
            "outer_fold": args.outer_fold,
            "train_queries": int(len(train_positions)),
            "inner_selected_queries": int(len(inner_positions)),
            "inner_all_graph_queries": int(len(inner_all_queries)),
            "outer_queries_untouched": int(len(outer_positions)),
            "train_unique_formulas": len(train_formulas),
            "inner_unique_formulas": len(inner_formulas),
            "outer_unique_formulas": len(outer_formulas),
            "formula_disjoint": True,
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_queries": args.batch_queries,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "teacher_alpha": args.teacher_alpha,
            "teacher_temperature": args.teacher_temperature,
            "student_temperature": args.student_temperature,
            "lambda_protect": args.lambda_protect,
            "lambda_preserve": args.lambda_preserve,
            "seeds": args.seeds,
            "matched_initialization": True,
            "matched_step_counts": True,
        },
        "gate": {
            "required": (
                "For every seed, correct teacher must beat identity-only on formula-cluster-bootstrap "
                "Hit@1 and margin lower bounds in the selected inner set, and beat both chemistry "
                "controls on margin lower bounds."
            ),
            "pass_identity": pass_identity,
            "pass_controls": pass_controls,
        },
        "runs": runs,
        "comparisons": comparisons,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"decision={report['status']} report={args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
