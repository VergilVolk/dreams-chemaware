"""Matched retrieval pilot after raw-spectrum formula pretraining.

Three branches start with an exactly zero residual and therefore reproduce the
official DreaMS embedding and ranks: an untrained chemical encoder, a correctly
formula-pretrained encoder, and a within-spectrum peak-permuted control.  During
this pilot only the common residual projection is trainable.  Thus any correct
arm advantage must already be present in the candidate-independent chemical
representation; identity-only retrieval training cannot rewrite that encoder.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_shared_v2_core import (  # noqa: E402
    formula_folds,
    paired_evaluation,
    split_allowed_molecules,
)
from dreams.models.chem_aware.hierarchical_chemical_adapter import (  # noqa: E402
    HierarchicalChemicalResidualAdapter,
    deployable_parameter_count,
)
from dreams.models.chem_aware.shared_embedding_v2 import (  # noqa: E402
    molecule_listwise_loss,
    molecule_scores_from_spectrum_pairs,
    protected_margin_loss,
)
from noise_final_core import CandidateGraph, seed_everything  # noqa: E402
from train_chemaware_shared_v2 import build_query_batch  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RawSpectrumStore:
    def __init__(self, token_dir: Path):
        self.rows = np.load(token_dir / "rows.npy").astype(np.int64)
        self.position = {int(row): index for index, row in enumerate(self.rows)}
        self.official = np.load(token_dir / "official_embeddings_f32.npy").astype(np.float32)
        self.mz = np.load(token_dir / "mz_f32.npy").astype(np.float32)
        self.intensity = np.load(token_dir / "intensity_f32.npy").astype(np.float32)
        self.valid = np.load(token_dir / "valid.npy").astype(bool)
        self.precursor = np.load(token_dir / "precursor_mz_f32.npy").astype(np.float32)
        self.dimension = int(self.official.shape[1])
        if len(self.rows) != len(self.official) or len(self.rows) != len(self.mz):
            raise RuntimeError("raw-spectrum cache arrays are not aligned")
        norms = np.linalg.norm(self.official, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-6):
            raise RuntimeError("official cached embeddings are not normalized")

    def positions(self, rows: np.ndarray) -> np.ndarray:
        try:
            return np.asarray([self.position[int(row)] for row in rows], dtype=np.int64)
        except KeyError as error:
            raise RuntimeError(f"spectrum row absent from cache: {error}") from error

    def batch(self, rows: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
        index = self.positions(rows)
        return (
            torch.from_numpy(self.official[index]).to(device),
            torch.from_numpy(self.mz[index]).to(device),
            torch.from_numpy(self.intensity[index]).to(device),
            torch.from_numpy(self.precursor[index]).to(device),
            torch.from_numpy(self.valid[index]).to(device),
        )


def encode_rows(
    model: HierarchicalChemicalResidualAdapter,
    store: RawSpectrumStore,
    rows: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    official, mz, intensity, precursor, valid = store.batch(rows, device)
    output = model(official, mz, intensity, precursor, valid)
    return output.embedding, official


def score_batch(
    model: HierarchicalChemicalResidualAdapter,
    store: RawSpectrumStore,
    batch: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    joined = np.concatenate((batch["query_rows"], batch["candidate_rows"]))
    unique, inverse = np.unique(joined, return_inverse=True)
    encoded, official = encode_rows(model, store, unique, device)
    n_query = len(batch["query_rows"])
    query_index = torch.from_numpy(inverse[:n_query]).to(device=device, dtype=torch.long)
    candidate_index = torch.from_numpy(inverse[n_query:]).to(device=device, dtype=torch.long)
    pair_query = torch.from_numpy(batch["pair_query"]).to(device=device, dtype=torch.long)
    molecule_ptr = torch.from_numpy(batch["molecule_ptr"]).to(device=device, dtype=torch.long)
    new_scores = molecule_scores_from_spectrum_pairs(
        encoded[query_index], encoded[candidate_index], pair_query, molecule_ptr,
    )
    old_scores = molecule_scores_from_spectrum_pairs(
        official[query_index], official[candidate_index], pair_query, molecule_ptr,
    )
    preserve = torch.mean(1.0 - torch.sum(encoded * official, dim=1))
    return new_scores, old_scores, preserve


@torch.no_grad()
def encode_all(
    model: HierarchicalChemicalResidualAdapter,
    store: RawSpectrumStore,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    encoded = np.empty_like(store.official)
    for left in range(0, len(store.rows), batch_size):
        right = min(left + batch_size, len(store.rows))
        value, _ = encode_rows(model, store, store.rows[left:right], device)
        encoded[left:right] = value.cpu().numpy()
    return encoded


def retrieval_arm(
    name: str,
    checkpoint: Path,
    store: RawSpectrumStore,
    graph: CandidateGraph,
    train_query: np.ndarray,
    inner_query: np.ndarray,
    allowed_molecule: np.ndarray,
    query_weight: dict[int, float],
    device: torch.device,
    epochs: int,
    batch_queries: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    lambda_protect: float,
    lambda_preserve: float,
    seed: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    seed_everything(seed)
    model = HierarchicalChemicalResidualAdapter(
        dropout=0.0, use_formula_moments=True,
    ).to(device)
    pretrained = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # Formula pretraining used the aggregate-only zero residual.  It was frozen
    # and contains no learned information, so all arms discard it and receive
    # the same freshly zero-initialized formula-moment residual projection.
    pretrained = {
        name: value for name, value in pretrained.items()
        if not name.startswith("residual_head.")
    }
    missing, unexpected = model.load_state_dict(pretrained, strict=False)
    if unexpected or set(missing) != {
        "residual_head.0.weight", "residual_head.0.bias",
        "residual_head.1.weight", "residual_head.1.bias",
    }:
        raise RuntimeError(f"unexpected formula checkpoint mismatch: {missing}, {unexpected}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.residual_head.parameters():
        parameter.requires_grad_(True)
    trainable = list(model.residual_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    official_encoded = store.official.copy()
    initial_encoded = encode_all(model, store, device, eval_batch_size)
    initial = paired_evaluation(initial_encoded, official_encoded, store, graph, inner_query)
    if not np.array_equal(initial["old_rank"], initial["new_rank"]):
        raise RuntimeError(f"{name} does not exactly reproduce official ranks at initialization")
    if np.max(np.abs(initial_encoded - official_encoded)) != 0:
        raise RuntimeError(f"{name} residual is not exactly zero at initialization")

    history = [{"epoch": 0, "inner": initial["summary"]}]
    rng = np.random.default_rng(seed + 707)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_query)
        components = {"loss": 0.0, "listwise": 0.0, "protect": 0.0, "preserve": 0.0}
        batches = 0
        for left in range(0, len(order), batch_queries):
            queries = order[left : left + batch_queries]
            batch = build_query_batch(graph, queries, allowed_molecule)
            new_scores, old_scores, preserve = score_batch(model, store, batch, device)
            query_ptr = torch.from_numpy(batch["query_ptr"]).to(device=device, dtype=torch.long)
            weights = torch.tensor(
                [query_weight[int(query)] for query in queries], device=device,
            )
            listwise = molecule_listwise_loss(new_scores, query_ptr, temperature, weights)
            protect = protected_margin_loss(new_scores, old_scores, query_ptr)
            loss = listwise + lambda_protect * protect + lambda_preserve * preserve
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            for key, value in (
                ("loss", loss), ("listwise", listwise),
                ("protect", protect), ("preserve", preserve),
            ):
                components[key] += float(value.detach())
            batches += 1
        encoded = encode_all(model, store, device, eval_batch_size)
        inner = paired_evaluation(encoded, official_encoded, store, graph, inner_query)
        history.append({
            "epoch": epoch,
            "train": {key: value / batches for key, value in components.items()},
            "inner": inner["summary"],
        })

    final_encoded = encode_all(model, store, device, eval_batch_size)
    final = paired_evaluation(final_encoded, official_encoded, store, graph, inner_query)
    return {
        "name": name,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "only_residual_head_trainable": True,
        "initial": initial["summary"],
        "final": final["summary"],
        "history": history,
    }, {"rank": final["new_rank"], "margin": final["new_margin"]}


def formula_cluster_bootstrap(
    difference: np.ndarray,
    formulas: np.ndarray,
    seed: int,
    draws: int = 10_000,
) -> dict:
    if difference.shape != formulas.shape:
        raise RuntimeError("paired difference/formula arrays are not aligned")
    unique = np.unique(formulas)
    cluster = np.asarray([np.mean(difference[formulas == value]) for value in unique])
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for draw in range(draws):
        estimates[draw] = np.mean(cluster[rng.integers(0, len(cluster), len(cluster))])
    return {
        "formula_macro_advantage": float(np.mean(cluster)),
        "formula_cluster_bootstrap_95ci": [
            float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975)),
        ],
        "formula_clusters": int(len(unique)),
        "draws": draws,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--formula-branch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1701)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-queries", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-protect", type=float, default=2.0)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    device = torch.device(args.device)
    graph = CandidateGraph(args.graph)
    store = RawSpectrumStore(args.token_dir)
    needed = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row)))
    if not np.all(np.isin(needed, store.rows)):
        raise RuntimeError("token store does not cover candidate graph")
    query_fold = formula_folds(graph.query_formula, args.folds, args.fold_seed)
    inner_fold = (args.outer_fold + 1) % args.folds
    train_query = np.flatnonzero(
        (query_fold != args.outer_fold) & (query_fold != inner_fold)
    )
    inner_query = np.flatnonzero(query_fold == inner_fold)
    outer_query = np.flatnonzero(query_fold == args.outer_fold)
    allowed_molecule = split_allowed_molecules(
        graph, args.outer_fold, inner_fold, args.folds, args.fold_seed,
    )
    train_query = np.asarray([
        int(query) for query in train_query
        if allowed_molecule[int(graph.query_ptr[query])]
        and int(np.sum(allowed_molecule[
            int(graph.query_ptr[query]) : int(graph.query_ptr[query + 1])
        ])) >= 2
    ], dtype=np.int64)
    identities = graph.query_ik14[train_query].astype(str)
    _, inverse, counts = np.unique(identities, return_inverse=True, return_counts=True)
    values = (1.0 / counts[inverse]).astype(np.float32)
    values /= values.mean()
    query_weight = {int(query): float(weight) for query, weight in zip(train_query, values)}

    checkpoints = {
        "untrained": args.formula_branch_dir / "initial_formula_branch.pt",
        "correct": args.formula_branch_dir / "correct_formula_branch.pt",
        "peak_permuted": args.formula_branch_dir / "peak_permuted_formula_branch.pt",
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    arms = {}
    ranks = {}
    for name, checkpoint in checkpoints.items():
        arms[name], ranks[name] = retrieval_arm(
            name, checkpoint, store, graph, train_query, inner_query,
            allowed_molecule, query_weight, device, args.epochs, args.batch_queries,
            args.eval_batch_size, args.learning_rate, args.weight_decay,
            args.temperature, args.lambda_protect, args.lambda_preserve, args.seed,
        )
        print(name, json.dumps(arms[name]["final"]), flush=True)

    comparisons = {
        "correct_vs_peak_permuted": formula_cluster_bootstrap(
            (ranks["correct"]["rank"] == 1).astype(float)
            - (ranks["peak_permuted"]["rank"] == 1).astype(float),
            graph.query_formula[inner_query], args.seed + 101,
        ),
        "correct_vs_untrained": formula_cluster_bootstrap(
            (ranks["correct"]["rank"] == 1).astype(float)
            - (ranks["untrained"]["rank"] == 1).astype(float),
            graph.query_formula[inner_query], args.seed + 102,
        ),
        "correct_vs_peak_permuted_margin": formula_cluster_bootstrap(
            ranks["correct"]["margin"] - ranks["peak_permuted"]["margin"],
            graph.query_formula[inner_query], args.seed + 103,
        ),
        "correct_vs_untrained_margin": formula_cluster_bootstrap(
            ranks["correct"]["margin"] - ranks["untrained"]["margin"],
            graph.query_formula[inner_query], args.seed + 104,
        ),
    }
    correct = arms["correct"]["final"]
    control = comparisons["correct_vs_peak_permuted"]
    clean = comparisons["correct_vs_untrained"]
    control_margin = comparisons["correct_vs_peak_permuted_margin"]
    clean_margin = comparisons["correct_vs_untrained_margin"]
    recall_specific = bool(
        control["formula_macro_advantage"] > 0
        and control["formula_cluster_bootstrap_95ci"][0] > 0
        and clean["formula_macro_advantage"] > 0
        and clean["formula_cluster_bootstrap_95ci"][0] > 0
    )
    margin_specific = bool(
        control_margin["formula_macro_advantage"] > 0
        and control_margin["formula_cluster_bootstrap_95ci"][0] > 0
        and clean_margin["formula_macro_advantage"] > 0
        and clean_margin["formula_cluster_bootstrap_95ci"][0] > 0
    )
    gate = bool(
        correct["delta_recall1"] >= 0
        and correct["corrected"] >= correct["introduced"]
        and correct["preservation_mean"] >= 0.995
        and (recall_specific or margin_specific)
    )
    report = {
        "status": "nonformal_raw_formula_retrieval_pilot",
        "formal_training_authorized": False,
        "official_dreams_parameters_updated": False,
        "outer_fold_evaluated": False,
        "selection": {
            "train_queries": int(len(train_query)),
            "inner_queries": int(len(inner_query)),
            "inner_formula_clusters": int(len(np.unique(graph.query_formula[inner_query]))),
            "outer_queries_untouched": int(len(outer_query)),
            "formula_fold_overlap": 0,
        },
        "architecture": {
            "deployable_parameters": deployable_parameter_count(
                HierarchicalChemicalResidualAdapter(use_formula_moments=True)
            ),
            "candidate_inputs_used_by_encoder": False,
            "retrieval_trainable_component": "residual_head_only",
        },
        "optimization": {
            "epochs_fixed_in_advance": args.epochs,
            "batch_queries": args.batch_queries,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "lambda_protect": args.lambda_protect,
            "lambda_preserve": args.lambda_preserve,
            "same_query_ledger_initialization_batch_order_optimizer": True,
            "checkpoint_selection": "none_final_epoch_only",
        },
        "arms": arms,
        "comparisons": comparisons,
        "pass_to_multifold_retrieval_confirmation": gate,
        "gate": (
            "correct arm must not reduce official recall or net corrections, preserve official "
            "embeddings, and beat both peak-permuted and untrained arms in either recall@1 or "
            "positive-vs-hardest-negative margin with formula-cluster bootstrap lower bounds >0"
        ),
        "claim_limit": (
            "This selected mass-dense local graph is a mechanism diagnostic. A pass is not a "
            "formal benchmark result and does not authorize claims before multi-fold confirmation."
        ),
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "formula_branch_report_sha256": sha256_file(
                args.formula_branch_dir / "report.json"
            ),
        },
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "arms": {name: value["final"] for name, value in arms.items()},
        "comparisons": comparisons,
        "pass_to_multifold_retrieval_confirmation": gate,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
