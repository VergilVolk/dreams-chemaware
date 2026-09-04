"""Compare PEFT chemical-gradient directions under matched teacher controls.

This is a mechanism diagnostic, not model selection or a retrieval benchmark.
One shared zero-init PEFT forward graph is reused for all controls, so any
gradient difference is caused by the frozen molecule geometry alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))

from chemaware_shared_v2_core import (  # noqa: E402
    ChemAwareTokenStore, MoleculeTeacherStore, formula_folds, split_allowed_molecules,
)
from dreams.models.chem_aware.peft_v3 import DreaMSPEFTConfig, install_dreams_peft  # noqa: E402
from dreams.models.chem_aware.shared_embedding_v2 import (  # noqa: E402
    chemical_weighted_listwise_loss, molecule_listwise_loss,
)
from noise_final_core import CandidateGraph, json_dump, seed_everything, sha256_file  # noqa: E402
from train_chemaware_shared_v2 import build_query_batch, gradient_geometry  # noqa: E402
from train_chemaware_shared_v3_peft import RawSpectrumStore, score_batch  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402


ARMS = (
    "correct", "identity_permuted", "random_marginal",
    "correct_same_formula_scope", "same_formula_mismatched",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--molecule-teacher-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1701)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--forward-batch-size", type=int, default=16)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--chemical-hardness-beta", type=float, default=4.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def cosine(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise RuntimeError("gradient signatures are not aligned")
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite gradient diagnostic: {args.output}")
    if args.queries < 1 or args.rank < 1 or args.alpha <= 0:
        raise ValueError("queries/rank/alpha must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_everything(args.seed)
    device = torch.device(args.device)
    graph = CandidateGraph(args.graph)
    store = ChemAwareTokenStore(
        args.token_dir, args.graph, args.official_checkpoint, require_formal=False
    )
    store.require_graph_coverage(graph)
    teacher_store = MoleculeTeacherStore(
        args.molecule_teacher_dir, args.graph, graph, require_formal=False
    )
    query_fold = formula_folds(graph.query_formula, args.folds, args.fold_seed)
    inner_fold = (args.outer_fold + 1) % args.folds
    allowed = split_allowed_molecules(
        graph, args.outer_fold, inner_fold, args.folds, args.fold_seed
    )
    candidates = []
    for query in np.flatnonzero(
        (query_fold != args.outer_fold) & (query_fold != inner_fold)
    ):
        left, right = map(int, graph.query_ptr[query:query + 2])
        count = int(np.sum(allowed[left:right]))
        if allowed[left] and count >= 2:
            candidates.append((count, int(query)))
    # Candidate count is label-free and maximizes the chance that a teacher can
    # alter relative negative directions rather than only rescale one negative.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    queries = np.asarray([query for _, query in candidates[:args.queries]], dtype=np.int64)
    if len(queries) != args.queries:
        raise RuntimeError("not enough split-eligible diagnostic queries")

    raw_store = RawSpectrumStore(args.data, store.rows, args.n_highest_peaks)
    model, initialization = load_base_model(
        args.official_checkpoint, args.raw_checkpoint, device, args.n_highest_peaks
    )
    model.eval()
    capacity = install_dreams_peft(
        model, DreaMSPEFTConfig(last_blocks=1, rank=args.rank, alpha=args.alpha)
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    batch = build_query_batch(graph, queries, allowed)
    new_scores, _, _, _, _, _ = score_batch(
        model, raw_store, store, batch, device, args.forward_batch_size, args.amp
    )
    query_ptr = torch.from_numpy(batch["query_ptr"]).to(device=device, dtype=torch.long)
    clean = molecule_listwise_loss(new_scores, query_ptr, args.temperature)
    audits = {}
    for arm in ARMS:
        values, observable, teacher_audit = teacher_store.graph_embeddings(
            graph, allowed, arm, args.seed + 1009 * args.outer_fold
        )
        teacher_values = torch.from_numpy(values[batch["molecule_index"]]).to(
            device=device, dtype=torch.float32
        )
        teacher_observable = torch.from_numpy(
            observable[batch["molecule_index"]]
        ).to(device=device, dtype=torch.bool)
        chemical = chemical_weighted_listwise_loss(
            new_scores, teacher_values, query_ptr, args.temperature,
            args.chemical_hardness_beta, teacher_observable,
            weighting="absolute_bounded",
        )
        audit = gradient_geometry(clean, chemical - clean, trainable)
        audit.update({
            "clean_loss": float(clean.detach()),
            "chemical_loss": float(chemical.detach()),
            "teacher": teacher_audit,
        })
        audits[arm] = audit
    pairwise = {}
    for left_index, left in enumerate(ARMS):
        for right in ARMS[left_index + 1:]:
            pairwise[f"{left}__vs__{right}"] = cosine(
                audits[left]["chemical_delta_gradient_signature"],
                audits[right]["chemical_delta_gradient_signature"],
            )
    report = {
        "status": "chemaware_shared_v3_teacher_gradient_diagnostic_complete",
        "formal": False,
        "initialization": initialization,
        "queries": queries.tolist(),
        "candidate_molecules_per_query": [
            int(batch["query_ptr"][index + 1] - batch["query_ptr"][index])
            for index in range(len(queries))
        ],
        "capacity": capacity,
        "audits": audits,
        "pairwise_signature_cosine": pairwise,
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
            "teacher_report_sha256": sha256_file(args.molecule_teacher_dir / "report.json"),
        },
        "claim_limit": (
            "single-batch optimization-mechanism diagnostic only; not model selection, "
            "retrieval performance, chemical attribution, or external validation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
