"""G1: clean full-candidate listwise training for ChemAware shared embedding v2.

This is the capacity/sampler/optimizer-matched control for every later chemical
supervision experiment.  It uses no rules, structures, fingerprints, P2b
scores, or candidate features.  Both query and reference spectra pass through
one zero-initialized candidate-independent adapter.  The primary loss covers
every candidate molecule allowed by the formula-isolated training split and
uses the deployment max-over-reference-spectra aggregation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))

from chemaware_shared_v2_core import (  # noqa: E402
    ChemAwareTokenStore, MoleculeTeacherStore, encode_all, formula_folds,
    paired_evaluation, split_allowed_molecules,
)
from dreams.models.chem_aware.shared_embedding_v2 import (  # noqa: E402
    SignedPeakResidualAdapter, chemical_weighted_listwise_loss,
    molecule_listwise_loss, molecule_scores_from_spectrum_pairs,
    protected_margin_loss,
)
from noise_final_core import CandidateGraph, json_dump, seed_everything, sha256_file  # noqa: E402


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def gradient_geometry(
    clean_loss: torch.Tensor,
    chemical_delta_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> dict[str, float | int]:
    """Audit whether chemistry changes gradients of deployable parameters."""
    clean = torch.autograd.grad(
        clean_loss, parameters, retain_graph=True, allow_unused=True
    )
    delta = torch.autograd.grad(
        chemical_delta_loss, parameters, retain_graph=True, allow_unused=True
    )
    clean_sq = torch.zeros((), device=clean_loss.device, dtype=torch.float64)
    delta_sq = torch.zeros_like(clean_sq)
    dot = torch.zeros_like(clean_sq)
    delta_nonzero = 0
    # Fixed CountSketch over every gradient coordinate.  It is compact enough
    # for JSON ledgers but, unlike sparse coordinate sampling, cannot silently
    # miss a difference merely because it fell between sampled indices.
    signature = torch.zeros(128, device=clean_loss.device, dtype=torch.float64)
    signature_offset = 0
    for left, right in zip(clean, delta):
        if left is not None:
            clean_sq += torch.sum(left.double() ** 2)
        if right is not None:
            delta_sq += torch.sum(right.double() ** 2)
            delta_nonzero += int(bool(torch.any(right != 0).detach()))
            flat = right.detach().double().reshape(-1)
            if flat.numel():
                coordinate = torch.arange(
                    signature_offset, signature_offset + flat.numel(),
                    device=flat.device, dtype=torch.int64,
                )
                hashed = coordinate * 1_103_515_245 + 12_345
                bucket = torch.remainder(hashed, signature.numel())
                sign = torch.where(
                    torch.remainder(torch.div(hashed, signature.numel(), rounding_mode="floor"), 2) == 0,
                    1.0, -1.0,
                )
                signature.scatter_add_(0, bucket, flat * sign)
                signature_offset += flat.numel()
        if left is not None and right is not None:
            dot += torch.sum(left.double() * right.double())
    clean_norm = torch.sqrt(clean_sq)
    delta_norm = torch.sqrt(delta_sq)
    cosine = dot / (clean_norm * delta_norm).clamp_min(1e-30)
    if signature_offset:
        signature = signature / torch.linalg.vector_norm(signature).clamp_min(1e-30)
        signature_values = signature.cpu().tolist()
    else:
        signature_values = []
    return {
        "clean_gradient_norm": float(clean_norm.detach().cpu()),
        "chemical_minus_clean_gradient_norm": float(delta_norm.detach().cpu()),
        "clean_vs_chemical_delta_cosine": float(cosine.detach().cpu()),
        "chemical_delta_nonzero_parameter_tensors": int(delta_nonzero),
        # A compact deterministic sketch lets paired teacher/control runs test
        # whether chemistry changes gradient direction rather than merely
        # multiplying the clean gradient by a scalar.
        "chemical_delta_gradient_signature": signature_values,
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_tokens")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--preflight", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_preflight.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_g1")
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1701)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-queries", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--delta-bound", type=float, default=0.12)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--gate-topk", type=int, default=0)
    parser.add_argument("--contextual-gate", action="store_true")
    parser.add_argument("--global-branch", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-listwise", type=float, default=1.0)
    parser.add_argument("--lambda-protect", type=float, default=2.0)
    parser.add_argument("--lambda-preserve", type=float, default=1.0)
    parser.add_argument("--protect-slack", type=float, default=0.0)
    parser.add_argument("--minimum-preservation", type=float, default=0.995)
    parser.add_argument("--recall1-floor", type=float, default=-5e-4)
    parser.add_argument("--near-recall1-floor", type=float, default=-1e-3)
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--near-selection-weight", type=float, default=0.25)
    parser.add_argument("--molecule-teacher-dir", type=Path)
    parser.add_argument(
        "--teacher-control",
        choices=(
            "correct", "identity_permuted", "random_marginal",
            "correct_same_formula_scope", "same_formula_mismatched",
        ),
    )
    parser.add_argument("--lambda-molecule", type=float, default=0.25)
    parser.add_argument("--molecule-temperature", type=float, default=0.07)
    parser.add_argument("--chemical-hardness-beta", type=float, default=4.0)
    parser.add_argument(
        "--chemical-weighting",
        choices=("relative_centered", "absolute_bounded"),
        default="relative_centered",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-queries", type=int, default=0, help="non-formal smoke only")
    parser.add_argument("--max-eval-queries", type=int, default=0, help="non-formal smoke only")
    return parser.parse_args()


def build_query_batch(
    graph: CandidateGraph,
    queries: np.ndarray,
    allowed_molecule: np.ndarray,
) -> dict[str, np.ndarray]:
    """Materialize complete split-eligible candidate groups for query batch."""
    query_rows: list[int] = []
    candidate_rows: list[int] = []
    pair_query: list[int] = []
    molecule_ptr = [0]
    query_ptr = [0]
    molecule_index: list[int] = []
    for local_query, query in enumerate(np.asarray(queries, dtype=np.int64)):
        molecule_left, molecule_right = map(int, graph.query_ptr[query:query + 2])
        selected = np.flatnonzero(allowed_molecule[molecule_left:molecule_right]) + molecule_left
        if not len(selected) or selected[0] != molecule_left:
            raise RuntimeError(f"positive molecule excluded for training query {query}")
        if len(selected) < 2:
            raise RuntimeError(f"training query {query} has no split-eligible negative molecule")
        query_rows.append(int(graph.query_row[query]))
        for molecule in selected:
            molecule_index.append(int(molecule))
            left, right = map(int, graph.molecule_ptr[molecule:molecule + 2])
            rows = graph.pair_candidate_row[left:right]
            candidate_rows.extend(map(int, rows))
            pair_query.extend([local_query] * len(rows))
            molecule_ptr.append(molecule_ptr[-1] + len(rows))
        query_ptr.append(query_ptr[-1] + len(selected))
    return {
        "query_rows": np.asarray(query_rows, dtype=np.int64),
        "candidate_rows": np.asarray(candidate_rows, dtype=np.int64),
        "pair_query": np.asarray(pair_query, dtype=np.int64),
        "molecule_ptr": np.asarray(molecule_ptr, dtype=np.int64),
        "query_ptr": np.asarray(query_ptr, dtype=np.int64),
        "molecule_index": np.asarray(molecule_index, dtype=np.int64),
    }


def score_batch(
    adapter: SignedPeakResidualAdapter,
    store: ChemAwareTokenStore,
    batch: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    joined = np.concatenate((batch["query_rows"], batch["candidate_rows"]))
    unique, inverse = np.unique(joined, return_inverse=True)
    official, adapted, _, _, _ = store.adapt(adapter, unique, device)
    n_query = len(batch["query_rows"])
    query_index = torch.from_numpy(inverse[:n_query]).to(device=device, dtype=torch.long)
    candidate_index = torch.from_numpy(inverse[n_query:]).to(device=device, dtype=torch.long)
    pair_query = torch.from_numpy(batch["pair_query"]).to(device=device, dtype=torch.long)
    molecule_ptr = torch.from_numpy(batch["molecule_ptr"]).to(device=device, dtype=torch.long)
    new_scores = molecule_scores_from_spectrum_pairs(
        adapted[query_index], adapted[candidate_index], pair_query, molecule_ptr
    )
    old_scores = molecule_scores_from_spectrum_pairs(
        official[query_index], official[candidate_index], pair_query, molecule_ptr
    )
    preservation = torch.mean(1.0 - torch.sum(adapted * official, dim=1))
    return new_scores, old_scores, preservation


def main() -> None:
    args = arguments()
    if args.folds < 3 or args.outer_fold not in range(args.folds):
        raise ValueError("invalid formula outer fold")
    if args.epochs < 1 or args.batch_queries < 1 or args.eval_batch_size < 1:
        raise ValueError("invalid training schedule")
    if args.gate_temperature <= 0 or args.gate_topk < 0:
        raise ValueError("invalid peak-gate temperature/top-k")
    if (args.molecule_teacher_dir is None) != (args.teacher_control is None):
        raise ValueError("molecule-teacher-dir and teacher-control must be supplied together")
    if args.molecule_teacher_dir is not None and not (0 < args.lambda_molecule <= 1):
        raise ValueError("G2 molecule convex weight must be in (0, 1]")
    if args.chemical_hardness_beta < 0:
        raise ValueError("chemical-hardness-beta must be nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("ChemAware shared-v2 training requires an available CUDA device")
    required = [args.graph, args.official_checkpoint, args.token_dir / "report.json"]
    if args.molecule_teacher_dir is not None:
        required.append(args.molecule_teacher_dir / "report.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    experiment = "G1_clean_listwise_control" if args.molecule_teacher_dir is None else f"G2_molformer_{args.teacher_control}"
    output_base = args.output_root if args.molecule_teacher_dir is None else args.output_root / str(args.teacher_control)
    output = output_base / f"seed_{args.seed}" / f"fold_{args.outer_fold}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite ChemAware output: {output}")

    seed_everything(args.seed)
    device = torch.device(args.device)
    graph = CandidateGraph(args.graph)
    formal = args.max_train_queries == 0 and args.max_eval_queries == 0
    if formal:
        if not args.preflight.is_file():
            raise FileNotFoundError(args.preflight)
        preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
        if (
            preflight.get("status") != "chemaware_corrected_training_preflight_passed"
            or preflight.get("formal") is not True
            or preflight.get("data_contract") != "train_primary_all_p3_disjoint_v1"
        ):
            raise RuntimeError(
                "formal training requires a corrected train_primary_all P3-disjoint preflight; "
                "historical chemaware_shared_v2_preflight_passed artifacts are not admissible"
            )
        preflight_sha256 = sha256_file(args.preflight)
        token_report = json.loads((args.token_dir / "report.json").read_text(encoding="utf-8"))
        if token_report.get("provenance", {}).get("preflight_sha256") != preflight_sha256:
            raise RuntimeError("token cache belongs to a different formal preflight")
        if args.molecule_teacher_dir is not None:
            teacher_report = json.loads(
                (args.molecule_teacher_dir / "report.json").read_text(encoding="utf-8")
            )
            if teacher_report.get("provenance", {}).get("preflight_sha256") != preflight_sha256:
                raise RuntimeError("molecule teacher belongs to a different formal preflight")
    store = ChemAwareTokenStore(
        args.token_dir, args.graph, args.official_checkpoint, require_formal=formal
    )
    store.require_graph_coverage(graph)
    graph_score_error = store.verify_official_graph_scores(graph)
    if store.dimension != 1024 or store.tokens.shape[2] != 1024:
        raise RuntimeError("formal ChemAware v2 requires full 1024-dimensional official tokens")

    query_fold = formula_folds(graph.query_formula, args.folds, args.fold_seed)
    inner_fold = (args.outer_fold + 1) % args.folds
    train_query = np.flatnonzero(
        (query_fold != args.outer_fold) & (query_fold != inner_fold)
    )
    inner_query = np.flatnonzero(query_fold == inner_fold)
    outer_query = np.flatnonzero(query_fold == args.outer_fold)
    allowed_molecule = split_allowed_molecules(
        graph, args.outer_fold, inner_fold, args.folds, args.fold_seed
    )
    # Full listwise remains meaningful only for groups retaining a negative.
    eligible = []
    for query in train_query:
        left, right = map(int, graph.query_ptr[query:query + 2])
        selected = allowed_molecule[left:right]
        if selected[0] and int(np.sum(selected)) >= 2:
            eligible.append(int(query))
    train_query = np.asarray(eligible, dtype=np.int64)
    if args.max_train_queries:
        train_query = train_query[:args.max_train_queries]
    if args.max_eval_queries:
        inner_query = inner_query[:args.max_eval_queries]
        outer_query = outer_query[:args.max_eval_queries]
    if not len(train_query) or not len(inner_query) or not len(outer_query):
        raise RuntimeError("empty train/inner/outer formula split")

    # Instantiate the deployable adapter before any training-only data branch.
    # For a fixed seed this makes initialization byte-identical in G1/G2.
    adapter = SignedPeakResidualAdapter(
        store.dimension, args.hidden_dim, args.delta_bound,
        args.gate_temperature, args.gate_topk, args.contextual_gate,
        args.global_branch,
    ).to(device)
    initial_adapter_sha256 = state_sha256(adapter.state_dict())
    teacher_store = None
    teacher_graph_embeddings = None
    teacher_graph_observable = None
    teacher_control_audit = None
    if args.molecule_teacher_dir is not None:
        teacher_store = MoleculeTeacherStore(
            args.molecule_teacher_dir, args.graph, graph, require_formal=formal
        )
        teacher_graph_embeddings, teacher_graph_observable, teacher_control_audit = teacher_store.graph_embeddings(
            graph, allowed_molecule, str(args.teacher_control), args.seed + 1009 * args.outer_fold
        )
        effective_queries = 0
        observable_negative_counts = []
        for query in train_query:
            left, right = map(int, graph.query_ptr[query:query + 2])
            selected = np.flatnonzero(allowed_molecule[left:right]) + left
            observed = teacher_graph_observable[selected]
            negatives = int(np.sum(observed[1:])) if bool(observed[0]) else 0
            observable_negative_counts.append(negatives)
            required_negatives = (
                2 if args.chemical_weighting == "relative_centered" else 1
            )
            effective_queries += negatives >= required_negatives
        teacher_control_audit.update({
            "chemical_effect_queries": int(effective_queries),
            "chemical_effect_query_fraction": float(effective_queries / len(train_query)),
            "observable_negative_median": float(np.median(observable_negative_counts)),
        })
        if effective_queries == 0:
            raise RuntimeError("molecule teacher changes no training candidate group")

    identities = graph.query_ik14[train_query].astype(str)
    _, identity_inverse, identity_count = np.unique(
        identities, return_inverse=True, return_counts=True
    )
    query_weight = (1.0 / identity_count[identity_inverse]).astype(np.float32)
    query_weight /= query_weight.mean()
    weight_by_query = {int(query): float(weight) for query, weight in zip(train_query, query_weight)}

    trainable = list(adapter.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    official_encoded = np.asarray(store.official_embeddings, dtype=np.float32).copy()
    encoded = encode_all(adapter, store, device, args.eval_batch_size)
    initial_inner = paired_evaluation(
        encoded, official_encoded, store, graph, inner_query
    )
    if (
        initial_inner["summary"]["corrected"]
        or initial_inner["summary"]["introduced"]
        or not np.array_equal(initial_inner["old_rank"], initial_inner["new_rank"])
    ):
        raise RuntimeError("zero-init ChemAware v2 does not exactly reproduce official ranks")

    best_state = copy.deepcopy(adapter.state_dict())
    best_epoch = 0
    best_utility = 0.0
    history = [{
        "epoch": 0,
        "inner": initial_inner["summary"],
        "selection_utility": 0.0,
        "eligible": True,
    }]
    rng = np.random.default_rng(args.seed)
    print(
        f"[ChemAware-v2 {experiment} fold={args.outer_fold} seed={args.seed}] epoch 0 "
        f"inner={initial_inner['summary']}", flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        order = rng.permutation(train_query)
        component_sum = {
            name: 0.0 for name in ("loss", "listwise", "molecule", "protect", "preserve")
        }
        candidate_molecule_sum = 0
        candidate_pair_sum = 0
        batches = 0
        gradient_audit = None
        gradient_audit_attempts = 0
        started = time.time()
        for left in range(0, len(order), args.batch_queries):
            queries = order[left:left + args.batch_queries]
            batch = build_query_batch(graph, queries, allowed_molecule)
            new_scores, old_scores, preserve = score_batch(
                adapter, store, batch, device
            )
            query_ptr = torch.from_numpy(batch["query_ptr"]).to(device=device, dtype=torch.long)
            weights = torch.tensor(
                [weight_by_query[int(query)] for query in queries],
                device=device, dtype=torch.float32,
            )
            listwise = molecule_listwise_loss(
                new_scores, query_ptr, args.temperature, weights
            )
            protect = protected_margin_loss(
                new_scores, old_scores, query_ptr, args.protect_slack
            )
            if teacher_store is None:
                molecule = new_scores.sum() * 0.0
                retrieval_objective = listwise
            else:
                teacher_values = torch.from_numpy(
                    teacher_graph_embeddings[batch["molecule_index"]]
                ).to(device=device, dtype=torch.float32)
                teacher_observable = torch.from_numpy(
                    teacher_graph_observable[batch["molecule_index"]]
                ).to(device=device, dtype=torch.bool)
                molecule = chemical_weighted_listwise_loss(
                    new_scores, teacher_values, query_ptr,
                    args.molecule_temperature, args.chemical_hardness_beta,
                    teacher_observable, weights, args.chemical_weighting,
                )
                retrieval_objective = (
                    (1.0 - args.lambda_molecule) * listwise
                    + args.lambda_molecule * molecule
                )
                if gradient_audit is None and gradient_audit_attempts < 10:
                    gradient_audit_attempts += 1
                    candidate_audit = gradient_geometry(
                        listwise, molecule - listwise, trainable
                    )
                    if candidate_audit["chemical_minus_clean_gradient_norm"] > 1e-12:
                        gradient_audit = candidate_audit
            loss = (
                args.lambda_listwise * retrieval_objective
                + args.lambda_protect * protect
                + args.lambda_preserve * preserve
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite ChemAware shared-v2 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            for name, value in (
                ("loss", loss), ("listwise", listwise), ("molecule", molecule),
                ("protect", protect), ("preserve", preserve),
            ):
                component_sum[name] += float(value.detach())
            candidate_molecule_sum += len(batch["molecule_ptr"]) - 1
            candidate_pair_sum += len(batch["candidate_rows"])
            batches += 1

        if teacher_store is not None and gradient_audit is None:
            raise RuntimeError(
                "chemical teacher produced no incremental gradient on deployable adapter"
            )

        encoded = encode_all(adapter, store, device, args.eval_batch_size)
        inner = paired_evaluation(encoded, official_encoded, store, graph, inner_query)
        summary = inner["summary"]
        risk_net = (
            summary["corrected"] - args.risk_penalty * summary["introduced"]
        ) / summary["n_queries"]
        near_delta = summary["delta_near_recall1"] or 0.0
        utility = float(risk_net + args.near_selection_weight * near_delta)
        eligible_epoch = (
            summary["preservation_mean"] >= args.minimum_preservation
            and summary["delta_recall1"] >= args.recall1_floor
            and (
                summary["delta_near_recall1"] is None
                or summary["delta_near_recall1"] >= args.near_recall1_floor
            )
        )
        if eligible_epoch and utility > best_utility + 1e-12:
            best_state = copy.deepcopy(adapter.state_dict())
            best_epoch = epoch
            best_utility = utility
        record = {
            "epoch": epoch,
            "train": {name: value / batches for name, value in component_sum.items()},
            "mean_candidate_molecules_per_batch": candidate_molecule_sum / batches,
            "mean_candidate_spectrum_pairs_per_batch": candidate_pair_sum / batches,
            "inner": summary,
            "risk_net_per_query": float(risk_net),
            "selection_utility": utility,
            "eligible": bool(eligible_epoch),
            "gradient_audit": gradient_audit,
            "seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"[ChemAware-v2 {experiment} fold={args.outer_fold} seed={args.seed}] {record}",
            flush=True,
        )

    adapter.load_state_dict(best_state)
    encoded = encode_all(adapter, store, device, args.eval_batch_size)
    outer = paired_evaluation(encoded, official_encoded, store, graph, outer_query)
    output.mkdir(parents=True, exist_ok=False)
    torch.save({
        "status": (
            "chemaware_shared_v2_clean_listwise"
            if teacher_store is None else "chemaware_shared_v2_molecule_teacher"
        ),
        "adapter_state": {key: value.cpu() for key, value in best_state.items()},
        "adapter_config": {
            "embedding_dim": store.dimension,
            "hidden_dim": args.hidden_dim,
            "delta_bound": args.delta_bound,
            "gate_temperature": args.gate_temperature,
            "gate_topk": args.gate_topk,
            "contextual_gate": args.contextual_gate,
            "global_branch": args.global_branch,
        },
        "objective": "complete_split_eligible_molecule_listwise",
        "chemical_supervision": teacher_store is not None,
        "chemical_objective": (
            f"frozen_teacher_candidate_hardness_{args.chemical_weighting}"
            if teacher_store is not None else None
        ),
        "training_only_projector_used": False,
        "teacher_control": args.teacher_control,
        "candidate_inputs_at_inference": False,
        "query_reference_encoder_shared": True,
        "P2b_used": False,
        "outer_fold": args.outer_fold,
        "inner_fold": inner_fold,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "formal": formal,
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "molecule_teacher_report_sha256": (
                sha256_file(args.molecule_teacher_dir / "report.json")
                if args.molecule_teacher_dir is not None else None
            ),
            "script_sha256": sha256_file(Path(__file__)),
        },
    }, output / "adapter.pt")
    np.savez_compressed(
        output / "outer_predictions.npz",
        query=outer["query"], old_rank=outer["old_rank"], new_rank=outer["new_rank"],
    )
    decision = {
        "status": (
            "chemaware_shared_v2_g1_fold_complete"
            if teacher_store is None else "chemaware_shared_v2_g2_fold_complete"
        ),
        "formal": formal,
        "experiment": experiment,
        "chemical_supervision": teacher_store is not None,
        "chemical_objective": (
            f"frozen_teacher_candidate_hardness_{args.chemical_weighting}"
            if teacher_store is not None else None
        ),
        "training_only_projector_used": False,
        "teacher_control": args.teacher_control,
        "teacher_control_audit": teacher_control_audit,
        "seed": args.seed,
        "outer_fold": args.outer_fold,
        "inner_fold": inner_fold,
        "train_queries": int(len(train_query)),
        "train_identities": int(len(np.unique(identities))),
        "initial_adapter_sha256": initial_adapter_sha256,
        "training_query_ledger_sha256": hashlib.sha256(train_query.tobytes()).hexdigest(),
        "allowed_molecule_ledger_sha256": hashlib.sha256(allowed_molecule.tobytes()).hexdigest(),
        "training_contract": {
            "epochs": args.epochs,
            "batch_queries": args.batch_queries,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "delta_bound": args.delta_bound,
            "gate_temperature": args.gate_temperature,
            "gate_topk": args.gate_topk,
            "contextual_gate": args.contextual_gate,
            "global_branch": args.global_branch,
            "temperature": args.temperature,
            "lambda_listwise": args.lambda_listwise,
            "lambda_protect": args.lambda_protect,
            "lambda_preserve": args.lambda_preserve,
            "protect_slack": args.protect_slack,
            "minimum_preservation": args.minimum_preservation,
            "recall1_floor": args.recall1_floor,
            "near_recall1_floor": args.near_recall1_floor,
            "risk_penalty": args.risk_penalty,
            "near_selection_weight": args.near_selection_weight,
            "folds": args.folds,
            "fold_seed": args.fold_seed,
        },
        "chemical_contract": (
            {
                "lambda_molecule": args.lambda_molecule,
                "molecule_temperature": args.molecule_temperature,
                "chemical_hardness_beta": args.chemical_hardness_beta,
                "chemical_weighting": args.chemical_weighting,
                "combination": "convex_mix_with_clean_listwise",
                "training_only_projector_used": False,
            }
            if teacher_store is not None else None
        ),
        "best_epoch": best_epoch,
        "best_inner_utility": best_utility,
        "outer": outer["summary"],
        "history": history,
        "preflight": {
            "exact_graph_coverage": True,
            "official_graph_max_abs_error": graph_score_error,
            "zero_init_exact_rank_reproduction": True,
            "heldout_formula_molecules_excluded_from_training": True,
        },
        "query_reference_encoder_shared": True,
        "candidate_inputs_at_inference": False,
        "P2b_used": False,
        "claim_limit": (
            "clean continuation control only; one formula outer fold; no chemical-attribution "
            "claim and no sealed external claim"
            if teacher_store is None else
            "one G2 formula outer fold; chemical attribution requires paired multifold superiority "
            "over G1 and all pseudo-teacher controls; no stereochemistry or sealed external claim"
        ),
    }
    json_dump(output / "decision.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
