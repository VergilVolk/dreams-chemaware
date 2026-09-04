"""Cross-fit ChemAware discovery adapters on a molecule-disjoint confirmation graph.

Each confirmation query is encoded by the discovery adapter whose outer fold
matches the query formula.  That adapter therefore excluded the formula from
both fitting and inner model selection.  The confirmation cohort remains a
biased, non-sealed mechanism check rather than a formal benchmark.
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
    ChemAwareTokenStore, formula_folds, paired_evaluation,
)
from dreams.models.chem_aware.shared_embedding_v2 import SignedPeakResidualAdapter  # noqa: E402
from noise_final_core import CandidateGraph, sha256_file  # noqa: E402
import pilot_multilevel_factor_activations as multi  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-graph", type=Path, required=True)
    parser.add_argument("--confirmation-graph", type=Path, required=True)
    parser.add_argument("--confirmation-token-dir", type=Path, required=True)
    parser.add_argument("--training-token-dir", type=Path)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--adapter-set",
        action="append",
        required=True,
        help="LABEL=adapter/root containing seed_N/fold_K/adapter.pt",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1701)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--calibrate-scale", action="store_true")
    parser.add_argument("--scale-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_adapter(path: Path, device: torch.device) -> tuple[SignedPeakResidualAdapter, dict]:
    package = multi.torch_load_compat(path, map_location="cpu")
    config = package.get("adapter_config", {})
    required = {"embedding_dim", "hidden_dim", "delta_bound"}
    if required - set(config) or not isinstance(package.get("adapter_state"), dict):
        raise RuntimeError(f"malformed ChemAware adapter: {path}")
    adapter = SignedPeakResidualAdapter(
        int(config["embedding_dim"]), int(config["hidden_dim"]),
        float(config["delta_bound"]), float(config.get("gate_temperature", 1.0)),
        int(config.get("gate_topk", 0)), bool(config.get("contextual_gate", False)),
        bool(config.get("global_branch", False)),
    ).to(device)
    adapter.load_state_dict(package["adapter_state"], strict=True)
    adapter.eval()
    return adapter, package


def paired_summary(old_rank: np.ndarray, new_rank: np.ndarray, near: np.ndarray) -> dict:
    old_correct = old_rank == 1
    new_correct = new_rank == 1
    output = {
        "n_queries": int(len(old_rank)),
        "baseline_recall1": float(np.mean(old_correct)),
        "recall1": float(np.mean(new_correct)),
        "delta_recall1": float(np.mean(new_correct) - np.mean(old_correct)),
        "baseline_mrr": float(np.mean(1.0 / old_rank)),
        "mrr": float(np.mean(1.0 / new_rank)),
        "delta_mrr": float(np.mean(1.0 / new_rank) - np.mean(1.0 / old_rank)),
        "corrected": int(np.sum(~old_correct & new_correct)),
        "introduced": int(np.sum(old_correct & ~new_correct)),
        "risk_net_lambda2": int(
            np.sum(~old_correct & new_correct) - 2 * np.sum(old_correct & ~new_correct)
        ),
        "near_n": int(np.sum(near)),
    }
    if np.any(near):
        output.update({
            "baseline_near_recall1": float(np.mean(old_correct[near])),
            "near_recall1": float(np.mean(new_correct[near])),
            "delta_near_recall1": float(
                np.mean(new_correct[near]) - np.mean(old_correct[near])
            ),
        })
    return output


def encode_components(
    adapter: SignedPeakResidualAdapter,
    store: ChemAwareTokenStore,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    official_blocks, delta_blocks = [], []
    with torch.inference_mode():
        for left in range(0, len(store.rows), batch_size):
            rows = store.rows[left:left + batch_size]
            official, _, delta, _, _ = store.adapt(adapter, rows, device)
            official_blocks.append(official.cpu().numpy())
            delta_blocks.append(delta.cpu().numpy())
    return np.concatenate(official_blocks), np.concatenate(delta_blocks)


def scaled_embedding(official: np.ndarray, delta: np.ndarray, scale: float) -> np.ndarray:
    if scale == 0:
        return official.copy()
    values = official + float(scale) * delta
    values /= np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
    return values.astype(np.float32, copy=False)


def selection_utility(summary: dict) -> float:
    risk = (summary["corrected"] - 2 * summary["introduced"]) / summary["n_queries"]
    return float(risk + 0.25 * (summary.get("delta_near_recall1") or 0.0))


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite confirmation output: {args.output_dir}")
    if args.folds < 3 or args.batch_size < 1:
        raise ValueError("invalid cross-fit configuration")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    scales = sorted({float(value) for value in args.scale_grid.split(",")})
    if not scales or scales[0] < 0 or scales[-1] > 1 or 0.0 not in scales:
        raise ValueError("scale grid must contain 0 and stay within [0, 1]")
    if args.calibrate_scale and args.training_token_dir is None:
        raise ValueError("--calibrate-scale requires --training-token-dir")
    sets: dict[str, Path] = {}
    for value in args.adapter_set:
        if "=" not in value:
            raise ValueError("--adapter-set must be LABEL=ROOT")
        label, raw_root = value.split("=", 1)
        if not label or label in sets:
            raise ValueError(f"duplicate or empty adapter label: {label}")
        sets[label] = Path(raw_root)

    training = CandidateGraph(args.training_graph)
    confirmation = CandidateGraph(args.confirmation_graph)
    training_identities = set(training.molecule_ik14.astype(str))
    confirmation_identities = set(confirmation.molecule_ik14.astype(str))
    identity_overlap = training_identities & confirmation_identities
    if identity_overlap:
        raise RuntimeError(
            f"discovery/confirmation molecule identity overlap: {len(identity_overlap)}"
        )
    store = ChemAwareTokenStore(
        args.confirmation_token_dir, args.confirmation_graph,
        args.official_checkpoint, require_formal=False,
    )
    store.require_graph_coverage(confirmation)
    graph_error = store.verify_official_graph_scores(confirmation)
    official = np.asarray(store.official_embeddings, dtype=np.float32).copy()
    query_fold = formula_folds(confirmation.query_formula, args.folds, args.fold_seed)
    training_store = None
    training_official = None
    training_query_fold = None
    if args.calibrate_scale:
        training_store = ChemAwareTokenStore(
            args.training_token_dir, args.training_graph,
            args.official_checkpoint, require_formal=False,
        )
        training_store.require_graph_coverage(training)
        training_store.verify_official_graph_scores(training)
        training_official = np.asarray(
            training_store.official_embeddings, dtype=np.float32
        ).copy()
        training_query_fold = formula_folds(
            training.query_formula, args.folds, args.fold_seed
        )
    device = torch.device(args.device)
    set_reports = {}
    args.output_dir.mkdir(parents=True)
    expected_training_hash = sha256_file(args.training_graph)
    for label, root in sets.items():
        query_blocks = []
        old_blocks = []
        new_blocks = []
        fold_reports = []
        common_contract = None
        for fold in range(args.folds):
            path = root / f"seed_{args.seed}" / f"fold_{fold}" / "adapter.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            adapter, package = load_adapter(path, device)
            if (
                int(package.get("outer_fold", -1)) != fold
                or int(package.get("seed", -1)) != args.seed
                or package.get("query_reference_encoder_shared") is not True
                or package.get("candidate_inputs_at_inference") is not False
                or package.get("P2b_used") is not False
            ):
                raise RuntimeError(f"invalid shared-encoder contract: {path}")
            if package.get("provenance", {}).get("graph_sha256") != expected_training_hash:
                raise RuntimeError(f"adapter was not trained on the declared graph: {path}")
            contract = json.dumps(
                {
                    "config": package.get("adapter_config"),
                    "objective": package.get("objective"),
                    "chemical_supervision": package.get("chemical_supervision"),
                    "teacher_control": package.get("teacher_control"),
                },
                sort_keys=True,
            )
            if common_contract is None:
                common_contract = contract
            elif contract != common_contract:
                raise RuntimeError(f"adapter set {label} mixes incompatible contracts")
            selected_scale = 1.0
            scale_audit = []
            if args.calibrate_scale:
                train_official_component, train_delta = encode_components(
                    adapter, training_store, device, args.batch_size
                )
                if not np.allclose(train_official_component, training_official, atol=1e-7):
                    raise RuntimeError("training cache official embeddings changed during scaling")
                inner_fold = (fold + 1) % args.folds
                inner_queries = np.flatnonzero(training_query_fold == inner_fold)
                best_utility = 0.0
                selected_scale = 0.0
                for scale in scales:
                    scaled = scaled_embedding(training_official, train_delta, scale)
                    candidate = paired_evaluation(
                        scaled, training_official, training_store, training, inner_queries
                    )
                    candidate_summary = candidate["summary"]
                    utility = selection_utility(candidate_summary)
                    eligible = (
                        candidate_summary["preservation_mean"] >= 0.995
                        and candidate_summary["delta_recall1"] >= -5e-4
                        and (
                            candidate_summary["delta_near_recall1"] is None
                            or candidate_summary["delta_near_recall1"] >= -1e-3
                        )
                    )
                    scale_audit.append({
                        "scale": scale,
                        "utility": utility,
                        "eligible": bool(eligible),
                        **candidate_summary,
                    })
                    if eligible and utility > best_utility + 1e-12:
                        best_utility = utility
                        selected_scale = scale
            confirmation_official_component, confirmation_delta = encode_components(
                adapter, store, device, args.batch_size
            )
            if not np.allclose(confirmation_official_component, official, atol=1e-7):
                raise RuntimeError("confirmation cache official embeddings changed during scaling")
            adapted = scaled_embedding(official, confirmation_delta, selected_scale)
            queries = np.flatnonzero(query_fold == fold)
            result = paired_evaluation(adapted, official, store, confirmation, queries)
            query_blocks.append(result["query"])
            old_blocks.append(result["old_rank"])
            new_blocks.append(result["new_rank"])
            fold_reports.append({
                "fold": fold,
                "adapter_sha256": sha256_file(path),
                "best_epoch": package.get("best_epoch"),
                "selected_residual_scale": selected_scale,
                "scale_selection_audit": scale_audit,
                **result["summary"],
            })
        query = np.concatenate(query_blocks)
        old_rank = np.concatenate(old_blocks)
        new_rank = np.concatenate(new_blocks)
        order = np.argsort(query)
        query, old_rank, new_rank = query[order], old_rank[order], new_rank[order]
        if not np.array_equal(query, np.arange(confirmation.n_queries)):
            raise RuntimeError(f"cross-fit adapter set {label} does not cover each query once")
        summary = paired_summary(old_rank, new_rank, confirmation.query_has_near)
        np.savez_compressed(
            args.output_dir / f"{label}_predictions.npz",
            query=query, old_rank=old_rank, new_rank=new_rank,
        )
        set_reports[label] = {"summary": summary, "folds": fold_reports}
    report = {
        "status": "chemaware_shared_v2_crossfit_confirmation_complete",
        "formal": False,
        "identity_disjoint": True,
        "training_identities": len(training_identities),
        "confirmation_identities": len(confirmation_identities),
        "identity_overlap": 0,
        "confirmation_queries": confirmation.n_queries,
        "official_graph_max_abs_error": graph_error,
        "residual_scale_calibrated_on_discovery_inner_fold": bool(args.calibrate_scale),
        "scale_grid": scales,
        "sets": set_reports,
        "claim_limit": (
            "molecule-disjoint cached-cohort confirmation only; biased gallery, no sealed "
            "benchmark and no chemical-attribution claim"
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
