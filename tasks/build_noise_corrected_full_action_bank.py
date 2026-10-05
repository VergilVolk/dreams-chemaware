"""Build mature N actions on every eligible corrected-graph outer-train query.

The registered S3A mechanisms are preserved: candidate-gradient attenuation
at dose 0.50 (prefixes 3..6) and confounder-role dropout at dose 1.00
(prefixes 1..5).  Paths are re-mined in the exact initialization geometry.
No target/control outcome is encoded or ranked by this builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from audit_noise_v3_candidate_gradient import top_reference_rows
from noise_final_core import CandidateGraph, load_embedding_cache, sha256_file, stable_fold
from noise_v3_core import (
    CONFOUNDER_ONLY, IDENTITY_ONLY, ROLE_NAMES, attenuate_and_renormalize,
    candidate_peak_roles_from_mz, candidate_representatives,
    matched_control_tokens_strict_excluding, rank_gradient_targets,
    rank_role_targets, stable_seed,
)
from train_e1_identity import load_base_model, torch_load_compat
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows, forward_embeddings


REGISTERED = {
    "candidate_gradient": {"attenuation": 0.50, "maximum_steps": 6, "publish_steps": (3, 4, 5, 6)},
    "role_confounder": {"attenuation": 1.00, "maximum_steps": 5, "publish_steps": (1, 2, 3, 4, 5)},
}

REGISTERED_FORMAL_ACTION_BANK_CONFIGURATION: dict[str, object] = {
    "formula_fold_seed": 20260825,
    "n_highest_peaks": 100,
    "top_k_negatives": 5,
    "softmax_temperature": 0.10,
    "fragment_tolerance": 0.02,
    "control_repeats": 2,
    "seed": 20260906,
    "max_queries": 0,
    "amp": False,
}


def _validate_registered_formal_configuration(args: argparse.Namespace) -> None:
    if args.max_queries != 0:
        return
    mismatches = {}
    for name, expected in REGISTERED_FORMAL_ACTION_BANK_CONFIGURATION.items():
        observed = getattr(args, name)
        equal = (
            bool(np.isclose(float(observed), float(expected), rtol=1e-12, atol=1e-12))
            if isinstance(expected, float) else
            observed == expected and type(observed) is type(expected)
        )
        if not equal:
            mismatches[name] = {"observed": observed, "expected": expected}
    if mismatches:
        raise RuntimeError(
            "formal mature-N action-bank configuration drifted: "
            + json.dumps(mismatches, sort_keys=True)
        )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-student-checkpoint", type=Path, default=None)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--selection-batch-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--top-k-negatives", type=int, default=5)
    parser.add_argument("--softmax-temperature", type=float, default=0.10)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--control-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--max-queries", type=int, default=0, help="Development subset; zero means all outer-train queries")
    parser.add_argument(
        "--development-sample-seed", type=int, default=20260906,
        help="Fixed random query sample used only when max-queries is nonzero",
    )
    parser.add_argument(
        "--development-query-scope", choices=("all", "official_errors"), default="all",
        help="Development-only sampling stratum; formal max-queries=0 always uses all outer-train queries",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def first_unused(values: np.ndarray, used: set[int]) -> int | None:
    for value in values:
        if int(value) not in used:
            return int(value)
    return None


def current_context(
    graph: CandidateGraph, query: int, vector: np.ndarray,
    embeddings: np.ndarray, row_index: dict[int, int],
    tensors: dict[int, torch.Tensor], state: torch.Tensor,
    top_k_negatives: int, tolerance: float,
) -> tuple[object, np.ndarray]:
    _, rows, ptr, _ = graph.query_block(query)
    candidate = embeddings[[row_index[int(row)] for row in rows]]
    scores = candidate @ np.asarray(vector, dtype=np.float32)
    representatives = candidate_representatives(scores, rows, ptr, top_k_negatives)
    positive_rows = top_reference_rows(scores, rows, ptr, 0, 3)
    hard_row = int(representatives.negative_rows[0])
    hard_pair = np.flatnonzero(rows == hard_row)
    if len(hard_pair) != 1:
        raise RuntimeError("hard-negative representative is not unique")
    hard_molecule = int(np.searchsorted(ptr[1:], hard_pair[0], side="right"))
    negative_rows = top_reference_rows(scores, rows, ptr, hard_molecule, 3)

    def mz_union(reference_rows: np.ndarray) -> np.ndarray:
        pieces = []
        for row in reference_rows:
            tensor = tensors[int(row)]
            valid = (tensor[1:, 0] > 0) & (tensor[1:, 1] > 0)
            pieces.append(tensor[1:, 0][valid].numpy())
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)

    roles = candidate_peak_roles_from_mz(
        state, mz_union(positive_rows), mz_union(negative_rows), tolerance,
    )
    return representatives, roles


def load_initial_model(args: argparse.Namespace, device: torch.device):
    model, kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    provenance: dict[str, object] = {
        "initialization": kind,
        "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
    }
    if args.initial_student_checkpoint is not None:
        package = torch_load_compat(args.initial_student_checkpoint, map_location="cpu")
        decision_path = args.initial_student_checkpoint.parent / "decision.json"
        if not decision_path.is_file():
            raise FileNotFoundError(decision_path)
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if (
            package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
            or package.get("inference_clean_only") is not True
            or package.get("P2b_used") is not False
            or int(package.get("outer_fold", -1)) != args.outer_fold
            or decision.get("status") != "noise_final_e4a_direct_augmentation_complete"
            or decision.get("formal") is not True
            or decision.get("pass_to_multifold") is not True
        ):
            raise RuntimeError("mature E4 initialization contract failed")
        model.load_state_dict(package["model_state"], strict=True)
        provenance.update({
            "initialization": "mature_e4_current_geometry",
            "initial_student_checkpoint_sha256": sha256_file(args.initial_student_checkpoint),
            "initial_decision_sha256": sha256_file(decision_path),
            "historical_held_ledger_reused": False,
        })
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, provenance


def action_id(query_row: int, selector: str, attenuation: float, step: int) -> str:
    payload = f"corrected-v1|{query_row}|{selector}|{attenuation:.2f}|{step}".encode()
    return "NC-" + hashlib.sha256(payload).hexdigest()[:24]


def publish_rows(
    graph: CandidateGraph, query: int, selector: str, attenuation: float,
    targets: list[int], roles: list[np.ndarray], hard_rows: list[int],
    controls: list[list[int]], publish_steps: tuple[int, ...], fold: int,
) -> list[dict[str, object]]:
    output = []
    for step in publish_steps:
        if len(targets) < step:
            continue
        target_path = ",".join(map(str, targets[:step]))
        control_complete = bool(controls) and all(len(path) >= step for path in controls)
        control_paths = (
            ";".join(",".join(map(str, path[:step])) for path in controls)
            if control_complete else ""
        )
        output.append({
            "action_id": action_id(int(graph.query_row[query]), selector, attenuation, step),
            "query_index": int(query), "query_row": int(graph.query_row[query]),
            "query_ik14": str(graph.query_ik14[query]),
            "query_formula": str(graph.query_formula[query]),
            "has_near": bool(graph.query_has_near[query]), "formula_fold": int(fold),
            "selector": selector, "attenuation": float(attenuation), "step": int(step),
            "target_path": target_path, "matched_control_paths": control_paths,
            "matched_controls_complete": control_complete,
            "hard_negative_row": int(hard_rows[step - 1]),
            "target_role": str(ROLE_NAMES[roles[step - 1][targets[step - 1]]]),
        })
    return output


def main() -> None:
    args = arguments()
    _validate_registered_formal_configuration(args)
    started = time.time()
    if args.outer_fold not in range(5) or args.max_queries < 0:
        raise ValueError("outer-fold must be 0..4 and max-queries nonnegative")
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    graph_path = args.graph_dir / "candidate_graph.npz"
    cache_path = args.graph_dir / "official_embeddings.npz"
    report_path = args.graph_dir / "report.json"
    for path in (graph_path, cache_path, report_path, args.data, args.official_checkpoint, args.architecture_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    graph_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        graph_report.get("formal_training_authorized") is not True
        or graph_report.get("provenance", {}).get("candidate_graph_sha256") != sha256_file(graph_path)
        or graph_report.get("provenance", {}).get("embedding_cache_sha256") != sha256_file(cache_path)
        or graph_report.get("contracts", {}).get("P3_consumed") is not False
    ):
        raise RuntimeError("corrected graph provenance contract failed")
    graph = CandidateGraph(graph_path)
    formula_fold = np.asarray([
        stable_fold(str(value), 5, args.formula_fold_seed) for value in graph.query_formula
    ], dtype=np.int8)
    queries = np.flatnonzero(formula_fold != args.outer_fold)
    if args.max_queries:
        if args.development_query_scope == "official_errors":
            official_molecule = np.maximum.reduceat(
                graph.features[:, graph.dreams_column], graph.molecule_ptr[:-1],
            )
            official_rank = np.asarray([
                1 + int(np.sum(
                    official_molecule[int(left) + 1:int(right)]
                    >= official_molecule[int(left)]
                ))
                for left, right in zip(graph.query_ptr[:-1], graph.query_ptr[1:])
            ], dtype=np.int16)
            queries = queries[official_rank[queries] > 1]
        rng = np.random.default_rng(args.development_sample_seed)
        queries = np.sort(rng.choice(
            queries, size=min(args.max_queries, len(queries)), replace=False,
        ))
    needed = set(map(int, graph.query_row[queries]))
    for query in queries:
        _, rows, _, _ = graph.query_block(int(query))
        needed.update(map(int, rows))
    store = SpectrumStore(args.data, np.asarray(sorted(needed), dtype=np.int64), args.n_highest_peaks)
    tensors = {int(row): store.tensor[index] for index, row in enumerate(store.rows)}
    device = torch.device(args.device)
    model, model_provenance = load_initial_model(args, device)
    current_embeddings = encode_rows(
        model, store, store.rows, device, args.encode_batch_size, args.amp,
        "corrected-action-bank-clean",
    )
    row_index = {int(row): index for index, row in enumerate(store.rows)}
    rows: list[dict[str, object]] = []
    eligibility: dict[str, dict[str, int]] = {}
    for selector, specification in REGISTERED.items():
        attenuation = float(specification["attenuation"])
        maximum_steps = int(specification["maximum_steps"])
        states = {int(query): tensors[int(graph.query_row[query])].clone() for query in queries}
        paths = {int(query): [] for query in queries}
        role_history = {int(query): [] for query in queries}
        hard_history = {int(query): [] for query in queries}
        active = set(map(int, queries))
        for step in range(1, maximum_steps + 1):
            ordered = np.asarray(sorted(active), dtype=np.int64)
            for left in range(0, len(ordered), args.selection_batch_size):
                local = ordered[left:left + args.selection_batch_size]
                block = torch.stack([states[int(query)] for query in local]).to(device)
                block.requires_grad_(selector == "candidate_gradient")
                current = forward_embeddings(model, block, False)
                current_np = current.detach().float().cpu().numpy()
                contexts = [current_context(
                    graph, int(query), vector, current_embeddings, row_index,
                    tensors, states[int(query)], args.top_k_negatives,
                    args.fragment_tolerance,
                ) for query, vector in zip(local, current_np)]
                gradients = None
                if selector == "candidate_gradient":
                    positive = torch.as_tensor(np.stack([
                        current_embeddings[row_index[int(context[0].positive_row)]] for context in contexts
                    ]), device=device, dtype=current.dtype)
                    max_k = max(len(context[0].negative_rows) for context in contexts)
                    negatives = torch.zeros(
                        (len(contexts), max_k, current_embeddings.shape[1]),
                        device=device, dtype=current.dtype,
                    )
                    valid = torch.zeros(
                        (len(contexts), max_k), device=device, dtype=torch.bool,
                    )
                    for context_index, context in enumerate(contexts):
                        negative_rows = context[0].negative_rows
                        negatives[context_index, :len(negative_rows)] = torch.as_tensor(
                            np.stack([
                                current_embeddings[row_index[int(row)]]
                                for row in negative_rows
                            ]), device=device, dtype=current.dtype,
                        )
                        valid[context_index, :len(negative_rows)] = True
                    positive_score = torch.sum(current * positive, dim=1)
                    negative_score = torch.einsum("bd,bkd->bk", current, negatives)
                    negative_score = negative_score.masked_fill(~valid, -1e9)
                    weight = torch.softmax(negative_score / args.softmax_temperature, dim=1).detach()
                    objective = positive_score - torch.sum(weight * negative_score, dim=1)
                    gradients = torch.autograd.grad(objective.sum(), block)[0][:, :, 1]
                for offset, query_value in enumerate(local):
                    query = int(query_value)
                    representatives, roles = contexts[offset]
                    if selector == "candidate_gradient":
                        ranked = rank_gradient_targets(
                            states[query], gradients[offset].detach().float().cpu().numpy(),
                            roles, attenuation, args.n_highest_peaks, True,
                        )
                    else:
                        ranked = rank_role_targets(
                            states[query], roles, CONFOUNDER_ONLY, args.n_highest_peaks,
                        )
                    target = first_unused(ranked, set(paths[query]))
                    if target is None:
                        active.discard(query)
                        continue
                    if target <= 0 or int(roles[target]) == IDENTITY_ONLY:
                        raise RuntimeError("identity/precursor protection failed")
                    paths[query].append(target)
                    role_history[query].append(roles.copy())
                    hard_history[query].append(int(representatives.negative_rows[0]))
                    states[query] = attenuate_and_renormalize(states[query], target, attenuation)
            print(f"[full action bank] {selector} step={step} active={len(active):,}", flush=True)
        published_queries = 0
        for query in queries:
            query = int(query)
            target_state = tensors[int(graph.query_row[query])].clone()
            targets = paths[query]
            controls = [[] for _ in range(args.control_repeats)]
            blocked = set(targets)
            complete = 0
            for index, target in enumerate(targets):
                selected = matched_control_tokens_strict_excluding(
                    target_state, target, role_history[query][index], args.control_repeats,
                    stable_seed(args.seed, int(graph.query_row[query]), selector, attenuation, index),
                    excluded=blocked | {token for path in controls for token in path},
                )
                if len(selected) != args.control_repeats:
                    break
                for repeat, token in enumerate(selected):
                    controls[repeat].append(int(token))
                target_state = attenuate_and_renormalize(target_state, target, attenuation)
                complete += 1
            local_rows = publish_rows(
                graph, query, selector, attenuation, targets,
                role_history[query], hard_history[query], controls,
                tuple(specification["publish_steps"]), int(formula_fold[query]),
            )
            rows.extend(local_rows)
            published_queries += bool(local_rows)
        eligibility[selector] = {
            "eligible_queries": int(published_queries),
            "ineligible_queries": int(len(queries) - published_queries),
            "published_actions": int(sum(row["selector"] == selector for row in rows)),
            "actions_with_complete_matched_controls": int(sum(
                row["selector"] == selector and bool(row["matched_controls_complete"])
                for row in rows
            )),
        }
    frame = pd.DataFrame(rows)
    forbidden = {"target_rank", "control_rank", "corrected", "introduced", "teacher_margin"}
    if frame.empty or forbidden & set(frame.columns) or frame.action_id.duplicated().any():
        raise RuntimeError("action bank is empty, duplicated, or outcome-contaminated")
    report = {
        "status": "noise_corrected_full_action_bank_complete",
        "formal": bool(args.max_queries == 0),
        "formal_training_authorized": bool(args.max_queries == 0),
        "outer_formula_fold": args.outer_fold,
        "configuration": {
            name: getattr(args, name)
            for name in REGISTERED_FORMAL_ACTION_BANK_CONFIGURATION
        },
        "source_queries": int(len(queries)),
        "development_query_scope": (
            args.development_query_scope if args.max_queries else "all"
        ),
        "action_rows": int(len(frame)),
        "action_queries": int(frame.query_index.nunique()),
        "action_identities": int(frame.query_ik14.nunique()),
        "action_formulas": int(frame.query_formula.nunique()),
        "eligibility": eligibility,
        "registered_actions": REGISTERED,
        "contracts": {
            "registered_formal_action_bank_configuration_verified": bool(
                args.max_queries == 0
            ),
            "outer_held_formulas_published": False,
            "current_geometry_remined_each_step": True,
            "candidate_context_recomputed_each_step": True,
            "target_action_eligibility_independent_of_matched_controls": True,
            "available_matched_controls_complete_and_distinct": True,
            "action_multiplicity_is_not_training_dose": True,
            "action_outcomes_computed": False,
            "teacher_embedding_or_margin_used": False,
            "P2b": "forbidden", "P3_consumed": False,
        },
        "model_provenance": model_provenance,
        "provenance": {
            "candidate_graph_sha256": sha256_file(graph_path),
            "embedding_cache_sha256": sha256_file(cache_path),
            "graph_report_sha256": sha256_file(report_path),
            "hdf5_sha256": sha256_file(args.data),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "runtime_seconds": time.time() - started,
        "claim_limit": "Outcome-free action construction only; no encoder improvement is claimed.",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_full_action_bank_", dir=args.output_dir.parent))
    try:
        frame.to_csv(staging / "training_actions.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
