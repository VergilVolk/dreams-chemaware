"""Route mature multi-step N actions in their exact E8 initialization geometry.

The action bank is outcome-free.  This audit materializes every target path and
all available matched-control paths, scores them on the corrected molecule-max
candidate boundary, and routes only outer-training formulas.  Missing matched
controls fall back to the detached clean view; they never delete a valid target
action from the audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from noise_corrected_action_routing import RoutingThresholds, route_action
from noise_corrected_action_routing_v3 import score_candidate_boundary
from noise_corrected_action_panel import lossless_selector_frontier
from noise_final_core import CandidateGraph, sha256_file
from noise_v3_core import attenuate_sequence
from train_e1_identity import load_base_model, torch_load_compat
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows, forward_embeddings, parse_path


REGISTERED_FORMAL_N_ROUTE_CONFIGURATION: dict[str, object] = {
    "paired_advantage_threshold": 0.01,
    "harm_margin_threshold": 0.01,
    "robustness_slack": 0.005,
    "maximum_corrective_frontier": 16,
    "maximum_harmful_frontier": 8,
    "maximum_robust_frontier": 8,
    "n_highest_peaks": 100,
    "amp": False,
}


def _validate_registered_formal_configuration(
    args: argparse.Namespace,
    *,
    formal: bool,
) -> None:
    if not formal:
        return
    mismatches = {}
    for name, expected in REGISTERED_FORMAL_N_ROUTE_CONFIGURATION.items():
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
            "formal N route configuration drifted: "
            + json.dumps(mismatches, sort_keys=True)
        )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--action-bank-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-student-checkpoint", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--paired-advantage-threshold", type=float, default=0.01)
    parser.add_argument("--harm-margin-threshold", type=float, default=0.01)
    parser.add_argument("--robustness-slack", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--maximum-corrective-frontier", type=int, default=16)
    parser.add_argument("--maximum-harmful-frontier", type=int, default=8)
    parser.add_argument("--maximum-robust-frontier", type=int, default=8)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_control_paths(value: object) -> tuple[tuple[int, ...], ...]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    paths = tuple(parse_path(piece) for piece in text.split(";") if piece.strip())
    if any(not path for path in paths):
        raise ValueError("matched control path contains an empty sequence")
    return paths


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if (
        args.query_chunk_size < 1
        or min(
            args.maximum_corrective_frontier,
            args.maximum_harmful_frontier,
            args.maximum_robust_frontier,
        ) < 1
    ):
        raise ValueError("N routing chunk/frontier limits must be positive")
    graph_path = args.graph_dir / "candidate_graph.npz"
    graph_report_path = args.graph_dir / "report.json"
    action_path = args.action_bank_dir / "training_actions.csv.gz"
    action_report_path = args.action_bank_dir / "report.json"
    initial_decision_path = args.initial_student_checkpoint.parent / "decision.json"
    required = (
        graph_path, graph_report_path, action_path, action_report_path, args.data,
        args.official_checkpoint, args.architecture_checkpoint,
        args.initial_student_checkpoint, initial_decision_path,
    )
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    action_report = json.loads(action_report_path.read_text(encoding="utf-8"))
    formal_route = action_report.get("formal_training_authorized") is True
    _validate_registered_formal_configuration(args, formal=formal_route)
    if graph_report.get("formal_training_authorized") is not True:
        raise RuntimeError("corrected graph is not training-authorized")
    if (
        action_report.get("status") != "noise_corrected_full_action_bank_complete"
        or action_report.get("outer_formula_fold") != args.outer_fold
        or action_report.get("contracts", {}).get("action_outcomes_computed") is not False
        or action_report.get("contracts", {}).get("target_action_eligibility_independent_of_matched_controls") is not True
        or action_report.get("contracts", {}).get("P3_consumed") is not False
        or action_report.get("contracts", {}).get(
            "registered_formal_action_bank_configuration_verified"
        ) is not True
    ):
        raise RuntimeError("outcome-free N action-bank contract failed")
    for key, path in (
        ("candidate_graph_sha256", graph_path),
        ("graph_report_sha256", graph_report_path),
    ):
        if action_report.get("provenance", {}).get(key) != sha256_file(path):
            raise RuntimeError(f"N action-bank provenance drifted: {key}")
    if (
        action_report.get("model_provenance", {}).get("initial_student_checkpoint_sha256")
        != sha256_file(args.initial_student_checkpoint)
    ):
        raise RuntimeError("N action bank and router initialization differ")

    graph = CandidateGraph(graph_path)
    actions = pd.read_csv(action_path, low_memory=False)
    required_columns = {
        "action_id", "query_index", "query_row", "query_ik14", "query_formula",
        "formula_fold", "selector", "attenuation", "step", "target_path",
        "matched_control_paths",
    }
    if missing := required_columns - set(actions.columns):
        raise RuntimeError(f"N action table misses {sorted(missing)}")
    if actions.action_id.astype(str).duplicated().any():
        raise RuntimeError("N action IDs are not unique")
    if actions.formula_fold.eq(args.outer_fold).any():
        raise RuntimeError("outer-held N action reached router")
    query = actions.query_index.to_numpy(np.int64)
    if not np.array_equal(actions.query_row.to_numpy(np.int64), graph.query_row[query]):
        raise RuntimeError("N action row/query alignment failed")
    if not np.array_equal(actions.query_ik14.astype(str).to_numpy(), graph.query_ik14[query]):
        raise RuntimeError("N action identity alignment failed")
    if not np.array_equal(actions.query_formula.astype(str).to_numpy(), graph.query_formula[query]):
        raise RuntimeError("N action formula alignment failed")

    needed: set[int] = set(map(int, actions.query_row))
    for query_value in np.unique(query):
        _, rows, _, _ = graph.query_block(int(query_value))
        needed.update(map(int, rows))
    reachable = np.asarray(sorted(needed), dtype=np.int64)
    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, _ = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device,
        args.n_highest_peaks,
    )
    package = torch_load_compat(args.initial_student_checkpoint, map_location="cpu")
    initial_decision = json.loads(initial_decision_path.read_text(encoding="utf-8"))
    if (
        package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
        or package.get("inference_clean_only") is not True
        or package.get("P2b_used") is not False
        or int(package.get("outer_fold", -1)) != args.outer_fold
        or initial_decision.get("formal") is not True
    ):
        raise RuntimeError("mature E4/E8 initialization contract failed")
    model.load_state_dict(package["model_state"], strict=True)
    model.eval()
    embeddings = encode_rows(
        model, store, reachable, device, args.batch_size, args.amp,
        "corrected-N-router-initial",
    )
    embedding_index = {int(row): index for index, row in enumerate(reachable)}

    thresholds = RoutingThresholds(
        paired_advantage=args.paired_advantage_threshold,
        harm_margin=args.harm_margin_threshold,
        robustness_slack=args.robustness_slack,
    )
    frontier_frames: list[pd.DataFrame] = []
    frontier_action_chunks: list[np.ndarray] = []
    frontier_control_chunks: list[np.ndarray] = []
    per_query_frames: list[pd.DataFrame] = []
    route_counts: Counter[str] = Counter()
    control_counts: Counter[str] = Counter()
    total_actions = 0
    maximum_resident_spectra = 0
    maximum_resident_embedding_bytes = 0
    unique_queries = np.unique(query)
    for chunk_left in range(0, len(unique_queries), args.query_chunk_size):
        query_chunk = unique_queries[chunk_left:chunk_left + args.query_chunk_size]
        action_chunk = actions.loc[actions.query_index.isin(set(map(int, query_chunk)))]
        spectra: list[torch.Tensor] = []
        layouts: list[dict[str, object]] = []
        for row in action_chunk.itertuples(index=False):
            clean = store.one(int(row.query_row))
            action_index = len(spectra)
            spectra.append(attenuate_sequence(
                clean, parse_path(row.target_path), float(row.attenuation),
            ))
            controls = parse_control_paths(row.matched_control_paths)
            control_indices = []
            for path in controls:
                control_indices.append(len(spectra))
                spectra.append(attenuate_sequence(clean, path, float(row.attenuation)))
            layouts.append({
                "row": row, "action_index": action_index,
                "control_indices": control_indices,
            })
        maximum_resident_spectra = max(maximum_resident_spectra, len(spectra))
        encoded = np.empty((len(spectra), embeddings.shape[1]), dtype=np.float32)
        maximum_resident_embedding_bytes = max(
            maximum_resident_embedding_bytes, int(encoded.nbytes),
        )
        with torch.inference_mode():
            for left in range(0, len(spectra), args.batch_size):
                right = min(left + args.batch_size, len(spectra))
                encoded[left:right] = forward_embeddings(
                    model, torch.stack(spectra[left:right]).to(device), args.amp,
                ).float().cpu().numpy()

        records: list[dict[str, object]] = []
        chunk_action_spectra: list[np.ndarray] = []
        chunk_control_spectra: list[np.ndarray] = []
        for layout in layouts:
            row = layout["row"]
            query_index = int(row.query_index)
            _, candidate_rows, ptr, _ = graph.query_block(query_index)
            candidate = embeddings[[embedding_index[int(value)] for value in candidate_rows]]

            def score(vector: np.ndarray):
                return score_candidate_boundary(
                    candidate_rows, ptr, candidate, vector,
                )

            clean_vector = embeddings[embedding_index[int(row.query_row)]]
            clean_score = score(clean_vector)
            clean_rank, clean_margin = clean_score.rank, clean_score.margin
            action_index = int(layout["action_index"])
            action_score = score(encoded[action_index])
            action_rank, action_margin = action_score.rank, action_score.margin
            indices = list(map(int, layout["control_indices"]))
            if indices:
                values = [score(encoded[index]) for index in indices]
                chosen = int(np.argmax([value.margin for value in values]))
                control_score = values[chosen]
                control_rank, control_margin = control_score.rank, control_score.margin
                control_tensor = spectra[indices[chosen]]
                control_kind = "matched_path"
            else:
                control_rank, control_margin = clean_rank, clean_margin
                control_score = clean_score
                control_tensor = store.one(int(row.query_row))
                control_kind = "clean_fallback"
            route = route_action(
                clean_rank=clean_rank, clean_margin=clean_margin,
                action_rank=action_rank, action_margin=action_margin,
                control_margin=control_margin, thresholds=thresholds,
            )
            tensor_index = len(chunk_action_spectra)
            chunk_action_spectra.append(
                spectra[action_index].detach().cpu().numpy().astype(np.float32),
            )
            chunk_control_spectra.append(
                control_tensor.detach().cpu().numpy().astype(np.float32),
            )
            records.append({
                "action_id": str(row.action_id), "query_index": query_index,
                "query_row": int(row.query_row), "query_ik14": str(row.query_ik14),
                "query_formula": str(row.query_formula),
                "formula_fold": int(row.formula_fold),
                "near": bool(graph.query_has_near[query_index]), "source": "N_mature",
                "family": str(row.selector),
                "recipe_id": (
                    f"{row.selector}|step={int(row.step)}|dose={float(row.attenuation):.2f}"
                ),
                "selector": str(row.selector), "step": int(row.step),
                "attenuation": float(row.attenuation), "target_path": str(row.target_path),
                "control_kind": control_kind,
                "control_semantic": (
                    "matched_neutral" if control_kind == "matched_path"
                    else "clean_fallback"
                ),
                "clean_rank": clean_rank,
                "clean_margin": clean_margin, "action_rank": action_rank,
                "action_margin": action_margin, "control_rank": control_rank,
                "control_margin": control_margin,
                "action_positive_row": action_score.positive_row,
                "action_hard_negative_molecule_index": (
                    action_score.hard_negative_molecule_index
                ),
                "action_hard_negative_row": action_score.hard_negative_row,
                "control_positive_row": control_score.positive_row,
                "control_hard_negative_molecule_index": (
                    control_score.hard_negative_molecule_index
                ),
                "control_hard_negative_row": control_score.hard_negative_row,
                "margin_change": action_margin - clean_margin,
                "paired_advantage": action_margin - control_margin,
                "route": route, "corrective_weight": float(route == "corrective"),
                "action_tensor_index": tensor_index,
            })
        chunk_result = pd.DataFrame(records)
        route_counts.update(map(str, chunk_result.route))
        control_counts.update(map(str, chunk_result.control_kind))
        total_actions += len(chunk_result)
        per_query_frames.append(chunk_result.groupby("query_index", sort=True).agg(
            clean_rank=("clean_rank", "first"),
            corrective_actions=("corrective_weight", "sum"),
            best_action_rank=("action_rank", "min"),
            best_margin_change=("margin_change", "max"),
            harmful_actions=("route", lambda value: int(np.sum(value == "harmful"))),
        ).reset_index())
        frontier = lossless_selector_frontier(
            chunk_result,
            maximum_corrective_per_query=args.maximum_corrective_frontier,
            maximum_harmful_per_query=args.maximum_harmful_frontier,
            maximum_robust_per_query=args.maximum_robust_frontier,
        )
        take = frontier.action_tensor_index.to_numpy(np.int64)
        if len(frontier):
            target_array = np.stack(chunk_action_spectra).astype(np.float32)
            control_array = np.stack(chunk_control_spectra).astype(np.float32)
            frontier_action_chunks.append(target_array[take])
            frontier_control_chunks.append(control_array[take])
            frontier_frames.append(frontier)
        processed = min(chunk_left + args.query_chunk_size, len(unique_queries))
        print(
            f"[corrected N router chunk] {processed:,}/{len(unique_queries):,}; "
            f"resident_spectra={len(spectra):,}; frontier={len(frontier):,}",
            flush=True,
        )
    if not frontier_frames:
        raise RuntimeError("N routing produced no corrective/harmful/robust selector frontier")
    result = pd.concat(frontier_frames, ignore_index=True, sort=False)
    action_spectra = np.concatenate(frontier_action_chunks, axis=0)
    control_spectra = np.concatenate(frontier_control_chunks, axis=0)
    if len(result) != len(action_spectra) or action_spectra.shape != control_spectra.shape:
        raise RuntimeError("N selector frontier tensor alignment failed")
    result["action_tensor_index"] = np.arange(len(result), dtype=np.int64)
    per_query = pd.concat(per_query_frames, ignore_index=True, sort=False)
    report = {
        "status": "noise_corrected_n_router_audit_complete",
        "formal_training_authorized": bool(
            action_report.get("formal_training_authorized") is True
        ),
        "outer_formula_fold": args.outer_fold,
        "configuration": {
            **{
                name: getattr(args, name)
                for name in REGISTERED_FORMAL_N_ROUTE_CONFIGURATION
            },
            "formula_fold_seed": int(
                action_report["configuration"]["formula_fold_seed"]
            ),
        },
        "queries": int(len(unique_queries)), "actions": int(total_actions),
        "actions_evaluated": int(total_actions),
        "selector_frontier_actions_retained": int(len(result)),
        "route_counts": {
            str(key): int(value) for key, value in route_counts.items()
        },
        "queries_with_corrective_action": int(per_query.corrective_actions.gt(0).sum()),
        "error_queries": int(per_query.clean_rank.gt(1).sum()),
        "error_queries_with_rank1_action": int(
            (per_query.clean_rank.gt(1) & per_query.best_action_rank.eq(1)).sum()
        ),
        "control_kinds": {
            str(key): int(value) for key, value in control_counts.items()
        },
        "bounded_memory": {
            "query_chunk_size": int(args.query_chunk_size),
            "maximum_resident_spectra": int(maximum_resident_spectra),
            "maximum_resident_action_embedding_bytes": int(maximum_resident_embedding_bytes),
            "monolithic_all_action_embedding_allocation": False,
        },
        "selector_frontier": {
            "maximum_corrective_per_query": int(args.maximum_corrective_frontier),
            "maximum_harmful_per_query": int(args.maximum_harmful_frontier),
            "maximum_robust_per_query": int(args.maximum_robust_frontier),
        },
        "contracts": {
            "registered_formal_route_configuration_verified": bool(formal_route),
            "target_action_eligibility_independent_of_matched_controls": True,
            "current_E8_geometry_rerouted": True,
            "selector_frontier_is_lossless_for_global_caps": True,
            "exact_action_control_candidate_switch_rows_recorded": True,
            "control_semantics_explicit": True,
            "nonfrontier_action_metadata_aggregated_not_materialized": True,
            "noncorrective_weight_exact_zero": bool(
                result.loc[~result.route.eq("corrective"), "corrective_weight"].eq(0).all()
            ),
            "outer_held_formula_consumed": False,
            "teacher_embedding_target_used": False,
            "P3_consumed": False,
        },
        "model_provenance": {
            "initial_student_checkpoint_sha256": sha256_file(args.initial_student_checkpoint),
            "initial_decision_sha256": sha256_file(initial_decision_path),
        },
        "provenance": {
            "candidate_graph_sha256": sha256_file(graph_path),
            "graph_report_sha256": sha256_file(graph_report_path),
            "action_bank_sha256": sha256_file(action_path),
            "action_bank_report_sha256": sha256_file(action_report_path),
            "script_sha256": sha256_file(Path(__file__)),
            "action_routing_v3_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_action_routing_v3.py")
            ),
        },
        "runtime_seconds": time.time() - started,
        "claim_limit": "Outer-train N action routing audit; not encoder performance.",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        result.to_csv(staging / "routed_actions.csv.gz", index=False, compression="gzip")
        per_query.to_csv(staging / "per_query.csv.gz", index=False, compression="gzip")
        np.savez_compressed(
            staging / "action_spectra.npz",
            action_ids=np.asarray(result.action_id.astype(str), dtype=str),
            action_spectra=action_spectra,
            control_spectra=control_spectra,
        )
        report["provenance"]["action_spectra_sha256"] = sha256_file(
            staging / "action_spectra.npz"
        )
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
