"""Re-evaluate the mature A4 exact-peak action in corrected E8 geometry.

Historical A4 supplies one formula-OOF selected peak action per query.  Old
query indices and old outcome labels are never reused.  Actions are remapped
by the stable spectrum row plus identity/formula, materialized from the raw
spectrum, paired with matched peak controls when available, and routed only
on outer-training formulas in the current initialization geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import h5py
import numpy as np
import pandas as pd
import torch

from noise_corrected_action_routing import RoutingThresholds, route_action
from noise_final_core import CandidateGraph, sha256_file, stable_fold, strict_rank
from noise_v3_core import (
    ROLE_NAMES, attenuate_and_renormalize, matched_control_tokens,
    matched_control_tokens_strict, stable_seed,
)
from train_e1_identity import load_base_model, torch_load_compat
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows, forward_embeddings


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--a4-scan-dir", type=Path, required=True)
    parser.add_argument("--a4-teacher-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-student-checkpoint", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument(
        "--query-scope", choices=("official_errors", "official_correct", "all"),
        default="all",
    )
    parser.add_argument("--max-queries", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260906)
    parser.add_argument("--control-repeats", type=int, default=2)
    parser.add_argument("--paired-advantage-threshold", type=float, default=0.01)
    parser.add_argument("--harm-margin-threshold", type=float, default=0.01)
    parser.add_argument("--robustness-slack", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def remap_selected_actions(
    graph: CandidateGraph,
    scan: pd.DataFrame,
    selected: pd.DataFrame,
    formula_folds: np.ndarray,
    *,
    outer_fold: int,
    query_scope: str,
    maximum_queries: int,
    sample_seed: int,
) -> pd.DataFrame:
    """Return exact row/identity/formula-aligned A4 actions for outer train."""
    required_scan = {
        "scan_position", "query_row", "query_ik14", "query_formula",
    }
    required_selected = {"scan_position", "action_index", "token", "role", "attenuation"}
    if missing := required_scan - set(scan.columns):
        raise KeyError(f"A4 scan table misses {sorted(missing)}")
    if missing := required_selected - set(selected.columns):
        raise KeyError(f"A4 selected table misses {sorted(missing)}")
    metadata = scan[list(required_scan)].copy()
    if metadata.scan_position.duplicated().any():
        raise RuntimeError("A4 scan_position is not unique")
    frame = selected.merge(metadata, on="scan_position", how="left", validate="many_to_one")
    if frame.query_row.isna().any():
        raise RuntimeError("selected A4 action lacks scan metadata")
    query_by_row = {int(row): query for query, row in enumerate(graph.query_row)}
    mapped = frame.query_row.astype(np.int64).map(query_by_row)
    frame = frame.loc[mapped.notna()].copy()
    frame["query_index"] = mapped[mapped.notna()].astype(np.int64).to_numpy()
    query = frame.query_index.to_numpy(np.int64)
    if not np.array_equal(frame.query_row.to_numpy(np.int64), graph.query_row[query]):
        raise RuntimeError("A4 spectrum row remap drifted")
    if not np.array_equal(frame.query_ik14.astype(str).to_numpy(), graph.query_ik14[query]):
        raise RuntimeError("A4 identity changed during row remap")
    if not np.array_equal(frame.query_formula.astype(str).to_numpy(), graph.query_formula[query]):
        raise RuntimeError("A4 formula changed during row remap")
    frame["formula_fold"] = formula_folds[query]
    frame = frame.loc[frame.formula_fold.ne(outer_fold)].copy()
    official_rank = np.asarray([
        strict_rank(graph.official_molecule_scores(int(value)))
        for value in frame.query_index.to_numpy(np.int64)
    ], dtype=np.int16)
    if query_scope == "official_errors":
        frame = frame.loc[official_rank > 1].copy()
    elif query_scope == "official_correct":
        frame = frame.loc[official_rank == 1].copy()
    if maximum_queries < 1:
        raise ValueError("bounded A4 router requires max-queries >= 1")
    if len(frame) > maximum_queries:
        rng = np.random.default_rng(sample_seed)
        keep = np.sort(rng.choice(len(frame), maximum_queries, replace=False))
        frame = frame.iloc[keep].copy()
    if frame.query_index.duplicated().any():
        raise RuntimeError("A4 policy must provide at most one selected action per query")
    return frame.sort_values("query_index", kind="stable").reset_index(drop=True)


def action_identifier(query_row: int, token: int, attenuation: float) -> str:
    payload = f"corrected-v1|A4|{query_row}|{token}|{attenuation:.2f}".encode()
    return "NC-A4-" + hashlib.sha256(payload).hexdigest()[:24]


def choose_control_tokens(
    clean: torch.Tensor,
    target: int,
    roles: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Prefer strict same-role controls but never discard the target action."""
    strict = matched_control_tokens_strict(clean, target, roles, repeats, seed)
    if len(strict) == repeats:
        return strict, "strict_same_role"
    relaxed = matched_control_tokens(
        clean, target, roles, repeats, seed, same_role=True,
    )
    if len(relaxed):
        return relaxed, "relaxed_peak_matched"
    return np.empty(0, dtype=np.int64), "clean_fallback"


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.outer_fold not in range(5) or args.control_repeats < 1:
        raise ValueError("outer fold/control repeats are invalid")
    graph_path = args.graph_dir / "candidate_graph.npz"
    graph_report_path = args.graph_dir / "report.json"
    scan_path = args.a4_scan_dir / "scan_queries.csv.gz"
    h5_path = args.a4_scan_dir / "exact_peak_scan.h5"
    selected_path = args.a4_teacher_dir / "oof_selected_actions.csv.gz"
    teacher_decision_path = args.a4_teacher_dir / "decision.json"
    initial_decision_path = args.initial_student_checkpoint.parent / "decision.json"
    required = (
        graph_path, graph_report_path, scan_path, h5_path, selected_path,
        teacher_decision_path, args.data, args.official_checkpoint,
        args.architecture_checkpoint, args.initial_student_checkpoint,
        initial_decision_path,
    )
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    if graph_report.get("formal_training_authorized") is not True:
        raise RuntimeError("corrected graph is not training-authorized")
    teacher_decision = json.loads(teacher_decision_path.read_text(encoding="utf-8"))
    if (
        teacher_decision.get("status") != "noise_v3_a4_nonlinear_action_teacher_complete"
        or teacher_decision.get("formal") is not True
        or teacher_decision.get("integrity", {}).get("formula_fold_overlap") != 0
    ):
        raise RuntimeError("A4 formula-OOF selected-action provenance failed")

    graph = CandidateGraph(graph_path)
    formula_folds = np.asarray([
        stable_fold(str(value), 5, args.formula_fold_seed) for value in graph.query_formula
    ], dtype=np.int8)
    scan = pd.read_csv(scan_path, low_memory=False)
    selected = pd.read_csv(selected_path, low_memory=False)
    actions = remap_selected_actions(
        graph, scan, selected, formula_folds, outer_fold=args.outer_fold,
        query_scope=args.query_scope, maximum_queries=args.max_queries,
        sample_seed=args.sample_seed,
    )
    if not len(actions):
        raise RuntimeError("A4 corrected-geometry panel is empty")

    needed: set[int] = set(map(int, actions.query_row))
    for query in actions.query_index:
        _, rows, _, _ = graph.query_block(int(query))
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
        "corrected-A4-router-initial",
    )
    embedding_index = {int(row): index for index, row in enumerate(reachable)}

    spectra: list[torch.Tensor] = []
    layouts: list[dict[str, object]] = []
    with h5py.File(h5_path, "r") as handle:
        action_query = handle["action_query"][:]
        action_token = handle["action_token"][:]
        action_role = handle["action_role"][:]
        action_mz = handle["action_mz"][:]
        action_intensity = handle["action_intensity"][:]
        ptr = handle["query_action_ptr"][:]
        for row in actions.itertuples(index=False):
            scan_position = int(row.scan_position)
            action_index = int(row.action_index)
            target = int(row.token)
            if (
                not int(ptr[scan_position]) <= action_index < int(ptr[scan_position + 1])
                or int(action_query[action_index]) != scan_position
                or int(action_token[action_index]) != target
                or int(action_role[action_index]) != int(row.role)
            ):
                raise RuntimeError("A4 selected action no longer matches exact scan payload")
            clean = store.one(int(row.query_row))
            clean_values = clean.detach().cpu().numpy()
            if (
                not np.isclose(clean_values[target, 0], action_mz[action_index], atol=1e-4)
                or not np.isclose(clean_values[target, 1], action_intensity[action_index], atol=1e-5)
            ):
                raise RuntimeError("A4 token no longer matches raw clean spectrum")
            roles = np.full(args.n_highest_peaks, -1, dtype=np.int8)
            left, right = int(ptr[scan_position]), int(ptr[scan_position + 1])
            tokens = action_token[left:right].astype(np.int64)
            if len(np.unique(tokens)) != len(tokens):
                raise RuntimeError("A4 scan contains duplicate token roles")
            roles[tokens] = action_role[left:right]
            controls, control_kind = choose_control_tokens(
                clean, target, roles, args.control_repeats,
                stable_seed(args.sample_seed, int(row.query_row), target, float(row.attenuation)),
            )
            action_tensor_index = len(spectra)
            spectra.append(attenuate_and_renormalize(clean, target, float(row.attenuation)))
            control_indices = []
            for token in controls:
                control_indices.append(len(spectra))
                spectra.append(attenuate_and_renormalize(clean, int(token), float(row.attenuation)))
            layouts.append({
                "row": row, "target": target, "target_role": int(row.role),
                "action_tensor_index": action_tensor_index,
                "control_tensor_indices": control_indices,
                "control_tokens": controls, "control_kind": control_kind,
            })

    encoded = np.empty((len(spectra), embeddings.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for left in range(0, len(spectra), args.batch_size):
            right = min(left + args.batch_size, len(spectra))
            encoded[left:right] = forward_embeddings(
                model, torch.stack(spectra[left:right]).to(device), args.amp,
            ).float().cpu().numpy()
            if right == len(spectra) or right % (args.batch_size * 8) == 0:
                print(f"[corrected A4 router encode] {right:,}/{len(spectra):,}", flush=True)

    thresholds = RoutingThresholds(
        paired_advantage=args.paired_advantage_threshold,
        harm_margin=args.harm_margin_threshold,
        robustness_slack=args.robustness_slack,
    )
    records: list[dict[str, object]] = []
    selected_action_spectra: list[np.ndarray] = []
    selected_control_spectra: list[np.ndarray] = []
    for layout in layouts:
        row = layout["row"]
        query = int(row.query_index)
        _, candidate_rows, molecule_ptr, _ = graph.query_block(query)
        candidate = embeddings[[embedding_index[int(value)] for value in candidate_rows]]

        def score(vector: np.ndarray) -> tuple[int, float]:
            molecule = np.maximum.reduceat(candidate @ vector, molecule_ptr[:-1])
            return strict_rank(molecule), float(molecule[0] - np.max(molecule[1:]))

        clean_vector = embeddings[embedding_index[int(row.query_row)]]
        clean_rank, clean_margin = score(clean_vector)
        action_index = int(layout["action_tensor_index"])
        action_rank, action_margin = score(encoded[action_index])
        control_indices = list(map(int, layout["control_tensor_indices"]))
        if control_indices:
            control_values = [score(encoded[index]) for index in control_indices]
            # The strongest matched control is deliberately conservative: an A4
            # action must beat the best ordinary peak attenuation, not their mean.
            chosen = int(np.argmax([value[1] for value in control_values]))
            control_rank, control_margin = control_values[chosen]
            control_tensor = spectra[control_indices[chosen]]
            control_token = int(layout["control_tokens"][chosen])
        else:
            control_rank, control_margin = clean_rank, clean_margin
            control_tensor = store.one(int(row.query_row))
            control_token = -1
        route = route_action(
            clean_rank=clean_rank, clean_margin=clean_margin,
            action_rank=action_rank, action_margin=action_margin,
            control_margin=control_margin, thresholds=thresholds,
        )
        identifier = action_identifier(
            int(row.query_row), int(layout["target"]), float(row.attenuation),
        )
        tensor_index = len(selected_action_spectra)
        selected_action_spectra.append(np.asarray(spectra[action_index], dtype=np.float32))
        selected_control_spectra.append(np.asarray(control_tensor, dtype=np.float32))
        role_name = str(ROLE_NAMES[int(layout["target_role"])])
        records.append({
            "action_id": identifier, "query_index": query,
            "query_row": int(row.query_row), "query_ik14": str(row.query_ik14),
            "query_formula": str(row.query_formula),
            "formula_fold": int(row.formula_fold), "near": bool(graph.query_has_near[query]),
            "source": "A4_exact", "family": f"exact_peak_{role_name}",
            "recipe_id": f"token={int(layout['target'])}|dose={float(row.attenuation):.2f}",
            "token": int(layout["target"]), "role": int(layout["target_role"]),
            "attenuation": float(row.attenuation), "control_kind": str(layout["control_kind"]),
            "control_token": control_token, "clean_rank": clean_rank,
            "clean_margin": clean_margin, "action_rank": action_rank,
            "action_margin": action_margin, "control_rank": control_rank,
            "control_margin": control_margin,
            "margin_change": action_margin - clean_margin,
            "paired_advantage": action_margin - control_margin,
            "route": route, "corrective_weight": float(route == "corrective"),
            "action_tensor_index": tensor_index,
        })
    result = pd.DataFrame(records)
    report = {
        "status": "noise_corrected_a4_router_audit_complete",
        "formal_training_authorized": False,
        "outer_formula_fold": args.outer_fold, "query_scope": args.query_scope,
        "queries": int(len(result)),
        "route_counts": {
            str(key): int(value) for key, value in result.route.value_counts().items()
        },
        "queries_with_corrective_action": int(result.route.eq("corrective").sum()),
        "current_geometry_rank1_actions": int(
            (result.clean_rank.gt(1) & result.action_rank.eq(1)).sum()
        ),
        "control_kinds": {
            str(key): int(value) for key, value in result.control_kind.value_counts().items()
        },
        "contracts": {
            "old_query_index_reused": False,
            "stable_row_identity_formula_remap": True,
            "old_A4_outcome_labels_consumed": False,
            "A4_selected_policy_used_as_action_source_only": True,
            "current_E8_geometry_rerouted": True,
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
            "A4_scan_sha256": sha256_file(h5_path),
            "A4_selected_actions_sha256": sha256_file(selected_path),
            "A4_teacher_decision_sha256": sha256_file(teacher_decision_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "runtime_seconds": time.time() - started,
        "claim_limit": "Bounded outer-train A4 routing audit; not encoder performance.",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        result.to_csv(staging / "routed_actions.csv.gz", index=False, compression="gzip")
        np.savez_compressed(
            staging / "action_spectra.npz",
            action_ids=result.action_id.astype(str).to_numpy(),
            action_spectra=np.stack(selected_action_spectra).astype(np.float32),
            control_spectra=np.stack(selected_control_spectra).astype(np.float32),
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
