"""Replay every E15 action in the mature, reference-filtered E4-A geometry.

This stage creates fixed training teachers.  It never reads M3 held outcomes;
held and sentinel identities are used only as a reference-exclusion list.
"""
from __future__ import annotations
import argparse, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from build_noise_final_e15_m3_identity_split import row_identities  # noqa: E402
from noise_final_core import CandidateGraph, json_dump, seed_everything, sha256_file  # noqa: E402
from train_e1_identity import load_base_model, torch_load_compat  # noqa: E402
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows, forward_embeddings  # noqa: E402
from train_noise_final_e15_m2_overfit import action_tensor, c1_target, SOURCES  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--initial-student-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def candidate_geometry(graph, query, row_ik14, excluded):
    _, rows, ptr, _ = graph.query_block(int(query)); groups = []
    for molecule in range(len(ptr) - 1):
        start, stop = map(int, ptr[molecule:molecule + 2])
        allowed = [int(rows[local]) for local in range(start, stop)
                   if molecule == 0 or row_ik14[int(rows[local])] not in excluded]
        if allowed: groups.append(tuple(allowed))
    return groups


def rank_margin(vector, groups, embeddings, index):
    scores = [max(float(embeddings[index[row]] @ vector) for row in group) for group in groups]
    if len(scores) < 2: return None
    return 1 + sum(score >= scores[0] for score in scores[1:]), scores[0] - max(scores[1:])


def main():
    args = arguments(); seed_everything(args.seed)
    if args.output_dir.exists(): raise FileExistsError(f"refusing to overwrite E16 replay: {args.output_dir}")
    required = {"split_report": args.split_dir / "report.json", "actions": args.split_dir / "train_corrective.csv.gz",
                "excluded": args.split_dir / "excluded_reference_identities.txt", "graph": args.graph, "data": args.data,
                "official_checkpoint": args.official_checkpoint, "architecture_checkpoint": args.architecture_checkpoint,
                "initial_student_checkpoint": args.initial_student_checkpoint}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing: raise FileNotFoundError(missing)
    split_report = json.loads(required["split_report"].read_text(encoding="utf-8"))
    if not split_report.get("pass_to_identity_holdout_training"): raise RuntimeError("E16 requires passing M3 split")
    actions = pd.read_csv(required["actions"], low_memory=False)
    excluded = set(required["excluded"].read_text(encoding="utf-8").splitlines())
    graph = CandidateGraph(args.graph)
    reachable = np.unique(np.concatenate([graph.query_row, graph.pair_candidate_row])).astype(np.int64)
    store = SpectrumStore(args.data, reachable, args.n_highest_peaks); device = torch.device(args.device)
    model, _ = load_base_model(args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks)
    package = torch_load_compat(args.initial_student_checkpoint, map_location="cpu")
    if package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder" or package.get("P2b_used") or not package.get("inference_clean_only"):
        raise RuntimeError("E16 initialization is not mature clean-only E4-A")
    model.load_state_dict(package["model_state"], strict=True); model.eval()
    initial = encode_rows(model, store, store.rows, device, args.batch_size, False, "E16-mature")
    index = {int(row): position for position, row in enumerate(store.rows)}
    row_ik14 = row_identities(args.data, store.rows)

    raw_positions, raw_spectra, teacher = [], [], [None] * len(actions)
    for position, (_, row) in enumerate(actions.iterrows()):
        if str(row["source"]) == "C1_support_disjoint":
            teacher[position] = c1_target(row, initial, index)
        else:
            variant = action_tensor(store, row)
            if variant is None: raise RuntimeError(f"non-C1 action has no executable spectrum: {position}")
            raw_positions.append(position); raw_spectra.append(variant)
    for left in range(0, len(raw_spectra), args.batch_size):
        batch = torch.stack(raw_spectra[left:left + args.batch_size]).to(device)
        encoded = forward_embeddings(model, batch, False).detach().cpu().numpy().astype(np.float32)
        for offset, vector in enumerate(encoded): teacher[raw_positions[left + offset]] = vector
        print(f"[E16 replay] {min(left + args.batch_size, len(raw_spectra)):,}/{len(raw_spectra):,}", flush=True)
    teacher_matrix = np.stack(teacher).astype(np.float32)

    geometry_cache = {}; records = []
    for position, (_, row) in enumerate(actions.iterrows()):
        query = int(row["query_index"])
        if query not in geometry_cache:
            geometry_cache[query] = candidate_geometry(graph, query, row_ik14, excluded)
        groups = geometry_cache[query]
        clean = rank_margin(initial[index[int(row["query_row"])]], groups, initial, index)
        action = rank_margin(teacher_matrix[position], groups, initial, index)
        if clean is None or action is None:
            records.append({"replay_index": position, "query_index": query, "eligible_geometry": False})
            continue
        clean_rank, clean_margin = clean; action_rank, action_margin = action; delta = action_margin - clean_margin
        error_corrected = clean_rank != 1 and action_rank == 1
        error_improved = clean_rank != 1 and action_rank < clean_rank and delta > 0
        safe_strengthened = clean_rank == 1 and action_rank == 1 and delta >= 0.005
        harmful = (clean_rank == 1 and action_rank != 1) or delta <= -0.005
        records.append({"replay_index": position, "query_index": query, "eligible_geometry": True,
                        "clean_rank": clean_rank, "action_rank": action_rank,
                        "clean_margin": clean_margin, "action_margin": action_margin, "margin_delta": delta,
                        "error_corrected": error_corrected, "error_improved": error_improved,
                        "safe_strengthened": safe_strengthened, "harmful_in_mature_geometry": harmful,
                        "teacher_eligible": bool(error_corrected or error_improved or safe_strengthened)})
    replay = actions.reset_index(drop=True).join(pd.DataFrame(records).drop(columns=["query_index"]))
    eligible = replay.loc[replay["teacher_eligible"].fillna(False)].copy()
    harmful = replay.loc[replay["harmful_in_mature_geometry"].fillna(False)].copy()
    query = replay.groupby("query_index", as_index=False).agg(
        query_ik14=("query_ik14", "first"), query_formula=("query_formula", "first"),
        clean_rank=("clean_rank", "first"), actions=("action_id", "size"),
        eligible_actions=("teacher_eligible", "sum"), correcting_actions=("error_corrected", "sum"),
        harmful_actions=("harmful_in_mature_geometry", "sum"), best_margin_delta=("margin_delta", "max"),
    )
    source = {name: {"actions": int((eligible["source"].astype(str) == name).sum()),
                     "queries": int(eligible.loc[eligible["source"].astype(str).eq(name), "query_index"].nunique()),
                     "identities": int(eligible.loc[eligible["source"].astype(str).eq(name), "query_ik14"].nunique())}
              for name in SOURCES}
    gates = {"all_actions_replayed": len(replay) == len(actions), "all_sources_have_fixed_teachers": all(value["actions"] > 0 for value in source.values()),
             "eligible_teacher_pool_nonempty": len(eligible) > 0, "harmful_pool_materialized": len(harmful) > 0,
             "P2b_forbidden": True, "P3_not_consumed": True}
    report = {"status": "noise_final_e16_mature_action_replay_complete", "formal": True,
              "actions_replayed": int(len(replay)), "queries": int(replay["query_index"].nunique()),
              "eligible_teacher_actions": int(len(eligible)), "eligible_teacher_queries": int(eligible["query_index"].nunique()),
              "eligible_teacher_identities": int(eligible["query_ik14"].nunique()),
              "error_queries_correctable": int(query["correcting_actions"].gt(0).sum()),
              "safe_strengthening_queries": int(eligible.loc[eligible["safe_strengthened"].fillna(False), "query_index"].nunique()),
              "harmful_actions": int(len(harmful)), "source_capacity": source, "gates": gates,
              "pass_to_fresh_identity_split": bool(all(gates.values())),
              "contracts": {"teacher_geometry": "mature E4-A with held/sentinel references excluded",
                            "teacher_vectors_frozen": True, "held_outcomes_read": False,
                            "P2b": "forbidden", "P3_consumed": False},
              "provenance": {name: sha256_file(path) for name, path in required.items()},
              "claim_limit": "Training-only mature action replay; not a trained encoder or held performance result."}
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="noise_e16_replay_", dir=args.output_dir.parent))
    try:
        replay.to_csv(staging / "action_replay.csv.gz", index=False, compression="gzip")
        query.to_csv(staging / "query_replay.csv.gz", index=False, compression="gzip")
        np.savez_compressed(
            staging / "teacher_embeddings.npz",
            replay_index=np.arange(len(actions), dtype=np.int64),
            source=actions["source"].astype(str).to_numpy(),
            action_id=actions["action_id"].astype(str).to_numpy(),
            query_index=actions["query_index"].to_numpy(np.int64),
            embeddings=teacher_matrix.astype(np.float16),
        )
        json_dump(staging / "report.json", report); staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__": main()
