"""Qualify structure-differential observed-peak actions without distillation.

ICEBERG predictions are used only to choose which *observed* intensities to
attenuate.  Frozen official DreaMS measures action headroom.  Formula folds
0-1 select one global action setting and fold 2 confirms it once.  Fold 3 is
reserved for development-time embedding evaluation and fold 4 for final outer
evaluation; actions are never generated for either evaluation fold.  Correct-
structure evidence must beat candidate-swapped and peak-permuted controls.  No
model weights are updated here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_chemaware_iceberg_synthetic_embedding import peak_permute  # noqa: E402
from chemaware_direct_action_core import (  # noqa: E402
    ROLE_CODE, classify_action_roles, formula_bootstrap,
)
from chemaware_retrieval_graph import RetrievalGraph  # noqa: E402
from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from chemaware_iceberg_peak_action_core import apply_peak_action, differential_evidence, hard_negative_indices  # noqa: E402
from noise_final_core import sha256_file, strict_rank  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_r2_shared_encoder import SpectrumStore  # noqa: E402


SETTINGS = (
    *(("conflict_attenuate", strength, top_k)
      for strength in (0.25, 0.50, 0.75) for top_k in (3, 5, 10)),
    *(("support_boost", strength, top_k)
      for strength in (0.10, 0.25, 0.50) for top_k in (3, 5, 10)),
    *(("signed_exp", strength, 0) for strength in (0.25, 0.50, 0.75)),
)
ARMS = ("correct", "candidate_swapped", "peak_permuted")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1/graph.npz")
    parser.add_argument("--teacher-dir", type=Path, default=ROOT / "data/validation/chemaware_full_manifest_iceberg_teacher_700_v1")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_direct_action_bank_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260935)
    parser.add_argument("--discovery-folds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--confirmation-fold", type=int, default=2)
    parser.add_argument("--embedding-evaluation-fold", type=int, default=3)
    parser.add_argument("--reserve-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260935)
    parser.add_argument("--minimum-margin-action", type=float, default=0.005)
    parser.add_argument("--harmful-margin", type=float, default=0.010)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def normalize_prediction(values: np.ndarray) -> np.ndarray:
    return values / max(float(np.max(values)), 1e-12)


def molecule_rank_margin(graph: RetrievalGraph, query: int, query_embedding: np.ndarray,
                         reference: np.ndarray, row_position: dict[int, int]) -> tuple[int, float]:
    _, rows, ptr, _ = graph.query_block(query)
    pair = reference[[row_position[int(row)] for row in rows]] @ query_embedding
    molecule = np.maximum.reduceat(pair, ptr[:-1])
    return strict_rank(molecule), float(molecule[0] - np.max(molecule[1:]))


def encode(model, spectra, device, batch_size: int) -> np.ndarray:
    import torch

    model.eval(); output = []
    with torch.no_grad():
        for left in range(0, len(spectra), batch_size):
            output.append(model(spectra[left:left + batch_size].to(device)).float().cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    import torch

    args = arguments(); started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    roles = set((
        *args.discovery_folds,
        args.confirmation_fold,
        args.embedding_evaluation_fold,
        args.reserve_fold,
    ))
    if len(roles) != len(args.discovery_folds) + 3 or args.bootstrap_draws < 10_000:
        raise ValueError(
            "action discovery, action confirmation, embedding evaluation and reserve roles "
            "must be distinct"
        )
    if min(roles) < 0 or max(roles) >= args.folds:
        raise ValueError("formula-fold role lies outside --folds")
    teacher_report = json.loads((args.teacher_dir / "report.json").read_text(encoding="utf-8"))
    if (teacher_report.get("status") != "PASS"
            or teacher_report.get("selected", {}).get("unique_identities") != teacher_report.get("selected", {}).get("queries")
            or teacher_report.get("inputs", {}).get("graph_sha256") != sha256_file(args.graph)):
        raise RuntimeError("requires a passed identity-unique ICEBERG teacher ledger")
    torch.set_num_threads(args.torch_threads)
    graph = RetrievalGraph(args.graph)
    selected = np.load(args.teacher_dir / "selected_queries.npy").astype(np.int64)
    ptr = np.load(args.teacher_dir / "query_ptr.npy").astype(np.int64)
    prediction = np.load(args.teacher_dir / "iceberg_predictions_f16.npy").astype(np.float32)
    score_file = np.load(args.teacher_dir / "scores_and_ranks.npz", allow_pickle=True)
    token_rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): i for i, row in enumerate(token_rows)}
    if not set(map(int, graph.query_row[selected])).issubset(row_position):
        raise RuntimeError("teacher queries are absent from official cache")

    formula = graph.query_formula[selected].astype(str)
    fold = stable_formula_folds(formula, args.folds, args.fold_seed)
    discovery = np.flatnonzero(np.isin(fold, args.discovery_folds))
    confirmation = np.flatnonzero(fold == args.confirmation_fold)
    embedding_evaluation = np.flatnonzero(fold == args.embedding_evaluation_fold)
    reserve = np.flatnonzero(fold == args.reserve_fold)
    audited = np.concatenate((discovery, confirmation))
    if min(len(discovery), len(confirmation), len(embedding_evaluation), len(reserve)) == 0:
        raise RuntimeError("one or more formula-fold roles are empty")
    if len(ptr) != len(selected) + 1 or int(ptr[0]) != 0 or int(ptr[-1]) != len(prediction):
        raise RuntimeError("teacher query pointer does not span predictions")
    required_scores = {
        "correct_score", "candidate_swapped_score", "peak_permuted_score",
    }
    if not required_scores.issubset(score_file.files):
        raise RuntimeError("teacher score ledger lacks matched action controls")
    if args.preflight_only:
        preflight = {
            "status": "CHEMAWARE_DIRECT_ACTION_BANK_PREFLIGHT_PASS",
            "weights_updated": False,
            "queries": int(len(selected)),
            "formulas": int(len(np.unique(formula))),
            "settings": int(len(SETTINGS)),
            "operators": sorted({value[0] for value in SETTINGS}),
            "fold_roles": {
                "discovery_queries": int(len(discovery)),
                "confirmation_queries": int(len(confirmation)),
                "embedding_evaluation_queries_action_unseen": int(len(embedding_evaluation)),
                "outer_queries_action_unseen": int(len(reserve)),
            },
            "planned_spectrum_encodes": int(
                3 * (len(SETTINGS) * len(discovery) + len(confirmation))
            ),
            "graph_sha256": sha256_file(args.graph),
            "teacher_report_sha256": sha256_file(args.teacher_dir / "report.json"),
        }
        print(json.dumps(preflight, indent=2))
        return

    swapped = np.concatenate([np.roll(prediction[int(left):int(right)], 1, axis=0)
                              for left, right in zip(ptr[:-1], ptr[1:])])
    permuted = peak_permute(prediction, args.seed + 41)
    arm_prediction = {"correct": prediction, "candidate_swapped": swapped, "peak_permuted": permuted}
    arm_score = {"correct": np.asarray(score_file["correct_score"], dtype=np.float32),
                 "candidate_swapped": np.asarray(score_file["candidate_swapped_score"], dtype=np.float32),
                 "peak_permuted": np.asarray(score_file["peak_permuted_score"], dtype=np.float32)}
    hard = {arm: hard_negative_indices(ptr, arm_score[arm]) for arm in ARMS}
    store = SpectrumStore(args.data, graph.query_row[selected], args.n_highest_peaks)
    clean = torch.stack([store.one(int(graph.query_row[q])) for q in selected])

    action_tensors = []; labels = []
    for setting_id, (mode, strength, top_k) in enumerate(SETTINGS):
        for arm in ARMS:
            block = []
            for position in discovery:
                evidence = differential_evidence(
                    arm_prediction[arm][int(ptr[position])],
                    arm_prediction[arm][int(hard[arm][position])],
                    clean[position, 1:, 0].numpy(),
                )
                block.append(apply_peak_action(clean[position], evidence, mode, strength, top_k))
            action_tensors.extend(block); labels.append((setting_id, arm, mode, strength, top_k))
    device = torch.device(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, _ = load_base_model(args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks)
    encoded = encode(model, torch.stack(action_tensors), device, args.batch_size)

    old_rank = np.empty(len(selected), dtype=np.int16); old_margin = np.empty(len(selected), dtype=np.float32)
    for position, query in enumerate(selected):
        qz = official[row_position[int(graph.query_row[query])]]
        old_rank[position], old_margin[position] = molecule_rank_margin(graph, int(query), qz, official, row_position)
    ranks = np.full((len(SETTINGS), len(ARMS), len(selected)), -1, dtype=np.int16)
    margins = np.full((len(SETTINGS), len(ARMS), len(selected)), np.nan, dtype=np.float32)
    block_size = len(discovery)
    for block_index, (setting_id, arm, _mode, _strength, _top_k) in enumerate(labels):
        arm_id = ARMS.index(arm); block = encoded[block_index * block_size:(block_index + 1) * block_size]
        for local_position, position in enumerate(discovery):
            query = selected[position]
            ranks[setting_id, arm_id, position], margins[setting_id, arm_id, position] = molecule_rank_margin(
                graph, int(query), block[local_position], official, row_position,
            )
    setting_reports = []
    for setting_id, (mode, strength, top_k) in enumerate(SETTINGS):
        delta = margins[setting_id] - old_margin[None, :]
        hit_delta = (ranks[setting_id, 0] == 1).astype(float) - (old_rank == 1).astype(float)
        report = {"setting_id": setting_id, "mode": mode, "strength": strength, "top_k": top_k,
                  "discovery": {
                      "absolute_margin": formula_bootstrap(delta[0, discovery], formula[discovery], args.seed + setting_id, args.bootstrap_draws),
                      "minus_candidate_swapped_margin": formula_bootstrap(
                          (margins[setting_id, 0] - margins[setting_id, 1])[discovery], formula[discovery], args.seed + 100 + setting_id, args.bootstrap_draws),
                      "minus_peak_permuted_margin": formula_bootstrap(
                          (margins[setting_id, 0] - margins[setting_id, 2])[discovery], formula[discovery], args.seed + 200 + setting_id, args.bootstrap_draws),
                      "hit1_delta": float(np.mean(hit_delta[discovery])),
                      "corrected": int(np.sum((old_rank[discovery] > 1) & (ranks[setting_id, 0, discovery] == 1))),
                      "introduced": int(np.sum((old_rank[discovery] == 1) & (ranks[setting_id, 0, discovery] > 1))),
                  }}
        components = [report["discovery"][key]["formula_macro_mean"] for key in
                      ("absolute_margin", "minus_candidate_swapped_margin", "minus_peak_permuted_margin")]
        report["discovery"]["minimum_margin_advantage"] = float(min(components))
        setting_reports.append(report)
    selected_setting = max(setting_reports, key=lambda item: (
        item["discovery"]["minimum_margin_advantage"],
        item["discovery"]["hit1_delta"], -item["strength"], -item["top_k"],
    ))
    setting_id = int(selected_setting["setting_id"])
    selected_mode, selected_strength, selected_top_k = SETTINGS[setting_id]
    confirmation_tensors = []
    for arm in ARMS:
        for position in confirmation:
            evidence = differential_evidence(
                arm_prediction[arm][int(ptr[position])],
                arm_prediction[arm][int(hard[arm][position])],
                clean[position, 1:, 0].numpy(),
            )
            confirmation_tensors.append(apply_peak_action(
                clean[position], evidence, selected_mode, selected_strength, selected_top_k,
            ))
    confirmation_encoded = encode(
        model, torch.stack(confirmation_tensors), device, args.batch_size,
    )
    for arm_id, arm in enumerate(ARMS):
        block = confirmation_encoded[
            arm_id * len(confirmation):(arm_id + 1) * len(confirmation)
        ]
        for local_position, position in enumerate(confirmation):
            query = selected[position]
            ranks[setting_id, arm_id, position], margins[setting_id, arm_id, position] = molecule_rank_margin(
                graph, int(query), block[local_position], official, row_position,
            )
    confirmation_metrics = {
        "absolute_margin": formula_bootstrap((margins[setting_id, 0] - old_margin)[confirmation], formula[confirmation], args.seed + 501, args.bootstrap_draws),
        "minus_candidate_swapped_margin": formula_bootstrap((margins[setting_id, 0] - margins[setting_id, 1])[confirmation], formula[confirmation], args.seed + 502, args.bootstrap_draws),
        "minus_peak_permuted_margin": formula_bootstrap((margins[setting_id, 0] - margins[setting_id, 2])[confirmation], formula[confirmation], args.seed + 503, args.bootstrap_draws),
    }
    confirmation_hit = (ranks[setting_id, 0, confirmation] == 1).astype(float) - (old_rank[confirmation] == 1).astype(float)
    confirmation_metrics.update({"hit1_delta": float(np.mean(confirmation_hit)),
        "corrected": int(np.sum((old_rank[confirmation] > 1) & (ranks[setting_id, 0, confirmation] == 1))),
        "introduced": int(np.sum((old_rank[confirmation] == 1) & (ranks[setting_id, 0, confirmation] > 1)))})
    gates = {
        f"{key}_formula_ci_positive": value["formula_cluster_bootstrap_95ci"][0] > 0
        for key, value in confirmation_metrics.items() if isinstance(value, dict)
    }
    gates["confirmation_hit1_nonnegative"] = confirmation_metrics["hit1_delta"] >= 0
    gates["confirmation_corrected_not_less_than_introduced"] = confirmation_metrics["corrected"] >= confirmation_metrics["introduced"]
    passed = bool(all(gates.values()))

    role_blocks = []
    for candidate_setting in range(len(SETTINGS)):
        roles_for_setting = np.full(len(selected), -1, dtype=np.int8)
        evaluated = audited if candidate_setting == setting_id else discovery
        delta = margins[candidate_setting, 0, evaluated] - old_margin[evaluated]
        roles_for_setting[evaluated] = classify_action_roles(
            old_rank[evaluated], ranks[candidate_setting, 0, evaluated], delta,
            margins[candidate_setting, 0, evaluated] - margins[candidate_setting, 1, evaluated],
            margins[candidate_setting, 0, evaluated] - margins[candidate_setting, 2, evaluated],
            args.minimum_margin_action, args.harmful_margin,
        )
        role_blocks.append(roles_for_setting)
    roles_array = np.stack(role_blocks)
    training_positions = np.concatenate((discovery, confirmation))
    selected_roles = roles_array[setting_id]
    eligible = np.zeros(len(selected), dtype=bool)
    eligible[training_positions] = np.isin(
        selected_roles[training_positions],
        [ROLE_CODE["corrective_rank"], ROLE_CODE["corrective_margin"]],
    )
    report = {
        "status": "CHEMAWARE_DIRECT_ACTION_BANK_PASS" if passed else "CHEMAWARE_DIRECT_ACTION_BANK_FAIL",
        "pass_to_direct_action_pilot": passed, "formal_training_authorized": False,
        "scope": {"weights_updated": False, "teacher_training_only": True,
                  "candidate_input_at_inference": False, "observed_peak_mz_unchanged": True,
                  "teacher_panel_biased_toward_official_errors": True,
                  "global_inner_outer_evaluated": False},
        "split": {"queries": len(selected), "formulas": len(np.unique(formula)),
                  "folds": args.folds, "fold_seed": args.fold_seed,
                  "discovery_folds": list(args.discovery_folds),
                  "confirmation_fold": args.confirmation_fold,
                  "embedding_evaluation_fold": args.embedding_evaluation_fold,
                  "reserve_fold": args.reserve_fold,
                  "discovery_queries": len(discovery), "discovery_formulas": len(np.unique(formula[discovery])),
                  "confirmation_queries": len(confirmation), "confirmation_formulas": len(np.unique(formula[confirmation])),
                  "embedding_evaluation_queries_not_action_evaluated": len(embedding_evaluation),
                  "embedding_evaluation_formulas": len(np.unique(formula[embedding_evaluation])),
                  "reserve_queries": len(reserve), "reserve_formulas": len(np.unique(formula[reserve])),
                  "formula_overlap": 0},
        "action_space": {"operators": ["conflict_attenuate", "support_boost", "signed_exp"],
                         "action_generation_seed": args.seed,
                         "settings": setting_reports,
                         "selected_setting": selected_setting,
                         "evaluated_correct_actions": int(len(SETTINGS) * len(discovery) + len(confirmation)),
                         "matched_control_actions": int(2 * (len(SETTINGS) * len(discovery) + len(confirmation))),
                         "eligible_training_actions_if_family_passes": int(np.sum(eligible)),
                         "role_counts_audited_only": {
                             name: int(np.sum(roles_array[:, audited] == code))
                             for name, code in ROLE_CODE.items()
                         }},
        "confirmation": confirmation_metrics, "gates": gates,
        "controls": {"candidate_swapped": True, "peak_evidence_permuted": True,
                     "same_operator_strength_topk": True},
        "direct_training_contract": {"teacher_score_loss": False, "teacher_embedding_loss": False,
                                     "clean_action_embedding_consistency": False,
                                     "action_is_second_query_view_with_same_identity_label": True},
        "provenance": {"graph_sha256": sha256_file(args.graph),
                       "teacher_report_sha256": sha256_file(args.teacher_dir / "report.json"),
                       "teacher_predictions_sha256": sha256_file(args.teacher_dir / "iceberg_predictions_f16.npy"),
                       "official_checkpoint_sha256": sha256_file(args.official_checkpoint)},
        "runtime_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    bank_path = args.output / "action_bank.npz"
    np.savez_compressed(bank_path, selected_query=selected, formula=formula,
                        action_fold=fold,
                        setting_mode=np.asarray([value[0] for value in SETTINGS]),
                        setting_strength=np.asarray([value[1] for value in SETTINGS], dtype=np.float32),
                        setting_top_k=np.asarray([value[2] for value in SETTINGS], dtype=np.int16),
                        old_rank=old_rank, old_margin=old_margin, ranks=ranks, margins=margins,
                        role_code=roles_array, selected_setting=np.asarray([setting_id], dtype=np.int16),
                        discovery_position=discovery, confirmation_position=confirmation,
                        embedding_evaluation_position=embedding_evaluation,
                        reserve_position=reserve,
                        direct_training_eligible=(eligible if passed else np.zeros_like(eligible)),
                        action_generation_seed=np.asarray([args.seed], dtype=np.int64))
    report["provenance"]["action_bank_sha256"] = sha256_file(bank_path)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
