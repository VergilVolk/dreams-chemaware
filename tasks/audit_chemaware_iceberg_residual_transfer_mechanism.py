"""Post-hoc mechanism audit of completed ICEBERG residual checkpoints.

No parameter is updated.  Each matched checkpoint is replayed on the frozen
2,048-query action graph.  The audit measures whether the 272 strict chemical
targets were actually fitted and decomposes score changes into query-only,
reference-only and shared query-plus-reference paths.  This separates an
optimization/injection failure from a held-formula generalization failure.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_direct_action_core import formula_bootstrap  # noqa: E402
from chemaware_direct_prior_objective import direct_prior_loss_and_gradient  # noqa: E402
from noise_final_core import sha256_file, strict_rank  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows  # noqa: E402


ARMS = ("clean_duplicate", "correct_alpha050", "structure_alpha050", "peak_alpha050")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--action-graph-dir", type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1",
    )
    parser.add_argument(
        "--token-dir", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--architecture-checkpoint", type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--huber", type=float, default=0.02)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args()


def exact_position(rows: np.ndarray) -> dict[int, int]:
    values = np.asarray(rows, dtype=np.int64)
    if len(np.unique(values)) != len(values):
        raise RuntimeError("embedding cache rows are not unique")
    return {int(row): index for index, row in enumerate(values)}


def candidate_score_vectors(
    body: dict[str, np.ndarray],
    eval_rows: np.ndarray,
    official_eval: np.ndarray,
    adapted_eval: np.ndarray,
) -> dict[str, np.ndarray]:
    """Score every molecule under both, query-only and reference-only paths."""
    local = exact_position(eval_rows)
    outputs = {
        "official": np.zeros(len(body["molecule_label"]), dtype=np.float32),
        "both": np.zeros(len(body["molecule_label"]), dtype=np.float32),
        "query_only": np.zeros(len(body["molecule_label"]), dtype=np.float32),
        "reference_only": np.zeros(len(body["molecule_label"]), dtype=np.float32),
    }
    for query, (left, right) in enumerate(zip(body["query_ptr"][:-1], body["query_ptr"][1:])):
        qpos = local[int(body["query_row"][query])]
        old_query = official_eval[qpos]
        new_query = adapted_eval[qpos]
        for molecule in range(int(left), int(right)):
            rleft, rright = map(int, body["molecule_ptr"][molecule : molecule + 2])
            position = np.asarray(
                [local[int(row)] for row in body["pair_candidate_row"][rleft:rright]],
                dtype=np.int64,
            )
            old_reference = official_eval[position]
            new_reference = adapted_eval[position]
            outputs["official"][molecule] = float(np.max(old_reference @ old_query))
            outputs["both"][molecule] = float(np.max(new_reference @ new_query))
            outputs["query_only"][molecule] = float(np.max(old_reference @ new_query))
            outputs["reference_only"][molecule] = float(np.max(new_reference @ old_query))
    return outputs


def centered_residual(
    student: np.ndarray, official: np.ndarray, query_ptr: np.ndarray
) -> np.ndarray:
    output = np.zeros_like(student, dtype=np.float64)
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left, right = int(left), int(right)
        block = np.asarray(student[left:right], dtype=np.float64) - np.asarray(
            official[left:right], dtype=np.float64
        )
        output[left:right] = block - np.mean(block)
    return output


def cosine_each_active(
    observed: np.ndarray,
    target: np.ndarray,
    query_ptr: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    values = []
    for query in np.flatnonzero(active):
        left, right = map(int, query_ptr[query : query + 2])
        x = observed[left:right]
        y = target[left:right]
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        values.append(0.0 if denominator == 0 else float(np.dot(x, y) / denominator))
    return np.asarray(values, dtype=np.float64)


def rank_and_boundary_metrics(
    student: np.ndarray,
    official: np.ndarray,
    body: dict[str, np.ndarray],
    active: np.ndarray,
    winning_negative: np.ndarray,
    target: np.ndarray,
    seed: int,
    draws: int,
) -> dict:
    old_rank, new_rank, observed_boundary, target_boundary, formula = [], [], [], [], []
    observed = centered_residual(student, official, body["query_ptr"])
    for selected_position, query in enumerate(np.flatnonzero(active)):
        left, right = map(int, body["query_ptr"][query : query + 2])
        label = body["molecule_label"][left:right]
        positive = int(np.flatnonzero(label == 1)[0])
        negative = int(winning_negative[query])
        old_rank.append(strict_rank(official[left:right]))
        new_rank.append(strict_rank(student[left:right]))
        observed_boundary.append(
            float(observed[left + positive] - observed[left + negative])
        )
        target_boundary.append(
            float(target[left + positive] - target[left + negative])
        )
        formula.append(str(body["query_formula"][query]))
    old_rank = np.asarray(old_rank, dtype=np.int64)
    new_rank = np.asarray(new_rank, dtype=np.int64)
    observed_boundary = np.asarray(observed_boundary)
    target_boundary = np.asarray(target_boundary)
    formula = np.asarray(formula)
    valid = target_boundary > 0
    ratio = observed_boundary[valid] / target_boundary[valid]
    return {
        "active_queries": int(np.sum(active)),
        "corrected": int(np.sum((old_rank != 1) & (new_rank == 1))),
        "still_wrong": int(np.sum(new_rank != 1)),
        "mean_observed_boundary_change": float(np.mean(observed_boundary)),
        "mean_target_boundary_change": float(np.mean(target_boundary)),
        "positive_boundary_change_fraction": float(np.mean(observed_boundary > 0)),
        "median_transfer_ratio": float(np.median(ratio)),
        "mean_transfer_ratio": float(np.mean(ratio)),
        "target_reached_fraction": float(np.mean(observed_boundary >= target_boundary)),
        "formula_bootstrap_observed_boundary_change": formula_bootstrap(
            observed_boundary, formula, seed, draws
        ),
    }


def safety_introductions(
    student: np.ndarray,
    official: np.ndarray,
    body: dict[str, np.ndarray],
    active: np.ndarray,
) -> dict:
    eligible, introduced = 0, 0
    for query, (left, right) in enumerate(zip(body["query_ptr"][:-1], body["query_ptr"][1:])):
        if active[query]:
            continue
        left, right = int(left), int(right)
        if strict_rank(official[left:right]) != 1:
            continue
        eligible += 1
        introduced += int(strict_rank(student[left:right]) != 1)
    return {"official_correct_nonaction_queries": eligible, "introduced": introduced}


def main() -> None:
    args = arguments()
    if args.alpha <= 0 or args.huber <= 0 or args.bootstrap_draws < 10_000:
        raise ValueError("invalid fixed mechanism-audit configuration")
    output = args.run_root / "transfer_mechanism_audit.json"
    if output.exists():
        raise FileExistsError(output)
    summary_path = args.run_root / "phase_a_summary.json"
    ledger_dir = args.run_root / "strict_corrective_ledger"
    required = [
        summary_path,
        ledger_dir / "report.json",
        ledger_dir / "ledger.npz",
        args.action_graph_dir / "graph.npz",
        args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy",
        args.data,
        args.official_checkpoint,
        args.architecture_checkpoint,
    ]
    required.extend(args.run_root / arm / "report.json" for arm in ARMS)
    required.extend(args.run_root / arm / "final_shared_encoder.pt" for arm in ARMS)
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    phase_a = json.loads(summary_path.read_text(encoding="utf-8"))
    if phase_a.get("status") != "CHEMAWARE_ICEBERG_RESIDUAL_SHARED_PHASE_A_DEVELOPMENT_CAUSAL_FAIL":
        raise RuntimeError("mechanism audit is reserved for the completed failed Phase A")

    with np.load(args.action_graph_dir / "graph.npz", allow_pickle=True) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    with np.load(ledger_dir / "ledger.npz", allow_pickle=False) as ledger:
        active = np.asarray(ledger["active_query"], dtype=bool)
        priors = {
            "correct": np.asarray(ledger["centered_residual"], dtype=np.float64),
            "structure_swapped": np.asarray(
                ledger["structure_swapped_centered_residual"], dtype=np.float64
            ),
            "peak_permuted": np.asarray(
                ledger["peak_permuted_centered_residual"], dtype=np.float64
            ),
        }
        selected_query = np.asarray(ledger["selected_teacher_query"], dtype=np.int64)
        selected_negative = np.asarray(
            ledger["selected_current_winning_negative"], dtype=np.int64
        )
    winning_negative = np.full(len(active), -1, dtype=np.int32)
    winning_negative[selected_query] = selected_negative
    if np.any(winning_negative[active] < 0):
        raise RuntimeError("active action lacks its frozen current winning negative")

    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    row_position = exact_position(rows)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    eval_rows = np.unique(np.r_[body["query_row"], body["pair_candidate_row"]]).astype(np.int64)
    eval_position = np.asarray([row_position[int(row)] for row in eval_rows], dtype=np.int64)
    official_eval = np.asarray(official[eval_position], dtype=np.float32)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("checkpoint mechanism audit requires one CUDA GPU")
    store = SpectrumStore(args.data, eval_rows, args.n_highest_peaks)

    scores_by_arm = {}
    arm_reports = {}
    for arm in ARMS:
        report = json.loads((args.run_root / arm / "report.json").read_text(encoding="utf-8"))
        checkpoint_path = args.run_root / arm / "final_shared_encoder.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("status") != "chemaware_iceberg_residual_shared_arm_checkpoint":
            raise RuntimeError(f"invalid checkpoint status: {arm}")
        model, _ = load_base_model(
            args.official_checkpoint,
            args.architecture_checkpoint,
            device,
            args.n_highest_peaks,
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        adapted = encode_rows(
            model, store, eval_rows, device, args.eval_batch_size, args.amp,
            f"mechanism-{arm}",
        )
        scores_by_arm[arm] = candidate_score_vectors(
            body, eval_rows, official_eval, adapted
        )
        arm_reports[arm] = report
        del model, checkpoint, adapted
        gc.collect()
        torch.cuda.empty_cache()

    official_score = scores_by_arm["clean_duplicate"]["official"].astype(np.float64)
    correct_target = args.alpha * priors["correct"]
    target_by_arm = {
        "clean_duplicate": np.zeros_like(correct_target),
        "correct_alpha050": correct_target,
        "structure_alpha050": args.alpha * priors["structure_swapped"],
        "peak_alpha050": args.alpha * priors["peak_permuted"],
    }
    audit = {}
    for arm in ARMS:
        pathways = {}
        for pathway in ("both", "query_only", "reference_only"):
            student = scores_by_arm[arm][pathway].astype(np.float64)
            observed = centered_residual(student, official_score, body["query_ptr"])
            own_loss, _ = direct_prior_loss_and_gradient(
                student,
                official_score,
                target_by_arm[arm] / args.alpha if arm != "clean_duplicate" else priors["correct"],
                body["query_ptr"],
                active,
                alpha=(args.alpha if arm != "clean_duplicate" else 0.0),
                huber_delta=args.huber,
            )
            correct_loss, _ = direct_prior_loss_and_gradient(
                student,
                official_score,
                priors["correct"],
                body["query_ptr"],
                active,
                alpha=args.alpha,
                huber_delta=args.huber,
            )
            correlation = cosine_each_active(
                observed, correct_target, body["query_ptr"], active
            )
            pathways[pathway] = {
                "own_target_huber": float(own_loss),
                "correct_target_huber": float(correct_loss),
                "correct_target_cosine_mean": float(np.mean(correlation)),
                "correct_target_cosine_median": float(np.median(correlation)),
                "correct_target_positive_cosine_fraction": float(np.mean(correlation > 0)),
                "boundary": rank_and_boundary_metrics(
                    student,
                    official_score,
                    body,
                    active,
                    winning_negative,
                    correct_target,
                    args.seed + 101 * ARMS.index(arm) + (
                        0 if pathway == "both" else 1 if pathway == "query_only" else 2
                    ),
                    args.bootstrap_draws,
                ),
                "safety": safety_introductions(
                    student, official_score, body, active
                ),
            }
        audit[arm] = {
            "pathways": pathways,
            "training_history": arm_reports[arm].get("history", []),
            "checkpoint_sha256": sha256_file(
                args.run_root / arm / "final_shared_encoder.pt"
            ),
        }

    clean = audit["clean_duplicate"]["pathways"]["both"]
    correct = audit["correct_alpha050"]["pathways"]["both"]
    controls = [
        audit[name]["pathways"]["both"]
        for name in ("structure_alpha050", "peak_alpha050")
    ]
    fit_better_than_clean = (
        correct["correct_target_huber"] < clean["correct_target_huber"]
    )
    alignment_better_than_controls = all(
        correct["correct_target_cosine_mean"] > control["correct_target_cosine_mean"]
        for control in controls
    )
    corrections_better_than_clean = (
        correct["boundary"]["corrected"] > clean["boundary"]["corrected"]
    )
    if fit_better_than_clean and alignment_better_than_controls and corrections_better_than_clean:
        diagnosis = "TRAIN_ACTION_FIT_PRESENT_HELD_FORMULA_GENERALIZATION_FAILED"
    elif not fit_better_than_clean and not corrections_better_than_clean:
        diagnosis = "DIRECT_RESIDUAL_INJECTION_FAILED_BEFORE_GENERALIZATION"
    else:
        diagnosis = "MIXED_TRANSFER_MECHANISM_REQUIRES_TARGETED_REDESIGN"
    report = {
        "status": "CHEMAWARE_ICEBERG_RESIDUAL_TRANSFER_MECHANISM_AUDIT_COMPLETE",
        "weights_updated": False,
        "retraining_performed": False,
        "diagnosis": diagnosis,
        "decision_facts": {
            "correct_target_fit_better_than_clean": bool(fit_better_than_clean),
            "correct_alignment_better_than_both_controls": bool(alignment_better_than_controls),
            "correct_active_corrections_exceed_clean": bool(corrections_better_than_clean),
            "held_formula_development_causal_pass": bool(
                phase_a.get("development_causal_chemistry_pass", False)
            ),
            "all_training_steps_were_gradient_clipped": all(
                arm_reports[arm].get("mean_clip_fraction") == 1.0 for arm in ARMS
            ),
        },
        "arm_mechanisms": audit,
        "provenance": {
            "phase_a_summary_sha256": sha256_file(summary_path),
            "ledger_sha256": sha256_file(ledger_dir / "ledger.npz"),
            "action_graph_sha256": sha256_file(args.action_graph_dir / "graph.npz"),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "diagnosis": diagnosis,
        "decision_facts": report["decision_facts"],
        "both_path_summary": {
            arm: audit[arm]["pathways"]["both"] for arm in ARMS
        },
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
