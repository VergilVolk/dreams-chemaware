"""Recompute the E4 direct-training bottleneck from immutable local logs.

This is a zero-update audit.  It separates generic clean/listwise continuation
from action-specific signal and checks whether a successful action receives
more or less rank pressure under the historical E4 objective.  It also records
the objective/evaluator aggregation mismatch in the later candidate-boundary
implementation.  No teacher quality metric is used as a promotion endpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--causal-logs", nargs=3, type=Path,
        default=[
            ROOT / "noise_e4a_causal_2328357_0.out",
            ROOT / "noise_e4a_causal_2328357_1.out",
            ROOT / "noise_e4a_causal_2328357_2.out",
        ],
    )
    parser.add_argument(
        "--dynamic-log", type=Path,
        default=ROOT / "noise_dd_resume_2331284_2331352.out",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/noise_e4_direct_training_bottleneck_20260905",
    )
    return parser.parse_args()


def reports(path: Path, status: str) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    marker = '{\n  "status": "' + status + '"'
    decoder = json.JSONDecoder()
    output: list[dict] = []
    position = 0
    while True:
        try:
            start = text.index(marker, position)
        except ValueError:
            break
        value, consumed = decoder.raw_decode(text[start:])
        output.append(value)
        position = start + consumed
    return output


def rank_pressure(margin: float, rank_margin: float, temperature: float) -> float:
    """Magnitude of d softplus((gamma-margin)/tau) / d margin."""
    value = (rank_margin - margin) / temperature
    sigmoid = 1.0 / (1.0 + math.exp(-value))
    return sigmoid / temperature


def source_checks() -> dict[str, object]:
    e4_path = ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py"
    dynamic_path = ROOT / "tasks/train_noise_final_dynamic_direct_phase_a.py"
    boundary_path = ROOT / "tasks/noise_final_candidate_boundary_core.py"
    e4 = e4_path.read_text(encoding="utf-8")
    dynamic = dynamic_path.read_text(encoding="utf-8")
    boundary = boundary_path.read_text(encoding="utf-8")
    return {
        "historical_e4_loss_has_clean_and_aug_rank_but_no_target_control_delta": bool(
            "args.lambda_clean_rank * clean_rank" in e4
            and "args.lambda_aug_rank * aug_rank" in e4
            and "target_margin - control_margin" not in e4[
                e4.index("def direct_action_loss"):e4.index("def safety_loss")
            ]
        ),
        "historical_e4_clean_duplicate_doubles_clean_rank": bool(
            'elif arm == "clean_duplicate"' in e4
            and 'selected_paths.append("")' in e4
            and "attenuate_sequence(" in e4[
                e4.index("def flatten_direct"):e4.index("def flatten_pmt")
            ]
            and "args.lambda_clean_rank * clean_rank" in e4
            and "args.lambda_aug_rank * aug_rank" in e4
        ),
        "dynamic_loss_replaces_clean_mass_with_action_loss": bool(
            "augmented = no_op * clean_loss + torch.sum(weight_tensor * action_losses)" in dynamic
            and "args.lambda_clean_rank * clean_loss" in dynamic
            and "args.lambda_action_rank * augmented" in dynamic
        ),
        "dynamic_loss_has_no_action_to_clean_transfer_term": not bool(re.search(
            r"(target|action).*?(clean).*?(advantage|delta|inherit|transfer)",
            dynamic[dynamic.index("def query_loss"):dynamic.index("def main")],
            flags=re.IGNORECASE | re.DOTALL,
        )),
        "boundary_training_averages_all_positive_reference_edges": bool(
            "clean_i" in boundary
            and "torch.sum(weight * F.softplus" in boundary
            and "clean_margin.min(dim=0)" in boundary
        ),
        "boundary_evaluator_semantics_are_not_molecule_max_in_objective": bool(
            "clean_margin.min(dim=0)" in boundary
            and "torch.max" not in boundary[
                boundary.index("def candidate_boundary_objective"):
                boundary.index("def candidate_safety_objective")
            ]
        ),
        "files": {
            "historical_e4": str(e4_path),
            "dynamic_direct": str(dynamic_path),
            "candidate_boundary": str(boundary_path),
        },
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    causal: dict[str, dict] = {}
    for path in args.causal_logs:
        values = reports(path, "noise_final_e4a_direct_augmentation_complete")
        if len(values) != 1:
            raise RuntimeError(f"expected one causal report in {path}, found {len(values)}")
        arm = values[0]["configuration"]["causal_arm"]
        causal[str(arm)] = values[0]
    if set(causal) != {"clean_duplicate", "matched_random", "targeted"}:
        raise RuntimeError(f"causal arms incomplete: {sorted(causal)}")

    dynamic_all = reports(args.dynamic_log, "noise_final_dynamic_direct_phase_a_arm_complete")
    dynamic_formal = [value for value in dynamic_all if value.get("formal")]
    dynamic = {str(value["arm"]): value for value in dynamic_formal}
    expected = {"clean_continuation", "matched_random", "static_target", "dynamic_np"}
    if set(dynamic) != expected:
        raise RuntimeError(f"dynamic arms incomplete: {sorted(dynamic)}")

    target = causal["targeted"]
    clean = causal["clean_duplicate"]
    random = causal["matched_random"]
    config = target["configuration"]
    gamma = float(config["rank_margin"])
    tau = float(config["temperature"])
    epoch_rows = []
    for clean_h, random_h, target_h in zip(clean["history"], random["history"], target["history"]):
        clean_pressure = rank_pressure(float(target_h["action_clean_margin"]), gamma, tau)
        target_pressure = rank_pressure(float(target_h["action_aug_margin"]), gamma, tau)
        explicit_consistency = float(config["lambda_consistency"]) * float(
            target_h["action_consistency"]
        )
        rank_total = (
            float(config["lambda_clean_rank"]) * float(target_h["action_clean_rank"])
            + float(config["lambda_aug_rank"]) * float(target_h["action_aug_rank"])
        )
        epoch_rows.append({
            "epoch": int(target_h["epoch"]),
            "target_action_minus_clean_margin": float(
                target_h["action_aug_margin"] - target_h["action_clean_margin"]
            ),
            "matched_random_action_minus_clean_margin": float(
                random_h["action_aug_margin"] - random_h["action_clean_margin"]
            ),
            "target_action_rank_loss_minus_clean_duplicate_action_rank_loss": float(
                target_h["action_aug_rank"] - clean_h["action_aug_rank"]
            ),
            "target_action_rank_pressure_vs_its_clean_pressure_ratio": float(
                target_pressure / clean_pressure
            ),
            "weighted_consistency_over_clean_plus_action_rank": float(
                explicit_consistency / rank_total
            ),
            "gradient_clip_fraction": float(target_h["gradient_clip_applied"]),
            "gradient_clip_scale": float(target_h["gradient_clip_scale"]),
        })

    clean_net = int(clean["held_clean"]["corrected"] - clean["held_clean"]["introduced"])
    target_net = int(target["held_clean"]["corrected"] - target["held_clean"]["introduced"])
    dynamic_rows = []
    for epoch in range(4):
        row = {"epoch": epoch + 1}
        for arm in sorted(dynamic):
            item = dynamic[arm]["history"][epoch]
            row[arm] = {
                "clean_listwise": float(item["clean_listwise"]),
                "action_mixture_listwise": float(item["action_mixture_listwise"]),
                "action_minus_clean_loss": float(
                    item["action_mixture_listwise"] - item["clean_listwise"]
                ),
                "action_mass": float(item["action_mass"]),
                "mean_gradient_norm": float(item["mean_gradient_norm"]),
                "clip_fraction": float(item["clip_fraction"]),
            }
        dynamic_rows.append(row)

    report = {
        "status": "noise_e4_direct_training_bottleneck_audit_complete",
        "optimizer_steps": 0,
        "question": (
            "Why strong peak actions produce little independent clean-encoder gain under "
            "the historical direct fine-tuning kernels."
        ),
        "causal_e4": {
            "clean_duplicate_delta_pp": 100.0 * float(clean["held_clean"]["delta_recall1"]),
            "targeted_delta_pp": 100.0 * float(target["held_clean"]["delta_recall1"]),
            "targeted_minus_clean_duplicate_pp": 100.0 * float(
                target["held_clean"]["delta_recall1"] - clean["held_clean"]["delta_recall1"]
            ),
            "clean_duplicate_net_corrections": clean_net,
            "targeted_net_corrections": target_net,
            "fraction_targeted_net_already_explained_by_clean_duplicate": float(
                clean_net / target_net
            ),
            "epochs": epoch_rows,
        },
        "dynamic_direct": {
            "epochs": dynamic_rows,
            "held_delta_vs_initial_pp": {
                arm: 100.0 * float(value["held"]["delta_recall1_vs_initial"])
                for arm, value in dynamic.items()
            },
        },
        "source_contract_checks": source_checks(),
        "bottleneck_findings": [
            "E4 never optimizes the target-versus-control improvement as a clean-query objective.",
            "A successful action is an easier positive view, so the monotone rank loss gives it less gradient pressure.",
            "Clean duplicate receives essentially the same two rank terms and explains most net corrections.",
            "The dynamic kernel substitutes easy target loss for part of clean continuation; better action scores can reduce total update dose.",
            "Global clipping acts after branch summation and cannot restore information absent from the objective.",
            "The DEB edge loss emphasizes every positive-reference edge, whereas retrieval uses the best reference per molecule; this changes the optimized boundary.",
        ],
        "claim_limit": (
            "Zero-update local audit of existing logs and source. It diagnoses the training "
            "objective; it is not a new embedding or a teacher/distillation result."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
