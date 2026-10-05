"""Formula-split audit of a cross-spectrum orthogonal ChemAware teacher.

For each held spectrum, utilities from other spectra of the same molecular
identity are aligned by candidate molecular identity and robustly aggregated.
One half of the fold-3 formulae selects the aggregation rule; the other half is
held for confirmation.  Identity labels group repeat observations only and are
never used to choose a candidate or compute its utility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval
from audit_chemaware_counterfactual_rule_kernel_natural import paired_rank_comparison
from chemaware_crossview_consensus_core import AGGREGATIONS, crossview_policy
from chemaware_iceberg_direct_core import stable_formula_folds


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("correct", "zero_contrast", "reversed_contrast", "alignment_permuted")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeat-ledger", type=Path,
        default=ROOT / "data/validation/chemaware_orthogonal_teacher_repeat_consistency_v2/repeat_ledger.npz",
    )
    parser.add_argument(
        "--teacher-report", type=Path,
        default=ROOT / "data/validation/chemaware_orthogonal_teacher_repeat_consistency_v2/report.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_crossview_consensus_teacher_v1",
    )
    parser.add_argument("--aggregations", nargs="+", choices=AGGREGATIONS, default=AGGREGATIONS)
    parser.add_argument("--min-context", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--threshold-multiplier", type=float, nargs="+", default=(0.50, 0.75, 1.0, 1.25, 1.50))
    parser.add_argument("--min-selected-formulas", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=20260913)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def active_formulas(selected: np.ndarray, formula: np.ndarray, mask: np.ndarray) -> int:
    active = mask & (np.asarray(selected) >= 0)
    return int(len(np.unique(np.asarray(formula).astype(str)[active])))


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_draws < 100 or args.min_selected_formulas < 1:
        raise ValueError("invalid audit size")
    if any(value <= 0 for value in args.min_context):
        raise ValueError("min-context must be positive")
    if any(value <= 0 for value in args.threshold_multiplier):
        raise ValueError("threshold multipliers must be positive")
    teacher_report = json.loads(args.teacher_report.read_text(encoding="utf-8"))
    if teacher_report.get("status") != "CHEMAWARE_ORTHOGONAL_TEACHER_REPEAT_CONSISTENCY_COMPLETE":
        raise RuntimeError("repeat teacher report is not complete")
    frozen_threshold = float(teacher_report["frozen_selection_replay"]["threshold"])
    with np.load(args.repeat_ledger, allow_pickle=False) as loaded:
        ledger = {key: np.asarray(loaded[key]) for key in loaded.files}
    required = {
        "query", "formula", "identity", "baseline_rank", "candidate_identity",
        "candidate_valid", "proposal_rank",
        *(f"{arm}_candidate_utility" for arm in ARMS),
    }
    if missing := required - set(ledger):
        raise RuntimeError(f"repeat ledger arrays missing: {sorted(missing)}")
    formula = ledger["formula"].astype(str)
    identity = ledger["identity"].astype(str)
    baseline = ledger["baseline_rank"].astype(np.int64)
    formula_role = stable_formula_folds(formula, 3, args.split_seed)
    screen = np.isin(formula_role, (0, 1)); confirmation = formula_role == 2
    if set(formula[screen]) & set(formula[confirmation]):
        raise RuntimeError("screen and confirmation formulas overlap")

    settings: list[dict[str, object]] = []
    ranks_by_setting: dict[tuple[str, int, float], dict[str, np.ndarray]] = {}
    selected_by_setting: dict[tuple[str, int, float], dict[str, np.ndarray]] = {}
    for aggregation in args.aggregations:
        for min_context in args.min_context:
            for multiplier in args.threshold_multiplier:
                key = (aggregation, int(min_context), float(multiplier))
                arm_ranks = {}; arm_selected = {}; arm_metrics = {}
                for arm in ARMS:
                    rank, selected, _score, _support = crossview_policy(
                        baseline, ledger["candidate_identity"], ledger["candidate_valid"],
                        ledger["proposal_rank"], ledger[f"{arm}_candidate_utility"], identity,
                        threshold=frozen_threshold * float(multiplier),
                        aggregation=aggregation, min_context=int(min_context),
                    )
                    arm_ranks[arm] = rank; arm_selected[arm] = selected
                    arm_metrics[arm] = retrieval(baseline[screen], rank[screen])
                correct = arm_metrics["correct"]
                control_risk = {
                    arm: int(arm_metrics[arm]["risk_utility_at_1"])
                    for arm in ARMS if arm != "correct"
                }
                selected_formulas = active_formulas(arm_selected["correct"], formula, screen)
                row = {
                    "aggregation": aggregation,
                    "min_context": int(min_context),
                    "threshold_multiplier": float(multiplier),
                    "threshold": float(frozen_threshold * multiplier),
                    "selected_formulas": selected_formulas,
                    "correct": correct,
                    "control_risk": control_risk,
                    "minimum_specific_risk_advantage": int(
                        min(int(correct["risk_utility_at_1"]) - value for value in control_risk.values())
                    ),
                }
                settings.append(row)
                ranks_by_setting[key] = arm_ranks
                selected_by_setting[key] = arm_selected

    eligible = [
        row for row in settings
        if row["selected_formulas"] >= args.min_selected_formulas
        and int(row["correct"]["risk_utility_at_1"]) >= 0
        and int(row["correct"]["corrected_at_1"]) >= 2 * int(row["correct"]["introduced_at_1"])
    ]
    if not eligible:
        raise RuntimeError("no cross-view consensus setting passed screen safety")
    selected_setting = max(
        eligible,
        key=lambda row: (
            int(row["minimum_specific_risk_advantage"]),
            int(row["correct"]["risk_utility_at_1"]),
            -int(row["correct"]["introduced_at_1"]),
            float(row["correct"]["delta_mrr"]),
            int(row["min_context"]),
            float(row["threshold_multiplier"]),
        ),
    )
    selected_key = (
        str(selected_setting["aggregation"]), int(selected_setting["min_context"]),
        float(selected_setting["threshold_multiplier"]),
    )
    held_ranks = ranks_by_setting[selected_key]
    held_selected = selected_by_setting[selected_key]
    held = {
        arm: {
            "retrieval": retrieval(baseline[confirmation], held_ranks[arm][confirmation]),
            "selected_formulas": active_formulas(held_selected[arm], formula, confirmation),
            "absolute_formula_bootstrap_ci95": bootstrap(
                formula[confirmation], baseline[confirmation], held_ranks[arm][confirmation],
                args.bootstrap_draws, args.split_seed + 100 + index,
            ),
        }
        for index, arm in enumerate(ARMS)
    }
    paired = {
        f"correct_minus_{arm}": paired_rank_comparison(
            held_ranks["correct"][confirmation], held_ranks[arm][confirmation],
            formula[confirmation], draws=args.bootstrap_draws,
            seed=args.split_seed + 200 + index,
        )
        for index, arm in enumerate(ARMS[1:])
    }
    correct = held["correct"]
    gates = {
        "screen_minimum_specific_advantage_positive": int(selected_setting["minimum_specific_risk_advantage"]) > 0,
        "confirmation_absolute_ci_positive": correct["absolute_formula_bootstrap_ci95"][0] > 0,
        "confirmation_corrected_exceeds_twice_introduced": (
            int(correct["retrieval"]["corrected_at_1"])
            > 2 * int(correct["retrieval"]["introduced_at_1"])
        ),
        **{
            f"confirmation_beats_{arm}_ci": paired[f"correct_minus_{arm}"][
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0
            for arm in ARMS[1:]
        },
        "outer_fold_untouched": True,
    }
    report = {
        "status": (
            "CHEMAWARE_CROSSVIEW_CONSENSUS_TEACHER_PASS"
            if all(gates.values()) else "CHEMAWARE_CROSSVIEW_CONSENSUS_TEACHER_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": False,
        "candidate_conditioned": True,
        "shared_embedding_result": False,
        "scope": (
            "Previously used inner fold 3 subdivided by formula for a repeat-consensus "
            "diagnostic; outer fold 4 remains untouched."
        ),
        "claim_limit": (
            "This validates or rejects a privileged multi-spectrum teacher. It does not "
            "establish a single-spectrum embedding gain."
        ),
        "method": {
            "name": "leave-one-spectrum-out candidate-identity consensus",
            "candidate_alignment": "molecular identity, never candidate slot",
            "teacher_input_for_each_held_spectrum": "other spectra of the same known training identity",
            "ground_truth_used_to_choose_candidate": False,
            "same_aggregation_and_threshold_all_arms": True,
            "frozen_single_spectrum_threshold": frozen_threshold,
        },
        "data": {
            "queries": int(len(formula)),
            "identities": int(len(np.unique(identity))),
            "screen_queries": int(np.sum(screen)),
            "confirmation_queries": int(np.sum(confirmation)),
            "screen_formulas": int(len(np.unique(formula[screen]))),
            "confirmation_formulas": int(len(np.unique(formula[confirmation]))),
            "formula_overlap": 0,
        },
        "selection": selected_setting,
        "selection_grid_size": int(len(settings)),
        "held_confirmation": held,
        "paired_confirmation": paired,
        "gates": gates,
        "provenance": {
            "repeat_ledger_sha256": sha256(args.repeat_ledger),
            "teacher_report_sha256": sha256(args.teacher_report),
        },
    }
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_crossview_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "confirmation_ranks.npz",
            query=ledger["query"][confirmation], formula=formula[confirmation],
            identity=identity[confirmation], baseline_rank=baseline[confirmation],
            **{f"{arm}_rank": held_ranks[arm][confirmation] for arm in ARMS},
            **{f"{arm}_selected": held_selected[arm][confirmation] for arm in ARMS},
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "selection": selected_setting,
        "held_confirmation": held, "paired_confirmation": paired,
        "gates": gates, "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
