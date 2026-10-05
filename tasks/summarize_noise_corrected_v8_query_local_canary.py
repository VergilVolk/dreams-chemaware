"""Summarize the four-arm V8 query-local direct-injection canary.

This is deliberately a development decision, not a promotion result.  It
separates action semantic locality from generic continuation and from the V7
shared-reference corrective residual on exactly aligned held queries.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import tempfile

ARMS = (
    "query_local_routed",
    "query_local_shuffled",
    "shared_routed",
    "clean_control",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm.replace('_', '-')}-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _parse_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"true", "1"}:
        return True
    if lowered in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _load(path: Path) -> tuple[dict, list[dict[str, object]]]:
    decision_path = path / "decision.json"
    paired_path = path / "held_per_query.csv.gz"
    if not decision_path.is_file() or not paired_path.is_file():
        raise FileNotFoundError(f"incomplete canary arm: {path}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    with gzip.open(paired_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        raw_rows = list(reader)
    required = {
        "query_index", "query_formula", "near", "initial_E8_rank",
        "candidate_rank", "corrected", "introduced",
    }
    if missing := required - columns:
        raise RuntimeError(f"held table misses {sorted(missing)}: {path}")
    paired = [{
        "query_index": int(row["query_index"]),
        "query_formula": str(row["query_formula"]),
        "near": _parse_bool(row["near"]),
        "initial_E8_rank": int(row["initial_E8_rank"]),
        "candidate_rank": int(row["candidate_rank"]),
    } for row in raw_rows]
    if decision.get("formal") is not False:
        raise RuntimeError("V8 canary accepted a formal arm")
    return decision, paired


def _paired_arm_delta(
    baseline: list[dict[str, object]], candidate: list[dict[str, object]],
) -> dict[str, float | int]:
    keys = ["query_index", "query_formula", "near", "initial_E8_rank"]
    if (
        [[row[key] for key in keys] for row in baseline]
        != [[row[key] for key in keys] for row in candidate]
    ):
        raise RuntimeError("V8 canary arm held ledgers are not exactly aligned")
    if not candidate:
        raise RuntimeError("V8 canary held ledger is empty")
    before = [int(row["candidate_rank"]) for row in baseline]
    after = [int(row["candidate_rank"]) for row in candidate]
    corrected = [left > 1 and right == 1 for left, right in zip(before, after)]
    introduced = [left == 1 and right > 1 for left, right in zip(before, after)]
    near = [bool(row["near"]) for row in candidate]
    near_count = sum(near)
    corrected_count = sum(corrected)
    introduced_count = sum(introduced)
    return {
        "queries": int(len(candidate)),
        "delta_recall1_pp": float(100.0 * (
            sum(value == 1 for value in after)
            - sum(value == 1 for value in before)
        ) / len(candidate)),
        "corrected": int(corrected_count),
        "introduced": int(introduced_count),
        "risk_net_lambda2": int(corrected_count - 2 * introduced_count),
        "near_delta_recall1_pp": float(
            100.0 * (
                sum(right == 1 for right, selected in zip(after, near) if selected)
                - sum(left == 1 for left, selected in zip(before, near) if selected)
            ) / near_count if near_count else 0.0
        ),
        "near_corrected": int(sum(
            value and selected for value, selected in zip(corrected, near)
        )),
        "near_introduced": int(sum(
            value and selected for value, selected in zip(introduced, near)
        )),
    }


def _role_gate(decision: dict, expected: str) -> dict[str, object]:
    # The trainer's frozen output contract names this block
    # ``gradient_calibration``.  The legacy fallback remains read-only support
    # for any early development artifact created before that contract was
    # audited; new jobs must exercise the primary path.
    calibration = decision.get("gradient_calibration")
    if calibration is None:
        calibration = decision.get("calibration", {})
    role = calibration.get(
        "corrective_embedding_role_energy", {}
    )
    observed = role.get("corrective_gradient_locality")
    if observed != expected:
        raise RuntimeError(
            f"corrective locality drifted: observed={observed}, expected={expected}"
        )
    return {
        "locality": observed,
        "reference_gradient_exact_zero": role.get(
            "corrective_reference_gradient_exact_zero"
        ),
        "required_clean_and_action_paths_live": role.get(
            "required_clean_and_action_paths_live"
        ),
        "query_action_only_gate_passed": role.get(
            "query_action_only_locality_gate_passed"
        ),
        "transfer_reference_dominates_query_energy": role.get(
            "transfer_reference_dominates_query_energy"
        ),
        "payload_reference_dominates_action_energy": role.get(
            "payload_reference_dominates_action_energy"
        ),
        "branches": role.get("branches", {}),
    }


def summarize(paths: dict[str, Path]) -> dict[str, object]:
    loaded = {arm: _load(paths[arm]) for arm in ARMS}
    decisions = {arm: value[0] for arm, value in loaded.items()}
    tables = {arm: value[1] for arm, value in loaded.items()}
    query_local_routed = decisions["query_local_routed"]
    query_local_shuffled = decisions["query_local_shuffled"]
    shared_routed = decisions["shared_routed"]
    clean = decisions["clean_control"]
    expected_arm = {
        "query_local_routed": "routed_direct",
        "query_local_shuffled": "shuffled_action_control",
        "shared_routed": "routed_direct",
        "clean_control": "clean_control",
    }
    for name, decision in decisions.items():
        if decision.get("arm") != expected_arm[name]:
            raise RuntimeError(f"V8 canary arm label drifted: {name}")
        if decision.get("corrective_objective_mode") != "v3_direct":
            raise RuntimeError("V8 canary escaped the direct objective")
        if decision.get("contracts", {}).get(
            "teacher_embedding_or_distillation_target_used"
        ) is not False:
            raise RuntimeError("V8 canary used a teacher/distillation target")
    role = {
        "query_local_routed": _role_gate(query_local_routed, "query_action_only"),
        "query_local_shuffled": _role_gate(
            query_local_shuffled, "query_action_only"
        ),
        "shared_routed": _role_gate(shared_routed, "shared"),
    }
    pairwise = {
        "query_local_routed_vs_query_local_shuffled": _paired_arm_delta(
            tables["query_local_shuffled"], tables["query_local_routed"]
        ),
        "query_local_routed_vs_shared_routed": _paired_arm_delta(
            tables["shared_routed"], tables["query_local_routed"]
        ),
        "query_local_routed_vs_clean_control": _paired_arm_delta(
            tables["clean_control"], tables["query_local_routed"]
        ),
    }
    semantic = pairwise["query_local_routed_vs_query_local_shuffled"]
    locality = pairwise["query_local_routed_vs_shared_routed"]
    clean_delta = pairwise["query_local_routed_vs_clean_control"]
    role_gate = bool(
        role["query_local_routed"]["query_action_only_gate_passed"] is True
        and role["query_local_shuffled"]["query_action_only_gate_passed"] is True
        and role["shared_routed"]["reference_gradient_exact_zero"] is False
    )
    semantic_gate = bool(
        semantic["delta_recall1_pp"] > 0
        and semantic["risk_net_lambda2"] > 0
    )
    locality_gain_gate = bool(
        locality["delta_recall1_pp"] > 0
        and clean_delta["delta_recall1_pp"] > 0
    )
    signal = {
        arm: {
            "action_retention_p10": decision.get("signal_transmission", {}).get(
                "action_retention_p10"
            ),
            "optimizer_action_attributable_update_fraction_p10": decision.get(
                "signal_transmission", {}
            ).get("optimizer_action_attributable_update_fraction_p10"),
            "legacy_90pct_end_to_end_loss_reproduced": decision.get(
                "signal_transmission", {}
            ).get("legacy_90pct_end_to_end_loss_reproduced"),
            "all_signal_gates_passed": decision.get(
                "signal_transmission", {}
            ).get("all_signal_gates_passed"),
        }
        for arm, decision in decisions.items()
    }
    optimizer_gate = bool(all(
        signal[arm]["all_signal_gates_passed"] is True
        for arm in ("query_local_routed", "query_local_shuffled")
    ))
    return {
        "status": "noise_corrected_v8_query_local_canary_complete",
        "formal": False,
        "training_or_embedding_promotion": False,
        "role_locality": role,
        "paired_clean_held_comparisons": pairwise,
        "signal_transmission": signal,
        "gates": {
            "corrective_reference_bypass_removed": role_gate,
            "routed_semantics_beat_matched_shuffle": semantic_gate,
            "query_locality_beats_shared_and_clean": locality_gain_gate,
            "optimizer_signal_gate": optimizer_gate,
            "advance_to_full_candidate": bool(
                role_gate and semantic_gate and locality_gain_gate and optimizer_gate
            ),
        },
        "interpretation": (
            "Role locality, routed-vs-shuffled causality, locality-vs-shared gain, "
            "and optimizer transmission are independent gates. No single canary "
            "metric authorizes a formal full run."
        ),
        "claim_limit": (
            "Bounded outer-formula-held development canary; not a 4 pp claim, "
            "not a promoted encoder, and not a teacher/distillation experiment."
        ),
    }


def main() -> None:
    args = arguments()
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    paths = {
        arm: getattr(args, f"{arm}_dir") for arm in ARMS
    }
    report = summarize(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
