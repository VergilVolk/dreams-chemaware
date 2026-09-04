"""Fail-closed validator for the L0 full-candidate action ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_noise_final_l0_action_learnability_ledger import (
    EXPECTED_R0_STATUS, advantage_label, transition_label,
)
from noise_final_core import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_l0_action_learnability_ledger",
    )
    parser.add_argument(
        "--r0-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    required = {
        "report": args.output_dir / "report.json",
        "labels": args.output_dir / "action_labels.csv.gz",
        "cells": args.output_dir / "cell_summary.csv",
        "r0_report": args.r0_dir / "report.json",
        "r0_actions": args.r0_dir / "training_actions.csv.gz",
        "r0_historical": args.r0_dir / "outcome_audit_only.csv.gz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"L0 output is incomplete: {missing}")
    report = json.loads(required["report"].read_text(encoding="utf-8"))
    r0_report = json.loads(required["r0_report"].read_text(encoding="utf-8"))
    if report.get("status") != "noise_final_l0_action_learnability_ledger_complete":
        raise RuntimeError("wrong L0 report status")
    if not report.get("formal"):
        raise RuntimeError("validator only accepts a formal L0 run")
    if r0_report.get("status") != EXPECTED_R0_STATUS or not r0_report.get("formal"):
        raise RuntimeError("R0 source is not formal")
    provenance = report.get("provenance", {})
    if provenance.get("r0_report_sha256") != sha256_file(required["r0_report"]):
        raise RuntimeError("R0 report changed after L0")
    if provenance.get("r0_actions_sha256") != sha256_file(required["r0_actions"]):
        raise RuntimeError("R0 actions changed after L0")
    if provenance.get("r0_historical_outcomes_sha256") != sha256_file(required["r0_historical"]):
        raise RuntimeError("R0 historical outcomes changed after L0")

    labels = pd.read_csv(required["labels"], low_memory=False)
    source = pd.read_csv(required["r0_actions"], low_memory=False)
    cells = pd.read_csv(required["cells"], low_memory=False)
    required_columns = {
        "action_index", "query_index", "query_ik14", "query_formula", "selector",
        "attenuation", "step", "baseline_rank", "baseline_margin", "target_rank",
        "target_margin", "control0_rank", "control0_margin", "control1_rank",
        "control1_margin", "target_margin_gain", "control0_margin_gain",
        "control1_margin_gain", "control_mean_margin_gain", "paired_advantage",
        "advantage_label", "transition", "historical_official_baseline_rank",
        "historical_official_target_rank", "historical_official_paired_advantage",
    }
    if missing_columns := required_columns - set(labels.columns):
        raise RuntimeError(f"L0 labels miss columns: {sorted(missing_columns)}")
    if len(labels) != len(source) or len(labels) != int(report.get("actions", -1)):
        raise RuntimeError("L0 does not contain exactly one label per R0 action")
    if len(source) != int(r0_report.get("training_action_rows", -1)):
        raise RuntimeError("formal R0 action count changed")
    if labels["action_index"].duplicated().any() or not np.array_equal(
        labels["action_index"].to_numpy(np.int64), np.arange(len(labels)),
    ):
        raise RuntimeError("L0 action index is not unique and contiguous")
    keys = ["query_index", "selector", "attenuation", "step"]
    if not labels[keys].equals(source[keys]):
        raise RuntimeError("L0 action order/identity differs from R0")
    numeric = labels[[
        "baseline_margin", "target_margin", "control0_margin", "control1_margin",
        "target_margin_gain", "control0_margin_gain", "control1_margin_gain",
        "control_mean_margin_gain", "paired_advantage",
    ]].to_numpy(float)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("L0 contains non-finite outcomes")
    expected_advantage = (
        labels["target_margin"]
        - (labels["control0_margin"] + labels["control1_margin"]) / 2.0
    )
    if not np.allclose(labels["paired_advantage"], expected_advantage, atol=2e-7):
        raise RuntimeError("L0 paired-advantage replay failed")
    threshold = float(report["advantage_threshold"])
    expected_labels = [advantage_label(value, threshold) for value in labels["paired_advantage"]]
    if expected_labels != labels["advantage_label"].astype(str).tolist():
        raise RuntimeError("L0 advantage labels are inconsistent")
    expected_transitions = [
        transition_label(old, new)
        for old, new in zip(labels["baseline_rank"], labels["target_rank"])
    ]
    if expected_transitions != labels["transition"].astype(str).tolist():
        raise RuntimeError("L0 transitions are inconsistent")
    if len(cells) != labels.groupby(keys[1:], sort=True).ngroups:
        raise RuntimeError("L0 cell summary does not cover every action cell")
    if int(report.get("queries", -1)) != labels["query_index"].nunique():
        raise RuntimeError("L0 query coverage summary is inconsistent")
    contracts = report.get("contracts", {})
    expected_contracts = {
        "complete_candidate_list_scored": True,
        "query_and_reference_encoder_shared": True,
        "target_and_two_frozen_matched_controls_scored": True,
        "local_positive_negative_surrogate_used": False,
        "optimizer_steps": 0,
        "outcome_labels_are_not_features": True,
        "next_stage_features_must_be_clean_query_visible": True,
        "formula_crossfit_required_next": True,
        "P2b": "forbidden",
        "P3_consumed": False,
    }
    if any(contracts.get(key) != value for key, value in expected_contracts.items()):
        raise RuntimeError("L0 scientific contract mismatch")
    print(
        f"[validate_noise_final_l0_action_learnability_ledger] PASS "
        f"actions={len(labels):,} cells={len(cells)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
