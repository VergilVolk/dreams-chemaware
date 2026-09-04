"""Freeze the executable E15-M2 panel without changing any model weights.

M1 proves ledger fidelity.  M2 additionally requires a concrete executor for
every corrective action.  R0/A4/C1 carry their executor payload in the M1
ledger.  E14 actions are executable only when the exact positive-reference
rows and teacher margin pair can be joined from the immutable E14 sidecars.
No substitute reference rows are invented.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import json_dump, sha256_file  # noqa: E402
from noise_final_e15_calibration import diverse_panel  # noqa: E402


SOURCES = ("R0_N", "A4_exact", "C1_support_disjoint", "E14_mature_P")
KEY = ["source", "query_index", "action_id"]
E14_COLUMNS = [
    "query_index", "action_id", "positive_reference_rows",
    "teacher_positive_row", "teacher_hard_negative_row",
    "teacher_pair_clean_margin", "guided_family", "guided_dose",
    "guided_auxiliary_dose", "guided_recurrence_prevalence",
    "guided_recurrence_max_peaks", "guided_support_weighted",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, required=True)
    parser.add_argument("--e14-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--per-source-kind", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def enrich_e14(frame: pd.DataFrame, sidecar: pd.DataFrame) -> pd.DataFrame:
    """Inner-join E14 to exact executor metadata; never fabricate references."""
    if sidecar.empty:
        return frame.iloc[0:0].copy()
    missing = set(E14_COLUMNS) - set(sidecar.columns)
    if missing:
        raise RuntimeError(f"E14 executor sidecar misses {sorted(missing)}")
    sidecar = sidecar[E14_COLUMNS].copy()
    if sidecar.duplicated(["query_index", "action_id"]).any():
        raise RuntimeError("E14 executor sidecar repeats query/action keys")
    output = frame.merge(
        sidecar, on=["query_index", "action_id"], how="inner",
        validate="one_to_one", suffixes=("", "_exact"),
    )
    output["executor"] = "E14_exact_references"
    return output


def executable(frame: pd.DataFrame, e14_sidecar: pd.DataFrame) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for source, block in frame.groupby("source", sort=True):
        block = block.copy()
        if source == "E14_mature_P":
            block = enrich_e14(block, e14_sidecar)
        elif source == "R0_N":
            block["executor"] = "R0_peak_path"
        elif source == "A4_exact":
            block["executor"] = "A4_exact_token"
        elif source == "C1_support_disjoint":
            block["executor"] = "C1_support_disjoint_prototype"
        else:
            raise RuntimeError(f"unregistered E15 source: {source}")
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True) if blocks else frame.iloc[0:0].copy()


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite E15-M2 panel: {args.output_dir}")
    required = {
        "m1_report": args.m1_dir / "report.json",
        "corrective": args.m1_dir / "calibrated_corrective_actions.csv.gz",
        "harmful": args.m1_dir / "calibrated_harmful_actions.csv.gz",
        "e14_selected": args.e14_dir / "selected_actions.csv.gz",
        "e14_risk": args.e14_dir / "risk_controls.csv.gz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    m1 = json.loads(required["m1_report"].read_text(encoding="utf-8"))
    if (
        m1.get("status") != "noise_final_e15_m1_replay_calibration_complete"
        or not m1.get("formal") or not m1.get("pass_to_32_query_overfit")
        or int(m1.get("outer_formula_fold", -1)) != args.outer_fold
    ):
        raise RuntimeError("M2 requires a passing formal E15-M1 artifact")

    corrective_raw = read_csv(required["corrective"])
    harmful_raw = read_csv(required["harmful"])
    if not corrective_raw["supervision_kind"].astype(str).eq("corrective").all():
        raise RuntimeError("corrective ledger contains another supervision kind")
    if not harmful_raw["supervision_kind"].astype(str).eq("harmful").all():
        raise RuntimeError("harmful ledger contains another supervision kind")

    selected = read_csv(required["e14_selected"])
    selected["source"] = "E14_mature_P"
    risk = read_csv(required["e14_risk"])
    risk = risk.loc[risk["control_kind"].astype(str).eq("introduced")].copy()
    risk["source"] = "E14_mature_P"
    corrective = executable(corrective_raw, selected)
    harmful = executable(harmful_raw, risk)
    for name, frame in (("corrective", corrective), ("harmful", harmful)):
        if frame.empty or frame.duplicated(KEY).any():
            raise RuntimeError(f"{name} executable ledger is empty or duplicated")
        if set(frame["source"].astype(str)) != set(SOURCES):
            raise RuntimeError(f"{name} does not retain all four sources")

    joined = pd.concat([corrective, harmful], ignore_index=True)
    calibration = diverse_panel(joined, args.per_source_kind, args.seed)
    expected = pd.MultiIndex.from_product([SOURCES, ("corrective", "harmful")])
    counts = calibration.groupby(["source", "supervision_kind"]).size()
    if not expected.isin(counts.index).all() or not (counts.loc[expected] == args.per_source_kind).all():
        raise RuntimeError("M2 calibration panel is not exactly source/kind balanced")
    calibration_query_counts = calibration.groupby(
        ["source", "supervision_kind"]
    )["query_index"].nunique()
    calibration_identity_counts = calibration.groupby(
        ["source", "supervision_kind"]
    )["query_ik14"].nunique()
    if not (
        (calibration_query_counts.loc[expected] == args.per_source_kind).all()
        and (calibration_identity_counts.loc[expected] == args.per_source_kind).all()
    ):
        raise RuntimeError(
            "M2 calibration panel must contain distinct queries and identities, "
            "not merely distinct action rows"
        )

    # Candidate pool is deliberately broader than the final 32-query overfit
    # panel.  The trainer freezes current-initialization ranks first, then takes
    # eight current errors per source.  That is a capacity test, not a held-set
    # estimate, and the selection is written to the result ledger.
    source_counts = {
        kind: frame.groupby("source").agg(
            actions=("action_id", "size"), queries=("query_index", "nunique"),
            identities=("query_ik14", "nunique"), formulas=("query_formula", "nunique"),
        ).astype(int).to_dict("index")
        for kind, frame in (("corrective", corrective), ("harmful", harmful))
    }
    source_query_action_profiles = {
        kind: {
            str(source): {
                "queries": int(len(counts)),
                "multiaction_queries": int(counts.ge(2).sum()),
                "maximum_actions_per_query": int(counts.max()),
            }
            for source, counts in frame.groupby(["source", "query_index"]).size().groupby(level=0)
        }
        for kind, frame in (("corrective", corrective), ("harmful", harmful))
    }
    gates = {
        "all_four_sources_corrective": set(corrective["source"].astype(str)) == set(SOURCES),
        "all_four_sources_harmful": set(harmful["source"].astype(str)) == set(SOURCES),
        "calibration_has_128_unique_actions": bool(
            len(calibration) == 8 * args.per_source_kind == 128
            and not calibration.duplicated(KEY + ["supervision_kind"]).any()
        ),
        "calibration_has_128_unique_source_kind_queries": bool(
            (calibration_query_counts.loc[expected] == args.per_source_kind).all()
        ),
        "calibration_has_128_unique_source_kind_identities": bool(
            (calibration_identity_counts.loc[expected] == args.per_source_kind).all()
        ),
        "E14_corrective_exact_rows_ge_100": int((corrective["source"] == "E14_mature_P").sum()) >= 100,
        "E14_harmful_exact_rows_ge_25": int((harmful["source"] == "E14_mature_P").sum()) >= 25,
        "P2b_forbidden": True,
        "P3_not_consumed": True,
    }
    report = {
        "status": "noise_final_e15_m2_executable_panel_complete",
        "formal": True,
        "outer_formula_fold": int(args.outer_fold),
        "corrective_actions": int(len(corrective)),
        "harmful_actions": int(len(harmful)),
        "source_counts": source_counts,
        "source_query_action_profiles": source_query_action_profiles,
        "calibration_actions": int(len(calibration)),
        "calibration_microbatches_at_batch4": int(len(calibration) // 4),
        "E14_nonexecutable_corrective_rows": int(
            (corrective_raw["source"] == "E14_mature_P").sum()
            - (corrective["source"] == "E14_mature_P").sum()
        ),
        "E14_nonexecutable_harmful_rows": int(
            (harmful_raw["source"] == "E14_mature_P").sum()
            - (harmful["source"] == "E14_mature_P").sum()
        ),
        "gates": gates,
        "pass_to_shared_encoder_overfit": bool(all(gates.values())),
        "contracts": {
            "unexecutable_E14_actions_are_reported_not_substituted": True,
            "multiple_actions_are_retained": True,
            "harmful_actions_are_routing_labels_only": True,
            "no_op_is_implemented_as_clean_initialization_protection": True,
            "P2b": "forbidden", "P3_consumed": False,
        },
        "provenance": {name: sha256_file(path) for name, path in required.items()},
        "claim_limit": "Executable training panel only; no trained embedding result.",
    }
    if not report["pass_to_shared_encoder_overfit"]:
        raise RuntimeError(f"E15-M2 executable gates failed: {gates}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="noise_e15_m2_panel_", dir=args.output_dir.parent))
    try:
        corrective.to_csv(staging / "executable_corrective.csv.gz", index=False, compression="gzip")
        harmful.to_csv(staging / "executable_harmful.csv.gz", index=False, compression="gzip")
        calibration.to_csv(staging / "gradient_panel.csv.gz", index=False, compression="gzip")
        json_dump(staging / "report.json", report)
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
