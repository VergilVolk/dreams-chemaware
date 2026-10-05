"""CPU-only post-hoc specificity audit for a completed ChemAware A2 screen.

The script reads the immutable per-action ledger and action bank.  It never
loads DreaMS, computes gradients, updates weights, or turns matched controls
into optimizer targets.  With no arguments it resolves the newest completed
``run_*/screen24`` directory and records that resolved path and every input
hash in the output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from chemaware_direct_action_core import ROLE_CODE, formula_bootstrap


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("correct", "candidate_swapped", "peak_permuted")
PRIMARY_METRICS = (
    "clean_margin_gain_per_unit_update_norm",
    "predicted_clean_margin_gain",
    "raw_gradient_norm",
    "learning_rate_scaled_update_norm",
    "clean_margin_descent_cosine",
    "clean_loss_decrease_per_unit_update_norm",
    "clean_loss_gradient_cosine",
)
PAIR_ROUTE = "action_routed_clean_pair"
FORWARD_ROUTE = "action_query_only"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", type=Path, default=None)
    parser.add_argument(
        "--action-bank", type=Path,
        default=ROOT / "data/validation/chemaware_direct_action_bank_v1",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260906)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_screen(path: Path | None) -> Path:
    if path is not None:
        candidates = [path.resolve()]
    else:
        base = ROOT / "data/validation/chemaware_action_transfer_gradients"
        candidates = [
            item.resolve() for item in base.glob("run_*/screen24")
            if (item / "report.json").is_file()
            and (item / "per_action.csv").is_file()
            and (item / "COMPLETE.json").is_file()
        ]
        candidates.sort(
            key=lambda item: ((item / "report.json").stat().st_mtime_ns, str(item))
        )
        candidates = candidates[-1:]
    if not candidates:
        raise FileNotFoundError(
            "no completed run_*/screen24 directory was found; copy the completed "
            "screen directory locally or pass --screen-dir"
        )
    screen = candidates[0]
    missing = [
        name for name in ("report.json", "per_action.csv", "COMPLETE.json")
        if not (screen / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"specificity screen is incomplete: {missing}")
    return screen


def optional_float(value: str | None) -> float | None:
    if value is None or value.strip() in {"", "None", "null"}:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RuntimeError(f"non-finite numeric value in per-action ledger: {value}")
    return parsed


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "action_position", "formula", "route", "arm", "route_forward_margin",
        "positive_reference", "negative_reference", *PRIMARY_METRICS,
    }
    if not rows or required - set(rows[0]):
        raise RuntimeError(
            f"per-action ledger is empty or lacks columns: {sorted(required - set(rows[0] if rows else []))}"
        )
    numeric = {
        "route_forward_margin", "positive_reference", "negative_reference",
        *PRIMARY_METRICS,
    }
    output = []
    for row in rows:
        parsed = dict(row)
        parsed["action_position"] = int(row["action_position"])
        for name in numeric:
            parsed[name] = optional_float(row[name])
        output.append(parsed)
    return output


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and values[order[right]] == values[order[left]]:
            right += 1
        ranks[order[left:right]] = 0.5 * (left + right - 1) + 1.0
        left = right
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(np.corrcoef(average_ranks(left), average_ranks(right))[0, 1])
    return value if math.isfinite(value) else None


def distribution(values: np.ndarray, formulas: np.ndarray, seed: int, draws: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("specificity distribution contains non-finite values")
    body = formula_bootstrap(values, formulas, seed, draws)
    body.update({
        "query_mean": float(np.mean(values)),
        "query_median": float(np.median(values)),
        "positive_fraction": float(np.mean(values > 0)),
    })
    return body


def analyze(screen: Path, action_bank: Path, draws: int, seed: int) -> dict:
    if draws < 100:
        raise ValueError("bootstrap-draws must be at least 100")
    report_path = screen / "report.json"
    csv_path = screen / "per_action.csv"
    bank_path = action_bank / "action_bank.npz"
    bank_report_path = action_bank / "report.json"
    for path in (bank_path, bank_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = json.loads(report_path.read_text(encoding="utf-8"))
    complete = json.loads((screen / "COMPLETE.json").read_text(encoding="utf-8"))
    if source.get("weights_updated") is not False or source.get("optimizer_steps") != 0:
        raise RuntimeError("source is not a zero-update A2 mechanism screen")
    if complete.get("status") != source.get("status"):
        raise RuntimeError("COMPLETE marker and source report status disagree")
    expected_bank_hash = source.get("preflight", {}).get("provenance", {}).get(
        "action_bank_sha256"
    )
    actual_bank_hash = sha256_file(bank_path)
    if expected_bank_hash != actual_bank_hash:
        raise RuntimeError("action bank hash differs from the completed A2 screen")

    rows = read_rows(csv_path)
    positions = sorted({int(row["action_position"]) for row in rows})
    routes = list(source.get("routes", {}))
    expected_keys = {
        (position, route, arm)
        for position in positions for route in routes for arm in ARMS
    }
    actual_keys = {
        (int(row["action_position"]), str(row["route"]), str(row["arm"]))
        for row in rows
    }
    if len(actual_keys) != len(rows) or actual_keys != expected_keys:
        raise RuntimeError("per-action ledger does not form a unique position/route/arm cube")
    reported_positions = source.get("preflight", {}).get("positions", [])
    if positions != sorted(map(int, reported_positions)):
        raise RuntimeError("per-action positions differ from the completed preflight")

    with np.load(bank_path, allow_pickle=False) as payload:
        required = {"action_fold", "role_code", "selected_setting", "formula"}
        if required - set(payload.files):
            raise RuntimeError(f"action bank lacks arrays: {sorted(required - set(payload.files))}")
        action_fold = np.asarray(payload["action_fold"], dtype=np.int64)
        role_cube = np.asarray(payload["role_code"], dtype=np.int64)
        selected_setting = int(np.asarray(payload["selected_setting"]).reshape(-1)[0])
        bank_formula = np.asarray(payload["formula"]).astype(str)
    if role_cube.ndim != 2 or not 0 <= selected_setting < len(role_cube):
        raise RuntimeError("action-bank role matrix or selected setting is invalid")
    if max(positions) >= len(action_fold) or len(action_fold) != len(bank_formula):
        raise RuntimeError("audited action position lies outside action bank")
    reverse_role = {code: name for name, code in ROLE_CODE.items()}
    selected_role = role_cube[selected_setting]

    bank_report = json.loads(bank_report_path.read_text(encoding="utf-8"))
    split = bank_report.get("split", {})
    discovery = set(map(int, split.get("discovery_folds", (0, 1))))
    confirmation = int(split.get("confirmation_fold", 2))

    by_key = {
        (int(row["action_position"]), str(row["route"]), str(row["arm"])): row
        for row in rows
    }
    formulas = np.asarray([str(by_key[(p, routes[0], ARMS[0])]["formula"]) for p in positions])
    if not np.array_equal(formulas, bank_formula[positions]):
        raise RuntimeError("per-action formula labels drifted from the action bank")
    fold_labels = np.asarray([
        "discovery" if int(action_fold[p]) in discovery else
        "confirmation" if int(action_fold[p]) == confirmation else
        f"other_fold_{int(action_fold[p])}"
        for p in positions
    ])
    role_labels = np.asarray([
        reverse_role.get(int(selected_role[p]), f"unknown_{int(selected_role[p])}")
        for p in positions
    ])

    def values(route: str, arm: str, metric: str) -> np.ndarray:
        result = [by_key[(p, route, arm)][metric] for p in positions]
        if any(value is None for value in result):
            raise RuntimeError(f"{route}/{arm}/{metric} contains missing values")
        return np.asarray(result, dtype=np.float64)

    forward_route = FORWARD_ROUTE if FORWARD_ROUTE in routes else routes[0]
    forward_margin = {
        arm: values(forward_route, arm, "route_forward_margin") for arm in ARMS
    }
    route_diagnostics = {}
    for route_index, route in enumerate(routes):
        metric_reports = {}
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            arm_values = {arm: values(route, arm, metric) for arm in ARMS}
            controls = {}
            for control_index, control in enumerate(ARMS[1:]):
                difference = arm_values["correct"] - arm_values[control]
                controls[control] = distribution(
                    difference, formulas,
                    seed + 10_000 * route_index + 100 * metric_index + control_index,
                    draws,
                )
            metric_reports[metric] = {
                "arms": {
                    arm: distribution(
                        arm_values[arm], formulas,
                        seed + 20_000 * route_index + 200 * metric_index + arm_index,
                        draws,
                    )
                    for arm_index, arm in enumerate(ARMS)
                },
                "correct_minus_controls": controls,
            }

        influence = {
            arm: values(route, arm, "clean_margin_gain_per_unit_update_norm")
            for arm in ARMS
        }
        relation = {}
        for control in ARMS[1:]:
            relation[control] = spearman(
                forward_margin["correct"] - forward_margin[control],
                influence["correct"] - influence[control],
            )
        subgroup = {}
        for label_name, labels in (("fold_role", fold_labels), ("action_role", role_labels)):
            subgroup[label_name] = {}
            for label_index, label in enumerate(sorted(set(labels.tolist()))):
                mask = labels == label
                subgroup[label_name][label] = {
                    "actions": int(np.sum(mask)),
                    "correct_minus_candidate_swapped": distribution(
                        (influence["correct"] - influence["candidate_swapped"])[mask],
                        formulas[mask], seed + 30_000 * route_index + 100 * label_index,
                        draws,
                    ),
                    "correct_minus_peak_permuted": distribution(
                        (influence["correct"] - influence["peak_permuted"])[mask],
                        formulas[mask], seed + 30_000 * route_index + 100 * label_index + 1,
                        draws,
                    ),
                }
        route_diagnostics[route] = {
            "metrics": metric_reports,
            "e1_forward_vs_e2_influence_spearman": relation,
            "subgroups": subgroup,
        }

    pair_overlap = None
    if PAIR_ROUTE in routes:
        def reference(arm: str, name: str) -> np.ndarray:
            result = [by_key[(p, PAIR_ROUTE, arm)][name] for p in positions]
            if any(value is None for value in result):
                raise RuntimeError(f"pair route lacks {name}")
            return np.asarray(result, dtype=np.int64)

        correct_positive = reference("correct", "positive_reference")
        correct_negative = reference("correct", "negative_reference")
        pair_overlap = {}
        for control in ARMS[1:]:
            same_positive = correct_positive == reference(control, "positive_reference")
            same_negative = correct_negative == reference(control, "negative_reference")
            pair_overlap[control] = {
                "actions": len(positions),
                "same_positive_fraction": float(np.mean(same_positive)),
                "same_negative_fraction": float(np.mean(same_negative)),
                "same_pair_fraction": float(np.mean(same_positive & same_negative)),
                "positive_only_changed_fraction": float(np.mean(~same_positive & same_negative)),
                "negative_only_changed_fraction": float(np.mean(same_positive & ~same_negative)),
                "both_changed_fraction": float(np.mean(~same_positive & ~same_negative)),
            }

    return {
        "status": "CHEMAWARE_ACTION_TRANSFER_SPECIFICITY_SUMMARY_COMPLETE",
        "source_status": source.get("status"),
        "weights_updated": False,
        "optimizer_steps": 0,
        "resolved_screen_dir": str(screen),
        "actions": len(positions),
        "formulas": int(len(np.unique(formulas))),
        "records": len(rows),
        "routes": routes,
        "route_diagnostics": route_diagnostics,
        "pair_selection_overlap": pair_overlap,
        "contracts": {
            "gpu_used": False,
            "model_loaded": False,
            "controls_used_as_optimizer_signal": False,
            "new_training_authorized": False,
            "result_is_posthoc_mechanism_decomposition": True,
        },
        "provenance": {
            "source_report_sha256": sha256_file(report_path),
            "per_action_csv_sha256": sha256_file(csv_path),
            "complete_sha256": sha256_file(screen / "COMPLETE.json"),
            "action_bank_sha256": actual_bank_hash,
            "action_bank_report_sha256": sha256_file(bank_report_path),
        },
        "claim_limit": (
            "Post-hoc decomposition of an existing 24-action zero-update screen; "
            "not a retrieval-performance result and not authorization to train."
        ),
    }


def main() -> None:
    args = arguments()
    screen = resolve_screen(args.screen_dir)
    output = args.output or (screen / "specificity_report.json")
    if output.exists():
        raise FileExistsError(
            f"specificity output already exists and will not be overwritten: {output}"
        )
    report = analyze(
        screen, args.action_bank.resolve(), args.bootstrap_draws, args.seed,
    )
    atomic_json(output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "resolved_screen_dir": report["resolved_screen_dir"],
        "actions": report["actions"],
        "pair_selection_overlap": report["pair_selection_overlap"],
        "output": str(output.resolve()),
    }, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
