"""Post-hoc mechanism audit for frozen P2b transitions on consumed P3.

This script never fits or selects a model.  It enriches the already-consumed
P3 per-query report with deployment-visible candidate relations and reports
how simple *diagnostic* fallback rules would partition corrected and
introduced errors.  Any rule suggested by this audit must be rebuilt and
selected on P2, then evaluated on a new sealed test set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data/validation/g8r_p2b_p3_final.per_query.csv"
DEFAULT_RESULT = ROOT / "data/validation/g8r_p2b_p3_final.json"
DEFAULT_PAIRS = ROOT / "tasks/massspecgym_isomers/pairs.json"
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_p3_transition_audit.json"
DEFAULT_ENRICHED = ROOT / "data/validation/g8r_p3_transition_audit.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--enriched", type=Path, default=DEFAULT_ENRICHED)
    return parser.parse_args()


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def text_array(dataset) -> np.ndarray:
    values = dataset[:]
    return np.asarray([
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=object)


def load_identity_metadata(path: Path) -> dict[str, dict[str, str]]:
    with h5py.File(path, "r") as handle:
        inchikey = text_array(handle["INCHIKEY"])
        formulas = text_array(handle["FORMULA"])
        smiles_key = "SMILES" if "SMILES" in handle else "smiles"
        smiles = text_array(handle[smiles_key])
    metadata: dict[str, dict[str, str]] = {}
    for key, formula, smile in zip(inchikey, formulas, smiles):
        ik14 = str(key)[:14]
        current = metadata.setdefault(ik14, {"formula": "", "smiles": ""})
        if not current["formula"] and formula not in {"", "nan", "None"}:
            current["formula"] = str(formula)
        if not current["smiles"] and smile not in {"", "nan", "None"}:
            current["smiles"] = str(smile)
    return metadata


def load_pair_relations(path: Path) -> dict[tuple[str, str], dict]:
    body = json.loads(path.read_text(encoding="utf-8"))
    relations: dict[tuple[str, str], dict] = {}
    for grade in ("near", "mid", "far", "uncomputed"):
        for row in body.get(grade, []):
            key = tuple(sorted((str(row["ik_a"])[:14], str(row["ik_b"])[:14])))
            distance = row.get("mces_raw")
            relations[key] = {
                "grade": grade,
                "mces": None if distance is None else float(distance),
            }
    return relations


def predicted_identity(row: dict[str, str], method: str) -> str:
    return row["ik14"] if as_bool(row[f"{method}_top1"]) else row[f"{method}_best_negative_ik14"]


def transition_counts(rows: list[dict], use_fusion: np.ndarray) -> dict:
    base = np.asarray([row["dreams_top1"] for row in rows], dtype=bool)
    fusion = np.asarray([row["p2b_frozen_top1"] for row in rows], dtype=bool)
    final = np.where(np.asarray(use_fusion, dtype=bool), fusion, base)
    corrected = int(np.sum((~base) & final))
    introduced = int(np.sum(base & (~final)))
    return {
        "n_queries": len(rows),
        "recall1": float(final.mean()),
        "delta_vs_dreams": float(final.mean() - base.mean()),
        "corrected": corrected,
        "introduced": introduced,
        "net": corrected - introduced,
        "intervention_rate": float(np.mean(use_fusion)),
    }


def categorical_transition_table(rows: list[dict], field: str) -> dict:
    table: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        base, fusion = row["dreams_top1"], row["p2b_frozen_top1"]
        transition = (
            "corrected" if not base and fusion else
            "introduced" if base and not fusion else
            "persistent_right" if base else "persistent_wrong"
        )
        table[str(row[field])][transition] += 1
    return {key: dict(counter) for key, counter in sorted(table.items())}


def audit_panel(rows: list[dict]) -> dict:
    support = np.asarray([int(row["p2b_support"]) for row in rows])
    same_formula = np.asarray([bool(row["prediction_pair_same_formula"]) for row in rows])
    grade = np.asarray([str(row["prediction_pair_grade"]) for row in rows], dtype=object)
    disagrees = np.asarray([bool(row["dreams_p2b_disagree"]) for row in rows])
    policies = {
        "frozen_p2b": np.ones(len(rows), dtype=bool),
        "support_ge_2": support >= 2,
        "support_ge_3": support >= 3,
        "fallback_same_formula_disagreement": ~(disagrees & same_formula),
        "fallback_cached_near_disagreement": ~(disagrees & (grade == "near")),
        "fallback_cached_nearmid_disagreement": ~(disagrees & np.isin(grade, ["near", "mid"])),
        "support_ge_2_and_not_same_formula": (support >= 2) & ~(disagrees & same_formula),
    }
    return {
        "n_queries": len(rows),
        "policies_diagnostic_only": {
            name: transition_counts(rows, mask) for name, mask in policies.items()
        },
        "transition_by_support": categorical_transition_table(rows, "p2b_support"),
        "transition_by_candidate_count_bucket": categorical_transition_table(
            rows, "candidate_count_bucket"
        ),
        "transition_by_prediction_pair_formula": categorical_transition_table(
            rows, "prediction_pair_same_formula"
        ),
        "transition_by_prediction_pair_grade": categorical_transition_table(
            rows, "prediction_pair_grade"
        ),
        "transition_by_raw_expert_agreement": categorical_transition_table(
            rows, "raw_experts_agree_with_p2b"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.enriched.exists():
        raise FileExistsError("refusing to overwrite an existing P3 post-hoc audit")
    for path in (args.input, args.result, args.pairs, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("status") not in {"g8r_p2b_p3_passed", "g8r_p2b_p3_failed"}:
        raise RuntimeError("input is not a completed sealed P3 result")
    metadata = load_identity_metadata(args.data)
    relations = load_pair_relations(args.pairs)
    enriched: list[dict] = []
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for method in ("dreams", "sqrt_cosine", "entropy", "neutral_loss", "p2b_frozen"):
                row[f"{method}_top1"] = as_bool(raw[f"{method}_top1"])
                row[f"{method}_prediction"] = predicted_identity(raw, method)
            row["p2b_intervened"] = as_bool(raw["p2b_intervened"])
            row["p2b_support"] = int(raw["p2b_support"])
            row["n_candidate_molecules"] = int(raw["n_candidate_molecules"])
            n_candidates = row["n_candidate_molecules"]
            row["candidate_count_bucket"] = (
                "2" if n_candidates == 2 else "3-4" if n_candidates <= 4 else
                "5-8" if n_candidates <= 8 else "9+"
            )
            dreams_prediction = row["dreams_prediction"]
            p2b_prediction = row["p2b_frozen_prediction"]
            row["dreams_p2b_disagree"] = dreams_prediction != p2b_prediction
            relation_key = tuple(sorted((dreams_prediction, p2b_prediction)))
            relation = relations.get(relation_key, {"grade": "not_cached", "mces": None})
            if dreams_prediction == p2b_prediction:
                relation = {"grade": "same_prediction", "mces": 0.0}
            row["prediction_pair_grade"] = relation["grade"]
            row["prediction_pair_mces"] = relation["mces"]
            dreams_formula = metadata.get(dreams_prediction, {}).get("formula", "")
            p2b_formula = metadata.get(p2b_prediction, {}).get("formula", "")
            row["dreams_prediction_formula"] = dreams_formula
            row["p2b_prediction_formula"] = p2b_formula
            row["prediction_pair_same_formula"] = bool(
                dreams_formula and dreams_formula == p2b_formula
            )
            raw_predictions = [
                row["sqrt_cosine_prediction"], row["entropy_prediction"],
                row["neutral_loss_prediction"],
            ]
            row["raw_experts_agree_with_p2b"] = int(sum(
                prediction == p2b_prediction for prediction in raw_predictions
            ))
            enriched.append(row)

    panels: dict[str, list[dict]] = defaultdict(list)
    for row in enriched:
        panels[str(row["panel"])].append(row)
    report = {
        "status": "g8r_p3_transition_mechanism_audit_complete",
        "source_result_sha256": sha256_file(args.result),
        "source_per_query_sha256": sha256_file(args.input),
        "pairs_sha256": sha256_file(args.pairs),
        "p3_is_consumed": True,
        "may_be_used_for_model_selection": False,
        "panels": {name: audit_panel(rows) for name, rows in panels.items()},
        "claim_boundary": (
            "Policy tables are post-hoc diagnostics on consumed P3. They cannot be reported "
            "as new model performance or used to choose thresholds. Any router must be "
            "trained on P2 and validated on a new sealed test set."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = list(enriched[0])
    with args.enriched.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
