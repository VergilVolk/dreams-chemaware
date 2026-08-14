"""Compare two molecule-disjoint multilevel DreaMS activation pilots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DISCOVERY = ROOT / "data/validation/multilevel_factor_pilot1000_qc"
DEFAULT_CONFIRMATION = ROOT / "data/validation/multilevel_factor_confirm1000_qc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def identity_by_layer(report: dict, representation: str, model: str) -> dict[int, dict]:
    return {
        item["layer"]: item
        for item in report[representation][model]
    }


def ci_direction(item: dict, minimum_n: int = 20) -> str:
    if item["n_pairs"] < minimum_n:
        return "insufficient_n"
    low, high = item["official_minus_raw_ci95"]
    if low > 0:
        return "positive"
    if high < 0:
        return "negative"
    return "uncertain"


def main() -> None:
    args = parse_args()
    output_path = args.output or args.confirmation / "replication_report.json"
    discovery_activation = read_json(args.discovery / "report.json")
    confirmation_activation = read_json(args.confirmation / "report.json")
    discovery_condition = read_json(args.discovery / "condition_invariance_report.json")
    confirmation_condition = read_json(args.confirmation / "condition_invariance_report.json")
    discovery_pairs = read_json(args.discovery / "pairs.json")
    confirmation_pairs = read_json(args.confirmation / "pairs.json")

    discovery_ik = {item["ik14"] for item in discovery_pairs}
    confirmation_ik = {item["ik14"] for item in confirmation_pairs}
    overlap = sorted(discovery_ik & confirmation_ik)
    if overlap:
        raise RuntimeError(f"Discovery/confirmation IK14 overlap: {overlap[:5]}")

    layers = discovery_activation["config"]["layers"]
    if confirmation_activation["config"]["layers"] != layers:
        raise RuntimeError("Layer definitions differ between runs")

    layer_replication = []
    for index, layer in enumerate(layers):
        d_prec = discovery_activation["precursor_same_layer_cka"][index]
        c_prec = confirmation_activation["precursor_same_layer_cka"][index]
        d_peak = discovery_activation["pooled_peak_same_layer_cka"][index]
        c_peak = confirmation_activation["pooled_peak_same_layer_cka"][index]
        layer_replication.append({
            "layer": layer,
            "precursor_cka": {
                "discovery": d_prec,
                "confirmation": c_prec,
                "absolute_difference": abs(d_prec - c_prec),
            },
            "pooled_peak_cka": {
                "discovery": d_peak,
                "confirmation": c_peak,
                "absolute_difference": abs(d_peak - c_peak),
            },
        })

    identity_replication = {}
    for representation in ("precursor_identity", "pooled_peak_identity"):
        identity_replication[representation] = []
        for layer in layers:
            d_raw = identity_by_layer(discovery_condition, representation, "raw")[layer]
            d_off = identity_by_layer(discovery_condition, representation, "official")[layer]
            c_raw = identity_by_layer(confirmation_condition, representation, "raw")[layer]
            c_off = identity_by_layer(confirmation_condition, representation, "official")[layer]
            d_gain = d_off["roc_auc"] - d_raw["roc_auc"]
            c_gain = c_off["roc_auc"] - c_raw["roc_auc"]
            identity_replication[representation].append({
                "layer": layer,
                "discovery_auc_gain": d_gain,
                "confirmation_auc_gain": c_gain,
                "replicated_positive_gain": bool(d_gain > 0 and c_gain > 0),
            })

    condition_replication = {}
    for representation in (
        "precursor_pair_invariance_by_condition",
        "pooled_peak_pair_invariance_by_condition",
    ):
        discovery_groups = discovery_condition[representation]
        confirmation_groups = confirmation_condition[representation]
        condition_replication[representation] = {}
        for category in sorted(set(discovery_groups) | set(confirmation_groups)):
            d_items = {item["layer"]: item for item in discovery_groups.get(category, [])}
            c_items = {item["layer"]: item for item in confirmation_groups.get(category, [])}
            category_results = []
            for layer in layers:
                d_item = d_items.get(layer)
                c_item = c_items.get(layer)
                if d_item is None or c_item is None:
                    category_results.append({
                        "layer": layer,
                        "status": "missing_category",
                    })
                    continue
                d_direction = ci_direction(d_item)
                c_direction = ci_direction(c_item)
                if "insufficient_n" in (d_direction, c_direction):
                    status = "insufficient_n"
                elif d_direction == c_direction and d_direction in ("positive", "negative"):
                    status = f"replicated_{d_direction}"
                elif d_direction == "uncertain" or c_direction == "uncertain":
                    status = "not_confirmed"
                else:
                    status = "direction_conflict"
                category_results.append({
                    "layer": layer,
                    "discovery_n": d_item["n_pairs"],
                    "confirmation_n": c_item["n_pairs"],
                    "discovery_effect": d_item["official_minus_raw"],
                    "confirmation_effect": c_item["official_minus_raw"],
                    "discovery_direction": d_direction,
                    "confirmation_direction": c_direction,
                    "status": status,
                })
            condition_replication[representation][category] = category_results

    criteria = {
        "zero_molecule_overlap": len(overlap) == 0,
        "all_layer_cka_abs_diff_le_0_03": all(
            item["precursor_cka"]["absolute_difference"] <= 0.03
            and item["pooled_peak_cka"]["absolute_difference"] <= 0.03
            for item in layer_replication
        ),
        "all_identity_auc_gains_positive": all(
            item["replicated_positive_gain"]
            for values in identity_replication.values() for item in values
        ),
        "deep_layers_more_changed_than_layer4": bool(
            layer_replication[-1]["precursor_cka"]["confirmation"]
            < layer_replication[0]["precursor_cka"]["confirmation"]
            and layer_replication[1]["pooled_peak_cka"]["confirmation"]
            < layer_replication[0]["pooled_peak_cka"]["confirmation"]
        ),
    }
    output = {
        "status": "independent_replication_audit",
        "discovery": str(args.discovery.resolve()),
        "confirmation": str(args.confirmation.resolve()),
        "molecules": {
            "discovery": len(discovery_ik),
            "confirmation": len(confirmation_ik),
            "overlap": len(overlap),
        },
        "criteria": criteria,
        "layer_replication": layer_replication,
        "identity_replication": identity_replication,
        "condition_replication": condition_replication,
        "decision": (
            "Layer candidates and identity gains replicated. Proceed only to a "
            "small Crosscoder smoke test; rare joint adduct-condition effects "
            "remain underpowered."
            if all(criteria.values()) else
            "Replication criteria not all met; do not train Crosscoder."
        ),
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Independent replication audit")
    print("  molecule overlap:", len(overlap))
    print("  criteria:", criteria)
    for item in layer_replication:
        print(
            f"  L{item['layer']}: precursor CKA "
            f"{item['precursor_cka']['discovery']:.4f} -> "
            f"{item['precursor_cka']['confirmation']:.4f}; peak CKA "
            f"{item['pooled_peak_cka']['discovery']:.4f} -> "
            f"{item['pooled_peak_cka']['confirmation']:.4f}"
        )
    print("  decision:", output["decision"])
    print("  saved:", output_path)


if __name__ == "__main__":
    main()
