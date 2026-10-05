#!/usr/bin/env python
"""Apply a frozen grand-fusion router to sealed GNPS score panels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel
from gnps_pair_score_cache import PANELS, write_pair_score_cache
from grand_fusion_router_core import predict_correctness, query_features, route
from noise_final_core import sha256_file


ROOT = Path(__file__).resolve().parents[1]
GNPS_METHOD_ALIASES = {
    "noise_p2b_v1": "p2b_noise_v1_frozen",
    "neutral_loss": "neutral_loss_sqrt_cosine",
    "sqrt_cosine": "p2b_sqrt_cosine",
    "spectral_entropy": "p2b_unweighted_entropy",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1")
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    package = joblib.load(args.router / "router.joblib")
    report = json.loads((args.router / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "GRAND_FUSION_ROUTER_FROZEN":
        raise RuntimeError("router is not frozen")
    methods = tuple(package["methods"])
    with np.load(args.score_bundle, allow_pickle=False) as body:
        bundle_methods = list(map(str, body["method_names"]))
        bundle_scores = {panel: np.asarray(body[f"scores_{panel}"], dtype=np.float32) for panel in PANELS}
    bundle_name = {name: GNPS_METHOD_ALIASES.get(name, name) for name in methods}
    missing = [name for name in methods if bundle_name[name] not in bundle_methods]
    if missing:
        raise RuntimeError(
            f"GNPS bundle misses router experts {missing}; extend the frozen bundle before application"
        )

    output_scores: dict[str, np.ndarray] = {}
    routing_reports: dict[str, dict] = {}
    routing_ledgers: dict[str, list[dict[str, object]]] = {}
    for panel in PANELS:
        graph = graph_from_panel(args.benchmark / f"panel_{panel}.npz")
        pair_matrix = bundle_scores[panel]
        molecule_scores = {
            name: np.maximum.reduceat(
                pair_matrix[bundle_methods.index(bundle_name[name])].astype(np.float64),
                np.asarray(graph.molecule_ptr, dtype=np.int64)[:-1],
            )
            for name in methods
        }
        features, winners = query_features(molecule_scores, graph.query_ptr, methods)
        probabilities = predict_correctness(package["models"], features, methods)
        chosen = route(
            probabilities, winners, methods, package["default_method"], package["threshold"],
        )
        query_edge_ptr = np.asarray(graph.molecule_ptr, dtype=np.int64)[np.asarray(graph.query_ptr, dtype=np.int64)]
        selected = np.empty(pair_matrix.shape[1], dtype=np.float32)
        for query, (left, right) in enumerate(zip(query_edge_ptr[:-1], query_edge_ptr[1:])):
            method = methods[int(chosen[query])]
            selected[int(left):int(right)] = pair_matrix[
                bundle_methods.index(bundle_name[method]), int(left):int(right)
            ]
        output_scores[panel] = selected
        counts = {method: int((chosen == index).sum()) for index, method in enumerate(methods)}
        default_index = methods.index(package["default_method"])
        panel_ledger: list[dict[str, object]] = []
        for query, (left, right) in enumerate(zip(graph.query_ptr[:-1], graph.query_ptr[1:])):
            selected_index = int(chosen[query])
            selected_method = methods[selected_index]
            expert_state = {
                method: {
                    "predicted_correctness": float(probabilities[query, index]),
                    "winner_molecule_index": (
                        int(left) + int(winners[query, index])
                        if int(winners[query, index]) >= 0 else None
                    ),
                }
                for index, method in enumerate(methods)
            }
            panel_ledger.append({
                "panel": panel,
                "query_index": int(query),
                "candidate_count": int(right) - int(left),
                "default_method": package["default_method"],
                "selected_method": selected_method,
                "switched": int(selected_index != default_index),
                "threshold": float(package["threshold"]),
                "default_predicted_correctness": float(probabilities[query, default_index]),
                "selected_predicted_correctness": float(probabilities[query, selected_index]),
                "default_winner_molecule_index": expert_state[package["default_method"]]["winner_molecule_index"],
                "selected_winner_molecule_index": expert_state[selected_method]["winner_molecule_index"],
                "expert_state_json": json.dumps(expert_state, sort_keys=True, separators=(",", ":")),
            })
        routing_ledgers[panel] = panel_ledger
        routing_reports[panel] = {
            "queries": int(graph.n_queries),
            "threshold": float(package["threshold"]),
            "selected_expert_counts": counts,
            "abstained_to_default": counts[package["default_method"]],
        }

    method = {
        "name": "grand_fusion_query_router_v1",
        "default_method": package["default_method"],
        "threshold": package["threshold"],
        "risk_lambda": package["risk_lambda"],
        "experts": list(methods),
        "routing": routing_reports,
        "gnps_used_in_training": False,
        "sources": {
            "score_bundle_sha256": sha256_file(args.score_bundle),
            "router_report_sha256": sha256_file(args.router / "report.json"),
        },
        "per_query_ledgers": {
            panel: f"routing_ledger_{panel}.csv" for panel in PANELS
        },
    }
    cache_report = write_pair_score_cache(args.output, args.benchmark, method, output_scores)
    ledger_hashes: dict[str, str] = {}
    fields = [
        "panel", "query_index", "candidate_count", "default_method", "selected_method",
        "switched", "threshold", "default_predicted_correctness",
        "selected_predicted_correctness", "default_winner_molecule_index",
        "selected_winner_molecule_index", "expert_state_json",
    ]
    for panel, rows in routing_ledgers.items():
        path = args.output / f"routing_ledger_{panel}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        ledger_hashes[panel] = sha256_file(path)
    ledger_manifest = {
        "schema": "grand_fusion_truthblind_routing_ledger_v1",
        "contains_ground_truth": False,
        "score_cache_report_sha256": sha256_file(args.output / "report.json"),
        "router_report_sha256": sha256_file(args.router / "report.json"),
        "ledgers": ledger_hashes,
    }
    (args.output / "routing_ledger_manifest.json").write_text(
        json.dumps(ledger_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"score_cache": cache_report, "routing_ledger": ledger_manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
