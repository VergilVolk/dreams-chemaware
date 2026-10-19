#!/usr/bin/env python
"""Replay consumed biology shards against the Contrast Atlas admission gates.

The three shards fail at different scientific layers.  This audit preserves
those distinctions instead of coercing incompatible outputs into one score.
It is deliberately an admission controller, not a model or a rescue analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def replay_chemprop(report_path: Path, retrieval_path: Path) -> dict:
    report = load_json(report_path)
    retrieval = pd.read_csv(retrieval_path)
    require(report["status"] == "chemprop_cross_design_response_v4_complete", "unexpected ChemProp report")
    require(report["evidence_role"] == "development_consumed", "ChemProp role drift")
    require(report["natural_feature_id_shared"] is False, "ChemProp must be an open mapping task")
    require(len(retrieval) > 0, "empty ChemProp retrieval table")

    expected = {
        "mass_nearest_10ppm": (7, 36),
        "mass_cosine_top1_0.7": (10, 36),
        "mass_wse_top1": (7, 36),
    }
    observed: dict[str, dict[str, float | int]] = {}
    for method, (correct, denominator) in expected.items():
        rows = retrieval.loc[retrieval["method"] == method]
        require(len(rows) == denominator, f"{method} denominator drift")
        actual_correct = int(rows["recall1"].astype(bool).sum())
        require(actual_correct == correct, f"{method} correct-count drift")
        observed[method] = {
            "correct": actual_correct,
            "denominator": denominator,
            "recall1": actual_correct / denominator,
            "mrr": float(rows["reciprocal_rank"].mean()),
        }

    bootstrap = report["spectral_minus_mass_nearest_cluster_bootstrap_by_drug"]
    eligible_methods = []
    for method, values in bootstrap.items():
        if values["recall1"]["ci95_low"] > 0 or values["mrr"]["ci95_low"] > 0:
            eligible_methods.append(method)
    require(not eligible_methods, "ChemProp unexpectedly contains a method with a strictly positive cluster CI")
    return {
        "shard_id": "ChemProp_cross_design_v4",
        "dataset": "MSV000094899 queries vs MSV000096724/MSV000093571 references",
        "data_role": "development/consumed",
        "independent_denominator": "12 drugs; 36 replicate queries",
        "failed_layer": "strict_entity_mapping_to_biological_response",
        "observed": observed,
        "primary_increment": {
            "method": "mass_cosine_top1_0.7",
            "recall1_delta": bootstrap["mass_cosine_top1_0.7"]["recall1"]["estimate"],
            "recall1_ci95": [
                bootstrap["mass_cosine_top1_0.7"]["recall1"]["ci95_low"],
                bootstrap["mass_cosine_top1_0.7"]["recall1"]["ci95_high"],
            ],
            "mrr_delta": bootstrap["mass_cosine_top1_0.7"]["mrr"]["estimate"],
            "mrr_ci95": [
                bootstrap["mass_cosine_top1_0.7"]["mrr"]["ci95_low"],
                bootstrap["mass_cosine_top1_0.7"]["mrr"]["ci95_high"],
            ],
        },
        "decision": "FAIL_ENTITY_INCREMENT",
        "admit_to_shared_pattern_or_study_association": False,
        "claim_limit": "Same-project cross-design development result; no cross-source, unknown-identity, or Atlas capability.",
    }


def replay_st003325(report_path: Path) -> dict:
    report = load_json(report_path)
    require(report["status"] == "ST003325_STRUCTURE_HIDDEN_MS1_BASELINE_COMPLETE", "unexpected ST003325 report")
    require(report["data_role"] == "development/consumed", "ST003325 role drift")
    require(report["unit_of_replication"].startswith("clone"), "ST003325 replication unit drift")
    primary = report["analyses"]["median_centered_log1p"]["MS1_all"]
    combined = primary["combined"]
    permutation = primary["clone_label_permutation"]
    require(combined["correct"] == 10 and combined["denominator"] == 27, "ST003325 primary count drift")
    require(permutation["repeats"] == 2000, "ST003325 permutation denominator drift")
    class_pass = all(item["accuracy"] > 0.5 for item in combined["by_held_edit_class"].values())
    signal_pass = class_pass and permutation["right_tail_p"] < 0.05
    require(not signal_pass, "ST003325 unexpectedly passes the frozen response-object gate")
    return {
        "shard_id": "ST003325_structure_hidden_perturbation",
        "dataset": "Metabolomics Workbench ST003325 / PR001844",
        "data_role": "development/consumed",
        "independent_denominator": "12 clones; 27 held-edit comparisons",
        "failed_layer": "within_study_response_object",
        "observed": {
            "correct": int(combined["correct"]),
            "denominator": int(combined["denominator"]),
            "accuracy": float(combined["accuracy"]),
            "median_margin": float(combined["median_margin"]),
            "class_accuracy": {key: float(value["accuracy"]) for key, value in combined["by_held_edit_class"].items()},
            "clone_label_permutation_p": float(permutation["right_tail_p"]),
        },
        "decision": "FAIL_RESPONSE_OBJECT",
        "admit_to_shared_pattern_or_study_association": False,
        "claim_limit": "Design-labelled plate-set response only; does not establish absence of an STS phenotype.",
    }


def replay_lcnec(headroom_path: Path, certificate_path: Path) -> dict:
    headroom = load_json(headroom_path)
    certificate = load_json(certificate_path)
    require(headroom["status"] == "phenotype_blind_candidate_headroom_complete", "unexpected LCNEC headroom report")
    require(headroom["phenotype_labels_read"] is False, "LCNEC headroom leakage status drift")
    require(headroom["discovery_entities"] == 263, "LCNEC discovery denominator drift")
    require(headroom["candidate_pairs"] == 6, "LCNEC candidate-pair count drift")
    require(headroom["pass_to_global_transport"] is False, "LCNEC global-transport decision drift")
    require(certificate["queries"] == 263 and certificate["assigned"] == 0, "LCNEC certificate count drift")
    require(certificate["contracts"]["explicit_dustbin"] is True, "LCNEC no-match contract drift")
    return {
        "shard_id": "LCNEC_cross_platform_and_public_certificate",
        "dataset": "LCNEC HSST3n discovery vs LIPn validation plus GNPS-negative public evidence",
        "data_role": "development/consumed",
        "independent_denominator": "263 discovery entities",
        "failed_layer": "strict_entity_coverage_and_public_evidence_assignment",
        "observed": {
            "candidate_pairs": int(headroom["candidate_pairs"]),
            "entities_with_candidate": int(headroom["discovery_entities_with_candidate"]),
            "discovery_entities": int(headroom["discovery_entities"]),
            "candidate_coverage": float(headroom["discovery_candidate_coverage"]),
            "minimum_global_coverage": float(headroom["min_global_coverage"]),
            "public_certificate_assigned": int(certificate["assigned"]),
            "public_certificate_queries": int(certificate["queries"]),
        },
        "decision": "ABSTAIN_INSUFFICIENT_COMPARABLE_EVIDENCE",
        "admit_to_shared_pattern_or_study_association": False,
        "claim_limit": "Six phenotype-blind candidate pairs are a coverage ceiling, not effect replication or biology.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemprop-report", type=Path, required=True)
    parser.add_argument("--chemprop-retrieval", type=Path, required=True)
    parser.add_argument("--st003325-report", type=Path, required=True)
    parser.add_argument("--lcnec-headroom-report", type=Path, required=True)
    parser.add_argument("--lcnec-certificate-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    paths = {
        "chemprop_report": args.chemprop_report,
        "chemprop_retrieval": args.chemprop_retrieval,
        "st003325_report": args.st003325_report,
        "lcnec_headroom_report": args.lcnec_headroom_report,
        "lcnec_certificate_report": args.lcnec_certificate_report,
    }
    shards = [
        replay_chemprop(args.chemprop_report, args.chemprop_retrieval),
        replay_st003325(args.st003325_report),
        replay_lcnec(args.lcnec_headroom_report, args.lcnec_certificate_report),
    ]
    ledger = pd.DataFrame([{key: value for key, value in shard.items() if key not in {"observed", "primary_increment"}} for shard in shards])
    report = {
        "status": "CONTRAST_ATLAS_NEGATIVE_SHARD_ADMISSION_REPLAY_COMPLETE",
        "formal": True,
        "analysis_role": "development/consumed negative-control replay",
        "shards_replayed": len(shards),
        "shards_admitted": int(sum(bool(shard["admit_to_shared_pattern_or_study_association"]) for shard in shards)),
        "all_historical_stops_preserved": all(not shard["admit_to_shared_pattern_or_study_association"] for shard in shards),
        "input_sha256": {key: sha256(path) for key, path in paths.items()},
        "shards": shards,
        "next_gate": "A real positive-control response object must pass before shared-pattern or study-association retrieval is allowed.",
        "claim_limit": "Admission replay only. It adds no positive biological result and does not compare new spectral algorithms.",
    }
    args.output_dir.mkdir(parents=True)
    ledger.to_csv(args.output_dir / "admission_ledger.csv", index=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "shards_replayed", "shards_admitted", "all_historical_stops_preserved", "next_gate")}, indent=2))


if __name__ == "__main__":
    main()
