#!/usr/bin/env python
"""Audit every BioAware B2 Top-1 flip against its decisive competitor.

This is an interpretation audit, not a model-selection stage.  It replays the
frozen outer-study ensemble, compares the truth candidate with the negative
that decided the relevant rank, and records whether either identity was
available to the training fold.  No outcome is fed back into training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "tasks"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from replay_bioaware_b2_leave_study_out import build_artifact_model, load_artifact
from train_bioaware_b2_leave_study_out import (
    Dataset, adapt_candidates, sha256, strict_rank,
)


def top_negative(scores: np.ndarray, positive: int) -> int:
    indices = np.arange(len(scores), dtype=np.int64)
    negatives = indices[indices != positive]
    if len(negatives) == 0:
        raise RuntimeError("candidate list has no negative")
    # A score tie is an error under the frozen rank contract.  Stable candidate
    # order resolves which co-maximal negative is displayed, without changing
    # any rank or transition count.
    return int(negatives[np.argmax(scores[negatives])])


def training_queries(data: Dataset, outer_study: str, isolation: str) -> np.ndarray:
    train = np.flatnonzero(data.study_ids != outer_study)
    heldout = np.flatnonzero(data.study_ids == outer_study)
    heldout_truth = set(data.truth_ids[heldout].tolist())
    heldout_formula = set(data.formulas[heldout].tolist())
    heldout_candidates = set(np.concatenate([
        data.candidate_ids[data.section(int(index))] for index in heldout
    ]).tolist())
    if isolation == "truth_identity":
        train = train[np.asarray([
            data.truth_ids[int(index)] not in heldout_truth for index in train
        ], dtype=bool)]
    elif isolation == "truth_formula":
        train = train[np.asarray([
            data.formulas[int(index)] not in heldout_formula for index in train
        ], dtype=bool)]
    if isolation != "study":
        train = np.asarray([
            int(index) for index in train
            if not (
                set(data.candidate_ids[data.section(int(index))].tolist())
                & heldout_candidates
            )
        ], dtype=np.int64)
    return train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    destination = args.input_dir / "flip_mechanism_audit.csv.gz"
    report_destination = args.input_dir / "flip_mechanism_audit.json"
    if destination.exists() or report_destination.exists():
        raise RuntimeError("fail-closed: flip audit output already exists")

    report_path = args.input_dir / "report.json"
    transitions_path = args.input_dir / "query_oof_transitions.csv.gz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["provenance"]["dataset_sha256"] != sha256(args.dataset):
        raise RuntimeError("dataset hash differs from training report")
    expected = pd.read_csv(transitions_path).set_index("query_id")
    device = torch.device(args.device)
    data = Dataset(args.dataset, device, report["configuration"]["evidence_mode"])

    by_study: dict[str, list[dict]] = {}
    for item in report["run_reports"]:
        by_study.setdefault(str(item["outer_study"]), []).append(item)
    records: list[dict] = []
    rank_mismatches = 0
    for outer_study in sorted(by_study):
        heldout = np.flatnonzero(data.study_ids == outer_study)
        train = training_queries(
            data, outer_study, str(report["configuration"]["training_isolation"]),
        )
        train_truth_ids = set(data.truth_ids[train].tolist())
        train_candidate_ids = set(np.concatenate([
            data.candidate_ids[data.section(int(index))] for index in train
        ]).tolist())
        ensemble_scores: dict[int, list[np.ndarray]] = {int(i): [] for i in heldout}
        ensemble_gates: dict[int, list[np.ndarray]] = {int(i): [] for i in heldout}
        for item in sorted(by_study[outer_study], key=lambda value: int(value["seed"])):
            artifact_path = args.input_dir / Path(str(item["artifact"])).name
            if sha256(artifact_path) != item["artifact_sha256"]:
                raise RuntimeError(f"artifact hash mismatch: {artifact_path.name}")
            body = load_artifact(artifact_path)
            configuration = body["configuration"]
            model = build_artifact_model(configuration, device)
            model.load_state_dict(body["state_dict"], strict=True)
            model.eval()
            mean = torch.from_numpy(np.asarray(body["evidence_mean"], dtype=np.float32)).to(device)
            scale = torch.from_numpy(np.asarray(body["evidence_scale"], dtype=np.float32)).to(device)
            with torch.no_grad():
                for index in heldout:
                    index = int(index)
                    section = data.section(index)
                    flat = torch.arange(section.start, section.stop, device=device)
                    adapted, _, gates = adapt_candidates(
                        model, data, index, flat, mean, scale,
                        str(configuration.get("adapter_type", "free")),
                    )
                    ensemble_scores[index].append(
                        (adapted @ data.query[index]).detach().cpu().numpy()
                    )
                    ensemble_gates[index].append(gates.detach().cpu().numpy())

        for index in heldout:
            index = int(index)
            query_id = str(data.query_ids[index])
            row = expected.loc[query_id]
            if not (bool(row.corrected) or bool(row.introduced)):
                continue
            section = data.section(index)
            baseline = (data.candidate[section] @ data.query[index]).detach().cpu().numpy()
            final = np.mean(np.stack(ensemble_scores[index]), axis=0)
            gates = np.mean(np.stack(ensemble_gates[index]), axis=0)
            positive = int(data.positive[index])
            baseline_rank = strict_rank(baseline, positive)
            final_rank = strict_rank(final, positive)
            rank_mismatches += int(
                baseline_rank != int(row.baseline_rank) or final_rank != int(row.final_rank)
            )
            transition = "corrected" if bool(row.corrected) else "introduced"
            decisive_scores = baseline if transition == "corrected" else final
            competitor = top_negative(decisive_scores, positive)
            truth_flat = section.start + positive
            competitor_flat = section.start + competitor
            truth_id = str(data.candidate_ids[truth_flat])
            competitor_id = str(data.candidate_ids[competitor_flat])
            record = {
                "query_id": query_id,
                "study_id": outer_study,
                "unit_id": str(data.unit_ids[index]),
                "truth_formula": str(data.formulas[index]),
                "transition": transition,
                "truth_candidate_id": truth_id,
                "decisive_competitor_id": competitor_id,
                "baseline_rank": baseline_rank,
                "final_rank": final_rank,
                "baseline_truth_score": float(baseline[positive]),
                "baseline_competitor_score": float(baseline[competitor]),
                "baseline_margin_vs_competitor": float(
                    baseline[positive] - baseline[competitor]
                ),
                "final_truth_score": float(final[positive]),
                "final_competitor_score": float(final[competitor]),
                "final_margin_vs_competitor": float(final[positive] - final[competitor]),
                "truth_score_delta": float(final[positive] - baseline[positive]),
                "competitor_score_delta": float(final[competitor] - baseline[competitor]),
                "truth_gate": float(gates[positive]),
                "competitor_gate": float(gates[competitor]),
                "truth_context": bool(data.context[truth_flat]),
                "competitor_context": bool(data.context[competitor_flat]),
                "truth_seen_as_training_truth": truth_id in train_truth_ids,
                "competitor_seen_as_training_truth": competitor_id in train_truth_ids,
                "truth_seen_in_training_candidate_lists": truth_id in train_candidate_ids,
                "competitor_seen_in_training_candidate_lists": competitor_id in train_candidate_ids,
            }
            truth_evidence = data.evidence_raw[truth_flat]
            competitor_evidence = data.evidence_raw[competitor_flat]
            family_differences = {"reaction": [], "smn": [], "rt": []}
            for column, truth_value, competitor_value in zip(
                data.evidence_columns, truth_evidence, competitor_evidence,
            ):
                name = str(column)
                difference = float(truth_value - competitor_value)
                record[f"truth__{name}"] = float(truth_value)
                record[f"competitor__{name}"] = float(competitor_value)
                record[f"truth_minus_competitor__{name}"] = difference
                if name.startswith(("known_edge_", "predicted_edge_")):
                    family_differences["reaction"].append(difference)
                elif name.startswith("smn_"):
                    family_differences["smn"].append(difference)
                elif name.startswith("rt_"):
                    family_differences["rt"].append(difference)
            for family, differences in family_differences.items():
                record[f"truth_minus_competitor__{family}_sum"] = (
                    float(sum(differences)) if differences else np.nan
                )
            records.append(record)

    if records:
        frame = pd.DataFrame(records).sort_values(
            ["transition", "study_id", "query_id"], kind="stable",
        )
    else:
        frame = pd.DataFrame(columns=[
            "transition", "study_id", "query_id", "truth_candidate_id",
            "decisive_competitor_id", "truth_seen_as_training_truth",
            "competitor_seen_as_training_truth",
            "truth_seen_in_training_candidate_lists",
            "competitor_seen_in_training_candidate_lists", "truth_gate",
            "competitor_gate", "truth_minus_competitor__reaction_sum",
            "truth_minus_competitor__smn_sum", "truth_minus_competitor__rt_sum",
        ])
    if rank_mismatches:
        raise RuntimeError(f"frozen transition replay mismatch: {rank_mismatches}")
    expected_flips = int(expected.corrected.sum() + expected.introduced.sum())
    if len(frame) != expected_flips:
        raise RuntimeError(f"flip coverage mismatch: {len(frame)} != {expected_flips}")
    frame.to_csv(destination, index=False, compression="gzip")
    unordered_pairs: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        pair = tuple(sorted((
            str(record["truth_candidate_id"]), str(record["decisive_competitor_id"]),
        )))
        unordered_pairs.setdefault(pair, []).append(record)
    reversed_pairs = []
    for pair, pair_records in unordered_pairs.items():
        truth_roles = {str(record["truth_candidate_id"]) for record in pair_records}
        if len(truth_roles) > 1:
            reversed_pairs.append({
                "candidate_pair": list(pair),
                "queries": [str(record["query_id"]) for record in pair_records],
                "transitions": [str(record["transition"]) for record in pair_records],
                "truth_roles": sorted(truth_roles),
            })
    summary = {
        "status": "bioaware_b2_flip_mechanism_audit_complete",
        "queries": int(len(frame)),
        "corrected": int((frame.transition == "corrected").sum()),
        "introduced": int((frame.transition == "introduced").sum()),
        "rank_mismatches": rank_mismatches,
        "truth_seen_as_training_truth": int(frame.truth_seen_as_training_truth.sum()),
        "competitor_seen_as_training_truth": int(
            frame.competitor_seen_as_training_truth.sum()
        ),
        "truth_seen_in_training_candidate_lists": int(
            frame.truth_seen_in_training_candidate_lists.sum()
        ),
        "competitor_seen_in_training_candidate_lists": int(
            frame.competitor_seen_in_training_candidate_lists.sum()
        ),
        "both_truth_and_competitor_seen_in_training_candidate_lists": int((
            frame.truth_seen_in_training_candidate_lists
            & frame.competitor_seen_in_training_candidate_lists
        ).sum()),
        "reversed_truth_competitor_pair_count": len(reversed_pairs),
        "reversed_truth_competitor_pairs": reversed_pairs,
        "corrected_with_truth_gate_advantage": int((
            (frame.transition == "corrected") & (frame.truth_gate > frame.competitor_gate)
        ).sum()),
        "introduced_with_competitor_gate_advantage": int((
            (frame.transition == "introduced") & (frame.competitor_gate > frame.truth_gate)
        ).sum()),
        "corrected_with_positive_truth_reaction_sum": int((
            (frame.transition == "corrected")
            & (frame["truth_minus_competitor__reaction_sum"].fillna(0) > 0)
        ).sum()),
        "corrected_with_positive_truth_smn_sum": int((
            (frame.transition == "corrected")
            & (frame["truth_minus_competitor__smn_sum"].fillna(0) > 0)
        ).sum()),
        "corrected_with_positive_truth_rt_sum": int((
            (frame.transition == "corrected")
            & (frame["truth_minus_competitor__rt_sum"].fillna(0) > 0)
        ).sum()),
        "output_sha256": sha256(destination),
        "claim_limit": (
            "Descriptive frozen-flip audit. Evidence differences and training exposure "
            "do not establish which feature causally produced a correction."
        ),
    }
    report_destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
