#!/usr/bin/env python
"""Frozen MS1 baselines for ST003325 structure-hidden perturbation replication."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PANELS = ("AN005444", "AN005445")
EDIT_CLASSES = ("CE", "KO", "PTC")
SEED = 20261012


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine(vector: np.ndarray, signature: np.ndarray) -> float:
    denominator = np.linalg.norm(vector) * np.linalg.norm(signature)
    return float(np.dot(vector, signature) / denominator) if denominator > 0 else float("nan")


def build_clone_response(
    table_path: Path, linkage_path: Path, design_path: Path, median_center: bool
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    table = pd.read_csv(table_path, sep="\t")
    linkage = pd.read_csv(linkage_path)
    design = pd.read_csv(design_path)
    if not np.array_equal(
        (design["analysis_id"].iloc[0] + ":" + table["id_number"].astype(str)).to_numpy(),
        linkage["query_id"].to_numpy(),
    ):
        raise ValueError(f"Feature order mismatch: {table_path.name}")
    qualified = linkage["measurement_qualified"].astype(bool).to_numpy()
    linked = qualified & linkage["linked_replicated"].astype(bool).to_numpy()
    sample_columns = design["sample_id"].tolist()
    values = np.log1p(table[sample_columns].to_numpy(float))
    if median_center:
        values = values - np.median(values[qualified, :], axis=0, keepdims=True)

    responses: dict[str, np.ndarray] = {}
    for clone, rows in design.groupby("clone"):
        basal = rows.loc[rows["condition"] == "Basal", "sample_id"].tolist()
        dheas = rows.loc[rows["condition"] == "Basal+DHEAS", "sample_id"].tolist()
        if len(basal) != 3 or len(dheas) != 3:
            raise ValueError(f"{clone}: expected 3+3 plates")
        basal_idx = [sample_columns.index(value) for value in basal]
        dheas_idx = [sample_columns.index(value) for value in dheas]
        responses[clone] = np.median(values[:, dheas_idx], axis=1) - np.median(values[:, basal_idx], axis=1)
    return responses, qualified, linked


def evaluate(
    panel_responses: dict[str, dict[str, np.ndarray]],
    masks: dict[str, np.ndarray],
    assignment: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    assignment = assignment or {clone: clone for clone in next(iter(panel_responses.values()))}
    role_vectors = {
        panel: {role: vectors[source] for role, source in assignment.items()}
        for panel, vectors in panel_responses.items()
    }
    rows = []
    wt_clones = ("WT_1", "WT_2", "WT_3")
    class_clones = {
        edit: sorted(clone for clone in assignment if clone.startswith(edit + "_"))
        for edit in EDIT_CLASSES
    }
    for held_edit in EDIT_CLASSES:
        train_edits = [edit for edit in EDIT_CLASSES if edit != held_edit]
        train_mutants = [clone for edit in train_edits for clone in class_clones[edit]]
        for held_wt in wt_clones:
            train_wt = [clone for clone in wt_clones if clone != held_wt]
            panel_scores: dict[str, dict[str, float]] = {}
            for panel, vectors in role_vectors.items():
                mask = masks[panel]
                mutant = np.stack([vectors[clone][mask] for clone in train_mutants])
                wt = np.stack([vectors[clone][mask] for clone in train_wt])
                wt_centroid = np.median(wt, axis=0)
                signature = np.median(mutant, axis=0) - wt_centroid
                test_clones = class_clones[held_edit] + [held_wt]
                panel_scores[panel] = {
                    clone: cosine(vectors[clone][mask] - wt_centroid, signature)
                    for clone in test_clones
                }
            for mutant_clone in class_clones[held_edit]:
                record: dict[str, object] = {
                    "held_edit_class": held_edit,
                    "held_wt_clone": held_wt,
                    "held_mutant_clone": mutant_clone,
                }
                combined_mutant = 0.0
                combined_wt = 0.0
                for panel in PANELS:
                    mutant_score = panel_scores[panel][mutant_clone]
                    wt_score = panel_scores[panel][held_wt]
                    record[f"{panel}_mutant_score"] = mutant_score
                    record[f"{panel}_wt_score"] = wt_score
                    record[f"{panel}_margin"] = mutant_score - wt_score
                    record[f"{panel}_correct"] = bool(mutant_score > wt_score)
                    combined_mutant += mutant_score / len(PANELS)
                    combined_wt += wt_score / len(PANELS)
                record["combined_mutant_score"] = combined_mutant
                record["combined_wt_score"] = combined_wt
                record["combined_margin"] = combined_mutant - combined_wt
                record["combined_correct"] = bool(combined_mutant > combined_wt)
                rows.append(record)
    return rows


def summarize(rows: list[dict[str, object]], suffix: str) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    correct = frame[f"{suffix}_correct"].astype(bool)
    by_edit = {
        edit: {
            "correct": int(frame.loc[frame["held_edit_class"] == edit, f"{suffix}_correct"].sum()),
            "denominator": 9,
            "accuracy": float(frame.loc[frame["held_edit_class"] == edit, f"{suffix}_correct"].mean()),
            "median_margin": float(frame.loc[frame["held_edit_class"] == edit, f"{suffix}_margin"].median()),
        }
        for edit in EDIT_CLASSES
    }
    return {
        "correct": int(correct.sum()),
        "denominator": int(len(frame)),
        "accuracy": float(correct.mean()),
        "median_margin": float(frame[f"{suffix}_margin"].median()),
        "by_held_edit_class": by_edit,
    }


def permutation_pvalue(
    panel_responses: dict[str, dict[str, np.ndarray]], masks: dict[str, np.ndarray], observed: int, repeats: int
) -> tuple[float, list[int]]:
    rng = np.random.default_rng(SEED)
    roles = sorted(next(iter(panel_responses.values())).keys())
    sources = np.array(roles, dtype=object)
    null = []
    for _ in range(repeats):
        permuted = rng.permutation(sources)
        assignment = dict(zip(roles, permuted.tolist()))
        rows = evaluate(panel_responses, masks, assignment)
        null.append(sum(bool(row["combined_correct"]) for row in rows))
    pvalue = (1 + sum(value >= observed for value in null)) / (repeats + 1)
    return float(pvalue), null


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--linkage", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    analyses = {}
    provenance = {}
    for normalization, median_center in (("median_centered_log1p", True), ("raw_log1p", False)):
        panel_responses = {}
        qualified_masks = {}
        linked_masks = {}
        for panel in PANELS:
            table_path = args.tables / f"ST003325_{panel}_Results.txt"
            linkage_path = args.linkage / f"{panel}_feature_linkage.csv.gz"
            design_path = args.design / f"{panel}_design.csv"
            responses, qualified, linked = build_clone_response(
                table_path, linkage_path, design_path, median_center
            )
            panel_responses[panel] = responses
            qualified_masks[panel] = qualified
            linked_masks[panel] = linked
            provenance[table_path.name] = sha256(table_path)
            provenance[linkage_path.name] = sha256(linkage_path)
            provenance[design_path.name] = sha256(design_path)

        analyses[normalization] = {}
        for universe, masks in (("MS1_all", qualified_masks), ("MS1_MS2_observed", linked_masks)):
            rows = evaluate(panel_responses, masks)
            pd.DataFrame(rows).to_csv(
                args.output / f"{normalization}_{universe}_folds.csv", index=False
            )
            summary = {
                "feature_denominators": {panel: int(masks[panel].sum()) for panel in PANELS},
                "combined": summarize(rows, "combined"),
                "by_panel": {panel: summarize(rows, panel) for panel in PANELS},
            }
            if normalization == "median_centered_log1p" and universe == "MS1_all":
                observed = summary["combined"]["correct"]
                pvalue, null = permutation_pvalue(panel_responses, masks, observed, args.permutations)
                summary["clone_label_permutation"] = {
                    "seed": SEED,
                    "repeats": args.permutations,
                    "right_tail_p": pvalue,
                    "null_mean_correct": float(np.mean(null)),
                    "null_q025_correct": float(np.quantile(null, 0.025)),
                    "null_q975_correct": float(np.quantile(null, 0.975)),
                }
            else:
                summary["clone_label_permutation"] = {
                    "status": "not_run_secondary_analysis",
                    "reason": "The frozen primary MS1-all analysis alone controls progression.",
                }
            analyses[normalization][universe] = summary

    primary = analyses["median_centered_log1p"]["MS1_all"]
    sensitivity = analyses["raw_log1p"]["MS1_all"]
    gates = {
        "primary_each_edit_accuracy_gt_50pct": all(
            value["accuracy"] > 0.5 for value in primary["combined"]["by_held_edit_class"].values()
        ),
        "primary_permutation_p_lt_0_05": primary["clone_label_permutation"]["right_tail_p"] < 0.05,
        "raw_log1p_overall_direction_consistent": sensitivity["combined"]["accuracy"] > 0.5,
        "raw_log1p_each_edit_direction_consistent": all(
            value["accuracy"] > 0.5 for value in sensitivity["combined"]["by_held_edit_class"].values()
        ),
    }
    report = {
        "status": "ST003325_STRUCTURE_HIDDEN_MS1_BASELINE_COMPLETE",
        "formal": True,
        "dataset": "Metabolomics Workbench ST003325 / PR001844",
        "data_role": "development/consumed",
        "sealed": False,
        "structure_or_pathway_labels_used": False,
        "unit_of_replication": "clone and held-out edit class; not injection",
        "analyses": analyses,
        "gates": gates,
        "pass_to_frozen_spectral_increment": all(gates.values()),
        "provenance": provenance,
        "claim_limit": (
            "Structure-hidden cross-edit replication of a design-labelled plate-set response. Because "
            "Basal and Basal+DHEAS occupy disjoint plate numbers, this is not an unconfounded causal "
            "DHEA-S effect and does not establish metabolite identity, STS flux, or cross-study transfer."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
