#!/usr/bin/env python
"""Build outcome-labelled *development-only* BioAware context matrices.

No model or threshold is fitted.  Both cohorts have already been inspected, so
the outputs are diagnostic artifacts and may not be used as final validation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.bioaware import (  # noqa: E402
    BioAwareConfig,
    build_one_hop_evidence,
    fuse_candidates,
    top1_transition_table,
)
from annotation.bioaware_context import extract_reaction_context_features  # noqa: E402
from evaluate_bioaware_v2_two_layer import build_expanded_seeds  # noqa: E402


def build_one(
    name: str,
    candidates: pd.DataFrame,
    seeds: pd.DataFrame,
    participants: pd.DataFrame,
    reaction_directions: pd.DataFrame,
    output_dir: Path,
) -> dict:
    config = BioAwareConfig()
    paths = build_one_hop_evidence(participants, seeds, config)
    features, details = extract_reaction_context_features(
        candidates,
        paths,
        participants,
        seeds,
        truth_col="truth_candidate_id",
        exclude_truth_identity=True,
        reaction_directions=reaction_directions,
    )
    score_columns = features[
        [
            "query_id",
            "candidate_id",
            "raw_network_support",
            "dependency_corrected_network_support",
            "raw_path_count",
        ]
    ]
    scored_input = candidates.merge(
        score_columns,
        on=["query_id", "candidate_id"],
        validate="one_to_one",
    ).rename(
        columns={
            "raw_network_support": "network_support",
            "raw_path_count": "network_path_count",
        }
    )
    scored, decisions = fuse_candidates(scored_input, config)
    transitions, result = top1_transition_table(
        scored, truth_col="truth_candidate_id"
    )
    dependency_input = candidates.merge(
        score_columns,
        on=["query_id", "candidate_id"],
        validate="one_to_one",
    ).rename(
        columns={
            "dependency_corrected_network_support": "network_support",
            "raw_path_count": "network_path_count",
        }
    )
    dependency_scored, dependency_decisions = fuse_candidates(
        dependency_input, config
    )
    dependency_transitions, dependency_result = top1_transition_table(
        dependency_scored, truth_col="truth_candidate_id"
    )
    candidate_label_columns = [
        "query_id",
        "candidate_id",
        "spectral_score",
        "truth_candidate_id",
    ]
    if "truth_formula" in candidates:
        candidate_label_columns.append("truth_formula")
    labelled = features.merge(
        candidates[candidate_label_columns],
        on=["query_id", "candidate_id"],
        validate="one_to_one",
    )
    labelled["is_pseudo_or_published_truth"] = (
        labelled["candidate_id"].astype(str)
        == labelled["truth_candidate_id"].astype(str)
    )
    labelled = labelled.merge(
        transitions[
            [
                "query_id",
                "baseline_top_candidate",
                "final_top_candidate",
                "baseline_correct",
                "final_correct",
                "corrected",
                "introduced",
            ]
        ],
        on="query_id",
        validate="many_to_one",
    )
    labelled.to_csv(output_dir / f"{name}__candidate_context.csv.gz", index=False)
    details.to_csv(output_dir / f"{name}__path_context.csv.gz", index=False)
    decisions.to_csv(output_dir / f"{name}__query_decisions.csv", index=False)
    transitions.to_csv(output_dir / f"{name}__transitions.csv", index=False)
    dependency_decisions.to_csv(
        output_dir / f"{name}__dependency_corrected_query_decisions.csv", index=False
    )
    dependency_transitions.to_csv(
        output_dir / f"{name}__dependency_corrected_transitions.csv", index=False
    )
    return {
        "queries": int(candidates["query_id"].nunique()),
        "candidate_rows": int(len(candidates)),
        "seed_rows": int(len(seeds)),
        "seed_compounds": int(seeds["seed_compound_id"].astype(str).nunique()),
        "raw_paths": int(len(paths)),
        "context_paths": int(len(details)),
        "candidates_with_any_evidence": int((features["raw_path_count"] > 0).sum()),
        "candidates_with_complete_evidence": int((features["complete_path_count"] > 0).sum()),
        "candidates_with_only_incomplete_evidence": int(
            ((features["raw_path_count"] > 0) & (features["complete_path_count"] == 0)).sum()
        ),
        "result_under_archived_raw_fusion": result,
        "posthoc_dependency_corrected_fusion": dependency_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", type=Path, default=Path("data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz"))
    parser.add_argument("--reaction-directions", type=Path, default=Path("data/reference/bioaware_rhea_offline_20260827/rhea2reactome.tsv"))
    parser.add_argument(
        "--mtbls1905-input-dir",
        type=Path,
        default=Path("data/external/MTBLS1905/bioaware_v1_input"),
    )
    parser.add_argument(
        "--mtbls1905-auto-seeds",
        type=Path,
        default=None,
        help=(
            "Optional frozen automatic-seed artifact. When omitted, use "
            "seeds_auto.csv from --mtbls1905-input-dir."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_reaction_context_development"))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(args.participants)
    reaction_directions = pd.read_csv(args.reaction_directions, sep="\t")

    mtbls13729_candidates = pd.read_csv(
        "data/mtbls13729/bioaware_v1_input/candidates.csv.gz"
    )
    mtbls13729_seeds = build_expanded_seeds(
        Path("data/mtbls13729/ms1_ms2_link"),
        ["neg_rp", "pos_rp"],
        set(participants["compound_id"].dropna().astype(str)),
        minimum_cosine=0.80,
        minimum_support_spectra=2,
        minimum_agreement=0.60,
    )
    reports = {
        "mtbls13729_expanded": build_one(
            "mtbls13729_expanded",
            mtbls13729_candidates,
            mtbls13729_seeds,
            participants,
            reaction_directions,
            out,
        )
    }

    mtbls1905_candidates = pd.read_csv(args.mtbls1905_input_dir / "candidates.csv.gz")
    auto_seed_path = (
        args.mtbls1905_auto_seeds
        if args.mtbls1905_auto_seeds is not None
        else args.mtbls1905_input_dir / "seeds_auto.csv"
    )
    for regime, seed_path in [
        ("auto", auto_seed_path),
        (
            "published_headroom",
            args.mtbls1905_input_dir / "seeds_published_leave_target_out.csv",
        ),
    ]:
        reports[f"mtbls1905_{regime}"] = build_one(
            f"mtbls1905_{regime}",
            mtbls1905_candidates,
            pd.read_csv(seed_path),
            participants,
            reaction_directions,
            out,
        )
    report = {
        "status": "bioaware_reaction_context_development_complete",
        "formal": False,
        "outcomes_used_for_model_or_threshold_selection": False,
        "cohorts_previously_exposed": ["MTBLS13729", "MTBLS1905"],
        "datasets": reports,
        "contract": (
            "Development-only feature audit. Freeze the schema before obtaining "
            "a new external cohort; do not report these matrices as blind validation."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
