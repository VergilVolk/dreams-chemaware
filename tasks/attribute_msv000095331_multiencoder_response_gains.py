#!/usr/bin/env python3
"""Attribute registered response-rank gains to changed identity hypotheses.

This analysis is evaluation-only.  It does not fit, select, or calibrate a
spectral method from biological responses.  For every prespecified context in
which a frozen encoder gains Top-1 relative to Noise V1, it decomposes the
change in the response numerator into gained and lost mutual identity edges.
The result identifies concrete entities for orthogonal follow-up without
calling an unverified identity hypothesis a correction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_msv000095331_integrated_algorithm_handoff import sha256


STATUS = "MSV000095331_MULTIENCODER_RESPONSE_GAIN_ATTRIBUTION_COMPLETE"
REFERENCE = "noise_v1"
DEFAULT_METHODS = ("noise_chemaware_fullmsg", "chemaware_phase_a")


def context_mask(table: pd.DataFrame, row: pd.Series, method: str) -> pd.Series:
    return (
        table["method_name"].eq(method)
        & table["query_contrast_id"].eq(row["query_contrast_id"])
        & table["candidate_study_id"].eq(row["candidate_study_id"])
        & table["candidate_contrast_id"].eq(row["candidate_contrast_id"])
        & table["prespecified_overlap"].eq(True)
    )


def edge_view(block: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "selected_candidate_pair_id", "query_id", "query_raw_entity_id",
        "selected_candidate_id", "selected_candidate_raw_entity_id",
        "selected_method_score", "selected_score_margin", "ranked_candidate_count",
        "selected_precursor_ppm_difference", "query_beta", "query_se",
        "candidate_beta", "candidate_se", "response_product", "direction_relation",
    ]
    output = block[columns].drop_duplicates("selected_candidate_pair_id").copy()
    return output.set_index("selected_candidate_pair_id", drop=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--increment-dir", type=Path,
        default=Path(
            "data/validation/msv000095331_entity_method_increment_multiencoder_run_2404230"
        ),
    )
    parser.add_argument(
        "--summary-dir", type=Path,
        default=Path(
            "data/validation/msv000095331_multiencoder_increment_summary_v1_20261024"
        ),
    )
    parser.add_argument(
        "--identity-handoff", type=Path,
        default=Path("data/validation/msv000095331_complete_identity_handoff_v3_20261023"),
    )
    parser.add_argument(
        "--product-registry", type=Path,
        default=Path(
            "data/validation/msv000095331_published_product_direction_check_v1_20261023/"
            "prespecified_product_registry.tsv"
        ),
    )
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    methods = tuple(args.method) if args.method else DEFAULT_METHODS

    rank_path = args.summary_dir / "prespecified_rank_comparison.tsv"
    support_path = args.increment_dir / "entity_response_support.tsv.gz"
    score_path = args.increment_dir / "response_retrieval_scores.tsv.gz"
    pair_path = args.identity_handoff / "complete_identity_candidate_pairs.tsv.gz"
    ranks = pd.read_csv(rank_path, sep="\t")
    ranks = ranks.loc[ranks["method_name"].isin(methods)].copy()
    top1_gains = ranks.loc[
        ranks["candidate_rank_method"].eq(1)
        & ranks["candidate_rank_noise"].gt(1)
    ].copy()
    support = pd.read_csv(
        support_path,
        sep="\t",
        dtype={
            "query_id": str, "query_raw_entity_id": str,
            "selected_candidate_id": str, "selected_candidate_raw_entity_id": str,
            "selected_candidate_pair_id": str,
        },
    )
    support = support.loc[
        support["method_name"].isin((REFERENCE, *methods))
        & support["prespecified_overlap"].eq(True)
    ].copy()
    scores = pd.read_csv(score_path, sep="\t")
    pairs = pd.read_csv(pair_path, sep="\t", dtype=str)
    pair_columns = [
        "candidate_pair_id", "query_identity_status", "candidate_identity_status",
        "query_precursor_mz", "candidate_precursor_mz", "precursor_ppm_difference",
        "wse_score", "wse_matched_fragments",
    ]
    pairs = pairs[pair_columns].drop_duplicates("candidate_pair_id")
    products = pd.read_csv(args.product_registry, sep="\t", dtype={"entity_id": str})
    product_by_entity = products.set_index("entity_id")["target_id"].to_dict()

    method_context_edge_sets = {
        (str(method), str(query), str(study), str(candidate)): set(
            block["selected_candidate_pair_id"].astype(str)
        )
        for (method, query, study, candidate), block in support.groupby(
            [
                "method_name", "query_contrast_id", "candidate_study_id",
                "candidate_contrast_id",
            ],
            sort=False,
        )
    }
    edge_rows: list[dict] = []
    context_rows: list[dict] = []
    for gain in top1_gains.to_dict("records"):
        gain_row = pd.Series(gain)
        method = str(gain["method_name"])
        context_key = (
            str(gain["query_contrast_id"]),
            str(gain["candidate_study_id"]),
            str(gain["candidate_contrast_id"]),
        )
        method_block = edge_view(support.loc[context_mask(support, gain_row, method)])
        noise_block = edge_view(support.loc[context_mask(support, gain_row, REFERENCE)])
        edge_ids = sorted(set(method_block.index) | set(noise_block.index))
        local_rows: list[dict] = []
        for edge_id in edge_ids:
            in_method = edge_id in method_block.index
            in_noise = edge_id in noise_block.index
            method_row = method_block.loc[edge_id] if in_method else None
            noise_row = noise_block.loc[edge_id] if in_noise else None
            source = method_row if in_method else noise_row
            method_product = float(method_row["response_product"]) if in_method else 0.0
            noise_product = float(noise_row["response_product"]) if in_noise else 0.0
            state = "retained" if in_method and in_noise else "gained" if in_method else "lost"
            local_rows.append({
                "method_name": method,
                "query_contrast_id": str(gain["query_contrast_id"]),
                "candidate_study_id": str(gain["candidate_study_id"]),
                "candidate_contrast_id": str(gain["candidate_contrast_id"]),
                "candidate_rank_noise": float(gain["candidate_rank_noise"]),
                "candidate_rank_method": float(gain["candidate_rank_method"]),
                "edge_state_vs_noise": state,
                "candidate_pair_id": str(edge_id),
                "query_id": str(source["query_id"]),
                "query_raw_entity_id": str(source["query_raw_entity_id"]),
                "candidate_id": str(source["selected_candidate_id"]),
                "candidate_raw_entity_id": str(source["selected_candidate_raw_entity_id"]),
                "method_score": (
                    float(method_row["selected_method_score"]) if in_method else np.nan
                ),
                "noise_score": (
                    float(noise_row["selected_method_score"]) if in_noise else np.nan
                ),
                "method_score_margin": (
                    float(method_row["selected_score_margin"]) if in_method else np.nan
                ),
                "noise_score_margin": (
                    float(noise_row["selected_score_margin"]) if in_noise else np.nan
                ),
                "query_beta": float(source["query_beta"]),
                "query_se": float(source["query_se"]),
                "candidate_beta": float(source["candidate_beta"]),
                "candidate_se": float(source["candidate_se"]),
                "method_response_product": method_product,
                "noise_response_product": noise_product,
                "delta_response_numerator": method_product - noise_product,
                "delta_role": (
                    "supports_gain" if method_product - noise_product > 0
                    else "opposes_gain" if method_product - noise_product < 0
                    else "neutral"
                ),
                "published_product_target": product_by_entity.get(
                    str(source["query_raw_entity_id"]), ""
                ),
                "also_selected_by_phase_a_same_context": edge_id in method_context_edge_sets.get(
                    ("chemaware_phase_a", *context_key), set()
                ),
                "also_selected_by_fullmsg_same_context": edge_id in method_context_edge_sets.get(
                    ("noise_chemaware_fullmsg", *context_key), set()
                ),
            })
        local = pd.DataFrame(local_rows)
        edge_rows.extend(local_rows)
        changed = local.loc[local["edge_state_vs_noise"].ne("retained")]
        positive = changed.loc[changed["delta_response_numerator"].gt(0)]
        negative = changed.loc[changed["delta_response_numerator"].lt(0)]
        method_score_row = scores.loc[
            scores["method_name"].eq(method)
            & scores["query_contrast_id"].eq(gain["query_contrast_id"])
            & scores["candidate_study_id"].eq(gain["candidate_study_id"])
            & scores["candidate_contrast_id"].eq(gain["candidate_contrast_id"])
        ].iloc[0]
        noise_score_row = scores.loc[
            scores["method_name"].eq(REFERENCE)
            & scores["query_contrast_id"].eq(gain["query_contrast_id"])
            & scores["candidate_study_id"].eq(gain["candidate_study_id"])
            & scores["candidate_contrast_id"].eq(gain["candidate_contrast_id"])
        ].iloc[0]
        observed_delta = float(
            method_score_row["response_numerator"] - noise_score_row["response_numerator"]
        )
        decomposed_delta = float(local["delta_response_numerator"].sum())
        if not np.isclose(observed_delta, decomposed_delta, atol=1e-9, rtol=1e-9):
            raise RuntimeError("edge decomposition does not reproduce response numerator")
        positive_sum = float(positive["delta_response_numerator"].sum())
        context_rows.append({
            **gain,
            "noise_mutual_edges": int(len(noise_block)),
            "method_mutual_edges": int(len(method_block)),
            "gained_edges": int((local["edge_state_vs_noise"] == "gained").sum()),
            "lost_edges": int((local["edge_state_vs_noise"] == "lost").sum()),
            "helpful_changed_edges": int(len(positive)),
            "harmful_changed_edges": int(len(negative)),
            "positive_delta_sum": positive_sum,
            "negative_delta_sum": float(negative["delta_response_numerator"].sum()),
            "net_response_numerator_delta": observed_delta,
            "largest_helpful_edge_fraction": (
                float(positive["delta_response_numerator"].max() / positive_sum)
                if positive_sum > 0 else np.nan
            ),
        })

    edge_table = pd.DataFrame(edge_rows)
    edge_table = edge_table.merge(
        pairs, left_on="candidate_pair_id", right_on="candidate_pair_id",
        how="left", validate="many_to_one",
    )
    edge_table["identity_evidence_class"] = np.select(
        [
            edge_table["query_identity_status"].eq("library_supported")
            & edge_table["candidate_identity_status"].eq("library_supported"),
            edge_table["query_identity_status"].eq("library_supported")
            | edge_table["candidate_identity_status"].eq("library_supported"),
        ],
        ["both_library_supported", "one_library_supported"],
        default="both_unresolved",
    )
    context_table = pd.DataFrame(context_rows)
    changed = edge_table.loc[edge_table["edge_state_vs_noise"].ne("retained")].copy()
    changed["absolute_delta_response_numerator"] = changed["delta_response_numerator"].abs()
    representatives = changed.sort_values(
        "absolute_delta_response_numerator", ascending=False, kind="mergesort"
    ).groupby(
        [
            "method_name", "query_contrast_id", "candidate_study_id",
            "candidate_contrast_id", "delta_role",
        ],
        sort=False,
    ).head(6)

    args.output_dir.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    outputs = {
        "top1_gain_contexts": args.output_dir / "top1_gain_contexts.tsv",
        "edge_attribution": args.output_dir / "top1_gain_edge_attribution.tsv.gz",
        "representative_entities": args.output_dir / "representative_changed_entities.tsv",
    }
    context_table.to_csv(outputs["top1_gain_contexts"], sep="\t", index=False)
    edge_table.to_csv(
        outputs["edge_attribution"], sep="\t", index=False, compression=compression
    )
    representatives.to_csv(outputs["representative_entities"], sep="\t", index=False)

    report = {
        "status": STATUS,
        "dataset_role": "development_consumed_not_external_validation",
        "reference_method": REFERENCE,
        "methods_attributed": list(methods),
        "denominators": {
            "top1_gain_contexts": int(len(context_table)),
            "changed_edges_in_gain_contexts": int(len(changed)),
            "helpful_changed_edges": int(changed["delta_response_numerator"].gt(0).sum()),
            "harmful_changed_edges": int(changed["delta_response_numerator"].lt(0).sum()),
            "both_unresolved_changed_edges": int(
                changed["identity_evidence_class"].eq("both_unresolved").sum()
            ),
            "changed_edges_touching_library_support": int(
                changed["identity_evidence_class"].ne("both_unresolved").sum()
            ),
            "changed_edges_on_published_product_queries": int(
                changed["published_product_target"].fillna("").ne("").sum()
            ),
            "helpful_changed_edges_both_unresolved": int(
                (
                    changed["delta_response_numerator"].gt(0)
                    & changed["identity_evidence_class"].eq("both_unresolved")
                ).sum()
            ),
            "helpful_changed_edges_touching_library_support": int(
                (
                    changed["delta_response_numerator"].gt(0)
                    & changed["identity_evidence_class"].ne("both_unresolved")
                ).sum()
            ),
            "helpful_gained_edges": int(
                (
                    changed["delta_response_numerator"].gt(0)
                    & changed["edge_state_vs_noise"].eq("gained")
                ).sum()
            ),
            "helpful_removed_edges": int(
                (
                    changed["delta_response_numerator"].gt(0)
                    & changed["edge_state_vs_noise"].eq("lost")
                ).sum()
            ),
        },
        "inputs": {
            "rank_comparison": {"path": str(rank_path), "sha256": sha256(rank_path)},
            "response_support": {"path": str(support_path), "sha256": sha256(support_path)},
            "response_scores": {"path": str(score_path), "sha256": sha256(score_path)},
            "identity_pairs": {"path": str(pair_path), "sha256": sha256(pair_path)},
            "product_registry": {
                "path": str(args.product_registry), "sha256": sha256(args.product_registry)
            },
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "claim_boundary": [
            "Biological responses were opened only for post-freeze evaluation and attribution.",
            "An edge that improves a registered response rank is not thereby a correct molecular identity.",
            "Repeated microbial strata are contexts within one experiment, not independent confirmations.",
            "Representative entities are follow-up priorities and cannot be fed back into model selection.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
