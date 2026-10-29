#!/usr/bin/env python3
"""Join frozen IBD decision blocks to truth-blind complete-axis WSE scores.

Spectral ranking is completed before response estimates or the manually frozen
identity hypotheses are opened.  The output is development/consumed evidence:
it identifies concrete WSE successes and failures for later same-axis model
comparison, but it is not external biological validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATUS = "DEVELOPMENT_CONSUMED_IBD_COMPLETE_BLOCK_WSE_ADJUDICATED"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("tasks/contrast_atlas_ibd_decision_blocks_v1.json"),
    )
    parser.add_argument(
        "--handoff",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_ibd_decision_blocks_v5_20261027/"
            "complete_block_algorithm_handoff.tsv.gz"
        ),
    )
    parser.add_argument(
        "--wse-dir",
        type=Path,
        default=Path("data/validation/contrast_atlas_complete_identity_wse_release_v1_20261029"),
    )
    parser.add_argument(
        "--axis-dir",
        type=Path,
        default=Path("data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027"),
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023/"
            "response_entities.tsv.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/contrast_atlas_ibd_wse_adjudication_v2_20261029"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (args.contract, args.handoff, args.wse_dir / "report.json", args.responses):
        if not path.is_file():
            raise FileNotFoundError(path)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    wse_report_path = args.wse_dir / "report.json"
    wse_report = json.loads(wse_report_path.read_text(encoding="utf-8"))
    if wse_report.get("status") != "DEVELOPMENT_CONSUMED_CONTRAST_ATLAS_COMPLETE_IDENTITY_WSE_RELEASED":
        raise RuntimeError("WSE input is not the compact complete-axis release")
    wse_path = args.wse_dir / "wse_scores_aligned.npz"
    pair_axis_path = args.axis_dir / "directed_complete_candidate_pairs.tsv.gz"
    if sha256(wse_path) != wse_report["outputs"]["wse_scores_aligned.npz"]:
        raise RuntimeError("WSE aligned score hash mismatch")
    if sha256(pair_axis_path) != wse_report["alignment_contract"]["pair_axis_sha256"]:
        raise RuntimeError("directed pair axis differs from the compact WSE release")

    handoff = pd.read_csv(args.handoff, sep="\t", low_memory=False)
    all_scores = pd.read_csv(pair_axis_path, sep="\t", low_memory=False)
    with np.load(wse_path, allow_pickle=False) as body:
        if str(body["schema"].item()) != "contrast_atlas_complete_identity_wse_aligned_v1":
            raise RuntimeError("unexpected compact WSE schema")
        all_scores["wse_score"] = np.asarray(body["wse_score"], dtype=np.float32)
        all_scores["wse_matched_fragments"] = np.asarray(
            body["wse_matched_fragments"], dtype=np.int16
        )
        all_scores["wse_available"] = np.asarray(body["wse_availability"], dtype=bool)
        all_scores["wse_rank_min"] = np.asarray(body["wse_rank_min"], dtype=np.int32)
        all_scores["wse_rank_stable"] = np.asarray(body["wse_rank_stable"], dtype=np.int32)
    scores = all_scores[[
        "directed_pair_id",
        "wse_score",
        "wse_matched_fragments",
        "wse_available",
        "wse_rank_min",
        "wse_rank_stable",
    ]]
    pairs = handoff.merge(scores, on="directed_pair_id", how="left", validate="one_to_one")
    if pairs["wse_score"].isna().any() or not pairs["wse_available"].all():
        raise RuntimeError("targeted handoff is not fully scored by WSE")

    candidate_sets = {row["candidate_set_id"]: row for row in contract["candidate_sets"]}
    stercobilin = set(candidate_sets["stercobilin_like_mz595_response_hypothesis"]["entities"])
    expected_5asa = frozenset(
        candidate_sets["deoxycholic_acid_5asa_identity_disambiguation"]["expected_identity_pair"]
    )
    competitor_5asa = candidate_sets["deoxycholic_acid_5asa_identity_disambiguation"]["known_competitor"]

    def role(row: pd.Series) -> str:
        unordered = frozenset((str(row["query_entity_id"]), str(row["candidate_entity_id"])))
        if unordered == expected_5asa:
            return "frozen_expected_5asa_identity_pair"
        if str(row["candidate_entity_id"]) == competitor_5asa:
            return "frozen_5asa_same_mass_competitor"
        if {str(row["query_entity_id"]), str(row["candidate_entity_id"])}.issubset(stercobilin):
            return "frozen_stercobilin_like_unresolved_hypothesis"
        return "other_complete_block_candidate"

    pairs["frozen_evidence_role"] = pairs.apply(role, axis=1)

    response = pd.read_csv(args.responses, sep="\t", low_memory=False)
    entities = set(pairs["query_entity_id"].astype(str)) | set(pairs["candidate_entity_id"].astype(str))
    response = response.loc[
        response["entity_id"].astype(str).isin(entities),
        [
            "study_id",
            "entity_id",
            "contrast_id",
            "beta",
            "se",
            "effect_model",
            "data_role",
        ],
    ].copy()
    if response["entity_id"].duplicated().any():
        raise RuntimeError("targeted entity has multiple response rows")
    response["z"] = response["beta"] / response["se"]
    lookup = response.set_index("entity_id")
    for side in ("query", "candidate"):
        ids = pairs[f"{side}_entity_id"].astype(str)
        pairs[f"{side}_contrast_id"] = ids.map(lookup["contrast_id"])
        pairs[f"{side}_beta"] = ids.map(lookup["beta"])
        pairs[f"{side}_se"] = ids.map(lookup["se"])
        pairs[f"{side}_z"] = ids.map(lookup["z"])

    block_rows: list[dict[str, object]] = []
    for block_id, block in pairs.groupby("query_block_id", sort=False):
        ordered = block.sort_values("wse_rank_stable", kind="stable")
        top = ordered.iloc[0]
        block_rows.append({
            "query_block_id": block_id,
            "query_entity_id": top["query_entity_id"],
            "candidate_study_id": top["candidate_study_id"],
            "candidate_count": int(len(ordered)),
            "wse_top1_candidate_entity_id": top["candidate_entity_id"],
            "wse_top1_score": float(top["wse_score"]),
            "wse_top1_evidence_role": top["frozen_evidence_role"],
            "wse_expected_hypothesis_rank": (
                int(ordered.loc[
                    ordered["frozen_evidence_role"].isin([
                        "frozen_expected_5asa_identity_pair",
                        "frozen_stercobilin_like_unresolved_hypothesis",
                    ]),
                    "wse_rank_min",
                ].min())
                if ordered["frozen_evidence_role"].isin([
                    "frozen_expected_5asa_identity_pair",
                    "frozen_stercobilin_like_unresolved_hypothesis",
                ]).any()
                else np.nan
            ),
        })
    blocks = pd.DataFrame(block_rows)

    # The late-retention-time m/z 595 pair was resolved after the original
    # targeted contract.  Keep it explicitly exploratory rather than silently
    # upgrading it to a preregistered hypothesis.
    late_ids = {"IBD200_GROUP::1286", "MSV000097610::28796"}
    late = all_scores.loc[
        all_scores["query_entity_id"].astype(str).isin(late_ids)
        & all_scores["candidate_entity_id"].astype(str).isin(late_ids)
    ].copy()
    late["evidence_role"] = "post_contract_exploratory_distinct_mz595_spectral_family"
    late = late.merge(
        response[["entity_id", "contrast_id", "beta", "se", "z"]].rename(
            columns={
                "entity_id": "query_entity_id",
                "contrast_id": "query_contrast_id",
                "beta": "query_beta",
                "se": "query_se",
                "z": "query_z",
            }
        ),
        on="query_entity_id",
        how="left",
        validate="many_to_one",
    )

    output = args.output_dir
    output.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    pair_path = output / "targeted_pair_scores.tsv.gz"
    block_path = output / "targeted_block_decisions.tsv"
    response_path = output / "targeted_entity_responses.tsv"
    late_path = output / "exploratory_distinct_mz595_pair.tsv"
    pairs.to_csv(pair_path, sep="\t", index=False, compression=compression)
    blocks.to_csv(block_path, sep="\t", index=False)
    response.sort_values(["study_id", "entity_id"]).to_csv(response_path, sep="\t", index=False)
    late.to_csv(late_path, sep="\t", index=False)

    expected_row = pairs.loc[
        pairs["frozen_evidence_role"].eq("frozen_expected_5asa_identity_pair")
        & pairs["query_entity_id"].eq("MSV000097610::37125")
    ].iloc[0]
    competitor_row = pairs.loc[
        pairs["query_entity_id"].eq("MSV000097610::37125")
        & pairs["candidate_entity_id"].eq(competitor_5asa)
    ].iloc[0]
    stercobilin_response = response.loc[response["entity_id"].isin(stercobilin)]
    report = {
        "schema": "contrast_atlas_ibd_wse_adjudication_v2_compact_release",
        "status": STATUS,
        "data_role": "development_consumed",
        "truth_order": "WSE scores frozen before response estimates and candidate-set labels were joined",
        "denominators": {
            "targeted_candidate_pairs": int(len(pairs)),
            "targeted_query_blocks": int(len(blocks)),
            "stercobilin_like_entities_with_response": int(len(stercobilin_response)),
            "post_contract_late_mz595_directed_pairs": int(len(late)),
        },
        "findings": {
            "stercobilin_like_response_direction": {
                "entities": sorted(stercobilin),
                "beta": {
                    str(row.entity_id): float(row.beta)
                    for row in stercobilin_response.itertuples(index=False)
                },
                "all_three_negative": bool((stercobilin_response["beta"] < 0).all()),
                "boundary": (
                    "descriptive three-study unresolved response support; effect scales differ, "
                    "strict-known baseline already retrieves the IBD study relation, and exact identity is unconfirmed"
                ),
            },
            "wse_5asa_identity_control": {
                "query": "MSV000097610::37125",
                "expected_candidate": str(expected_row["candidate_entity_id"]),
                "expected_rank": int(expected_row["wse_rank_min"]),
                "expected_score": float(expected_row["wse_score"]),
                "wrong_top1_candidate": str(competitor_row["candidate_entity_id"]),
                "wrong_top1_score": float(competitor_row["wse_score"]),
                "interpretation": (
                    "frozen WSE failure on a full-InChIKey identity positive control; medication confounding "
                    "precludes an IBD-mechanism claim but creates a legal same-axis algorithm-correction test"
                ),
            },
            "late_mz595_pair": {
                "entities": sorted(late_ids),
                "directed_pairs": int(len(late)),
                "both_rank1": bool(len(late) == 2 and late["wse_rank_min"].eq(1).all()),
                "boundary": "post-contract exploratory pair; distinct from the early m/z 595 family and not pooled",
            },
        },
        "success_gate_remaining": [
            "Score the identical complete blocks with Official DreaMS, Noise V1 and Full23 v1.",
            "Determine whether a self-developed method corrects the WSE 5-ASA error without using response labels.",
            "Confirm any unresolved IBD identity and response link in an unconsumed cohort or orthogonal chemical experiment.",
        ],
        "claim_boundary": [
            "This is a consumed development adjudication, not external confirmation.",
            "Response concordance did not select or rerank any candidate.",
            "The unresolved m/z 595 entities are identity hypotheses; beta and SE are not pooled.",
            "The 5-ASA block is an identity positive control, not an IBD mechanism endpoint.",
        ],
        "provenance": {
            "contract": {"path": str(args.contract), "sha256": sha256(args.contract)},
            "handoff": {"path": str(args.handoff), "sha256": sha256(args.handoff)},
            "wse_release_report": {
                "path": str(wse_report_path),
                "sha256": sha256(wse_report_path),
            },
            "wse_aligned_scores": {"path": str(wse_path), "sha256": sha256(wse_path)},
            "directed_pair_axis": {"path": str(pair_axis_path), "sha256": sha256(pair_axis_path)},
            "responses": {"path": str(args.responses), "sha256": sha256(args.responses)},
            "implementation": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))},
        },
        "outputs": {
            "targeted_pair_scores.tsv.gz": sha256(pair_path),
            "targeted_block_decisions.tsv": sha256(block_path),
            "targeted_entity_responses.tsv": sha256(response_path),
            "exploratory_distinct_mz595_pair.tsv": sha256(late_path),
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["findings"], indent=2), flush=True)


if __name__ == "__main__":
    main()
