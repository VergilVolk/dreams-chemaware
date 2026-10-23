#!/usr/bin/env python3
"""Open cross-project retrieval of drug-microbiome response contrasts.

MSV000095331 provides four mixture-level microbiota responses.  This script
queries them against every drug contrast in three earlier ChemProp sources.
Spectral edges are built before response values or drug names are loaded.

The experiment deliberately reports two different scientific tracks:

* precursor-compatible mutual top-1 edges are identity hypotheses;
* mass-incompatible mutual top-1 edges are chemical-relation hypotheses.

Neither track calls a spectrum a metabolite.  The primary output is the open
candidate rank of the independently prespecified overlapping drugs, together
with the concrete unresolved response edges that support those ranks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

from run_contrast_atlas_sparse_fragment_measure_baseline import (
    iter_mgf,
    load_study_spectra,
    normalise_entity_id,
    pad_columns,
    spectrum_id,
)
from build_contrast_atlas_open_response_collection import load_contract


STATUS = "MSV000095331_CHEMPROP_OPEN_RESPONSE_RETRIEVAL_COMPLETE"
NEW_STUDY = "MSV000095331"
WORLDS = ("all", "known", "unresolved")
TRUE_OVERLAPS = {
    "drug_mix_1_pooled_microbiota": "drug_sertraline",
    "drug_mix_3_pooled_microbiota": "drug_omeprazole",
    "drug_mix_4_pooled_microbiota": "drug_loratadine",
}
TRUE_DRUG_BY_MIXTURE = {
    1: "drug_sertraline",
    3: "drug_omeprazole",
    4: "drug_loratadine",
}
CHEMPROP_MGF = {
    "MSV000094899": Path("data/reference/chemprop2_m1_20261011/consensus.mgf"),
    "MSV000096724": Path(
        "data/reference/chemprop_cross_design_20261011/chemprop_23drug/spectra/specs_ms.mgf"
    ),
    "MSV000093571": Path(
        "data/reference/chemprop_cross_design_20261011/chemprop_45drug/spectra/specs_ms.mgf"
    ),
}
CANDIDATE_CONTRACT = Path("tasks/contrast_atlas_open_collection_chemprop_extension_v1.json")


def candidate_spectra(scope: str, contract_path: Path) -> dict[str, Path]:
    if scope == "chemprop":
        return dict(CHEMPROP_MGF)
    contract, _ = load_contract(contract_path, Path("."))
    return {
        str(source["study_id"]): Path(source["mgf_path"])
        for source in contract["sources"]
    }


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    output = np.full(len(pvalues), np.nan)
    finite = np.flatnonzero(np.isfinite(pvalues))
    if not len(finite):
        return output
    order = finite[np.argsort(pvalues[finite], kind="mergesort")]
    ranked = pvalues[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output[order] = np.minimum(ranked, 1.0)
    return output


def add_inference(table: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    for _, block in table.groupby(["study_id", "contrast_id"], sort=False):
        block = block.copy()
        z = np.divide(
            block["beta"].to_numpy(float),
            block["se"].to_numpy(float),
            out=np.zeros(len(block), dtype=float),
            where=block["se"].to_numpy(float) > 0,
        )
        block["pvalue"] = 2.0 * stats.norm.sf(np.abs(z))
        block["qvalue"] = bh_qvalues(block["pvalue"].to_numpy(float))
        denominator = np.sqrt(
            block["beta"].to_numpy(float) ** 2
            + block["se"].to_numpy(float) ** 2
        )
        block["response_weight"] = np.divide(
            block["beta"].to_numpy(float),
            denominator,
            out=np.zeros(len(block), dtype=float),
            where=denominator > 0,
        )
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def load_new_effects(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", dtype={"entity_id": str})
    table = table.loc[
        table["contrast_id"].str.match(
            r"^drug_mix_[1-4]_(pooled_microbiota|group_3_medium_slow|"
            r"group_4_slow|full_syncom|syncom_red)$"
        )
    ].copy()
    extracted = table["contrast_id"].str.extract(r"^drug_mix_([1-4])_(.+)$")
    table["mixture"] = extracted[0].astype(int)
    table["query_stratum"] = extracted[1]
    table.insert(0, "study_id", NEW_STUDY)
    table["raw_entity_id"] = table["entity_id"].map(normalise_entity_id)
    table["entity_id"] = NEW_STUDY + "::" + table["raw_entity_id"]
    return add_inference(table)


def load_candidate_effects(path: Path, candidate_studies: set[str]) -> pd.DataFrame:
    use = [
        "study_id", "raw_entity_id", "entity_id", "contrast_id", "beta", "se",
        "prevalence",
    ]
    table = pd.read_csv(path, sep="\t", usecols=use, dtype=str)
    table = table.loc[table["study_id"].isin(candidate_studies)].copy()
    for column in ("beta", "se", "prevalence"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table["raw_entity_id"] = table["raw_entity_id"].map(normalise_entity_id)
    if table[["beta", "se"]].isna().any().any():
        raise ValueError("ChemProp response table contains invalid beta/SE")
    return add_inference(table)


def load_identity(
    new_identity: Path,
    common_root: Path,
    candidate_studies: set[str],
) -> pd.DataFrame:
    new = pd.read_csv(new_identity, sep="\t", dtype=str)
    new.insert(0, "study_id", NEW_STUDY)
    new["raw_entity_id"] = new["entity_id"].map(normalise_entity_id)
    new["entity_id"] = NEW_STUDY + "::" + new["raw_entity_id"]
    pieces = [new[["study_id", "raw_entity_id", "entity_id", "identity_status"]]]
    for study in sorted(candidate_studies):
        path = common_root / study / "identity_partition.tsv.gz"
        block = pd.read_csv(
            path,
            sep="\t",
            usecols=["study_id", "raw_entity_id", "entity_id", "identity_status"],
            dtype=str,
        )
        block["raw_entity_id"] = block["raw_entity_id"].map(normalise_entity_id)
        pieces.append(block)
    output = pd.concat(pieces, ignore_index=True)
    if output.duplicated(["study_id", "raw_entity_id"]).any():
        raise ValueError("duplicate entity in identity partitions")
    return output


def precursor_map(path: Path, required: set[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for metadata, _ in iter_mgf(path):
        entity = spectrum_id(metadata)
        if entity not in required:
            continue
        text = metadata.get("PEPMASS", "").split()[0]
        try:
            value = float(text)
        except ValueError:
            value = math.nan
        output[entity] = value if np.isfinite(value) and value > 0 else math.nan
    return output


def sparse_top1(
    source: sparse.csr_matrix,
    target: sparse.csr_matrix,
    minimum: float,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.full(source.shape[0], -1, dtype=np.int64)
    scores = np.full(source.shape[0], np.nan, dtype=np.float32)
    target_t = target.T.tocsc()
    for start in range(0, source.shape[0], block_size):
        stop = min(start + block_size, source.shape[0])
        similarity = (source[start:stop] @ target_t).toarray()
        selected = np.argmax(similarity, axis=1)
        maximum = similarity[np.arange(stop - start), selected]
        keep = maximum >= minimum
        indices[start:stop][keep] = selected[keep]
        scores[start:stop][keep] = maximum[keep]
    return indices, scores


def freeze_mutual_edges(
    left_ids: np.ndarray,
    left: sparse.csr_matrix,
    right_ids: np.ndarray,
    right: sparse.csr_matrix,
    left_status: dict[str, str],
    right_status: dict[str, str],
    left_mz: dict[str, float],
    right_mz: dict[str, float],
    minimum: float,
    block_size: int,
    ppm_limit: float,
) -> pd.DataFrame:
    rows = []
    for world in WORLDS:
        if world == "all":
            li = np.arange(len(left_ids))
            ri = np.arange(len(right_ids))
        else:
            status = "library_supported" if world == "known" else "unresolved_entity"
            li = np.flatnonzero([left_status.get(str(value)) == status for value in left_ids])
            ri = np.flatnonzero([right_status.get(str(value)) == status for value in right_ids])
        if not len(li) or not len(ri):
            continue
        forward, fs = sparse_top1(left[li], right[ri], minimum, block_size)
        reverse, _ = sparse_top1(right[ri], left[li], minimum, block_size)
        for local_left, local_right in enumerate(forward):
            if local_right < 0 or reverse[local_right] != local_left:
                continue
            left_entity = str(left_ids[li[local_left]])
            right_entity = str(right_ids[ri[local_right]])
            lmz, rmz = left_mz.get(left_entity, np.nan), right_mz.get(right_entity, np.nan)
            ppm = (
                abs(lmz - rmz) / ((lmz + rmz) / 2.0) * 1e6
                if np.isfinite(lmz) and np.isfinite(rmz) and lmz > 0 and rmz > 0
                else np.nan
            )
            rows.append({
                "evidence_world": world,
                "left_raw_entity_id": left_entity,
                "right_raw_entity_id": right_entity,
                "chemical_similarity": float(fs[local_left]),
                "left_precursor_mz": lmz,
                "right_precursor_mz": rmz,
                "precursor_ppm_difference": ppm,
                "edge_track": (
                    "precursor_compatible_identity_hypothesis"
                    if np.isfinite(ppm) and ppm <= ppm_limit
                    else "chemical_relation_only"
                ),
            })
    return pd.DataFrame(rows)


def profile_map(table: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    return {
        (str(study), str(contrast)): block.set_index("raw_entity_id")
        for (study, contrast), block in table.groupby(["study_id", "contrast_id"], sort=False)
    }


def score_candidates(
    edges: pd.DataFrame,
    new_effects: pd.DataFrame,
    chem_effects: pd.DataFrame,
    identity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = profile_map(pd.concat([new_effects, chem_effects], ignore_index=True))
    identity_counts = (
        identity.groupby(["study_id", "identity_status"]).size().to_dict()
    )
    query_registry = new_effects[
        ["contrast_id", "mixture", "query_stratum"]
    ].drop_duplicates().set_index("contrast_id")
    query_contrasts = sorted(new_effects["contrast_id"].unique())
    candidates = (
        chem_effects[["study_id", "contrast_id"]].drop_duplicates()
        .sort_values(["study_id", "contrast_id"])
    )
    score_rows, support_rows = [], []
    for query in query_contrasts:
        query_mixture = int(query_registry.loc[query, "mixture"])
        query_stratum = str(query_registry.loc[query, "query_stratum"])
        query_profile = profiles[(NEW_STUDY, query)]
        for candidate in candidates.itertuples(index=False):
            study, contrast = str(candidate.study_id), str(candidate.contrast_id)
            target_profile = profiles[(study, contrast)]
            pair_edges = edges.loc[edges["right_study_id"].eq(study)]
            for world in WORLDS:
                block = pair_edges.loc[pair_edges["evidence_world"].eq(world)]
                if world == "all":
                    qn = identity.loc[identity["study_id"].eq(NEW_STUDY), "raw_entity_id"].nunique()
                    cn = identity.loc[identity["study_id"].eq(study), "raw_entity_id"].nunique()
                else:
                    status = "library_supported" if world == "known" else "unresolved_entity"
                    qn = int(identity_counts.get((NEW_STUDY, status), 0))
                    cn = int(identity_counts.get((study, status), 0))
                similar = opposite = composition = 0.0
                dual_same = dual_opposite = 0
                identity_same = relation_same = 0
                for edge in block.itertuples(index=False):
                    if edge.left_raw_entity_id not in query_profile.index:
                        continue
                    if edge.right_raw_entity_id not in target_profile.index:
                        continue
                    left = query_profile.loc[edge.left_raw_entity_id]
                    right = target_profile.loc[edge.right_raw_entity_id]
                    product = float(left.response_weight) * float(right.response_weight)
                    weighted = product * float(edge.chemical_similarity)
                    similar += max(weighted, 0.0)
                    opposite += max(-weighted, 0.0)
                    composition += float(edge.chemical_similarity)
                    dual = float(left.qvalue) <= 0.05 and float(right.qvalue) <= 0.05
                    same = np.sign(float(left.beta)) == np.sign(float(right.beta))
                    if dual and same:
                        dual_same += 1
                        if edge.edge_track == "precursor_compatible_identity_hypothesis":
                            identity_same += 1
                        else:
                            relation_same += 1
                        support_rows.append({
                            "query_contrast_id": query,
                            "query_mixture": query_mixture,
                            "query_stratum": query_stratum,
                            "candidate_study_id": study,
                            "candidate_contrast_id": contrast,
                            "evidence_world": world,
                            "left_entity_id": NEW_STUDY + "::" + str(edge.left_raw_entity_id),
                            "right_entity_id": study + "::" + str(edge.right_raw_entity_id),
                            "chemical_similarity": edge.chemical_similarity,
                            "edge_track": edge.edge_track,
                            "left_beta": left.beta,
                            "left_qvalue": left.qvalue,
                            "right_beta": right.beta,
                            "right_qvalue": right.qvalue,
                            "registered_overlap": (
                                TRUE_DRUG_BY_MIXTURE.get(query_mixture) == contrast
                            ),
                        })
                    elif dual:
                        dual_opposite += 1
                symmetric_denominator = 2.0 / (1.0 / qn + 1.0 / cn) if qn and cn else np.nan
                score_rows.append({
                    "query_study_id": NEW_STUDY,
                    "query_contrast_id": query,
                    "query_mixture": query_mixture,
                    "query_stratum": query_stratum,
                    "candidate_study_id": study,
                    "candidate_contrast_id": contrast,
                    "evidence_world": world,
                    "query_universe_entities": qn,
                    "candidate_universe_entities": cn,
                    "mutual_top1_edges": len(block),
                    "similar_response_score": similar / symmetric_denominator if qn and cn else np.nan,
                    "opposite_response_score": opposite / symmetric_denominator if qn and cn else np.nan,
                    "composition_score": composition / symmetric_denominator if qn and cn else np.nan,
                    "dual_fdr_same_direction_edges": dual_same,
                    "dual_fdr_opposite_direction_edges": dual_opposite,
                    "identity_hypothesis_same_direction_edges": identity_same,
                    "chemical_relation_same_direction_edges": relation_same,
                    "registered_overlap": (
                        TRUE_DRUG_BY_MIXTURE.get(query_mixture) == contrast
                    ),
                })
    scores = pd.DataFrame(score_rows)
    scores["signed_response_score"] = (
        scores["similar_response_score"] - scores["opposite_response_score"]
    )
    # A candidate can score highly against every query because it carries a
    # generic culture/time response.  Remove that candidate-specific response
    # propensity across the complete query panel.  This uses no drug name or
    # registered relation and is the treatment-specific analogue of centering
    # out a batch/source main effect.
    candidate_background = scores.groupby(
        [
            "candidate_study_id", "candidate_contrast_id", "evidence_world",
            "query_stratum",
        ]
    )["signed_response_score"].transform("mean")
    scores["response_specificity_score"] = (
        scores["signed_response_score"] - candidate_background
    )
    scores["similar_response_rank"] = scores.groupby(
        ["query_contrast_id", "evidence_world"]
    )["similar_response_score"].rank(ascending=False, method="min")
    scores["response_specificity_rank"] = scores.groupby(
        ["query_contrast_id", "evidence_world"]
    )["response_specificity_score"].rank(ascending=False, method="min")
    scores["dual_fdr_support_rank"] = scores.groupby(
        ["query_contrast_id", "evidence_world"]
    )["dual_fdr_same_direction_edges"].rank(ascending=False, method="min")
    return scores, pd.DataFrame(support_rows)


def evaluation(scores: pd.DataFrame) -> pd.DataFrame:
    scores = scores.loc[scores["query_stratum"].eq("pooled_microbiota")].copy()
    rows = []
    for world, block in scores.groupby("evidence_world", sort=False):
        positive = block.loc[block["registered_overlap"]]
        query_rows = []
        for query in TRUE_OVERLAPS:
            q = block.loc[block["query_contrast_id"].eq(query)]
            truth = q.loc[q["registered_overlap"]]
            query_rows.append({
                "query": query,
                "best_true_rank": float(truth["similar_response_rank"].min()),
                "best_true_specificity_rank": float(
                    truth["response_specificity_rank"].min()
                ),
                "best_true_support_rank": float(truth["dual_fdr_support_rank"].min()),
                "top1_correct": bool(q.loc[q["similar_response_rank"].eq(1), "registered_overlap"].any()),
            })
        rows.append({
            "evidence_world": world,
            "open_candidates_per_query": int(block.groupby("query_contrast_id").size().min()),
            "prespecified_overlap_queries": len(query_rows),
            "prespecified_true_source_relations": len(positive),
            "top1_queries_correct": sum(item["top1_correct"] for item in query_rows),
            "mean_reciprocal_best_true_rank": float(np.mean([1.0 / item["best_true_rank"] for item in query_rows])),
            "median_best_true_rank": float(np.median([item["best_true_rank"] for item in query_rows])),
            "specificity_top1_queries_correct": sum(
                item["best_true_specificity_rank"] == 1 for item in query_rows
            ),
            "specificity_mean_reciprocal_best_true_rank": float(
                np.mean([1.0 / item["best_true_specificity_rank"] for item in query_rows])
            ),
            "specificity_median_best_true_rank": float(
                np.median([item["best_true_specificity_rank"] for item in query_rows])
            ),
            "specificity_true_relations_top3": int(
                positive["response_specificity_rank"].le(3).sum()
            ),
            "specificity_true_relations_top5": int(
                positive["response_specificity_rank"].le(5).sum()
            ),
            "true_relations_top3": int(positive["similar_response_rank"].le(3).sum()),
            "true_relations_top5": int(positive["similar_response_rank"].le(5).sum()),
            "true_dual_fdr_same_direction_edges": int(positive["dual_fdr_same_direction_edges"].sum()),
            "true_identity_hypothesis_edges": int(positive["identity_hypothesis_same_direction_edges"].sum()),
            "true_chemical_relation_edges": int(positive["chemical_relation_same_direction_edges"].sum()),
        })
    return pd.DataFrame(rows)


def community_replication(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate four community-specific retrievals without pooling effects."""
    local = scores.loc[~scores["query_stratum"].eq("pooled_microbiota")].copy()
    groups = [
        "query_mixture", "candidate_study_id", "candidate_contrast_id",
        "evidence_world", "registered_overlap",
    ]
    aggregate = local.groupby(groups, as_index=False).agg(
        communities=("query_stratum", "nunique"),
        mean_response_specificity=("response_specificity_score", "mean"),
        median_response_specificity=("response_specificity_score", "median"),
        positive_specificity_communities=(
            "response_specificity_score", lambda value: int((value > 0).sum())
        ),
        top5_communities=(
            "response_specificity_rank", lambda value: int((value <= 5).sum())
        ),
    )
    aggregate["community_mean_rank"] = aggregate.groupby(
        ["query_mixture", "evidence_world"]
    )["mean_response_specificity"].rank(ascending=False, method="min")

    summaries = []
    eligible = aggregate.loc[aggregate["query_mixture"].isin(TRUE_DRUG_BY_MIXTURE)]
    for world, block in eligible.groupby("evidence_world", sort=False):
        truth = block.loc[block["registered_overlap"]]
        best = truth.groupby("query_mixture")["community_mean_rank"].min()
        summaries.append({
            "evidence_world": world,
            "prespecified_overlap_mixtures": len(TRUE_DRUG_BY_MIXTURE),
            "prespecified_true_source_relations": int(len(truth)),
            "community_mean_top1_mixtures_correct": int((best == 1).sum()),
            "community_mean_mrr": float(np.mean(1.0 / best)),
            "community_mean_median_best_true_rank": float(np.median(best)),
            "true_relations_positive_in_all_four_communities": int(
                truth["positive_specificity_communities"].eq(4).sum()
            ),
            "true_relations_top5_in_at_least_three_communities": int(
                truth["top5_communities"].ge(3).sum()
            ),
        })
    return aggregate, pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new-response-dir", type=Path,
        default=Path("data/validation/msv000095331_directed_drug_microbiome_responses_v2_20261023"),
    )
    parser.add_argument(
        "--new-mgf", type=Path,
        default=Path("data/reference/msv000095331_drug_microbiome_20261023/consensus.mgf"),
    )
    parser.add_argument(
        "--collection", type=Path,
        default=Path("data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023"),
    )
    parser.add_argument(
        "--common-identity-root", type=Path,
        default=Path("data/validation/contrast_atlas_open_v9_common_identity_shards_20261023"),
    )
    parser.add_argument(
        "--candidate-scope", choices=("chemprop", "all"), default="chemprop",
        help="three ChemProp sources or all 15 studies / 51 contrasts in the collection",
    )
    parser.add_argument("--candidate-contract", type=Path, default=CANDIDATE_CONTRACT)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bin-width", type=float, default=0.1)
    parser.add_argument("--minimum-cosine", type=float, default=0.70)
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    # Phase 1: load only entity registries and spectra, then freeze all edges.
    candidate_mgf = candidate_spectra(args.candidate_scope, args.candidate_contract)
    identity = load_identity(
        args.new_response_dir / "source_native_identity.tsv",
        args.common_identity_root,
        set(candidate_mgf),
    )
    spectra_paths = {NEW_STUDY: args.new_mgf, **candidate_mgf}
    spectra: dict[str, tuple[np.ndarray, sparse.csr_matrix]] = {}
    precursors: dict[str, dict[str, float]] = {}
    spectrum_stats = {}
    for study, mgf in spectra_paths.items():
        required = set(identity.loc[identity["study_id"].eq(study), "raw_entity_id"])
        ids, matrix, stats_block = load_study_spectra(mgf, required, args.bin_width)
        spectra[study] = (np.asarray(ids, dtype=str), matrix)
        precursors[study] = precursor_map(mgf, required)
        spectrum_stats[study] = stats_block
    dimensions = max(matrix.shape[1] for _, matrix in spectra.values())
    spectra = {
        study: (ids, pad_columns(matrix, dimensions))
        for study, (ids, matrix) in spectra.items()
    }
    status_maps = {
        study: dict(zip(block["raw_entity_id"], block["identity_status"]))
        for study, block in identity.groupby("study_id", sort=False)
    }
    edge_blocks = []
    left_ids, left_matrix = spectra[NEW_STUDY]
    for study in candidate_mgf:
        right_ids, right_matrix = spectra[study]
        block = freeze_mutual_edges(
            left_ids, left_matrix, right_ids, right_matrix,
            status_maps[NEW_STUDY], status_maps[study],
            precursors[NEW_STUDY], precursors[study],
            args.minimum_cosine, args.block_size, args.precursor_ppm,
        )
        block.insert(0, "left_study_id", NEW_STUDY)
        block.insert(1, "right_study_id", study)
        edge_blocks.append(block)
    edges = pd.concat(edge_blocks, ignore_index=True)

    # Phase 2: only after edge freezing, open effects and registered drug overlap.
    new_effects = load_new_effects(args.new_response_dir / "effects.tsv")
    chem_effects = load_candidate_effects(
        args.collection / "response_entities.tsv.gz", set(candidate_mgf)
    )
    scores, supports = score_candidates(edges, new_effects, chem_effects, identity)
    metrics = evaluation(scores)
    community_scores, community_metrics = community_replication(scores)

    args.output_dir.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    edges.to_csv(
        args.output_dir / "phenotype_blind_mutual_top1_edges.tsv.gz",
        sep="\t", index=False, compression=compression,
    )
    scores.to_csv(
        args.output_dir / "open_candidate_scores.tsv.gz",
        sep="\t", index=False, compression=compression,
    )
    supports.to_csv(
        args.output_dir / "dual_fdr_same_direction_supports.tsv.gz",
        sep="\t", index=False, compression=compression,
    )
    metrics.to_csv(args.output_dir / "retrieval_summary.tsv", sep="\t", index=False)
    community_scores.to_csv(
        args.output_dir / "community_replication_scores.tsv.gz",
        sep="\t", index=False, compression=compression,
    )
    community_metrics.to_csv(
        args.output_dir / "community_replication_summary.tsv", sep="\t", index=False,
    )
    report = {
        "schema": "msv000095331_chemprop_open_response_retrieval_v1",
        "status": STATUS,
        "data_role": "development; MSV000095331 is new to this method but not sealed and not independent-laboratory confirmation",
        "question": (
            "Can a four-mixture microbial response retrieve prespecified overlapping drug "
            "responses from an open multi-study candidate collection?"
        ),
        "parameters": {
            "fragment_bin_width_da": args.bin_width,
            "minimum_sparse_cosine": args.minimum_cosine,
            "neighbors_each_direction": 1,
            "candidate_scope": args.candidate_scope,
            "edge_rule": "mutual top-1, frozen before effects and drug overlap",
            "precursor_compatible_identity_hypothesis_ppm": args.precursor_ppm,
            "response_weight": "beta/sqrt(beta^2+se^2)",
            "response_specificity_score": "signed aligned-response score minus that candidate contrast's mean signed score across all four query mixtures; no drug name or registered overlap enters centering",
            "dual_response_support": "BH q<=0.05 within each contrast at both edge endpoints and equal beta sign",
        },
        "denominators": {
            "new_project_samples": 120,
            "new_project_query_contrasts": int(new_effects["contrast_id"].nunique()),
            "primary_pooled_query_contrasts": 4,
            "community_specific_query_contrasts": 16,
            "open_chemprop_candidate_contrasts": int(chem_effects[["study_id", "contrast_id"]].drop_duplicates().shape[0]),
            "chemprop_candidate_sources": int(chem_effects["study_id"].nunique()),
            "prespecified_overlap_queries": len(TRUE_OVERLAPS),
            "prespecified_true_source_relations": int(
                scores.loc[
                    scores.evidence_world.eq("all")
                    & scores.query_stratum.eq("pooled_microbiota")
                    & scores.registered_overlap
                ].shape[0]
            ),
            "phenotype_blind_edge_rows_all_worlds": int(len(edges)),
            "dual_fdr_same_direction_support_rows": int(len(supports)),
        },
        "retrieval_summary": metrics.to_dict("records"),
        "community_replication_summary": community_metrics.to_dict("records"),
        "spectrum_loading": spectrum_stats,
        "no_match_query": "drug_mix_2_pooled_microbiota has no prespecified overlapping ChemProp drug and is retained as a no-match pressure test; no calibrated abstention threshold is claimed",
        "claim_boundary": [
            "A drug-mixture response cannot identify which co-administered drug caused an individual feature.",
            "Ten-ppm precursor compatibility is an identity hypothesis, not proof against isomers, adduct ambiguity, or in-source fragments.",
            "Mass-incompatible edges are chemical relations; their beta values are not meta-analysed as one molecule.",
            "Unresolved means not supported by the stated library policy, not a unique metabolite.",
            "Known and unresolved worlds use different-sized entity universes; this is not a measurement-matched formal unknown-increment estimate.",
            "Ion/adduct-family collapse is absent, so support counts are spectral-response edges rather than metabolite counts.",
            "The sparse 0.1-Da cosine is a transparent baseline; final algorithm replacement must use the same samples, contrasts, candidates and response scoring.",
            "Response-specificity centering was introduced during development after the uncentered v1 exposed a generic candidate-response bias; its performance is not sealed confirmation.",
            "Four community-specific responses are repeated biological contexts inside one experiment, not four independent studies.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
