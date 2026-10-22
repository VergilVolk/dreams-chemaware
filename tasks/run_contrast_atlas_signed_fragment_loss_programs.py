#!/usr/bin/env python3
"""Truth-blind signed fragment/neutral-loss response programs.

The previous centred kernel compresses every response into one signed vector.
Here up- and down-response programs remain separate, and spectra are represented
either by fragments alone or by an equal-norm fragment plus neutral-loss view.
The complete open candidate table is frozen before registered relations are read.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from build_contrast_atlas_open_response_collection import load_contract
from run_contrast_atlas_sparse_fragment_measure_baseline import (
    WORLDS,
    iter_mgf,
    normalise_entity_id,
    pad_columns,
    sha256,
    sparse_spectrum,
    spectrum_id,
    unweighted_centroid,
)
from run_contrast_atlas_centered_response_kernel import (
    l2_normalise,
    stable_seed,
    stratified_permutation,
)


REPRESENTATIONS = ("fragment", "fragment_loss")
SCORES = (
    "similar_score", "opposite_score", "alignment_strength_score",
    "signed_margin_score", "composition_score",
)


def null_top1_summary(
    observed_top1: int, null_block: pd.DataFrame
) -> dict[str, int | float | str]:
    """Summarise whether Top-1 relations survive response reassignment.

    This is a development sensitivity statistic, not a randomised-experiment
    p-value. The plus-one fraction prevents a zero estimate with few nulls.
    """
    counts = (
        null_block.groupby("null_replicate", sort=True)
        .alignment_strength_absolute_rank
        .apply(lambda values: int(values.eq(1).sum()))
        .to_numpy(dtype=int)
    )
    if not len(counts):
        return {
            "observed_alignment_top1": int(observed_top1),
            "null_replicates": 0,
            "null_top1_min": math.nan,
            "null_top1_median": math.nan,
            "null_top1_max": math.nan,
            "null_top1_mean": math.nan,
            "plus_one_fraction_null_at_least_observed": math.nan,
            "specificity_gate": "NOT_EVALUABLE",
        }
    fraction = float((1 + np.sum(counts >= observed_top1)) / (1 + len(counts)))
    return {
        "observed_alignment_top1": int(observed_top1),
        "null_replicates": int(len(counts)),
        "null_top1_min": int(counts.min()),
        "null_top1_median": float(np.median(counts)),
        "null_top1_max": int(counts.max()),
        "null_top1_mean": float(counts.mean()),
        "plus_one_fraction_null_at_least_observed": fraction,
        "specificity_gate": (
            "PASS_DEVELOPMENT_SENSITIVITY"
            if observed_top1 > 0 and fraction <= 0.05
            else "FAIL_RESPONSE_REASSIGNMENT_SPECIFICITY"
        ),
    }


def loss_spectrum(
    precursor_mz: float, peaks: list[tuple[float, float]], bin_width: float
) -> tuple[list[int], list[float]]:
    binned: dict[int, float] = defaultdict(float)
    if not math.isfinite(precursor_mz) or precursor_mz <= 0:
        return [], []
    for mz, intensity in peaks:
        loss = precursor_mz - mz
        if not (math.isfinite(loss) and math.isfinite(intensity)):
            continue
        if loss < 0 or intensity <= 0:
            continue
        index = int(math.floor(loss / bin_width + 1e-10))
        binned[index] += math.sqrt(intensity)
    if not binned:
        return [], []
    norm = math.sqrt(sum(value * value for value in binned.values()))
    indices = sorted(binned)
    return indices, [binned[index] / norm for index in indices]


def _matrix(
    row_indices: list[int], columns: list[int], values: list[float], rows: int, max_bin: int
) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (np.asarray(row_indices, dtype=int), np.asarray(columns, dtype=int)),
        ),
        shape=(rows, max_bin + 1 if max_bin >= 0 else 0),
        dtype=np.float32,
    )


def load_dual_spectra(
    path: Path, required: set[str], bin_width: float
) -> tuple[list[str], sparse.csr_matrix, sparse.csr_matrix, dict[str, int]]:
    rows: list[str] = []
    seen: set[str] = set()
    fragment_row: list[int] = []
    fragment_col: list[int] = []
    fragment_val: list[float] = []
    loss_row: list[int] = []
    loss_col: list[int] = []
    loss_val: list[float] = []
    fragment_max = -1
    loss_max = -1
    total = 0
    with_loss = 0
    for metadata, peaks in iter_mgf(path):
        total += 1
        entity = normalise_entity_id(spectrum_id(metadata))
        if entity not in required:
            continue
        if entity in seen:
            raise ValueError(f"duplicate representative spectrum {entity} in {path}")
        seen.add(entity)
        f_index, f_value = sparse_spectrum(peaks, bin_width)
        if not f_index:
            continue
        try:
            precursor = float(str(metadata.get("PEPMASS", "")).split()[0])
        except (ValueError, IndexError):
            precursor = float("nan")
        l_index, l_value = loss_spectrum(precursor, peaks, bin_width)
        row = len(rows)
        rows.append(entity)
        fragment_row.extend([row] * len(f_index))
        fragment_col.extend(f_index)
        fragment_val.extend(f_value)
        fragment_max = max(fragment_max, f_index[-1])
        if l_index:
            with_loss += 1
            loss_row.extend([row] * len(l_index))
            loss_col.extend(l_index)
            loss_val.extend(l_value)
            loss_max = max(loss_max, l_index[-1])
    missing = required - set(rows)
    if missing:
        raise ValueError(f"{len(missing)} required entities lack nonempty spectra in {path}")
    fragments = _matrix(
        fragment_row, fragment_col, fragment_val, len(rows), fragment_max
    )
    losses = _matrix(loss_row, loss_col, loss_val, len(rows), loss_max)
    return rows, fragments, losses, {
        "mgf_spectra_total": total,
        "response_spectra_nonempty": len(rows),
        "response_spectra_with_neutral_loss": with_loss,
        "fragment_nonzero_bins": int(fragments.nnz),
        "loss_nonzero_bins": int(losses.nnz),
        "maximum_fragment_bin": fragment_max,
        "maximum_loss_bin": loss_max,
    }


def row_l2_normalise(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    return sparse.diags(inverse.astype(np.float32)) @ matrix


def signed_program(
    matrix: sparse.csr_matrix, weights: np.ndarray
) -> dict[str, object]:
    centred = np.asarray(weights, dtype=float) - float(np.mean(weights))
    sum_squares = float(np.dot(centred, centred))
    effective_n = (
        float(np.square(np.abs(centred).sum()) / sum_squares)
        if sum_squares > 0 else 0.0
    )
    positive = np.maximum(centred, 0.0)
    negative = np.maximum(-centred, 0.0)
    up, up_norm = l2_normalise(sparse.csr_matrix(positive.astype(np.float32)[None, :] @ matrix))
    down, down_norm = l2_normalise(sparse.csr_matrix(negative.astype(np.float32)[None, :] @ matrix))
    return {
        "n_entities": int(matrix.shape[0]),
        "effective_n": effective_n,
        "up": up,
        "down": down,
        "up_raw_norm": up_norm,
        "down_raw_norm": down_norm,
        "composition": unweighted_centroid(matrix),
    }


def sparse_dot(left: sparse.csr_matrix, right: sparse.csr_matrix) -> float:
    return float(left.multiply(right).sum())


def program_scores(left: dict[str, object], right: dict[str, object]) -> dict[str, float]:
    similar = 0.5 * (
        sparse_dot(left["up"], right["up"])
        + sparse_dot(left["down"], right["down"])
    )
    opposite = 0.5 * (
        sparse_dot(left["up"], right["down"])
        + sparse_dot(left["down"], right["up"])
    )
    return {
        "similar_score": similar,
        "opposite_score": opposite,
        "alignment_strength_score": max(similar, opposite),
        "signed_margin_score": similar - opposite,
        "composition_score": sparse_dot(left["composition"], right["composition"]),
    }


def score_candidates(
    contrasts: pd.DataFrame,
    sketches: dict[tuple[int, str, str], dict[str, object]],
    minimum_entities: int,
    minimum_effective_n: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for qi, query in contrasts.iterrows():
        for ti, target in contrasts.iterrows():
            if str(query.source_family) == str(target.source_family):
                continue
            for world in WORLDS:
                for representation in REPRESENTATIONS:
                    left = sketches[(qi, world, representation)]
                    right = sketches[(ti, world, representation)]
                    available = (
                        int(left["n_entities"]) >= minimum_entities
                        and int(right["n_entities"]) >= minimum_entities
                        and float(left["effective_n"]) >= minimum_effective_n
                        and float(right["effective_n"]) >= minimum_effective_n
                        and float(left["up_raw_norm"]) > 0
                        and float(left["down_raw_norm"]) > 0
                        and float(right["up_raw_norm"]) > 0
                        and float(right["down_raw_norm"]) > 0
                    )
                    record: dict[str, object] = {
                        "query_study_id": str(query.study_id),
                        "query_source_family": str(query.source_family),
                        "query_contrast_id": str(query.contrast_id),
                        "candidate_study_id": str(target.study_id),
                        "candidate_source_family": str(target.source_family),
                        "candidate_contrast_id": str(target.contrast_id),
                        "evidence_world": world,
                        "representation": representation,
                        "query_entities": int(left["n_entities"]),
                        "candidate_entities": int(right["n_entities"]),
                        "query_effective_n": float(left["effective_n"]),
                        "candidate_effective_n": float(right["effective_n"]),
                        "score_available": bool(available),
                        "unavailability_reason": "" if available else "insufficient_signed_program",
                    }
                    record.update(
                        program_scores(left, right) if available
                        else {score: np.nan for score in SCORES}
                    )
                    records.append(record)
    scores = pd.DataFrame(records)
    groups = scores.groupby(
        ["query_study_id", "query_contrast_id", "evidence_world", "representation"],
        sort=False,
    )
    for score in SCORES:
        rank = score.replace("_score", "_absolute_rank")
        scores[rank] = np.nan
        for indices in groups.groups.values():
            values = scores.loc[indices, score]
            present = values.notna()
            scores.loc[values.index[present], rank] = values.loc[present].rank(
                ascending=False, method="min"
            )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bin-width-da", type=float, default=0.1)
    parser.add_argument("--minimum-entities", type=int, default=10)
    parser.add_argument("--minimum-effective-n", type=float, default=5.0)
    parser.add_argument("--null-replicates", type=int, default=31)
    parser.add_argument("--null-seed", type=int, default=20261022)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    root = Path(__file__).resolve().parents[1]
    contract, lineage = load_contract(args.contract.resolve(), root)
    entities = pd.read_csv(args.collection / "response_entities.tsv.gz", sep="\t", dtype={"raw_entity_id": str})
    entities["raw_entity_id"] = entities.raw_entity_id.map(normalise_entity_id)
    contrasts = pd.read_csv(args.collection / "contrast_manifest.tsv", sep="\t").reset_index(drop=True)
    source_contract = {str(item["study_id"]): item for item in contract["sources"]}

    source_matrices: dict[str, tuple[list[str], sparse.csr_matrix, sparse.csr_matrix]] = {}
    source_reports: list[dict[str, object]] = []
    fragment_dim = 0
    loss_dim = 0
    for study_id in contrasts.study_id.astype(str).drop_duplicates():
        required = set(entities.loc[entities.study_id.astype(str).eq(study_id), "raw_entity_id"])
        mgf_path = root / source_contract[study_id]["mgf_path"]
        ids, fragment, loss, details = load_dual_spectra(mgf_path, required, args.bin_width_da)
        source_matrices[study_id] = (ids, fragment, loss)
        fragment_dim = max(fragment_dim, fragment.shape[1])
        loss_dim = max(loss_dim, loss.shape[1])
        source_reports.append({"study_id": study_id, "mgf_sha256": sha256(mgf_path), **details})

    prepared: dict[tuple[int, str, str], tuple[sparse.csr_matrix, np.ndarray, np.ndarray]] = {}
    sketches: dict[tuple[int, str, str], dict[str, object]] = {}
    for ci, contrast in contrasts.iterrows():
        study_id = str(contrast.study_id)
        ids, raw_fragment, raw_loss = source_matrices[study_id]
        fragment = pad_columns(raw_fragment, fragment_dim)
        loss = pad_columns(raw_loss, loss_dim)
        dual = row_l2_normalise(sparse.hstack([fragment, loss], format="csr", dtype=np.float32))
        representations = {"fragment": fragment, "fragment_loss": dual}
        lookup = {entity: index for index, entity in enumerate(ids)}
        block = entities.loc[
            entities.study_id.astype(str).eq(study_id)
            & entities.contrast_id.astype(str).eq(str(contrast.contrast_id))
        ].copy()
        block["spectrum_row"] = block.raw_entity_id.map(lookup)
        if block.spectrum_row.isna().any() or block.raw_entity_id.duplicated().any():
            raise ValueError(f"invalid response-to-spectrum join for {study_id}/{contrast.contrast_id}")
        beta = block.beta.to_numpy(float)
        se = block.se.to_numpy(float)
        denominator = np.sqrt(beta * beta + se * se)
        weights = np.divide(beta, denominator, out=np.zeros_like(beta), where=denominator > 0)
        for world in WORLDS:
            if world == "all":
                mask = np.ones(len(block), dtype=bool)
            elif world == "known":
                mask = block.identity_status.astype(str).eq("library_supported").to_numpy()
            else:
                mask = block.identity_status.astype(str).eq("unresolved_entity").to_numpy()
            strata = block.loc[mask, "identity_status"].astype(str).to_numpy()
            for representation, matrix in representations.items():
                selected = matrix[block.loc[mask, "spectrum_row"].to_numpy(int)]
                prepared[(ci, world, representation)] = (selected, weights[mask], strata)
                sketches[(ci, world, representation)] = signed_program(selected, weights[mask])

    scores = score_candidates(
        contrasts, sketches, args.minimum_entities, args.minimum_effective_n
    )
    args.output_dir.mkdir(parents=True)
    score_path = args.output_dir / "fixed_candidate_scores.tsv.gz"
    scores.to_csv(
        score_path, sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    score_hash_before_truth = sha256(score_path)

    null_frames: list[pd.DataFrame] = []
    for replicate in range(args.null_replicates):
        null_sketches: dict[tuple[int, str, str], dict[str, object]] = {}
        for key, (matrix, weights, strata) in prepared.items():
            ci, world, representation = key
            seed = stable_seed(args.null_seed, replicate, ci, world + representation)
            permuted = stratified_permutation(weights, strata, np.random.default_rng(seed))
            null_sketches[key] = signed_program(matrix, permuted)
        frame = score_candidates(
            contrasts, null_sketches, args.minimum_entities, args.minimum_effective_n
        )
        frame.insert(0, "null_replicate", replicate)
        null_frames.append(frame)
    null_scores = pd.concat(null_frames, ignore_index=True)
    null_path = args.output_dir / "response_reassignment_scores.tsv.gz"
    null_scores.to_csv(
        null_path, sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    relations = pd.read_csv(args.collection / "registered_relations.tsv", sep="\t")
    keys = relations.rename(columns={
        "left_study_id": "query_study_id",
        "left_contrast_id": "query_contrast_id",
        "right_study_id": "candidate_study_id",
        "right_contrast_id": "candidate_contrast_id",
    })[["query_study_id", "query_contrast_id", "candidate_study_id", "candidate_contrast_id"]]
    audit = scores.merge(
        keys,
        on=["query_study_id", "query_contrast_id", "candidate_study_id", "candidate_contrast_id"],
        how="inner",
        validate="many_to_one",
    )
    audit.to_csv(args.output_dir / "registered_relation_ranks.tsv", sep="\t", index=False)
    null_audit = null_scores.merge(
        keys,
        on=["query_study_id", "query_contrast_id", "candidate_study_id", "candidate_contrast_id"],
        how="inner",
        validate="many_to_one",
    )
    sensitivity: list[dict[str, object]] = []
    identity = [
        "query_study_id", "query_contrast_id", "candidate_study_id",
        "candidate_contrast_id", "evidence_world", "representation",
    ]
    for observed in audit.to_dict("records"):
        block = null_audit
        for column in identity:
            block = block.loc[block[column].astype(str).eq(str(observed[column]))]
        null_rank = block.alignment_strength_absolute_rank.dropna().to_numpy(float)
        rank = float(observed["alignment_strength_absolute_rank"])
        sensitivity.append({
            **{column: observed[column] for column in identity},
            "observed_alignment_rank": rank,
            "null_replicates": int(len(null_rank)),
            "fraction_null_rank_as_good_or_better": (
                float(np.mean(null_rank <= rank)) if len(null_rank) else np.nan
            ),
        })
    pd.DataFrame(sensitivity).to_csv(
        args.output_dir / "registered_relation_reassignment_sensitivity.tsv",
        sep="\t", index=False,
    )

    top1: dict[str, dict[str, dict[str, int]]] = {}
    null_top1: dict[str, dict[str, dict[str, int | float | str]]] = {}
    for representation in REPRESENTATIONS:
        top1[representation] = {}
        null_top1[representation] = {}
        for world in WORLDS:
            block = audit.loc[
                audit.representation.eq(representation) & audit.evidence_world.eq(world)
            ]
            available = block.loc[block.score_available.astype(bool)]
            top1[representation][world] = {
                "registered_relations": int(len(block)),
                "score_available": int(len(available)),
                "similar_top1": int(available.similar_absolute_rank.eq(1).sum()),
                "opposite_top1": int(available.opposite_absolute_rank.eq(1).sum()),
                "alignment_top1": int(available.alignment_strength_absolute_rank.eq(1).sum()),
                "composition_top1": int(available.composition_absolute_rank.eq(1).sum()),
            }
            null_block = null_audit.loc[
                null_audit.representation.eq(representation)
                & null_audit.evidence_world.eq(world)
                & null_audit.score_available.astype(bool)
            ]
            null_top1[representation][world] = null_top1_summary(
                top1[representation][world]["alignment_top1"], null_block
            )

    primary_gate = null_top1["fragment_loss"]["unresolved"]
    report = {
        "schema": "contrast_atlas_signed_fragment_loss_programs_v1",
        "status": "DEVELOPMENT_CONSUMED_TRUTH_BLIND_RESULT",
        "method": {
            "response_weight": "beta/sqrt(beta^2+se^2), centred within contrast and evidence world",
            "programs": "positive and negative centred weights aggregated separately; no beta/SE merging across entities",
            "fragment_view": "sqrt-intensity 0.1-Da absolute fragment bins, spectrum L2-normalised",
            "fragment_loss_view": "equal-norm concatenation of absolute fragment and precursor-minus-fragment bins",
            "similar": "mean(up-up, down-down)",
            "opposite": "mean(up-down, down-up)",
            "parameter_selection": "fixed before registered relations were opened",
        },
        "denominators": {
            "source_families": int(contrasts.source_family.nunique()),
            "study_level_contrasts": int(len(contrasts)),
            "fixed_candidate_rows": int(len(scores)),
            "score_available_rows": int(scores.score_available.sum()),
            "registered_directed_relations": int(len(relations)),
            "fragment_dimensions": fragment_dim,
            "neutral_loss_dimensions": loss_dim,
        },
        "registered_relation_descriptive_top1": top1,
        "registered_relation_response_reassignment_top1": null_top1,
        "primary_development_gate": {
            "representation": "fragment_loss",
            "evidence_world": "unresolved",
            **primary_gate,
            "decision": (
                "FAIL_UNKNOWN_RESPONSE_SPECIFICITY"
                if primary_gate["specificity_gate"]
                != "PASS_DEVELOPMENT_SENSITIVITY"
                else "DEVELOPMENT_SENSITIVITY_ONLY_NOT_UNKNOWN_INCREMENT"
            ),
        },
        "truth_boundary": {
            "score_sha256_before_registered_relations": score_hash_before_truth,
            "signed_direction_truth": "absent for all registered relations",
            "unregistered_pairs": "unadjudicated, never negative",
            "unknown_increment": "requires a prespecified correct-answer and abstention gate; descriptive rank alone is insufficient",
        },
        "source_reports": source_reports,
        "input_sha256": {
            str(args.contract): sha256(args.contract),
            **{f"contract_lineage::{path.relative_to(root)}": sha256(path) for path in lineage},
            str(args.collection / "response_entities.tsv.gz"): sha256(args.collection / "response_entities.tsv.gz"),
            str(args.collection / "contrast_manifest.tsv"): sha256(args.collection / "contrast_manifest.tsv"),
            str(args.collection / "registered_relations.tsv"): sha256(args.collection / "registered_relations.tsv"),
        },
        "outputs_sha256": {
            "fixed_candidate_scores.tsv.gz": sha256(score_path),
            "response_reassignment_scores.tsv.gz": sha256(null_path),
        },
        "claim_boundary": [
            "All sources and registered relations are development/consumed.",
            "This relation-layer score never asserts that related spectra are the same entity.",
            "Unresolved means no strict direct library-supported control hit, not a unique metabolite.",
            "A rank change is not a biological discovery without independent relation truth.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"top1": top1, "denominators": report["denominators"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
