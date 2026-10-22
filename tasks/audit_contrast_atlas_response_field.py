#!/usr/bin/env python3
"""Falsify a covariance-corrected spectral response field on consumed studies.

For each query contrast and evidence world, a ridge model learns the mapping
from frozen, phenotype-blind spectrum embeddings to uncertainty-shrunk response
weights.  It is then applied without refitting to every source-disjoint target
study.  The candidate score is the centred correlation between the transferred
response field and the target's observed response weights.

This does not assert entity identity or merge effects across different ions.
The entire fixed score table is written before HFD/obesity family labels are
audited.  Query-field learnability is screened only within the query source;
target response labels never select the ridge penalty or availability gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

WORLDS = ("all", "known", "unresolved")
ALPHA = 10.0
CV_FOLDS = 5
MINIMUM_ENTITIES = 20
NULL_REPLICATES = 31
NULL_SEED = 20261022


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signed_reliability(beta: np.ndarray, se: np.ndarray) -> np.ndarray:
    denominator = np.sqrt(beta * beta + se * se)
    return np.divide(beta, denominator, out=np.zeros_like(beta), where=denominator > 0)


def centred_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float) - float(np.mean(left))
    right = np.asarray(right, dtype=float) - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")


def world_mask(block: pd.DataFrame, world: str) -> np.ndarray:
    if world == "all":
        return np.ones(len(block), dtype=bool)
    if world == "known":
        return block.identity_status.astype(str).eq("library_supported").to_numpy()
    if world == "unresolved":
        return block.identity_status.astype(str).eq("unresolved_entity").to_numpy()
    raise ValueError(world)


def load_embeddings(manifest_path: Path) -> tuple[dict[str, tuple[dict[str, int], np.ndarray]], list[dict[str, object]]]:
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    stores: dict[str, tuple[dict[str, int], np.ndarray]] = {}
    reports: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        path = Path(row.embedding_path)
        archive = np.load(path, allow_pickle=False)
        ids = archive["entity_id"].astype(str)
        vectors = np.asarray(archive["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        if len(ids) != len(set(ids)) or vectors.ndim != 2:
            raise ValueError(f"invalid embedding archive: {path}")
        if not np.isfinite(vectors).all() or float(np.max(np.abs(norms - 1.0))) > 2e-5:
            raise ValueError(f"non-unit or non-finite embeddings: {path}")
        stores[str(row.study_id)] = ({entity: i for i, entity in enumerate(ids)}, vectors)
        reports.append(
            {
                "study_id": str(row.study_id),
                "entities": int(len(ids)),
                "dimensions": int(vectors.shape[1]),
                "embedding_sha256": sha256(path),
            }
        )
    return stores, reports


def prepare_block(
    entities: pd.DataFrame,
    stores: dict[str, tuple[dict[str, int], np.ndarray]],
    study: str,
    contrast: str,
    world: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    block = entities.loc[
        entities.study_id.astype(str).eq(study)
        & entities.contrast_id.astype(str).eq(contrast)
    ].copy()
    mask = world_mask(block, world)
    block = block.loc[mask].reset_index(drop=True)
    lookup, matrix = stores[study]
    missing = sorted(set(block.entity_id.astype(str)) - set(lookup))
    if missing:
        raise ValueError(f"{study}/{contrast} has {len(missing)} entities without embeddings")
    rows = [lookup[entity] for entity in block.entity_id.astype(str)]
    x = matrix[rows]
    y = signed_reliability(block.beta.to_numpy(float), block.se.to_numpy(float))
    strata = block.identity_status.astype(str).to_numpy()
    ids = block.entity_id.astype(str).to_numpy()
    return x, y, strata, ids


def fit_query_field(
    x: np.ndarray,
    y: np.ndarray,
    *,
    study: str,
    contrast: str,
    world: str,
) -> tuple[Ridge | None, dict[str, object]]:
    if len(y) < MINIMUM_ENTITIES or float(np.var(y)) <= 0:
        return None, {
            "query_entities": int(len(y)),
            "query_oof_r2": float("nan"),
            "query_field_identifiable": False,
            "query_unavailability_reason": "insufficient_entities_or_response_variance",
        }
    folds = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=stable_seed(NULL_SEED, study, contrast, world, "query_cv"),
    )
    model = Ridge(alpha=ALPHA, fit_intercept=True, solver="lsqr", tol=1e-6)
    oof = cross_val_predict(model, x, y, cv=folds, method="predict")
    denominator = float(np.sum(np.square(y - np.mean(y))))
    oof_r2 = 1.0 - float(np.sum(np.square(y - oof))) / denominator
    oof_correlation = centred_cosine(oof, y)
    identifiable = bool(oof_r2 > 0 and np.isfinite(oof_correlation))
    model.fit(x, y)
    return model, {
        "query_entities": int(len(y)),
        "query_oof_r2": float(oof_r2),
        "query_oof_correlation": float(oof_correlation),
        "query_field_identifiable": identifiable,
        "query_unavailability_reason": "" if identifiable else "nonpositive_query_oof_r2",
    }


def permute_within_strata(
    values: np.ndarray, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    for stratum in sorted(set(strata.astype(str))):
        indices = np.flatnonzero(strata.astype(str) == stratum)
        if len(indices) > 1:
            result[indices] = result[rng.permutation(indices)]
    return result


def add_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    scores = scores.copy()
    groups = scores.groupby(
        ["query_study_id", "query_contrast_id", "evidence_world"], sort=False
    ).groups
    for score in (
        "signed_similar_score", "signed_opposite_score",
        "response_alignment_strength_score",
    ):
        rank = score.replace("_score", "_absolute_rank")
        scores[rank] = np.nan
        for indices in groups.values():
            values = scores.loc[indices, score]
            present = values.notna()
            scores.loc[values.index[present], rank] = values.loc[present].rank(
                ascending=False, method="min"
            )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection", type=Path,
        default=Path("data/validation/contrast_atlas_open_response_collection_v2_20261019"),
    )
    parser.add_argument(
        "--embedding-manifest", type=Path,
        default=Path(
            "data/validation/contrast_atlas_open_response_embeddings_official_v1_20261019/"
            "embedding_manifest.tsv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/contrast_atlas_response_field_official_v1_20261022"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    entities_path = args.collection / "response_entities.tsv.gz"
    contrasts_path = args.collection / "contrast_manifest.tsv"
    entities = pd.read_csv(entities_path, sep="\t", dtype={"entity_id": str})
    contrasts = pd.read_csv(contrasts_path, sep="\t", dtype=str).reset_index(drop=True)
    stores, embedding_reports = load_embeddings(args.embedding_manifest)
    if set(contrasts.study_id.astype(str)) != set(stores):
        raise ValueError("embedding and contrast study universes differ")

    prepared: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    fields: dict[tuple[str, str, str], tuple[Ridge | None, dict[str, object]]] = {}
    for contrast in contrasts.itertuples(index=False):
        key_base = (str(contrast.study_id), str(contrast.contrast_id))
        for world in WORLDS:
            key = (*key_base, world)
            prepared[key] = prepare_block(entities, stores, *key)
            x, y, _, _ = prepared[key]
            fields[key] = fit_query_field(
                x, y, study=key[0], contrast=key[1], world=world
            )
            print(
                f"[field] {key[0]}/{key[1]}/{world}: "
                f"n={len(y)} oof_r2={fields[key][1]['query_oof_r2']}",
                flush=True,
            )

    score_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    for query in contrasts.itertuples(index=False):
        for candidate in contrasts.itertuples(index=False):
            if str(query.source_family) == str(candidate.source_family):
                continue
            for world in WORLDS:
                query_key = (str(query.study_id), str(query.contrast_id), world)
                candidate_key = (str(candidate.study_id), str(candidate.contrast_id), world)
                model, diagnostics = fields[query_key]
                target_x, target_y, target_strata, _ = prepared[candidate_key]
                prediction = model.predict(target_x) if model is not None else np.full(len(target_y), np.nan)
                raw_score = centred_cosine(prediction, target_y)
                prediction_sd = float(np.std(prediction)) if len(prediction) else float("nan")
                available = bool(
                    diagnostics["query_field_identifiable"]
                    and len(target_y) >= MINIMUM_ENTITIES
                    and np.isfinite(raw_score)
                    and prediction_sd > 0
                )
                base = {
                    "query_study_id": str(query.study_id),
                    "query_source_family": str(query.source_family),
                    "query_contrast_id": str(query.contrast_id),
                    "candidate_study_id": str(candidate.study_id),
                    "candidate_source_family": str(candidate.source_family),
                    "candidate_contrast_id": str(candidate.contrast_id),
                    "evidence_world": world,
                    **diagnostics,
                    "candidate_entities": int(len(target_y)),
                    "transferred_prediction_sd": prediction_sd,
                    "raw_signed_field_score": raw_score,
                    "score_available": available,
                    "unavailability_reason": (
                        "" if available else str(diagnostics["query_unavailability_reason"])
                        or "insufficient_target_entities_or_prediction_variance"
                    ),
                }
                score_rows.append(
                    {
                        **base,
                        "signed_similar_score": raw_score if available else np.nan,
                        "signed_opposite_score": -raw_score if available else np.nan,
                        "response_alignment_strength_score": abs(raw_score) if available else np.nan,
                    }
                )
                for replicate in range(NULL_REPLICATES):
                    rng = np.random.default_rng(
                        stable_seed(
                            NULL_SEED, replicate, *query_key, *candidate_key,
                            "candidate_response_reassignment",
                        )
                    )
                    permuted = permute_within_strata(target_y, target_strata, rng)
                    null_score = centred_cosine(prediction, permuted)
                    null_rows.append(
                        {
                            "null_replicate": replicate,
                            **base,
                            "raw_signed_field_score": null_score,
                            "signed_similar_score": null_score if available else np.nan,
                            "signed_opposite_score": -null_score if available else np.nan,
                            "response_alignment_strength_score": abs(null_score) if available else np.nan,
                        }
                    )

    scores = add_ranks(pd.DataFrame(score_rows))
    null_scores = pd.concat(
        [
            add_ranks(block)
            for _, block in pd.DataFrame(null_rows).groupby("null_replicate", sort=True)
        ],
        ignore_index=True,
    )
    args.output_dir.mkdir(parents=True)
    score_path = args.output_dir / "fixed_candidate_scores.tsv.gz"
    null_path = args.output_dir / "stratified_response_reassignment_scores.tsv.gz"
    diagnostics_path = args.output_dir / "query_field_diagnostics.tsv"
    compression = {"method": "gzip", "mtime": 0}
    scores.to_csv(score_path, sep="\t", index=False, compression=compression)
    null_scores.to_csv(null_path, sep="\t", index=False, compression=compression)
    scores[
        [
            "query_study_id", "query_contrast_id", "evidence_world",
            "query_entities", "query_oof_r2", "query_oof_correlation",
            "query_unavailability_reason", "query_field_identifiable",
        ]
    ].drop_duplicates().to_csv(diagnostics_path, sep="\t", index=False)

    diagnostics = pd.read_csv(diagnostics_path, sep="\t")
    report = {
        "schema": "contrast_atlas_covariance_corrected_response_field_v1",
        "status": "DEVELOPMENT_CONSUMED_FALSIFICATION_RESULT",
        "method": {
            "response_weight": "beta/sqrt(beta^2+se^2)",
            "field": "ridge mapping from frozen unit spectrum embedding to response weight",
            "ridge_alpha": ALPHA,
            "ridge_alpha_selection": "fixed before execution; no target outcome used",
            "query_identifiability": "five-fold out-of-fold R2 strictly greater than zero",
            "candidate_score": "centred cosine of transferred prediction and candidate response weight",
            "null": "candidate response weights reassigned within identity stratum",
            "null_replicates": NULL_REPLICATES,
            "minimum_entities": MINIMUM_ENTITIES,
        },
        "denominators": {
            "source_families": int(contrasts.source_family.nunique()),
            "study_level_contrasts": int(len(contrasts)),
            "query_worlds": int(len(diagnostics)),
            "identifiable_query_worlds": int(diagnostics.query_field_identifiable.astype(bool).sum()),
            "fixed_candidate_rows": int(len(scores)),
            "available_candidate_rows": int(scores.score_available.astype(bool).sum()),
            "null_candidate_rows": int(len(null_scores)),
        },
        "input_sha256": {
            str(entities_path): sha256(entities_path),
            str(contrasts_path): sha256(contrasts_path),
            str(args.embedding_manifest): sha256(args.embedding_manifest),
        },
        "embedding_reports": embedding_reports,
        "claim_boundary": [
            "All sources and response labels are development/consumed.",
            "This relation-layer field does not identify molecules or merge beta/SE across ions.",
            "Unresolved means not library-supported, not a unique metabolite.",
            "Feature-level cross-validation may share ion families and is an identifiability screen, not external evidence.",
            "HFD and obesity family results are audited only after the complete fixed score table exists.",
            "A positive source-conditioned control cannot by itself establish open biological discovery or unknown increment.",
        ],
        "output_sha256": {
            score_path.name: sha256(score_path),
            null_path.name: sha256(null_path),
            diagnostics_path.name: sha256(diagnostics_path),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
