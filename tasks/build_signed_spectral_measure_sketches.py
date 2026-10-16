#!/usr/bin/env python3
"""Build scalable, signed study-level spectral-response sketches.

The scientific object is a biological contrast, not a spectrum match.  Within
each study/contrast, every QC-qualified LC-MS/MS-linked entity receives an
effect and uncertainty estimated from biological samples.  Positive and
negative effects are then integrated separately in a frozen spectral embedding
space.  This is a finite-dimensional kernel mean embedding of the signed
response measure described in UNBIASED_SIGNED_SPECTRAL_RESPONSE_METHOD_20261014.

This program deliberately does not read cross-study truth or condition-family
labels.  It also does not claim that DDA spectrum counts are metabolite
abundances: input values must be MS1-linked quantitative values or another
explicitly declared quantitative assay.  Embeddings and entity identities are
fixed before biological effects are estimated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY_STATES = {"confirmed_known", "library_supported", "unresolved_entity"}
REQUIRED_COLUMNS = {
    "study_id", "source_family", "contrast_id", "sample_id", "group",
    "entity_id", "value", "embedding_row", "identity_status", "qc_pass",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(values: pd.Series, name: str) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError(f"{name} contains values other than true/false/1/0")
    return normalized.map(mapping).astype(bool)


def validate_input(frame: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"input table lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["qc_pass"] = parse_bool(frame["qc_pass"], "qc_pass")
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    frame["embedding_row"] = pd.to_numeric(frame["embedding_row"], errors="raise").astype(int)
    if (~np.isfinite(frame["value"].to_numpy(float))).any() or (frame["value"] < 0).any():
        raise ValueError("quantitative values must be finite and non-negative")
    if not set(frame["identity_status"].astype(str)).issubset(IDENTITY_STATES):
        raise ValueError("unrecognized identity_status")
    if (frame["embedding_row"] < 0).any() or (frame["embedding_row"] >= len(embeddings)).any():
        raise ValueError("embedding_row lies outside the embedding matrix")
    key = ["study_id", "contrast_id", "sample_id", "entity_id"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate entity observation within one sample contrast")
    # One entity must have one phenotype-blind spectral representation and one
    # identity state inside a study.  Otherwise biological values could choose
    # which spectrum represents the entity.
    entity_key = ["study_id", "contrast_id", "entity_id"]
    for column in ("embedding_row", "identity_status", "source_family"):
        if (frame.groupby(entity_key, sort=False)[column].nunique() > 1).any():
            raise ValueError(f"entity maps to multiple {column} values")
    if (~np.isfinite(embeddings)).any():
        raise ValueError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if (norms <= 0).any():
        raise ValueError("embedding matrix contains a zero vector")
    return frame.loc[frame["qc_pass"]].copy()


def transform_panel(raw: pd.DataFrame, transform: str) -> pd.DataFrame:
    """Return samples x entities on an explicitly chosen quantitative scale."""
    matrix = raw.pivot(index="sample_id", columns="entity_id", values="value")
    if matrix.isna().any().any():
        # Absence from a long-form table is not silently interpreted as zero.
        # Callers must provide the complete analysed sample/entity rectangle.
        raise ValueError("panel is not a complete sample-by-entity rectangle")
    values = matrix.to_numpy(dtype=np.float64)
    if transform == "none":
        transformed = values
    elif transform == "log2_pseudocount_median_center":
        positive = np.where(values > 0, values, np.nan)
        minimum = np.nanmin(positive, axis=0)
        if np.isnan(minimum).any():
            raise ValueError("an entity is zero in every sample")
        pseudocount = minimum / 2.0
        transformed = np.log2(values + pseudocount[None, :])
        transformed -= np.median(transformed, axis=1, keepdims=True)
    else:
        raise ValueError(transform)
    return pd.DataFrame(transformed, index=matrix.index, columns=matrix.columns)


def welch_effect(a: np.ndarray, b: np.ndarray, se_floor: float) -> tuple[np.ndarray, np.ndarray]:
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each group needs at least two biological samples")
    beta = b.mean(axis=0) - a.mean(axis=0)
    se = np.sqrt(a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b))
    se = np.maximum(se, se_floor)
    return beta, se


def normalized_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).astype(np.float32)


def response_sketch(
    embedding: np.ndarray,
    beta: np.ndarray,
    se: np.ndarray,
    prevalence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Bounded response prevents one unstable high-fold-change entity from
    # dominating a study. Prevalence is an observable measurement-quality
    # weight, not a structure or outcome label.
    response = beta / np.sqrt(beta * beta + se * se)
    quality = np.sqrt(np.clip(prevalence, 0.0, 1.0))
    signed_weight = response * quality
    positive_weight = np.maximum(signed_weight, 0.0)
    negative_weight = np.maximum(-signed_weight, 0.0)

    def integrate(weight: np.ndarray) -> np.ndarray:
        total = float(weight.sum())
        if total <= 0:
            return np.zeros(embedding.shape[1], dtype=np.float32)
        return (weight @ embedding / total).astype(np.float32)

    return integrate(positive_weight), integrate(negative_weight), signed_weight


def build(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    transform: str,
    min_group_n: int,
    min_prevalence: float,
    se_floor: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict[str, object]]]:
    normalized = normalized_embeddings(embeddings)
    effect_rows: list[dict[str, object]] = []
    vectors: dict[str, list[np.ndarray]] = {
        "positive_all": [], "negative_all": [],
        "positive_known": [], "negative_known": [],
        "positive_unresolved": [], "negative_unresolved": [],
    }
    manifest: list[dict[str, object]] = []

    group_columns = ["study_id", "source_family", "contrast_id"]
    for key, block in frame.groupby(group_columns, sort=True):
        study_id, source_family, contrast_id = map(str, key)
        sample_group = block[["sample_id", "group"]].drop_duplicates()
        if sample_group.groupby("sample_id")["group"].nunique().max() != 1:
            raise ValueError(f"sample belongs to multiple groups in {study_id}/{contrast_id}")
        groups = sorted(sample_group["group"].astype(str).unique(), key=str.casefold)
        if len(groups) != 2:
            raise ValueError(f"contrast {study_id}/{contrast_id} must have exactly two groups")
        group_a, group_b = groups
        counts = sample_group.groupby("group")["sample_id"].nunique()
        if min(int(counts.get(group_a, 0)), int(counts.get(group_b, 0))) < min_group_n:
            continue

        panel = transform_panel(block, transform)
        ids_a = sample_group.loc[sample_group["group"].astype(str).eq(group_a), "sample_id"]
        ids_b = sample_group.loc[sample_group["group"].astype(str).eq(group_b), "sample_id"]
        a = panel.loc[ids_a].to_numpy(float)
        b = panel.loc[ids_b].to_numpy(float)
        beta, se = welch_effect(a, b, se_floor)
        original = block.pivot(index="sample_id", columns="entity_id", values="value").loc[panel.index]
        prevalence = (original.to_numpy(float) > 0).mean(axis=0)
        eligible = prevalence >= min_prevalence

        entity_meta = (
            block[["entity_id", "embedding_row", "identity_status"]]
            .drop_duplicates("entity_id").set_index("entity_id").loc[panel.columns]
        )
        entity_embedding = normalized[entity_meta["embedding_row"].to_numpy(int)]
        identity = entity_meta["identity_status"].astype(str).to_numpy()
        worlds = {
            "all": eligible,
            "known": eligible & np.isin(identity, ["confirmed_known", "library_supported"]),
            "unresolved": eligible & (identity == "unresolved_entity"),
        }
        world_counts: dict[str, int] = {}
        signed_all = np.zeros(len(panel.columns), dtype=float)
        for world, mask in worlds.items():
            world_counts[world] = int(mask.sum())
            pos, neg, signed = response_sketch(
                entity_embedding[mask], beta[mask], se[mask], prevalence[mask]
            )
            vectors[f"positive_{world}"].append(pos)
            vectors[f"negative_{world}"].append(neg)
            if world == "all":
                signed_all[mask] = signed

        for index, entity_id in enumerate(panel.columns.astype(str)):
            effect_rows.append({
                "study_id": study_id,
                "source_family": source_family,
                "contrast_id": contrast_id,
                "group_a": group_a,
                "group_b": group_b,
                "entity_id": entity_id,
                "identity_status": identity[index],
                "embedding_row": int(entity_meta.iloc[index]["embedding_row"]),
                "beta_b_minus_a": float(beta[index]),
                "se": float(se[index]),
                "bounded_response_weight": float(signed_all[index]),
                "prevalence": float(prevalence[index]),
                "eligible": bool(eligible[index]),
            })
        manifest.append({
            "sketch_row": len(manifest),
            "study_id": study_id,
            "source_family": source_family,
            "contrast_id": contrast_id,
            "group_a": group_a,
            "group_b": group_b,
            "n_group_a": int(counts[group_a]),
            "n_group_b": int(counts[group_b]),
            "n_entities_all": world_counts["all"],
            "n_entities_known": world_counts["known"],
            "n_entities_unresolved": world_counts["unresolved"],
        })

    packed = {
        name: np.stack(items).astype(np.float32)
        if items else np.empty((0, embeddings.shape[1]), dtype=np.float32)
        for name, items in vectors.items()
    }
    return pd.DataFrame(effect_rows), packed, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--transform", choices=("none", "log2_pseudocount_median_center"),
        default="log2_pseudocount_median_center",
    )
    parser.add_argument("--min-group-n", type=int, default=5)
    parser.add_argument("--min-prevalence", type=float, default=0.2)
    parser.add_argument("--se-floor", type=float, default=1e-6)
    args = parser.parse_args()
    if not 0 <= args.min_prevalence <= 1:
        raise ValueError("--min-prevalence must lie in [0,1]")

    observations = pd.read_csv(args.observations, sep="\t")
    embeddings = np.load(args.embeddings, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError("embedding matrix must be two-dimensional")
    observations = validate_input(observations, embeddings)
    effects, sketches, manifest = build(
        observations, embeddings, args.transform, args.min_group_n,
        args.min_prevalence, args.se_floor,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(args.output_dir / "entity_effects.tsv", sep="\t", index=False)
    np.savez_compressed(args.output_dir / "contrast_sketches.npz", **sketches)
    pd.DataFrame(manifest).to_csv(args.output_dir / "contrast_manifest.tsv", sep="\t", index=False)
    report = {
        "schema": "signed_spectral_measure_sketch_v1",
        "scientific_role": "study_level_response_index_not_structure_annotation",
        "inputs": {
            "observations": str(args.observations.resolve()),
            "observations_sha256": file_sha256(args.observations),
            "embeddings": str(args.embeddings.resolve()),
            "embeddings_sha256": file_sha256(args.embeddings),
        },
        "parameters": {
            "transform": args.transform,
            "min_group_n": args.min_group_n,
            "min_prevalence": args.min_prevalence,
            "se_floor": args.se_floor,
        },
        "denominators": {
            "input_rows": int(len(observations)),
            "input_studies": int(observations["study_id"].nunique()),
            "built_contrasts": len(manifest),
            "eligible_entity_effects": int(effects["eligible"].sum()) if len(effects) else 0,
            "embedding_dimensions": int(embeddings.shape[1]),
        },
        "leakage_status": "no_cross_study_truth_or_contrast_family_read",
        "claim_boundary": [
            "Input values must be MS1-linked quantitative observations; DDA scan counts alone are not abundance.",
            "Embedding similarity is chemical relation evidence, not molecular identity.",
            "The output is a response index; biological association requires held-out cross-study truth and controls.",
            "Known/unresolved increment requires fixed-universe matched-known and shuffled-MS2 comparisons.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["denominators"], indent=2))


if __name__ == "__main__":
    main()
