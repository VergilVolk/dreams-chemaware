#!/usr/bin/env python3
"""Truth-blind count-matched unresolved response retrieval for factorial HFD.

For each panel and resample, unresolved ion-family representatives are sampled
to the number of strict-known representatives.  The same sampled entity set is
used for all five contrasts, preventing response-specific entity selection.
The registered Hi-MYC relationship is never read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from assemble_factorial_hfd_response_index import CONTRAST_ORDER, integrate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def response_vectors(block: pd.DataFrame, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta = block.beta.to_numpy(float)
    se = block.se_hc3.to_numpy(float)
    prevalence = block.unit_prevalence.to_numpy(float)
    signed = beta / np.sqrt(beta * beta + se * se)
    signed *= np.sqrt(np.clip(prevalence, 0.0, 1.0))
    emb = embedding[block.embedding_row.to_numpy(int)]
    positive = np.maximum(signed, 0)
    negative = np.maximum(-signed, 0)
    response_mass = float(positive.sum() + negative.sum())
    return (
        integrate(emb, positive, response_mass),
        integrate(emb, negative, response_mass),
        integrate(emb, np.sqrt(np.clip(prevalence, 0.0, 1.0))),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-jobs", required=True, type=Path)
    parser.add_argument("--panel-registry", required=True, type=Path)
    parser.add_argument("--effects-root", required=True, type=Path)
    parser.add_argument("--embeddings-root", required=True, type=Path)
    parser.add_argument("--families-root", required=True, type=Path)
    parser.add_argument("--identity-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resamples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--minimum-entities", type=int, default=12)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.resamples < 1:
        raise ValueError("resamples must be positive")

    jobs = pd.read_csv(args.panel_jobs, sep="\t", dtype=str)
    registry = pd.read_csv(args.panel_registry, sep="\t", dtype=str).set_index("panel_id", verify_integrity=True)
    identity = pd.read_csv(args.identity_table, sep="\t", dtype=str)
    if identity.duplicated(["panel_id", "anchor_id"]).any():
        raise ValueError("identity table contains duplicate panel/anchor rows")
    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=args.resamples, dtype=np.int64)

    # Cache complete phenotype-blind panel inputs once.
    panels: dict[str, dict[str, object]] = {}
    dimension: int | None = None
    source_hashes: list[dict[str, str]] = []
    for job in jobs.sort_values("job_index", key=lambda value: value.astype(int)).itertuples(index=False):
        panel_id = str(job.panel_id)
        effect_path = args.effects_root / panel_id / "factorial_contrast_effects.tsv.gz"
        embedding_path = args.embeddings_root / panel_id / "anchor_embeddings.npz"
        family_path = args.families_root / panel_id / "ion_family_membership.tsv"
        effects = pd.read_csv(effect_path, sep="\t", dtype={"anchor_id": str})
        with np.load(embedding_path, allow_pickle=False) as packed:
            anchor_ids = np.asarray(packed["anchor_id"], dtype=str)
            embedding = np.asarray(packed["embeddings"], dtype=np.float32)
        if len(anchor_ids) != len(set(anchor_ids)) or embedding.shape[0] != len(anchor_ids):
            raise RuntimeError(f"invalid embedding map for {panel_id}")
        if dimension is None:
            dimension = int(embedding.shape[1])
        if embedding.ndim != 2 or embedding.shape[1] != dimension:
            raise RuntimeError("embedding dimensions differ across panels")
        effects = effects.merge(
            pd.DataFrame({"anchor_id": anchor_ids, "embedding_row": np.arange(len(anchor_ids))}),
            on="anchor_id", how="left", validate="many_to_one",
        )
        families = pd.read_csv(family_path, sep="\t", dtype={"anchor_id": str})
        effects = effects.merge(
            families[["anchor_id", "representative"]],
            on="anchor_id", how="left", validate="many_to_one",
        )
        panel_identity = identity.loc[
            identity.panel_id.eq(panel_id), ["anchor_id", "identity_status"]
        ]
        effects = effects.merge(
            panel_identity, on="anchor_id", how="left", validate="many_to_one",
        )
        if effects[["embedding_row", "representative", "identity_status"]].isna().any().any():
            raise RuntimeError(f"incomplete panel joins for {panel_id}")
        effects = effects.loc[
            effects.representative.astype(str).str.lower().isin(["true", "1"])
        ].copy()
        representative_identity = effects[["anchor_id", "identity_status"]].drop_duplicates()
        known_ids = representative_identity.loc[
            representative_identity.identity_status.isin(["confirmed_known", "library_supported"]),
            "anchor_id",
        ].to_numpy(str)
        unresolved_ids = representative_identity.loc[
            representative_identity.identity_status.eq("unresolved_entity"), "anchor_id",
        ].to_numpy(str)
        if len(unresolved_ids) < len(known_ids):
            raise RuntimeError(
                f"panel {panel_id} cannot support known-count matched unresolved sampling: "
                f"known={len(known_ids)} unresolved={len(unresolved_ids)}"
            )
        panels[panel_id] = {
            "study_id": str(registry.loc[panel_id, "dataset_accession"]),
            "effects": effects,
            "embedding": embedding,
            "known_ids": known_ids,
            "unresolved_ids": unresolved_ids,
        }
        source_hashes.append({
            "panel_id": panel_id,
            "effects_sha256": sha256(effect_path),
            "embeddings_sha256": sha256(embedding_path),
            "families_sha256": sha256(family_path),
        })

    assert dimension is not None
    score_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    for resample, seed in enumerate(seeds):
        prng = np.random.default_rng(int(seed))
        panel_vectors: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
        for panel_id, panel in panels.items():
            known_ids = panel["known_ids"]
            unresolved_ids = panel["unresolved_ids"]
            selected = np.sort(
                prng.choice(unresolved_ids, size=len(known_ids), replace=False)
            ) if len(known_ids) else np.asarray([], dtype=str)
            sample_rows.append({
                "resample": resample, "seed": int(seed), "panel_id": panel_id,
                "study_id": panel["study_id"], "known_representatives": len(known_ids),
                "unresolved_representatives": len(unresolved_ids),
                "sampled_unresolved_representatives": len(selected),
                "sampled_anchor_sha256": hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest(),
            })
            effects = panel["effects"]
            embedding = panel["embedding"]
            for contrast in CONTRAST_ORDER:
                block = effects.loc[
                    effects.contrast_type.eq(contrast)
                    & effects.anchor_id.isin(selected)
                    & effects.passes_effect_qc.astype(str).str.lower().isin(["true", "1"])
                ].copy()
                positive, negative, chemistry = response_vectors(block, embedding)
                panel_vectors[(panel["study_id"], panel_id, contrast)] = (
                    positive, negative, chemistry, len(block)
                )

        study_vectors: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
        for study_id in sorted({str(value["study_id"]) for value in panels.values()}):
            study_panels = sorted(
                panel_id for panel_id, value in panels.items() if value["study_id"] == study_id
            )
            for contrast in CONTRAST_ORDER:
                pieces = [panel_vectors[(study_id, panel_id, contrast)] for panel_id in study_panels]
                study_vectors[(study_id, contrast)] = (
                    np.mean(np.stack([piece[0] for piece in pieces]), axis=0),
                    np.mean(np.stack([piece[1] for piece in pieces]), axis=0),
                    np.mean(np.stack([piece[2] for piece in pieces]), axis=0),
                    int(sum(piece[3] for piece in pieces)),
                )
                object_rows.append({
                    "resample": resample, "study_id": study_id, "contrast_id": contrast,
                    "sampled_effect_qc_entities": study_vectors[(study_id, contrast)][3],
                })

        raw_rows: list[dict[str, object]] = []
        for (query_study, query_contrast), query in sorted(study_vectors.items()):
            for (target_study, target_contrast), target in sorted(study_vectors.items()):
                if query_study == target_study:
                    continue
                answerable = query[3] >= args.minimum_entities and target[3] >= args.minimum_entities
                raw_rows.append({
                    "resample": resample,
                    "query_study_id": query_study, "query_contrast_id": query_contrast,
                    "target_study_id": target_study, "target_contrast_id": target_contrast,
                    "query_entities": query[3], "target_entities": target[3],
                    "answerable": answerable,
                    "signed_response_score": float((query[0] - query[1]) @ (target[0] - target[1])) if answerable else np.nan,
                    "magnitude_only_score": float((query[0] + query[1]) @ (target[0] + target[1])) if answerable else np.nan,
                    "chemistry_only_score": float(query[2] @ target[2]) if answerable else np.nan,
                })
        scores = pd.DataFrame(raw_rows)
        group = ["resample", "query_study_id", "query_contrast_id"]
        for score in ("signed_response_score", "magnitude_only_score", "chemistry_only_score"):
            scores[f"rank_{score}"] = scores.groupby(group)[score].rank(ascending=False, method="min")
        score_rows.extend(scores.to_dict("records"))

    args.output_dir.mkdir(parents=True)
    scores = pd.DataFrame(score_rows)
    samples = pd.DataFrame(sample_rows)
    objects = pd.DataFrame(object_rows)
    scores.to_csv(args.output_dir / "count_matched_retrieval_scores.tsv.gz", sep="\t", index=False, compression="gzip")
    samples.to_csv(args.output_dir / "resample_entity_ledger.tsv.gz", sep="\t", index=False, compression="gzip")
    objects.to_csv(args.output_dir / "resample_response_objects.tsv.gz", sep="\t", index=False, compression="gzip")
    report = {
        "schema": "factorial_hfd_count_matched_unresolved_v1",
        "status": "PRETRUTH_COUNT_MATCHED_UNRESOLVED_SCORES_FROZEN",
        "data_role": "development_consumed",
        "denominators": {
            "studies": int(objects.study_id.nunique()),
            "panels": len(panels),
            "response_objects_per_resample": int(objects.groupby("resample").size().iloc[0]),
            "resamples": args.resamples,
            "retrieval_rows": len(scores),
        },
        "sampling_unit": "ion-family representative within panel; one sampled set reused across all five contrasts",
        "truth_opened": False,
        "leakage_status": "sampling and all-query scoring used identity state, effects and embeddings but never read the registered cross-study relationship",
        "provenance": {
            "panel_jobs_sha256": sha256(args.panel_jobs),
            "panel_registry_sha256": sha256(args.panel_registry),
            "identity_table_sha256": sha256(args.identity_table),
            "sources": source_hashes,
            "scores_sha256": sha256(args.output_dir / "count_matched_retrieval_scores.tsv.gz"),
            "resample_ledger_sha256": sha256(args.output_dir / "resample_entity_ledger.tsv.gz"),
            "implementation_sha256": sha256(Path(__file__).resolve()),
        },
        "claim_boundary": "Count matching tests coverage dependence; it is not a substitute for the separately calibrated response null or an external validation.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
