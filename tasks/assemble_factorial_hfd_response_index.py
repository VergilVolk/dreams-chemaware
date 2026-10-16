#!/usr/bin/env python3
"""Build panel- and study-level signed response measures for the HFD wave.

Each study-level object is an equal-panel mixture, preventing an assay with
more recurrent anchors from receiving more biological weight.  The operation
uses effect estimates and phenotype-blind spectral coordinates, but never
reads the predeclared cross-study relationship.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORLDS = ("all", "known", "unresolved")
CONTRAST_ORDER = (
    "diet_reference", "diet_modified", "genotype_control",
    "genotype_hfd", "genotype_by_hfd",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def integrate(embedding: np.ndarray, weights: np.ndarray, denominator: float | None = None) -> np.ndarray:
    total = float(weights.sum()) if denominator is None else float(denominator)
    if total <= 0:
        return np.zeros(embedding.shape[1], dtype=np.float32)
    return (weights @ embedding / total).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-jobs", required=True, type=Path)
    parser.add_argument("--panel-registry", required=True, type=Path)
    parser.add_argument("--effects-root", required=True, type=Path)
    parser.add_argument("--embeddings-root", required=True, type=Path)
    parser.add_argument("--families-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--identity-table", type=Path)
    parser.add_argument("--minimum-entities", type=int, default=12)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    jobs = pd.read_csv(args.panel_jobs, sep="\t", dtype=str)
    registry = pd.read_csv(args.panel_registry, sep="\t", dtype=str)
    registry = registry.set_index("panel_id", verify_integrity=True)
    identity = None
    if args.identity_table:
        identity = pd.read_csv(args.identity_table, sep="\t", dtype=str)
        required_identity = {"panel_id", "anchor_id", "identity_status"}
        if missing := required_identity - set(identity):
            raise ValueError(f"identity table lacks {sorted(missing)}")
        if identity.duplicated(["panel_id", "anchor_id"]).any():
            raise ValueError("identity table contains duplicate panel/anchor rows")
        allowed = {"confirmed_known", "library_supported", "unresolved_entity"}
        if not set(identity.identity_status).issubset(allowed):
            raise ValueError("unsupported identity state")

    panel_rows: list[dict[str, object]] = []
    panel_vectors: dict[str, list[np.ndarray]] = {
        f"{kind}_{world}": []
        for kind in ("positive", "negative", "chemistry") for world in WORLDS
    }
    sources: list[dict[str, str]] = []
    dimension: int | None = None
    for job in jobs.sort_values("job_index", key=lambda value: value.astype(int)).itertuples(index=False):
        panel_id = str(job.panel_id)
        effect_dir = args.effects_root / panel_id
        embedding_dir = args.embeddings_root / panel_id
        effect_report_path = effect_dir / "report.json"
        embedding_report_path = embedding_dir / "report.json"
        effect_report = json.loads(effect_report_path.read_text(encoding="utf-8"))
        embedding_report = json.loads(embedding_report_path.read_text(encoding="utf-8"))
        if effect_report.get("status") != "development_factorial_effect_objects_not_cross_study_evidence":
            raise RuntimeError(f"invalid effect status for {panel_id}")
        if embedding_report.get("status") != "phenotype_blind_frozen_spectral_embeddings":
            raise RuntimeError(f"invalid embedding status for {panel_id}")
        effects = pd.read_csv(effect_dir / "factorial_contrast_effects.tsv.gz", sep="\t")
        with np.load(embedding_dir / "anchor_embeddings.npz", allow_pickle=False) as packed:
            anchor_ids = np.asarray(packed["anchor_id"], dtype=str)
            embedding = np.asarray(packed["embeddings"], dtype=np.float32)
        if len(anchor_ids) != len(set(anchor_ids)) or embedding.shape[0] != len(anchor_ids):
            raise RuntimeError(f"invalid embedding map for {panel_id}")
        if dimension is None:
            dimension = int(embedding.shape[1])
        if embedding.ndim != 2 or embedding.shape[1] != dimension:
            raise RuntimeError("embedding dimensions differ across panels")
        lookup = pd.DataFrame({"anchor_id": anchor_ids, "embedding_row": np.arange(len(anchor_ids))})
        effects = effects.merge(lookup, on="anchor_id", how="left", validate="many_to_one")
        if effects.embedding_row.isna().any():
            raise RuntimeError(f"effect anchor lacks embedding for {panel_id}")
        if args.families_root:
            family_path = args.families_root / panel_id / "ion_family_membership.tsv"
            families = pd.read_csv(family_path, sep="\t", dtype={"anchor_id": str})
            needed_family = {"panel_id", "anchor_id", "family_id", "representative"}
            if missing := needed_family - set(families):
                raise ValueError(f"family table lacks {sorted(missing)} for {panel_id}")
            if families.anchor_id.duplicated().any() or not families.panel_id.astype(str).eq(panel_id).all():
                raise ValueError(f"invalid family map for {panel_id}")
            effects = effects.merge(
                families[["anchor_id", "family_id", "representative"]],
                on="anchor_id", how="left", validate="many_to_one",
            )
            if effects.family_id.isna().any():
                raise RuntimeError(f"family map incomplete for {panel_id}")
            representative = effects.representative.astype(str).str.lower().isin(["true", "1"])
            effects = effects.loc[representative].copy()
        else:
            effects["family_id"] = effects["anchor_id"]
        if identity is None:
            effects["identity_status"] = "unpartitioned"
        else:
            panel_identity = identity.loc[identity.panel_id.eq(panel_id), ["anchor_id", "identity_status"]]
            effects = effects.merge(panel_identity, on="anchor_id", how="left", validate="many_to_one")
            if effects.identity_status.isna().any():
                raise RuntimeError(f"identity partition incomplete for {panel_id}")

        panel_meta = registry.loc[panel_id]
        observed_contrasts = tuple(effects.contrast_type.drop_duplicates())
        if set(observed_contrasts) != set(CONTRAST_ORDER):
            raise RuntimeError(f"incomplete contrast set for {panel_id}: {observed_contrasts}")
        for contrast_type in CONTRAST_ORDER:
            block = effects.loc[effects.contrast_type.eq(contrast_type)].copy()
            block = block.loc[block.passes_effect_qc.astype(str).str.lower().isin(["true", "1"])]
            beta = block.beta.to_numpy(float)
            se = block.se_hc3.to_numpy(float)
            prevalence = block.unit_prevalence.to_numpy(float)
            bounded = beta / np.sqrt(beta * beta + se * se)
            signed_weight = bounded * np.sqrt(np.clip(prevalence, 0.0, 1.0))
            emb = embedding[block.embedding_row.to_numpy(int)]
            states = block.identity_status.astype(str).to_numpy()
            masks = {
                "all": np.ones(len(block), dtype=bool),
                "known": np.isin(states, ["confirmed_known", "library_supported"]),
                "unresolved": states == "unresolved_entity",
            }
            row: dict[str, object] = {
                "sketch_row": len(panel_rows),
                "study_id": str(panel_meta.dataset_accession),
                "source_family": str(panel_meta.source_family),
                "panel_id": panel_id,
                "contrast_id": contrast_type,
                "contrast_type": contrast_type,
            }
            for world, mask in masks.items():
                row[f"n_entities_{world}"] = int(mask.sum())
                positive = np.maximum(signed_weight[mask], 0)
                negative = np.maximum(-signed_weight[mask], 0)
                response_mass = float(positive.sum() + negative.sum())
                panel_vectors[f"positive_{world}"].append(integrate(emb[mask], positive, response_mass))
                panel_vectors[f"negative_{world}"].append(integrate(emb[mask], negative, response_mass))
                panel_vectors[f"chemistry_{world}"].append(
                    integrate(emb[mask], np.sqrt(np.clip(prevalence[mask], 0.0, 1.0)))
                )
            row["answerable_all"] = bool(row["n_entities_all"] >= args.minimum_entities)
            row["identity_partition_available"] = identity is not None
            panel_rows.append(row)
        sources.append({
            "panel_id": panel_id,
            "effect_report_sha256": sha256(effect_report_path),
            "embedding_report_sha256": sha256(embedding_report_path),
        })

    if not panel_rows or dimension is None:
        raise RuntimeError("no response objects assembled")
    panel_manifest = pd.DataFrame(panel_rows)
    panel_packed = {name: np.stack(items).astype(np.float32) for name, items in panel_vectors.items()}

    study_rows: list[dict[str, object]] = []
    study_vectors: dict[str, list[np.ndarray]] = {name: [] for name in panel_packed}
    for (study_id, contrast_type), block in panel_manifest.groupby(["study_id", "contrast_type"], sort=True):
        indices = block.sketch_row.to_numpy(int)
        row = {
            "sketch_row": len(study_rows),
            "study_id": study_id,
            "source_family": block.source_family.iloc[0],
            "panel_id": "STUDY_EQUAL_PANEL_MIXTURE",
            "contrast_id": contrast_type,
            "contrast_type": contrast_type,
            "assay_panels": len(block),
            "identity_partition_available": identity is not None,
        }
        for world in WORLDS:
            row[f"n_entities_{world}"] = int(block[f"n_entities_{world}"].sum())
            for kind in ("positive", "negative", "chemistry"):
                name = f"{kind}_{world}"
                study_vectors[name].append(panel_packed[name][indices].mean(axis=0).astype(np.float32))
        row["answerable_all"] = bool(
            block.answerable_all.all()
            and (np.linalg.norm(study_vectors["positive_all"][-1]) > 0
                 or np.linalg.norm(study_vectors["negative_all"][-1]) > 0)
        )
        study_rows.append(row)

    args.output_dir.mkdir(parents=True)
    panel_manifest.to_csv(args.output_dir / "panel_contrast_manifest.tsv", sep="\t", index=False)
    np.savez_compressed(args.output_dir / "panel_contrast_sketches.npz", **panel_packed)
    study_manifest = pd.DataFrame(study_rows)
    study_manifest.to_csv(args.output_dir / "study_contrast_manifest.tsv", sep="\t", index=False)
    np.savez_compressed(
        args.output_dir / "study_contrast_sketches.npz",
        **{name: np.stack(items).astype(np.float32) for name, items in study_vectors.items()},
    )
    report = {
        "schema": "factorial_hfd_signed_response_index_v1",
        "status": "development_pretruth_factorial_response_index",
        "denominators": {
            "studies": int(study_manifest.study_id.nunique()),
            "panels": int(panel_manifest.panel_id.nunique()),
            "panel_contrast_objects": len(panel_manifest),
            "study_contrast_objects": len(study_manifest),
            "embedding_dimensions": dimension,
        },
        "study_aggregation": "equal weight per analytical panel within study and contrast",
        "minimum_entities_per_panel": args.minimum_entities,
        "identity_partition_available": identity is not None,
        "ion_family_partition_available": args.families_root is not None,
        "leakage_status": "effect_and_spectral_objects_joined_without_opening_cross_study_relationship",
        "sources": sources,
        "claim_boundary": [
            "The index is pre-truth and is not evidence that Hi-MYC interactions retrieve one another.",
            "MTBLS135 and MTBLS3316 are separate experiments from a shared model lineage, not independent-laboratory replication.",
            "MTBLS2085 NQO1-by-HFD direction is not assumed to match or oppose Hi-MYC.",
            "Unknown-only claims require a complete frozen identity partition.",
            "Primary metabolite-level claims require the frozen conservative ion-family partition.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
