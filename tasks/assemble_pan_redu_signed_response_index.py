#!/usr/bin/env python3
"""Assemble panel effects and frozen spectral embeddings into response sketches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def integrate(embedding: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= 0:
        return np.zeros(embedding.shape[1], dtype=np.float32)
    return (weights @ embedding / total).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-jobs", required=True, type=Path)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--effects-root", required=True, type=Path)
    parser.add_argument("--embeddings-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--identity-table", type=Path)
    parser.add_argument("--minimum-entities", type=int, default=12)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    jobs = pd.read_csv(args.panel_jobs, sep="\t", dtype=str)
    topology = pd.read_csv(args.topology, sep="\t", dtype=str)
    topology = topology.loc[topology["topology_status"].str.startswith("PASS")].copy()
    identity = None
    if args.identity_table:
        identity = pd.read_csv(args.identity_table, sep="\t", dtype=str)
        needed = {"panel_id", "anchor_id", "identity_status"}
        if missing := needed - set(identity):
            raise ValueError(f"identity table lacks {sorted(missing)}")
        if identity.duplicated(["panel_id", "anchor_id"]).any():
            raise ValueError("identity table has duplicate panel/anchor rows")
        allowed = {"confirmed_known", "library_supported", "unresolved_entity"}
        if not set(identity["identity_status"]).issubset(allowed):
            raise ValueError("identity table contains unsupported states")

    vectors = {f"{direction}_{world}": [] for direction in ("positive", "negative") for world in ("all", "known", "unresolved")}
    manifest: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    dimension: int | None = None
    for job in jobs.sort_values("job_index", key=lambda s: s.astype(int)).itertuples(index=False):
        panel_id = str(job.panel_id)
        effect_dir = args.effects_root / panel_id
        embedding_dir = args.embeddings_root / panel_id
        effect_report_path = effect_dir / "report.json"
        embedding_report_path = embedding_dir / "report.json"
        if not effect_report_path.is_file() or not embedding_report_path.is_file():
            raise FileNotFoundError(f"incomplete panel products: {panel_id}")
        effect_report = json.loads(effect_report_path.read_text(encoding="utf-8"))
        embedding_report = json.loads(embedding_report_path.read_text(encoding="utf-8"))
        if effect_report.get("status") != "development_effect_objects_not_cross_study_evidence":
            raise RuntimeError(f"invalid effect status: {panel_id}")
        if embedding_report.get("status") != "phenotype_blind_frozen_spectral_embeddings":
            raise RuntimeError(f"invalid embedding status: {panel_id}")
        effects = pd.read_csv(effect_dir / "contrast_effects.tsv.gz", sep="\t")
        with np.load(embedding_dir / "anchor_embeddings.npz", allow_pickle=False) as packed:
            anchor_ids = np.asarray(packed["anchor_id"], dtype=str)
            embedding = np.asarray(packed["embeddings"], dtype=np.float32)
        if len(anchor_ids) != len(set(anchor_ids)) or embedding.shape[0] != len(anchor_ids):
            raise RuntimeError(f"invalid anchor embedding map: {panel_id}")
        if dimension is None:
            dimension = int(embedding.shape[1])
        if embedding.ndim != 2 or embedding.shape[1] != dimension:
            raise RuntimeError("embedding dimensions differ across panels")
        embedding_frame = pd.DataFrame({"anchor_id": anchor_ids, "embedding_row": np.arange(len(anchor_ids))})
        effects = effects.merge(embedding_frame, on="anchor_id", how="left", validate="many_to_one")
        if effects["embedding_row"].isna().any():
            raise RuntimeError(f"effect anchors lack embeddings: {panel_id}")
        panel_topology = topology.loc[topology["panel_id"].eq(panel_id)]
        if identity is not None:
            effects = effects.merge(
                identity.loc[identity["panel_id"].eq(panel_id), ["anchor_id", "identity_status"]],
                on="anchor_id", how="left", validate="many_to_one",
            )
            if effects["identity_status"].isna().any():
                raise RuntimeError(f"identity partition incomplete: {panel_id}")
        else:
            effects["identity_status"] = "unpartitioned"

        for definition in panel_topology.itertuples(index=False):
            block = effects.loc[effects["contrast_id"].astype(str).eq(str(definition.contrast_id))].copy()
            block = block.loc[block["passes_effect_qc"].astype(str).str.lower().isin(["true", "1"])]
            beta = block["beta_b_minus_a"].to_numpy(float)
            se = block["se"].to_numpy(float)
            prevalence = block["unit_prevalence"].to_numpy(float)
            response = beta / np.sqrt(beta * beta + se * se)
            signed_weight = response * np.sqrt(np.clip(prevalence, 0, 1))
            emb = embedding[block["embedding_row"].to_numpy(int)]
            states = block["identity_status"].astype(str).to_numpy()
            masks = {
                "all": np.ones(len(block), dtype=bool),
                "known": np.isin(states, ["confirmed_known", "library_supported"]),
                "unresolved": states == "unresolved_entity",
            }
            row: dict[str, object] = {
                "sketch_row": len(manifest), "study_id": str(definition.dataset_accession),
                "source_family": str(definition.preliminary_source_family),
                "panel_id": panel_id, "contrast_id": str(definition.contrast_id),
                "comparison_stratum": str(definition.comparison_stratum),
                "contrast_field": str(definition.contrast_field),
                "level_a": str(definition.level_a), "level_b": str(definition.level_b),
            }
            for world, mask in masks.items():
                row[f"n_entities_{world}"] = int(mask.sum())
                pos = np.maximum(signed_weight[mask], 0)
                neg = np.maximum(-signed_weight[mask], 0)
                vectors[f"positive_{world}"].append(integrate(emb[mask], pos))
                vectors[f"negative_{world}"].append(integrate(emb[mask], neg))
            row["answerable_all"] = bool(row["n_entities_all"] >= args.minimum_entities)
            row["identity_partition_available"] = identity is not None
            manifest.append(row)
        sources.append({
            "panel_id": panel_id, "effect_report_sha256": sha256(effect_report_path),
            "embedding_report_sha256": sha256(embedding_report_path),
        })

    if not manifest or dimension is None:
        raise RuntimeError("no response sketches assembled")
    packed = {name: np.stack(items).astype(np.float32) for name, items in vectors.items()}
    args.output_dir.mkdir(parents=True)
    pd.DataFrame(manifest).to_csv(args.output_dir / "contrast_manifest.tsv", sep="\t", index=False)
    np.savez_compressed(args.output_dir / "contrast_sketches.npz", **packed)
    report = {
        "schema": "pan_redu_signed_response_index_v1",
        "status": "development_pretruth_response_index",
        "denominators": {
            "panels": len(sources), "contrasts": len(manifest),
            "source_families": int(pd.DataFrame(manifest)["source_family"].nunique()),
            "answerable_all_contrasts": int(sum(bool(row["answerable_all"]) for row in manifest)),
            "embedding_dimensions": dimension,
        },
        "identity_partition_available": identity is not None,
        "minimum_entities": args.minimum_entities,
        "leakage_status": "study effects and phenotype_blind_embeddings_joined_without_cross_study_truth",
        "sources": sources,
        "claim_boundary": [
            "This is a pre-truth index, not evidence of a correct biological association.",
            "Known/unresolved claims require an explicit frozen identity table; missing identity is never called unresolved.",
            "Pan-ReDU ontology labels remain candidate retrieval endpoints, not assumed biological truth.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
