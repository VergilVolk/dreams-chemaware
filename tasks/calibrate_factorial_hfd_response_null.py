#!/usr/bin/env python3
"""Truth-blind within-panel response-to-chemistry permutation null.

The marginal effect distribution, spectral coordinates, prevalence, panel
composition, and study design are preserved.  Only which chemical entity owns
which response is broken.  A maximum-candidate null is stored for every query,
so later adjudication cannot select a favorable target without multiplicity
control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-manifest", required=True, type=Path)
    parser.add_argument("--panel-registry", required=True, type=Path)
    parser.add_argument("--effects-root", required=True, type=Path)
    parser.add_argument("--embeddings-root", required=True, type=Path)
    parser.add_argument("--families-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.permutations < 99:
        raise ValueError("at least 99 permutations are required")

    manifest = pd.read_csv(args.study_manifest, sep="\t")
    if manifest.duplicated(["study_id", "contrast_id"]).any():
        raise ValueError("study manifest contains duplicate response objects")
    registry = pd.read_csv(args.panel_registry, sep="\t", dtype=str).set_index("panel_id", verify_integrity=True)
    rng = np.random.default_rng(args.seed)
    object_vectors: dict[tuple[str, str], list[np.ndarray]] = {}
    panel_counts: dict[tuple[str, str], int] = {}
    dimension: int | None = None

    for panel_id, panel in registry.iterrows():
        effects = pd.read_csv(
            args.effects_root / panel_id / "factorial_contrast_effects.tsv.gz", sep="\t"
        )
        with np.load(args.embeddings_root / panel_id / "anchor_embeddings.npz", allow_pickle=False) as packed:
            anchor_id = np.asarray(packed["anchor_id"], dtype=str)
            embedding = np.asarray(packed["embeddings"], dtype=np.float32)
        if len(anchor_id) != len(set(anchor_id)) or embedding.shape[0] != len(anchor_id):
            raise RuntimeError(f"invalid embedding map for {panel_id}")
        if dimension is None:
            dimension = int(embedding.shape[1])
        if embedding.ndim != 2 or embedding.shape[1] != dimension:
            raise RuntimeError("embedding dimension mismatch")
        lookup = pd.DataFrame({"anchor_id": anchor_id, "embedding_row": np.arange(len(anchor_id))})
        effects = effects.merge(lookup, on="anchor_id", how="left", validate="many_to_one")
        if effects.embedding_row.isna().any():
            raise RuntimeError(f"missing embedding for effect anchor in {panel_id}")
        if args.families_root:
            families = pd.read_csv(
                args.families_root / panel_id / "ion_family_membership.tsv", sep="\t", dtype={"anchor_id": str}
            )
            if families.anchor_id.duplicated().any():
                raise ValueError(f"duplicate family anchor for {panel_id}")
            effects = effects.merge(
                families[["anchor_id", "representative"]], on="anchor_id", how="left", validate="many_to_one"
            )
            if effects.representative.isna().any():
                raise RuntimeError(f"incomplete family map for {panel_id}")
            effects = effects.loc[
                effects.representative.astype(str).str.lower().isin(["true", "1"])
            ].copy()
        for contrast_id, block in effects.groupby("contrast_type", sort=True):
            block = block.loc[block.passes_effect_qc.astype(str).str.lower().isin(["true", "1"])]
            beta = block.beta.to_numpy(float)
            se = block.se_hc3.to_numpy(float)
            prevalence = block.unit_prevalence.to_numpy(float)
            weight = beta / np.sqrt(beta * beta + se * se)
            weight *= np.sqrt(np.clip(prevalence, 0.0, 1.0))
            denominator = float(np.abs(weight).sum())
            if denominator <= 0 or len(weight) == 0:
                vectors = np.zeros((args.permutations, int(dimension)), dtype=np.float32)
            else:
                shuffled = np.stack([rng.permutation(weight) for _ in range(args.permutations)])
                vectors = (shuffled @ embedding[block.embedding_row.to_numpy(int)] / denominator).astype(np.float32)
            key = (str(panel.dataset_accession), str(contrast_id))
            object_vectors.setdefault(key, []).append(vectors)
            panel_counts[key] = panel_counts.get(key, 0) + 1

    study_null: list[np.ndarray] = []
    for row in manifest.itertuples(index=False):
        key = (str(row.study_id), str(row.contrast_id))
        pieces = object_vectors.get(key)
        if not pieces:
            raise RuntimeError(f"null lacks response object {key}")
        if len(pieces) != int(row.assay_panels):
            raise RuntimeError(f"panel count differs for {key}")
        study_null.append(np.mean(np.stack(pieces), axis=0).astype(np.float32))
    null = np.stack(study_null, axis=1)  # permutation x object x embedding
    pair_scores = np.full((args.permutations, len(manifest), len(manifest)), np.nan, dtype=np.float32)
    max_candidate = np.full((len(manifest), args.permutations), np.nan, dtype=np.float32)
    study_ids = manifest.study_id.astype(str).to_numpy()
    for query_row in range(len(manifest)):
        candidate_rows = np.flatnonzero(study_ids != study_ids[query_row])
        scores = np.einsum("pd,pcd->pc", null[:, query_row], null[:, candidate_rows])
        pair_scores[:, query_row, candidate_rows] = scores
        max_candidate[query_row] = scores.max(axis=1)

    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / "response_assignment_null.npz",
        query_sketch_row=manifest.sketch_row.to_numpy(int),
        max_candidate_score=max_candidate,
        pair_scores=np.transpose(pair_scores, (1, 2, 0)),
    )
    report = {
        "schema": "factorial_hfd_response_assignment_null_v1",
        "status": "pretruth_max_candidate_null_frozen",
        "permutations": args.permutations,
        "seed": args.seed,
        "response_objects": len(manifest),
        "embedding_dimensions": dimension,
        "truth_opened": False,
        "ion_family_partition_available": args.families_root is not None,
        "null_operation": "permute bounded prevalence-weighted responses across anchors within panel and contrast",
        "preserved": [
            "study_and_panel_composition", "spectral_coordinates", "response_marginal_distribution",
            "prevalence_weights", "candidate_universe",
        ],
        "claim_boundary": "Null calibration only; no biological relationship has been selected or tested.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
