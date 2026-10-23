#!/usr/bin/env python3
"""Retrieve study contrasts by locally matched, directed chemical responses.

The existing factorial-HFD baseline compresses every panel to a single mean
response vector.  This implementation instead matches individual ion-family
representatives between panels before comparing their response directions.

Two evidence tracks are kept separate:

* ``identity_compatible``: mutual spectral neighbours whose precursor ions are
  within the same 10-ppm window.  These are identity *candidates*, not proven
  identities.
* ``chemical_relation``: mutual spectral neighbours outside that mass window.
  Their effects are never pooled as measurements of one molecule; they only
  support a local chemical-response relation.

Cross-study truth, compound names, pathways, and target ranks are not inputs.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


WORLDS = ("all", "known", "unresolved")
TRACKS = ("local_all", "identity_compatible", "chemical_relation")
CONTRAST_ORDER = (
    "diet_reference", "diet_modified", "genotype_control",
    "genotype_hfd", "genotype_by_hfd",
)


def infer_polarity(assay_file: str) -> str:
    upper = assay_file.upper()
    positive, negative = "POS" in upper, "NEG" in upper
    if positive == negative:
        raise ValueError(f"cannot infer unique polarity from {assay_file}")
    return "positive" if positive else "negative"


def as_bool(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.lower().isin(["true", "1"]).to_numpy()


def bounded_response(beta: np.ndarray, se: np.ndarray, prevalence: np.ndarray) -> np.ndarray:
    denominator = np.sqrt(beta * beta + se * se)
    signal = np.divide(beta, denominator, out=np.zeros_like(beta), where=denominator > 0)
    return signal * np.sqrt(np.clip(prevalence, 0.0, 1.0))


def mutual_top1(
    left: np.ndarray, right: np.ndarray, minimum_cosine: float, block_size: int = 512,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic mutual top-1 pairs without materializing all similarities."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("embedding matrices have incompatible shapes")
    if len(left) == 0 or len(right) == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.empty(0, dtype=np.float32)
    if device != "cpu":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("a non-CPU matching device requires PyTorch") from exc
        target = torch.device(device)
        right_tensor = torch.from_numpy(np.asarray(right, np.float32)).to(target)
        right_best_score_tensor = torch.full(
            (len(right),), -torch.inf, dtype=torch.float32, device=target,
        )
        right_best_left_tensor = torch.full(
            (len(right),), -1, dtype=torch.int64, device=target,
        )
        left_best_score_parts: list[np.ndarray] = []
        left_best_right_parts: list[np.ndarray] = []
        with torch.no_grad():
            for l0 in range(0, len(left), block_size):
                l1 = min(l0 + block_size, len(left))
                left_tensor = torch.from_numpy(np.asarray(left[l0:l1], np.float32)).to(target)
                similarity = left_tensor @ right_tensor.T
                row_score, row_arg = torch.max(similarity, dim=1)
                left_best_score_parts.append(row_score.cpu().numpy())
                left_best_right_parts.append(row_arg.cpu().numpy())
                col_score, col_arg = torch.max(similarity, dim=0)
                update = col_score > right_best_score_tensor
                right_best_score_tensor[update] = col_score[update]
                right_best_left_tensor[update] = l0 + col_arg[update]
        left_best_score = np.concatenate(left_best_score_parts).astype(np.float32, copy=False)
        left_best_right = np.concatenate(left_best_right_parts).astype(np.int64, copy=False)
        right_best_left = right_best_left_tensor.cpu().numpy()
        left_index = np.arange(len(left), dtype=np.int64)
        mutual = right_best_left[left_best_right] == left_index
        mutual &= left_best_score >= minimum_cosine
        return left_index[mutual], left_best_right[mutual], left_best_score[mutual]
    left_best_score = np.full(len(left), -np.inf, dtype=np.float32)
    left_best_right = np.full(len(left), -1, dtype=np.int64)
    right_best_score = np.full(len(right), -np.inf, dtype=np.float32)
    right_best_left = np.full(len(right), -1, dtype=np.int64)
    for l0 in range(0, len(left), block_size):
        l1 = min(l0 + block_size, len(left))
        for r0 in range(0, len(right), block_size):
            r1 = min(r0 + block_size, len(right))
            similarity = left[l0:l1] @ right[r0:r1].T
            row_arg = np.argmax(similarity, axis=1)
            row_score = similarity[np.arange(l1 - l0), row_arg]
            current = left_best_score[l0:l1]
            update = row_score > current
            left_best_score[l0:l1][update] = row_score[update]
            left_best_right[l0:l1][update] = r0 + row_arg[update]
            col_arg = np.argmax(similarity, axis=0)
            col_score = similarity[col_arg, np.arange(r1 - r0)]
            current = right_best_score[r0:r1]
            update = col_score > current
            right_best_score[r0:r1][update] = col_score[update]
            right_best_left[r0:r1][update] = l0 + col_arg[update]
    left_index = np.arange(len(left), dtype=np.int64)
    right_index = left_best_right
    mutual = (right_index >= 0) & (right_best_left[right_index] == left_index)
    mutual &= left_best_score >= minimum_cosine
    return left_index[mutual], right_index[mutual], left_best_score[mutual]


def local_response_statistic(
    left_response: np.ndarray,
    right_response: np.ndarray,
    left_edge: np.ndarray,
    right_edge: np.ndarray,
    similarity: np.ndarray,
    minimum_edges: int,
) -> dict[str, float | int | bool]:
    """Score signed concordance and penalize unmatched response mass.

    The edge weight is the smaller response magnitude on the two endpoints,
    multiplied by spectral similarity.  Thus a strong response on only one
    side cannot create support.  The final score is shrunk by the geometric
    mean of matched response-mass coverage on the two panels.
    """
    if not (len(left_edge) == len(right_edge) == len(similarity)):
        raise ValueError("edge arrays differ in length")
    if len(left_edge) == 0:
        return {
            "answerable": False, "n_edges": 0, "n_effective_edges": 0,
            "signed_concordance": np.nan, "left_coverage": 0.0,
            "right_coverage": 0.0, "score": np.nan,
        }
    left_value = left_response[left_edge]
    right_value = right_response[right_edge]
    weights = similarity * np.minimum(np.abs(left_value), np.abs(right_value))
    effective = weights > 0
    weight_sum = float(weights.sum())
    left_mass = float(np.abs(left_response).sum())
    right_mass = float(np.abs(right_response).sum())
    left_coverage = (
        float(np.abs(left_value).sum()) / left_mass if left_mass > 0 else 0.0
    )
    right_coverage = (
        float(np.abs(right_value).sum()) / right_mass if right_mass > 0 else 0.0
    )
    answerable = bool(len(left_edge) >= minimum_edges and weight_sum > 0)
    if not answerable:
        concordance = score = np.nan
    else:
        directions = np.sign(left_value) * np.sign(right_value)
        concordance = float(np.dot(weights, directions) / weight_sum)
        score = float(concordance * np.sqrt(left_coverage * right_coverage))
    return {
        "answerable": answerable,
        "n_edges": int(len(left_edge)),
        "n_effective_edges": int(effective.sum()),
        "signed_concordance": concordance,
        "left_coverage": left_coverage,
        "right_coverage": right_coverage,
        "score": score,
    }


@dataclass
class Panel:
    panel_id: str
    study_id: str
    source_family: str
    polarity: str
    entities: pd.DataFrame
    embeddings: np.ndarray
    responses: dict[str, np.ndarray]


def load_panels(args: argparse.Namespace) -> dict[str, Panel]:
    jobs = pd.read_csv(args.panel_jobs, sep="\t", dtype=str)
    registry = pd.read_csv(args.panel_registry, sep="\t", dtype=str).set_index(
        "panel_id", verify_integrity=True,
    )
    identity = pd.read_csv(args.identity_table, sep="\t", dtype=str)
    needed_identity = {"panel_id", "anchor_id", "identity_status"}
    if missing := needed_identity - set(identity):
        raise ValueError(f"identity table lacks {sorted(missing)}")
    if identity.duplicated(["panel_id", "anchor_id"]).any():
        raise ValueError("identity table contains duplicate panel/anchor rows")

    panels: dict[str, Panel] = {}
    dimension: int | None = None
    for job in jobs.sort_values("job_index", key=lambda value: value.astype(int)).itertuples(index=False):
        panel_id = str(job.panel_id)
        meta = registry.loc[panel_id]
        effects = pd.read_csv(
            args.effects_root / panel_id / "factorial_contrast_effects.tsv.gz", sep="\t",
            dtype={"anchor_id": str},
        )
        families = pd.read_csv(
            args.families_root / panel_id / "ion_family_membership.tsv", sep="\t",
            dtype={"anchor_id": str},
        )
        panel_object = Path(str(job.panel_object))
        anchors = pd.read_csv(
            panel_object / "anchor_manifest.tsv", sep="\t",
            dtype={"anchor_id": str},
        )
        with np.load(
            args.embeddings_root / panel_id / "anchor_embeddings.npz", allow_pickle=False,
        ) as packed:
            embedding_ids = np.asarray(packed["anchor_id"], dtype=str)
            embeddings = np.asarray(packed["embeddings"], dtype=np.float32)
        if dimension is None:
            dimension = int(embeddings.shape[1])
        if embeddings.shape != (len(embedding_ids), dimension):
            raise RuntimeError(f"invalid embedding shape for {panel_id}")
        if len(embedding_ids) != len(set(embedding_ids)):
            raise RuntimeError(f"duplicate embedding anchors for {panel_id}")

        representatives = families.loc[
            as_bool(families["representative"]), ["anchor_id", "family_id"]
        ].copy()
        entities = representatives.merge(
            anchors[["anchor_id", "precursor_mz"]], on="anchor_id", how="left",
            validate="one_to_one",
        ).merge(
            identity.loc[identity.panel_id.eq(panel_id), ["anchor_id", "identity_status"]],
            on="anchor_id", how="left", validate="one_to_one",
        ).merge(
            pd.DataFrame({"anchor_id": embedding_ids, "embedding_row": np.arange(len(embedding_ids))}),
            on="anchor_id", how="left", validate="one_to_one",
        )
        if entities.isna().any().any():
            raise RuntimeError(f"incomplete entity join for {panel_id}")
        entity_embeddings = embeddings[entities.embedding_row.to_numpy(int)]
        norms = np.linalg.norm(entity_embeddings, axis=1)
        if not np.isfinite(entity_embeddings).all() or np.max(np.abs(norms - 1.0)) > 2e-5:
            raise RuntimeError(f"embeddings are not finite unit vectors for {panel_id}")

        responses: dict[str, np.ndarray] = {}
        for contrast in CONTRAST_ORDER:
            block = effects.loc[effects.contrast_type.eq(contrast)].copy()
            if len(block) != effects.anchor_id.nunique():
                raise RuntimeError(f"duplicate or missing effects for {panel_id}/{contrast}")
            block = entities[["anchor_id"]].merge(
                block[["anchor_id", "beta", "se_hc3", "unit_prevalence", "passes_effect_qc"]],
                on="anchor_id", how="left", validate="one_to_one",
            )
            if block.isna().any().any() or not as_bool(block.passes_effect_qc).all():
                raise RuntimeError(f"invalid representative effects for {panel_id}/{contrast}")
            responses[contrast] = bounded_response(
                block.beta.to_numpy(float), block.se_hc3.to_numpy(float),
                block.unit_prevalence.to_numpy(float),
            )
        panels[panel_id] = Panel(
            panel_id=panel_id,
            study_id=str(meta.dataset_accession),
            source_family=str(meta.source_family),
            polarity=infer_polarity(str(meta.assay_file)),
            entities=entities.reset_index(drop=True),
            embeddings=entity_embeddings,
            responses=responses,
        )
    return panels


def world_mask(entities: pd.DataFrame, world: str) -> np.ndarray:
    status = entities.identity_status.astype(str).to_numpy()
    if world == "all":
        return np.ones(len(entities), dtype=bool)
    if world == "known":
        return np.isin(status, ["confirmed_known", "library_supported"])
    if world == "unresolved":
        return status == "unresolved_entity"
    raise ValueError(world)


def build_panel_edges(
    left: Panel, right: Panel, world: str, minimum_cosine: float,
    identity_ppm: float, block_size: int, device: str,
) -> pd.DataFrame:
    left_rows = np.flatnonzero(world_mask(left.entities, world))
    right_rows = np.flatnonzero(world_mask(right.entities, world))
    li, ri, similarity = mutual_top1(
        left.embeddings[left_rows], right.embeddings[right_rows], minimum_cosine,
        block_size, device,
    )
    li, ri = left_rows[li], right_rows[ri]
    left_mz = left.entities.precursor_mz.to_numpy(float)[li]
    right_mz = right.entities.precursor_mz.to_numpy(float)[ri]
    ppm = np.abs(left_mz - right_mz) / ((left_mz + right_mz) / 2.0) * 1e6
    return pd.DataFrame({
        "left_row": li, "right_row": ri, "cosine": similarity,
        "precursor_ppm": ppm,
        "track": np.where(ppm <= identity_ppm, "identity_compatible", "chemical_relation"),
    })


def summarize_track(
    left: Panel, right: Panel, edges: pd.DataFrame, left_contrast: str,
    right_contrast: str, track: str, minimum_edges: int,
) -> dict[str, object]:
    selected = edges if track == "local_all" else edges.loc[edges.track.eq(track)]
    statistic = local_response_statistic(
        left.responses[left_contrast], right.responses[right_contrast],
        selected.left_row.to_numpy(int), selected.right_row.to_numpy(int),
        selected.cosine.to_numpy(float), minimum_edges,
    )
    statistic.update({"track": track})
    return statistic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-jobs", required=True, type=Path)
    parser.add_argument("--panel-registry", required=True, type=Path)
    parser.add_argument("--effects-root", required=True, type=Path)
    parser.add_argument("--embeddings-root", required=True, type=Path)
    parser.add_argument("--families-root", required=True, type=Path)
    parser.add_argument("--identity-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-cosine", type=float, default=0.70)
    parser.add_argument("--identity-ppm", type=float, default=10.0)
    parser.add_argument("--minimum-local-edges", type=int, default=3)
    parser.add_argument("--minimum-track-edges", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    panels = load_panels(args)
    studies = sorted({panel.study_id for panel in panels.values()})
    by_study = {
        study: [panel for panel in panels.values() if panel.study_id == study]
        for study in studies
    }

    edge_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    pair_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for query_study in studies:
        for target_study in studies:
            if target_study == query_study:
                continue
            panel_pairs = [
                (left, right)
                for left in by_study[query_study] for right in by_study[target_study]
                if left.polarity == right.polarity
            ]
            for world in WORLDS:
                for left, right in panel_pairs:
                    key = (left.panel_id, right.panel_id, world)
                    if key not in edge_cache:
                        edge_cache[key] = build_panel_edges(
                            left, right, world, args.minimum_cosine,
                            args.identity_ppm, args.block_size, args.device,
                        )
                for query_contrast in CONTRAST_ORDER:
                    for target_contrast in CONTRAST_ORDER:
                        aggregate: dict[str, list[float]] = {track: [] for track in TRACKS}
                        counts = {track: 0 for track in TRACKS}
                        effective = {track: 0 for track in TRACKS}
                        for left, right in panel_pairs:
                            edges = edge_cache[(left.panel_id, right.panel_id, world)]
                            for track in TRACKS:
                                minimum = (
                                    args.minimum_local_edges if track == "local_all"
                                    else args.minimum_track_edges
                                )
                                summary = summarize_track(
                                    left, right, edges, query_contrast,
                                    target_contrast, track, minimum,
                                )
                                counts[track] += int(summary["n_edges"])
                                effective[track] += int(summary["n_effective_edges"])
                                if bool(summary["answerable"]):
                                    aggregate[track].append(float(summary["score"]))
                                pair_rows.append({
                                    "query_study_id": query_study,
                                    "query_contrast_id": query_contrast,
                                    "target_study_id": target_study,
                                    "target_contrast_id": target_contrast,
                                    "evidence_world": world,
                                    "query_panel_id": left.panel_id,
                                    "target_panel_id": right.panel_id,
                                    **summary,
                                })
                            for edge in edges.itertuples(index=False):
                                lr, rr = int(edge.left_row), int(edge.right_row)
                                lv = float(left.responses[query_contrast][lr])
                                rv = float(right.responses[target_contrast][rr])
                                contribution_rows.append({
                                    "query_study_id": query_study,
                                    "query_contrast_id": query_contrast,
                                    "target_study_id": target_study,
                                    "target_contrast_id": target_contrast,
                                    "evidence_world": world,
                                    "query_panel_id": left.panel_id,
                                    "target_panel_id": right.panel_id,
                                    "polarity": left.polarity,
                                    "query_anchor_id": str(left.entities.anchor_id.iloc[lr]),
                                    "target_anchor_id": str(right.entities.anchor_id.iloc[rr]),
                                    "query_precursor_mz": float(left.entities.precursor_mz.iloc[lr]),
                                    "target_precursor_mz": float(right.entities.precursor_mz.iloc[rr]),
                                    "cosine": float(edge.cosine),
                                    "precursor_ppm": float(edge.precursor_ppm),
                                    "evidence_track": str(edge.track),
                                    "query_response": lv,
                                    "target_response": rv,
                                    "direction_agreement": int(np.sign(lv) * np.sign(rv)),
                                    "response_support_weight": float(
                                        edge.cosine * min(abs(lv), abs(rv))
                                    ),
                                })
                        row: dict[str, object] = {
                            "query_study_id": query_study,
                            "query_contrast_id": query_contrast,
                            "target_study_id": target_study,
                            "target_contrast_id": target_contrast,
                            "evidence_world": world,
                            "same_polarity_panel_pairs": len(panel_pairs),
                        }
                        for track in TRACKS:
                            values = aggregate[track]
                            row[f"{track}_score"] = float(np.mean(values)) if values else np.nan
                            row[f"{track}_answerable_panel_pairs"] = len(values)
                            row[f"{track}_edges"] = counts[track]
                            row[f"{track}_effective_edges"] = effective[track]
                        score_rows.append(row)

    scores = pd.DataFrame(score_rows)
    group = ["query_study_id", "query_contrast_id", "evidence_world"]
    for track in TRACKS:
        column = f"{track}_score"
        scores[f"rank_{column}"] = scores.groupby(group)[column].rank(
            ascending=False, method="min",
        )
        scores[f"rank_opposed_{column}"] = scores.groupby(group)[column].rank(
            ascending=True, method="min",
        )

    args.output_dir.mkdir(parents=True)
    scores.to_csv(
        args.output_dir / "local_directed_retrieval_scores.tsv.gz", sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(pair_rows).to_csv(
        args.output_dir / "panel_pair_local_scores.tsv.gz", sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(contribution_rows).to_csv(
        args.output_dir / "local_edge_contributions.tsv.gz", sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    report = {
        "schema": "factorial_hfd_local_directed_response_retrieval_v1",
        "status": "development_local_directed_response_scores",
        "denominators": {
            "studies": len(studies), "panels": len(panels),
            "study_contrasts": len(studies) * len(CONTRAST_ORDER),
            "candidate_rows": len(scores),
            "edge_contribution_rows": len(contribution_rows),
        },
        "method": {
            "chemical_correspondence": "same-polarity mutual top-1 embedding neighbour",
            "minimum_cosine": args.minimum_cosine,
            "identity_compatible_ppm": args.identity_ppm,
            "panel_score": (
                "spectral-similarity weighted direction concordance, multiplied by the "
                "geometric mean of matched response-mass coverage"
            ),
            "study_score": "equal mean over answerable analytical-panel pairs",
            "worlds": list(WORLDS),
            "matching_device": args.device,
        },
        "claim_boundary": [
            "Identity-compatible edges are same-mass hypotheses, not confirmed metabolite identities.",
            "Chemical-relation edges never pool beta or standard error across different entities.",
            "The method compares directed local responses rather than mean study embeddings.",
            "A biological relationship is not established by an unadjudicated retrieval rank.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
