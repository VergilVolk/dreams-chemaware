#!/usr/bin/env python
"""One-shot frozen BioAware evaluation on ST001154 HILIC-negative MS2."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from annotation.bioaware_negative_expert import (  # noqa: E402
    FrozenNegativeBioAwareExpert,
    apply_frozen_negative_bioaware_expert,
)
from annotation.embed import load_embedder  # noqa: E402
from audit_st001154_bioaware_external_readiness import checksum  # noqa: E402
from evaluate_bioaware_kgmn200std_hidden_seed import (  # noqa: E402
    adjacency,
    best_path_bottleneck,
    shortest_paths,
)


ALL_FEATURES = (
    "spectral_score",
    "known_mass_candidate_fraction",
    "known_path_fraction",
    "known_inverse_depth_mean",
    "known_log_seed_support_mean",
    "known_log_degree",
    "edge0_complete_fraction",
    "edge0_bottleneck_mean",
    "edge1_complete_fraction",
    "edge1_bottleneck_mean",
    "predicted_edge_increment",
)


def encode_queries(tensors: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model, weight, bias = load_embedder(device=device, n_highest_peaks=100)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(tensors.astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    dtype = next(model.parameters()).dtype
    output = []
    with torch.inference_mode():
        for batch in loader:
            values = batch[0].to(device=device, dtype=dtype)
            precursor = model(values, None)[:, 0]
            output.append(
                F.normalize(F.linear(precursor, weight, bias), dim=-1)
                .float()
                .cpu()
                .numpy()
            )
    return np.concatenate(output).astype(np.float32)


def unique_top(group: pd.DataFrame, score_column: str) -> tuple[str, bool]:
    maximum = float(group[score_column].max())
    top = group[np.isclose(group[score_column], maximum, rtol=0, atol=1e-12)]
    return str(top.sort_values("candidate_id").iloc[0]["candidate_id"]), len(top) == 1


def cluster_bootstrap(
    frame: pd.DataFrame, cluster_column: str, repeats: int, seed: int
) -> dict:
    grouped = frame.groupby(cluster_column, sort=False)["delta"].agg(["sum", "count"])
    if grouped.empty:
        raise RuntimeError(f"empty cluster bootstrap: {cluster_column}")
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sampled = rng.integers(0, len(grouped), len(grouped))
        values[index] = sums[sampled].sum() / counts[sampled].sum()
    return {
        "mean": float(frame["delta"].mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "clusters": int(len(grouped)),
        "resamples": int(repeats),
    }


def subgroup_summary(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    corrected = int(frame["corrected"].sum())
    introduced = int(frame["introduced"].sum())
    discordant = corrected + introduced
    return {
        "queries": int(len(frame)),
        "truth_identities": int(frame["truth_candidate_id"].nunique()),
        "baseline_recall1": float(frame["baseline_correct"].mean()),
        "bioaware_recall1": float(frame["final_correct"].mean()),
        "delta_recall1": float(frame["delta"].mean()),
        "corrected": corrected,
        "introduced": introduced,
        "interventions": int(frame["intervene"].sum()),
        "identity_cluster_bootstrap": cluster_bootstrap(frame, "truth_candidate_id", repeats, seed),
        "formula_cluster_bootstrap": cluster_bootstrap(frame, "truth_formula", repeats, seed + 1),
        "sample_cluster_bootstrap": cluster_bootstrap(frame, "sample_id", repeats, seed + 2),
        "mcnemar_exact_p": (
            float(binomtest(corrected, discordant, 0.5, alternative="two-sided").pvalue)
            if discordant
            else 1.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=Path("data/validation/bioaware_st001154_hilic_external_manifest_v1"),
    )
    parser.add_argument(
        "--library-manifest", type=Path,
        default=Path("data/models/mona_neg_dreams_emb/manifest.csv"),
    )
    parser.add_argument(
        "--library-embeddings", type=Path,
        default=Path("data/models/mona_neg_dreams_emb/embeddings.npy"),
    )
    parser.add_argument(
        "--network", type=Path,
        default=Path("data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz"),
    )
    parser.add_argument(
        "--artifact", type=Path,
        default=Path(
            "data/validation/bioaware_metdna3_negative_network_expert_v2_chemically_filtered/"
            "artifact.json"
        ),
    )
    parser.add_argument(
        "--artifact-report", type=Path, default=None,
        help="Optional freeze report whose artifact SHA256 must match --artifact.",
    )
    parser.add_argument(
        "--library-integrity", type=Path,
        default=Path(
            "data/validation/mona_negative_library_chemical_integrity_v1/"
            "library_row_integrity.csv.gz"
        ),
    )
    parser.add_argument(
        "--candidate-protocol", choices=("mass_10ppm", "same_formula_10ppm"),
        default="mass_10ppm",
    )
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=Path("data/e1/official_embedding_slim.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_st001154_hilic_frozen_evaluation_v1"),
    )
    parser.add_argument("--maximum-depth", type=int, default=2)
    parser.add_argument("--maximum-paths", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    paths = {
        "report": args.manifest_dir / "report.json",
        "queries": args.manifest_dir / "queries.csv.gz",
        "candidates": args.manifest_dir / "candidate_references.csv.gz",
        "seeds": args.manifest_dir / "seed_features.csv.gz",
        "query_tensors": args.manifest_dir / "query_tensors.npz",
        "seed_tensors": args.manifest_dir / "seed_tensors.npz",
        "library_manifest": args.library_manifest,
        "library_embeddings": args.library_embeddings,
        "network": args.network,
        "artifact": args.artifact,
        "official_checkpoint": args.official_checkpoint,
        "library_integrity": args.library_integrity,
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    manifest_report = json.loads(paths["report"].read_text(encoding="utf-8"))
    if manifest_report.get("pass_to_frozen_evaluation") is not True:
        raise RuntimeError("external manifest did not pass its frozen gates")
    for name in ("queries", "candidates", "seeds", "query_tensors", "seed_tensors"):
        expected = manifest_report["provenance"][f"{name}_sha256"]
        if checksum(paths[name]) != expected:
            raise RuntimeError(f"external manifest component hash mismatch: {name}")
    if checksum(args.official_checkpoint) != "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245":
        raise RuntimeError("official DreaMS checkpoint hash mismatch")
    if args.artifact_report is None:
        expected_artifact_sha256 = "a04f9a7d02f726702f1c03ec4bac2e9ac2e471a3f722422165555774e7944c74"
    else:
        if not args.artifact_report.is_file() or args.artifact_report.stat().st_size == 0:
            raise FileNotFoundError(args.artifact_report)
        freeze_report = json.loads(args.artifact_report.read_text(encoding="utf-8"))
        if freeze_report.get("ready_for_new_holdout") is not True:
            raise RuntimeError("BioAware artifact freeze report is not holdout-ready")
        expected_artifact_sha256 = str(freeze_report["artifact_sha256"])
    if checksum(args.artifact) != expected_artifact_sha256:
        raise RuntimeError("frozen BioAware artifact hash mismatch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    queries = pd.read_csv(paths["queries"])
    candidate_references = pd.read_csv(paths["candidates"])
    seeds = pd.read_csv(paths["seeds"])
    query_tensors = np.load(paths["query_tensors"], allow_pickle=False)["query_tensor"]
    seed_tensors = np.load(paths["seed_tensors"], allow_pickle=False)["seed_tensor"]
    if len(queries) != len(query_tensors) or len(seeds) != len(seed_tensors):
        raise RuntimeError("external tensor/table lengths disagree")
    total_manifest_queries = int(len(queries))
    if args.candidate_protocol == "same_formula_10ppm":
        integrity = pd.read_csv(
            args.library_integrity,
            usecols=["library_row", "calculated_formula", "approved_m_h_reference"],
        )
        candidate_references = candidate_references.merge(
            integrity, on="library_row", how="left", validate="many_to_one"
        ).merge(
            queries[["query_id", "truth_formula"]], on="query_id", how="left",
            validate="many_to_one",
        )
        candidate_references = candidate_references.loc[
            candidate_references["approved_m_h_reference"].astype(bool)
            & candidate_references["calculated_formula"].astype(str).eq(
                candidate_references["truth_formula"].astype(str)
            )
        ].copy()
        candidate_counts = candidate_references.groupby("query_id")["candidate_id"].nunique()
        truth_present = candidate_references.merge(
            queries[["query_id", "truth_candidate_id"]], on="query_id", validate="many_to_one"
        ).assign(
            is_truth=lambda frame: frame["candidate_id"].astype(str).eq(
                frame["truth_candidate_id"].astype(str)
            )
        ).groupby("query_id")["is_truth"].any()
        scoped_query_ids = set(
            candidate_counts[(candidate_counts >= 2) & truth_present.reindex(candidate_counts.index).fillna(False)].index
        )
        expected_scope = set(
            queries.loc[queries["same_formula_candidate_identities"] >= 2, "query_id"]
        )
        if scoped_query_ids != expected_scope:
            raise RuntimeError("same-formula candidate scope disagrees with frozen manifest")
        original_positions = queries.loc[
            queries["query_id"].isin(scoped_query_ids), "query_tensor_index"
        ].to_numpy(int)
        queries = queries.loc[queries["query_id"].isin(scoped_query_ids)].copy().reset_index(drop=True)
        query_tensors = query_tensors[original_positions]
        queries["query_tensor_index"] = np.arange(len(queries), dtype=int)
        candidate_references = candidate_references.loc[
            candidate_references["query_id"].isin(scoped_query_ids),
            ["query_id", "candidate_id", "library_row", "library_precursor_mz"],
        ].copy()
    elif args.candidate_protocol != "mass_10ppm":  # pragma: no cover
        raise RuntimeError("unsupported candidate protocol")
    query_index = dict(zip(queries["query_id"].astype(str), queries["query_tensor_index"].astype(int), strict=True))
    seed_index = dict(zip(seeds["seed_id"].astype(str), seeds["tensor_index"].astype(int), strict=True))

    library_manifest = pd.read_csv(args.library_manifest)
    library_embeddings = np.load(args.library_embeddings, mmap_mode="r")
    if library_embeddings.shape != (len(library_manifest), 1024):
        raise RuntimeError("MONA manifest/embedding shape mismatch")
    query_embeddings = encode_queries(query_tensors, device, args.batch_size)
    if query_embeddings.shape != (len(queries), 1024):
        raise RuntimeError("official query embedding shape mismatch")
    if np.max(np.abs(np.linalg.norm(query_embeddings, axis=1) - 1)) > 1e-5:
        raise RuntimeError("official query embeddings are not normalized")

    score_rows = []
    for (query_id, candidate_id), group in candidate_references.groupby(
        ["query_id", "candidate_id"], sort=False
    ):
        rows = group["library_row"].to_numpy(np.int64)
        similarities = np.asarray(library_embeddings[rows]) @ query_embeddings[query_index[str(query_id)]]
        best = int(np.argmax(similarities))
        score_rows.append(
            {
                "query_id": str(query_id),
                "candidate_id": str(candidate_id),
                "spectral_score": float(similarities[best]),
                "best_library_row": int(rows[best]),
                "reference_spectra": int(len(rows)),
            }
        )
    scores = pd.DataFrame(score_rows)
    baseline_rows = []
    for query_id, group in scores.groupby("query_id", sort=False):
        top, unique = unique_top(group, "spectral_score")
        truth = str(queries.loc[queries["query_id"].eq(query_id), "truth_candidate_id"].iloc[0])
        baseline_rows.append(
            {
                "query_id": query_id,
                "baseline_candidate_id": top,
                "baseline_unique": bool(unique),
                "baseline_correct": bool(unique and top == truth),
            }
        )
    baseline = pd.DataFrame(baseline_rows)

    edge_table = pd.read_csv(args.network)
    graph0 = adjacency(edge_table, 0)
    graph1 = adjacency(edge_table, 1)
    tensor_lookup = {
        str(query_id): query_tensors[int(position)]
        for query_id, position in query_index.items()
    }
    tensor_lookup.update(
        {
            str(seed_id): seed_tensors[int(position)]
            for seed_id, position in seed_index.items()
        }
    )
    feature_rows = []
    similarity_cache: dict[tuple[str, str], float] = {}
    for sample_id, local_queries in queries.groupby("sample_id", sort=False):
        query_ids = set(local_queries["query_id"].astype(str))
        local_scores = scores.loc[scores["query_id"].isin(query_ids)]
        local_seeds = seeds.loc[seeds["sample_id"].astype(str).eq(str(sample_id))]
        mass_options: dict[str, set[str]] = defaultdict(set)
        for row in local_scores.itertuples(index=False):
            mass_options[str(row.candidate_id)].add(str(row.query_id))
        exact_options: dict[str, set[str]] = defaultdict(set)
        for row in local_seeds.itertuples(index=False):
            exact_options[str(row.ik14)].add(str(row.seed_id))
        all_seed_identities = set(local_seeds["ik14"].astype(str))
        for query in local_queries.itertuples(index=False):
            query_id = str(query.query_id)
            truth = str(query.truth_candidate_id)
            visible_seeds = all_seed_identities - {truth}
            observed = set(local_scores["candidate_id"].astype(str)) | visible_seeds
            node_options = {
                identity: sorted(
                    mass_options.get(identity, set())
                    | (exact_options.get(identity, set()) if identity in visible_seeds else set())
                )
                for identity in observed
            }
            for row in local_scores.loc[local_scores["query_id"].eq(query_id)].itertuples(index=False):
                candidate = str(row.candidate_id)
                known_paths = shortest_paths(
                    candidate, graph0, visible_seeds, observed,
                    args.maximum_depth, args.maximum_paths,
                )
                predicted_paths = shortest_paths(
                    candidate, graph1, visible_seeds, observed,
                    args.maximum_depth, args.maximum_paths,
                )
                complete0, bottleneck0 = best_path_bottleneck(
                    known_paths, query_id, node_options, tensor_lookup, similarity_cache
                )
                complete1, bottleneck1 = best_path_bottleneck(
                    predicted_paths, query_id, node_options, tensor_lookup, similarity_cache
                )
                minimum_depth = len(known_paths[0]) - 1 if known_paths else 0
                shortest_seed_count = len({path[-1] for path in known_paths})
                feature_rows.append(
                    {
                        "query_id": query_id,
                        "candidate_id": candidate,
                        "spectral_score": float(row.spectral_score),
                        "known_mass_candidate_fraction": 1.0,
                        "known_path_fraction": float(bool(known_paths)),
                        "known_inverse_depth_mean": 1.0 / minimum_depth if minimum_depth else 0.0,
                        "known_log_seed_support_mean": float(np.log1p(shortest_seed_count)),
                        "known_log_degree": float(np.log1p(len(graph0.get(candidate, set())))),
                        "edge0_complete_fraction": float(complete0 > 0),
                        "edge0_bottleneck_mean": float(bottleneck0),
                        "edge1_complete_fraction": float(complete1 > 0),
                        "edge1_bottleneck_mean": float(bottleneck1),
                        "predicted_edge_increment": float(bottleneck1 - bottleneck0),
                        "known_shortest_paths": int(len(known_paths)),
                        "predicted_shortest_paths": int(len(predicted_paths)),
                        "raw_complete_known_paths": int(complete0),
                        "raw_complete_predicted_paths": int(complete1),
                    }
                )
    candidate_features = pd.DataFrame(feature_rows)
    expert = FrozenNegativeBioAwareExpert.load(args.artifact)
    if not set(expert.feature_names).issubset(ALL_FEATURES):
        raise RuntimeError("frozen BioAware feature contract is unsupported")
    inference = candidate_features[["query_id", "candidate_id", *expert.feature_names]].copy()
    scored, decisions = apply_frozen_negative_bioaware_expert(inference, expert)
    if len(decisions) != len(queries):
        raise RuntimeError("frozen expert did not cover every external query")

    truth_meta = queries[[
        "query_id", "sample_id", "truth_candidate_id", "truth_formula",
        "same_formula_candidate_identities", "emrn_seed_reachable_depth2",
    ]]
    evaluated = decisions.merge(baseline, on="query_id", validate="one_to_one").merge(
        truth_meta, on="query_id", validate="one_to_one"
    )
    evaluated["final_correct"] = evaluated["final_candidate_id"].astype(str).eq(
        evaluated["truth_candidate_id"].astype(str)
    )
    evaluated["corrected"] = ~evaluated["baseline_correct"].astype(bool) & evaluated["final_correct"]
    evaluated["introduced"] = evaluated["baseline_correct"].astype(bool) & ~evaluated["final_correct"]
    evaluated["delta"] = evaluated["final_correct"].astype(int) - evaluated["baseline_correct"].astype(int)

    overall = subgroup_summary(evaluated, args.bootstrap_resamples, args.seed)
    network_supported_frame = evaluated.loc[evaluated["emrn_seed_reachable_depth2"].astype(bool)]
    same_formula_frame = evaluated.loc[evaluated["same_formula_candidate_identities"] >= 2]
    network_supported = subgroup_summary(
        network_supported_frame, args.bootstrap_resamples, args.seed + 10
    )
    same_formula = subgroup_summary(same_formula_frame, args.bootstrap_resamples, args.seed + 20)
    gates = {
        "overall_corrected_gt_introduced": overall["corrected"] > overall["introduced"],
        "overall_identity_cluster_ci_positive": overall["identity_cluster_bootstrap"]["ci_low"] > 0,
        "overall_formula_cluster_ci_positive": overall["formula_cluster_bootstrap"]["ci_low"] > 0,
        "network_supported_nonnegative": network_supported["delta_recall1"] >= 0,
        "same_formula_nonnegative": same_formula["delta_recall1"] >= 0,
    }
    report = {
        "status": "bioaware_st001154_hilic_frozen_evaluation_complete",
        "formal": True,
        "protocol": (
            "one-shot frozen official DreaMS plus frozen BioAware v2; eight phenotype-blind "
            "acquisition-order-spaced biological samples; development identity purged; "
            f"candidate_protocol={args.candidate_protocol}"
        ),
        "candidate_protocol": args.candidate_protocol,
        "manifest_queries": total_manifest_queries,
        "expert_scope_queries": int(len(queries)),
        "overall": overall,
        "network_supported": network_supported,
        "same_formula_hard": same_formula,
        "abstention_reasons": decisions["abstention_reasons"].value_counts().to_dict(),
        "gates": gates,
        "confirmatory_pass": all(gates.values()),
        "contracts": {
            "artifact_frozen": True,
            "thresholds_unchanged": True,
            "candidate_protocol_unchanged": True,
            "truth_removed_from_network_seeds": True,
            "phenotype_used": False,
            "P2b": "forbidden",
            "shared_embedding_changed": False,
        },
        "provenance": {
            "manifest_report_sha256": checksum(paths["report"]),
            "official_checkpoint_sha256": checksum(args.official_checkpoint),
            "artifact_sha256": checksum(args.artifact),
            "artifact_report_sha256": (
                checksum(args.artifact_report) if args.artifact_report is not None else None
            ),
            "library_integrity_sha256": checksum(args.library_integrity),
            "library_manifest_sha256": checksum(args.library_manifest),
            "library_embeddings_sha256": checksum(args.library_embeddings),
            "network_sha256": checksum(args.network),
            "script_sha256": checksum(Path(__file__)),
        },
        "claim_limit": (
            "External author-structure raw-MS2 benchmark, not locally reinjected MSI Level 1. "
            "A passing result supports incremental ranking on this protocol only; it does not "
            "establish universal SOTA, flux, enzyme activity, or shared-embedding improvement."
        ),
    }
    args.output_dir.mkdir(parents=True)
    output_paths = {
        "candidate_features": args.output_dir / "candidate_features.csv.gz",
        "per_query": args.output_dir / "per_query.csv.gz",
        "query_embeddings": args.output_dir / "query_embeddings.npy",
        "report": args.output_dir / "report.json",
    }
    candidate_features.to_csv(output_paths["candidate_features"], index=False, compression="gzip")
    evaluated.to_csv(output_paths["per_query"], index=False, compression="gzip")
    np.save(output_paths["query_embeddings"], query_embeddings)
    for name in ("candidate_features", "per_query", "query_embeddings"):
        report["provenance"][f"{name}_sha256"] = checksum(output_paths[name])
    output_paths["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
