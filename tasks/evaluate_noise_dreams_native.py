"""Evaluate one native DreaMS Noise checkpoint on the frozen held graph."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from build_noise_dreams_native_triplets import stable_fold
from dreams.models.heads.heads import ContrastiveHead
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from noise_corrected_fullgraph_evaluation import (
    full_metrics,
    held_metric_evidence,
    paired_outcome_table,
    score_embeddings,
)
from noise_final_core import CandidateGraph
from train_e1_identity import load_base_model
from train_chemaware_dreams_native import make_spectrum as make_hdf5_spectrum


NATIVE_EVALUATION_BASELINE_VERSION = "official_slim_reencoded_v3"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--official-slim-checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--official-finetuned-checkpoint", type=Path)
    parser.add_argument("--mature-e8-checkpoint", type=Path)
    parser.add_argument("--e8-architecture-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=("primary", "replicate", "targeted", "control"),
        required=True,
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260920)
    parser.add_argument(
        "--write-held-metric-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write the large pair-level replay NPZ. Metrics and per-query ledgers "
            "are unchanged when this supplementary artifact is disabled."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    return value


def formula_cluster_ci(
    formulas: np.ndarray,
    values: np.ndarray,
    *,
    repeats: int,
    seed: int,
    hypotheses: int,
) -> dict[str, float | int]:
    grouped = pd.DataFrame({
        "formula": np.asarray(formulas, dtype=str),
        "value": np.asarray(values, dtype=np.float64),
    }).groupby("formula", sort=True).value.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    boot = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        take = rng.integers(0, len(grouped), len(grouped))
        boot[index] = 100.0 * sums[take].sum() / counts[take].sum()
    tail = 0.05 / (2 * hypotheses)
    return {
        "delta_pp": float(100.0 * np.mean(values)),
        "ci_low_pp": float(np.quantile(boot, tail)),
        "ci_high_pp": float(np.quantile(boot, 1.0 - tail)),
        "formula_clusters": int(len(grouped)),
        "bootstrap_resamples": int(repeats),
        "familywise_hypotheses": int(hypotheses),
    }


@torch.no_grad()
def encode_rows(
    model: ContrastiveHead,
    rows: np.ndarray,
    data: Path,
    preprocessor: SpectrumPreprocessor,
    *,
    batch_size: int,
    device: torch.device,
    label: str,
) -> np.ndarray:
    output = np.empty((len(rows), 1024), dtype=np.float32)
    model.eval()
    started = time.time()
    with h5py.File(data, "r") as handle:
        for left in range(0, len(rows), batch_size):
            right = min(left + batch_size, len(rows))
            spectra = []
            for row in rows[left:right]:
                spectrum = make_hdf5_spectrum(
                    handle["spectrum"][int(row)],
                    float(handle["precursor_mz"][int(row)]),
                )
                spectra.append(preprocessor(
                    spectrum.get_peak_list(),
                    prec_mz=spectrum.get_precursor_mz(),
                    high_form=False,
                ))
            tensor = torch.from_numpy(np.stack(spectra).astype(np.float32)).to(device)
            encoded = F.normalize(model(tensor).float(), dim=1)
            output[left:right] = encoded.cpu().numpy()
            if right == len(rows) or right % (batch_size * 20) == 0:
                print(
                    f"[{label}-held-encode] {right:,}/{len(rows):,}; "
                    f"{time.time() - started:.0f}s",
                    flush=True,
                )
    norms = np.linalg.norm(output, axis=1)
    if not np.all(np.isfinite(norms)) or np.max(np.abs(norms - 1.0)) > 2e-3:
        raise RuntimeError("native held embeddings are not finite unit vectors")
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("native held evaluation requires an allocated GPU")
    for path in (
        args.graph,
        args.embedding_cache,
        args.official_slim_checkpoint,
        args.source_manifest,
        args.data,
        args.checkpoint,
        args.architecture_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    comparator_arguments = (
        args.mature_e8_checkpoint,
        args.official_finetuned_checkpoint,
        args.e8_architecture_checkpoint,
    )
    if any(value is None for value in comparator_arguments) and not all(
        value is None for value in comparator_arguments
    ):
        raise RuntimeError(
            "mature E8 comparison requires checkpoint, official initialization, and raw architecture"
        )
    for optional in comparator_arguments:
        if optional is not None and not optional.is_file():
            raise FileNotFoundError(optional)
    graph = CandidateGraph(args.graph)
    with np.load(args.embedding_cache, allow_pickle=False) as cache:
        rows = np.asarray(cache["rows"], dtype=np.int64)
        selection_embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
    if (
        selection_embeddings.shape != (len(rows), 1024)
    ):
        raise RuntimeError(
            "selection embedding cache and candidate-graph registry disagree"
        )
    with np.load(args.source_manifest, allow_pickle=False) as manifest:
        query_adduct = np.asarray(manifest["query_adduct"], dtype=str)
        if not np.array_equal(np.asarray(manifest["query_row"]), graph.query_row):
            raise RuntimeError("source manifest query rows differ from candidate graph")
    held_queries = np.flatnonzero(np.asarray([
        stable_fold(value, 5, args.formula_fold_seed) == args.outer_fold
        for value in graph.query_formula
    ], dtype=bool)).astype(np.int64)
    if len(held_queries) != 18333:
        raise RuntimeError(f"outer-held query count drifted: {len(held_queries)}")

    # Use the exact loader exercised by the sealed ChemAware native-triplet
    # result.  It reconstructs the official architecture and loads both the
    # shared backbone and the 1024-D projection head from either the official
    # slim initialization or a native Lightning continuation checkpoint.
    model, candidate_checkpoint_kind = load_base_model(
        args.checkpoint, args.architecture_checkpoint,
        torch.device("cuda"), 100,
    )
    preprocessor = SpectrumPreprocessor(
        dformat=DataFormatA(),
        prec_intens=1.1,
        n_highest_peaks=100,
        spec_entropy_cleaning=False,
        precision=32,
        mz_shift_aug_p=0,
        mz_shift_aug_max=0,
    )
    embeddings = encode_rows(
        model,
        rows,
        args.data,
        preprocessor,
        batch_size=args.batch_size,
        device=torch.device("cuda"),
        label=args.arm,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    candidate_scores = score_embeddings(graph, rows, embeddings)
    # The corrected graph was mined with an older lightweight selection cache.
    # It remains valid for candidate membership and action provenance, but the
    # native continuation must be compared with the same ContrastiveHead path
    # used to initialize training. Never mix its action embeddings with legacy
    # positive/negative embeddings or call the legacy geometry the baseline.
    official_model, official_checkpoint_kind = load_base_model(
        args.official_slim_checkpoint, args.architecture_checkpoint,
        torch.device("cuda"), 100,
    )
    official_embeddings = encode_rows(
        official_model, rows, args.data, preprocessor,
        batch_size=args.batch_size, device=torch.device("cuda"),
        label="official-slim",
    )
    del official_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    baseline_scores = score_embeddings(graph, rows, official_embeddings)
    selection_geometry_scores = score_embeddings(
        graph, rows, selection_embeddings
    )
    official_metric, official_table = full_metrics(
        graph, baseline_scores, query_adduct, held_queries
    )
    selection_geometry_metric, _ = full_metrics(
        graph, selection_geometry_scores, query_adduct, held_queries
    )
    candidate_metric, candidate_table = full_metrics(
        graph, candidate_scores, query_adduct, held_queries
    )
    mature_e8_metric = None
    mature_e8_table = None
    mature_e8_scores = None
    if args.mature_e8_checkpoint is not None:
        reference_model, _ = load_base_model(
            args.official_finetuned_checkpoint,
            args.e8_architecture_checkpoint,
            torch.device("cuda"),
            100,
        )
        reference_package = torch.load(
            args.mature_e8_checkpoint, map_location="cpu", weights_only=False
        )
        if (
            "model_state" not in reference_package
            or not reference_package.get("inference_clean_only")
            or reference_package.get("P2b_used")
        ):
            raise RuntimeError("mature E8 comparator violates clean shared-encoder contract")
        reference_model.load_state_dict(reference_package["model_state"], strict=True)
        reference_embeddings = encode_rows(
            reference_model,
            rows,
            args.data,
            preprocessor,
            batch_size=args.batch_size,
            device=torch.device("cuda"),
            label="mature-e8",
        )
        del reference_model, reference_package
        mature_e8_scores = score_embeddings(graph, rows, reference_embeddings)
        mature_e8_metric, mature_e8_table = full_metrics(
            graph, mature_e8_scores, query_adduct, held_queries
        )
    outcome = paired_outcome_table(official_table, candidate_table)
    near = outcome["near"].to_numpy(bool)
    corrected = outcome["corrected"].to_numpy(bool)
    introduced = outcome["introduced"].to_numpy(bool)
    r1_delta = (outcome["candidate_rank"].to_numpy(int) == 1).astype(float) - (
        outcome["official_rank"].to_numpy(int) == 1
    ).astype(float)
    # Familywise correction is frozen for the joint official/mature-E8,
    # all/near Recall@1 and MRR promotion family across both seeds.
    mrr_delta = (
        1.0 / outcome["candidate_rank"].to_numpy(float)
        - 1.0 / outcome["official_rank"].to_numpy(float)
    )
    ci = {
        "recall1": formula_cluster_ci(
            outcome["query_formula"].to_numpy(str), r1_delta,
            repeats=args.bootstrap_resamples, seed=args.bootstrap_seed, hypotheses=16,
        ),
        "near_recall1": formula_cluster_ci(
            outcome.loc[near, "query_formula"].to_numpy(str), r1_delta[near],
            repeats=args.bootstrap_resamples, seed=args.bootstrap_seed + 1, hypotheses=16,
        ),
        "mrr": formula_cluster_ci(
            outcome["query_formula"].to_numpy(str), mrr_delta,
            repeats=args.bootstrap_resamples, seed=args.bootstrap_seed + 2, hypotheses=16,
        ),
        "near_mrr": formula_cluster_ci(
            outcome.loc[near, "query_formula"].to_numpy(str), mrr_delta[near],
            repeats=args.bootstrap_resamples, seed=args.bootstrap_seed + 3, hypotheses=16,
        ),
    }
    risk = {
        "corrected": int(np.sum(corrected)),
        "introduced": int(np.sum(introduced)),
        "risk_net_lambda2": int(np.sum(corrected) - 2 * np.sum(introduced)),
        "near_corrected": int(np.sum(corrected & near)),
        "near_introduced": int(np.sum(introduced & near)),
        "near_risk_net_lambda2": int(
            np.sum(corrected & near) - 2 * np.sum(introduced & near)
        ),
    }
    candidate_vs_mature_e8 = None
    mature_e8_ci = None
    if mature_e8_table is not None:
        e8_rank = mature_e8_table["rank"].to_numpy(np.int64)
        outcome["mature_e8_rank"] = e8_rank
        e8_corrected = (e8_rank > 1) & (
            outcome["candidate_rank"].to_numpy(np.int64) == 1
        )
        e8_introduced = (e8_rank == 1) & (
            outcome["candidate_rank"].to_numpy(np.int64) > 1
        )
        candidate_vs_mature_e8 = {
            "corrected": int(np.sum(e8_corrected)),
            "introduced": int(np.sum(e8_introduced)),
            "risk_net_lambda2": int(np.sum(e8_corrected) - 2 * np.sum(e8_introduced)),
            "near_corrected": int(np.sum(e8_corrected & near)),
            "near_introduced": int(np.sum(e8_introduced & near)),
            "near_risk_net_lambda2": int(
                np.sum(e8_corrected & near) - 2 * np.sum(e8_introduced & near)
            ),
        }
        r1_vs_e8 = (
            outcome["candidate_rank"].to_numpy(np.int64) == 1
        ).astype(float) - (e8_rank == 1).astype(float)
        mrr_vs_e8 = (
            1.0 / outcome["candidate_rank"].to_numpy(float) - 1.0 / e8_rank
        )
        mature_e8_ci = {
            "recall1": formula_cluster_ci(
                outcome["query_formula"].to_numpy(str), r1_vs_e8,
                repeats=args.bootstrap_resamples,
                seed=args.bootstrap_seed + 10,
                hypotheses=16,
            ),
            "near_recall1": formula_cluster_ci(
                outcome.loc[near, "query_formula"].to_numpy(str), r1_vs_e8[near],
                repeats=args.bootstrap_resamples,
                seed=args.bootstrap_seed + 11,
                hypotheses=16,
            ),
            "mrr": formula_cluster_ci(
                outcome["query_formula"].to_numpy(str), mrr_vs_e8,
                repeats=args.bootstrap_resamples,
                seed=args.bootstrap_seed + 12,
                hypotheses=16,
            ),
            "near_mrr": formula_cluster_ci(
                outcome.loc[near, "query_formula"].to_numpy(str), mrr_vs_e8[near],
                repeats=args.bootstrap_resamples,
                seed=args.bootstrap_seed + 13,
                hypotheses=16,
            ),
        }
    evidence = None
    if args.write_held_metric_evidence:
        evidence = held_metric_evidence(
            graph,
            {
                "official": baseline_scores,
                "candidate": candidate_scores,
                "selection_geometry": selection_geometry_scores,
                **(
                    {"mature_e8": mature_e8_scores}
                    if mature_e8_scores is not None else {}
                ),
            },
            query_adduct,
            held_queries,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output.name}.", dir=args.output.parent
    ))
    outcome.to_csv(staging / "held_per_query.csv.gz", index=False, compression="gzip")
    if evidence is not None:
        np.savez_compressed(staging / "held_metric_evidence.npz", **evidence)
    report = {
        "status": "NOISE_DREAMS_NATIVE_HELD_EVALUATION_COMPLETE",
        "arm": args.arm,
        "evaluation_baseline_version": NATIVE_EVALUATION_BASELINE_VERSION,
        "held_queries": int(len(held_queries)),
        "official": official_metric,
        "official_geometry": "sealed official_embedding_slim native DreaMS",
        "candidate_checkpoint_kind": candidate_checkpoint_kind,
        "official_checkpoint_kind": official_checkpoint_kind,
        "selection_geometry_diagnostic": selection_geometry_metric,
        "candidate": candidate_metric,
        "candidate_vs_official": risk,
        "formula_cluster_paired_ci": ci,
        "mature_e8": mature_e8_metric,
        "candidate_vs_mature_e8": candidate_vs_mature_e8,
        "mature_e8_formula_cluster_paired_ci": mature_e8_ci,
        "provenance": {
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "mature_e8_checkpoint_sha256": (
                sha256_file(args.mature_e8_checkpoint)
                if args.mature_e8_checkpoint is not None else None
            ),
            "mature_e8_architecture_checkpoint_sha256": (
                sha256_file(args.e8_architecture_checkpoint)
                if args.e8_architecture_checkpoint is not None else None
            ),
            "candidate_graph_sha256": sha256_file(args.graph),
            "selection_embedding_cache_sha256": sha256_file(
                args.embedding_cache
            ),
            "official_slim_checkpoint_sha256": sha256_file(
                args.official_slim_checkpoint
            ),
            "embedding_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
            "held_per_query_sha256": sha256_file(staging / "held_per_query.csv.gz"),
            "held_metric_evidence_sha256": (
                sha256_file(staging / "held_metric_evidence.npz")
                if evidence is not None else None
            ),
        },
        "storage_contract": {
            "held_per_query_written": True,
            "held_metric_evidence_written": evidence is not None,
            "all_registered_metrics_computed": True,
            "omitted_artifact_is_supplementary_pair_level_replay_only": (
                evidence is None
            ),
        },
        "claim_limit": (
            "Corrected MassSpecGym formula-held graph; not an external NIST20 replication."
        ),
    }
    try:
        (staging / "report.json").write_text(
            json.dumps(json_native(report), indent=2), encoding="utf-8"
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(json_native(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
