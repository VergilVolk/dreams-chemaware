#!/usr/bin/env python
"""Build the missing MetDNA3 external negative-ion DreaMS benchmark.

The positive external benchmark used the MassSpecGym reference HDF5, which
contains only positive adducts.  This script closes that protocol gap for the
dominant negative adduct only: ``[M-H]-``.  It reuses the already decoded,
ambiguity-free external Level-1 spectra and the local MONA-negative reference
library.  No BioAware/network score is used here; the output is the frozen
official-DreaMS baseline on which a later network increment must be evaluated.

Truth labels are used only to require an evaluable retrieval query and to score
the final rank.  Candidate generation is feature-m/z-only (10 ppm), exactly as
in the positive benchmark.  Each candidate molecule is represented by the
maximum cosine over its MONA reference spectra, and ties count against truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation.embed import load_embedder  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def candidate_window(
    sorted_mz: np.ndarray,
    sorted_ik14: np.ndarray,
    feature_mz: float,
    ppm: float,
) -> tuple[int, int, np.ndarray]:
    tolerance = feature_mz * ppm * 1e-6
    lower = int(np.searchsorted(sorted_mz, feature_mz - tolerance, side="left"))
    upper = int(np.searchsorted(sorted_mz, feature_mz + tolerance, side="right"))
    return lower, upper, np.unique(sorted_ik14[lower:upper])


def strict_truth_top1(group: pd.DataFrame) -> bool:
    maximum = float(group["spectral_score"].max())
    top = group[np.isclose(group["spectral_score"], maximum, rtol=0, atol=1e-12)]
    truth = str(group["truth_candidate_id"].iloc[0])
    return bool(len(top) == 1 and str(top.iloc[0]["candidate_id"]) == truth)


def cluster_bootstrap_delta(
    transitions: pd.DataFrame, cluster_col: str, resamples: int, seed: int
) -> dict:
    local = transitions[[cluster_col, "baseline_correct", "final_correct"]].copy()
    local["delta"] = local["final_correct"].astype(float) - local["baseline_correct"].astype(float)
    grouped = local.groupby(cluster_col, sort=False)["delta"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    for position in range(resamples):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        values[position] = float(sums[sampled].sum() / counts[sampled].sum())
    return {
        "mean": float(local["delta"].mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "clusters": int(len(grouped)),
        "resamples": int(resamples),
    }


def encode_queries(tensors: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model, weight, bias = load_embedder(device=device, n_highest_peaks=100)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(tensors.astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    dtype = next(model.parameters()).dtype
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for position, batch in enumerate(loader, 1):
            x = batch[0].to(device=device, dtype=dtype)
            precursor = model(x, None)[:, 0]
            embedding = F.normalize(F.linear(precursor, weight, bias), dim=-1)
            output.append(embedding.float().cpu().numpy())
            if position % 10 == 0:
                print(f"[negative query encode] batches={position}", flush=True)
    return np.concatenate(output).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-root", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_v3_v1"),
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
        "--library-mgf", type=Path, default=Path("data/models/mona_neg_full.mgf")
    )
    parser.add_argument(
        "--approved-library-rows", type=Path, default=None,
        help="Optional .npy row index approved for the declared [M-H]- scope.",
    )
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=Path("data/e1/official_embedding_slim.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_dreams_v1"),
    )
    parser.add_argument("--candidate-ppm", type=float, default=10.0)
    parser.add_argument("--minimum-precursor-mz", type=float, default=50.0)
    parser.add_argument("--maximum-precursor-mz", type=float, default=1500.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    required = [
        args.external_root, args.library_manifest, args.library_embeddings,
        args.library_mgf, args.official_checkpoint,
    ]
    if args.approved_library_rows is not None:
        required.append(args.approved_library_rows)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256(args.official_checkpoint) != "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245":
        raise RuntimeError("official checkpoint hash mismatch")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is not empty: {output}")

    manifest = pd.read_csv(args.library_manifest)
    required_columns = {"inchikey", "precursor_mz"}
    if not required_columns.issubset(manifest.columns):
        raise RuntimeError(f"MONA manifest lacks {sorted(required_columns-manifest.columns)}")
    library_embedding = np.load(args.library_embeddings, mmap_mode="r")
    if library_embedding.shape != (len(manifest), 1024):
        raise RuntimeError("MONA manifest/embedding shape mismatch")
    norms = np.linalg.norm(library_embedding, axis=1)
    if not np.isfinite(library_embedding).all() or float(np.max(np.abs(norms - 1))) > 1e-5:
        raise RuntimeError("MONA embedding cache is non-finite or not unit-normalized")

    manifest = manifest.copy()
    manifest["ik14"] = manifest["inchikey"].fillna("").astype(str).str[:14].str.upper()
    precursor = pd.to_numeric(manifest["precursor_mz"], errors="coerce").to_numpy(float)
    valid = (
        np.isfinite(precursor)
        & (precursor >= args.minimum_precursor_mz)
        & (precursor <= args.maximum_precursor_mz)
        & manifest["ik14"].str.len().eq(14).to_numpy()
    )
    if args.approved_library_rows is not None:
        approved_rows = np.load(args.approved_library_rows, allow_pickle=False)
        if approved_rows.ndim != 1 or not np.issubdtype(approved_rows.dtype, np.integer):
            raise RuntimeError("approved library rows must be a one-dimensional integer array")
        if len(approved_rows) == 0 or int(approved_rows.min()) < 0 or int(approved_rows.max()) >= len(manifest):
            raise RuntimeError("approved library row index is empty or out of range")
        if len(np.unique(approved_rows)) != len(approved_rows):
            raise RuntimeError("approved library rows contain duplicates")
        approved_mask = np.zeros(len(manifest), dtype=bool)
        approved_mask[approved_rows] = True
        valid &= approved_mask
    valid_rows = np.flatnonzero(valid)
    order = np.argsort(precursor[valid_rows], kind="stable")
    sorted_rows = valid_rows[order]
    sorted_mz = precursor[sorted_rows]
    sorted_ik14 = manifest.loc[sorted_rows, "ik14"].to_numpy(str)

    query_rows: list[dict] = []
    tensor_rows: list[np.ndarray] = []
    candidate_rows: list[dict] = []
    unit_coverage: list[dict] = []
    cache_hashes: dict[str, dict] = {}
    for unit_dir in sorted(path for path in args.external_root.iterdir() if path.is_dir()):
        cache = unit_dir / "cache"
        spectrum_path = cache / "external_spectra.csv.gz"
        tensor_path = cache / "external_tensors.npz"
        report_path = cache / "report.json"
        for path in (spectrum_path, tensor_path, report_path):
            if not path.exists():
                raise FileNotFoundError(path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report.get("formal") or report.get("scope") != "external":
            raise RuntimeError(f"invalid external cache contract: {unit_dir.name}")
        external = pd.read_csv(spectrum_path)
        tensors = np.load(tensor_path, allow_pickle=False)["external_tensor"]
        if len(external) != len(tensors):
            raise RuntimeError(f"external manifest/tensor mismatch: {unit_dir.name}")
        negative = external[external["adduct"].eq("[M-H]-")]
        truth_present = 0
        evaluable = 0
        for position, source in negative.iterrows():
            feature_mz = float(source["feature_mz"])
            lower, upper, identities = candidate_window(
                sorted_mz, sorted_ik14, feature_mz, args.candidate_ppm
            )
            truth = str(source["truth_ik14"]).upper()
            if truth in identities:
                truth_present += 1
            if truth not in identities or len(identities) < 2:
                continue
            query_id = f"M3NEG:{unit_dir.name}:{int(source['truth_row']):04d}:{source['spectrum_id']}"
            if any(row["query_id"] == query_id for row in query_rows):
                raise RuntimeError(f"duplicate query id: {query_id}")
            query_rows.append({
                **source.to_dict(), "query_id": query_id,
                "candidate_identities": int(len(identities)),
            })
            tensor_rows.append(tensors[int(position)])
            for library_row in sorted_rows[lower:upper]:
                candidate_rows.append({
                    "query_id": query_id,
                    "candidate_id": str(manifest.iloc[int(library_row)]["ik14"]),
                    "library_row": int(library_row),
                    "library_precursor_mz": float(precursor[int(library_row)]),
                    "truth_candidate_id": truth,
                    "truth_formula": str(source["truth_formula"]),
                    "adduct": "[M-H]-",
                    "unit_id": unit_dir.name,
                })
            evaluable += 1
        unit_coverage.append({
            "unit_id": unit_dir.name,
            "negative_m_h_level1_spectra": int(len(negative)),
            "truth_identity_in_mona": int(truth_present),
            "evaluable_queries": int(evaluable),
        })
        cache_hashes[unit_dir.name] = {
            "report": sha256(report_path), "external_spectra": sha256(spectrum_path),
            "external_tensors": sha256(tensor_path),
        }

    query = pd.DataFrame(query_rows)
    candidate = pd.DataFrame(candidate_rows)
    query_tensor = np.stack(tensor_rows).astype(np.float32)
    if len(query) < 500 or query["unit_id"].nunique() != 8:
        raise RuntimeError("negative benchmark coverage gate failed")
    if query["spectrum_key"].duplicated().any():
        # A physical external spectrum must never represent two truth labels.
        duplicates = query[query["spectrum_key"].duplicated(False)]
        if duplicates.groupby("spectrum_key")["truth_ik14"].nunique().max() > 1:
            raise RuntimeError("one external spectrum maps to multiple truth identities")

    device = torch.device(args.device)
    query_embedding = encode_queries(query_tensor, device, args.batch_size)
    if query_embedding.shape != (len(query), 1024):
        raise RuntimeError("query embedding shape mismatch")

    query_position = {query_id: position for position, query_id in enumerate(query["query_id"])}
    score_rows: list[dict] = []
    for (query_id, candidate_id), group in candidate.groupby(
        ["query_id", "candidate_id"], sort=False
    ):
        q = query_embedding[query_position[query_id]]
        rows = group["library_row"].to_numpy(np.int64)
        similarities = np.asarray(library_embedding[rows]) @ q
        best = int(np.argmax(similarities))
        source = group.iloc[best]
        score_rows.append({
            "query_id": query_id, "candidate_id": candidate_id,
            "spectral_score": float(similarities[best]),
            "best_library_row": int(source["library_row"]),
            "reference_spectra": int(len(group)),
            "truth_candidate_id": str(source["truth_candidate_id"]),
            "truth_formula": str(source["truth_formula"]),
            "adduct": "[M-H]-", "unit_id": str(source["unit_id"]),
        })
    scores = pd.DataFrame(score_rows)
    transition_rows: list[dict] = []
    for query_id, group in scores.groupby("query_id", sort=False):
        correct = strict_truth_top1(group)
        truth = str(group["truth_candidate_id"].iloc[0])
        ordered = group.sort_values(
            ["spectral_score", "candidate_id"], ascending=[False, True]
        )
        truth_score = float(group.loc[group["candidate_id"].eq(truth), "spectral_score"].iloc[0])
        wrong_score = float(group.loc[~group["candidate_id"].eq(truth), "spectral_score"].max())
        qrow = query.iloc[query_position[query_id]]
        transition_rows.append({
            "query_id": query_id, "unit_id": str(qrow["unit_id"]),
            "truth_ik14": truth, "truth_formula": str(qrow["truth_formula"]),
            "candidate_identities": int(qrow["candidate_identities"]),
            "baseline_correct": correct, "final_correct": correct,
            "top_candidate_id": str(ordered.iloc[0]["candidate_id"]),
            "truth_score": truth_score, "hardest_wrong_score": wrong_score,
            "margin": truth_score - wrong_score,
        })
    transitions = pd.DataFrame(transition_rows)

    unit_results = []
    for unit_id, group in transitions.groupby("unit_id", sort=True):
        unit_results.append({
            "unit_id": unit_id, "queries": int(len(group)),
            "identities": int(group["truth_ik14"].nunique()),
            "formulas": int(group["truth_formula"].nunique()),
            "official_dreams_recall1": float(group["baseline_correct"].mean()),
            "errors": int((~group["baseline_correct"]).sum()),
            "median_margin": float(group["margin"].median()),
        })

    files = {
        "queries": output / "queries.csv.gz",
        "candidates": output / "candidate_references.csv.gz",
        "scores": output / "candidate_scores.csv.gz",
        "transitions": output / "transitions.csv.gz",
        "embeddings": output / "query_embeddings.npz",
        "report": output / "report.json",
    }
    query.to_csv(files["queries"], index=False, compression="gzip")
    candidate.to_csv(files["candidates"], index=False, compression="gzip")
    scores.to_csv(files["scores"], index=False, compression="gzip")
    transitions.to_csv(files["transitions"], index=False, compression="gzip")
    np.savez_compressed(files["embeddings"], query_embedding=query_embedding)
    payload = {
        "status": "bioaware_metdna3_external_negative_dreams_complete",
        "formal": True,
        "protocol": "external Level-1 [M-H]- query; MONA-negative reference; 10 ppm feature-m/z candidates; max cosine per IK14; ties fail",
        "queries": int(len(query)),
        "identities": int(query["truth_ik14"].nunique()),
        "formulas": int(query["truth_formula"].nunique()),
        "candidate_molecules": int(scores["candidate_id"].nunique()),
        "candidate_pairs": int(len(scores)),
        "official_dreams_recall1": float(transitions["baseline_correct"].mean()),
        "official_dreams_errors": int((~transitions["baseline_correct"]).sum()),
        "formula_cluster_baseline_reproduction": cluster_bootstrap_delta(
            transitions, "truth_formula", args.bootstrap_resamples, args.seed
        ),
        "unit_coverage": unit_coverage,
        "unit_results": unit_results,
        "library_audit": {
            "rows": int(len(manifest)), "valid_rows": int(len(valid_rows)),
            "valid_identities": int(manifest.loc[valid_rows, "ik14"].nunique()),
            "norm_min": float(norms.min()), "norm_max": float(norms.max()),
            "maximum_unit_norm_error": float(np.max(np.abs(norms - 1))),
            "adduct_scope": "MGF declares negative ion mode but lacks per-record adduct; benchmark is conservatively limited to external [M-H]- feature-m/z matching",
            "chemically_filtered_for_m_h": args.approved_library_rows is not None,
        },
        "contracts": {
            "candidate_generation_uses_truth_identity": False,
            "truth_used_only_for_evaluability_and_scoring": True,
            "one_external_spectrum_one_truth_identity": True,
            "shared_official_encoder": True,
            "BioAware_or_network_used": False,
            "P2b": "forbidden",
            "unsupported_negative_adducts_excluded": True,
        },
        "provenance": {
            "official_checkpoint_sha256": sha256(args.official_checkpoint),
            "library_mgf_sha256": sha256(args.library_mgf),
            "library_manifest_sha256": sha256(args.library_manifest),
            "library_embeddings_sha256": sha256(args.library_embeddings),
            "approved_library_rows_sha256": (
                sha256(args.approved_library_rows)
                if args.approved_library_rows is not None else None
            ),
            "external_caches": cache_hashes,
            **{f"{name}_sha256": sha256(path) for name, path in files.items() if name != "report"},
        },
        "gates": {
            "queries_ge_500": bool(len(query) >= 500),
            "all_eight_units_present": bool(query["unit_id"].nunique() == 8),
            "official_checkpoint_exact": True,
            "embedding_cache_unit_normalized": True,
        },
        "claim_limit": "This repairs the missing negative-ion official-DreaMS baseline. It is not a BioAware gain, a 16-panel combined effect, or a SOTA claim.",
    }
    payload["pass_to_negative_bioaware_increment"] = all(payload["gates"].values())
    atomic_json(files["report"], payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
