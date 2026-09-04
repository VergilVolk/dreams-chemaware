"""Phenotype-blind MTBLS13729 inference with official DreaMS and frozen P2b.

Unlike the legacy annotation tables, this builds the precursor-mass candidate
set *before* ranking.  Both systems see the identical library spectra and the
same 10 ppm candidate graph.  Only MS2 spectra linkable to a uniformly
requantified MS1 feature are evaluated, because these are the observations that
can enter the downstream biology analysis.

MTBLS13729 has no structure truth for most observations.  Consequently this
script reports retained/changed/abstained annotations and agreement evidence;
it never labels application changes as corrected or introduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from audit_large_observability_residual import symmetric_features  # noqa: E402
from g8r_p2_rank_fusion_core import (  # noqa: E402
    fuse_one_query,
    fusion_configuration_from_mapping,
    grouped_max,
    normalize_pair_features,
    unique_top_index,
)


EXPECTED_FEATURES = (
    "dreams_similarity", "sqrt_cosine", "entropy_similarity", "neutral_loss_sqrt_cosine"
)


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_query_embeddings(directory: Path) -> Path:
    normal = directory / "embeddings.npy"
    partial = directory / "embeddings.npy.filepart"
    for path in (normal, partial):
        if path.exists():
            try:
                values = np.load(path, mmap_mode="r")
            except Exception:
                continue
            if values.ndim == 2 and values.shape[1] == 1024:
                return path
    raise FileNotFoundError(f"no readable 1024-d query embedding array in {directory}")


def nearest_target_links(
    manifest: pd.DataFrame, targets: pd.DataFrame, ppm: float, rt_seconds: float
) -> pd.DataFrame:
    target = targets.sort_values("mz").reset_index(drop=True)
    masses = target["mz"].to_numpy(np.float64)
    target_rt = target["rt_sec"].to_numpy(np.float64)
    target_feature = target["feature_id"].to_numpy(np.int64)
    query_mass = pd.to_numeric(manifest["precursor_mz"], errors="coerce").to_numpy(np.float64)
    query_rt = pd.to_numeric(manifest["RT"], errors="coerce").to_numpy(np.float64)
    if np.nanquantile(query_rt, 0.99) < 100:
        query_rt *= 60.0
    # Vectorized nearest-mass prefilter removes the overwhelming majority of
    # MS2 rows before the exact multi-target/RT check below.
    insert = np.searchsorted(masses, query_mass, side="left")
    lower = np.clip(insert - 1, 0, len(masses) - 1)
    upper = np.clip(insert, 0, len(masses) - 1)
    lower_error = np.abs(masses[lower] - query_mass)
    upper_error = np.abs(masses[upper] - query_mass)
    nearest = np.where(lower_error <= upper_error, lower, upper)
    finite = np.isfinite(query_mass) & np.isfinite(query_rt) & (query_mass > 0)
    possible = finite & (np.abs(masses[nearest] - query_mass) <= query_mass * ppm * 1e-6)
    links = []
    for query_idx in np.flatnonzero(possible):
        mass, rt = query_mass[query_idx], query_rt[query_idx]
        delta = mass * ppm * 1e-6
        left = int(np.searchsorted(masses, mass - delta, side="left"))
        right = int(np.searchsorted(masses, mass + delta, side="right"))
        if left == right:
            continue
        dppm = np.abs(masses[left:right] - mass) / mass * 1e6
        drt = np.abs(target_rt[left:right] - rt)
        eligible = np.flatnonzero(drt <= rt_seconds)
        if not len(eligible):
            continue
        cost = (dppm[eligible] / ppm) ** 2 + (drt[eligible] / rt_seconds) ** 2
        # lexsort preserves the feature-id tie breaker in the original contract.
        local = int(eligible[np.lexsort((target_feature[left:right][eligible], cost))[0]])
        links.append({
            "query_idx": query_idx,
            "feature_id": int(target_feature[left + local]),
            "query_rt_sec": float(rt),
            "feature_dppm": float(dppm[local]),
            "feature_drt_sec": float(drt[local]),
        })
    return pd.DataFrame(links)


def top_spectrum_index(pair_scores: np.ndarray, molecule_ptr: np.ndarray, molecule_index: int) -> int:
    left, right = int(molecule_ptr[molecule_index]), int(molecule_ptr[molecule_index + 1])
    return left + int(np.argmax(pair_scores[left:right]))


def summarize_features(per_query: pd.DataFrame, method: str) -> pd.DataFrame:
    prefix = f"{method}_"
    usable = per_query[per_query[prefix + "ik14"].fillna("").astype(str).str.len() == 14].copy()
    if usable.empty:
        return pd.DataFrame()
    usable["query_spectrum_id"] = usable["query_file"].astype(str) + "::" + usable["query_scan"].astype(str)
    grouped = (
        usable.groupby(["feature_id", prefix + "ik14"], as_index=False)
        .agg(
            inchikey=(prefix + "inchikey", "first"),
            name=(prefix + "name", "first"),
            smiles=(prefix + "smiles", "first"),
            n_support_spectra=("query_spectrum_id", "nunique"),
            n_support_samples=("query_file", "nunique"),
            maximum_dreams_similarity=(prefix + "dreams_similarity", "max"),
            median_dreams_similarity=(prefix + "dreams_similarity", "median"),
            maximum_method_score=(prefix + "score", "max"),
            median_method_score=(prefix + "score", "median"),
        )
        .rename(columns={prefix + "ik14": "ik14"})
    )
    grouped = grouped.sort_values(
        ["feature_id", "n_support_samples", "n_support_spectra", "median_method_score", "maximum_method_score", "ik14"],
        ascending=[True, False, False, False, False, True],
        kind="stable",
    )
    total = usable.groupby("feature_id")["query_spectrum_id"].nunique()
    best = grouped.groupby("feature_id", sort=False).head(1).copy()
    best["n_linked_ms2"] = best.feature_id.map(total).astype(int)
    best["agreement_fraction"] = best.n_support_spectra / best.n_linked_ms2
    best["method"] = method
    best["annotation_evidence_tier"] = np.select(
        [
            (best.maximum_dreams_similarity >= 0.8) & (best.n_support_spectra >= 2) & (best.agreement_fraction >= 0.6),
            best.maximum_dreams_similarity >= 0.7,
            best.maximum_dreams_similarity >= 0.5,
        ],
        ["Level 2a-supported", "Level 2a-single/ambiguous", "Level 3-candidate"],
        default="unassigned",
    )
    return best.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=["neg_rp", "pos_rp"], required=True)
    parser.add_argument("--query-embedding-dir", type=Path)
    parser.add_argument("--query-hdf5-root", type=Path, default=ROOT / "data/mtbls13729/mzml")
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--library-cache", type=Path, required=True)
    parser.add_argument("--p2b-artifact", type=Path, default=ROOT / "data/validation/g8r_p2b_rank_fusion.json")
    parser.add_argument("--p3-result", type=Path, default=ROOT / "data/validation/g8r_p2b_p3_final.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--rt-seconds", type=float, default=20.0)
    parser.add_argument("--peak-tolerance", type=float, default=0.02)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_dir = args.query_embedding_dir or ROOT / f"data/mtbls13729/embeddings/{args.panel}"
    target_path = args.targets or ROOT / f"data/mtbls13729/ms1_consensus/{args.panel}__requantification_targets.csv.gz"
    required = [query_dir / "manifest.csv", target_path, args.p2b_artifact, args.p3_result]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    artifact = json.loads(args.p2b_artifact.read_text(encoding="utf-8"))
    if artifact.get("status") != "g8r_p2b_rank_fusion_frozen":
        raise RuntimeError("P2b artifact is not frozen")
    if tuple(artifact.get("selected_features", [])) != EXPECTED_FEATURES:
        raise RuntimeError("P2b feature schema drift")
    if artifact.get("p3_used_for_training_or_selection") is not False:
        raise RuntimeError("P2b artifact used sealed P3")
    configuration = fusion_configuration_from_mapping(artifact["configuration"])
    p3 = json.loads(args.p3_result.read_text(encoding="utf-8"))

    manifest = pd.read_csv(query_dir / "manifest.csv")
    targets = pd.read_csv(target_path)
    raw_links = nearest_target_links(manifest, targets, args.ppm, args.rt_seconds)
    raw_links["query_file"] = manifest.iloc[raw_links.query_idx.to_numpy(int)].file_name.to_numpy()
    raw_links["link_cost"] = (
        (raw_links.feature_dppm / args.ppm) ** 2
        + (raw_links.feature_drt_sec / args.rt_seconds) ** 2
    )
    # One deterministic DDA spectrum per feature and biological sample avoids
    # giving repeated fragmentation events from one injection extra voting
    # weight in feature-level consensus.
    links = (
        raw_links.sort_values(["feature_id", "query_file", "link_cost", "query_idx"], kind="stable")
        .groupby(["feature_id", "query_file"], sort=False, as_index=False)
        .head(1)
        .sort_values("query_idx", kind="stable")
        .reset_index(drop=True)
    )
    if args.max_queries:
        links = links.iloc[: args.max_queries].copy()
    if links.empty:
        raise RuntimeError("no query spectra link to MS1 targets")
    preflight = {
        "status": "mtbls13729_p2b_application_preflight",
        "panel": args.panel,
        "query_spectra": int(len(manifest)),
        "ms1_targets": int(len(targets)),
        "raw_linked_query_spectra": int(len(raw_links)),
        "selected_query_spectra": int(len(links)),
        "linked_features": int(links.feature_id.nunique()),
        "replicate_policy": "one closest MS2 per feature and sample; phenotype-blind",
        "legacy_protocol_warning": "legacy annotations ranked globally before mass filtering; not reused as the comparison graph",
        "candidate_protocol": "10 ppm exact precursor within polarity-specific unified_v2 library; query adduct unavailable",
        "P2b_boundary": "frozen downstream rank fusion, not a new DreaMS embedding",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = args.output_dir / f"{args.panel}__preflight.json"
    report_path = args.output_dir / f"{args.panel}__report.json"
    per_query_path = args.output_dir / f"{args.panel}__per_query.csv.gz"
    if not args.preflight_only and (report_path.exists() or per_query_path.exists()):
        raise FileExistsError(f"refusing to overwrite completed/partial panel output: {args.output_dir}")
    preflight_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    print(json.dumps(preflight, indent=2), flush=True)
    if args.preflight_only:
        return

    library_required = [args.library_cache / name for name in ("report.json", "manifest.csv", "embeddings.npy", "spectra.npy")]
    for path in library_required:
        if not path.exists():
            raise FileNotFoundError(path)
    library_report = json.loads((args.library_cache / "report.json").read_text(encoding="utf-8"))
    if library_report.get("status") != "unified_library_p2b_cache_complete":
        raise RuntimeError("library cache is incomplete")
    lib_manifest = pd.read_csv(args.library_cache / "manifest.csv")
    lib_embeddings = np.load(args.library_cache / "embeddings.npy", mmap_mode="r")
    lib_spectra = np.load(args.library_cache / "spectra.npy", mmap_mode="r")
    if not (len(lib_manifest) == len(lib_embeddings) == len(lib_spectra)):
        raise RuntimeError("library cache alignment failure")
    lib_mass = pd.to_numeric(lib_manifest.precursor_mz, errors="coerce").to_numpy(np.float64)
    lib_ik14 = lib_manifest.ik14.fillna("").astype(str).to_numpy(object)
    lib_inchikey = lib_manifest.inchikey.fillna("").astype(str).to_numpy(object)
    lib_name = lib_manifest["name"].fillna("").astype(str).to_numpy(object)
    lib_smiles = lib_manifest.smiles.fillna("").astype(str).to_numpy(object)
    valid_library_identity = np.asarray([len(value) == 14 for value in lib_ik14], dtype=bool)
    order = np.argsort(lib_mass, kind="stable")
    sorted_mass = lib_mass[order]
    query_embedding_path = resolve_query_embeddings(query_dir)
    query_embeddings = np.load(query_embedding_path, mmap_mode="r")
    if len(query_embeddings) != len(manifest):
        raise RuntimeError("query embedding and manifest length mismatch")

    links_by_file: dict[str, list[dict]] = defaultdict(list)
    link_lookup = links.set_index("query_idx").to_dict("index")
    for query_idx in links.query_idx.astype(int):
        row = manifest.iloc[query_idx]
        links_by_file[str(row.file_name)].append({"query_idx": query_idx, "row_in_file": int(row.row_in_file)})
    records = []
    total = len(links)
    completed = 0
    for query_file, requests in links_by_file.items():
        hdf5_path = args.query_hdf5_root / args.panel / f"{query_file}.hdf5"
        if not hdf5_path.exists():
            raise FileNotFoundError(hdf5_path)
        with h5py.File(hdf5_path, "r") as handle:
            for request in requests:
                query_idx = request["query_idx"]
                row_in_file = request["row_in_file"]
                query = manifest.iloc[query_idx]
                query_mass = float(query.precursor_mz)
                tolerance = query_mass * args.ppm * 1e-6
                left = int(np.searchsorted(sorted_mass, query_mass - tolerance, side="left"))
                right = int(np.searchsorted(sorted_mass, query_mass + tolerance, side="right"))
                candidate_rows = order[left:right]
                candidate_rows = candidate_rows[valid_library_identity[candidate_rows]]
                if not len(candidate_rows):
                    completed += 1
                    continue
                grouped: dict[str, list[int]] = defaultdict(list)
                for candidate_row in candidate_rows:
                    grouped[str(lib_ik14[int(candidate_row)])].append(int(candidate_row))
                identities = sorted(grouped)
                pair_rows, molecule_ptr, pair_library_rows = [], [0], []
                query_spectrum = np.asarray(handle["spectrum"][row_in_file], dtype=np.float32)
                query_embedding = np.asarray(query_embeddings[query_idx], dtype=np.float32)
                for identity in identities:
                    for candidate_row in sorted(grouped[identity]):
                        raw = symmetric_features(
                            query_spectrum, query_mass,
                            np.asarray(lib_spectra[candidate_row]), float(lib_mass[candidate_row]),
                            args.peak_tolerance,
                        )
                        dreams = float(query_embedding @ np.asarray(lib_embeddings[candidate_row], dtype=np.float32))
                        pair_rows.append([
                            dreams, float(raw["sqrt_cosine"]), float(raw["entropy_similarity"]),
                            float(raw["neutral_loss_sqrt_cosine"]),
                        ])
                        pair_library_rows.append(candidate_row)
                    molecule_ptr.append(len(pair_rows))
                values = np.asarray(pair_rows, dtype=np.float64)
                ptr = np.asarray(molecule_ptr, dtype=np.int64)
                normalized = normalize_pair_features(values, np.asarray([0, len(values)]), configuration.normalization)
                p2b_scores, intervened, support = fuse_one_query(
                    normalized, values[:, 0], ptr, np.asarray(configuration.weights),
                    (1, 2, 3), configuration.min_support, configuration.min_advantage,
                )
                dreams_scores = grouped_max(values[:, 0], ptr)
                dreams_top = unique_top_index(dreams_scores)
                p2b_top = unique_top_index(p2b_scores)
                link = link_lookup[query_idx]
                result = {
                    "panel": args.panel,
                    "query_idx": query_idx,
                    "query_file": query_file,
                    "query_scan": int(query.scan_number),
                    "query_precursor_mz": query_mass,
                    "query_rt_sec": float(link["query_rt_sec"]),
                    "feature_id": int(link["feature_id"]),
                    "feature_dppm": float(link["feature_dppm"]),
                    "feature_drt_sec": float(link["feature_drt_sec"]),
                    "n_candidate_spectra": int(len(values)),
                    "n_candidate_molecules": int(len(identities)),
                    "p2b_intervened": bool(intervened),
                    "p2b_support": int(support),
                }
                p2b_pair_score = (
                    normalized @ np.asarray(configuration.weights) if intervened else values[:, 0]
                )
                for method, top, scores, pair_score in (
                    ("dreams", dreams_top, dreams_scores, values[:, 0]),
                    ("p2b", p2b_top, p2b_scores, p2b_pair_score),
                ):
                    result[f"{method}_abstained_tie"] = top is None
                    if top is None:
                        for field in ("ik14", "inchikey", "name", "smiles"):
                            result[f"{method}_{field}"] = ""
                        result[f"{method}_score"] = math.nan
                        result[f"{method}_dreams_similarity"] = math.nan
                        continue
                    spectrum_position = top_spectrum_index(pair_score, ptr, top)
                    library_row = pair_library_rows[spectrum_position]
                    result[f"{method}_ik14"] = identities[top]
                    result[f"{method}_inchikey"] = str(lib_inchikey[library_row])
                    result[f"{method}_name"] = str(lib_name[library_row])
                    result[f"{method}_smiles"] = str(lib_smiles[library_row])
                    result[f"{method}_score"] = float(scores[top])
                    result[f"{method}_dreams_similarity"] = float(values[spectrum_position, 0])
                result["decision"] = (
                    "abstained" if dreams_top is None or p2b_top is None
                    else ("retained" if result["dreams_ik14"] == result["p2b_ik14"] else "changed")
                )
                records.append(result)
                completed += 1
                if completed % 1000 == 0 or completed == total:
                    print(f"[{args.panel}] {completed:,}/{total:,}; scored={len(records):,}", flush=True)

    per_query = pd.DataFrame(records)
    if per_query.empty:
        raise RuntimeError("no mass-constrained candidates were scored")
    per_query.to_csv(per_query_path, index=False)
    summaries = {}
    for method in ("dreams", "p2b"):
        frame = summarize_features(per_query, method)
        frame.to_csv(args.output_dir / f"{args.panel}__{method}_feature_annotations.csv.gz", index=False)
        summaries[method] = {
            "annotated_features": int(len(frame)),
            "level2a_supported": int((frame.annotation_evidence_tier == "Level 2a-supported").sum()),
            "level2a_single_or_ambiguous": int((frame.annotation_evidence_tier == "Level 2a-single/ambiguous").sum()),
            "level3_candidate": int((frame.annotation_evidence_tier == "Level 3-candidate").sum()),
            "median_agreement_fraction": float(frame.agreement_fraction.median()) if len(frame) else math.nan,
        }
    report = {
        "status": "mtbls13729_p2b_vs_dreams_inference_complete",
        "formal": True,
        "panel": args.panel,
        "queries_linked_to_ms1": int(len(links)),
        "queries_with_candidates": int(len(per_query)),
        "features_with_candidates": int(per_query.feature_id.nunique()),
        "decisions": {str(k): int(v) for k, v in per_query.decision.value_counts().items()},
        "systems": summaries,
        "engineering_benchmark": {
            "source": "frozen P2b sealed-P3 result",
            "status": p3.get("status"),
            "primary": p3.get("panels", {}).get("P3-main-real-pristine", {}),
            "near": p3.get("panels", {}).get("P3-near-core-real-pristine", {}),
        },
        "protocol": {
            "candidate_graph": "10 ppm before ranking; polarity-specific unified_v2",
            "feature_link": f"nearest MS1 target within {args.ppm:g} ppm and {args.rt_seconds:g} seconds",
            "phenotype_used": False,
            "application_truth_available": False,
            "application_transition_names": "retained/changed/abstained, never corrected/introduced",
            "P2b": "frozen downstream expert; official DreaMS embeddings unchanged",
        },
        "provenance": {
            "p2b_artifact_sha256": sha256_file(args.p2b_artifact),
            "p3_result_sha256": sha256_file(args.p3_result),
            "query_embeddings_sha256": sha256_file(query_embedding_path),
            "query_manifest_sha256": sha256_file(query_dir / "manifest.csv"),
            "targets_sha256": sha256_file(target_path),
            "library_manifest_sha256": sha256_file(args.library_cache / "manifest.csv"),
            "library_mgf_sha256": library_report["provenance"]["mgf_sha256"],
        },
        "claim_limit": "Application candidates lack structure truth. Accuracy superiority is supported only by the frozen engineering benchmark; MTBLS evidence tests consistency and biological usefulness.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
