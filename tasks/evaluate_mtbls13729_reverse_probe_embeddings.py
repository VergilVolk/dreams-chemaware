"""Evaluate one frozen shared embedding in reference-to-sample direction.

For every frozen reference spectrum and every biological sample, select the
highest-cosine precursor-matched DDA spectrum.  Calibration asks whether that
selected spectrum maps back to the source-paper Level-1 feature.  This is a
direction-specific application protocol, although cosine itself is symmetric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from evaluate_mtbls13729_reverse_probe_raw import nearest_target, parse_feature_id_set  # noqa: E402
from infer_mtbls13729_p2b_vs_dreams import resolve_query_embeddings  # noqa: E402


def sha256(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=["neg_rp", "pos_rp"], required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=ROOT / "data/mtbls13729/reverse_probe_manifest_v4",
    )
    parser.add_argument(
        "--alignment", type=Path,
        default=ROOT / "data/mtbls13729/reverse_probe_mgf_v1/alignment.csv",
    )
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--query-embedding-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--rt-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.method.replace("_", "").isalnum():
        raise ValueError("method must contain only letters, digits and underscores")
    target_path = args.targets or ROOT / f"data/mtbls13729/ms1_consensus/{args.panel}__requantification_targets.csv.gz"
    required = [
        args.manifest_dir / "reference_spectra.csv.gz",
        args.manifest_dir / "reference_identities.csv",
        args.manifest_dir / "sample_manifest.csv",
        args.alignment,
        args.reference_cache / "manifest.csv",
        args.reference_cache / "embeddings.npy",
        args.reference_cache / "report.json",
        args.query_embedding_dir / "manifest.csv",
        target_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    out = args.output_dir.resolve()
    report_path = out / f"{args.panel}__{args.method}__report.json"
    evidence_path = out / f"{args.panel}__{args.method}__identity_sample.csv.gz"
    if report_path.exists() or evidence_path.exists():
        raise RuntimeError(f"refusing to overwrite reverse embedding result in {out}")
    out.mkdir(parents=True, exist_ok=True)

    references = pd.read_csv(args.manifest_dir / "reference_spectra.csv.gz")
    references = references[references.panel.eq(args.panel)].copy()
    identities = pd.read_csv(args.manifest_dir / "reference_identities.csv")
    identities = identities[identities.panel.eq(args.panel)].copy()
    samples = pd.read_csv(args.manifest_dir / "sample_manifest.csv")
    samples = samples[samples.panel.eq(args.panel)].copy()
    alignment = pd.read_csv(args.alignment)
    alignment = alignment[alignment.panel.eq(args.panel)].sort_values("subset_row", kind="stable")
    cache_manifest = pd.read_csv(args.reference_cache / "manifest.csv")
    reference_embedding = np.load(args.reference_cache / "embeddings.npy", mmap_mode="r")
    if not (len(alignment) == len(cache_manifest) == len(reference_embedding) == len(references)):
        raise RuntimeError("reference subset/cache/alignment length mismatch")
    reference_by_id = references.set_index("reference_spectrum_id", drop=False)
    ordered_reference = reference_by_id.loc[alignment.reference_spectrum_id].reset_index(drop=True)
    if not np.array_equal(
        ordered_reference.ik14.fillna("").to_numpy(), cache_manifest.ik14.fillna("").to_numpy()
    ):
        raise RuntimeError("reference cache identity order differs from frozen alignment")
    reference_embedding = np.asarray(reference_embedding, dtype=np.float32)
    reference_embedding /= np.clip(np.linalg.norm(reference_embedding, axis=1, keepdims=True), 1e-12, None)

    query_manifest = pd.read_csv(args.query_embedding_dir / "manifest.csv")
    query_path = resolve_query_embeddings(args.query_embedding_dir)
    query_embedding = np.load(query_path, mmap_mode="r")
    if len(query_manifest) != len(query_embedding):
        raise RuntimeError("query manifest/embedding length mismatch")
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    query_embedding /= np.clip(np.linalg.norm(query_embedding, axis=1, keepdims=True), 1e-12, None)
    mass = pd.to_numeric(query_manifest.precursor_mz, errors="coerce").to_numpy(np.float64)
    rt = pd.to_numeric(query_manifest.RT, errors="coerce").to_numpy(np.float64)
    if np.nanquantile(rt, 0.99) < 100:
        rt *= 60.0
    query_manifest = query_manifest.copy()
    query_manifest["sample_id"] = query_manifest.file_name.astype(str)
    sample_groups = {
        sample_id: np.asarray(index, dtype=np.int64)
        for sample_id, index in query_manifest.groupby("sample_id", sort=False).groups.items()
    }
    expected_samples = set(samples.sample_id.astype(str))
    missing_samples = expected_samples - set(sample_groups)
    if missing_samples:
        raise RuntimeError(f"query embedding manifest misses samples: {sorted(missing_samples)[:5]}")

    targets = pd.read_csv(target_path).sort_values(["mz", "rt_sec", "feature_id"], kind="stable")
    tmz = targets.mz.to_numpy(np.float64)
    trt = targets.rt_sec.to_numpy(np.float64)
    tid = targets.feature_id.to_numpy(np.int64)
    sample_metadata = samples.set_index("sample_id").to_dict("index")
    rows: list[dict[str, object]] = []
    for ref_position, ref in enumerate(ordered_reference.itertuples(index=False)):
        ref_mass = float(ref.precursor_mz)
        delta = ref_mass * args.ppm * 1e-6
        ref_vector = reference_embedding[ref_position]
        for sample_id in samples.sample_id.astype(str):
            indices = sample_groups[sample_id]
            eligible = indices[(mass[indices] >= ref_mass - delta) & (mass[indices] <= ref_mass + delta)]
            if not len(eligible):
                continue
            scores = query_embedding[eligible] @ ref_vector
            best_local = int(np.flatnonzero(scores == np.max(scores))[0])
            query_index = int(eligible[best_local])
            feature, dppm, drt = nearest_target(
                float(mass[query_index]), float(rt[query_index]), tmz, trt, tid,
                args.ppm, args.rt_seconds,
            )
            metadata = sample_metadata[sample_id]
            rows.append({
                "panel": args.panel,
                "method": args.method,
                "reference_spectrum_id": ref.reference_spectrum_id,
                "ik14": ref.ik14,
                "reference_name": ref.name,
                "hypothesis_family": ref.hypothesis_family,
                "calibration_panel": bool(ref.calibration_panel),
                "identity_claim_scope": ref.identity_claim_scope,
                "sample_id": sample_id,
                "patient_id": metadata["patient_id"],
                "side": metadata["side"],
                "tissue": metadata["tissue"],
                "histology": metadata["histology"],
                "query_index": query_index,
                "query_scan": int(query_manifest.iloc[query_index].scan_number),
                "query_rt_sec": float(rt[query_index]),
                "query_precursor_mz": float(mass[query_index]),
                "embedding_similarity": float(scores[best_local]),
                "linked_feature_id": feature,
                "feature_delta_ppm": dppm,
                "feature_delta_rt_sec": drt,
            })
    spectrum_sample = pd.DataFrame(rows)
    if spectrum_sample.empty:
        raise RuntimeError("no precursor-matched reference/sample embedding pairs")
    evidence = (
        spectrum_sample.sort_values(
            ["ik14", "sample_id", "embedding_similarity", "reference_spectrum_id"],
            ascending=[True, True, False, True], kind="stable",
        )
        .groupby(["panel", "ik14", "sample_id"], sort=False, as_index=False)
        .head(1).reset_index(drop=True)
    )
    expected = identities[[
        "panel", "ik14", "author_level1_feature_ids", "frozen_hypothesis_feature_id"
    ]]
    evidence = evidence.merge(expected, on=["panel", "ik14"], how="left", validate="many_to_one")
    evidence["calibration_feature_match"] = [
        bool(pd.notna(feature) and int(feature) in parse_feature_id_set(ids))
        for feature, ids in zip(evidence.linked_feature_id, evidence.author_level1_feature_ids)
    ]
    evidence["hypothesis_feature_match"] = (
        evidence.linked_feature_id.notna()
        & evidence.frozen_hypothesis_feature_id.notna()
        & evidence.linked_feature_id.eq(evidence.frozen_hypothesis_feature_id)
    )
    evidence.to_csv(evidence_path, index=False, compression="gzip")

    calibration = evidence[evidence.calibration_panel & evidence.linked_feature_id.notna()]
    y = calibration.calibration_feature_match.astype(int).to_numpy()
    auc = (
        float(roc_auc_score(y, calibration.embedding_similarity))
        if len(np.unique(y)) == 2 else math.nan
    )
    frozen_hypothesis = evidence[evidence.frozen_hypothesis_feature_id.notna()]
    report = {
        "status": "mtbls13729_reverse_probe_embedding_complete",
        "formal": False,
        "panel": args.panel,
        "method": args.method,
        "reference_spectra_attempted": int(len(ordered_reference)),
        "identity_sample_pairs": int(len(evidence)),
        "calibration_pairs_with_ms1_feature_link": int(len(calibration)),
        "calibration_feature_matches": int(calibration.calibration_feature_match.sum()),
        "calibration_feature_match_fraction": (
            float(calibration.calibration_feature_match.mean()) if len(calibration) else math.nan
        ),
        "calibration_similarity_auc_for_expected_feature": auc,
        "frozen_hypothesis_pairs_with_ms1_feature_link": int(frozen_hypothesis.linked_feature_id.notna().sum()),
        "frozen_hypothesis_feature_matches": int(frozen_hypothesis.hypothesis_feature_match.sum()),
        "protocol": {
            "direction": "reference embedding -> every biological sample",
            "candidate_prefilter": f"same panel and {args.ppm:g} ppm precursor window",
            "selection": "maximum cosine per reference spectrum/sample, then maximum per identity/sample",
            "phenotype_used": False,
            "DDA_miss_is_absence": False,
        },
        "provenance": {
            "reference_cache_report_sha256": sha256(args.reference_cache / "report.json"),
            "query_embeddings_sha256": sha256(query_path),
            "evidence_sha256": sha256(evidence_path),
        },
        "claim_limit": (
            "Calibration and method comparison only. Library matches remain putative; phenotype association is separate."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
