#!/usr/bin/env python3
"""Score every frozen Contrast Atlas identity candidate with canonical WSE.

The candidate universe is read verbatim from the model-independent identity
axis.  No response estimate, phenotype, registered relation or identity truth
is read.  This script supplies the classical baseline and a query-major score
contract that later methods must join by ``directed_pair_id``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ms_entropy import calculate_entropy_similarity, clean_spectrum

from run_contrast_atlas_sparse_fragment_measure_baseline import iter_mgf, spectrum_id


STATUS = "DEVELOPMENT_CONSUMED_CONTRAST_ATLAS_COMPLETE_IDENTITY_WSE_SCORED"
SCHEMA = "contrast_atlas_complete_identity_wse_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(peaks: list[tuple[float, float]]) -> np.ndarray:
    if not peaks:
        return np.empty((0, 2), dtype=np.float32)
    values = np.asarray(peaks, dtype=np.float32)
    keep = (
        np.isfinite(values[:, 0])
        & np.isfinite(values[:, 1])
        & (values[:, 0] > 0)
        & (values[:, 1] > 0)
    )
    values = values[keep]
    if not len(values):
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(
        clean_spectrum(
            values,
            noise_threshold=0.01,
            min_ms2_difference_in_da=0.05,
            max_peak_num=-1,
        ),
        dtype=np.float32,
    )


def matched_fragments(left: np.ndarray, right: np.ndarray, tolerance: float) -> int:
    """Count deterministic one-to-one matches after 0.05-Da cleaning."""
    i = j = count = 0
    while i < len(left) and j < len(right):
        delta = float(left[i, 0] - right[j, 0])
        if abs(delta) <= tolerance:
            count += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return count


def load_spectra(
    registry: pd.DataFrame,
    source_report: dict,
    source_root: Path,
    expected_hashes: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    spectra: dict[str, np.ndarray] = {}
    provenance: dict[str, dict[str, object]] = {}
    source_paths = {
        str(row["study_id"]): Path(str(row["mgf_path"]).replace("\\", "/"))
        for row in source_report["sources"]
    }
    studies = set(registry["study_id"].astype(str))
    if studies != set(source_paths):
        raise RuntimeError("source report and identity registry study sets differ")

    for study in sorted(studies):
        path = source_root / source_paths[study]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hash = sha256(path)
        if observed_hash != expected_hashes[study]:
            raise RuntimeError(f"{study} MGF hash differs from the frozen identity axis")
        required = dict(
            zip(
                registry.loc[registry["study_id"].eq(study), "raw_entity_id"].astype(str),
                registry.loc[registry["study_id"].eq(study), "entity_id"].astype(str),
                strict=True,
            )
        )
        found: set[str] = set()
        for metadata, peaks in iter_mgf(path):
            raw_id = spectrum_id(metadata)
            entity_id = required.get(raw_id)
            if entity_id is None:
                continue
            if entity_id in found:
                raise RuntimeError(f"duplicate representative spectrum: {entity_id}")
            found.add(entity_id)
            spectra[entity_id] = clean(peaks)
        missing = set(required.values()) - found
        if missing:
            raise RuntimeError(f"{study} misses {len(missing)} frozen spectra")
        provenance[study] = {
            "path": str(path),
            "sha256": observed_hash,
            "spectra": len(found),
        }
        print(f"loaded {study}: {len(found)} spectra", flush=True)
    if len(spectra) != len(registry):
        raise RuntimeError("loaded spectrum denominator differs from registry")
    return spectra, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis-dir",
        type=Path,
        default=Path("data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027"),
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=Path("data/validation/contrast_atlas_sparse_local_alignment_v3_chemprop_20261023/report.json"),
    )
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/contrast_atlas_complete_identity_wse_v2_20261029"),
    )
    parser.add_argument("--fragment-tolerance-da", type=float, default=0.02)
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.fragment_tolerance_da <= 0:
        raise ValueError("fragment tolerance must be positive")

    axis_report_path = args.axis_dir / "report.json"
    registry_path = args.axis_dir / "observed_ion_registry.tsv.gz"
    pairs_path = args.axis_dir / "directed_complete_candidate_pairs.tsv.gz"
    blocks_path = args.axis_dir / "query_target_source_blocks.tsv.gz"
    for path in (axis_report_path, registry_path, pairs_path, blocks_path, args.source_report):
        if not path.is_file():
            raise FileNotFoundError(path)

    axis_report = json.loads(axis_report_path.read_text(encoding="utf-8"))
    if axis_report.get("status") != "DEVELOPMENT_CONSUMED_MODEL_INDEPENDENT_IDENTITY_AXIS_FROZEN":
        raise RuntimeError("refusing a non-frozen identity axis")
    expected = axis_report["outputs"]
    for filename, path in (
        ("observed_ion_registry.tsv.gz", registry_path),
        ("directed_complete_candidate_pairs.tsv.gz", pairs_path),
        ("query_target_source_blocks.tsv.gz", blocks_path),
    ):
        if sha256(path) != expected[filename]:
            raise RuntimeError(f"frozen axis hash mismatch: {filename}")

    registry = pd.read_csv(
        registry_path,
        sep="\t",
        dtype={"study_id": str, "raw_entity_id": str, "entity_id": str},
    )
    pairs = pd.read_csv(
        pairs_path,
        sep="\t",
        dtype={
            "query_block_id": str,
            "directed_pair_id": str,
            "query_entity_id": str,
            "candidate_entity_id": str,
        },
        low_memory=False,
    )
    blocks = pd.read_csv(
        blocks_path,
        sep="\t",
        dtype={"query_block_id": str},
    )
    denominators = axis_report["denominators"]
    if len(registry) != int(denominators["entities"]):
        raise RuntimeError("entity denominator drifted")
    if len(pairs) != int(denominators["directed_candidate_pairs"]):
        raise RuntimeError("candidate-pair denominator drifted")
    if len(blocks) != int(denominators["query_target_source_blocks"]):
        raise RuntimeError("query-block denominator drifted")
    if pairs["directed_pair_id"].duplicated().any():
        raise RuntimeError("duplicate directed_pair_id")
    if not np.array_equal(
        blocks["query_block_index"].to_numpy(np.int64), np.arange(len(blocks), dtype=np.int64)
    ):
        raise RuntimeError("query blocks are not in canonical contiguous order")
    starts = blocks["candidate_pair_offset_start"].to_numpy(np.int64)
    stops = blocks["candidate_pair_offset_end"].to_numpy(np.int64)
    if starts[0] != 0 or stops[-1] != len(pairs) or np.any(starts > stops):
        raise RuntimeError("query block offsets are invalid")
    if len(blocks) > 1 and not np.array_equal(starts[1:], stops[:-1]):
        raise RuntimeError("query block offsets are not contiguous")
    repeated = np.repeat(blocks["query_block_id"].astype(str), stops - starts)
    if not np.array_equal(repeated, pairs["query_block_id"].astype(str).to_numpy()):
        raise RuntimeError("candidate rows do not follow the canonical block order")

    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    spectra, mgf_provenance = load_spectra(
        registry,
        source_report,
        args.source_root.resolve(),
        axis_report["provenance"]["mgf_sha256_by_study"],
    )

    score = np.empty(len(pairs), dtype=np.float32)
    matched = np.empty(len(pairs), dtype=np.int16)
    available = np.empty(len(pairs), dtype=bool)
    started = time.time()
    for index, row in enumerate(pairs.itertuples(index=False)):
        left = spectra[str(row.query_entity_id)]
        right = spectra[str(row.candidate_entity_id)]
        available[index] = bool(len(left) and len(right))
        if available[index]:
            score[index] = float(
                calculate_entropy_similarity(
                    left,
                    right,
                    ms2_tolerance_in_da=args.fragment_tolerance_da,
                    clean_spectra=False,
                )
            )
            matched[index] = matched_fragments(left, right, args.fragment_tolerance_da)
        else:
            score[index] = 0.0
            matched[index] = 0
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            elapsed = time.time() - started
            print(
                f"scored {index + 1}/{len(pairs)} pairs ({(index + 1) / elapsed:.1f}/s)",
                flush=True,
            )
    if not np.all(np.isfinite(score)) or np.any((score < 0) | (score > 1 + 1e-6)):
        raise RuntimeError("WSE emitted invalid scores")

    rank_min = np.empty(len(pairs), dtype=np.int32)
    rank_stable = np.empty(len(pairs), dtype=np.int32)
    decision_rows: list[dict[str, object]] = []
    candidate_ids = pairs["candidate_entity_id"].astype(str).to_numpy()
    ppm = pd.to_numeric(pairs["symmetric_ppm"], errors="raise").to_numpy(float)
    for block in blocks.itertuples(index=False):
        start = int(block.candidate_pair_offset_start)
        stop = int(block.candidate_pair_offset_end)
        size = stop - start
        base = {
            "query_block_id": str(block.query_block_id),
            "query_block_index": int(block.query_block_index),
            "query_study_id": str(block.query_study_id),
            "query_entity_id": str(block.query_entity_id),
            "candidate_study_id": str(block.candidate_study_id),
            "candidate_count": size,
        }
        if size == 0:
            decision_rows.append({
                **base,
                "wse_decision_status": "abstain_no_physicochemical_candidate",
                "wse_answered": False,
                "wse_top1_tie_count": 0,
                "wse_top1_candidate_entity_id": "",
                "wse_top1_score": np.nan,
                "wse_top1_matched_fragments": 0,
                "wse_top1_minus_top2_margin": np.nan,
            })
            continue
        block_score = score[start:stop].astype(np.float64)
        block_matched = matched[start:stop].astype(np.int64)
        block_available = available[start:stop]
        order = sorted(
            range(size),
            key=lambda local: (
                -float(block_score[local]),
                -int(block_matched[local]),
                float(ppm[start + local]),
                str(candidate_ids[start + local]),
            ),
        )
        for stable_position, local in enumerate(order, start=1):
            rank_stable[start + local] = stable_position
            rank_min[start + local] = 1 + int(
                np.sum(block_score > float(block_score[local]) + 1e-12)
            )
        top_score = float(np.max(block_score))
        tied = np.flatnonzero(block_score >= top_score - 1e-12)
        all_available = bool(np.all(block_available))
        unique = len(tied) == 1
        chosen = int(tied[0]) if all_available and unique else None
        if not all_available:
            status = "abstain_missing_candidate_evidence"
        elif not unique:
            status = "abstain_tied_top1"
        elif size == 1:
            status = "singleton_identity_hypothesis"
        else:
            status = "unique_top1_identity_hypothesis"
        second = float(np.partition(block_score, -2)[-2]) if size > 1 else np.nan
        decision_rows.append({
            **base,
            "wse_decision_status": status,
            "wse_answered": bool(chosen is not None),
            "wse_top1_tie_count": int(len(tied)),
            "wse_top1_candidate_entity_id": "" if chosen is None else candidate_ids[start + chosen],
            "wse_top1_score": np.nan if chosen is None else float(block_score[chosen]),
            "wse_top1_matched_fragments": 0 if chosen is None else int(block_matched[chosen]),
            "wse_top1_minus_top2_margin": (
                np.nan if size == 1 or chosen is None else float(top_score - second)
            ),
        })

    output = args.output_dir
    output.mkdir(parents=True)
    scored = pairs[[
        "query_block_id",
        "query_block_index",
        "candidate_index_within_block",
        "directed_pair_id",
        "query_study_id",
        "query_entity_id",
        "candidate_study_id",
        "candidate_entity_id",
        "symmetric_ppm",
        "physicochemical_compatibility_grade",
        "candidate_axis",
    ]].copy()
    scored["wse_score"] = score
    scored["wse_matched_fragments"] = matched.astype(np.int32)
    scored["wse_available"] = available
    scored["wse_rank_min"] = rank_min
    scored["wse_rank_stable"] = rank_stable
    compression = {"method": "gzip", "mtime": 0}
    score_path = output / "candidate_scores.tsv.gz"
    scored.to_csv(score_path, sep="\t", index=False, compression=compression)

    query_path = output / "query_registry.tsv.gz"
    query = blocks[[
        "query_block_id",
        "query_block_index",
        "candidate_pair_offset_start",
        "candidate_pair_offset_end",
        "query_study_id",
        "query_entity_id",
        "candidate_study_id",
        "result_class",
    ]].copy()
    query["candidate_count"] = stops - starts
    query.to_csv(query_path, sep="\t", index=False, compression=compression)

    decision_path = output / "block_decisions.tsv.gz"
    decisions = pd.DataFrame(decision_rows)
    decisions.to_csv(decision_path, sep="\t", index=False, compression=compression)

    long_path = output / "method_scores_long.tsv.gz"
    long = scored[["directed_pair_id", "query_block_id"]].copy()
    long["method_name"] = "weighted_spectral_entropy"
    long["method_score"] = score
    long["evidence_available"] = available
    long["rank_min"] = rank_min
    long["rank_stable"] = rank_stable
    long.to_csv(long_path, sep="\t", index=False, compression=compression)

    axis_path = output / "wse_identity_inference_axis.npz"
    np.savez_compressed(
        axis_path,
        schema=np.asarray(SCHEMA),
        query_ptr=np.concatenate((starts[:1], stops)).astype(np.int64),
        query_block_id=blocks["query_block_id"].astype(str).to_numpy(dtype="U"),
        directed_pair_id=pairs["directed_pair_id"].astype(str).to_numpy(dtype="U"),
        wse_score=score,
        wse_matched_fragments=matched,
        wse_availability=available,
        wse_rank_min=rank_min,
        wse_rank_stable=rank_stable,
    )

    counts = stops - starts
    report = {
        "schema": SCHEMA,
        "status": STATUS,
        "data_role": "development_consumed",
        "truth_blinding": {
            "response_beta_se_read": False,
            "phenotype_read": False,
            "registered_relations_read": False,
            "identity_truth_read": False,
            "candidate_admission_changed": False,
        },
        "method": {
            "name": "weighted_spectral_entropy",
            "backend": "ms_entropy",
            "fragment_tolerance_da": args.fragment_tolerance_da,
            "cleaning": {
                "noise_threshold": 0.01,
                "minimum_ms2_difference_da": 0.05,
                "maximum_peak_number": -1,
            },
        },
        "denominators": {
            "studies": int(registry["study_id"].nunique()),
            "entities": int(len(registry)),
            "query_blocks": int(len(blocks)),
            "zero_candidate_blocks": int(np.sum(counts == 0)),
            "singleton_blocks": int(np.sum(counts == 1)),
            "multiple_candidate_blocks": int(np.sum(counts > 1)),
            "directed_candidate_pairs": int(len(pairs)),
            "finite_scores": int(np.isfinite(score).sum()),
            "available_scores": int(available.sum()),
            "answered_blocks": int(decisions["wse_answered"].sum()),
            "tied_top1_blocks": int(decisions["wse_decision_status"].eq("abstain_tied_top1").sum()),
        },
        "score_summary": {
            "minimum": float(score.min(initial=0.0)),
            "median": float(np.median(score)),
            "maximum": float(score.max(initial=0.0)),
            "positive": int(np.sum(score > 0)),
            "matched_fragment_positive": int(np.sum(matched > 0)),
        },
        "runtime_seconds": float(time.time() - started),
        "claim_boundary": [
            "WSE is a truth-blind classical identity-candidate baseline, not molecular-identity proof.",
            "This run does not evaluate unresolved-information, algorithm, response-retrieval or biological increment.",
            "All blocks, including zero- and one-candidate blocks, remain in the denominator.",
        ],
        "provenance": {
            "axis_report": {"path": str(axis_report_path), "sha256": sha256(axis_report_path)},
            "registry": {"path": str(registry_path), "sha256": sha256(registry_path)},
            "pairs": {"path": str(pairs_path), "sha256": sha256(pairs_path)},
            "blocks": {"path": str(blocks_path), "sha256": sha256(blocks_path)},
            "source_report": {"path": str(args.source_report), "sha256": sha256(args.source_report)},
            "mgf_by_study": mgf_provenance,
            "implementation": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))},
        },
        "outputs": {
            "candidate_scores.tsv.gz": sha256(score_path),
            "query_registry.tsv.gz": sha256(query_path),
            "block_decisions.tsv.gz": sha256(decision_path),
            "method_scores_long.tsv.gz": sha256(long_path),
            "wse_identity_inference_axis.npz": sha256(axis_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "checksums.sha256").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in report["outputs"].items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["denominators"], indent=2), flush=True)
    print(json.dumps(report["score_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
