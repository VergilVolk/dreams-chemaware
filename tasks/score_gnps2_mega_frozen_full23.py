#!/usr/bin/env python3
"""Apply frozen Full23 v1.3.2 and its experts to the GNPS2 mega stress axis.

No fitting, threshold selection, or post-score routing occurs here. Natural
10-ppm panels are candidate-ranking tasks. The separate 14.95M row artifact is
an explicitly known-overlap pairwise throughput/ROC stress task; Full23 is not
fabricated on that pair axis because its semantic unit is a candidate molecule
with a complete query block and up to three reference spectra.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

from encode_gnps_gold_silver_10ppm_checkpoint import load_selected_spectra, sha256_file
from noise_gnps_article_spectral_scores import (
    apply_frozen_p2b, frozen_p2b_pair_features, prepare_spectrum,
    weighted_entropy_backend, weighted_entropy_similarity,
)
from score_crossstudy_identity_multiencoder import MSG_FAMILY, score_frozen_noise_msg

try:
    import resource
except ImportError:  # Windows contract tests; production scoring is Linux/Slurm.
    resource = None


ENCODERS = ("noise_v1", "chemaware_stage1", "chemaware_phasea", "noise_chemaware_fullmsg")
ALL_EMBEDDINGS = ("official_dreams", *ENCODERS)


def named(items: list[str]) -> dict[str, Path]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--embedding must use NAME=PATH")
        name, raw = item.split("=", 1)
        if name in result:
            raise ValueError(f"duplicate embedding name: {name}")
        result[name] = Path(raw)
    return result


def load_banks(paths: dict[str, Path], rows: int) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, str]]:
    banks, hashes = {}, {}
    registry = None
    for name in ALL_EMBEDDINGS:
        path = paths[name]
        with np.load(path, allow_pickle=False) as body:
            selected = np.asarray(body["rows"], dtype=np.int64)
            values = np.asarray(body["embeddings"], dtype=np.float32)
        if values.shape != (len(selected), 1024) or len(np.unique(selected)) != len(selected):
            raise RuntimeError(f"malformed embedding cache: {name}")
        if np.any(selected < 0) or np.any(selected >= rows):
            raise RuntimeError(f"embedding row outside manifest: {name}")
        if registry is None:
            registry = selected
        elif not np.array_equal(registry, selected):
            raise RuntimeError("frozen encoder caches do not share one row registry")
        banks[name] = values
        hashes[name] = sha256_file(path)
    assert registry is not None
    row_to_position = np.full(rows, -1, dtype=np.int64)
    row_to_position[registry] = np.arange(len(registry), dtype=np.int64)
    return banks, row_to_position, hashes


def aggregate_max(values: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.maximum.reduceat(values, np.asarray(ptr[:-1], dtype=np.int64)).astype(np.float32)


def ranks_and_metrics(scores: np.ndarray, labels: np.ndarray, ptr: np.ndarray) -> dict:
    ranks = np.empty(len(ptr) - 1, dtype=np.int64)
    for i, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
        left, right = int(left), int(right)
        order = np.lexsort((np.arange(right-left), -scores[left:right]))
        positive = int(np.flatnonzero(labels[left:right] == 1)[0])
        ranks[i] = int(np.flatnonzero(order == positive)[0]) + 1
    result = {f"recall@{k}": float(np.mean(ranks <= k)) for k in (1, 2, 3, 5, 10, 20)}
    result.update({"mrr": float(np.mean(1.0 / ranks)), "mean_rank": float(np.mean(ranks)),
                   "median_rank": float(np.median(ranks)), "queries": int(len(ranks))})
    return result


def top3_positions(
    pair_score: np.ndarray, molecule_ptr: np.ndarray, query_ptr: np.ndarray,
    query_rows: np.ndarray, candidate_rows: np.ndarray, row_to_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_molecule = len(molecule_ptr) - 1
    query_position = np.empty(n_molecule, dtype=np.int64)
    refs = np.full((n_molecule, 3), -1, dtype=np.int64)
    mask = np.zeros((n_molecule, 3), dtype=bool)
    for qindex, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        for molecule in range(int(left), int(right)):
            query_position[molecule] = row_to_position[int(query_rows[qindex])]
            a, b = int(molecule_ptr[molecule]), int(molecule_ptr[molecule + 1])
            order = np.argsort(-pair_score[a:b], kind="stable")[:3]
            chosen = candidate_rows[a:b][order]
            positions = row_to_position[chosen]
            refs[molecule, :len(positions)] = positions
            mask[molecule, :len(positions)] = positions >= 0
    return query_position, refs, mask


def downsample(x: np.ndarray, y: np.ndarray, n: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= n:
        return x, y
    index = np.unique(np.linspace(0, len(x) - 1, n).astype(np.int64))
    return x[index], y[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--embedding", action="append", default=[])
    parser.add_argument("--msg-fusion-model", type=Path, required=True)
    parser.add_argument("--full23-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fragment-da", type=float, default=0.02)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--skip-natural-spectral", action="store_true")
    parser.add_argument("--pairwise-wse", action="store_true")
    args = parser.parse_args()
    global_started = time.time()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    embedding_paths = named(args.embedding)
    if set(embedding_paths) != set(ALL_EMBEDDINGS):
        raise RuntimeError(f"embedding registry must be {ALL_EMBEDDINGS}")
    report = json.loads((args.benchmark / "report.json").read_text(encoding="utf-8"))
    if report.get("schema") != "gnps2_mega_frozen_scoring_axis_v1":
        raise RuntimeError("unsupported GNPS2 mega scoring axis")
    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz")
    if not np.array_equal(manifest.row.to_numpy(np.int64), manifest.global_row.to_numpy(np.int64)):
        raise RuntimeError("row/global_row drifted")
    banks, row_to_position, bank_hashes = load_banks(embedding_paths, len(manifest))
    required_rows = []
    panels = {}
    for panel in ("identity_disjoint", "formula_disjoint"):
        with np.load(args.benchmark / f"panel_{panel}.npz", allow_pickle=False) as body:
            raw = {key: np.asarray(body[key]) for key in body.files}
        panels[panel] = raw
        required_rows.extend((raw["query_row"], raw["candidate_row"]))
    with np.load(args.benchmark / "pairwise_verification_indices.npz", allow_pickle=False) as body:
        qrow = np.asarray(body["query_row"], dtype=np.int64)
        rrow = np.asarray(body["reference_row"], dtype=np.int64)
        labels = np.asarray(body["label"], dtype=np.uint8)
        pair_type = np.asarray(body["pair_type"], dtype=np.uint8)
    required_rows.extend((qrow, rrow))
    selected = np.unique(np.concatenate(required_rows)).astype(np.int64)
    spectra_list = load_selected_spectra(args.benchmark / "spectra.mgf", selected, expected_records=len(manifest))
    spectra = {int(row): (prepare_spectrum(peaks, 100), precursor) for row, (peaks, precursor) in zip(selected, spectra_list, strict=True)}

    runtime = args.full23_package / "runtime"
    import sys
    sys.path.insert(0, str(runtime))
    from full23_v132_release import FrozenFull23V132
    from unified_additive_release import validate_axis
    scorer = FrozenFull23V132(device=args.device)
    msg_root = args.msg_fusion_model
    if msg_root.is_file():
        msg_root = msg_root.parent
    if not (msg_root / "fusion_model.joblib").is_file():
        raise FileNotFoundError(msg_root / "fusion_model.joblib")
    del joblib  # scoring helper owns the frozen artifact loading contract

    args.output_dir.mkdir(parents=True)
    natural_reports = {}
    for panel, raw in panels.items():
        started = time.time()
        query_rows = raw["query_row"].astype(np.int64)
        candidate_rows = raw["candidate_row"].astype(np.int64)
        query_ptr = raw["query_ptr"].astype(np.int64)
        molecule_ptr = raw["molecule_ptr"].astype(np.int64)
        n_molecule = len(molecule_ptr) - 1
        pair_query_rows = np.empty(len(candidate_rows), dtype=np.int64)
        for qi, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
            a, b = int(molecule_ptr[int(left)]), int(molecule_ptr[int(right)])
            pair_query_rows[a:b] = int(query_rows[qi])
        pair_embedding = {}
        molecule_embedding = {}
        for name in ALL_EMBEDDINGS:
            qpos = row_to_position[pair_query_rows]; rpos = row_to_position[candidate_rows]
            if np.any(qpos < 0) or np.any(rpos < 0):
                raise RuntimeError(f"{panel} requires rows absent from {name} cache")
            pair_embedding[name] = np.einsum(
                "ij,ij->i", banks[name][qpos], banks[name][rpos], optimize=True,
            ).astype(np.float32)
            molecule_embedding[name] = aggregate_max(pair_embedding[name], molecule_ptr)

        if args.skip_natural_spectral:
            natural_reports[panel] = {"status": "embedding_only", "queries": len(query_rows)}
            continue
        wse = np.empty(len(candidate_rows), dtype=np.float32)
        sqrt = np.empty_like(wse); entropy = np.empty_like(wse); neutral = np.empty_like(wse)
        for index, (qrow, rrow) in enumerate(zip(pair_query_rows, candidate_rows, strict=True)):
            qspec, qprec = spectra[int(qrow)]; rspec, rprec = spectra[int(rrow)]
            wse[index] = weighted_entropy_similarity(qspec, rspec, args.fragment_da)
            sqrt[index], entropy[index], neutral[index] = frozen_p2b_pair_features(
                qspec, qprec, rspec, rprec, tolerance=args.fragment_da,
            )
            if args.progress_every and (index + 1) % args.progress_every == 0:
                print(f"[{panel}] spectral {index+1:,}/{len(candidate_rows):,}", flush=True)
        graph = SimpleNamespace(query_ptr=query_ptr, molecule_ptr=molecule_ptr)
        p2b = apply_frozen_p2b(graph, pair_embedding["noise_v1"], sqrt, entropy, neutral)
        molecule = {
            "weighted_spectral_entropy": aggregate_max(wse, molecule_ptr),
            "p2b_noise_v1_frozen": p2b.molecule.astype(np.float32),
            **{name: molecule_embedding[name] for name in ENCODERS},
        }
        msg_table = pd.DataFrame({
            "noise_v1_score": molecule["noise_v1"],
            "official_dreams_score": molecule_embedding["official_dreams"],
            "WSE_score": molecule["weighted_spectral_entropy"],
            "WSE_available": True,
            "P2b_sqrt_cosine": aggregate_max(sqrt, molecule_ptr),
            "P2b_entropy": aggregate_max(entropy, molecule_ptr),
            "P2b_neutral_loss": aggregate_max(neutral, molecule_ptr),
            "P2b_score": molecule["p2b_noise_v1_frozen"],
        })
        msg, msg_report = score_frozen_noise_msg(msg_table, query_ptr, msg_root)
        molecule[MSG_FAMILY] = msg
        query_ids = np.asarray([f"row={row}" for row in query_rows])
        row_query_ids = np.repeat(query_ids, np.diff(query_ptr))
        pair_ids = np.asarray([f"{q}|ik14={c}" for q, c in zip(row_query_ids, raw["molecule_ik14"].astype("U"), strict=True)])
        axis = validate_axis(query_ptr, query_ids, row_query_ids, raw["molecule_ik14"], pair_ids)

        # Build the exact four-encoder top-3 reference tensor consumed by v1.3.2.
        qp, rp, masks = [], [], []
        for name in ENCODERS:
            qpos, refs, mask = top3_positions(
                pair_embedding[name], molecule_ptr, query_ptr, query_rows,
                candidate_rows, row_to_position,
            )
            qp.append(qpos); rp.append(refs); masks.append(mask)
        query_positions = np.stack(qp, axis=1)
        reference_positions = np.stack(rp, axis=1)
        reference_mask = np.stack(masks, axis=1)
        full23, full23_report = scorer.score(
            axis, molecule, pair_ids, query_positions, reference_positions,
            reference_mask, {name: banks[name] for name in ENCODERS},
            {name: bank_hashes[name] for name in ENCODERS}, chemical=None,
        )
        scores = {
            "weighted_spectral_entropy": molecule["weighted_spectral_entropy"],
            "official_dreams": molecule_embedding["official_dreams"],
            "noise_v1": molecule["noise_v1"],
            "p2b_noise_v1_frozen": molecule["p2b_noise_v1_frozen"],
            MSG_FAMILY: msg,
            "chemaware_stage1": molecule["chemaware_stage1"],
            "chemaware_phasea": molecule["chemaware_phasea"],
            "noise_chemaware_fullmsg": molecule["noise_chemaware_fullmsg"],
            "full23_v132": full23,
        }
        metrics = {name: ranks_and_metrics(value, raw["molecule_label"], query_ptr) for name, value in scores.items()}
        np.savez_compressed(
            args.output_dir / f"natural_topk_{panel}.npz", query_ptr=query_ptr,
            query_row=query_rows, candidate_ik14=raw["molecule_ik14"], labels=raw["molecule_label"],
            **scores,
        )
        natural_reports[panel] = {
            "queries": len(query_rows), "candidate_molecules": n_molecule,
            "reference_comparisons": len(candidate_rows), "metrics": metrics,
            "full23": full23_report, "msg": msg_report,
            "elapsed_seconds": time.time() - started,
        }

    # Known-overlap pairwise stress: exact embedding cosine only. Full23 is not
    # coerced into a non-molecule singleton-pair task.
    pairwise_report = {}
    pair_dir = args.output_dir / "pairwise_scores"
    pair_dir.mkdir()
    for name in ALL_EMBEDDINGS:
        method_started = time.time()
        score_path = pair_dir / f"{name}.npy"
        score = np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float32, shape=(len(qrow),))
        for left in range(0, len(qrow), 500_000):
            right = min(left + 500_000, len(qrow))
            qpos = row_to_position[qrow[left:right]]; rpos = row_to_position[rrow[left:right]]
            if np.any(qpos < 0) or np.any(rpos < 0):
                raise RuntimeError(f"pairwise axis requires rows absent from {name} cache")
            score[left:right] = np.einsum(
                "ij,ij->i", banks[name][qpos], banks[name][rpos], optimize=True,
            )
        score.flush()
        auc = float(roc_auc_score(labels, score)); ap = float(average_precision_score(labels, score))
        fpr, tpr, _ = roc_curve(labels, score); precision, recall, _ = precision_recall_curve(labels, score)
        fpr, tpr = downsample(fpr, tpr); recall, precision = downsample(recall, precision)
        np.savez_compressed(pair_dir / f"{name}_curves.npz", fpr=fpr, tpr=tpr, recall=recall, precision=precision)
        pairwise_report[name] = {
            "auroc": auc, "auprc": ap, "score_path": str(score_path),
            "score_bytes": score_path.stat().st_size,
            "elapsed_seconds": time.time() - method_started,
        }
        del score
    if args.pairwise_wse:
        method_started = time.time()
        score_path = pair_dir / "weighted_spectral_entropy.npy"
        score = np.lib.format.open_memmap(
            score_path, mode="w+", dtype=np.float32, shape=(len(qrow),),
        )
        for index, (qvalue, rvalue) in enumerate(zip(qrow, rrow, strict=True)):
            qspec, _ = spectra[int(qvalue)]; rspec, _ = spectra[int(rvalue)]
            score[index] = weighted_entropy_similarity(qspec, rspec, args.fragment_da)
            if args.progress_every and (index + 1) % args.progress_every == 0:
                print(f"[pairwise WSE] {index+1:,}/{len(qrow):,}", flush=True)
        score.flush()
        auc = float(roc_auc_score(labels, score)); ap = float(average_precision_score(labels, score))
        fpr, tpr, _ = roc_curve(labels, score); precision, recall, _ = precision_recall_curve(labels, score)
        fpr, tpr = downsample(fpr, tpr); recall, precision = downsample(recall, precision)
        np.savez_compressed(
            pair_dir / "weighted_spectral_entropy_curves.npz",
            fpr=fpr, tpr=tpr, recall=recall, precision=precision,
        )
        pairwise_report["weighted_spectral_entropy"] = {
            "auroc": auc, "auprc": ap, "score_path": str(score_path),
            "score_bytes": score_path.stat().st_size,
            "elapsed_seconds": time.time() - method_started,
        }
        del score

    final = {
        "schema": "gnps2_mega_frozen_full23_stress_result_v1",
        "status": "GNPS2_MEGA_FROZEN_FULL23_STRESS_COMPLETE",
        "data_role": "known_overlap_development_stress_not_independent_external",
        "training_or_tuning_performed": False,
        "full23_v2_mode": "unavailable_exact_noop",
        "natural_topk": natural_reports,
        "pairwise": {
            "pairs": len(labels), "positive": int(labels.sum()),
            "methods": pairwise_report,
            "boundary": "Pairwise ROC/PR is a separate known-overlap throughput axis; Full23 is not fabricated on it.",
        },
        "embedding_sha256": bank_hashes,
        "weighted_entropy_backend": weighted_entropy_backend(),
        "resource_usage": {
            "wall_seconds": time.time() - global_started,
            "peak_rss_kib_linux_ru_maxrss": (
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                if resource is not None else None
            ),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": final["status"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
