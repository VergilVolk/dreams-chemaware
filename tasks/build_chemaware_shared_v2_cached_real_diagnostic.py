"""Convert the cached mass-dense discovery cohort into a ChemAware-v2 diagnostic.

This is deliberately *not* a benchmark.  The source cohort was selected for
mass-dense molecular pairs, and candidates are restricted to spectra present
in that cohort rather than the complete MassSpecGym gallery.  Its purpose is
to exercise the deployable shared adapter on substantially more real, hard
candidate groups than the tiny CPU pilot while preserving the exact v2 graph
and cache contracts.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_shared_v2_core import TOKEN_STATUS  # noqa: E402
from e1_checkpoint_io import official_head_state  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
import pilot_multilevel_factor_activations as multi  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data/validation/mass_dense_all_peak_discovery",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--official-checkpoint",
        type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/validation/chemaware_shared_v2_cached_real_diagnostic",
    )
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--minimum-molecules", type=int, default=2)
    parser.add_argument("--head-batch-size", type=int, default=256)
    return parser.parse_args()


def read_rows(handle: h5py.File, key: str, rows: np.ndarray) -> np.ndarray:
    order = np.argsort(rows)
    inverse = np.argsort(order)
    dataset = handle[key]
    if h5py.check_string_dtype(dataset.dtype) is not None:
        return np.asarray(dataset.asstr()[rows[order]][inverse], dtype=str)
    return np.asarray(dataset[rows[order]][inverse])


def official_embeddings(
    precursor_tokens: np.ndarray,
    checkpoint: Path,
    batch_size: int,
) -> np.ndarray:
    package = multi.torch_load_compat(checkpoint, map_location="cpu")
    state = official_head_state(package)
    dimension = int(precursor_tokens.shape[1])
    head = torch.nn.Linear(dimension, dimension, bias=True)
    head.load_state_dict(state, strict=True)
    head.eval()
    blocks = []
    with torch.inference_mode():
        for left in range(0, len(precursor_tokens), batch_size):
            values = torch.from_numpy(
                np.asarray(
                    precursor_tokens[left:left + batch_size], dtype=np.float32
                ).copy()
            )
            blocks.append(F.normalize(head(values), dim=-1).cpu().numpy())
    return np.concatenate(blocks).astype(np.float32, copy=False)


def main() -> None:
    args = arguments()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic: {args.output_root}")
    if args.ppm <= 0 or args.minimum_molecules < 2 or args.head_batch_size < 1:
        raise ValueError("invalid diagnostic configuration")
    required = (
        args.source_dir / "report.json",
        args.source_dir / "rows.npy",
        args.source_dir / "official_precursor.npy",
        args.source_dir / "official_peak.npy",
        args.source_dir / "peak_values.npy",
        args.source_dir / "peak_mask.npy",
        args.data,
        args.official_checkpoint,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    source_report = json.loads(
        (args.source_dir / "report.json").read_text(encoding="utf-8")
    )
    if source_report.get("status") != "mass_dense_all_peak_tokens":
        raise RuntimeError("wrong source cache status")
    source_split = str(source_report.get("config", {}).get("split"))
    if source_split not in {"discovery", "confirmation"}:
        raise RuntimeError("source cache is neither discovery nor confirmation")
    if int(source_report.get("config", {}).get("layer", -1)) != 7:
        raise RuntimeError("source peak tokens are not from the final audited layer")

    source_rows = np.asarray(np.load(args.source_dir / "rows.npy"), dtype=np.int64)
    precursor_tokens = np.load(args.source_dir / "official_precursor.npy", mmap_mode="r")
    peak_tokens = np.load(args.source_dir / "official_peak.npy", mmap_mode="r")
    peak_values = np.load(args.source_dir / "peak_values.npy", mmap_mode="r")
    peak_mask = np.load(args.source_dir / "peak_mask.npy", mmap_mode="r")
    if (
        precursor_tokens.shape != (len(source_rows), 1024)
        or peak_tokens.shape[:2] != peak_mask.shape
        or peak_tokens.shape[0] != len(source_rows)
        or peak_tokens.shape[2] != 1024
        or peak_values.shape != peak_mask.shape + (2,)
    ):
        raise RuntimeError("malformed source activation cache")
    embeddings = official_embeddings(
        precursor_tokens, args.official_checkpoint, args.head_batch_size
    )

    with h5py.File(args.data, "r") as handle:
        precursor = read_rows(handle, "precursor_mz", source_rows).astype(np.float64)
        adduct = read_rows(handle, "adduct", source_rows).astype(str)
        inchikey = read_rows(handle, "INCHIKEY", source_rows).astype(str)
        formula = read_rows(handle, "FORMULA", source_rows).astype(str)
        fold = read_rows(handle, "fold", source_rows).astype(str)
    if not np.all(fold == "val"):
        raise RuntimeError("mass-dense diagnostic source is not entirely validation fold")
    ik14 = np.char.partition(inchikey, "-")[:, 0]

    mass_order = np.argsort(precursor, kind="stable")
    sorted_mass = precursor[mass_order]
    candidates: dict[int, np.ndarray] = {}
    for query in range(len(source_rows)):
        tolerance = float(precursor[query]) * args.ppm * 1e-6
        left = np.searchsorted(sorted_mass, precursor[query] - tolerance, side="left")
        right = np.searchsorted(sorted_mass, precursor[query] + tolerance, side="right")
        positions = mass_order[left:right]
        positions = positions[(positions != query) & (adduct[positions] == adduct[query])]
        identities = np.unique(ik14[positions])
        if ik14[query] in identities and len(identities) >= args.minimum_molecules:
            candidates[query] = positions.astype(np.int64)
    selected = np.asarray(sorted(candidates), dtype=np.int64)
    if len(selected) < 50:
        raise RuntimeError(f"only {len(selected)} eligible diagnostic queries")

    pair_features: list[list[float]] = []
    pair_rows: list[int] = []
    molecule_ptr = [0]
    molecule_label: list[int] = []
    molecule_identity: list[str] = []
    molecule_formula: list[str] = []
    molecule_grade: list[int] = []
    query_ptr = [0]
    query_has_near: list[bool] = []
    ranks: list[int] = []
    candidate_counts: list[int] = []
    for query in selected:
        grouped: dict[str, list[int]] = {}
        for position in candidates[int(query)]:
            grouped.setdefault(str(ik14[position]), []).append(int(position))
        positive = str(ik14[query])
        positive_score = max(
            float(embeddings[query] @ embeddings[position])
            for position in grouped[positive]
        )
        negatives = sorted(
            (identity for identity in grouped if identity != positive),
            key=lambda identity: (
                -max(
                    float(embeddings[query] @ embeddings[position])
                    for position in grouped[identity]
                ),
                identity,
            ),
        )
        ordered = [positive] + negatives
        negative_scores = []
        near = False
        for identity in ordered:
            identity_positions = sorted(
                grouped[identity],
                key=lambda position: -float(embeddings[query] @ embeddings[position]),
            )
            identity_score = -np.inf
            for position in identity_positions:
                score = float(embeddings[query] @ embeddings[position])
                identity_score = max(identity_score, score)
                pair_rows.append(int(source_rows[position]))
                pair_features.append([score])
            molecule_ptr.append(len(pair_rows))
            molecule_label.append(int(identity == positive))
            molecule_identity.append(identity)
            molecule_formula.append(str(formula[identity_positions[0]]))
            molecule_grade.append(-2 if identity == positive else -1)
            if identity != positive:
                negative_scores.append(identity_score)
                near |= str(formula[identity_positions[0]]) == str(formula[query])
        query_ptr.append(len(molecule_label))
        query_has_near.append(near)
        ranks.append(1 + int(np.sum(np.asarray(negative_scores) >= positive_score)))
        candidate_counts.append(len(ordered))

    reachable_rows = np.unique(
        np.concatenate((source_rows[selected], np.asarray(pair_rows, dtype=np.int64)))
    )
    source_position = {int(row): i for i, row in enumerate(source_rows)}
    reachable_position = np.asarray(
        [source_position[int(row)] for row in reachable_rows], dtype=np.int64
    )
    ranks_array = np.asarray(ranks, dtype=np.int32)
    near_array = np.asarray(query_has_near, dtype=bool)

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".chemaware_cached_real_", dir=args.output_root.parent)
    )
    try:
        graph_path = staging / "graph.npz"
        np.savez_compressed(
            graph_path,
            feature_names=np.asarray(["dreams_similarity"], dtype=object),
            features=np.asarray(pair_features, dtype=np.float32),
            pair_candidate_row=np.asarray(pair_rows, dtype=np.int64),
            query_ptr=np.asarray(query_ptr, dtype=np.int64),
            molecule_ptr=np.asarray(molecule_ptr, dtype=np.int64),
            molecule_label=np.asarray(molecule_label, dtype=np.int8),
            molecule_ik14=np.asarray(molecule_identity, dtype=object),
            molecule_formula=np.asarray(molecule_formula, dtype=object),
            molecule_mces_grade=np.asarray(molecule_grade, dtype=np.int8),
            query_row=source_rows[selected],
            query_ik14=ik14[selected].astype(object),
            query_formula=formula[selected].astype(object),
            query_has_near=near_array,
        )
        token_dir = staging / "tokens"
        token_dir.mkdir()
        np.save(token_dir / "rows.npy", reachable_rows)
        np.save(
            token_dir / "tokens_f16.npy",
            np.asarray(peak_tokens[reachable_position], dtype=np.float16),
        )
        np.save(
            token_dir / "mz_f32.npy",
            np.asarray(peak_values[reachable_position, :, 0], dtype=np.float32),
        )
        np.save(
            token_dir / "intensity_f32.npy",
            np.asarray(peak_values[reachable_position, :, 1], dtype=np.float32),
        )
        np.save(
            token_dir / "valid.npy",
            np.asarray(peak_mask[reachable_position], dtype=bool),
        )
        np.save(
            token_dir / "precursor_mz_f32.npy",
            np.asarray(precursor[reachable_position], dtype=np.float32),
        )
        np.save(
            token_dir / "official_embeddings_f32.npy",
            np.asarray(embeddings[reachable_position], dtype=np.float32),
        )
        token_report = {
            "status": TOKEN_STATUS,
            "formal": False,
            "spectra": int(len(reachable_rows)),
            "source": (
                f"cached real mass-dense {source_split} spectra and official "
                "DreaMS layer-7 tokens"
            ),
            "candidate_inputs_used": False,
            "provenance": {
                "graph_sha256": sha256_file(graph_path),
                "source_report_sha256": sha256_file(args.source_dir / "report.json"),
                "hdf5_sha256": sha256_file(args.data),
                "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            },
        }
        (token_dir / "report.json").write_text(
            json.dumps(token_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = {
            "status": "chemaware_shared_v2_cached_real_diagnostic_complete",
            "formal": False,
            "queries": int(len(selected)),
            "query_identities": int(len(np.unique(ik14[selected]))),
            "query_formulas": int(len(np.unique(formula[selected]))),
            "reachable_spectra": int(len(reachable_rows)),
            "candidate_molecules": int(len(molecule_label)),
            "candidate_molecules_median": float(np.median(candidate_counts)),
            "candidate_molecules_max": int(np.max(candidate_counts)),
            "baseline_recall1": float(np.mean(ranks_array == 1)),
            "baseline_errors": int(np.sum(ranks_array != 1)),
            "near_queries": int(np.sum(near_array)),
            "near_baseline_recall1": (
                float(np.mean(ranks_array[near_array] == 1)) if np.any(near_array) else None
            ),
            "candidate_protocol": (
                f"{args.ppm:g}ppm same-adduct self-row-excluded; molecule max; "
                f"gallery restricted to cached mass-dense {source_split} cohort"
            ),
            "selection_bias": (
                f"source {source_split} cohort preselected for mass-dense paired molecules; "
                "validation fold; "
                "not representative of the frozen full candidate graph"
            ),
            "P3_seal": False,
            "claim_limit": (
                "mechanism and failure diagnosis only; no benchmark, no external, and no "
                "chemical-attribution claim"
            ),
        }
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging.replace(args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
