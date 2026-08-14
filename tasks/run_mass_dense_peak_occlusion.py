"""Case-level peak occlusion audit for strict mass-dense DreaMS retrieval."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import evaluate_mass_dense_factor_retrieval as retrieval
import pilot_multilevel_factor_activations as multi
from pilot_paired_layer_cka import preprocess_spectrum


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTIVATIONS = ROOT / "data/validation/mass_dense_factor_confirmation"
DEFAULT_TAXONOMY = ROOT / "data/validation/mass_dense_failure_taxonomy/confirmation_taxonomy.csv"
DEFAULT_HDF5 = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OUTPUT = ROOT / "data/validation/mass_dense_peak_occlusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases-per-group", type=int, default=3)
    parser.add_argument("--single-peaks", type=int, default=32)
    parser.add_argument("--random-repeats", type=int, default=12)
    parser.add_argument("--mask-fraction", type=float, default=0.2)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def select_cases(frame: pd.DataFrame, per_group: int) -> pd.DataFrame:
    strategies = {
        "raw_wrong_official_correct": ("official_margin_gain", False),
        "both_wrong": ("official_margin", True),
        "official_correct_factor_wrong": ("factor_margin", True),
        "both_correct": ("official_margin", False),
    }
    selected = []
    used_pairs = set()
    for group, (column, ascending) in strategies.items():
        subset = frame.loc[frame["group"] == group].sort_values(column, ascending=ascending)
        count = 0
        for row in subset.itertuples(index=False):
            if row.pair_id in used_pairs:
                continue
            selected.append(row._asdict())
            used_pairs.add(row.pair_id)
            count += 1
            if count >= per_group:
                break
    return pd.DataFrame(selected)


def valid_peak_indices(raw: np.ndarray) -> np.ndarray:
    return np.flatnonzero(
        np.isfinite(raw[0]) & np.isfinite(raw[1]) & (raw[0] > 0) & (raw[1] > 0)
    )


def masked_tensor(raw: np.ndarray, precursor_mz: float, indices, n_highest: int):
    modified = np.asarray(raw).copy()
    if len(indices):
        modified[:, np.asarray(indices, dtype=int)] = 0
    return preprocess_spectrum(modified, precursor_mz, n_highest)


def build_variants(cases: pd.DataFrame, hdf5_path: Path, args: argparse.Namespace):
    tensors, metadata = [], []
    rng = np.random.RandomState(args.seed)
    with h5py.File(hdf5_path, "r") as handle:
        for case_id, case in enumerate(cases.itertuples(index=False)):
            raw = np.asarray(handle["spectrum"][int(case.query_row)])
            precursor = float(handle["precursor_mz"][int(case.query_row)])
            valid = valid_peak_indices(raw)
            ranked = valid[np.argsort(raw[1, valid])[::-1]]
            represented = ranked[: args.n_highest_peaks]
            single = represented[: min(args.single_peaks, len(represented))]
            mask_n = max(1, int(round(args.mask_fraction * len(represented))))

            def append(kind: str, removed, repeat: int = 0):
                removed = np.asarray(removed, dtype=int)
                tensors.append(masked_tensor(raw, precursor, removed, args.n_highest_peaks))
                metadata.append({
                    "case_id": case_id,
                    "group": case.group,
                    "pair_id": int(case.pair_id),
                    "query_view": int(case.query_view),
                    "query_row": int(case.query_row),
                    "positive_row": int(case.positive_row),
                    "negative_pair_id": int(case.official_best_negative_pair_id),
                    "negative_row": int(case.official_best_negative_row),
                    "variant_kind": kind,
                    "repeat": repeat,
                    "removed_count": int(len(removed)),
                    "removed_indices": removed.tolist(),
                    "removed_mz": raw[0, removed].tolist(),
                    "removed_relative_intensity": (
                        raw[1, removed] / max(float(raw[1, valid].max()), 1e-12)
                    ).tolist() if len(removed) else [],
                })

            append("original", [])
            for peak_index in single:
                append("single_peak", [peak_index])
            append("top_intensity_fraction", represented[:mask_n])
            append("low_intensity_fraction", represented[-mask_n:])
            for repeat in range(args.random_repeats):
                removed = rng.choice(represented, size=mask_n, replace=False)
                append("random_fraction", removed, repeat)
    return torch.stack(tensors), metadata


def infer(model, tensors: torch.Tensor, batch_size: int, device: torch.device):
    loader = DataLoader(TensorDataset(tensors), batch_size=batch_size, shuffle=False)
    outputs = []
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for (batch,) in loader:
            encoded = model(batch.to(device=device, dtype=dtype), None)[:, 0]
            outputs.append(encoded.float().cpu().numpy())
    values = np.concatenate(outputs)
    return retrieval.normalize(values)


def locate_view(pair_manifest: list[dict], pair_id: int, row: int) -> int:
    rows = pair_manifest[pair_id]["rows"]
    if row not in rows:
        raise ValueError(f"Row {row} is not part of pair {pair_id}")
    return rows.index(row)


def score_variants(
    model_name: str,
    encoded: np.ndarray,
    metadata: list[dict],
    cached: np.ndarray,
    pair_manifest: list[dict],
) -> list[dict]:
    cached = retrieval.normalize(cached)
    rows = []
    original_margin = {}
    for index, item in enumerate(metadata):
        pair_id = item["pair_id"]
        query_view = item["query_view"]
        negative_pair = item["negative_pair_id"]
        negative_view = locate_view(pair_manifest, negative_pair, item["negative_row"])
        positive = cached[pair_id, 1 - query_view]
        negative = cached[negative_pair, negative_view]
        margin = float(encoded[index] @ positive - encoded[index] @ negative)
        if item["variant_kind"] == "original":
            original_margin[item["case_id"]] = margin
            cached_query = cached[pair_id, query_view]
            recompute_cosine = float(encoded[index] @ cached_query)
        else:
            recompute_cosine = None
        row = dict(item)
        row.update({
            "model": model_name,
            "margin": margin,
            "original_recompute_cosine": recompute_cosine,
        })
        rows.append(row)
    for row in rows:
        row["margin_drop"] = original_margin[row["case_id"]] - row["margin"]
    return rows


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = pd.read_csv(args.taxonomy)
    cases = select_cases(taxonomy, args.cases_per_group)
    cases.to_csv(args.output_dir / "selected_cases.csv", index=False)
    tensors, metadata = build_variants(cases, args.hdf5, args)
    pair_manifest = json.loads(
        (args.activations / "pairs.json").read_text(encoding="utf-8")
    )
    raw_cached = retrieval.load_activations(args.activations, "raw", 7)
    official_cached = retrieval.load_activations(args.activations, "official", 7)
    device = torch.device(args.device)
    all_rows = []

    raw_package = multi.torch_load_compat(multi.DEFAULT_RAW, map_location="cpu")
    print(f"variants={len(tensors)}; running raw SSL", flush=True)
    raw_model = multi.reconstruct_backbone(raw_package, raw_package["state_dict"], device)
    raw_encoded = infer(raw_model, tensors, args.batch_size, device)
    all_rows.extend(score_variants("raw_ssl", raw_encoded, metadata, raw_cached, pair_manifest))
    del raw_model
    gc.collect()

    official_package = multi.torch_load_compat(multi.DEFAULT_OFFICIAL, map_location="cpu")
    print("running official fine-tuned", flush=True)
    official_model = multi.reconstruct_backbone(
        raw_package, multi.official_backbone_state(official_package), device
    )
    official_encoded = infer(official_model, tensors, args.batch_size, device)
    all_rows.extend(score_variants(
        "official_finetuned", official_encoded, metadata, official_cached, pair_manifest
    ))
    del official_model, official_package, raw_package
    gc.collect()

    frame = pd.DataFrame(all_rows)
    frame.to_csv(args.output_dir / "occlusion_results.csv", index=False)
    original_check = frame.loc[frame["variant_kind"] == "original"].groupby("model")[
        "original_recompute_cosine"
    ].agg(["min", "median", "mean"])
    grouped = frame.loc[frame["variant_kind"] != "original"].groupby(
        ["model", "group", "variant_kind"]
    )["margin_drop"].agg(["count", "mean", "median", "std"]).reset_index()
    grouped.to_csv(args.output_dir / "occlusion_summary.csv", index=False)
    report = {
        "status": "mass_dense_peak_occlusion",
        "n_cases": len(cases),
        "n_variants_per_model": len(metadata),
        "groups": cases["group"].value_counts().to_dict(),
        "original_recompute_cosine": original_check.to_dict(orient="index"),
        "interpretation_limit": (
            "Single-case occlusion localizes model dependence on peaks. It does not "
            "prove a fragmentation mechanism or justify adding a rule without replicated "
            "m/z or neutral-loss enrichment in independent molecules."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
