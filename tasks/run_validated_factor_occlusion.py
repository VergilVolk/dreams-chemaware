"""Causal peak-deletion audit for validated DreaMS peak factors.

For every confirmation spectrum containing a fixed target m/z, delete that
peak and compare the change in strict 10-ppm retrieval margin with deletion of
five within-spectrum peaks matched on m/z and intensity.  Factor activation at
the deleted token is intentionally not used: its disappearance would be
tautological rather than causal evidence.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, TensorDataset

import pilot_multilevel_factor_activations as multi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factor-ids", type=int, nargs="+", default=[117, 176])
    parser.add_argument("--control-peaks", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def robust_scale(values: np.ndarray) -> float:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(max(1.4826 * mad, 1e-6))


def matched_controls(
    mz: np.ndarray,
    intensity: np.ndarray,
    valid: np.ndarray,
    target_slot: int,
    target_mass: float,
    tolerance: float,
    n_controls: int,
) -> np.ndarray:
    candidate = valid & (np.abs(mz - target_mass) > tolerance)
    indices = np.flatnonzero(candidate)
    if not len(indices):
        return np.asarray([], dtype=np.int64)
    log_intensity = np.log1p(intensity * 1000.0)
    mz_scale = robust_scale(mz[valid])
    intensity_scale = robust_scale(log_intensity[valid])
    cost = (
        np.abs(mz[indices] - mz[target_slot]) / mz_scale
        + np.abs(log_intensity[indices] - log_intensity[target_slot])
        / intensity_scale
    )
    return indices[np.argsort(cost)[:n_controls]]


def infer(model, tensors: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(tensors), batch_size=batch_size, shuffle=False)
    output = []
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for (batch,) in loader:
            encoded = model(batch.to(device=device, dtype=dtype), None)[:, 0]
            output.append(encoded.float().cpu().numpy())
    return normalize(np.concatenate(output))


def bootstrap_ci(values: np.ndarray, groups: np.ndarray, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    grouped = [values[groups == group] for group in unique]
    draws = np.empty(n, dtype=float)
    for i in range(n):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        draws[i] = np.concatenate([grouped[j] for j in selected]).mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(args.catalog)
    catalog = catalog.loc[
        catalog["factor"].isin(args.factor_ids)
        & (catalog["spectral_kind"] == "fragment_mz")
    ]
    if len(catalog) != len(set(args.factor_ids)):
        raise RuntimeError("Every requested factor must be a validated fragment_mz factor")
    spectra = json.loads(
        (args.activations / "spectra.json").read_text(encoding="utf-8")
    )
    pairs = json.loads(
        (args.activations / "pairs.json").read_text(encoding="utf-8")
    )
    values = np.load(args.activations / "peak_values.npy")
    mask = np.load(args.activations / "peak_mask.npy")
    cached = normalize(np.load(args.activations / "official_precursor.npy"))
    if len(spectra) != len(values) or cached.shape[0] != len(values):
        raise RuntimeError("Activation metadata shapes do not align")

    tensors = []
    metadata = []
    for factor_row in catalog.itertuples(index=False):
        factor = int(factor_row.factor)
        target_mass = float(factor_row.mass_da)
        for spectrum_index, record in enumerate(spectra):
            valid = mask[spectrum_index]
            target_candidates = np.flatnonzero(
                valid & (np.abs(values[spectrum_index, :, 0] - target_mass) <= args.tolerance)
            )
            if not len(target_candidates):
                continue
            # One target per spectrum: closest exact mass, then strongest peak.
            candidate_error = np.abs(
                values[spectrum_index, target_candidates, 0] - target_mass
            )
            best_error = candidate_error.min()
            tied = target_candidates[candidate_error == best_error]
            target_slot = int(
                tied[np.argmax(values[spectrum_index, tied, 1])]
            )
            controls = matched_controls(
                values[spectrum_index, :, 0],
                values[spectrum_index, :, 1],
                valid,
                target_slot,
                target_mass,
                args.tolerance,
                args.control_peaks,
            )
            if len(controls) < args.control_peaks:
                continue
            base = np.zeros((101, 2), dtype=np.float32)
            base[0] = [float(record["precursor_mz"]), 1.1]
            base[1:] = values[spectrum_index]

            def append(kind: str, removed_slot: int | None, control_rank: int = -1):
                tensor = base.copy()
                if removed_slot is not None:
                    tensor[1 + removed_slot] = 0
                    # Reproduce preprocessing after deleting a raw peak: the
                    # remaining fragment intensities are normalized to their
                    # new base peak.  The precursor token remains fixed at 1.1.
                    remaining_max = float(tensor[1:, 1].max())
                    if remaining_max > 0:
                        tensor[1:, 1] /= remaining_max
                tensors.append(torch.from_numpy(tensor))
                pair_id = spectrum_index // 2
                view = spectrum_index % 2
                metadata.append({
                    "factor": factor,
                    "target_mass": target_mass,
                    "spectrum_index": spectrum_index,
                    "pair_id": pair_id,
                    "query_view": view,
                    "ik14": record["ik14"],
                    "variant": kind,
                    "control_rank": control_rank,
                    "removed_slot": removed_slot,
                    "removed_mz": None if removed_slot is None else float(values[spectrum_index, removed_slot, 0]),
                    "removed_intensity": None if removed_slot is None else float(values[spectrum_index, removed_slot, 1]),
                })

            append("original", None)
            append("target", target_slot)
            for rank, control in enumerate(controls):
                append("matched_control", int(control), rank)
    if not tensors:
        raise RuntimeError("No occlusion cases were constructed")

    raw_package = multi.torch_load_compat(multi.DEFAULT_RAW, map_location="cpu")
    official_package = multi.torch_load_compat(
        multi.DEFAULT_OFFICIAL, map_location="cpu"
    )
    device = torch.device(args.device)
    model = multi.reconstruct_backbone(
        raw_package, multi.official_backbone_state(official_package), device
    )
    del raw_package, official_package
    gc.collect()
    encoded = infer(model, torch.stack(tensors), args.batch_size, device)
    del model
    gc.collect()

    rows = []
    for item, embedding in zip(metadata, encoded):
        pair_id = item["pair_id"]
        view = item["query_view"]
        positive = cached[2 * pair_id + (1 - view)]
        negative_candidates = []
        for negative_pair in pairs[pair_id]["negative_pair_ids"]:
            negative_candidates.extend(
                [cached[2 * negative_pair], cached[2 * negative_pair + 1]]
            )
        negative_similarity = max(float(embedding @ value) for value in negative_candidates)
        positive_similarity = float(embedding @ positive)
        baseline_query = cached[item["spectrum_index"]]
        row = dict(item)
        row.update({
            "positive_similarity": positive_similarity,
            "best_10ppm_negative_similarity": negative_similarity,
            "retrieval_margin": positive_similarity - negative_similarity,
            "cosine_to_original_cached_embedding": float(embedding @ baseline_query),
        })
        rows.append(row)
    frame = pd.DataFrame(rows)
    summaries = []
    for factor, subset in frame.groupby("factor"):
        original = subset.loc[subset["variant"] == "original"].set_index("spectrum_index")
        target = subset.loc[subset["variant"] == "target"].set_index("spectrum_index")
        controls = subset.loc[subset["variant"] == "matched_control"].groupby("spectrum_index").agg(
            control_margin=("retrieval_margin", "median"),
            control_cosine=("cosine_to_original_cached_embedding", "median"),
        )
        common = original.index.intersection(target.index).intersection(controls.index)
        table = pd.DataFrame(index=common)
        table["ik14"] = original.loc[common, "ik14"]
        table["target_margin_drop"] = (
            original.loc[common, "retrieval_margin"]
            - target.loc[common, "retrieval_margin"]
        )
        table["control_margin_drop"] = (
            original.loc[common, "retrieval_margin"]
            - controls.loc[common, "control_margin"]
        )
        table["selective_margin_drop"] = (
            table["target_margin_drop"] - table["control_margin_drop"]
        )
        table["target_embedding_shift"] = 1 - target.loc[
            common, "cosine_to_original_cached_embedding"
        ]
        table["control_embedding_shift"] = 1 - controls.loc[
            common, "control_cosine"
        ]
        table["selective_embedding_shift"] = (
            table["target_embedding_shift"] - table["control_embedding_shift"]
        )
        table.to_csv(args.output_dir / f"factor_{int(factor)}_paired_effects.csv")
        difference = table["selective_margin_drop"].to_numpy(float)
        shift = table["selective_embedding_shift"].to_numpy(float)
        margin_wilcoxon = wilcoxon(difference, alternative="greater", zero_method="zsplit")
        shift_wilcoxon = wilcoxon(shift, alternative="greater", zero_method="zsplit")
        summaries.append({
            "factor": int(factor),
            "target_mass": float(subset["target_mass"].iloc[0]),
            "spectra": len(table),
            "unique_molecules": int(table["ik14"].nunique()),
            "target_margin_drop_mean": float(table["target_margin_drop"].mean()),
            "matched_control_margin_drop_mean": float(table["control_margin_drop"].mean()),
            "selective_margin_drop_mean": float(difference.mean()),
            "selective_margin_drop_molecule_bootstrap_ci95": bootstrap_ci(
                difference, table["ik14"].to_numpy(), args.bootstrap, args.seed + int(factor)
            ),
            "selective_margin_drop_wilcoxon_p": float(margin_wilcoxon.pvalue),
            "target_embedding_shift_mean": float(table["target_embedding_shift"].mean()),
            "matched_control_embedding_shift_mean": float(table["control_embedding_shift"].mean()),
            "selective_embedding_shift_mean": float(shift.mean()),
            "selective_embedding_shift_molecule_bootstrap_ci95": bootstrap_ci(
                shift, table["ik14"].to_numpy(), args.bootstrap, args.seed + 1000 + int(factor)
            ),
            "selective_embedding_shift_wilcoxon_p": float(shift_wilcoxon.pvalue),
            "original_recompute_cosine_min": float(
                original.loc[common, "cosine_to_original_cached_embedding"].min()
            ),
            "original_recompute_cosine_median": float(
                original.loc[common, "cosine_to_original_cached_embedding"].median()
            ),
        })
    frame.to_csv(args.output_dir / "all_variants.csv", index=False)
    report = {
        "status": "validated_factor_peak_occlusion",
        "checkpoint": "official DreaMS embedding checkpoint",
        "control": f"Median of {args.control_peaks} within-spectrum peaks matched on robust-scaled m/z and log intensity.",
        "causal_endpoint": "Selective change in global embedding and strict 10-ppm retrieval margin; deleted-token factor loss is not counted because it is tautological.",
        "factors": summaries,
        "claim_rule": "Causal support requires positive selective effect, molecule-bootstrap CI above zero, and one-sided Wilcoxon p <= 0.05.",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
