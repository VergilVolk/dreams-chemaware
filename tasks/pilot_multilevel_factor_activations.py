"""Balanced precursor/peak-token activation pilot for DreaMS model diffing.

The cohort contains two spectra per molecule. Pairs are chosen to maximize
differences in instrument, adduct, and collision energy, then sampled across
condition-difference categories and precursor-mass bins. The raw SSL and
official fine-tuned backbones see the exact same preprocessed tensors.

This stage only prepares and audits paired activations. It does not train a
Crosscoder and does not use chemical rules as labels.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import MethodType

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from e1_checkpoint_io import (  # noqa: E402
    checkpoint_kind,
    official_backbone_state,
    torch_load_compat,
)
from pilot_paired_layer_cka import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_OFFICIAL,
    DEFAULT_RAW,
    LightweightDreaMS,
    SpectrumRows,
    linear_cka,
)


DEFAULT_OUTPUT = ROOT / "data/validation/multilevel_factor_pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balanced DreaMS precursor/peak-token activation pilot"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--raw-checkpoint", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--exclude-pairs",
        type=Path,
        default=None,
        help="pairs.json from a previous run; all listed IK14 molecules are excluded",
    )
    parser.add_argument("--fold", default="val")
    parser.add_argument("--n-molecules", type=int, default=500)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 6, 7])
    parser.add_argument("--peak-tokens", type=int, default=24)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def pair_category(
    i: int,
    j: int,
    instrument: np.ndarray,
    adduct: np.ndarray,
    collision_energy: np.ndarray,
) -> tuple[bool, bool, bool]:
    inst_diff = instrument[i] != instrument[j]
    adduct_diff = adduct[i] != adduct[j]
    cei, cej = collision_energy[i], collision_energy[j]
    ce_diff = bool(np.isfinite(cei) and np.isfinite(cej) and abs(cei - cej) >= 10)
    return inst_diff, adduct_diff, ce_diff


def pair_score(
    i: int,
    j: int,
    instrument: np.ndarray,
    adduct: np.ndarray,
    collision_energy: np.ndarray,
) -> float:
    inst_diff, adduct_diff, ce_diff = pair_category(
        i, j, instrument, adduct, collision_energy
    )
    score = 5.0 * inst_diff + 4.0 * adduct_diff + 3.0 * ce_diff
    cei, cej = collision_energy[i], collision_energy[j]
    if np.isfinite(cei) and np.isfinite(cej):
        score += min(abs(cei - cej), 80.0) / 80.0
    elif np.isfinite(cei) != np.isfinite(cej):
        score += 0.25
    return score


def candidate_members(
    members: list[int],
    instrument: np.ndarray,
    adduct: np.ndarray,
    collision_energy: np.ndarray,
    rng: np.random.RandomState,
) -> list[int]:
    """Limit very large replicate groups without losing condition extremes."""
    if len(members) <= 32:
        return members
    selected: set[int] = set()
    members_arr = np.asarray(members, dtype=np.int64)
    for inst in np.unique(instrument[members_arr]):
        for add in np.unique(adduct[members_arr]):
            subset = members_arr[
                (instrument[members_arr] == inst) & (adduct[members_arr] == add)
            ]
            if not len(subset):
                continue
            finite = subset[np.isfinite(collision_energy[subset])]
            if len(finite):
                selected.add(int(finite[np.argmin(collision_energy[finite])]))
                selected.add(int(finite[np.argmax(collision_energy[finite])]))
            selected.add(int(subset[0]))
    remaining = [m for m in members if m not in selected]
    rng.shuffle(remaining)
    selected.update(remaining[: max(0, 32 - len(selected))])
    return sorted(selected)


def build_balanced_pair_cohort(
    data_path: Path,
    fold: str,
    n_molecules: int,
    seed: int,
    excluded_ik14: set[str] | None = None,
) -> tuple[np.ndarray, list[dict], dict]:
    rng = np.random.RandomState(seed)
    excluded_ik14 = excluded_ik14 or set()
    with h5py.File(data_path, "r") as handle:
        folds = handle["fold"].asstr()[:]
        fold_rows = np.flatnonzero(folds == fold)
        if not len(fold_rows):
            raise ValueError(f"No spectra found for fold={fold!r}")
        # DataFormatA is a domain definition, not a peak-trimming operation.
        # Apply its spectrum-level checks before cohort balancing so model
        # differences are not driven by inputs outside the DreaMS training
        # range. Padding zeros are excluded from peak-count/intensity checks.
        spectra = np.asarray(handle["spectrum"][fold_rows])
        mzs = spectra[:, 0, :]
        intensities = spectra[:, 1, :]
        valid_peak = (
            np.isfinite(mzs) & np.isfinite(intensities)
            & (mzs > 0) & (intensities > 0)
        )
        peak_count = valid_peak.sum(axis=1)
        max_mz = np.where(valid_peak, mzs, -np.inf).max(axis=1)
        max_intensity = np.where(valid_peak, intensities, -np.inf).max(axis=1)
        min_intensity = np.where(valid_peak, intensities, np.inf).min(axis=1)
        amplitude = np.divide(
            max_intensity,
            min_intensity,
            out=np.zeros_like(max_intensity),
            where=np.isfinite(max_intensity) & np.isfinite(min_intensity)
            & (min_intensity > 0),
        )
        relative = np.divide(
            intensities,
            max_intensity[:, None],
            out=np.zeros_like(intensities),
            where=np.isfinite(max_intensity[:, None])
            & (max_intensity[:, None] > 0),
        )
        high_peak_count = ((relative > 0.1) & valid_peak).sum(axis=1)
        fold_precursor = np.asarray(handle["precursor_mz"][fold_rows])
        precursor_valid = (
            np.isfinite(fold_precursor)
            & (fold_precursor > 0)
            & (fold_precursor <= 1000)
        )
        quality_mask = (
            precursor_valid
            & (peak_count >= 3)
            & (peak_count <= 128)
            & (max_mz <= 1000)
            & (amplitude >= 20)
            & (high_peak_count >= 3)
        )
        quality_audit = {
            "fold_spectra": int(len(fold_rows)),
            "quality_valid_spectra": int(quality_mask.sum()),
            "quality_valid_fraction": float(quality_mask.mean()),
            "excluded_precursor": int((~precursor_valid).sum()),
            "excluded_peak_count": int(((peak_count < 3) | (peak_count > 128)).sum()),
            "excluded_max_mz": int((max_mz > 1000).sum()),
            "excluded_intensity_amplitude": int((amplitude < 20).sum()),
            "excluded_high_peak_count": int((high_peak_count < 3).sum()),
            "note": "Exclusion counts overlap; DataFormatA-like spectrum-level QC.",
        }
        global_rows = fold_rows[quality_mask]
        ik14 = np.asarray([value[:14] for value in handle["INCHIKEY"].asstr()[global_rows]])
        instrument = handle["INSTRUMENT_TYPE"].asstr()[global_rows]
        adduct = handle["adduct"].asstr()[global_rows]
        collision_energy = np.asarray(handle["COLLISION_ENERGY"][global_rows])
        precursor_mz = np.asarray(handle["precursor_mz"][global_rows])

        groups: dict[str, list[int]] = defaultdict(list)
        for local_row, key in enumerate(ik14):
            if key in excluded_ik14:
                continue
            if np.isfinite(precursor_mz[local_row]) and 0 < precursor_mz[local_row] <= 1000:
                groups[key].append(local_row)

        best_pairs = []
        for key, members in groups.items():
            if len(members) < 2:
                continue
            candidates = candidate_members(
                members, instrument, adduct, collision_energy, rng
            )
            scored = []
            for i, j in itertools.combinations(candidates, 2):
                scored.append((
                    pair_score(i, j, instrument, adduct, collision_energy),
                    rng.uniform(0, 1),
                    i,
                    j,
                ))
            _, _, i, j = max(scored)
            category = pair_category(i, j, instrument, adduct, collision_energy)
            pair_mz = float(np.nanmean([precursor_mz[i], precursor_mz[j]]))
            best_pairs.append({
                "ik14": key,
                "local_i": i,
                "local_j": j,
                "category": category,
                "pair_mz": pair_mz,
            })

        if len(best_pairs) < n_molecules:
            raise ValueError(
                f"Only {len(best_pairs):,} molecules have two valid spectra; "
                f"requested {n_molecules:,}"
            )

        mass_edges = np.quantile(
            [pair["pair_mz"] for pair in best_pairs], np.linspace(0, 1, 9)
        )
        strata: dict[tuple, list[dict]] = defaultdict(list)
        for pair in best_pairs:
            mass_bin = int(np.clip(
                np.digitize(pair["pair_mz"], mass_edges[1:-1]), 0, 7
            ))
            pair["mass_bin"] = mass_bin
            strata[tuple(pair["category"]) + (mass_bin,)].append(pair)
        for values in strata.values():
            rng.shuffle(values)

        selected: list[dict] = []
        ordered_strata = sorted(strata)
        while len(selected) < n_molecules:
            progressed = False
            rng.shuffle(ordered_strata)
            for stratum in ordered_strata:
                if strata[stratum] and len(selected) < n_molecules:
                    selected.append(strata[stratum].pop())
                    progressed = True
            if not progressed:
                break

        rows: list[int] = []
        pair_manifest: list[dict] = []
        for pair_id, pair in enumerate(selected):
            li, lj = pair["local_i"], pair["local_j"]
            gi, gj = int(global_rows[li]), int(global_rows[lj])
            rows.extend([gi, gj])
            category = pair["category"]
            pair_manifest.append({
                "pair_id": pair_id,
                "ik14": pair["ik14"],
                "rows": [gi, gj],
                "mass_bin": pair["mass_bin"],
                "instrument": [str(instrument[li]), str(instrument[lj])],
                "adduct": [str(adduct[li]), str(adduct[lj])],
                "collision_energy": [
                    None if not np.isfinite(collision_energy[li]) else float(collision_energy[li]),
                    None if not np.isfinite(collision_energy[lj]) else float(collision_energy[lj]),
                ],
                "precursor_mz": [float(precursor_mz[li]), float(precursor_mz[lj])],
                "condition_difference": {
                    "instrument": bool(category[0]),
                    "adduct": bool(category[1]),
                    "collision_energy_ge_10": bool(category[2]),
                },
            })

    category_counts = Counter(
        tuple(item["condition_difference"].values()) for item in pair_manifest
    )
    audit = {
        "n_spectra": 2 * len(pair_manifest),
        "n_molecules": len(pair_manifest),
        "spectra_per_molecule": 2,
        "available_multi_spectrum_molecules": len(best_pairs),
        "excluded_molecules_requested": len(excluded_ik14),
        "quality_control": quality_audit,
        "category_counts": {
            f"instrument={key[0]},adduct={key[1]},ce={key[2]}": value
            for key, value in sorted(category_counts.items())
        },
        "mass_bin_counts": dict(sorted(Counter(
            item["mass_bin"] for item in pair_manifest
        ).items())),
    }
    return np.asarray(rows, dtype=np.int64), pair_manifest, audit


def reconstruct_backbone(
    architecture_package: dict,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> LightweightDreaMS:
    from argparse import Namespace

    model_args = Namespace(**architecture_package["args"])
    model_args.dformat = Namespace(
        max_mz=float(architecture_package["args"]["max_mz"]),
        max_tbxic_stdev=float(architecture_package["args"]["max_tbxic_stdev"]),
    )
    model = LightweightDreaMS(model_args)
    forward_state = {
        key: value for key, value in state_dict.items()
        if not key.startswith(("ff_out.", "ro_out."))
    }
    model.load_state_dict(forward_state, strict=True)
    return model.eval().to(device)


def select_representative_peaks(
    spectra: torch.Tensor, n_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select high/mid/low-intensity peaks, returning full token indices."""
    batch_size = len(spectra)
    indices = torch.zeros((batch_size, n_tokens), dtype=torch.long)
    masks = torch.zeros((batch_size, n_tokens), dtype=torch.bool)
    quotas = [n_tokens // 3] * 3
    for i in range(n_tokens % 3):
        quotas[i] += 1
    for row in range(batch_size):
        valid = torch.nonzero(
            spectra[row, 1:, 0] > 0, as_tuple=False
        ).flatten()
        if not len(valid):
            continue
        ranked = valid[torch.argsort(spectra[row, 1 + valid, 1], descending=True)]
        bands = torch.tensor_split(ranked, 3)
        chosen = []
        for band, quota in zip(bands, quotas):
            if not len(band) or quota == 0:
                continue
            if len(band) <= quota:
                picked = band
            else:
                positions = torch.linspace(0, len(band) - 1, quota).round().long()
                picked = band[positions]
            chosen.extend(picked.tolist())
        # Fill unused slots from remaining peaks in intensity order.
        chosen_set = set(chosen)
        for peak in ranked.tolist():
            if len(chosen) >= n_tokens:
                break
            if peak not in chosen_set:
                chosen.append(peak)
                chosen_set.add(peak)
        chosen = sorted(chosen[:n_tokens])
        if chosen:
            count = len(chosen)
            # +1 maps peak-array indices to the full sequence after precursor.
            indices[row, :count] = torch.tensor(chosen, dtype=torch.long) + 1
            masks[row, :count] = True
    return indices, masks


def extract_multilevel(
    model: LightweightDreaMS,
    loader: DataLoader,
    device: torch.device,
    layer_numbers: list[int],
    peak_tokens: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    layer_indices = [number - 1 for number in layer_numbers]
    encoder = model.transformer_encoder
    original = encoder._layer_forward
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    current_peak_indices: torch.Tensor | None = None

    def wrapped(this, layer_index, *args, **kwargs):
        output = original(layer_index, *args, **kwargs)
        if layer_index in layer_indices:
            gather_idx = current_peak_indices.to(output.device)
            gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, output.shape[-1])
            captured[layer_index] = (
                output[:, 0, :].detach().float().cpu(),
                torch.gather(output, 1, gather_idx).detach().to(torch.float16).cpu(),
            )
        return output

    encoder._layer_forward = MethodType(wrapped, encoder)
    all_precursor, all_peaks = [], []
    all_indices, all_masks, all_peak_values = [], [], []
    started = time.time()
    model_dtype = next(model.parameters()).dtype
    try:
        with torch.inference_mode():
            for spectra in loader:
                peak_indices, peak_masks = select_representative_peaks(
                    spectra, peak_tokens
                )
                current_peak_indices = peak_indices
                captured.clear()
                spectra_device = spectra.to(device=device, dtype=model_dtype)
                final = model(spectra_device, None)
                if sorted(captured) != sorted(layer_indices):
                    raise RuntimeError(
                        f"Captured {sorted(captured)}; expected {sorted(layer_indices)}"
                    )
                # Layer 7 receives the encoder's final normalization.
                if model.n_layers - 1 in layer_indices:
                    final_layer = model.n_layers - 1
                    gather_idx = peak_indices.to(device).unsqueeze(-1).expand(
                        -1, -1, final.shape[-1]
                    )
                    captured[final_layer] = (
                        final[:, 0, :].detach().float().cpu(),
                        torch.gather(final, 1, gather_idx).detach().to(torch.float16).cpu(),
                    )
                all_precursor.append(torch.stack(
                    [captured[i][0] for i in layer_indices], dim=1
                ).numpy())
                all_peaks.append(torch.stack(
                    [captured[i][1] for i in layer_indices], dim=1
                ).numpy())
                all_indices.append(peak_indices.numpy())
                all_masks.append(peak_masks.numpy())
                batch_row = torch.arange(len(spectra)).unsqueeze(1)
                values = spectra[batch_row, peak_indices]
                values[~peak_masks] = 0
                all_peak_values.append(values.numpy())
    finally:
        encoder._layer_forward = original

    precursor = np.concatenate(all_precursor).astype(np.float32, copy=False)
    peaks = np.concatenate(all_peaks).astype(np.float16, copy=False)
    peak_indices = np.concatenate(all_indices)
    peak_masks = np.concatenate(all_masks)
    peak_values = np.concatenate(all_peak_values).astype(np.float32, copy=False)
    diagnostics = {
        "seconds": time.time() - started,
        "precursor_shape": list(precursor.shape),
        "peak_shape": list(peaks.shape),
        "finite_precursor_fraction": float(np.isfinite(precursor).mean()),
        "finite_peak_fraction": float(np.isfinite(peaks).mean()),
    }
    return precursor, peaks, peak_indices, peak_masks, peak_values, diagnostics


def mean_peak_representation(peaks: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = mask[:, None, :, None].astype(np.float32)
    summed = (peaks.astype(np.float32) * weights).sum(axis=2)
    counts = weights.sum(axis=2).clip(min=1)
    return summed / counts


def paired_peak_cosine(
    raw: np.ndarray, official: np.ndarray, mask: np.ndarray
) -> list[float]:
    results = []
    valid = mask.reshape(-1)
    for layer in range(raw.shape[1]):
        x = raw[:, layer].reshape(-1, raw.shape[-1])[valid].astype(np.float32)
        y = official[:, layer].reshape(-1, official.shape[-1])[valid].astype(np.float32)
        numerator = (x * y).sum(axis=1)
        denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        results.append(float(np.mean(numerator / np.clip(denominator, 1e-12, None))))
    return results


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but this PyTorch environment has no CUDA")
    if any(layer < 1 or layer > 7 for layer in args.layers):
        raise ValueError("--layers must contain DreaMS layer numbers from 1 to 7")
    if args.peak_tokens < 3 or args.peak_tokens > args.n_highest_peaks:
        raise ValueError("--peak-tokens must be between 3 and --n-highest-peaks")
    args.layers = sorted(set(args.layers))
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Building molecule- and condition-balanced cohort", flush=True)
    excluded_ik14: set[str] = set()
    if args.exclude_pairs is not None:
        previous_pairs = json.loads(args.exclude_pairs.read_text(encoding="utf-8"))
        excluded_ik14 = {item["ik14"] for item in previous_pairs}
        print(
            f"  excluding {len(excluded_ik14):,} molecules from {args.exclude_pairs}",
            flush=True,
        )
    rows, pair_manifest, cohort_audit = build_balanced_pair_cohort(
        args.data, args.fold, args.n_molecules, args.seed, excluded_ik14
    )
    print(json.dumps(cohort_audit, ensure_ascii=False, indent=2), flush=True)
    loader = DataLoader(
        SpectrumRows(args.data, rows, args.n_highest_peaks),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("[2/6] Loading raw SSL backbone", flush=True)
    raw_package = torch_load_compat(args.raw_checkpoint, map_location="cpu")
    if checkpoint_kind(raw_package) != "raw_ssl":
        raise ValueError("--raw-checkpoint must be raw SSL format")
    raw_model = reconstruct_backbone(
        raw_package, raw_package["state_dict"], device
    )
    raw_precursor, raw_peaks, peak_indices, peak_masks, peak_values, raw_diag = (
        extract_multilevel(
            raw_model, loader, device, args.layers, args.peak_tokens
        )
    )
    del raw_model
    gc.collect()
    print(f"  raw: {raw_diag}", flush=True)

    print("[3/6] Loading official fine-tuned backbone", flush=True)
    official_package = torch_load_compat(args.official_checkpoint, map_location="cpu")
    if checkpoint_kind(official_package) not in (
        "official_embedding", "official_embedding_slim"
    ):
        raise ValueError("--official-checkpoint must be official embedding format")
    official_model = reconstruct_backbone(
        raw_package, official_backbone_state(official_package), device
    )
    official_precursor, official_peaks, indices_2, masks_2, values_2, official_diag = (
        extract_multilevel(
            official_model, loader, device, args.layers, args.peak_tokens
        )
    )
    del official_model, official_package, raw_package
    gc.collect()
    if not (
        np.array_equal(peak_indices, indices_2)
        and np.array_equal(peak_masks, masks_2)
        and np.array_equal(peak_values, values_2)
    ):
        raise RuntimeError("Raw and official runs did not use identical peak tokens")
    print(f"  official: {official_diag}", flush=True)

    print("[4/6] Computing global and pooled-peak diagnostics", flush=True)
    precursor_cka = [
        linear_cka(raw_precursor[:, i], official_precursor[:, i])
        for i in range(len(args.layers))
    ]
    raw_peak_mean = mean_peak_representation(raw_peaks, peak_masks)
    official_peak_mean = mean_peak_representation(official_peaks, peak_masks)
    pooled_peak_cka = [
        linear_cka(raw_peak_mean[:, i], official_peak_mean[:, i])
        for i in range(len(args.layers))
    ]
    peak_cosine = paired_peak_cosine(raw_peaks, official_peaks, peak_masks)

    print("[5/6] Saving activation artifacts", flush=True)
    np.save(args.output_dir / "rows.npy", rows)
    np.save(args.output_dir / "peak_indices.npy", peak_indices)
    np.save(args.output_dir / "peak_mask.npy", peak_masks)
    np.save(args.output_dir / "peak_values.npy", peak_values)
    np.save(args.output_dir / "raw_precursor.npy", raw_precursor)
    np.save(args.output_dir / "official_precursor.npy", official_precursor)
    np.save(args.output_dir / "raw_peak.npy", raw_peaks)
    np.save(args.output_dir / "official_peak.npy", official_peaks)
    (args.output_dir / "pairs.json").write_text(
        json.dumps(pair_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "status": "multilevel_activation_pilot",
        "warning": (
            "This run validates balanced extraction and identifies candidate "
            "layers. It does not establish chemical-factor semantics."
        ),
        "config": {
            "data": str(args.data.resolve()),
            "fold": args.fold,
            "n_molecules": args.n_molecules,
            "n_spectra": 2 * args.n_molecules,
            "layers": args.layers,
            "peak_tokens_per_spectrum": args.peak_tokens,
            "n_highest_peaks": args.n_highest_peaks,
            "seed": args.seed,
            "quality_control": "DataFormatA-like spectrum-level filtering",
            "exclude_pairs": (
                None if args.exclude_pairs is None else str(args.exclude_pairs.resolve())
            ),
        },
        "cohort_audit": cohort_audit,
        "raw_diagnostics": raw_diag,
        "official_diagnostics": official_diag,
        "precursor_same_layer_cka": precursor_cka,
        "pooled_peak_same_layer_cka": pooled_peak_cka,
        "paired_peak_token_cosine": peak_cosine,
        "interpretation_limits": [
            "CKA measures representational similarity, not chemical quality.",
            "Mean-pooled peak CKA is a pipeline diagnostic, not a local-factor test.",
            "Crosscoder training must wait for independent cohort validation.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[6/6] Complete", flush=True)
    for i, layer in enumerate(args.layers):
        print(
            f"  layer {layer}: precursor CKA={precursor_cka[i]:.4f}; "
            f"pooled-peak CKA={pooled_peak_cka[i]:.4f}; "
            f"paired peak cosine={peak_cosine[i]:.4f}",
            flush=True,
        )
    print(f"  output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
