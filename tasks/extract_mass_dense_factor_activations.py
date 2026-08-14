"""Extract paired raw/official DreaMS activations for the mass-dense cohort."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import pilot_multilevel_factor_activations as multi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/validation/mass_dense_factor_cohort_split.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=multi.DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, default=multi.DEFAULT_RAW)
    parser.add_argument("--official-checkpoint", type=Path, default=multi.DEFAULT_OFFICIAL)
    parser.add_argument("--layers", type=int, nargs="+", default=[7])
    parser.add_argument("--peak-tokens", type=int, default=3)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def connected_components(units: dict[int, dict]) -> list[list[int]]:
    adjacency = {unit_id: set() for unit_id in units}
    for unit_id, unit in units.items():
        for neighbor in unit["negative_unit_ids"]:
            if neighbor in units:
                adjacency[unit_id].add(neighbor)
                adjacency[neighbor].add(unit_id)
    components = []
    seen = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), min(values)))


def select_units(manifest: dict, split: str, max_units: int) -> list[dict]:
    units = {
        int(unit["unit_id"]): unit
        for unit in manifest["units"]
        if unit["split"] == split
    }
    if max_units <= 0 or max_units >= len(units):
        return [units[key] for key in sorted(units)]
    selected_ids = []
    for component in connected_components(units):
        if selected_ids and len(selected_ids) + len(component) > max_units:
            continue
        selected_ids.extend(component)
        if len(selected_ids) >= max_units:
            break
    if not selected_ids:
        raise ValueError("No complete mass-neighbor component fits --max-units")
    return [units[key] for key in sorted(selected_ids)]


def load_metadata(data: Path, units: list[dict]) -> tuple[np.ndarray, list[dict]]:
    rows = np.asarray(
        [row for unit in units for row in unit["positive_rows"]], dtype=np.int64
    )
    order = np.argsort(rows)
    inverse = np.argsort(order)
    sorted_rows = rows[order]
    with h5py.File(data, "r") as handle:
        instruments = handle["INSTRUMENT_TYPE"].asstr()[sorted_rows][inverse]
        adducts = handle["adduct"].asstr()[sorted_rows][inverse]
        energies = np.asarray(
            handle["COLLISION_ENERGY"][sorted_rows], dtype=float
        )[inverse]
        precursor = np.asarray(
            handle["precursor_mz"][sorted_rows], dtype=float
        )[inverse]
    mass_values = np.asarray([unit["precursor_mz"] for unit in units])
    edges = np.quantile(mass_values, np.linspace(0, 1, 9))
    unit_to_pair = {int(unit["unit_id"]): i for i, unit in enumerate(units)}
    pair_manifest = []
    for pair_id, unit in enumerate(units):
        offset = 2 * pair_id
        energy_pair = energies[offset : offset + 2]
        pair_manifest.append({
            "pair_id": pair_id,
            "original_unit_id": int(unit["unit_id"]),
            "ik14": unit["ik14"],
            "rows": [int(value) for value in rows[offset : offset + 2]],
            "mass_bin": int(np.clip(
                np.digitize(unit["precursor_mz"], edges[1:-1]), 0, 7
            )),
            "instrument": instruments[offset : offset + 2].tolist(),
            "adduct": adducts[offset : offset + 2].tolist(),
            "collision_energy": [
                None if not np.isfinite(value) else float(value)
                for value in energy_pair
            ],
            "precursor_mz": [
                float(value) for value in precursor[offset : offset + 2]
            ],
            "condition_difference": {
                "instrument": bool(unit["positive_instrument_diff"]),
                "adduct": False,
                "collision_energy_ge_10": bool(
                    unit["positive_ce_diff_ge_threshold"]
                ),
            },
            "negative_pair_ids": [
                unit_to_pair[neighbor]
                for neighbor in unit["negative_unit_ids"]
                if neighbor in unit_to_pair
            ],
            "nearest_negative_ppm": unit["nearest_negative_ppm"],
        })
    if any(not pair["negative_pair_ids"] for pair in pair_manifest):
        raise RuntimeError("Selection broke a mass-neighbor component")
    return rows, pair_manifest


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is unavailable")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    units = select_units(manifest, args.split, args.max_units)
    rows, pairs = load_metadata(args.data, units)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"split={args.split}; units={len(units)}; spectra={len(rows)}; "
        f"negative links={sum(len(p['negative_pair_ids']) for p in pairs)}",
        flush=True,
    )
    loader = DataLoader(
        multi.SpectrumRows(args.data, rows, args.n_highest_peaks),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device(args.device)
    layers = sorted(set(args.layers))

    print("Loading raw SSL backbone", flush=True)
    raw_package = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    if multi.checkpoint_kind(raw_package) != "raw_ssl":
        raise ValueError("--raw-checkpoint must use raw SSL format")
    raw_model = multi.reconstruct_backbone(
        raw_package, raw_package["state_dict"], device
    )
    raw, raw_peak, peak_indices, peak_mask, peak_values, raw_diag = (
        multi.extract_multilevel(
            raw_model, loader, device, layers, args.peak_tokens
        )
    )
    del raw_model
    gc.collect()

    print("Loading official fine-tuned backbone", flush=True)
    official_package = multi.torch_load_compat(
        args.official_checkpoint, map_location="cpu"
    )
    if multi.checkpoint_kind(official_package) not in (
        "official_embedding", "official_embedding_slim"
    ):
        raise ValueError("--official-checkpoint must use official embedding format")
    official_model = multi.reconstruct_backbone(
        raw_package, multi.official_backbone_state(official_package), device
    )
    official, official_peak, indices_2, mask_2, values_2, official_diag = (
        multi.extract_multilevel(
            official_model, loader, device, layers, args.peak_tokens
        )
    )
    del official_model, official_package, raw_package
    gc.collect()
    if not (
        np.array_equal(peak_indices, indices_2)
        and np.array_equal(peak_mask, mask_2)
        and np.array_equal(peak_values, values_2)
    ):
        raise RuntimeError("Raw and official runs used different peak tokens")

    np.save(args.output_dir / "rows.npy", rows)
    np.save(args.output_dir / "raw_precursor.npy", raw)
    np.save(args.output_dir / "official_precursor.npy", official)
    np.save(args.output_dir / "raw_peak.npy", raw_peak)
    np.save(args.output_dir / "official_peak.npy", official_peak)
    np.save(args.output_dir / "peak_indices.npy", peak_indices)
    np.save(args.output_dir / "peak_mask.npy", peak_mask)
    np.save(args.output_dir / "peak_values.npy", peak_values)
    (args.output_dir / "pairs.json").write_text(
        json.dumps(pairs, indent=2), encoding="utf-8"
    )
    report = {
        "status": "mass_dense_factor_activations",
        "config": {
            "data": str(args.data),
            "manifest": str(args.manifest),
            "split": args.split,
            "n_molecules": len(units),
            "n_spectra": len(rows),
            "layers": layers,
            "peak_tokens_per_spectrum": args.peak_tokens,
            "max_units": args.max_units,
        },
        "audit": {
            "directed_negative_links": int(
                sum(len(pair["negative_pair_ids"]) for pair in pairs)
            ),
            "all_pairs_have_10ppm_negative": all(
                pair["negative_pair_ids"] for pair in pairs
            ),
            "maximum_nearest_negative_ppm": float(
                max(pair["nearest_negative_ppm"] for pair in pairs)
            ),
        },
        "raw_diagnostics": raw_diag,
        "official_diagnostics": official_diag,
        "same_layer_cka": [
            multi.linear_cka(raw[:, i], official[:, i])
            for i in range(len(layers))
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["audit"], indent=2), flush=True)
    print(f"Saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
