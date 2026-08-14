"""Prepare a formula-disjoint cached-backbone pilot for lipid hard negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tasks"))
import pilot_multilevel_factor_activations as multi  # noqa: E402
from pilot_paired_layer_cka import preprocess_spectrum  # noqa: E402
from e1_checkpoint_io import official_head_state  # noqa: E402


def peak_hash(spectrum: np.ndarray) -> str:
    mz, intensity = spectrum[0], spectrum[1]
    keep = (mz > 0) & (intensity > 0)
    packed = np.stack((np.rint(mz[keep] / 0.01), np.rint(intensity[keep] / 0.01)), axis=1).astype(np.int32)
    return hashlib.blake2b(packed.tobytes(), digest_size=8).hexdigest()


def stable_number(text: str, seed: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}|{text}".encode(), digest_size=8).digest(), "little")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--pool-manifest", type=Path, default=Path("data/validation/massspecgym_train_lipid_pool_gate/manifest.json"))
    parser.add_argument("--external-dir", type=Path, default=Path("data/validation/external_ring_balanced_pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/e1/lipid_projection_pilot"))
    parser.add_argument("--official-checkpoint", type=Path, default=multi.DEFAULT_OFFICIAL)
    parser.add_argument("--raw-checkpoint", type=Path, default=multi.DEFAULT_RAW)
    parser.add_argument("--max-spectra-per-molecule", type=int, default=4)
    parser.add_argument("--val-formula-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    units = json.loads(args.pool_manifest.read_text(encoding="utf-8"))["units"]
    external_formulas = set()
    for split in ("discovery", "confirmation"):
        external = json.loads((args.external_dir / f"{split}_manifest.json").read_text(encoding="utf-8"))["units"]
        external_formulas.update(unit["formula"] for unit in external)
    excluded_formula = sorted({unit["formula"] for unit in units} & external_formulas)
    units = [unit for unit in units if unit["formula"] not in external_formulas]

    by_formula = defaultdict(list)
    for unit in units: by_formula[unit["formula"]].append(unit)
    formulas = sorted(by_formula, key=lambda value: stable_number(value, args.seed))
    target_val = max(1, round(len(formulas) * args.val_formula_fraction))
    val_formulas = set(formulas[:target_val])

    records, tensors = [], []
    with h5py.File(args.hdf5, "r") as handle:
        for unit in units:
            candidates = []
            seen = set()
            for row in unit["hdf5_rows"]:
                spectrum = np.asarray(handle["spectrum"][row])
                token = peak_hash(spectrum)
                if token in seen: continue
                seen.add(token); candidates.append((int(row), spectrum, float(handle["precursor_mz"][row])))
                if len(candidates) >= args.max_spectra_per_molecule: break
            if len(candidates) < 2:
                continue
            split = "val" if unit["formula"] in val_formulas else "train"
            for row, spectrum, precursor in candidates:
                records.append({"hdf5_row": row, "ik14": unit["ik14"], "formula": unit["formula"], "split": split})
                tensors.append(preprocess_spectrum(spectrum, precursor, args.n_highest_peaks))

    device = torch.device(args.device); torch.set_num_threads(min(torch.get_num_threads(), 8))
    raw = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    model = multi.reconstruct_backbone(raw, multi.official_backbone_state(official), device).eval()
    head = official_head_state(official)
    weight, bias = head["weight"].float(), head["bias"].float()
    output = []
    loader = DataLoader(TensorDataset(torch.stack(tensors)), batch_size=args.batch_size, shuffle=False)
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for (batch,) in loader:
            output.append(model(batch.to(device=device, dtype=dtype), None)[:, 0].float().cpu().numpy())
    backbone = np.concatenate(output).astype(np.float32)
    official_embedding = backbone @ weight.numpy().T + bias.numpy()
    official_embedding /= np.clip(np.linalg.norm(official_embedding, axis=1, keepdims=True), 1e-12, None)
    np.savez_compressed(args.output_dir / "cache.npz", backbone=backbone, official=official_embedding.astype(np.float32))
    (args.output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    torch.save({"weight": weight, "bias": bias}, args.output_dir / "official_head.pt")

    split_counts = {}
    for split in ("train", "val"):
        part = [record for record in records if record["split"] == split]
        split_counts[split] = {
            "spectra": len(part), "molecules": len({r["ik14"] for r in part}),
            "formulas": len({r["formula"] for r in part}),
        }
    report = {
        "status": "lipid_projection_pilot_cache", "excluded_external_formulas": excluded_formula,
        "formula_split_overlap": 0, "split_counts": split_counts,
        "embedding_definition": "unmodified official backbone precursor tokens cached; official head stored separately",
        "pilot_scope": "mechanism test only; not a formal full-model fine-tune",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
