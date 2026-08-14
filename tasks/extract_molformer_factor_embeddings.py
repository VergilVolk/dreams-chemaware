"""Extract frozen MolFormer embeddings for the DreaMS factor cohorts.

The script follows the model card: canonical, non-isomeric SMILES are used
because stereochemistry was removed during MoLFormer pretraining.  Both the
loss of stereochemical identity and any resulting SMILES collisions are
explicitly audited.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_DISCOVERY = ROOT / "data/validation/mass_dense_factor_discovery"
DEFAULT_CONFIRMATION = ROOT / "data/validation/mass_dense_factor_confirmation"
DEFAULT_OUTPUT = ROOT / "data/validation/molformer_factor_embeddings"
DEFAULT_CACHE = ROOT / "data/models/huggingface"
DEFAULT_MODEL = "ibm-research/MoLFormer-XL-both-10pct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="compat-v4")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-per-split", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_records(data: Path, directory: Path, split: str, limit: int) -> list[dict]:
    pairs = json.loads((directory / "pairs.json").read_text(encoding="utf-8"))
    if limit > 0:
        pairs = pairs[:limit]
    rows = np.asarray([pair["rows"][0] for pair in pairs], dtype=np.int64)
    order = np.argsort(rows)
    inverse = np.argsort(order)
    with h5py.File(data, "r") as handle:
        raw_smiles = handle["smiles"].asstr()[rows[order]][inverse].tolist()
        inchikeys = handle["INCHIKEY"].asstr()[rows[order]][inverse].tolist()
    records = []
    for pair, row, smiles, inchikey in zip(pairs, rows, raw_smiles, inchikeys):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse row {row}: {smiles}")
        canonical_isomeric = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        canonical_molformer = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=False
        )
        records.append({
            "split": split,
            "pair_id": int(pair["pair_id"]),
            "ik14": pair["ik14"],
            "inchikey": inchikey,
            "row": int(row),
            "raw_smiles": smiles,
            "canonical_isomeric_smiles": canonical_isomeric,
            "molformer_smiles": canonical_molformer,
            "stereochemistry_removed": canonical_isomeric != canonical_molformer,
        })
    return records


def collision_audit(records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["molformer_smiles"]].append(record)
    collisions = {
        smiles: values for smiles, values in groups.items()
        if len({value["canonical_isomeric_smiles"] for value in values}) > 1
    }
    return {
        "n_records": len(records),
        "n_unique_molformer_smiles": len(groups),
        "n_records_with_stereochemistry_removed": int(sum(
            record["stereochemistry_removed"] for record in records
        )),
        "n_colliding_smiles": len(collisions),
        "n_records_in_collisions": int(sum(len(values) for values in collisions.values())),
        "collision_examples": [
            {
                "molformer_smiles": smiles,
                "ik14": sorted({value["ik14"] for value in values}),
                "canonical_isomeric_smiles": sorted({
                    value["canonical_isomeric_smiles"] for value in values
                }),
                "splits": sorted({value["split"] for value in values}),
            }
            for smiles, values in list(collisions.items())[:20]
        ],
    }


def extract(
    records: list[dict], tokenizer, model, batch_size: int, device: torch.device
) -> tuple[np.ndarray, dict]:
    smiles = [record["molformer_smiles"] for record in records]
    token_lengths = [
        len(tokenizer(value, add_special_tokens=True)["input_ids"])
        for value in smiles
    ]
    outputs = []
    repeat_error = None
    for start in range(0, len(smiles), batch_size):
        batch = smiles[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            pooled = model(**encoded).pooler_output
            if start == 0:
                repeated = model(**encoded).pooler_output
                repeat_error = float((pooled - repeated).abs().max().cpu())
        outputs.append(pooled.detach().cpu().float().numpy())
        print(f"embedded {min(start + len(batch), len(smiles))}/{len(smiles)}", flush=True)
    matrix = np.concatenate(outputs, axis=0)
    return matrix, {
        "n_records": len(records),
        "embedding_dim": int(matrix.shape[1]),
        "all_finite": bool(np.isfinite(matrix).all()),
        "zero_variance_dimensions": int(np.sum(np.std(matrix, axis=0) < 1e-12)),
        "embedding_norm_min": float(np.linalg.norm(matrix, axis=1).min()),
        "embedding_norm_max": float(np.linalg.norm(matrix, axis=1).max()),
        "token_length_min": int(min(token_lengths)),
        "token_length_median": float(np.median(token_lengths)),
        "token_length_max": int(max(token_lengths)),
        "token_lengths_over_202": int(sum(value > 202 for value in token_lengths)),
        "first_batch_repeat_max_abs_error": repeat_error,
    }


def write_records(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.cache_dir)
    # Import after setting HF_HOME so remote-code modules and model files use
    # the same project-local cache rather than two different user caches.
    from transformers import AutoModel, AutoTokenizer

    discovery = load_records(
        args.data, args.discovery, "discovery", args.max_per_split
    )
    confirmation = load_records(
        args.data, args.confirmation, "confirmation", args.max_per_split
    )
    overlap = {record["ik14"] for record in discovery} & {
        record["ik14"] for record in confirmation
    }
    if overlap:
        raise RuntimeError(f"Molecule leakage detected: {len(overlap)} IK14")
    all_records = discovery + confirmation
    print(json.dumps(collision_audit(all_records), indent=2), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        cache_dir=args.cache_dir / "hub",
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        deterministic_eval=True,
        cache_dir=args.cache_dir / "hub",
        local_files_only=args.local_files_only,
    )
    device = torch.device(args.device)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    discovery_embedding, discovery_audit = extract(
        discovery, tokenizer, model, args.batch_size, device
    )
    confirmation_embedding, confirmation_audit = extract(
        confirmation, tokenizer, model, args.batch_size, device
    )
    np.save(args.output_dir / "discovery.npy", discovery_embedding)
    np.save(args.output_dir / "confirmation.npy", confirmation_embedding)
    write_records(args.output_dir / "discovery_records.csv", discovery)
    write_records(args.output_dir / "confirmation_records.csv", confirmation)
    report = {
        "status": "frozen_molformer_embeddings",
        "config": {
            "model": args.model,
            "revision": args.revision,
            "data": str(args.data),
            "discovery": str(args.discovery),
            "confirmation": str(args.confirmation),
            "batch_size": args.batch_size,
            "device": args.device,
            "max_per_split": args.max_per_split,
            "smiles_policy": "RDKit canonical, isomericSmiles=False",
        },
        "audit": {
            "molecule_overlap": len(overlap),
            "collisions": collision_audit(all_records),
            "discovery": discovery_audit,
            "confirmation": confirmation_audit,
        },
        "interpretation_limit": (
            "MoLFormer was pretrained after removing isomeric information. "
            "Pairs that collapse to the same non-isomeric SMILES cannot be used "
            "to claim stereochemical factor recovery."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
