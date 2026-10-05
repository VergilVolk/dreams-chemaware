"""Build one frozen ChemBERTa vector per corrected-manifest molecule identity."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from noise_final_core import sha256_file  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--model", default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/chemaware_chemberta_teacher_v1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-identities", type=int, default=0)
    return parser.parse_args()


def decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def canonical_connectivity(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.torch_threads)
    with np.load(args.manifest) as body:
        identity = body["molecule_ik14"].astype(str)
        formula = body["molecule_formula"].astype(str)
        representative = body["pair_candidate_row"][body["molecule_ptr"][:-1]].astype(np.int64)
    first: dict[str, tuple[int, str]] = {}
    for index, value in enumerate(identity):
        first.setdefault(str(value), (index, str(formula[index])))
    identities = np.asarray(sorted(first), dtype="U14")
    if args.max_identities:
        identities = identities[:args.max_identities]
    formulas = np.asarray([first[value][1] for value in identities])
    rows = np.asarray([representative[first[value][0]] for value in identities], dtype=np.int64)
    with h5py.File(args.data, "r") as handle:
        raw_smiles = [decode(handle["smiles"][int(row)]) for row in rows]
    smiles = np.asarray([canonical_connectivity(value) for value in raw_smiles])

    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)
    model.eval()
    dimension = int(model.config.hidden_size)
    embedding = np.empty((len(smiles), dimension), dtype=np.float16)
    with torch.inference_mode():
        for left in range(0, len(smiles), args.batch_size):
            right = min(left + args.batch_size, len(smiles))
            tokens = tokenizer(
                smiles[left:right].tolist(), padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt",
            ).to(args.device)
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = torch.sum(hidden * mask, dim=1) / torch.sum(mask, dim=1).clamp_min(1)
            embedding[left:right] = pooled.float().cpu().numpy().astype(np.float16)
            if right == len(smiles) or right % (args.batch_size * 25) == 0:
                print(f"[ChemBERTa teacher] {right:,}/{len(smiles):,}", flush=True)
    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "identities.npy", identities)
    np.save(args.output_dir / "formulas.npy", formulas)
    np.save(args.output_dir / "representative_rows.npy", rows)
    np.save(args.output_dir / "canonical_smiles.npy", smiles)
    np.save(args.output_dir / "embeddings_f16.npy", embedding)
    report = {
        "status": "chemaware_chemberta_teacher_complete",
        "identities": int(len(identities)), "dimension": dimension,
        "model": args.model, "pooling": "attention-mask mean of final hidden state",
        "stereochemistry": "removed; identity is connectivity-level InChIKey14",
        "training_only": True,
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "hdf5_sha256": sha256_file(args.data),
        },
        "runtime_seconds": time.time() - started,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
