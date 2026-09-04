"""Encode the locked G8R candidate molecules with a frozen MoLFormer teacher.

This cache is training-only.  The shared spectrum encoder never reads it at
inference.  MoLFormer canonicalization deliberately removes stereochemistry,
matching its pretraining protocol; all cross-identity collapses are audited so
the cache cannot be used to claim stereochemical discrimination.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem

try:
    from noise_final_core import CandidateGraph, sha256_file
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .noise_final_core import CandidateGraph, sha256_file


ROOT = Path(__file__).resolve().parent.parent
STATUS = "chemaware_shared_v2_frozen_molformer_cache_complete"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_molformer")
    parser.add_argument("--preflight", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_preflight.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/models/huggingface")
    parser.add_argument("--model", default="ibm-research/MoLFormer-XL-both-10pct")
    parser.add_argument("--revision", default="compat-v4")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-molecules", type=int, default=0, help="non-formal smoke only")
    return parser.parse_args()


def hdf5_strings(handle: h5py.File, key: str, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    # h5py requires fancy indices to be strictly increasing.  Candidate rows
    # legitimately repeat across queries, so read each HDF5 row once and then
    # restore the original (possibly duplicated) graph order.
    unique_values = np.asarray(handle[key].asstr()[unique_rows], dtype=str)
    return unique_values[inverse]


def graph_molecule_records(graph: CandidateGraph, data: Path) -> list[dict]:
    pair_molecule = np.repeat(np.arange(len(graph.molecule_ik14)), np.diff(graph.molecule_ptr))
    candidate_rows = graph.pair_candidate_row
    with h5py.File(data, "r") as handle:
        smiles_key = "smiles" if "smiles" in handle else "SMILES"
        if smiles_key not in handle or "INCHIKEY" not in handle:
            raise RuntimeError("HDF5 lacks smiles/INCHIKEY structure fields")
        pair_ik14 = np.char.upper(
            np.char.partition(hdf5_strings(handle, "INCHIKEY", candidate_rows), "-")[:, 0]
        )
        expected_ik14 = np.char.upper(graph.molecule_ik14[pair_molecule].astype(str))
        if not np.array_equal(pair_ik14, expected_ik14):
            mismatch = np.flatnonzero(pair_ik14 != expected_ik14)
            raise RuntimeError(
                f"graph/HDF5 molecule identity mismatch for {len(mismatch)} pairs; "
                f"first={mismatch[:10].tolist()}"
            )
        unique_ik14, first_molecule = np.unique(
            graph.molecule_ik14.astype(str), return_index=True
        )
        representative_rows = graph.pair_candidate_row[graph.molecule_ptr[first_molecule]]
        raw_smiles = hdf5_strings(handle, smiles_key, representative_rows)
    records = []
    for ik14, molecule_index, row, smiles in zip(
        unique_ik14, first_molecule, representative_rows, raw_smiles
    ):
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            raise RuntimeError(f"RDKit failed for candidate {ik14} at HDF5 row {row}")
        isomeric = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        molformer = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        records.append({
            "ik14": str(ik14),
            "formula": str(graph.molecule_formula[int(molecule_index)]),
            "representative_hdf5_row": int(row),
            "canonical_isomeric_smiles": isomeric,
            "molformer_smiles": molformer,
            "stereochemistry_removed": isomeric != molformer,
        })
    return records


def collision_audit(records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["molformer_smiles"]].append(record)
    collisions = [
        values for values in groups.values()
        if len({value["ik14"] for value in values}) > 1
    ]
    return {
        "molecules": len(records),
        "unique_nonisomeric_smiles": len(groups),
        "molecules_with_stereochemistry_removed": int(sum(
            record["stereochemistry_removed"] for record in records
        )),
        "cross_identity_collapsed_smiles": len(collisions),
        "cross_identity_collapsed_molecules": int(sum(len(values) for values in collisions)),
        "examples": [{
            "molformer_smiles": values[0]["molformer_smiles"],
            "ik14": sorted({value["ik14"] for value in values}),
            "canonical_isomeric_smiles": sorted({
                value["canonical_isomeric_smiles"] for value in values
            }),
        } for values in collisions[:20]],
    }


def main() -> None:
    args = arguments()
    required = [args.graph, args.data]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite MoLFormer cache: {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    graph = CandidateGraph(args.graph)
    records = graph_molecule_records(graph, args.data)
    formal = args.max_molecules == 0
    if formal:
        if not args.preflight.is_file():
            raise FileNotFoundError(args.preflight)
        preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
        if preflight.get("status") != "chemaware_shared_v2_preflight_passed" or not preflight.get("formal"):
            raise RuntimeError("formal molecule cache requires passing ChemAware-v2 preflight")
    if args.max_molecules:
        records = records[:args.max_molecules]
    if formal and len(records) != 3472:
        raise RuntimeError(f"formal G8R teacher expects 3,472 identities, observed {len(records):,}")
    audit = collision_audit(records)
    print(json.dumps(audit, indent=2), flush=True)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.cache_dir)
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True,
        cache_dir=args.cache_dir / "hub", local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True,
        deterministic_eval=True, cache_dir=args.cache_dir / "hub",
        local_files_only=args.local_files_only,
    )
    commit = getattr(model.config, "_commit_hash", None)
    if formal and not commit:
        raise RuntimeError("formal MoLFormer cache requires a resolved immutable model commit")
    device = torch.device(args.device)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    smiles = [record["molformer_smiles"] for record in records]
    output = []
    repeat_error = None
    with torch.inference_mode():
        for left in range(0, len(smiles), args.batch_size):
            right = min(left + args.batch_size, len(smiles))
            encoded = tokenizer(smiles[left:right], padding=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            pooled = model(**encoded).pooler_output.float()
            if left == 0:
                repeated = model(**encoded).pooler_output.float()
                repeat_error = float(torch.max(torch.abs(pooled - repeated)).cpu())
                if repeat_error > 1e-6:
                    raise RuntimeError(f"MoLFormer eval is not deterministic: {repeat_error}")
            output.append(F.normalize(pooled, dim=-1).cpu().numpy())
            print(f"[ChemAware-v2 MoLFormer] {right:,}/{len(smiles):,}", flush=True)
    embeddings = np.concatenate(output).astype(np.float32)
    if not np.all(np.isfinite(embeddings)):
        raise RuntimeError("non-finite MoLFormer teacher embeddings")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".chemaware_v2_molformer_", dir=args.output_dir.parent))
    try:
        np.save(staging / "molecule_ik14.npy", np.asarray([r["ik14"] for r in records], dtype="U14"))
        np.save(staging / "molecule_formula.npy", np.asarray([r["formula"] for r in records], dtype="U64"))
        np.save(staging / "canonical_isomeric_smiles.npy", np.asarray(
            [r["canonical_isomeric_smiles"] for r in records], dtype=str
        ))
        np.save(staging / "molformer_smiles.npy", np.asarray(
            [r["molformer_smiles"] for r in records], dtype=str
        ))
        np.save(staging / "embeddings_f32.npy", embeddings)
        report = {
            "status": STATUS,
            "formal": formal,
            "molecules": len(records),
            "embedding_dimension": int(embeddings.shape[1]),
            "model": args.model,
            "requested_revision": args.revision,
            "resolved_commit": commit,
            "normalization": "L2",
            "smiles_policy": "RDKit canonical isomericSmiles=False, matching MoLFormer pretraining",
            "repeat_max_abs_error": repeat_error,
            "collision_audit": audit,
            "training_only": True,
            "provenance": {
                "graph_sha256": sha256_file(args.graph),
                "hdf5_sha256": sha256_file(args.data),
                "preflight_sha256": sha256_file(args.preflight) if formal else None,
                "script_sha256": sha256_file(Path(__file__)),
            },
            "claim_limit": (
                "MoLFormer removes stereochemistry. Cross-identity collapsed SMILES and "
                "stereochemical distinctions cannot support stereochemistry claims."
            ),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
