"""Build a graph-aligned frozen Morgan fingerprint teacher.

The formal path reconstructs and audits the identity-to-SMILES ledger directly
from the frozen candidate graph and HDF5 structure fields; it never loads a
learned molecule model.  ``--source-dir`` remains available only for replaying
older non-formal diagnostics that used an audited MolFormer cache as the ledger.
Radius-2 binary Morgan fingerprints are L2-normalized so the training objective
computes Ochiai similarity, deliberately not mislabeled as Tanimoto.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_shared_v2_core import MOLFORMER_STATUS, MORGAN_STATUS  # noqa: E402
from encode_chemaware_shared_v2_molformer import (  # noqa: E402
    collision_audit as smiles_collision_audit,
    graph_molecule_records,
)
from noise_final_core import CandidateGraph, sha256_file  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path,
        default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--preflight", type=Path,
        default=ROOT / "data/validation/g8r_chemaware_shared_v2_preflight.json",
    )
    parser.add_argument(
        "--source-dir", type=Path,
        help="legacy audited MolFormer ledger; omitted for the formal direct-HDF5 path",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/g8r_chemaware_shared_v3_morgan",
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--dimensions", type=int, default=2048)
    parser.add_argument("--max-molecules", type=int, default=0, help="non-formal smoke only")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Morgan teacher: {args.output_dir}")
    if args.radius < 1 or args.dimensions < 64:
        raise ValueError("Morgan radius/dimensions are too small")
    if args.max_molecules < 0:
        raise ValueError("max-molecules must be nonnegative")
    required = [args.graph]
    if args.source_dir is None:
        required.append(args.data)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    graph = CandidateGraph(args.graph)
    graph_sha256 = sha256_file(args.graph)
    source_report_path = None
    source_report = None
    structure_audit = None
    if args.source_dir is not None:
        source_report_path = args.source_dir / "report.json"
        source_required = (
            source_report_path,
            args.source_dir / "molecule_ik14.npy",
            args.source_dir / "molecule_formula.npy",
            args.source_dir / "molformer_smiles.npy",
        )
        source_missing = [str(path) for path in source_required if not path.is_file()]
        if source_missing:
            raise FileNotFoundError(source_missing)
        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        if source_report.get("status") != MOLFORMER_STATUS:
            raise RuntimeError("legacy Morgan source is not an audited MolFormer ledger")
        if source_report.get("provenance", {}).get("graph_sha256") != graph_sha256:
            raise RuntimeError("Morgan source ledger belongs to a different graph")
        ik14 = np.load(args.source_dir / "molecule_ik14.npy").astype(str)
        formula = np.load(args.source_dir / "molecule_formula.npy").astype(str)
        smiles = np.load(args.source_dir / "molformer_smiles.npy").astype(str)
        formal = bool(source_report.get("formal")) and args.max_molecules == 0
        preflight_sha256 = source_report.get("provenance", {}).get("preflight_sha256")
        hdf5_sha256 = source_report.get("provenance", {}).get("hdf5_sha256")
        ledger_kind = "legacy_audited_molformer_cache"
    else:
        formal = args.max_molecules == 0
        observed_hdf5_sha256 = sha256_file(args.data)
        if formal:
            if not args.preflight.is_file():
                raise FileNotFoundError(args.preflight)
            preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
            if (
                preflight.get("status") != "chemaware_shared_v2_preflight_passed"
                or preflight.get("formal") is not True
            ):
                raise RuntimeError("formal Morgan cache requires the frozen passing preflight")
            expected = preflight.get("hashes", {})
            if (
                expected.get("graph_sha256") != graph_sha256
                or expected.get("hdf5_sha256") != observed_hdf5_sha256
            ):
                raise RuntimeError("formal Morgan graph/HDF5 differs from frozen preflight")
        records = graph_molecule_records(graph, args.data)
        if args.max_molecules:
            records = records[:args.max_molecules]
        ik14 = np.asarray([record["ik14"] for record in records], dtype="U14")
        formula = np.asarray([record["formula"] for record in records], dtype="U64")
        smiles = np.asarray([record["molformer_smiles"] for record in records], dtype=str)
        structure_audit = smiles_collision_audit(records)
        preflight_sha256 = sha256_file(args.preflight) if formal else None
        hdf5_sha256 = observed_hdf5_sha256
        ledger_kind = "direct_frozen_graph_hdf5_identity_smiles"
    if not (ik14.shape == formula.shape == smiles.shape):
        raise RuntimeError("Morgan source arrays are not aligned")
    expected_identities = np.unique(graph.molecule_ik14.astype(str))
    if formal and len(ik14) != 3472:
        raise RuntimeError(f"formal G8R Morgan cache expects 3,472 identities, observed {len(ik14):,}")
    if not args.max_molecules and not np.array_equal(np.sort(ik14), expected_identities):
        raise RuntimeError("Morgan source identities do not exactly cover graph")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=args.radius, fpSize=args.dimensions, includeChirality=False
    )
    fingerprints = np.zeros((len(ik14), args.dimensions), dtype=np.float32)
    invalid = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            invalid.append({"index": index, "ik14": str(ik14[index]), "smiles": str(value)})
            continue
        bit_vector = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(bit_vector, fingerprints[index])
    if invalid:
        raise RuntimeError(f"Morgan fingerprint parsing failed: {invalid[:10]}")
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("Morgan teacher produced an empty fingerprint")
    fingerprints /= norms
    packed = np.packbits(fingerprints > 0, axis=1)
    collision_groups: dict[bytes, list[int]] = {}
    for index, value in enumerate(packed):
        collision_groups.setdefault(value.tobytes(), []).append(index)
    cross_identity = [values for values in collision_groups.values() if len(values) > 1]
    formula_groups: dict[str, list[int]] = {}
    for index, value in enumerate(formula):
        formula_groups.setdefault(str(value), []).append(index)
    same_formula_pairs = 0
    same_formula_identical_pairs = 0
    for values in formula_groups.values():
        for left_offset, left in enumerate(values):
            for right in values[left_offset + 1:]:
                same_formula_pairs += 1
                same_formula_identical_pairs += int(np.array_equal(packed[left], packed[right]))

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".chemaware_morgan_", dir=args.output_dir.parent))
    try:
        np.save(staging / "molecule_ik14.npy", ik14)
        np.save(staging / "molecule_formula.npy", formula)
        np.save(staging / "connectivity_smiles.npy", smiles)
        np.save(staging / "embeddings_f32.npy", fingerprints)
        report = {
            "status": MORGAN_STATUS,
            "formal": formal,
            "molecules": int(len(ik14)),
            "embedding_dimension": int(args.dimensions),
            "teacher_kind": "morgan_binary_connectivity",
            "fingerprint": {
                "radius": int(args.radius),
                "dimensions": int(args.dimensions),
                "include_chirality": False,
                "similarity_in_training": "cosine_of_L2_normalized_binary_bits_Ochiai_not_Tanimoto",
            },
            "rdkit_version": rdkit.__version__,
            "smiles_policy": (
                "RDKit canonical non-isomeric SMILES reconstructed directly from frozen "
                "graph/HDF5 identity rows"
                if args.source_dir is None else
                "RDKit canonical non-isomeric SMILES from legacy audited MolFormer ledger"
            ),
            "identity_smiles_ledger": {
                "kind": ledger_kind,
                "stereochemistry_and_smiles_collision_audit": structure_audit,
            },
            "collision_audit": {
                "cross_identity_identical_fingerprint_groups": int(len(cross_identity)),
                "cross_identity_identical_fingerprint_molecules": int(sum(map(len, cross_identity))),
                "same_formula_identity_pairs": int(same_formula_pairs),
                "same_formula_identical_fingerprint_pairs": int(same_formula_identical_pairs),
                "examples": [
                    {
                        "ik14": [str(ik14[index]) for index in values],
                        "formula": sorted({str(formula[index]) for index in values}),
                        "smiles": sorted({str(smiles[index]) for index in values}),
                    }
                    for values in cross_identity[:10]
                ],
            },
            "training_only": True,
            "provenance": {
                "graph_sha256": graph_sha256,
                "hdf5_sha256": hdf5_sha256,
                "source_report_sha256": (
                    sha256_file(source_report_path) if source_report_path is not None else None
                ),
                "preflight_sha256": preflight_sha256,
                "script_sha256": sha256_file(Path(__file__)),
            },
            "claim_limit": (
                "Connectivity-only frozen teacher. It cannot support stereochemistry claims; "
                "Morgan cosine is not reported as Tanimoto."
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
