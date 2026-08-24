"""Build a tiny real strict-10ppm cache for local Noise-v3 end-to-end smoke."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
for item in (ROOT, ROOT / "tasks"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


def decode(values) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--manifest", type=Path, default=ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json")
    parser.add_argument("--official", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/noise_v3_local_smoke_inputs")
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))["entries"]
    with h5py.File(args.data, "r") as handle:
        ik = decode(handle["INCHIKEY"][:])
        ik14 = np.asarray([value[:14] for value in ik], dtype=object)
        adduct = decode(handle["adduct"][:])
        precursor = np.asarray(handle["precursor_mz"][:], dtype=float)
        formula = decode(handle["FORMULA"][:])
        by_adduct = {}
        for ion in np.unique(adduct):
            rows = np.flatnonzero(adduct == ion)
            order = np.argsort(precursor[rows], kind="mergesort")
            by_adduct[str(ion)] = precursor[rows][order], rows[order]
        selected = []
        for entry in entries:
            row = int(entry["anchor_row"])
            masses, rows = by_adduct[str(adduct[row])]
            tolerance = precursor[row] * args.ppm * 1e-6
            left = np.searchsorted(masses, precursor[row] - tolerance, side="left")
            right = np.searchsorted(masses, precursor[row] + tolerance, side="right")
            candidates = rows[left:right]
            candidates = candidates[candidates != row]
            identities = defaultdict(list)
            for candidate in candidates:
                identities[str(ik14[candidate])].append(int(candidate))
            if str(ik14[row]) not in identities or len(identities) < 2:
                continue
            ordered = [str(ik14[row])] + sorted(key for key in identities if key != str(ik14[row]))
            selected.append((row, ordered, identities, entry))
            if len(selected) == args.queries:
                break
        if len(selected) < args.queries:
            raise RuntimeError(f"only {len(selected)} real queries survived")
        needed = sorted({row for query, order, groups, entry in selected for row in [query, *[x for key in order for x in groups[key]]]})
        tensors = [preprocess_spectrum(
            np.asarray(handle["spectrum"][row]), float(precursor[row]), 100,
        ) for row in needed]

    device = torch.device(args.device)
    # This workstation currently has a too-new huggingface-hub installed by an
    # unrelated MolFormer workflow. DreaMS does not use Transformers here, but
    # torchmetrics imports it eagerly and its version guard blocks Lightning.
    # Patch only the in-process metadata answer during optional imports; no
    # package or environment is modified.
    original_version = importlib.metadata.version
    def compatible_optional_version(package: str) -> str:
        if package.replace("_", "-").lower() == "huggingface-hub":
            observed = original_version(package)
            if observed.startswith("1."):
                return "0.36.0"
        return original_version(package)
    importlib.metadata.version = compatible_optional_version
    try:
        model, _ = load_base_model(args.official, args.architecture, device, 100)
    finally:
        importlib.metadata.version = original_version
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoded = []
    with torch.inference_mode():
        for left in range(0, len(tensors), args.batch_size):
            batch = torch.stack(tensors[left:left + args.batch_size]).to(device)
            encoded.append(model(batch).float().cpu().numpy())
            print(f"[smoke-encode] {min(left + args.batch_size, len(tensors))}/{len(tensors)}", flush=True)
    embeddings = np.concatenate(encoded).astype(np.float32)
    row_to_embedding = {row: embeddings[index] for index, row in enumerate(needed)}

    features = []
    pair_rows = []
    molecule_ptr = [0]
    molecule_label = []
    molecule_ik14 = []
    molecule_formula = []
    molecule_grade = []
    query_ptr = [0]
    query_rows = []
    query_ik14 = []
    query_formula = []
    query_has_near = []
    for query, order, groups, entry in selected:
        query_rows.append(query)
        query_ik14.append(str(ik14[query]))
        query_formula.append(str(formula[query]))
        explicit_grade = {str(item["ik14"]): str(item["grade"]) for item in entry.get("neg", [])}
        has_near = False
        for position, identity in enumerate(order):
            rows = groups[identity]
            for candidate in rows:
                features.append([float(row_to_embedding[query] @ row_to_embedding[candidate])])
                pair_rows.append(candidate)
            molecule_ptr.append(len(features))
            molecule_label.append(1 if position == 0 else 0)
            molecule_ik14.append(identity)
            molecule_formula.append(str(formula[rows[0]]))
            grade_name = explicit_grade.get(identity, "unknown")
            grade = {"near": 0, "mid": 1, "far": 2}.get(grade_name, -1)
            molecule_grade.append(-2 if position == 0 else grade)
            has_near |= grade == 0
        query_ptr.append(len(molecule_label))
        query_has_near.append(has_near)
    np.savez_compressed(
        args.output_dir / "cache.npz",
        feature_names=np.asarray(["dreams_similarity"]),
        features=np.asarray(features, dtype=np.float32),
        pair_candidate_row=np.asarray(pair_rows, dtype=np.int64),
        query_ptr=np.asarray(query_ptr, dtype=np.int64),
        molecule_ptr=np.asarray(molecule_ptr, dtype=np.int64),
        molecule_label=np.asarray(molecule_label, dtype=np.int8),
        molecule_ik14=np.asarray(molecule_ik14),
        molecule_formula=np.asarray(molecule_formula),
        molecule_mces_grade=np.asarray(molecule_grade, dtype=np.int8),
        query_row=np.asarray(query_rows, dtype=np.int64),
        query_ik14=np.asarray(query_ik14),
        query_formula=np.asarray(query_formula),
        query_has_near=np.asarray(query_has_near, dtype=bool),
    )
    np.savez_compressed(
        args.output_dir / "embeddings.npz",
        rows=np.asarray(needed, dtype=np.int64), embeddings=embeddings,
    )
    p3 = args.output_dir / "p3"
    p3.mkdir()
    (p3 / "p3_smoke_manifest.json").write_text(
        json.dumps({"queries": [{"ik14": "SMOKE_P3_DUMMY"}]}), encoding="utf-8",
    )
    print(json.dumps({
        "status": "noise_v3_real_local_smoke_inputs_built",
        "queries": len(selected), "reachable_rows": len(needed),
        "candidate_molecules": len(molecule_label), "pair_edges": len(features),
        "cache": str(args.output_dir / "cache.npz"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
