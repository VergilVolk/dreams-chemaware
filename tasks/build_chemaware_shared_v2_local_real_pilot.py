"""Build a small real-spectrum, explicitly non-formal ChemAware-v2 CPU pilot.

The pilot uses the same strict-10ppm/same-adduct/self-excluded candidate rule
on the local MassSpecGym HDF5, but it is intentionally selected for tiny
candidate groups and has no P3 seal.  It validates real model/data execution;
it must never be reported as a performance benchmark.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

import pilot_multilevel_factor_activations as multi  # noqa: E402
from chemaware_shared_v2_core import TOKEN_STATUS  # noqa: E402
from e1_checkpoint_io import official_head_state  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--raw-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/validation/chemaware_shared_v2_local_real_pilot")
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--minimum-molecules", type=int, default=3)
    parser.add_argument("--maximum-candidate-spectra", type=int, default=3)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.queries < 10 or args.minimum_molecules < 3 or args.maximum_candidate_spectra < 3:
        raise ValueError("pilot is too small or has no meaningful negatives")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite local real pilot: {args.output_root}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    for path in (args.data, args.raw_checkpoint, args.official_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    with h5py.File(args.data, "r") as handle:
        precursor = np.asarray(handle["precursor_mz"][:], dtype=np.float64)
        adduct = np.asarray(handle["adduct"].asstr()[:], dtype=str)
        inchikey = np.asarray(handle["INCHIKEY"].asstr()[:], dtype=str)
        ik14 = np.char.partition(inchikey, "-")[:, 0]
        formula = np.asarray(handle["FORMULA"].asstr()[:], dtype=str)
    mass_order = np.argsort(precursor, kind="stable")
    sorted_mass = precursor[mass_order]
    candidates: dict[int, np.ndarray] = {}
    valid = []
    for query in range(len(precursor)):
        tolerance = float(precursor[query]) * 1e-5
        left = np.searchsorted(sorted_mass, precursor[query] - tolerance, side="left")
        right = np.searchsorted(sorted_mass, precursor[query] + tolerance, side="right")
        rows = mass_order[left:right]
        rows = rows[(rows != query) & (adduct[rows] == adduct[query])]
        identities = np.unique(ik14[rows])
        if (
            len(rows) <= args.maximum_candidate_spectra
            and ik14[query] in identities
            and len(identities) >= args.minimum_molecules
        ):
            candidates[query] = rows.astype(np.int64)
            valid.append(query)
    selected = []
    seen_identity = set()
    for query in sorted(valid, key=lambda value: (len(candidates[value]), value)):
        identity = str(ik14[query])
        if identity in seen_identity:
            continue
        selected.append(query)
        seen_identity.add(identity)
        if len(selected) == args.queries:
            break
    if len(selected) != args.queries:
        raise RuntimeError(f"only {len(selected)} eligible identity-disjoint pilot queries")
    selected = np.asarray(selected, dtype=np.int64)
    reachable = np.unique(np.concatenate((selected, *(candidates[int(q)] for q in selected))))

    device = torch.device(args.device)
    raw = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    backbone = multi.reconstruct_backbone(
        raw, multi.official_backbone_state(official), device
    ).eval()
    head = torch.nn.Linear(int(backbone.d_model), int(backbone.d_model), bias=True).to(device)
    head.load_state_dict(official_head_state(official), strict=True)
    head.eval()
    for parameter in list(backbone.parameters()) + list(head.parameters()):
        parameter.requires_grad_(False)
    dtype = next(backbone.parameters()).dtype
    loader = DataLoader(
        multi.SpectrumRows(args.data, reachable, args.n_highest_peaks),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    token_blocks, mz_blocks, intensity_blocks = [], [], []
    valid_blocks, precursor_blocks, embedding_blocks = [], [], []
    with torch.inference_mode():
        for batch_index, spectra in enumerate(loader, start=1):
            spectra = spectra.to(device=device, dtype=dtype)
            contextual = backbone(spectra, None)
            peak_valid = spectra[:, 1:, 0] > 0
            token_blocks.append(
                contextual[:, 1:, :].masked_fill(~peak_valid.unsqueeze(-1), 0).half().cpu().numpy()
            )
            mz_blocks.append(spectra[:, 1:, 0].float().cpu().numpy())
            intensity_blocks.append(spectra[:, 1:, 1].float().cpu().numpy())
            valid_blocks.append(peak_valid.cpu().numpy())
            precursor_blocks.append(spectra[:, 0, 0].float().cpu().numpy())
            embedding_blocks.append(
                F.normalize(head(contextual[:, 0, :].float()), dim=-1).cpu().numpy()
            )
            print(
                f"[local-real-cache] {min(batch_index * args.batch_size, len(reachable))}/"
                f"{len(reachable)}", flush=True,
            )
    tokens = np.concatenate(token_blocks)
    mz = np.concatenate(mz_blocks)
    intensity = np.concatenate(intensity_blocks)
    peak_valid = np.concatenate(valid_blocks)
    cached_precursor = np.concatenate(precursor_blocks)
    embeddings = np.concatenate(embedding_blocks).astype(np.float32)
    position = {int(row): index for index, row in enumerate(reachable)}

    pair_features, pair_rows = [], []
    molecule_ptr = [0]
    molecule_label, molecule_identity, molecule_formula, molecule_grade = [], [], [], []
    query_ptr = [0]
    query_has_near = []
    for query in selected:
        rows = candidates[int(query)]
        grouped = {}
        for row in rows:
            grouped.setdefault(str(ik14[row]), []).append(int(row))
        query_identity = str(ik14[query])
        if query_identity not in grouped:
            raise RuntimeError("pilot positive disappeared")
        query_embedding = embeddings[position[int(query)]]
        ordered = [query_identity] + sorted(
            (value for value in grouped if value != query_identity),
            key=lambda value: (
                -max(float(query_embedding @ embeddings[position[row]]) for row in grouped[value]),
                value,
            ),
        )
        near = False
        for identity in ordered:
            identity_rows = sorted(
                grouped[identity],
                key=lambda row: -float(query_embedding @ embeddings[position[row]]),
            )
            for row in identity_rows:
                pair_rows.append(row)
                pair_features.append([float(query_embedding @ embeddings[position[row]])])
            molecule_ptr.append(len(pair_rows))
            molecule_label.append(int(identity == query_identity))
            molecule_identity.append(identity)
            molecule_formula.append(str(formula[identity_rows[0]]))
            molecule_grade.append(-2 if identity == query_identity else -1)
            near |= identity != query_identity and formula[identity_rows[0]] == formula[query]
        query_ptr.append(len(molecule_label))
        query_has_near.append(near)

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".chemaware_local_real_", dir=args.output_root.parent))
    try:
        graph_path = staging / "graph.npz"
        np.savez_compressed(
            graph_path,
            feature_names=np.asarray(["dreams_similarity"], dtype=object),
            features=np.asarray(pair_features, dtype=np.float32),
            pair_candidate_row=np.asarray(pair_rows, dtype=np.int64),
            query_ptr=np.asarray(query_ptr, dtype=np.int64),
            molecule_ptr=np.asarray(molecule_ptr, dtype=np.int64),
            molecule_label=np.asarray(molecule_label, dtype=np.int8),
            molecule_ik14=np.asarray(molecule_identity, dtype=object),
            molecule_formula=np.asarray(molecule_formula, dtype=object),
            molecule_mces_grade=np.asarray(molecule_grade, dtype=np.int8),
            query_row=selected,
            query_ik14=ik14[selected].astype(object),
            query_formula=formula[selected].astype(object),
            query_has_near=np.asarray(query_has_near, dtype=bool),
        )
        cache = staging / "tokens"
        cache.mkdir()
        np.save(cache / "rows.npy", reachable)
        np.save(cache / "tokens_f16.npy", tokens)
        np.save(cache / "mz_f32.npy", mz)
        np.save(cache / "intensity_f32.npy", intensity)
        np.save(cache / "valid.npy", peak_valid)
        np.save(cache / "precursor_mz_f32.npy", cached_precursor)
        np.save(cache / "official_embeddings_f32.npy", embeddings)
        token_report = {
            "status": TOKEN_STATUS,
            "formal": False,
            "spectra": int(len(reachable)),
            "source": "real local MassSpecGym spectra; official DreaMS CPU encoding",
            "candidate_inputs_used": False,
            "provenance": {
                "graph_sha256": sha256_file(graph_path),
                "hdf5_sha256": sha256_file(args.data),
                "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
                "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            },
        }
        (cache / "report.json").write_text(json.dumps(token_report, indent=2), encoding="utf-8")
        baseline_scores = np.asarray(pair_features, dtype=np.float32)[:, 0]
        molecule_scores = np.maximum.reduceat(baseline_scores, np.asarray(molecule_ptr[:-1]))
        ranks = np.asarray([
            1 + np.sum(
                molecule_scores[int(query_ptr[q]) + 1:int(query_ptr[q + 1])]
                >= molecule_scores[int(query_ptr[q])]
            )
            for q in range(len(selected))
        ])
        report = {
            "status": "chemaware_shared_v2_local_real_pilot_complete",
            "formal": False,
            "queries": int(len(selected)),
            "query_identities": int(len(np.unique(ik14[selected]))),
            "query_formulas": int(len(np.unique(formula[selected]))),
            "reachable_spectra": int(len(reachable)),
            "candidate_molecules": int(len(molecule_label)),
            "baseline_recall1": float(np.mean(ranks == 1)),
            "baseline_errors": int(np.sum(ranks != 1)),
            "near_queries": int(np.sum(query_has_near)),
            "candidate_protocol": "strict-10ppm same-adduct self-row-excluded",
            "selection_bias": "one identity each; smallest eligible candidate-spectrum groups first",
            "P3_seal": False,
            "claim_limit": "real execution pilot only; biased tiny graph, no performance claim",
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
