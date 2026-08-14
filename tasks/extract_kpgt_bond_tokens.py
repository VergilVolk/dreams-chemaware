"""Extract KPGT bond-anchored tokens without using its global readout.

This is an intentionally small bridge for the fragmentation-factor pilot.  It
keeps only LiGhT nodes whose ``vavn`` indicator is zero (real molecular bonds)
and writes both the input triplet representation and the contextualized bond
representation.  KPGT's fingerprint and molecular-descriptor virtual nodes
are required by the published checkpoint during the forward pass, but they
are never exported as bond evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
KPGT_ROOT = ROOT / "data" / "external" / "KPGT"
sys.path.insert(0, str(KPGT_ROOT))

from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized  # noqa: E402
from src.data.featurizer import (  # noqa: E402
    N_ATOM_TYPES,
    N_BOND_TYPES,
    Vocab,
    smiles_to_graph_tune,
)
from src.model.light import LiGhTPredictor  # noqa: E402
from src.model_config import config_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smiles", nargs="*", default=[])
    parser.add_argument("--smiles-file", type=Path)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--featurization-only",
        action="store_true",
        help="Validate bond metadata and KPGT graph construction without a checkpoint.",
    )
    parser.add_argument(
        "--random-init-smoke",
        action="store_true",
        help="Run the full 12-layer shape/runtime smoke test without scientific output.",
    )
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for i, smiles in enumerate(args.smiles):
        rows.append((f"cli_{i:04d}", smiles))
    if args.smiles_file:
        with args.smiles_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if args.smiles_column not in (reader.fieldnames or []):
                raise ValueError(f"Missing SMILES column: {args.smiles_column}")
            for i, row in enumerate(reader):
                smiles = (row.get(args.smiles_column) or "").strip()
                if smiles:
                    item_id = (row.get(args.id_column) or f"row_{i:06d}").strip()
                    rows.append((item_id, smiles))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("Provide --smiles or --smiles-file")
    return rows


def canonicalized_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    # Replicate KPGT's atom renumbering exactly so bond order matches graph nodes.
    new_order = list(Chem.rdmolfiles.CanonicalRankAtoms(mol))
    return Chem.rdmolops.RenumberAtoms(mol, new_order)


def atom_environment_smarts(mol: Chem.Mol, atom_ids: tuple[int, int], radius: int) -> str:
    bonds: set[int] = set()
    atoms: set[int] = set(atom_ids)
    frontier = set(atom_ids)
    for _ in range(radius):
        next_frontier: set[int] = set()
        for atom_id in frontier:
            atom = mol.GetAtomWithIdx(atom_id)
            for bond in atom.GetBonds():
                bonds.add(bond.GetIdx())
                other = bond.GetOtherAtomIdx(atom_id)
                if other not in atoms:
                    next_frontier.add(other)
                atoms.add(other)
        frontier = next_frontier
    return Chem.MolFragmentToSmarts(
        mol,
        atomsToUse=sorted(atoms),
        bondsToUse=sorted(bonds),
        isomericSmarts=True,
    )


def bond_metadata(smiles: str) -> tuple[Chem.Mol, list[dict[str, object]]]:
    mol = canonicalized_mol(smiles)
    metadata: list[dict[str, object]] = []
    for token_id, bond in enumerate(mol.GetBonds()):
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        metadata.append(
            {
                "token_id": token_id,
                "begin_atom": begin,
                "end_atom": end,
                "begin_element": mol.GetAtomWithIdx(begin).GetSymbol(),
                "end_element": mol.GetAtomWithIdx(end).GetSymbol(),
                "bond_type": str(bond.GetBondType()),
                "aromatic": bool(bond.GetIsAromatic()),
                "conjugated": bool(bond.GetIsConjugated()),
                "in_ring": bool(bond.IsInRing()),
                "radius1_smarts": atom_environment_smarts(mol, (begin, end), 1),
                "radius2_smarts": atom_environment_smarts(mol, (begin, end), 2),
            }
        )
    return mol, metadata


def make_kpgt_inputs(
    smiles: str,
    descriptor_generator: RDKit2DNormalized,
    n_virtual_nodes: int = 2,
):
    graph = smiles_to_graph_tune(
        smiles, max_length=5, n_virtual_nodes=n_virtual_nodes
    )
    if graph is None:
        raise ValueError(f"KPGT could not featurize: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    fp = torch.tensor(
        list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)),
        dtype=torch.float32,
    ).reshape(1, -1)
    descriptor_row = descriptor_generator.process(smiles)
    md = np.asarray(descriptor_row[1:], dtype=np.float32)
    md = np.nan_to_num(md, nan=0.0, posinf=0.0, neginf=0.0)
    return graph, fp, torch.from_numpy(md).reshape(1, -1)


def build_model(model_path: Path = None) -> LiGhTPredictor:
    config = config_dict["base"]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    model = LiGhTPredictor(
        d_node_feats=config["d_node_feats"],
        d_edge_feats=config["d_edge_feats"],
        d_g_feats=config["d_g_feats"],
        d_hpath_ratio=config["d_hpath_ratio"],
        n_mol_layers=config["n_mol_layers"],
        path_length=config["path_length"],
        n_heads=config["n_heads"],
        n_ffn_dense_layers=config["n_ffn_dense_layers"],
        input_drop=0,
        attn_drop=0,
        feat_drop=0,
        n_node_types=vocab.vocab_size,
    )
    if model_path is not None:
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {key.replace("module.", ""): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def extract_one(model: LiGhTPredictor, graph, local_graph, fp, md):
    indicators = graph.ndata["vavn"]
    node_h = model.node_emb(graph.ndata["begin_end"], indicators)
    edge_h = model.edge_emb(graph.ndata["edge"], indicators)

    def forward_with_priors(run_fp, run_md):
        run_input_h = model.triplet_emb(
            node_h, edge_h, run_fp, run_md, indicators
        )
        run_contextual_h = model.model(graph, run_input_h.clone())
        return run_input_h, run_contextual_h

    input_h, contextual_h = forward_with_priors(fp, md)
    _, contextual_zero_priors_h = forward_with_priors(
        torch.zeros_like(fp), torch.zeros_like(md)
    )
    real_bonds = indicators == 0
    prior_delta = contextual_h[real_bonds] - contextual_zero_priors_h[real_bonds]

    # Stronger ablation: remove the FP/MD virtual nodes and their attention
    # edges entirely.  Calling TripletEmbedding.forward would still try to
    # assign virtual-node projections, so apply only its shared local in_proj.
    local_indicators = local_graph.ndata["vavn"]
    local_node_h = model.node_emb(
        local_graph.ndata["begin_end"], local_indicators
    )
    local_edge_h = model.edge_emb(local_graph.ndata["edge"], local_indicators)
    local_input_h = model.triplet_emb.in_proj(
        torch.cat([local_node_h, local_edge_h], dim=-1)
    )
    contextual_no_virtual_h = model.model(local_graph, local_input_h.clone())
    local_real_bonds = local_indicators == 0
    if int(real_bonds.sum()) != int(local_real_bonds.sum()):
        raise RuntimeError("Virtual-node ablation changed the real bond count")
    return (
        input_h[real_bonds].cpu().numpy().astype(np.float32),
        contextual_h[real_bonds].cpu().numpy().astype(np.float32),
        contextual_zero_priors_h[real_bonds].cpu().numpy().astype(np.float32),
        contextual_no_virtual_h[local_real_bonds]
        .cpu()
        .numpy()
        .astype(np.float32),
        prior_delta.cpu().numpy().astype(np.float32),
        indicators.cpu().numpy(),
    )


def safe_name(item_id: str, index: int) -> str:
    clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in item_id)
    return f"{index:05d}_{clean[:80]}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_inputs(args)
    if args.featurization_only and args.random_init_smoke:
        raise ValueError("Choose only one smoke-test mode")
    if (
        not args.featurization_only
        and not args.random_init_smoke
        and args.model_path is None
    ):
        raise ValueError("--model-path is required unless --featurization-only is set")
    model = None if args.featurization_only else build_model(args.model_path)
    descriptor_generator = RDKit2DNormalized()
    manifest: list[dict[str, object]] = []

    for index, (item_id, smiles) in enumerate(rows):
        record: dict[str, object] = {"id": item_id, "smiles": smiles}
        try:
            canonical_mol, bonds = bond_metadata(smiles)
            graph, fp, md = make_kpgt_inputs(smiles, descriptor_generator)
            local_graph = smiles_to_graph_tune(
                smiles, max_length=5, n_virtual_nodes=0
            )
            if local_graph is None:
                raise ValueError(f"KPGT local graph failed: {smiles}")
            indicators = graph.ndata["vavn"].cpu().numpy()
            n_real_bonds = int(np.sum(indicators == 0))
            if len(bonds) != n_real_bonds:
                raise RuntimeError(
                    f"Bond/token mismatch: {len(bonds)} bonds vs "
                    f"{n_real_bonds} tokens"
                )
            stem = safe_name(item_id, index)
            token_dim = None
            if model is not None:
                (
                    input_tokens,
                    contextual_tokens,
                    contextual_zero_priors_tokens,
                    contextual_no_virtual_tokens,
                    global_prior_delta,
                    indicators,
                ) = extract_one(model, graph, local_graph, fp, md)
                token_dim = int(contextual_tokens.shape[1])
                np.savez_compressed(
                    args.output_dir / f"{stem}.npz",
                    input_bond_tokens=input_tokens,
                    contextual_bond_tokens=contextual_tokens,
                    contextual_bond_tokens_zero_global_priors=(
                        contextual_zero_priors_tokens
                    ),
                    contextual_bond_tokens_no_virtual_nodes=(
                        contextual_no_virtual_tokens
                    ),
                    global_prior_delta=global_prior_delta,
                    vavn_indicators=indicators,
                )
            with (args.output_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "id": item_id,
                        "input_smiles": smiles,
                        "canonical_smiles": Chem.MolToSmiles(canonical_mol, isomericSmiles=True),
                        "bonds": bonds,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            if model is None:
                mode = "featurization"
            elif args.random_init_smoke:
                mode = "random_init_smoke_not_scientific"
            else:
                mode = "checkpoint"
            record.update(
                {
                    "status": "ok",
                    "n_bonds": len(bonds),
                    "token_dim": token_dim,
                    "fingerprint_dim": int(fp.shape[1]),
                    "descriptor_dim": int(md.shape[1]),
                    "mode": mode,
                    "file_stem": stem,
                }
            )
        except Exception as exc:  # keep smoke runs auditable molecule-by-molecule
            record.update({"status": "error", "error": repr(exc)})
        manifest.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    failures = sum(row["status"] != "ok" for row in manifest)
    if failures:
        raise SystemExit(f"{failures}/{len(manifest)} molecules failed")


if __name__ == "__main__":
    main()
