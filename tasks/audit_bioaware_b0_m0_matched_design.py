#!/usr/bin/env python
"""Build the model-free matched design for BioAware reaction identifiability.

The primary unit is one observed Rhea-neighbour pair plus one chemically and
observationally matched, unrecorded non-neighbour.  Controls are produced by a
closed-cycle permutation of the observed Rhea target slots.  Therefore the
matched subset preserves the target marginal exactly even when some reaction
contrasts cannot be matched.  This stage deliberately does not read embedding
values and does not fit a classifier.  Its sole job is to establish whether a
future reaction-signal comparison is identifiable.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation.bioaware_relations import typed_reaction_pairs  # noqa: E402


CONTINUOUS = (
    "tanimoto", "mass_delta", "heavy_atom_delta", "element_l1_delta",
    "carbon_delta", "nitrogen_delta", "oxygen_delta", "phosphorus_delta",
    "sulfur_delta", "halogen_delta", "ring_delta",
    "hbd_delta", "hba_delta", "target_log_degree", "target_log_spectra",
)
CATEGORICAL = (
    "same_formula", "shared_instrument", "shared_adduct", "charge_delta",
    "rare_element_signature",
)
P3_MANIFEST_FILES = (
    "p3_main_real_pristine_manifest.json",
    "p3_isomer_real_pristine_manifest.json",
    "p3_near_core_real_pristine_manifest.json",
    "p3_nearmid_real_pristine_manifest.json",
    "p3_isomer_real_exposed_extension_manifest.json",
    "p3_sim_to_real_secondary_manifest.json",
)
TANIMOTO_CALIPER_LADDER = (0.04, 0.06, 0.08, 0.10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
        for value in values
    ], dtype=str)


def sorted_take(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    """h5py fancy indexing requires increasing unique coordinates."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("universe rows must be unique")
    order = np.argsort(rows, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(dataset[rows[order]])[inverse]


def allowed_training_rows(path: Path) -> np.ndarray:
    body = json.loads(path.read_text(encoding="utf-8"))
    section = body.get("real_train_primary")
    if not isinstance(section, dict) or not isinstance(section.get("rows"), list):
        raise RuntimeError("allowed-training manifest lacks real_train_primary.rows")
    rows = np.asarray(section["rows"], dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise RuntimeError("invalid allowed-training row vector")
    if int(section.get("n_rows", -1)) != len(rows):
        raise RuntimeError("allowed-training n_rows does not match row vector")
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("allowed-training rows are not unique")
    return rows


def _elements(mol: Chem.Mol) -> dict[str, int]:
    counts = defaultdict(int)
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1
    return {
        "C": counts["C"], "N": counts["N"], "O": counts["O"],
        "P": counts["P"], "S": counts["S"],
        "X": sum(counts[value] for value in ("F", "Cl", "Br", "I")),
    }


def molecular_universe(hdf5_path: Path, rows: np.ndarray):
    with h5py.File(hdf5_path, "r") as handle:
        ik14 = np.asarray([value[:14] for value in decode(sorted_take(handle["INCHIKEY"], rows))])
        formula = decode(sorted_take(handle["FORMULA"], rows))
        smiles = decode(sorted_take(handle["smiles"], rows))
        instrument = decode(sorted_take(handle["INSTRUMENT_TYPE"], rows))
        adduct = decode(sorted_take(handle["adduct"], rows))
    frame = pd.DataFrame({
        "ik14": ik14, "formula": formula, "smiles": smiles,
        "instrument": instrument, "adduct": adduct,
    })
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    records, fingerprints = [], {}
    for identity, group in frame.groupby("ik14", sort=True):
        if not identity or identity.lower() == "nan" or group.formula.nunique() != 1:
            continue
        molecule = None
        selected_smiles = ""
        for value in group.smiles.astype(str):
            candidate = Chem.MolFromSmiles(value)
            if candidate is not None:
                molecule, selected_smiles = candidate, value
                break
        if molecule is None:
            continue
        identity = str(identity)
        elements = _elements(molecule)
        fingerprints[identity] = generator.GetFingerprint(molecule)
        records.append({
            "ik14": identity,
            "formula": str(group.formula.iloc[0]),
            "smiles": selected_smiles,
            "exact_mass": float(Descriptors.ExactMolWt(molecule)),
            "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
            "rings": int(rdMolDescriptors.CalcNumRings(molecule)),
            "formal_charge": int(Chem.GetFormalCharge(molecule)),
            "hbd": int(Lipinski.NumHDonors(molecule)),
            "hba": int(Lipinski.NumHAcceptors(molecule)),
            "carbon": elements["C"], "nitrogen": elements["N"],
            "oxygen": elements["O"], "phosphorus": elements["P"],
            "sulfur": elements["S"], "halogen": elements["X"],
            "spectra": int(len(group)),
            "instruments": tuple(sorted(set(group.instrument.astype(str)) - {"", "nan"})),
            "adducts": tuple(sorted(set(group.adduct.astype(str)) - {"", "nan"})),
        })
    result = pd.DataFrame(records)
    if result.ik14.duplicated().any():
        raise RuntimeError("molecular universe contains duplicate identities")
    return result, fingerprints


def p3_identities(p3_dir: Path) -> set[str]:
    output: set[str] = set()
    missing = [name for name in P3_MANIFEST_FILES if not (p3_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"P3 is incomplete; missing manifests: {missing}")
    for name in P3_MANIFEST_FILES:
        path = p3_dir / name
        body = json.loads(path.read_text(encoding="utf-8"))
        stack = [body]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                if "ik14" in value and isinstance(value["ik14"], str):
                    output.add(value["ik14"][:14])
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    if not output:
        raise RuntimeError("P3 manifests contain no query identities")
    return output


def transformation(source: dict, target: dict, similarity: float, degree: dict[str, int]) -> dict:
    element_names = ("carbon", "nitrogen", "oxygen", "phosphorus", "sulfur", "halogen")
    element_delta = {name: abs(int(target[name]) - int(source[name])) for name in element_names}
    target_degree = int(degree.get(str(target["ik14"]), 0))
    target_spectra = int(target["spectra"])
    degree_bin = 0 if target_degree == 0 else (1 if target_degree == 1 else (
        2 if target_degree <= 3 else (3 if target_degree <= 7 else 4)
    ))
    spectra_bin = 0 if target_spectra == 1 else (1 if target_spectra <= 3 else (
        2 if target_spectra <= 7 else (3 if target_spectra <= 15 else 4)
    ))
    return {
        "tanimoto": float(similarity),
        "mass_delta": abs(float(target["exact_mass"]) - float(source["exact_mass"])),
        "heavy_atom_delta": abs(int(target["heavy_atoms"]) - int(source["heavy_atoms"])),
        "element_l1_delta": int(sum(element_delta.values())),
        **{f"{name}_delta": value for name, value in element_delta.items()},
        "ring_delta": abs(int(target["rings"]) - int(source["rings"])),
        "charge_delta": abs(int(target["formal_charge"]) - int(source["formal_charge"])),
        "hbd_delta": abs(int(target["hbd"]) - int(source["hbd"])),
        "hba_delta": abs(int(target["hba"]) - int(source["hba"])),
        "target_log_degree": float(np.log1p(target_degree)),
        "target_log_spectra": float(np.log1p(target_spectra)),
        "target_degree_bin": int(degree_bin),
        "target_spectra_bin": int(spectra_bin),
        "rare_element_signature": ":".join(str(element_delta[name]) for name in (
            "phosphorus", "sulfur", "halogen",
        )),
        "same_formula": bool(source["formula"] == target["formula"]),
        "shared_instrument": bool(set(source["instruments"]) & set(target["instruments"])),
        "shared_adduct": bool(set(source["adducts"]) & set(target["adducts"])),
    }


def admissible(truth: dict, candidate: dict, tanimoto_caliper: float = 0.10) -> bool:
    if any(candidate[name] != truth[name] for name in CATEGORICAL):
        return False
    return bool(
        abs(candidate["tanimoto"] - truth["tanimoto"]) <= tanimoto_caliper
        and abs(candidate["mass_delta"] - truth["mass_delta"])
        <= max(1.0, 0.15 * max(truth["mass_delta"], 5.0))
        and abs(candidate["heavy_atom_delta"] - truth["heavy_atom_delta"]) <= 2
        and abs(candidate["element_l1_delta"] - truth["element_l1_delta"]) <= 3
        and abs(candidate["carbon_delta"] - truth["carbon_delta"]) <= 2
        and abs(candidate["nitrogen_delta"] - truth["nitrogen_delta"]) <= 2
        and abs(candidate["oxygen_delta"] - truth["oxygen_delta"]) <= 2
        and abs(candidate["ring_delta"] - truth["ring_delta"]) <= 2
        and abs(candidate["hbd_delta"] - truth["hbd_delta"]) <= 2
        and abs(candidate["hba_delta"] - truth["hba_delta"]) <= 2
    )


def match_cost(truth: dict, candidate: dict) -> float:
    scales = {
        "tanimoto": 0.05, "mass_delta": max(1.0, 0.10 * max(truth["mass_delta"], 5.0)),
        "heavy_atom_delta": 1.0, "element_l1_delta": 2.0,
        "carbon_delta": 1.0, "nitrogen_delta": 1.0, "oxygen_delta": 1.0,
        "phosphorus_delta": 1.0, "sulfur_delta": 1.0, "halogen_delta": 1.0,
        "ring_delta": 1.0, "hbd_delta": 1.0,
        "hba_delta": 1.0, "target_log_degree": 0.75, "target_log_spectra": 0.75,
    }
    return float(sum(abs(candidate[name] - truth[name]) / scales[name] for name in CONTINUOUS))


def canonical_edges(participants: pd.DataFrame, identities: set[str]):
    typed, audit = typed_reaction_pairs(participants, identities)
    aggregate = defaultdict(lambda: {"types": set(), "reactions": set()})
    for pair in typed:
        left, right = sorted((pair.identity_a, pair.identity_b))
        aggregate[(left, right)]["types"].add(pair.relation_type)
        aggregate[(left, right)]["reactions"].update(pair.reaction_ids)
    undirected_edges = [
        (left, right, ";".join(sorted(values["types"])), ";".join(sorted(values["reactions"])))
        for (left, right), values in sorted(aggregate.items())
    ]
    oriented_edges = []
    for edge_id, (left, right, relation_types, reaction_ids) in enumerate(undirected_edges):
        oriented_edges.append((edge_id, left, right, relation_types, reaction_ids))
        oriented_edges.append((edge_id, right, left, relation_types, reaction_ids))
    return undirected_edges, oriented_edges, audit


def build_matches(
    molecules: pd.DataFrame,
    fingerprints: dict,
    undirected_edges,
    oriented_edges,
    controls: int,
    tanimoto_caliper: float,
):
    """Globally rewire the observed target multiset under chemical calipers.

    Each oriented true edge contributes exactly one target slot.  The square
    Hungarian assignment either maps an edge to an admissible *other* target
    slot or maps it to its own diagonal, which means ``unmatched``.  A used
    off-diagonal slot therefore belongs to another used row: matched rows form
    closed permutation cycles.  This makes the true- and control-target slot
    multisets identical on the retained subset, rather than only when every
    oriented edge happens to be matchable.
    """
    meta = molecules.set_index("ik14").to_dict("index")
    neighbours = defaultdict(set)
    degree = defaultdict(int)
    for left, right, _, _ in undirected_edges:
        neighbours[left].add(right); neighbours[right].add(left)
        degree[left] += 1; degree[right] += 1
    n = len(oriented_edges)
    invalid_cost = 1e12
    dummy_cost = 1e8
    cost_matrix = np.full((n, n), invalid_cost, dtype=np.float64)
    candidate_payload: dict[tuple[int, int], dict] = {}
    truth_payload: list[dict] = []
    stage_totals = defaultdict(int)
    for row_index, (edge_id, source, truth_id, relation_types, reaction_ids) in enumerate(oriented_edges):
        truth = transformation(
            {"ik14": source, **meta[source]}, {"ik14": truth_id, **meta[truth_id]},
            DataStructs.TanimotoSimilarity(fingerprints[source], fingerprints[truth_id]), degree,
        )
        truth_payload.append(truth)
        stage = defaultdict(int)
        for column_index, (_other_edge, _other_source, candidate_id, _types, _reactions) in enumerate(oriented_edges):
            if candidate_id == source or candidate_id in neighbours[source]:
                continue
            stage["unrecorded_non_neighbour"] += 1
            candidate = transformation(
                {"ik14": source, **meta[source]}, {"ik14": candidate_id, **meta[candidate_id]},
                DataStructs.TanimotoSimilarity(fingerprints[source], fingerprints[candidate_id]), degree,
            )
            if any(candidate[name] != truth[name] for name in CATEGORICAL):
                continue
            stage["categorical_exact"] += 1
            if abs(candidate["tanimoto"] - truth["tanimoto"]) > tanimoto_caliper:
                continue
            stage["tanimoto_caliper"] += 1
            if abs(candidate["mass_delta"] - truth["mass_delta"]) > max(
                1.0, 0.15 * max(truth["mass_delta"], 5.0)
            ):
                continue
            stage["mass_caliper"] += 1
            if abs(candidate["heavy_atom_delta"] - truth["heavy_atom_delta"]) > 2:
                continue
            stage["heavy_atom_caliper"] += 1
            if abs(candidate["element_l1_delta"] - truth["element_l1_delta"]) > 3:
                continue
            stage["element_caliper"] += 1
            if any(abs(candidate[name] - truth[name]) > 2 for name in (
                "carbon_delta", "nitrogen_delta", "oxygen_delta", "ring_delta",
                "hbd_delta", "hba_delta",
            )):
                continue
            stage["common_element_and_function_caliper"] += 1
            if not admissible(truth, candidate, tanimoto_caliper=tanimoto_caliper):
                raise AssertionError("staged calipers disagree with admissible()")
            value = match_cost(truth, candidate)
            # Deterministic epsilon makes exact cost ties reproducible without
            # changing the scientific objective.
            cost_matrix[row_index, column_index] = value + 1e-10 * column_index
            candidate_payload[(row_index, column_index)] = candidate
        for name, value in stage.items():
            stage_totals[name] += value
        # The diagonal is an unmatched sentinel that also reserves this row's
        # target slot.  Independent dummy columns would break target-marginal
        # preservation whenever the assignment is incomplete.
        cost_matrix[row_index, row_index] = dummy_cost
        if (row_index + 1) % 50 == 0 or row_index + 1 == n:
            print(
                f"[B0-M0 graph] {row_index + 1:,}/{n:,} oriented contrasts; "
                f"admissible_pairs={len(candidate_payload):,}",
                flush=True,
            )
    print(
        f"[B0-M0 assignment] caliper={tanimoto_caliper:.2f}; "
        f"solving {n:,} x {n:,} closed-cycle assignment",
        flush=True,
    )
    assigned_rows, assigned_columns = linear_sum_assignment(cost_matrix)
    assignment = dict(zip(assigned_rows.tolist(), assigned_columns.tolist()))
    rows, unmatched = [], []
    for row_index, (edge_id, source, truth_id, relation_types, reaction_ids) in enumerate(oriented_edges):
        column_index = assignment[row_index]
        if column_index == row_index or cost_matrix[row_index, column_index] >= dummy_cost:
            unmatched.append({
                "source_identity": source, "truth_identity": truth_id,
                "available_controls": int(np.sum(cost_matrix[row_index, :n] < dummy_cost)),
                "relation_types": relation_types,
            })
            continue
        candidate_id = oriented_edges[column_index][2]
        candidate = candidate_payload[(row_index, column_index)]
        truth = truth_payload[row_index]
        common = {
            "group_id": len(rows) // (controls + 1), "reaction_edge_id": int(edge_id),
            "source_identity": source,
            "relation_types": relation_types, "reaction_ids": reaction_ids,
        }
        rows.append({
            **common, "target_identity": truth_id, "label": 1,
            "truth_target_slot": int(row_index), "match_cost": 0.0, **truth,
        })
        rows.append({
            **common, "target_identity": candidate_id, "label": 0,
            "control_target_slot": int(column_index),
            "match_cost": float(cost_matrix[row_index, column_index]), **candidate,
        })
    matched_row_indices = {
        int(row_index) for row_index, column_index in assignment.items()
        if column_index != row_index and cost_matrix[row_index, column_index] < dummy_cost
    }
    matched_column_indices = {
        int(column_index) for row_index, column_index in assignment.items()
        if column_index != row_index and cost_matrix[row_index, column_index] < dummy_cost
    }
    closed = matched_row_indices == matched_column_indices
    if not closed:
        raise RuntimeError("closed-cycle assignment failed to preserve target slots")
    assignment_audit = {
        "strategy": "square Hungarian assignment with diagonal unmatched sentinels",
        "tanimoto_caliper": float(tanimoto_caliper),
        "matched_rows": int(len(matched_row_indices)),
        "matched_target_slots": int(len(matched_column_indices)),
        "closed_target_slot_multiset": bool(closed),
    }
    return pd.DataFrame(rows), pd.DataFrame(unmatched), dict(stage_totals), assignment_audit


def balance(matches: pd.DataFrame) -> tuple[dict, dict]:
    smd, p90 = {}, {}
    for column in CONTINUOUS:
        positive = matches.loc[matches.label == 1, column].to_numpy(float)
        negative = matches.loc[matches.label == 0, column].to_numpy(float)
        pooled = np.sqrt((positive.var(ddof=1) + negative.var(ddof=1)) / 2.0)
        smd[column] = float(abs(positive.mean() - negative.mean()) / max(pooled, 1e-12))
        truth = matches.loc[matches.label == 1, ["group_id", column]].set_index("group_id")[column]
        joined = matches.loc[matches.label == 0, ["group_id", column]].copy()
        joined["truth"] = joined.group_id.map(truth)
        pooled_values = matches[column].to_numpy(float)
        scale = max(float(pooled_values.std(ddof=1)), 1e-12)
        p90[column] = float(np.quantile(np.abs(joined[column] - joined.truth) / scale, 0.90))
    categorical = {}
    for column in CATEGORICAL:
        truth = matches.loc[matches.label == 1, ["group_id", column]].set_index("group_id")[column]
        control = matches.loc[matches.label == 0, ["group_id", column]].copy()
        categorical[column] = int((control[column] != control.group_id.map(truth)).sum())
    return smd, {"continuous_p90": p90, "categorical_mismatches": categorical}


def self_test() -> None:
    truth = {
        "tanimoto": .5, "mass_delta": 10., "heavy_atom_delta": 1,
        "element_l1_delta": 2, "carbon_delta": 1, "nitrogen_delta": 0,
        "oxygen_delta": 1, "phosphorus_delta": 0, "sulfur_delta": 0,
        "halogen_delta": 0, "ring_delta": 0, "charge_delta": 0,
        "hbd_delta": 1, "hba_delta": 1, "target_log_degree": 1.,
        "target_log_spectra": 1., "same_formula": False,
        "shared_instrument": True, "shared_adduct": True,
        "rare_element_signature": "0:0:0", "target_degree_bin": 1,
        "target_spectra_bin": 1,
    }
    assert admissible(truth, dict(truth), tanimoto_caliper=0.04)
    broken = dict(truth); broken["same_formula"] = True
    assert not admissible(truth, broken)
    assert match_cost(truth, dict(truth)) == 0.0
    molecule = Chem.MolFromSmiles("CC")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
    fingerprint = generator.GetFingerprint(molecule)
    toy = pd.DataFrame([{
        "ik14": identity, "formula": "C2H6", "smiles": "CC",
        "exact_mass": float(Descriptors.ExactMolWt(molecule)), "heavy_atoms": 2,
        "rings": 0, "formal_charge": 0, "hbd": 0, "hba": 0,
        "carbon": 2, "nitrogen": 0, "oxygen": 0, "phosphorus": 0,
        "sulfur": 0, "halogen": 0, "spectra": 1,
        "instruments": ("Orbitrap",), "adducts": ("[M+H]+",),
    } for identity in ("A", "B", "C", "D")])
    toy_edges = [("A", "B", "unknown", "R1"), ("C", "D", "unknown", "R2")]
    toy_oriented = [
        (0, "A", "B", "unknown", "R1"), (0, "B", "A", "unknown", "R1"),
        (1, "C", "D", "unknown", "R2"), (1, "D", "C", "unknown", "R2"),
    ]
    toy_matches, toy_unmatched, _, toy_assignment = build_matches(
        toy, {identity: fingerprint for identity in ("A", "B", "C", "D")},
        toy_edges, toy_oriented, 1, 0.04,
    )
    assert len(toy_unmatched) == 0 and toy_matches.group_id.nunique() == 4
    assert toy_assignment["closed_target_slot_multiset"]
    positives = toy_matches.loc[toy_matches.label == 1, "truth_target_slot"].astype(int)
    negatives = toy_matches.loc[toy_matches.label == 0, "control_target_slot"].astype(int)
    assert sorted(positives) == sorted(negatives)
    known = {tuple(sorted(pair[:2])) for pair in toy_edges}
    controls = toy_matches[toy_matches.label == 0]
    assert all(
        tuple(sorted((row.source_identity, row.target_identity))) not in known
        for row in controls.itertuples(index=False)
    )
    print("[BioAware B0-M0 unit checks] PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--allowed-training", type=Path)
    parser.add_argument("--participants", type=Path, default=ROOT / "data/reference/bioaware_rhea_reactome_direction_20260830/rhea_participants.csv.gz")
    parser.add_argument("--p3-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/bioaware_b0_m0_matched_design_local")
    parser.add_argument("--controls-per-edge", type=int, default=1)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.allowed_training is None:
        parser.error("--allowed-training is required unless --self-test is used")
    if args.controls_per_edge != 1:
        raise ValueError("the revised B0-M0 primary design requires exactly one control")
    for path in (args.hdf5, args.allowed_training, args.participants):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.formal and (args.p3_dir is None or not args.p3_dir.is_dir()):
        raise FileNotFoundError("formal B0-M0 requires --p3-dir")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output {args.output_dir}")

    rows = allowed_training_rows(args.allowed_training)
    molecules, fingerprints = molecular_universe(args.hdf5, rows)
    identities = set(molecules.ik14.astype(str))
    overlap = set()
    if args.p3_dir is not None and args.p3_dir.is_dir():
        overlap = identities & p3_identities(args.p3_dir)
    participants = pd.read_csv(args.participants)
    undirected_edges, oriented_edges, reaction_audit = canonical_edges(participants, identities)
    designs = []
    for tanimoto_caliper in TANIMOTO_CALIPER_LADDER:
        matches_i, unmatched_i, stage_totals_i, assignment_i = build_matches(
            molecules, fingerprints, undirected_edges, oriented_edges,
            args.controls_per_edge, tanimoto_caliper,
        )
        if matches_i.empty:
            designs.append({
                "tanimoto_caliper": tanimoto_caliper, "matches": matches_i,
                "unmatched": unmatched_i, "stage_totals": stage_totals_i,
                "assignment": assignment_i, "smd": {}, "within": {},
                "source_ids": set(), "source_formulas": 0, "gates": {},
                "eligible": False, "maximum_smd": float("inf"),
            })
            continue
        counts = matches_i.groupby("group_id").label.agg(["sum", "count"])
        if not ((counts["sum"] == 1) & (counts["count"] == 2)).all():
            raise RuntimeError("incomplete or malformed 1:1 groups")
        positives = matches_i.loc[matches_i.label == 1, "truth_target_slot"].astype(int)
        negatives = matches_i.loc[matches_i.label == 0, "control_target_slot"].astype(int)
        if sorted(positives) != sorted(negatives):
            raise RuntimeError("retained true/control target-slot multisets differ")
        smd_i, within_i = balance(matches_i)
        for marginal_column in ("target_log_degree", "target_log_spectra"):
            if smd_i[marginal_column] > 1e-10:
                raise RuntimeError(
                    f"closed target marginal was not preserved for {marginal_column}: "
                    f"SMD={smd_i[marginal_column]}"
                )
        source_ids_i = set(matches_i.source_identity.astype(str))
        source_formula_i = molecules.set_index("ik14").loc[sorted(source_ids_i), "formula"]
        gates_i = {
            "groups_ge_500": bool(matches_i.group_id.nunique() >= 500),
            "source_identities_ge_300": bool(len(source_ids_i) >= 300),
            "source_formulas_ge_200": bool(source_formula_i.nunique() >= 200),
            "all_continuous_smd_le_0_10": bool(max(smd_i.values()) <= 0.10),
            "categorical_mismatches_zero": bool(
                max(within_i["categorical_mismatches"].values()) == 0
            ),
            "closed_target_marginal_preserved": bool(
                assignment_i["closed_target_slot_multiset"]
            ),
            "p3_identity_overlap_zero": bool(len(overlap) == 0),
        }
        designs.append({
            "tanimoto_caliper": tanimoto_caliper, "matches": matches_i,
            "unmatched": unmatched_i, "stage_totals": stage_totals_i,
            "assignment": assignment_i, "smd": smd_i, "within": within_i,
            "source_ids": source_ids_i,
            "source_formulas": int(source_formula_i.nunique()),
            "gates": gates_i, "eligible": bool(all(gates_i.values())),
            "maximum_smd": float(max(smd_i.values())),
        })
    eligible_designs = [design for design in designs if design["eligible"]]
    if eligible_designs:
        # Among balance-qualified designs, retain the largest sample.  This
        # choice uses covariates only; embeddings and downstream outcomes have
        # not been read.
        selected = sorted(
            eligible_designs,
            key=lambda value: (
                -value["matches"].group_id.nunique(), value["tanimoto_caliper"],
            ),
        )[0]
    else:
        # Preserve the best balance-only diagnostic without authorising a
        # signal test.  No threshold is relaxed after seeing an outcome.
        selected = sorted(
            designs,
            key=lambda value: (
                value["maximum_smd"], -value["matches"].group_id.nunique(),
            ),
        )[0]
    matches = selected["matches"]
    unmatched = selected["unmatched"]
    stage_totals = selected["stage_totals"]
    assignment_audit = selected["assignment"]
    smd = selected["smd"]
    within = selected["within"]
    source_ids = selected["source_ids"]
    source_formula_count = selected["source_formulas"]
    gates = selected["gates"]
    if matches.empty:
        raise RuntimeError("no complete matched groups in any frozen balance design")
    report = {
        "status": "bioaware_b0_m0_matched_design_complete",
        "formal": bool(args.formal),
        "model_fitted": False,
        "embedding_values_read": False,
        "universe": {"rows": int(len(rows)), "identities": int(len(identities)), "source": "P3 real_train_primary rows"},
        "reaction_pairs": {
            "undirected_available": int(len(undirected_edges)),
            "oriented_contrasts_available": int(len(oriented_edges)),
            **reaction_audit,
        },
        "matched": {
            "groups": int(matches.group_id.nunique()), "rows": int(len(matches)),
            "source_identities": int(len(source_ids)), "source_formulas": int(source_formula_count),
            "unmatched_oriented_contrasts": int(len(unmatched)), "controls_per_group": 1,
            "assignment": assignment_audit,
            "candidate_survival_totals_across_edges": {
                str(key): int(value) for key, value in stage_totals.items()
            },
        },
        "balance": {
            "standardised_mean_differences": smd,
            **within,
        },
        "balance_design_ladder": [{
            "tanimoto_caliper": float(design["tanimoto_caliper"]),
            "groups": int(design["matches"].group_id.nunique())
            if not design["matches"].empty else 0,
            "source_identities": int(len(design["source_ids"])),
            "source_formulas": int(design["source_formulas"]),
            "maximum_smd": None if not np.isfinite(design["maximum_smd"])
            else float(design["maximum_smd"]),
            "gates": design["gates"],
            "eligible": bool(design["eligible"]),
            "assignment": design["assignment"],
        } for design in designs],
        "selected_tanimoto_caliper": float(selected["tanimoto_caliper"]),
        "p3_identity_overlap": int(len(overlap)),
        "gates": gates,
        "pass_to_signal_test": bool(args.formal and selected["eligible"]),
        "contracts": {
            "source_fixed_within_group": True,
            "control_target_pool": "eligible observed Rhea-edge target-slot multiset",
            "controls_from_full_observed_universe": False,
            "controls_restricted_to_rhea_covered_non_neighbours": True,
            "both_orientations_per_undirected_reaction_edge": True,
            "controls_are_unrecorded_non_neighbours": True,
            "matched_subset_target_slots_preserved_exactly": True,
            "balance_design_selection_uses_embeddings_or_outcomes": False,
            "outcome_or_embedding_used_for_matching": False,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "provenance": {
            "hdf5_sha256": sha256(args.hdf5),
            "allowed_training_sha256": sha256(args.allowed_training),
            "participants_sha256": sha256(args.participants),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "A balance and identifiability preflight only. Passing does not show reaction signal, "
            "candidate-ranking gain, biological mechanism, or shared-embedding improvement."
        ),
    }
    args.output_dir.mkdir(parents=True)
    matches.to_csv(args.output_dir / "matched_groups.csv.gz", index=False, compression="gzip")
    unmatched.to_csv(args.output_dir / "unmatched_edges.csv.gz", index=False, compression="gzip")
    ladder_path = args.output_dir / "balance_design_ladder.json"
    ladder_path.write_text(
        json.dumps(report["balance_design_ladder"], indent=2), encoding="utf-8"
    )
    report["provenance"]["matched_groups_sha256"] = sha256(args.output_dir / "matched_groups.csv.gz")
    report["provenance"]["balance_design_ladder_sha256"] = sha256(ladder_path)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
