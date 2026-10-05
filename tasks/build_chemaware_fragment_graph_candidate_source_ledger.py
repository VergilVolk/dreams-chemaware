"""Build a truth-blind candidate ledger from connected fragment-graph evidence.

No candidate label, retrieval rank, or DreaMS embedding is read.  Candidate
structures are expanded with the pinned MAGMa enumerator and scored by a
rooted connected explanation of the observed query peaks.  Three equal-input
controls are exported with every score: within-candidate fragment-assignment
permutation, parent-topology permutation, and deterministic peak-mass shift.

The output is deliberately *unqualified*.  It must pass the existing
``qualify_chemaware_candidate_source_ledger.py`` formula-disjoint confirmation
gate before it may select native DreaMS triplets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from chemaware_fragment_graph_core import (
    SUPPORTED_ADDUCTS,
    ConnectedExplanation,
    FragmentGraph,
    build_fragment_graph,
    explain_candidate_set,
    fragment_assignment_control,
    fragment_topology_control,
    mass_shift_control,
    source_hashes,
    stable_u64,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_HASHES = {
    "fragmentation.py": "5fc6ac347a09ea8486154ce827eea0ea810a0c7f6a277279ffd4c5abd1e93744",
    "chem_utils.py": "c9842ef76eb831394c53fdd5050d87842edaebd46427a722d43983286495f96e",
}
SOURCE_FAMILY = "fragment_graph_connected_all_candidates"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", type=Path,
        default=(ROOT / "data/validation/chemaware_high_coverage_native"
                 / "run_2340524/evidence/train_triplet_evidence.npz"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=(ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1"
                 / "manifest.npz"),
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--ms-pred-source-root", type=Path,
        default=ROOT / "data/external/ms-pred-src",
    )
    parser.add_argument(
        "--evidence-contract", type=Path,
        default=ROOT / "dreams/models/chem_aware/fragment_graph_evidence_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-peaks", type=int, default=64)
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--floor-da", type=float, default=0.01)
    parser.add_argument("--max-tree-depth", type=int, default=3)
    parser.add_argument("--max-broken-bonds", type=int, default=6)
    parser.add_argument("--edge-penalty-scale", type=float, default=0.025)
    parser.add_argument("--coherence-scale", type=float, default=0.5)
    parser.add_argument("--minimum-matched-peaks", type=int, default=2)
    parser.add_argument(
        "--maximum-queries", type=int,
        help="Deterministic engineering subset only; omitted for the formal full ledger.",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def instrument_profile(value: str) -> str:
    normalized = str(value).strip().lower()
    if "orbitrap" in normalized:
        return "orbitrap"
    if "qtof" in normalized or "q-tof" in normalized or normalized == "tof":
        return "qtof"
    return "default"


def connectivity_smiles(value: str) -> str:
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise ValueError(f"RDKit could not parse candidate SMILES: {value!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def choose_structure_representative(values: set[str]) -> str:
    canonical = {connectivity_smiles(value) for value in values}
    if not canonical:
        raise ValueError("candidate identity has no structure")

    def charge_key(value: str) -> tuple[int, int, str]:
        molecule = Chem.MolFromSmiles(value)
        assert molecule is not None
        charges = [atom.GetFormalCharge() for atom in molecule.GetAtoms()]
        return sum(abs(charge) for charge in charges), sum(charge != 0 for charge in charges), value

    return min(canonical, key=charge_key)


def structure_formula(value: str) -> str:
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise ValueError(f"RDKit could not parse candidate SMILES: {value!r}")
    return rdMolDescriptors.CalcMolFormula(molecule)


def candidate_reference_rows(
    manifest: dict[str, np.ndarray], query: int, local_candidate: int,
) -> np.ndarray:
    left, right = map(int, manifest["query_ptr"][query:query + 2])
    molecule = left + int(local_candidate)
    if not left <= molecule < right:
        raise IndexError(f"candidate {local_candidate} outside query {query}")
    pair_left, pair_right = map(int, manifest["molecule_ptr"][molecule:molecule + 2])
    return np.asarray(
        manifest["pair_candidate_row"][pair_left:pair_right], dtype=np.int64,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def spectrum_peaks(
    spectrum: np.ndarray,
    precursor_mz: float,
    top_peaks: int,
) -> tuple[np.ndarray, np.ndarray]:
    mz = np.asarray(spectrum[0], dtype=np.float64)
    intensity = np.asarray(spectrum[1], dtype=np.float64)
    keep = (
        np.isfinite(mz) & np.isfinite(intensity)
        & (mz > 0) & (intensity > 0) & (mz < precursor_mz + 0.05)
    )
    mz, intensity = mz[keep], intensity[keep]
    if len(mz) > top_peaks:
        selected = np.argsort(-intensity, kind="stable")[:top_peaks]
        mz, intensity = mz[selected], intensity[selected]
    order = np.argsort(mz, kind="stable")
    return mz[order], intensity[order]


def deterministic_queries(queries: Sequence[int], maximum: int | None) -> list[int]:
    values = sorted(map(int, queries))
    if maximum is None:
        return values
    if maximum < 1:
        raise ValueError("maximum queries must be positive")
    ranked = sorted(values, key=lambda query: (stable_u64(query, seed=20260928), query))
    return sorted(ranked[:maximum])


def _nullable(value: float | None) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.17g}"


def _explanation_row(
    query: int,
    candidate: int,
    arm: str,
    explanation: ConnectedExplanation,
    graph: FragmentGraph,
    adduct: str,
    instrument_profile: str,
    collision_energy: str,
) -> dict[str, object]:
    return {
        "manifest_query": query,
        "local_candidate": candidate,
        "arm": arm,
        "adduct": adduct,
        "instrument_profile": instrument_profile,
        "collision_energy": collision_energy,
        "score": f"{explanation.score:.17g}",
        "objective": f"{explanation.objective:.17g}",
        "reward": f"{explanation.reward:.17g}",
        "complexity_cost": f"{explanation.complexity_cost:.17g}",
        "matched_peaks": explanation.matched_peaks,
        "matched_nodes": explanation.matched_nodes,
        "selected_edges": explanation.selected_edges,
        "explained_weight_fraction": f"{explanation.explained_weight_fraction:.17g}",
        "maximum_depth": explanation.maximum_depth,
        "regular_matches": explanation.regular_matches,
        "semiresolved_matches": explanation.semiresolved_matches,
        "ring_fission_edges": explanation.ring_fission_edges,
        "charge_migration_edges": explanation.charge_migration_edges,
        "observed_lineage_pairs": explanation.observed_lineage_pairs,
        "shared_branch_pairs": explanation.shared_branch_pairs,
        "coherence_bonus": f"{explanation.coherence_bonus:.17g}",
        "graph_nodes": len(graph.nodes),
        "graph_variants": len(graph.variant_mz),
        "matched_peak_indices": ";".join(
            str(match.peak_index) for match in explanation.peak_matches
        ),
        "matched_node_keys": ";".join(
            graph.nodes[match.node_index].key for match in explanation.peak_matches
        ),
        "hydrogen_shifts": ";".join(
            str(match.hydrogen_shift) for match in explanation.peak_matches
        ),
        "hydrogen_rule_labels": ";".join(
            match.rule for match in explanation.peak_matches
        ),
        "mass_errors_da": ";".join(
            f"{match.error_da:.9g}" for match in explanation.peak_matches
        ),
    }


def _score_arm(
    graphs: Sequence[FragmentGraph],
    mz: np.ndarray,
    intensity: np.ndarray,
    args: argparse.Namespace,
) -> tuple[ConnectedExplanation, ...]:
    return explain_candidate_set(
        graphs, mz, intensity,
        ppm=args.ppm,
        floor_da=args.floor_da,
        edge_penalty_scale=args.edge_penalty_scale,
        coherence_scale=args.coherence_scale,
        minimum_matched_peaks=args.minimum_matched_peaks,
    )


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.top_peaks < 16 or args.top_peaks > 256:
        raise ValueError("top-peaks must be in [16, 256]")
    if not 5 <= args.ppm <= 30 or not 0.001 <= args.floor_da <= 0.02:
        raise ValueError("mass tolerance falls outside the preregistered range")
    if not 0 < args.edge_penalty_scale <= 0.1:
        raise ValueError("edge penalty scale falls outside the preregistered range")
    if not 0 < args.coherence_scale <= 1:
        raise ValueError("coherence scale falls outside the preregistered range")
    if args.minimum_matched_peaks < 2:
        raise ValueError("connected evidence requires at least two observed peaks")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")

    evidence = json.loads(args.evidence_contract.read_text(encoding="utf-8"))
    if evidence.get("schema") != "chemaware.fragment-graph-evidence.v1":
        raise RuntimeError("fragment-graph evidence contract is not recognized")
    observed_source_hashes = source_hashes(args.ms_pred_source_root)
    if observed_source_hashes != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(
            "pinned ms-pred source hash mismatch: "
            f"expected={EXPECTED_SOURCE_HASHES} observed={observed_source_hashes}"
        )

    evidence_rows = load_npz(args.evidence)
    manifest = load_npz(args.manifest)
    if "query" not in evidence_rows:
        raise RuntimeError("training evidence lacks its formal query registry")
    formal_queries = np.asarray(evidence_rows["query"], dtype=np.int64)
    if len(formal_queries) != 4032 or len(np.unique(formal_queries)) != 4032:
        raise RuntimeError(
            f"expected 4,032 unique formula-role-0/1 queries, got {len(formal_queries)}"
        )
    formal_formulas = np.asarray(manifest["query_formula"])[formal_queries].astype(str)
    if len(np.unique(formal_formulas)) != 2518:
        raise RuntimeError(
            f"expected 2,518 formula-role-0/1 formulas, got {len(np.unique(formal_formulas))}"
        )
    if np.any(formal_queries < 0) or np.any(formal_queries >= len(manifest["query_row"])):
        raise RuntimeError("formal training queries fall outside the corrected manifest")
    sampled_queries = deterministic_queries(formal_queries, args.maximum_queries)
    selected_queries = [
        query for position, query in enumerate(sampled_queries)
        if position % args.shard_count == args.shard_index
    ]
    if not selected_queries:
        raise RuntimeError("fragment-graph shard contains no queries")
    graph_cache: dict[str, FragmentGraph] = {}
    structure_cache: dict[str, tuple[str, str]] = {}

    def graph_for(smiles: str) -> FragmentGraph:
        if smiles not in graph_cache:
            graph_cache[smiles] = build_fragment_graph(
                smiles,
                str(args.ms_pred_source_root.resolve()),
                args.max_tree_depth,
                args.max_broken_bonds,
            )
        return graph_cache[smiles]

    output_rows: list[dict[str, object]] = []
    explanation_rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    audit = Counter()
    active_queries = 0
    with h5py.File(args.data, "r") as data:
        for ordinal, query in enumerate(selected_queries, start=1):
            spectrum_row = int(manifest["query_row"][query])
            adduct = text(data["adduct"][spectrum_row])
            profile = instrument_profile(text(data["INSTRUMENT_TYPE"][spectrum_row]))
            collision = float(data["COLLISION_ENERGY"][spectrum_row])
            collision_energy = "" if not np.isfinite(collision) else f"{collision:.8g}"
            left, right = map(int, manifest["query_ptr"][query:query + 2])
            local_rows: list[dict[str, object]] = []
            for local_candidate, molecule in enumerate(range(left, right)):
                ik14 = str(manifest["molecule_ik14"][molecule])
                formula = str(manifest["molecule_formula"][molecule])
                reference_rows = candidate_reference_rows(
                    manifest, query, local_candidate,
                )
                if not len(reference_rows):
                    raise RuntimeError(
                        f"query {query} candidate {local_candidate} has no reference spectra"
                    )
                if ik14 not in structure_cache:
                    full_keys = {
                        text(data["INCHIKEY"][int(row)]) for row in reference_rows
                    }
                    if any(key[:14] != ik14 for key in full_keys):
                        raise RuntimeError(
                            f"query {query} candidate {local_candidate} InChIKey drift"
                        )
                    smiles = choose_structure_representative({
                        text(data["smiles"][int(row)]) for row in reference_rows
                    })
                    observed_formula = structure_formula(smiles)
                    if observed_formula != formula:
                        raise RuntimeError(
                            f"query {query} candidate {local_candidate} structure/formula "
                            f"mismatch: {observed_formula} versus {formula}"
                        )
                    structure_cache[ik14] = (smiles, formula)
                smiles, cached_formula = structure_cache[ik14]
                if cached_formula != formula:
                    raise RuntimeError(f"connectivity {ik14} maps to multiple formulas")
                local_rows.append({
                    "ik14": ik14,
                    "formula": formula,
                    "smiles": smiles,
                })
            scores: list[ConnectedExplanation] | None = None
            assignment_control: list[ConnectedExplanation] | None = None
            topology_control: list[ConnectedExplanation] | None = None
            mass_control: list[ConnectedExplanation] | None = None
            actual_graphs: list[FragmentGraph] = []
            control_graphs: list[FragmentGraph] = []
            topology_graphs: list[FragmentGraph] = []
            reason = ""
            if adduct not in SUPPORTED_ADDUCTS:
                reason = f"unsupported_adduct:{adduct}"
                audit["unsupported_adduct_queries"] += 1
            else:
                try:
                    actual_graphs = [graph_for(str(row["smiles"])) for row in local_rows]
                    control_graphs = [
                        fragment_assignment_control(graph, (query, local))
                        for local, graph in enumerate(actual_graphs)
                    ]
                    topology_graphs = [
                        fragment_topology_control(graph, (query, local))
                        for local, graph in enumerate(actual_graphs)
                    ]
                    precursor_mz = float(data["precursor_mz"][spectrum_row])
                    mz, intensity = spectrum_peaks(
                        np.asarray(data["spectrum"][spectrum_row], dtype=np.float64),
                        precursor_mz,
                        args.top_peaks,
                    )
                    if len(mz) < args.minimum_matched_peaks:
                        reason = "insufficient_observed_peaks"
                        audit["insufficient_peak_queries"] += 1
                    else:
                        scores = list(_score_arm(actual_graphs, mz, intensity, args))
                        assignment_control = list(
                            _score_arm(control_graphs, mz, intensity, args)
                        )
                        topology_control = list(
                            _score_arm(topology_graphs, mz, intensity, args)
                        )
                        mass_control = list(_score_arm(
                            actual_graphs,
                            mass_shift_control(mz, precursor_mz, query),
                            intensity,
                            args,
                        ))
                except (RuntimeError, ValueError) as error:
                    reason = f"fragment_graph_error:{type(error).__name__}:{error}"
                    audit["fragment_graph_error_queries"] += 1

            query_scores = []
            source_discriminative = bool(
                scores is not None
                and len(scores) > 1
                and np.ptp(np.asarray([value.score for value in scores])) > 0
            )
            if scores is not None and not source_discriminative:
                reason = "nondiscriminative_source_abstention"
                audit["nondiscriminative_source_queries"] += 1
            for local, row in enumerate(local_rows):
                available = all(value is not None for value in (
                    scores, assignment_control, topology_control, mass_control,
                )) and source_discriminative
                source_score = scores[local].score if available and scores is not None else None
                assignment_score = (
                    assignment_control[local].score
                    if available and assignment_control is not None else None
                )
                topology_score = (
                    topology_control[local].score
                    if available and topology_control is not None else None
                )
                mass_score = mass_control[local].score if available and mass_control is not None else None
                output_rows.append({
                    "manifest_query": query,
                    "local_candidate": local,
                    "ik14": row["ik14"],
                    "formula": row["formula"],
                    "source_family": SOURCE_FAMILY,
                    "scope": "all_candidates",
                    "source_score": _nullable(source_score),
                    "control_fragment_assignment_permuted_score": _nullable(assignment_score),
                    "control_fragment_topology_permuted_score": _nullable(topology_score),
                    "control_mass_shifted_score": _nullable(mass_score),
                    "controls_available": int(available),
                })
                if available:
                    query_scores.append(float(source_score))
                    assert scores is not None
                    assert assignment_control is not None
                    assert topology_control is not None
                    assert mass_control is not None
                    explanation_rows.extend((
                        _explanation_row(
                            query, local, "correct_structure", scores[local],
                            actual_graphs[local], adduct, profile,
                            collision_energy,
                        ),
                        _explanation_row(
                            query, local, "fragment_assignment_permuted",
                            assignment_control[local], control_graphs[local], adduct,
                            profile, collision_energy,
                        ),
                        _explanation_row(
                            query, local, "fragment_topology_permuted",
                            topology_control[local], topology_graphs[local], adduct,
                            profile, collision_energy,
                        ),
                        _explanation_row(
                            query, local, "mass_shifted",
                            mass_control[local], actual_graphs[local], adduct,
                            profile, collision_energy,
                        ),
                    ))
                else:
                    exclusions.append({
                        "manifest_query": query,
                        "local_candidate": local,
                        "ik14": row["ik14"],
                        "reason": reason,
                    })
            if query_scores:
                active_queries += 1
            if ordinal % 25 == 0 or ordinal == len(selected_queries):
                print(
                    f"FRAGMENT_GRAPH_PROGRESS queries={ordinal}/{len(selected_queries)} "
                    f"graphs={len(graph_cache)} active_queries={active_queries}",
                    flush=True,
                )

    if not output_rows:
        raise RuntimeError("fragment-graph candidate ledger is empty")
    report = {
        "status": "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_UNQUALIFIED",
        "truth_fields_exported": False,
        "source_families": {
            SOURCE_FAMILY: {
                "scope": "all_candidates",
                "larger_is_better": True,
                "confidence_tier": "C_calibration_required",
                "matched_controls": [
                    "within_candidate_fragment_assignment_permutation",
                    "within_candidate_fragment_topology_permutation",
                    "deterministic_mass_shift",
                ],
                "specificity_gate_passed": False,
            }
        },
        "queries": len(selected_queries),
        "candidates": len(output_rows),
        "candidate_source_rows": len(output_rows),
        "active_queries": active_queries,
        "unique_fragment_graphs": len(graph_cache),
        "excluded_candidate_rows": len(exclusions),
        "audit": dict(audit),
        "scope": "all_candidates",
        "formal_triplet_mining_authorized": False,
        "engineering_subset": args.maximum_queries is not None,
        "shard": {
            "count": args.shard_count,
            "index": args.shard_index,
            "selection_order": "stable_hash_then_modulo_then_manifest_sort",
            "pre_shard_query_count": len(sampled_queries),
        },
        "configuration": {
            "top_peaks": args.top_peaks,
            "ppm": args.ppm,
            "floor_da": args.floor_da,
            "max_tree_depth": args.max_tree_depth,
            "max_broken_bonds": args.max_broken_bonds,
            "edge_penalty_scale": args.edge_penalty_scale,
            "coherence_scale": args.coherence_scale,
            "minimum_matched_peaks": args.minimum_matched_peaks,
            "adducts": sorted(SUPPORTED_ADDUCTS),
            "candidate_frequency_weighting": "log((candidate_count+1)/(matching_candidates+1))",
            "connected_solver": "deterministic greedy rooted prize-collecting approximation",
            "instrument_and_collision_context": (
                "retained in the explanation ledger but not scored before OOF calibration"
            ),
            "unused_audit_arm": (
                "intensity-rank permutation remains implemented in the core but is "
                "not computed in the formal full run because it neither qualifies "
                "the source nor selects a triplet"
            ),
            "truth_blind_applicability": (
                "all candidate source scores finite and at least two candidates have "
                "different connected-explanation scores; otherwise the full query abstains"
            ),
        },
        "scientific_contract": {
            "changed_component": "truth-blind candidate relation evidence only",
            "fragmentation_scores_are_training_targets": False,
            "candidate_truth_read": False,
            "dreams_embeddings_read": False,
            "loss_sampler_optimizer_changed": False,
            "required_next_gate": (
                "formula-disjoint specificity qualification against all three controls"
            ),
        },
        "provenance": {
            "training_evidence": str(args.evidence.resolve()),
            "training_evidence_sha256": sha256_file(args.evidence),
            "corrected_manifest": str(args.manifest.resolve()),
            "corrected_manifest_sha256": sha256_file(args.manifest),
            "data": str(args.data.resolve()),
            "data_sha256_not_computed_due_size": True,
            "ms_pred_source_root": str(args.ms_pred_source_root.resolve()),
            "ms_pred_source_hashes": observed_source_hashes,
            "evidence_contract": str(args.evidence_contract.resolve()),
            "evidence_contract_sha256": sha256_file(args.evidence_contract),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_fragment_graph_", dir=args.output.parent))
    try:
        with (temporary / "candidate_scores.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(output_rows[0]), delimiter="\t", lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(output_rows)
        if explanation_rows:
            with (temporary / "connected_explanations.tsv").open(
                "w", encoding="utf-8", newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(explanation_rows[0]),
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(explanation_rows)
        if exclusions:
            with (temporary / "exclusions.tsv").open(
                "w", encoding="utf-8", newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(exclusions[0]), delimiter="\t", lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(exclusions)
        report["candidate_scores_sha256"] = sha256_file(temporary / "candidate_scores.tsv")
        if explanation_rows:
            report["connected_explanations_sha256"] = sha256_file(
                temporary / "connected_explanations.tsv"
            )
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
