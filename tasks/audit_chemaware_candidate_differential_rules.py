"""No-training preflight for candidate-differential chemical evidence.

This audit tests a necessary condition for ChemAware before any optimiser is
allowed to run.  A useful chemical rule must do more than match an absolute
mass in a query spectrum: the peak must recur in a candidate reference
spectrum, be compatible with that candidate's structure, and be less
compatible with the competing structures in the same candidate set.

The audit is deliberately score-blind and uses only the corrected training
manifest.  It compares four paired rankings on exactly the same query/reference
edges:

1. raw recurrent peak evidence;
2. recurrent evidence filtered by candidate structure;
3. candidate-differential (IDF-weighted) structural evidence;
4. two negative controls: within-query structure rotation and cardinality-
   matched peak-mask rotation.

Passing this audit is necessary, not sufficient, for adding the rule signal to
training.  No model parameters or official DreaMS scores are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import types
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[1]
PROTON = 1.007276466621
HYDROGEN = 1.00782503223


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def stable_u64(*values: object, seed: int = 0) -> int:
    body = "\x1f".join(map(str, values)).encode("utf-8")
    digest = hashlib.blake2b(body, digest_size=8, person=f"cw{seed}".encode()).digest()
    return int.from_bytes(digest, "little", signed=False)


def spectrum_peaks(spectrum: np.ndarray, precursor_mz: float) -> tuple[np.ndarray, np.ndarray]:
    mz = np.asarray(spectrum[0], dtype=np.float64)
    intensity = np.asarray(spectrum[1], dtype=np.float64)
    keep = (
        np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) &
        (intensity > 0) & (mz < precursor_mz + 0.05)
    )
    mz, intensity = mz[keep], intensity[keep]
    order = np.argsort(mz)
    mz, intensity = mz[order], intensity[order]
    total = float(np.sum(intensity))
    return mz, intensity / total if total > 0 else intensity


def greedy_matches(a: np.ndarray, b: np.ndarray, tolerance_da: float) -> list[tuple[int, int]]:
    proposals: list[tuple[float, int, int]] = []
    for i, value in enumerate(a):
        lo = int(np.searchsorted(b, value - tolerance_da, side="left"))
        hi = int(np.searchsorted(b, value + tolerance_da, side="right"))
        proposals.extend((abs(value - b[j]), i, j) for j in range(lo, hi))
    used_a: set[int] = set()
    used_b: set[int] = set()
    output: list[tuple[int, int]] = []
    for _, i, j in sorted(proposals):
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); output.append((i, j))
    return output


def components_after_cuts(molecule: Chem.Mol, cut_bonds: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    blocked = set(cut_bonds)
    adjacency = [[] for _ in range(molecule.GetNumAtoms())]
    for bond in molecule.GetBonds():
        if bond.GetIdx() in blocked:
            continue
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[left].append(right); adjacency[right].append(left)
    components: list[tuple[int, ...]] = []
    unseen = set(range(molecule.GetNumAtoms()))
    while unseen:
        root = unseen.pop()
        stack, component = [root], [root]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor); stack.append(neighbor); component.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


@lru_cache(maxsize=None)
def cut_fragment_masses(smiles: str, max_cuts: int) -> np.ndarray:
    """Masses of connected atom subsets exposed by <= ``max_cuts`` cuts.

    Hydrogens are counted from the intact molecule.  We intentionally do not
    sanitize disconnected fragments, because sanitization silently caps broken
    valences and turns an explicit chemical assumption into hidden label noise.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() < 2:
        return np.empty(0, dtype=np.float64)
    table = Chem.GetPeriodicTable()
    atom_mass = np.empty(molecule.GetNumAtoms(), dtype=np.float64)
    for atom in molecule.GetAtoms():
        isotope = atom.GetIsotope()
        heavy = (
            table.GetMassForIsotope(atom.GetAtomicNum(), isotope)
            if isotope else table.GetMostCommonIsotopeMass(atom.GetAtomicNum())
        )
        atom_mass[atom.GetIdx()] = heavy + atom.GetTotalNumHs() * HYDROGEN
    bonds = tuple(bond.GetIdx() for bond in molecule.GetBonds())
    full = frozenset(range(molecule.GetNumAtoms()))
    subsets: set[frozenset[int]] = set()
    for n_cuts in range(1, min(max_cuts, len(bonds)) + 1):
        for cuts in itertools.combinations(bonds, n_cuts):
            components = components_after_cuts(molecule, cuts)
            if len(components) == 1:
                continue
            for component in components:
                subset = frozenset(component)
                if subset and subset != full:
                    subsets.add(subset)
    masses = [float(np.sum(atom_mass[list(subset)])) for subset in subsets]
    return np.unique(np.round(np.asarray(masses, dtype=np.float64), 6))


_MAGMA_MODULE = None


def load_magma_module(source_root: str):
    """Load only the MIT ms-pred MAGMa heuristic without its NN stack."""
    global _MAGMA_MODULE
    if _MAGMA_MODULE is not None:
        return _MAGMA_MODULE
    root = Path(source_root).resolve() / "src"
    if not (root / "ms_pred/magma/fragmentation.py").is_file():
        raise FileNotFoundError(root / "ms_pred/magma/fragmentation.py")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # ms_pred.common.__init__ imports optional torch-scatter.  The heuristic
    # fragmenter only needs chem_utils, so load that module directly and avoid
    # turning unavailable neural dependencies into a chemistry requirement.
    common_package = types.ModuleType("ms_pred.common")
    common_package.__path__ = [str(root / "ms_pred/common")]
    sys.modules["ms_pred.common"] = common_package
    spec = importlib.util.spec_from_file_location(
        "ms_pred.common.chem_utils", root / "ms_pred/common/chem_utils.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load ms-pred chem_utils")
    chem_utils = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = chem_utils
    spec.loader.exec_module(chem_utils)
    from ms_pred.magma import fragmentation  # type: ignore
    _MAGMA_MODULE = fragmentation
    return fragmentation


@lru_cache(maxsize=None)
def magma_fragment_masses(
    smiles: str,
    source_root: str,
    max_tree_depth: int,
    max_broken_bonds: int,
) -> np.ndarray:
    fragmentation = load_magma_module(source_root)
    engine = fragmentation.FragmentEngine(
        smiles,
        max_tree_depth=max_tree_depth,
        max_broken_bonds=max_broken_bonds,
    )
    engine.generate_fragments()
    masses = np.asarray(engine.get_frag_masses()[3], dtype=np.float64)
    return np.unique(np.round(masses[np.isfinite(masses) & (masses > 0)], 6))


def nearest_mask(observed: np.ndarray, theory: np.ndarray, ppm: float, floor_da: float) -> np.ndarray:
    if len(observed) == 0 or len(theory) == 0:
        return np.zeros(len(observed), dtype=bool)
    theory = np.sort(theory)
    right = np.searchsorted(theory, observed)
    left = np.clip(right - 1, 0, len(theory) - 1)
    right = np.clip(right, 0, len(theory) - 1)
    distance = np.minimum(abs(observed - theory[left]), abs(observed - theory[right]))
    tolerance = np.maximum(floor_da, abs(observed) * ppm * 1e-6)
    return distance <= tolerance


def structure_masks(
    mz: np.ndarray,
    precursor_mz: float,
    smiles: str,
    max_cuts: int,
    hydrogen_shift: int,
    ppm: float,
    floor_da: float,
    fragment_backend: str = "simple_cut",
    magma_source_root: str = "",
    magma_tree_depth: int = 3,
    magma_max_broken_bonds: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    if fragment_backend == "magma":
        masses = magma_fragment_masses(
            smiles, magma_source_root, magma_tree_depth, magma_max_broken_bonds,
        )
        # MAGMa already enumerates hydrogen shifts constrained by the number of
        # bonds broken; never apply the broad simple-cut shift expansion again.
        expanded = masses
    elif fragment_backend == "simple_cut":
        masses = cut_fragment_masses(smiles, max_cuts)
        shifts = np.arange(-hydrogen_shift, hydrogen_shift + 1, dtype=np.float64) * HYDROGEN
        expanded = np.unique((masses[:, None] + shifts[None, :]).ravel()) if len(masses) else masses
    else:
        raise ValueError(f"unknown fragment backend: {fragment_backend}")
    direct = nearest_mask(mz, expanded + PROTON, ppm, floor_da)
    loss = nearest_mask(precursor_mz - mz, expanded, ppm, floor_da)
    return direct, loss


def recurrent_score(
    query_mz: np.ndarray,
    query_intensity: np.ndarray,
    query_precursor: float,
    reference_mz: np.ndarray,
    reference_intensity: np.ndarray,
    reference_precursor: float,
    direct_weight: np.ndarray,
    loss_weight: np.ndarray,
    match_da: float,
) -> float:
    direct = greedy_matches(query_mz, reference_mz, match_da)
    loss_q = query_precursor - query_mz
    loss_r = reference_precursor - reference_mz
    q_order, r_order = np.argsort(loss_q), np.argsort(loss_r)
    loss = greedy_matches(loss_q[q_order], loss_r[r_order], match_da)
    direct_value = sum(
        math.sqrt(query_intensity[i] * reference_intensity[j]) * direct_weight[i]
        for i, j in direct
    )
    loss_value = sum(
        math.sqrt(query_intensity[q_order[i]] * reference_intensity[r_order[j]]) * loss_weight[q_order[i]]
        for i, j in loss
    )
    return 0.5 * (direct_value + loss_value)


def strict_rank(scores: np.ndarray) -> int:
    return 1 + int(np.sum(scores[1:] >= scores[0]))


def rotate_mask(mask: np.ndarray, shift: int) -> np.ndarray:
    if len(mask) < 2:
        return mask.copy()
    return np.roll(mask, 1 + shift % (len(mask) - 1))


def summarize(rows: list[dict], prefix: str) -> dict:
    ranks = np.asarray([row[f"{prefix}_rank"] for row in rows], dtype=np.int64)
    margins = np.asarray([row[f"{prefix}_margin"] for row in rows], dtype=np.float64)
    return {
        "queries": int(len(rows)),
        "strict_hit1": float(np.mean(ranks == 1)) if len(rows) else None,
        "mrr": float(np.mean(1.0 / ranks)) if len(rows) else None,
        "mean_margin": float(np.mean(margins)) if len(rows) else None,
        "median_margin": float(np.median(margins)) if len(rows) else None,
        "positive_margin_fraction": float(np.mean(margins > 0)) if len(rows) else None,
    }


def conditional_summary(rows: list[dict]) -> dict:
    """Separate novel correction headroom from preservation of easy cases."""
    raw_errors = [row for row in rows if row["raw_rank"] != 1]
    raw_correct = [row for row in rows if row["raw_rank"] == 1]
    methods = ("structural", "differential", "structure_permuted", "mask_rotated")
    return {
        "raw_errors": {
            "queries": len(raw_errors),
            **{
                name: {
                    "rescued_to_strict_hit1": (
                        float(np.mean([row[f"{name}_rank"] == 1 for row in raw_errors]))
                        if raw_errors else None
                    ),
                    "positive_margin_fraction": (
                        float(np.mean([row[f"{name}_margin"] > 0 for row in raw_errors]))
                        if raw_errors else None
                    ),
                }
                for name in methods
            },
        },
        "raw_correct": {
            "queries": len(raw_correct),
            **{
                name: {
                    "preserved_strict_hit1": (
                        float(np.mean([row[f"{name}_rank"] == 1 for row in raw_correct]))
                        if raw_correct else None
                    )
                }
                for name in methods
            },
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=600)
    parser.add_argument("--max-references", type=int, default=4)
    parser.add_argument("--formula-folds", type=int, default=5)
    parser.add_argument("--confirmation-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260903)
    parser.add_argument("--max-cuts", type=int, default=2)
    parser.add_argument("--hydrogen-shift", type=int, default=0)
    parser.add_argument("--fragment-backend", choices=("simple_cut", "magma"), default="simple_cut")
    parser.add_argument(
        "--magma-source-root", type=Path,
        default=ROOT / "data/external/ms-pred-src",
    )
    parser.add_argument("--magma-tree-depth", type=int, default=3)
    parser.add_argument("--magma-max-broken-bonds", type=int, default=6)
    parser.add_argument("--match-da", type=float, default=0.02)
    parser.add_argument("--structure-ppm", type=float, default=20.0)
    parser.add_argument("--structure-floor-da", type=float, default=0.01)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    if args.output.exists():
        raise FileExistsError(args.output)
    body = np.load(args.manifest)
    query_ptr = np.asarray(body["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(body["molecule_ptr"], dtype=np.int64)
    pair_rows = np.asarray(body["pair_candidate_row"], dtype=np.int64)
    query_rows = np.asarray(body["query_row"], dtype=np.int64)
    query_formula = body["query_formula"].astype(str)
    query_identity = body["query_ik14"].astype(str)
    query_adduct = body["query_adduct"].astype(str)
    molecule_formula = body["molecule_formula"].astype(str)

    # One query spectrum per identity prevents prolific compounds from owning
    # the audit.  Restrict to same-formula competition and [M+H]+ so the first
    # test does not hide adduct chemistry inside broad mass-shift rules.
    best_by_identity: dict[str, tuple[int, int]] = {}
    for query in range(len(query_formula)):
        if query_adduct[query] != "[M+H]+":
            continue
        left, right = map(int, query_ptr[query:query + 2])
        formulas = molecule_formula[left:right]
        if np.sum(formulas == query_formula[query]) < 2:
            continue
        key = query_identity[query]
        priority = stable_u64(key, int(query_rows[query]), seed=args.seed)
        if key not in best_by_identity or priority < best_by_identity[key][0]:
            best_by_identity[key] = (priority, query)
    selected = [item[1] for item in sorted(best_by_identity.values())]
    selected.sort(key=lambda q: stable_u64(query_formula[q], query_identity[q], seed=args.seed + 1))
    selected = selected[: args.max_queries]

    rows: list[dict] = []
    with h5py.File(args.hdf5, "r") as handle:
        for sequence, query in enumerate(selected):
            molecule_left, molecule_right = map(int, query_ptr[query:query + 2])
            local_molecules = [
                m for m in range(molecule_left, molecule_right)
                if molecule_formula[m] == query_formula[query]
            ]
            # The positive is guaranteed first by the manifest contract.
            if not local_molecules or local_molecules[0] != molecule_left:
                raise RuntimeError(f"positive ordering/formula contract failed at query {query}")
            query_row = int(query_rows[query])
            q_precursor = float(handle["precursor_mz"][query_row])
            q_mz, q_intensity = spectrum_peaks(np.asarray(handle["spectrum"][query_row]), q_precursor)

            candidate_smiles: list[str] = []
            candidate_refs: list[list[int]] = []
            direct_masks: list[np.ndarray] = []
            loss_masks: list[np.ndarray] = []
            for molecule in local_molecules:
                pair_left, pair_right = map(int, molecule_ptr[molecule:molecule + 2])
                refs = list(map(int, pair_rows[pair_left:pair_right]))
                refs.sort(key=lambda row: stable_u64(query_identity[query], row, seed=args.seed + 2))
                refs = refs[: args.max_references]
                smiles = decode(handle["smiles"][refs[0]])
                direct, loss = structure_masks(
                    q_mz, q_precursor, smiles, args.max_cuts, args.hydrogen_shift,
                    args.structure_ppm, args.structure_floor_da,
                    args.fragment_backend, str(args.magma_source_root),
                    args.magma_tree_depth, args.magma_max_broken_bonds,
                )
                candidate_smiles.append(smiles); candidate_refs.append(refs)
                direct_masks.append(direct); loss_masks.append(loss)

            direct_matrix = np.asarray(direct_masks, dtype=bool)
            loss_matrix = np.asarray(loss_masks, dtype=bool)
            n_candidates = len(local_molecules)
            # IDF makes common explanations weak and candidate-exclusive ones
            # strong.  It is computed only within the current candidate set.
            direct_idf = np.log((n_candidates + 1.0) / (1.0 + direct_matrix.sum(axis=0)))
            loss_idf = np.log((n_candidates + 1.0) / (1.0 + loss_matrix.sum(axis=0)))
            raw_scores, structural_scores, differential_scores = [], [], []
            permuted_scores, rotated_scores = [], []
            for c, refs in enumerate(candidate_refs):
                raw_best = structural_best = differential_best = 0.0
                permuted_best = rotated_best = 0.0
                permuted = (c + 1) % n_candidates
                rotated_direct = rotate_mask(direct_matrix[c], sequence + 3 * c)
                rotated_loss = rotate_mask(loss_matrix[c], sequence + 5 * c)
                for reference_row in refs:
                    r_precursor = float(handle["precursor_mz"][reference_row])
                    r_mz, r_intensity = spectrum_peaks(np.asarray(handle["spectrum"][reference_row]), r_precursor)
                    ones = np.ones(len(q_mz), dtype=np.float64)
                    raw_best = max(raw_best, recurrent_score(
                        q_mz, q_intensity, q_precursor, r_mz, r_intensity, r_precursor,
                        ones, ones, args.match_da,
                    ))
                    structural_best = max(structural_best, recurrent_score(
                        q_mz, q_intensity, q_precursor, r_mz, r_intensity, r_precursor,
                        direct_matrix[c].astype(float), loss_matrix[c].astype(float), args.match_da,
                    ))
                    differential_best = max(differential_best, recurrent_score(
                        q_mz, q_intensity, q_precursor, r_mz, r_intensity, r_precursor,
                        direct_matrix[c] * direct_idf, loss_matrix[c] * loss_idf, args.match_da,
                    ))
                    permuted_best = max(permuted_best, recurrent_score(
                        q_mz, q_intensity, q_precursor, r_mz, r_intensity, r_precursor,
                        direct_matrix[permuted] * direct_idf,
                        loss_matrix[permuted] * loss_idf, args.match_da,
                    ))
                    rotated_best = max(rotated_best, recurrent_score(
                        q_mz, q_intensity, q_precursor, r_mz, r_intensity, r_precursor,
                        rotated_direct * direct_idf, rotated_loss * loss_idf, args.match_da,
                    ))
                raw_scores.append(raw_best); structural_scores.append(structural_best)
                differential_scores.append(differential_best); permuted_scores.append(permuted_best)
                rotated_scores.append(rotated_best)

            score_sets = {
                "raw": np.asarray(raw_scores),
                "structural": np.asarray(structural_scores),
                "differential": np.asarray(differential_scores),
                "structure_permuted": np.asarray(permuted_scores),
                "mask_rotated": np.asarray(rotated_scores),
            }
            row = {
                "query": int(query), "query_row": query_row,
                "query_ik14": query_identity[query], "formula": query_formula[query],
                "split": "confirmation" if stable_u64(query_formula[query], seed=args.seed) % args.formula_folds == args.confirmation_fold else "development",
                "n_same_formula_candidates": n_candidates,
                "positive_reference_count_used": len(candidate_refs[0]),
                "positive_direct_compatible_peaks": int(direct_matrix[0].sum()),
                "positive_loss_compatible_peaks": int(loss_matrix[0].sum()),
                "positive_strict_exclusive_peaks": int(np.sum(direct_matrix[0] & (direct_matrix.sum(axis=0) == 1)) + np.sum(loss_matrix[0] & (loss_matrix.sum(axis=0) == 1))),
            }
            for name, scores in score_sets.items():
                rank = strict_rank(scores)
                best_negative = float(np.max(scores[1:]))
                row[f"{name}_rank"] = rank
                row[f"{name}_margin"] = float(scores[0] - best_negative)
                row[f"{name}_truth"] = float(scores[0])
            rows.append(row)

    args.output.mkdir(parents=True)
    with (args.output / "per_query.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader(); writer.writerows(rows)
    report = {
        "status": "no_training_candidate_differential_rule_preflight",
        "formal_training_authorized": False,
        "score_blind": True,
        "selection": {
            "one_query_per_identity": True,
            "adduct": "[M+H]+",
            "same_formula_candidates_only": True,
            "selected_queries": len(rows),
            "formula_disjoint_confirmation_fold": args.confirmation_fold,
            "formula_folds": args.formula_folds,
        },
        "rules": {
            "max_bond_cuts": args.max_cuts,
            "fragment_backend": args.fragment_backend,
            "magma_tree_depth": args.magma_tree_depth if args.fragment_backend == "magma" else None,
            "magma_max_broken_bonds": args.magma_max_broken_bonds if args.fragment_backend == "magma" else None,
            "hidden_valence_capping": False,
            "hydrogen_shift": args.hydrogen_shift,
            "structure_ppm": args.structure_ppm,
            "structure_floor_da": args.structure_floor_da,
            "query_reference_match_da": args.match_da,
            "max_references_per_candidate": args.max_references,
            "candidate_differential_weight": "within-query log((C+1)/(compatible_candidates+1))",
            "magma_source_root": (
                str(args.magma_source_root.resolve()) if args.fragment_backend == "magma" else None
            ),
            "magma_fragmentation_sha256": (
                sha256_file(args.magma_source_root / "src/ms_pred/magma/fragmentation.py")
                if args.fragment_backend == "magma" else None
            ),
        },
        "development": {}, "confirmation": {},
        "gate": (
            "Proceed only if candidate-differential evidence improves over raw recurrence and both "
            "matched controls on the formula-disjoint confirmation split at nontrivial coverage."
        ),
        "claim_limit": (
            "This is a necessary-condition audit of training supervision. It is not a trained-model "
            "result, a fragmentation-mechanism proof, or an annotation benchmark."
        ),
    }
    for split in ("development", "confirmation"):
        subset = [row for row in rows if row["split"] == split]
        report[split] = {
            name: summarize(subset, name)
            for name in ("raw", "structural", "differential", "structure_permuted", "mask_rotated")
        }
        if subset:
            report[split]["coverage"] = {
                "any_positive_structure_peak": float(np.mean([
                    row["positive_direct_compatible_peaks"] + row["positive_loss_compatible_peaks"] > 0
                    for row in subset
                ])),
                "any_positive_strict_exclusive_peak": float(np.mean([
                    row["positive_strict_exclusive_peaks"] > 0 for row in subset
                ])),
            }
            report[split]["conditional_on_raw"] = conditional_summary(subset)
    confirmation = report["confirmation"]
    report["pass_as_standalone_ranker"] = bool(
        confirmation["differential"]["strict_hit1"] > confirmation["raw"]["strict_hit1"]
        and confirmation["differential"]["strict_hit1"] > confirmation["structure_permuted"]["strict_hit1"]
        and confirmation["differential"]["strict_hit1"] > confirmation["mask_rotated"]["strict_hit1"]
        and confirmation["coverage"]["any_positive_strict_exclusive_peak"] >= 0.10
    )
    raw_error = confirmation.get("conditional_on_raw", {}).get("raw_errors", {})
    differential_rescue = raw_error.get("differential", {}).get("rescued_to_strict_hit1")
    permuted_rescue = raw_error.get("structure_permuted", {}).get("rescued_to_strict_hit1")
    rotated_rescue = raw_error.get("mask_rotated", {}).get("rescued_to_strict_hit1")
    report["pass_as_privileged_teacher_signal"] = bool(
        raw_error.get("queries", 0) >= 30
        and differential_rescue is not None
        and differential_rescue > permuted_rescue
        and differential_rescue > rotated_rescue
        and confirmation["coverage"]["any_positive_strict_exclusive_peak"] >= 0.10
    )
    report["pass_to_training_mechanism"] = bool(
        report["pass_as_standalone_ranker"] or report["pass_as_privileged_teacher_signal"]
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
