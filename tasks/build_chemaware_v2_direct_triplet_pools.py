"""Build direct DreaMS triplet pools from frozen ChemAware V2 actions.

ChemAware is used only to mine hard negative molecules on formula role 2.
The emitted files use the native ``train_e1_identity.py`` candidate-pool
schema, so optimization is the standard DreaMS cosine triplet objective.  No
teacher score, candidate utility, or chemical feature reaches the encoder.

Formula role 3 is emitted only as a development pool.  Formula role 4 has no
input argument and cannot be reached by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_DIR = (
    ROOT / "data/validation/chemaware_multinull_deployment_safe_full_20260919"
)
DEFAULT_MANIFEST = (
    ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
)
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OUTPUT = ROOT / "data/validation/chemaware_v2_direct_triplets_20260920"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def load_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        missing = sorted(required - set(loaded.files))
        if missing:
            raise RuntimeError(f"{path} lacks required arrays: {missing}")
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def query_molecules(
    manifest: Mapping[str, np.ndarray], query: int,
) -> tuple[int, int, np.ndarray]:
    left, right = map(int, manifest["query_ptr"][query:query + 2])
    return left, right, np.asarray(manifest["molecule_ptr"][left:right + 1], dtype=np.int64)


def candidate_count(manifest: Mapping[str, np.ndarray], query: int) -> int:
    left, right = map(int, manifest["query_ptr"][query:query + 2])
    return right - left


def molecule_rows(
    manifest: Mapping[str, np.ndarray], query: int, local_molecule: int,
) -> np.ndarray:
    left, right, ptr = query_molecules(manifest, query)
    if not (0 <= local_molecule < right - left):
        raise IndexError(
            f"local molecule {local_molecule} outside query {query} candidate block"
        )
    pair_left = int(ptr[local_molecule])
    pair_right = int(ptr[local_molecule + 1])
    return np.asarray(manifest["pair_candidate_row"][pair_left:pair_right], dtype=np.int64)


def selected_candidate(policy: Mapping[str, np.ndarray], row: int) -> int:
    slot = int(policy["correct_selected_candidate_slot"][row])
    if slot < 0:
        return -1
    if not bool(policy["valid_candidate"][row, slot]):
        raise RuntimeError(f"selected invalid candidate slot at policy row {row}")
    return int(policy["proposed_candidate"][row, slot])


def correction_rows(policy: Mapping[str, np.ndarray]) -> np.ndarray:
    baseline = np.asarray(policy["baseline_rank"])
    corrected = np.asarray(policy["correct_rank"])
    selected = np.asarray([
        selected_candidate(policy, row) for row in range(len(baseline))
    ])
    rows = np.flatnonzero((baseline > 1) & (corrected == 1) & (selected == 0))
    # A claimed correction must promote the unique true molecule (local index 0).
    if np.any(np.asarray(policy["baseline_candidate"])[rows] == 0):
        raise RuntimeError("correction ledger contains a true baseline candidate")
    return rows.astype(np.int64)


def eligible_error_rows(
    policy: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
) -> np.ndarray:
    output = []
    for row, query in enumerate(np.asarray(policy["query"], dtype=np.int64)):
        baseline = int(policy["baseline_candidate"][row])
        if int(policy["baseline_rank"][row]) <= 1 or baseline == 0:
            continue
        anchor = int(manifest["query_row"][query])
        positives = molecule_rows(manifest, int(query), 0)
        negatives = molecule_rows(manifest, int(query), baseline)
        if np.any(positives != anchor) and len(negatives):
            output.append(row)
    return np.asarray(output, dtype=np.int64)


def match_controls(
    policy: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
    target_rows: np.ndarray, seed: int,
) -> np.ndarray:
    """Greedy no-replacement controls matched before any encoder update.

    Matching priority is same formula, then official rank, candidate count and
    positive-reference multiplicity.  Outcome fields from ChemAware are not in
    the distance function.
    """
    corrected = set(map(int, target_rows))
    candidates = [
        int(row) for row in eligible_error_rows(policy, manifest)
        if int(row) not in corrected
    ]
    rng = np.random.default_rng(seed)
    jitter = {row: float(rng.random()) for row in candidates}
    chosen: list[int] = []
    available = set(candidates)
    for target in target_rows:
        target = int(target)
        tq = int(policy["query"][target])
        target_formula = str(policy["formula"][target])
        target_rank = int(policy["baseline_rank"][target])
        target_count = candidate_count(manifest, tq)
        target_positive_n = len(molecule_rows(manifest, tq, 0))
        if not available:
            raise RuntimeError("insufficient non-ChemAware baseline errors for matched control")

        def distance(row: int) -> tuple[int, int, int, int, float, int]:
            query = int(policy["query"][row])
            return (
                int(str(policy["formula"][row]) != target_formula),
                abs(int(policy["baseline_rank"][row]) - target_rank),
                abs(candidate_count(manifest, query) - target_count),
                abs(len(molecule_rows(manifest, query, 0)) - target_positive_n),
                jitter[row],
                row,
            )

        best = min(available, key=distance)
        chosen.append(best)
        available.remove(best)
    return np.asarray(chosen, dtype=np.int64)


def build_pool(
    policy: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
    policy_rows: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    anchors: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    positives_flat: list[int] = []
    negatives_flat: list[int] = []
    kept_policy_rows: list[int] = []
    queries: list[int] = []
    skipped_self_only = 0
    for row in np.asarray(policy_rows, dtype=np.int64):
        query = int(policy["query"][row])
        anchor = int(manifest["query_row"][query])
        baseline = int(policy["baseline_candidate"][row])
        positive = molecule_rows(manifest, query, 0)
        positive = np.unique(positive[positive != anchor])
        negative = np.unique(molecule_rows(manifest, query, baseline))
        if not len(positive):
            skipped_self_only += 1
            continue
        if not len(negative):
            raise RuntimeError(f"query {query} has no baseline-negative spectrum")
        anchors.append(anchor)
        positives_flat.extend(map(int, positive))
        negatives_flat.extend(map(int, negative))
        positive_ptr.append(len(positives_flat))
        negative_ptr.append(len(negatives_flat))
        kept_policy_rows.append(int(row))
        queries.append(query)
    if not anchors:
        raise RuntimeError("triplet construction produced no eligible anchors")
    pool = {
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positives_flat, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negatives_flat, dtype=np.int64),
        "source_policy_row": np.asarray(kept_policy_rows, dtype=np.int64),
        "source_query": np.asarray(queries, dtype=np.int64),
    }
    audit = {
        "requested_actions": int(len(policy_rows)),
        "eligible_anchors": int(len(anchors)),
        "unique_formulas": int(len(np.unique(policy["formula"][kept_policy_rows]))),
        "positive_edges": int(len(positives_flat)),
        "negative_edges": int(len(negatives_flat)),
        "skipped_self_only_positive": int(skipped_self_only),
    }
    return pool, audit


def matching_diagnostics(
    policy: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
    target_rows: np.ndarray, control_rows: np.ndarray,
) -> dict[str, float]:
    if len(target_rows) != len(control_rows):
        raise ValueError("matching diagnostic rows differ in length")
    target_queries = np.asarray(policy["query"])[target_rows].astype(np.int64)
    control_queries = np.asarray(policy["query"])[control_rows].astype(np.int64)
    target_count = np.asarray([
        candidate_count(manifest, int(query)) for query in target_queries
    ])
    control_count = np.asarray([
        candidate_count(manifest, int(query)) for query in control_queries
    ])
    target_rank = np.asarray(policy["baseline_rank"])[target_rows]
    control_rank = np.asarray(policy["baseline_rank"])[control_rows]
    return {
        "same_formula_fraction": float(np.mean(
            np.asarray(policy["formula"])[target_rows]
            == np.asarray(policy["formula"])[control_rows]
        )),
        "same_official_rank_fraction": float(np.mean(target_rank == control_rank)),
        "official_rank_mean_absolute_difference": float(np.mean(np.abs(
            target_rank.astype(np.float64) - control_rank.astype(np.float64)
        ))),
        "candidate_count_mean_absolute_difference": float(np.mean(np.abs(
            target_count.astype(np.float64) - control_count.astype(np.float64)
        ))),
    }


def validate_pool_against_hdf5(
    pool: Mapping[str, np.ndarray], data: Path,
) -> dict[str, object]:
    all_rows = np.unique(np.concatenate((
        pool["anchor_idx"], pool["positive_idx"], pool["negative_idx"],
    )))
    with h5py.File(data, "r") as handle:
        n_rows = len(handle["INCHIKEY"])
        if np.any((all_rows < 0) | (all_rows >= n_rows)):
            raise RuntimeError("triplet pool contains an out-of-range HDF5 row")
        ik14 = decode(handle["INCHIKEY"][:])
        ik14 = np.asarray([value[:14] for value in ik14])
    positive_edges = negative_edges = 0
    for row, anchor in enumerate(pool["anchor_idx"]):
        p0, p1 = map(int, pool["positive_ptr"][row:row + 2])
        n0, n1 = map(int, pool["negative_ptr"][row:row + 2])
        positive = pool["positive_idx"][p0:p1]
        negative = pool["negative_idx"][n0:n1]
        if np.any(positive == anchor):
            raise RuntimeError("anchor leakage reached positive candidates")
        if not np.all(ik14[positive] == ik14[int(anchor)]):
            raise RuntimeError("positive edge crosses molecular identity")
        if not np.all(ik14[negative] != ik14[int(anchor)]):
            raise RuntimeError("negative edge shares anchor molecular identity")
        positive_edges += len(positive)
        negative_edges += len(negative)
    return {
        "hdf5_rows": int(n_rows),
        "identity_contract_passed": True,
        "positive_edges_checked": int(positive_edges),
        "negative_edges_checked": int(negative_edges),
    }


def save_pool(path: Path, pool: Mapping[str, np.ndarray]) -> None:
    np.savez_compressed(path, **pool)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite triplet directory: {args.output}")
    policy_required = {
        "query", "formula", "identity", "baseline_rank", "correct_rank",
        "correct_selected_candidate_slot", "valid_candidate", "proposed_candidate",
        "baseline_candidate",
    }
    manifest_required = {
        "query_row", "query_ik14", "query_formula", "query_ptr", "molecule_ptr",
        "molecule_label", "molecule_ik14", "molecule_formula", "pair_candidate_row",
    }
    role2_path = args.policy_dir / "validation_policy.npz"
    role3_path = args.policy_dir / "inner_policy.npz"
    role2 = load_npz(role2_path, policy_required)
    role3 = load_npz(role3_path, policy_required)
    manifest = load_npz(args.manifest, manifest_required)

    if np.intersect1d(role2["formula"].astype(str), role3["formula"].astype(str)).size:
        raise RuntimeError("formula leakage between role 2 training and role 3 development")
    role2_corrections = correction_rows(role2)
    role3_corrections = correction_rows(role3)
    role2_controls = match_controls(role2, manifest, role2_corrections, args.seed)
    match_audit = matching_diagnostics(
        role2, manifest, role2_corrections, role2_controls,
    )

    chem_pool, chem_audit = build_pool(role2, manifest, role2_corrections)
    control_pool, control_audit = build_pool(role2, manifest, role2_controls)
    development_pool, development_audit = build_pool(role3, manifest, role3_corrections)
    if len(chem_pool["anchor_idx"]) != len(control_pool["anchor_idx"]):
        raise RuntimeError("matched control lost anchor-count parity")

    temporary = Path(tempfile.mkdtemp(prefix="chemaware_triplets_", dir=args.output.parent))
    try:
        files = {
            "role2_chemaware_train_pool.npz": chem_pool,
            "role2_matched_control_train_pool.npz": control_pool,
            "role3_development_pool.npz": development_pool,
        }
        hdf5_audits = {}
        for name, pool in files.items():
            save_pool(temporary / name, pool)
            hdf5_audits[name] = validate_pool_against_hdf5(pool, args.data)
        report = {
            "status": "CHEMAWARE_V2_DIRECT_TRIPLET_POOLS_COMPLETE",
            "objective": "standard DreaMS cosine triplet fine-tuning; no distillation",
            "encoder_inputs": ["anchor spectrum", "positive spectrum", "negative spectrum"],
            "chemical_role": "role-2 hard-negative mining only",
            "optimization_role": 2,
            "development_role": 3,
            "outer_role_4_accessed": False,
            "pool_schema": "train_e1_identity.CandidatePool",
            "triplet": {
                "anchor": "query spectrum",
                "positive": "different reference spectrum of the unique true molecule",
                "negative": "reference spectrum of the official DreaMS top-1 false molecule",
            },
            "chemaware_selection": (
                "baseline rank > 1, frozen ChemAware V2 rank = 1, selected molecule = truth"
            ),
            "matched_control": (
                "nonselected role-2 official error; greedy same-formula/rank/candidate-count match"
            ),
            "role2_chemaware": chem_audit,
            "role2_matched_control": control_audit,
            "role2_matching_diagnostics": match_audit,
            "role3_development": development_audit,
            "hdf5_contracts": hdf5_audits,
            "provenance": {
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": sha256(args.manifest),
                "role2_policy": str(role2_path.resolve()),
                "role2_policy_sha256": sha256(role2_path),
                "role3_policy": str(role3_path.resolve()),
                "role3_policy_sha256": sha256(role3_path),
                "hdf5": str(args.data.resolve()),
                "hdf5_sha256": sha256(args.data),
            },
        }
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
