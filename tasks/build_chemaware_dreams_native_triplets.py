"""Convert frozen ChemAware actions into native DreaMS triplet pools.

Only triplet construction is ChemAware-specific.  The emitted candidate lists
are consumed by DreaMS ``ContrastiveSpectraDataset`` and ``ContrastiveHead``.

Directional events:

* correction: the chemical action promotes the true molecule over the
  official false top-1; true references are positive and official top-1
  references are negative;
* protection: the official top-1 is true but the chemical action promotes a
  false molecule; true references are positive and the promoted false
  references are negative.

Wrong-to-wrong actions have no valid chemical direction and are audited but
never converted into a training triplet.
"""
from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT = ROOT / "data/validation/chemaware_dreams_native_triplets_20260920"

CORRECTION = 1
PROTECTION = 2


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def molecule_rows(
    manifest: Mapping[str, np.ndarray], query: int, local_molecule: int,
) -> np.ndarray:
    molecule_left, molecule_right = map(
        int, manifest["query_ptr"][query:query + 2],
    )
    count = molecule_right - molecule_left
    if not (0 <= local_molecule < count):
        raise IndexError(f"candidate {local_molecule} outside query {query}")
    molecule = molecule_left + local_molecule
    pair_left, pair_right = map(
        int, manifest["molecule_ptr"][molecule:molecule + 2],
    )
    return np.asarray(
        manifest["pair_candidate_row"][pair_left:pair_right], dtype=np.int64,
    )


def selected_candidate(policy: Mapping[str, np.ndarray], row: int) -> int:
    slot = int(policy["correct_selected_candidate_slot"][row])
    if slot < 0:
        return -1
    if not bool(policy["valid_candidate"][row, slot]):
        raise RuntimeError(f"invalid selected action slot at row {row}")
    return int(policy["proposed_candidate"][row, slot])


def directional_events(policy: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    event_rows: list[int] = []
    event_type: list[int] = []
    action_candidate: list[int] = []
    negative_candidate: list[int] = []
    excluded_wrong_to_wrong = 0
    active = 0
    for row in range(len(policy["query"])):
        proposed = selected_candidate(policy, row)
        if proposed < 0:
            continue
        active += 1
        baseline = int(policy["baseline_candidate"][row])
        baseline_rank = int(policy["baseline_rank"][row])
        action_rank = int(policy["correct_rank"][row])
        if baseline_rank > 1 and action_rank == 1 and proposed == 0 and baseline != 0:
            kind = CORRECTION
            negative = baseline
        elif baseline_rank == 1 and action_rank > 1 and baseline == 0 and proposed != 0:
            kind = PROTECTION
            negative = proposed
        else:
            excluded_wrong_to_wrong += 1
            continue
        event_rows.append(row)
        event_type.append(kind)
        action_candidate.append(proposed)
        negative_candidate.append(negative)
    return {
        "policy_row": np.asarray(event_rows, dtype=np.int64),
        "event_type": np.asarray(event_type, dtype=np.int8),
        "action_candidate": np.asarray(action_candidate, dtype=np.int16),
        "negative_candidate": np.asarray(negative_candidate, dtype=np.int16),
        "active_actions": np.asarray(active, dtype=np.int64),
        "excluded_wrong_to_wrong": np.asarray(excluded_wrong_to_wrong, dtype=np.int64),
    }


def build_pool(
    policy: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    events = directional_events(policy)
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    kept: list[int] = []
    kept_type: list[int] = []
    kept_action: list[int] = []
    kept_negative: list[int] = []
    skipped_no_distinct_positive = 0
    for index, policy_row in enumerate(events["policy_row"]):
        policy_row = int(policy_row)
        query = int(policy["query"][policy_row])
        anchor = int(manifest["query_row"][query])
        positive = np.unique(molecule_rows(manifest, query, 0))
        positive = positive[positive != anchor]
        negative_local = int(events["negative_candidate"][index])
        negative = np.unique(molecule_rows(manifest, query, negative_local))
        if not len(positive):
            skipped_no_distinct_positive += 1
            continue
        if not len(negative):
            raise RuntimeError(f"query {query} has no negative references")
        anchors.append(anchor)
        positives.extend(map(int, positive))
        negatives.extend(map(int, negative))
        positive_ptr.append(len(positives))
        negative_ptr.append(len(negatives))
        kept.append(policy_row)
        kept_type.append(int(events["event_type"][index]))
        kept_action.append(int(events["action_candidate"][index]))
        kept_negative.append(negative_local)
    if not anchors:
        raise RuntimeError("no directional ChemAware triplets were constructed")
    kept_array = np.asarray(kept, dtype=np.int64)
    pool = {
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positives, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negatives, dtype=np.int64),
        "source_policy_row": kept_array,
        "source_query": np.asarray(policy["query"])[kept_array].astype(np.int64),
        "event_type": np.asarray(kept_type, dtype=np.int8),
        "action_candidate": np.asarray(kept_action, dtype=np.int16),
        "negative_candidate": np.asarray(kept_negative, dtype=np.int16),
    }
    audit = {
        "active_actions": int(events["active_actions"]),
        "directional_triplets": int(len(anchors)),
        "correction_triplets": int(np.sum(pool["event_type"] == CORRECTION)),
        "protection_triplets": int(np.sum(pool["event_type"] == PROTECTION)),
        "excluded_wrong_to_wrong": int(events["excluded_wrong_to_wrong"]),
        "skipped_no_distinct_positive": int(skipped_no_distinct_positive),
        "unique_formulas": int(len(np.unique(
            np.asarray(policy["formula"])[kept_array].astype(str)
        ))),
        "positive_edges": int(len(positives)),
        "negative_edges": int(len(negatives)),
    }
    return pool, audit


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def audit_identity_edges(pool: Mapping[str, np.ndarray], data: Path) -> dict[str, int | bool]:
    with h5py.File(data, "r") as handle:
        ik14 = np.asarray([value[:14] for value in decode(handle["INCHIKEY"][:])])
    positive_edges = negative_edges = 0
    for row, anchor in enumerate(pool["anchor_idx"]):
        p0, p1 = map(int, pool["positive_ptr"][row:row + 2])
        n0, n1 = map(int, pool["negative_ptr"][row:row + 2])
        pos = pool["positive_idx"][p0:p1]
        neg = pool["negative_idx"][n0:n1]
        if np.any(pos == anchor) or not np.all(ik14[pos] == ik14[int(anchor)]):
            raise RuntimeError("native DreaMS positive edge violates identity contract")
        if not np.all(ik14[neg] != ik14[int(anchor)]):
            raise RuntimeError("native DreaMS negative edge violates identity contract")
        positive_edges += len(pos)
        negative_edges += len(neg)
    return {
        "identity_contract_passed": True,
        "positive_edges_checked": int(positive_edges),
        "negative_edges_checked": int(negative_edges),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    role2_path = args.policy_dir / "validation_policy.npz"
    role3_path = args.policy_dir / "inner_policy.npz"
    role2, role3, manifest = map(load_npz, (role2_path, role3_path, args.manifest))
    if np.intersect1d(role2["formula"].astype(str), role3["formula"].astype(str)).size:
        raise RuntimeError("role-2/role-3 formula leakage")
    train_pool, train_audit = build_pool(role2, manifest)
    val_pool, val_audit = build_pool(role3, manifest)
    temporary = Path(tempfile.mkdtemp(prefix="chem_dreams_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **train_pool)
        np.savez_compressed(temporary / "val_pool.npz", **val_pool)
        report = {
            "status": "CHEMAWARE_DREAMS_NATIVE_TRIPLETS_COMPLETE",
            "custom_component": "triplet construction only",
            "training_stack": [
                "dreams.utils.data.ContrastiveSpectraDataset",
                "dreams.models.heads.heads.ContrastiveHead",
                "DreaMS SpectrumPreprocessor",
                "DreaMS cosine triplet-margin loss",
            ],
            "train_formula_role": 2,
            "validation_formula_role": 3,
            "outer_role_4_accessed": False,
            "train": train_audit,
            "validation": val_audit,
            "identity_audit": {
                "train": audit_identity_edges(train_pool, args.data),
                "validation": audit_identity_edges(val_pool, args.data),
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
