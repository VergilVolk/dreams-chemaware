"""Audit and build a representative small training pool for noise fine-tuning.

The historical ``entries[:N]`` shortcut is invalid because the isomer manifest
is grouped by InChIKey.  It can make a nominal 10k-anchor experiment contain
only a few hundred molecules.  This tool records the *effective* molecular and
condition diversity before any GPU job is submitted, and can write a locked
balanced subset for a reproducible pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "tasks/massspecgym_isomers/dataset_manifest.json"
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"


def decode(value: object) -> str:
    return value.decode("utf-8", "ignore") if isinstance(value, bytes) else str(value)


def cross_condition(row: int, peers: list[int], instrument: np.ndarray, ce: np.ndarray) -> bool:
    for other in peers:
        if other == row:
            continue
        if instrument[row] != instrument[other]:
            return True
        if np.isfinite(ce[row]) and np.isfinite(ce[other]) and abs(ce[row] - ce[other]) >= 10:
            return True
    return False


def balanced_subset(entries: list[dict], n: int, seed: int, max_per_identity: int) -> list[dict]:
    """Sample identity/adduct groups uniformly, never as a sorted prefix.

    Each retained group contributes up to ``max_per_identity`` spectra.  Groups
    with at least two spectra are considered first so explicit real-spectrum
    positives are available to the subsequent training objective.
    """
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in entries:
        groups[(entry["ik14"], entry["adduct"])].append(entry)
    multi = [items for items in groups.values() if len(items) >= 2]
    singleton = [items for items in groups.values() if len(items) == 1]
    rng.shuffle(multi); rng.shuffle(singleton)
    selected: list[dict] = []
    for pool in (multi, singleton):
        for items in pool:
            if len(selected) >= n:
                break
            order = rng.permutation(len(items))
            take = min(max_per_identity, len(items), n - len(selected))
            selected.extend(items[int(i)] for i in order[:take])
        if len(selected) >= n:
            break
    rng.shuffle(selected)
    return selected


def audit(entries: list[dict], h5_path: Path) -> dict:
    rows = np.asarray([int(e["anchor_row"]) for e in entries], dtype=int)
    with h5py.File(h5_path, "r") as handle:
        smiles = np.asarray([decode(x) for x in handle["smiles"][:]], dtype=object)
        formula = np.asarray([decode(x) for x in handle["FORMULA"][:]], dtype=object)
        instrument = np.asarray([decode(x) for x in handle["INSTRUMENT_TYPE"][:]], dtype=object)
        ce = np.asarray(handle["COLLISION_ENERGY"][:], dtype=float)
    by_identity: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_ik: dict[str, list[int]] = defaultdict(list)
    grade = Counter()
    with_neg = 0
    for entry in entries:
        row = int(entry["anchor_row"])
        by_identity[(entry["ik14"], entry["adduct"])].append(row)
        by_ik[entry["ik14"]].append(row)
        if entry["neg"]:
            with_neg += 1
        for neg in entry["neg"]:
            grade[str(neg["grade"])] += 1
    peer_rows = sum(len(v) for v in by_identity.values() if len(v) >= 2)
    cross_rows = sum(
        cross_condition(row, group, instrument, ce)
        for group in by_identity.values() for row in group
    )
    group_sizes = np.asarray([len(v) for v in by_identity.values()], dtype=float)
    return {
        "n_anchors": len(entries),
        "n_unique_ik14": len(by_ik),
        "n_unique_identity_adduct": len(by_identity),
        "n_unique_smiles": len(set(smiles[rows])),
        "n_unique_formula": len(set(formula[rows])),
        "anchors_with_same_identity_adduct_peer": int(peer_rows),
        "fraction_with_same_identity_adduct_peer": float(peer_rows / len(entries)) if entries else 0.0,
        "anchors_with_cross_condition_peer": int(cross_rows),
        "fraction_with_cross_condition_peer": float(cross_rows / len(entries)) if entries else 0.0,
        "anchors_with_hard_negative": int(with_neg),
        "fraction_with_hard_negative": float(with_neg / len(entries)) if entries else 0.0,
        "negative_grade_counts": dict(grade),
        "identity_adduct_group_size": {
            "median": float(np.median(group_sizes)) if len(group_sizes) else 0.0,
            "p90": float(np.quantile(group_sizes, 0.9)) if len(group_sizes) else 0.0,
            "max": int(group_sizes.max()) if len(group_sizes) else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--n-anchors", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--max-per-identity", type=int, default=4)
    parser.add_argument("--out", type=Path, default=ROOT / "tasks/massspecgym_isomers/noise_train_balanced_10k.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest[args.split]
    legacy = entries[:args.n_anchors]
    selected = balanced_subset(entries, min(args.n_anchors, len(entries)), args.seed, args.max_per_identity)
    report = {
        "purpose": "preflight audit; sorted-prefix training is prohibited",
        "source_manifest": str(args.manifest), "source_split": args.split,
        "selection": {"method": "uniform_identity_adduct_then_shuffle", "seed": args.seed,
                      "max_per_identity_adduct": args.max_per_identity},
        "legacy_sorted_prefix": audit(legacy, args.data),
        "balanced_selected": audit(selected, args.data),
    }
    payload = {"meta": report, "entries": selected}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    report["locked_subset"] = str(args.out)
    report["locked_subset_sha256"] = digest
    (args.out.with_suffix(".audit.json")).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
