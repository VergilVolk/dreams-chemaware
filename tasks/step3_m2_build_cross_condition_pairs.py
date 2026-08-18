"""M2: build the TRAIN-fold cross-condition same-molecule pair cohort for M3.

Mirrors `eval_condition_invariance_benchmark.build_locked_cohort` (same pair
definition: real-only, same adduct, instrument differs OR |CE| >= 10), but for
the TRAIN fold and enumerating *many* cross-condition pairs per molecule instead
of the single "best" pair the benchmark locks for evaluation.

Each sampled pair carries a condition-matched different-molecule negative row
(nearest precursor m/z, matching instrument+adduct first, then adduct only), so
M3's head-only training has an explicit FP guard.

The manifest is written once with a sha256 and locked (re-run refuses to drift),
exactly like the benchmark cohort.

This script does NO model forward passes -- it is pure metadata, cheap to run,
and is the "data gate" before M3 spends CPU on frozen-backbone embedding.

Performance note: the train fold is ~5x the val fold, so the negative matcher is
vectorised (np.argmin over a per-pool m/z array) rather than a Python min() over
an ~80k-row pool -- the latter is what the benchmark does once per molecule on
the small val fold and is fine there, but O(pool) x 2000 pairs here would crawl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OUTPUT = ROOT / "data/validation/cross_condition_m3"
LOCK_SEED = 20260816


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", default="train")
    parser.add_argument("--max-members", type=int, default=40,
                        help="cap replicate-group size while keeping CE extremes")
    parser.add_argument("--max-pairs", type=int, default=2000,
                        help="deterministically stride-sample down to this many cross pairs")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing locked manifest if present")
    return parser.parse_args()


def representative_members(
    members: list[int],
    instrument: np.ndarray,
    collision_energy: np.ndarray,
    max_members: int,
) -> list[int]:
    """Bound replicate-group size while keeping each instrument's CE extremes."""
    if len(members) <= max_members:
        return sorted(members)
    selected: set[int] = set()
    members_arr = np.asarray(members, dtype=np.int64)
    for inst in np.unique(instrument[members_arr]):
        subset = members_arr[instrument[members_arr] == inst]
        finite = subset[np.isfinite(collision_energy[subset])]
        if len(finite):
            selected.add(int(finite[np.argmin(collision_energy[finite])]))
            selected.add(int(finite[np.argmax(collision_energy[finite])]))
        selected.add(int(subset[0]))
    remaining = [m for m in members if m not in selected]
    step = max(1, len(remaining) // max(0, max_members - len(selected)))
    selected.update(remaining[::step][: max(0, max_members - len(selected))])
    return sorted(selected)


def build_train_cohort(data_path: Path, fold: str, max_members: int, max_pairs: int) -> dict:
    """Enumerate train-fold cross-condition same-molecule pairs + negatives.

    Same pair definition as the locked benchmark cohort (instrument differs or
    CE delta >= 10), so the training signal targets exactly the condition gap the
    benchmark measures.
    """
    cross_pairs: list[dict] = []

    with h5py.File(data_path, "r") as handle:
        folds = handle["fold"].asstr()[:]
        sim = handle["SIMULATION_CHALLENGE"].asstr()[:]
        ik = handle["INCHIKEY"].asstr()[:]
        instrument = handle["INSTRUMENT_TYPE"].asstr()[:]
        adduct = handle["adduct"].asstr()[:]
        collision_energy = np.asarray(handle["COLLISION_ENERGY"][:])
        precursor_mz = np.asarray(handle["precursor_mz"][:])
        print(f"[m2] loaded {len(folds):,} rows", flush=True)

        valid = (
            (folds == fold)
            & (sim == "False")
            & np.isfinite(precursor_mz)
            & (precursor_mz > 0)
            & (precursor_mz <= 1000)
        )
        rows = np.flatnonzero(valid)
        print(f"[m2] fold={fold} real spectra: {len(rows):,}", flush=True)

        groups: dict[tuple, list[int]] = defaultdict(list)
        for row in rows:
            groups[(ik[row][:14], adduct[row])].append(int(row))
        print(f"[m2] {len(groups):,} same-molecule groups", flush=True)

        # Negative-matching index (vectorised below): (instrument, adduct) -> rows.
        inst_add: dict[tuple, list[int]] = defaultdict(list)
        add_only: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            inst_add[(instrument[row], adduct[row])].append(int(row))
            add_only[adduct[row]].append(int(row))
        ik14_all = np.array([s[:14] for s in ik], dtype=object)
        inst_add = {k: np.asarray(v, dtype=np.int64) for k, v in inst_add.items()}
        add_only = {k: np.asarray(v, dtype=np.int64) for k, v in add_only.items()}

        def match_negative(anchor_row: int, own_ik14: str) -> int:
            amz = float(precursor_mz[anchor_row])
            for pool in (
                inst_add.get((instrument[anchor_row], adduct[anchor_row])),
                add_only.get(adduct[anchor_row]),
            ):
                if pool is None or len(pool) == 0:
                    continue
                diff = np.abs(precursor_mz[pool] - amz)
                diff[ik14_all[pool] == own_ik14] = np.inf
                diff[pool == anchor_row] = np.inf
                k = int(np.argmin(diff))
                if np.isfinite(diff[k]):
                    return int(pool[k])
            raise RuntimeError(f"No negative candidate for row {anchor_row}")

        def to_pair(ik14: str, i: int, j: int) -> dict:
            return {
                "ik14": ik14,
                "rows": [i, j],
                "instrument": [instrument[i], instrument[j]],
                "adduct": [adduct[i], adduct[j]],
                "collision_energy": [
                    None if not np.isfinite(collision_energy[i]) else float(collision_energy[i]),
                    None if not np.isfinite(collision_energy[j]) else float(collision_energy[j]),
                ],
                "precursor_mz": [float(precursor_mz[i]), float(precursor_mz[j])],
            }

        n_cross_total = 0
        n_same_total = 0
        for gi, ((ik14, _adduct), members) in enumerate(groups.items()):
            if len(members) < 2:
                continue
            members = representative_members(members, instrument, collision_energy, max_members)
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    i, j = members[a], members[b]
                    inst_diff = instrument[i] != instrument[j]
                    cei, cej = collision_energy[i], collision_energy[j]
                    both_ce = np.isfinite(cei) and np.isfinite(cej)
                    ce_delta = abs(cei - cej) if both_ce else None
                    if inst_diff or (both_ce and ce_delta >= 10):
                        n_cross_total += 1
                        cross_pairs.append(to_pair(ik14, i, j))
                    elif (not inst_diff) and both_ce and ce_delta < 10:
                        n_same_total += 1
            if (gi + 1) % 10000 == 0:
                print(f"[m2] enumerated {gi + 1:,}/{len(groups):,} groups "
                      f"({n_cross_total:,} cross so far)", flush=True)

        # ---- attach negatives (uses the materialised arrays, no re-open) ----
        print(f"[m2] enumerated {n_cross_total:,} cross pairs total; "
              f"stride-sampling to {max_pairs:,}...", flush=True)
        cross_pairs.sort(key=lambda p: (p["ik14"], p["rows"][0], p["rows"][1]))
        if len(cross_pairs) > max_pairs:
            step = len(cross_pairs) / max_pairs
            cross_pairs = [cross_pairs[int(i * step)] for i in range(max_pairs)]

        for pi, pair in enumerate(cross_pairs):
            neg = match_negative(pair["rows"][0], pair["ik14"])
            pair["negative_row"] = neg
            pair["negative_instrument"] = instrument[neg]
            pair["negative_adduct"] = adduct[neg]
            pair["negative_precursor_mz"] = float(precursor_mz[neg])
            if (pi + 1) % 500 == 0:
                print(f"[m2] matched negatives {pi + 1:,}/{len(cross_pairs):,}", flush=True)

    manifest = {
        "status": "cross_condition_m3_train_cohort",
        "lock_seed": LOCK_SEED,
        "fold": fold,
        "simulation_challenge": "False only",
        "adduct": "same-adduct only",
        "pair_definition": "instrument differs OR |CE| >= 10 (same as locked benchmark)",
        "cross_pairs": cross_pairs,
        "audit": {
            "n_cross_total_enumerated": n_cross_total,
            "n_same_condition_total": n_same_total,
            "n_cross_sampled": len(cross_pairs),
            "n_negative_attached": sum("negative_row" in p for p in cross_pairs),
        },
    }
    return manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "train_pairs.json"

    manifest = build_train_cohort(args.data, args.fold, args.max_members, args.max_pairs)
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8")

    if manifest_path.exists():
        existing = manifest_path.read_bytes()
        if hashlib.sha256(existing).digest() != hashlib.sha256(manifest_bytes).digest():
            if args.force:
                manifest_path.write_bytes(manifest_bytes)
            else:
                raise RuntimeError(
                    f"Manifest drift vs {manifest_path}. Re-run with --force to relock "
                    "(or move/delete the file)."
                )
    else:
        manifest_path.write_bytes(manifest_bytes)

    audit = manifest["audit"]
    print(
        f"M2 cohort locked: fold={args.fold}, real-only, same-adduct | "
        f"enumerated {audit['n_cross_total_enumerated']:,} cross pairs "
        f"({audit['n_same_condition_total']:,} same-condition), "
        f"sampled {audit['n_cross_sampled']:,} with negatives",
        flush=True,
    )
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
