"""Build leakage-free, cross-condition G8R training and validation subsets.

G5--G7 repeatedly selected a sorted prefix and tuned on the historical eval
split.  This utility creates a fresh *inner* development split from the old
training fold, grouped by IK14 across every adduct.  Thus no connectivity label
or spectrum of a validation molecule can appear as a training anchor, positive,
or explicit hard negative.

Only anchors that have a genuine same-IK14/same-adduct cross-condition peer are
retained.  The resulting JSON files are locked inputs to G8R; this script is
metadata-only and performs no neural-network forward pass.
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
DEFAULT_MANIFEST = ROOT / "tasks/massspecgym_isomers/dataset_manifest.json"
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OUT = ROOT / "tasks/massspecgym_isomers/g8r_locked"


def decode(x: object) -> str:
    return x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)


def args_() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--train-anchors", type=int, default=10000)
    p.add_argument("--val-anchors", type=int, default=2000)
    p.add_argument("--max-per-identity-adduct", type=int, default=4)
    p.add_argument("--neg-grade", choices=("near", "mid", "far", "near+mid", "all"), default="near+mid",
                   help="Which MCES isomer grade(s) to keep as explicit hard negatives. "
                        "near=MCES 0-2, mid=3-5, far=6-10+; '+' joins grades. "
                        "G8R locks near+mid (0-5): far is trivially separable and would "
                        "dilute the hard-negative gradient; near-only leaves the val gate "
                        "too thin (220 pairs). 'all' reproduces the old far-dominated mix.")
    p.add_argument("--val-fraction", type=float, default=0.18)
    p.add_argument("--seed", type=int, default=20260821)
    return p.parse_args()


def stable_val(ik14: str, fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{ik14}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def cross_condition_rows(entries: list[dict], instrument: np.ndarray, ce: np.ndarray) -> set[int]:
    by_identity: dict[tuple[str, str], list[int]] = defaultdict(list)
    for e in entries:
        by_identity[(e["ik14"], e["adduct"])].append(int(e["anchor_row"]))
    out: set[int] = set()
    for rows in by_identity.values():
        for row in rows:
            for peer in rows:
                if peer == row:
                    continue
                if instrument[row] != instrument[peer] or (
                    np.isfinite(ce[row]) and np.isfinite(ce[peer]) and abs(ce[row] - ce[peer]) >= 10
                ):
                    out.add(row)
                    break
    return out


def is_cross_pair(left: int, right: int, instrument: np.ndarray, ce: np.ndarray) -> bool:
    return instrument[left] != instrument[right] or (
        np.isfinite(ce[left]) and np.isfinite(ce[right]) and abs(ce[left] - ce[right]) >= 10
    )


def select(entries: list[dict], allowed_ik: set[str], eligible_rows: set[int], ik_all: np.ndarray,
           instrument: np.ndarray, ce: np.ndarray, n: int, max_per_identity: int, seed: int,
           neg_grade: str = "near+mid") -> list[dict]:
    # far (MCES >= 6) same-formula isomers are trivially separable and would
    # dilute the hard-negative gradient; near+mid (MCES 0-5) is the locked,
    # audited "mass-proximal hard negative" set (G8R review 2026-08-21).
    grades: set[str] | None = None if neg_grade == "all" else set(neg_grade.split("+"))
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        row = int(e["anchor_row"])
        if e["ik14"] in allowed_ik and row in eligible_rows:
            # A hard-negative spectrum must also remain inside this split, and
            # must match the locked MCES isomer grade(s).
            copied = dict(e)
            copied["neg"] = [x for x in e["neg"]
                             if ik_all[int(x["row"])] in allowed_ik
                             and (grades is None or x.get("grade") in grades)]
            groups[(copied["ik14"], copied["adduct"])].append(copied)
    rng = np.random.default_rng(seed)
    pools = list(groups.values())
    rng.shuffle(pools)
    chosen: list[dict] = []
    # A true cross-condition pair, not an isolated anchor, is the sampling
    # unit.  Otherwise the chosen subset can silently remove an anchor's only
    # real positive and revert the experiment to synthetic-noise training.
    for items in pools:
        if len(chosen) + 2 > n:
            break
        pairs = [
            (i, j) for i in range(len(items)) for j in range(i + 1, len(items))
            if is_cross_pair(int(items[i]["anchor_row"]), int(items[j]["anchor_row"]), instrument, ce)
        ]
        if not pairs:
            continue
        i, j = pairs[int(rng.integers(len(pairs)))]
        chosen.extend((items[i], items[j]))
    rng.shuffle(chosen)
    return chosen


def audit(entries: list[dict], instrument: np.ndarray, ce: np.ndarray) -> dict:
    ids = {(e["ik14"], e["adduct"]) for e in entries}
    iks = {e["ik14"] for e in entries}
    grade_counts: dict[str, int] = {}
    n_neg = 0
    for e in entries:
        for x in e["neg"]:
            n_neg += 1
            grade_counts[x.get("grade", "?")] = grade_counts.get(x.get("grade", "?"), 0) + 1
    return {
        "n_anchors": len(entries),
        "n_ik14": len(iks),
        "n_identity_adduct": len(ids),
        "cross_condition_anchor_fraction": len(cross_condition_rows(entries, instrument, ce)) / len(entries) if entries else 0.0,
        "hard_negative_anchor_fraction": sum(bool(e["neg"]) for e in entries) / len(entries) if entries else 0.0,
        "max_entries_per_identity_adduct": max((sum(1 for x in entries if (x["ik14"], x["adduct"]) == key) for key in ids), default=0),
        "n_hard_negative_pairs": n_neg,
        "hard_negative_grade_counts": grade_counts,
    }


def write_locked(path: Path, payload: dict) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    a = args_()
    if not 0 < a.val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    entries = manifest["train"]
    with h5py.File(a.data, "r") as h:
        ik_all = np.asarray([decode(x)[:14] for x in h["INCHIKEY"][:]], dtype=object)
        inst = np.asarray([decode(x) for x in h["INSTRUMENT_TYPE"][:]], dtype=object)
        ce = np.asarray(h["COLLISION_ENERGY"][:], dtype=float)
    all_ik = sorted({e["ik14"] for e in entries})
    val_ik = {ik for ik in all_ik if stable_val(ik, a.val_fraction, a.seed)}
    train_ik = set(all_ik) - val_ik
    eligible = cross_condition_rows(entries, inst, ce)
    train = select(entries, train_ik, eligible, ik_all, inst, ce, a.train_anchors, a.max_per_identity_adduct, a.seed, a.neg_grade)
    val = select(entries, val_ik, eligible, ik_all, inst, ce, a.val_anchors, a.max_per_identity_adduct, a.seed + 1, a.neg_grade)
    train_iks = {e["ik14"] for e in train}; val_iks = {e["ik14"] for e in val}
    if train_iks & val_iks:
        raise RuntimeError("IK14 leakage between G8R train and validation")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "purpose": "G8R inner development split; do not use historical eval as final evidence",
        "source_manifest": str(a.manifest), "source_fold": "train",
        "grouping": "IK14 across all adducts", "pair_requirement": "same IK14 + same adduct and instrument differs OR |CE|>=10",
        "seed": a.seed, "val_fraction_at_ik14_level": a.val_fraction,
        "max_per_identity_adduct": a.max_per_identity_adduct,
        "neg_grade": a.neg_grade,
    }
    train_payload = {"meta": common | {"role": "train", "audit": audit(train, inst, ce)}, "entries": train}
    val_payload = {"meta": common | {"role": "validation", "audit": audit(val, inst, ce)}, "entries": val}
    train_sha = write_locked(a.output_dir / "train.json", train_payload)
    val_sha = write_locked(a.output_dir / "val.json", val_payload)
    report = {"train": train_payload["meta"]["audit"] | {"sha256": train_sha},
              "validation": val_payload["meta"]["audit"] | {"sha256": val_sha},
              "n_source_train_ik14": len(all_ik), "n_partition_train_ik14": len(train_ik), "n_partition_val_ik14": len(val_ik),
              "ik14_overlap": 0}
    (a.output_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Locked train: {a.output_dir / 'train.json'}")
    print(f"Locked validation: {a.output_dir / 'val.json'}")


if __name__ == "__main__":
    main()
