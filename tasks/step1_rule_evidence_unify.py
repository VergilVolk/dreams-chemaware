"""Step 1 — unify per-spectrum / pair-level rule evidence and reproduce P1.

This is the data foundation for the weighted-rule + noise fine-tuning direction
(docs/WEIGHTED_RULE_NOISE_TRAINING_PLAN_20260816.md, Step 1). It does NOT train
and does NOT change any label. It produces, and *verifies against recorded
results*, the tables Step 2 will weight:

  1. G0 (matcher consistency): the 3,486-rule matcher used here reproduces the
     existing 335-rule per-spectrum cache on a held-out sample, bit-for-bit.
     This closes the loop that "the newest per-spectrum work" and the P1 rule
     engine share one code path.
  2. G1 (P1 reproduction): re-runs the two original P1 validation scripts
     (molecule-disjoint and within-molecule) under a fresh output dir and
     compares their panel ROC-AUCs to the recorded values. Because every seed
     is fixed, a faithful environment reproduces them to ~1e-6.
  3. Pool scale confirmation: the strict-10ppm pools match the master-plan
     counts (train 112,601 / 5,125,411 / 4,828,326).
  4. Per-spectrum rule profile: a 3,486-bit binary hit vector + per-category
     counts for every unique spectrum in the train/val 10ppm pools.
  5. Pair-level evidence table: per candidate edge (positive and negative) the
     shared-hit and conflict-hit rule counts, overall and per category.
  6. Cross-condition data gate: counts of same-molecule / same-adduct spectra in
     the train fold that differ by instrument or CE (the FN training-data gate).

Canonical matcher: ``pilot_rule_noise_stress.FastRuleMatcher`` fed by the same
light preprocessing (normalize intensity to base peak, prepend a precursor
token) used throughout P1 — so this table and the P1 numbers are on one path.

Memory: peak resident ~1 GB (labels matrix is memory-mapped). Runtime on CPU:
label computation ~15-30 min; pair evidence ~10-20 min; everything else minutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

import pilot_rule_noise_stress as pilot  # noqa: E402


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    p.add_argument("--pool-dir", type=Path, default=ROOT / "data/e1")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/weighted_rule_step1")
    p.add_argument("--rule-tolerance", type=float, default=0.02)
    p.add_argument("--n-consistency", type=int, default=200,
                   help="Number of held-out spectra for the G0 matcher consistency check")
    p.add_argument("--edge-chunk", type=int, default=20000,
                   help="Edges per chunk in the pair-evidence stage")
    p.add_argument("--skip-g1", action="store_true")
    p.add_argument("--skip-labels", action="store_true")
    p.add_argument("--skip-pairs", action="store_true")
    p.add_argument("--skip-cross", action="store_true")
    p.add_argument("--force", action="store_true", help="Overwrite existing stage outputs")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def decode_array(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else str(v)
        for v in values
    ])


def light_preprocess(raw: np.ndarray, precursor: float) -> np.ndarray:
    """Byte-equivalent of ``pilot.preprocess_spectrum(handle, row, None)``.

    ``raw`` is a (2, n_peaks) slice of ``handle["spectrum"]`` (m/z row 0,
    intensity row 1, zero-padded). Returns (n_valid+1, 2) float32 with a
    precursor token prepended; the matcher drops that token internally.
    """
    valid = raw[0] > 0
    peaks = raw[:, valid].T.copy()
    if len(peaks) and peaks[:, 1].max() > 0:
        peaks[:, 1] /= peaks[:, 1].max()
    return np.concatenate([
        np.asarray([[precursor, 1.0]], dtype=np.float32), peaks.astype(np.float32)
    ], axis=0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Stage G0 — matcher consistency against the existing 335-rule cache
# --------------------------------------------------------------------------- #

def stage_g0(args: argparse.Namespace, engine: Any, matcher: Any) -> dict[str, Any]:
    cache_path = ROOT / "data/validation/double_mapping/spectrum_rule_labels.npz"
    cache = np.load(cache_path)
    cache_labels = cache["labels"]          # (45185, 335) uint8
    cache_rows = cache["hdf5_row"]          # (45185,) int64
    cache_names = cache["rule_name"]        # (335,) U160

    # The engine's first 335 rules must be exactly the cache's 335 in the same order.
    engine_names = np.asarray([r.name for r in engine.rules[:335]], dtype="U160")
    name_match = bool(np.array_equal(engine_names, cache_names))

    rng = np.random.default_rng(20260817)
    sample = rng.choice(len(cache_rows), size=min(args.n_consistency, len(cache_rows)),
                        replace=False)

    mismatches = 0
    with h5py.File(args.data, "r") as handle:
        for k in sample:
            row = int(cache_rows[k])
            raw = np.asarray(handle["spectrum"][row], dtype=np.float32)
            precursor = float(handle["precursor_mz"][row])
            # 1) the exact P1 path (handle + preprocess + matcher)
            spec_p1 = pilot.preprocess_spectrum(handle, row, None)
            vec_p1 = matcher(spec_p1, precursor)
            # 2) the bulk path used in Stage D below
            spec_bulk = light_preprocess(raw, precursor)
            vec_bulk = matcher(spec_bulk, precursor)
            if not np.array_equal(vec_p1, vec_bulk):
                mismatches += 1
            # 3) against the existing cache (335-rule prefix)
            if not np.array_equal(vec_p1[:335].astype(np.uint8), cache_labels[k]):
                mismatches += 1

    result = {
        "cache_path": str(cache_path.resolve()),
        "cache_rule_count": int(cache_labels.shape[1]),
        "engine_rule_count": int(len(engine.rules)),
        "engine_prefix_matches_cache_names": name_match,
        "n_sampled": int(len(sample)),
        "n_mismatches": int(mismatches),
        "pass": bool(name_match and mismatches == 0),
    }
    save_json(args.output_dir / "g0_matcher_consistency.json", result)
    return result


# --------------------------------------------------------------------------- #
# Stage G1 — reproduce the P1 validation numbers
# --------------------------------------------------------------------------- #

def _recorded_p1_numbers() -> dict[str, dict[str, float]]:
    def load(name: str, key: str) -> dict[str, dict[str, float]]:
        path = ROOT / f"data/validation/conflict_rule_validation/{name}"
        summary = json.loads(path.read_text(encoding="utf-8"))
        return {
            panel: {
                "auc": float(v["roc_auc_error_detection"]),
                "lo": float(v["roc_auc_95ci"][0]),
                "hi": float(v["roc_auc_95ci"][1]),
            }
            for panel, v in summary["panels"].items()
        }

    return {
        "molecule_disjoint": load("validation_summary.json", "panels"),
        "within_molecule": load("within_molecule_summary.json", "panels"),
    }


def stage_g1(args: argparse.Namespace) -> dict[str, Any]:
    repro_dir = args.output_dir / "g1_repro"
    repro_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg")

    scripts = [
        ("molecule_disjoint", "validate_conflict_rule_panel.py"),
        ("within_molecule", "validate_conflict_rule_panel_paired.py"),
    ]
    runs: dict[str, dict[str, Any]] = {}
    for key, script in scripts:
        cmd = [sys.executable, str(ROOT / "tasks" / script), "--output-dir", str(repro_dir)]
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
        runs[key] = {
            "script": script,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
        }

    recorded = _recorded_p1_numbers()
    comparison: dict[str, Any] = {}
    for key, out in (("molecule_disjoint", "validation_summary.json"),
                     ("within_molecule", "within_molecule_summary.json")):
        repro_path = repro_dir / out
        comparison[key] = {"panels": {}, "present": repro_path.exists()}
        if not repro_path.exists():
            continue
        repro = json.loads(repro_path.read_text(encoding="utf-8"))
        for panel, v in repro["panels"].items():
            r = recorded[key][panel]
            d_auc = abs(float(v["roc_auc_error_detection"]) - r["auc"])
            d_lo = abs(float(v["roc_auc_95ci"][0]) - r["lo"])
            d_hi = abs(float(v["roc_auc_95ci"][1]) - r["hi"])
            comparison[key]["panels"][panel] = {
                "recorded_auc": r["auc"],
                "reproduced_auc": float(v["roc_auc_error_detection"]),
                "abs_diff_auc": d_auc,
                "abs_diff_ci": max(d_lo, d_hi),
            }

    max_diff = 0.0
    present = all(comparison[k]["present"] for k in comparison)
    if present:
        for key in comparison:
            for panel, c in comparison[key]["panels"].items():
                max_diff = max(max_diff, c["abs_diff_auc"])
    tol = 1e-4  # far below the ~0.04 bootstrap CI width; faithful env is ~1e-9
    result = {
        "recorded": recorded,
        "comparison": comparison,
        "tolerance": tol,
        "max_abs_diff_auc": max_diff,
        "returncodes": {k: runs[k]["returncode"] for k in runs},
        "pass": bool(present and all(runs[k]["returncode"] == 0 for k in runs) and max_diff < tol),
    }
    save_json(args.output_dir / "g1_p1_reproduction.json", result)
    return result


# --------------------------------------------------------------------------- #
# Pool scale confirmation
# --------------------------------------------------------------------------- #

EXPECTED_POOL = {
    "e1_train_triplet_pool_10ppm.npz": {"anchors": 112601, "pos": 5125411, "neg": 4828326},
    "e1_val_triplet_pool_10ppm.npz": {"anchors": 21163, "pos": 924136, "neg": 772886},
}


def stage_pool_audit(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    pools: dict[str, Any] = {}
    unique_rows = set()
    for name, expected in EXPECTED_POOL.items():
        p = np.load(args.pool_dir / name)
        anchors = p["anchor_idx"].astype(np.int64)
        pos = p["positive_idx"].astype(np.int64)
        neg = p["negative_idx"].astype(np.int64)
        unique = np.unique(np.concatenate([anchors, pos, neg]))
        unique_rows.update(unique.tolist())
        ok = (len(anchors) == expected["anchors"]
              and len(pos) == expected["pos"] and len(neg) == expected["neg"])
        pools[name] = {
            "anchors": int(len(anchors)), "pos": int(len(pos)), "neg": int(len(neg)),
            "unique_rows": int(len(unique)), "matches_master_plan": ok,
        }
    all_ok = all(v["matches_master_plan"] for v in pools.values())
    result = {
        "pools": pools,
        "union_unique_rows": len(unique_rows),
        "pass": all_ok,
    }
    save_json(args.output_dir / "pool_audit.json", result)
    return result, {"union_unique_rows": np.sort(np.asarray(sorted(unique_rows), dtype=np.int64))}


# --------------------------------------------------------------------------- #
# Stage D — per-spectrum 3,486-rule labels for every unique pool row
# --------------------------------------------------------------------------- #

def stage_labels(args: argparse.Namespace, engine: Any, matcher: Any,
                 unique_rows: np.ndarray) -> dict[str, Any]:
    out_path = args.output_dir / "spectrum_labels.npz"
    if out_path.exists() and not args.force:
        return {"status": "skipped_exists", "output": str(out_path.resolve())}

    n_rules = len(engine.rules)
    categories = sorted({r.category for r in engine.rules})
    cat_idx = {c: np.asarray([i for i, r in enumerate(engine.rules) if r.category == c],
                             dtype=np.int64) for c in categories}
    rule_names = np.asarray([r.name for r in engine.rules], dtype="U160")
    rule_cats = np.asarray([r.category for r in engine.rules], dtype="U8")

    labels = np.zeros((len(unique_rows), n_rules), dtype=np.uint8)
    cat_counts = np.zeros((len(unique_rows), len(categories)), dtype=np.uint16)

    order = np.argsort(unique_rows)
    sorted_rows = unique_rows[order]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))

    with h5py.File(args.data, "r") as handle:
        spectra = np.asarray(handle["spectrum"][sorted_rows], dtype=np.float32)
        precursor = np.asarray(handle["precursor_mz"][sorted_rows], dtype=np.float32)

    for pos in range(len(sorted_rows)):
        row = int(sorted_rows[pos])
        spec = light_preprocess(spectra[pos], float(precursor[pos]))
        vec = matcher(spec, float(precursor[pos])).astype(np.uint8)
        local = int(inverse[pos])
        labels[local] = vec
        for cj, c in enumerate(categories):
            cat_counts[local, cj] = int(vec[cat_idx[c]].sum())
        if (pos + 1) % 10000 == 0 or pos + 1 == len(sorted_rows):
            print(f"  labels {pos + 1:,}/{len(sorted_rows):,}", flush=True)

    np.savez_compressed(
        out_path,
        labels=labels,
        hdf5_row=unique_rows.astype(np.int64),
        cat_counts=cat_counts,
        rule_name=rule_names,
        rule_category=rule_cats,
    )
    report = {
        "status": "spectrum_rule_labels_3486_complete",
        "spectra": int(len(unique_rows)),
        "rules": int(n_rules),
        "categories": categories,
        "mean_rule_prevalence": float(labels.mean()),
        "rules_prevalence_1_to_50_percent": int(
            ((labels.mean(axis=0) >= 0.01) & (labels.mean(axis=0) <= 0.5)).sum()
        ),
        "label_semantics": "observed spectral motif in this spectrum (binary)",
        "matcher": "pilot_rule_noise_stress.FastRuleMatcher (P1 code path)",
        "output": str(out_path.resolve()),
        "sha256": sha256(out_path),
    }
    save_json(args.output_dir / "spectrum_labels.json", report)
    return report


# --------------------------------------------------------------------------- #
# Stage E — pair-level shared / conflict rule evidence per pool edge
# --------------------------------------------------------------------------- #

def _packed_byte_count(a: np.ndarray, b: np.ndarray, idx: np.ndarray | None) -> np.ndarray:
    """Count of positions where both / only-one of ``a`` and ``b`` are set."""
    if idx is not None:
        a, b = a[:, idx], b[:, idx]
    shared = (a & b).sum(axis=1)
    conflict = (a ^ b).sum(axis=1)
    assert int(shared.max()) < 32768 and int(conflict.max()) < 32768
    return shared.astype(np.int16), conflict.astype(np.int16)


def stage_pairs(args: argparse.Namespace, engine: Any, unique_rows: np.ndarray) -> dict[str, Any]:
    labels_mmap = np.load(args.output_dir / "spectrum_labels.npz", mmap_mode="r")["labels"]
    global_to_local = {int(r): i for i, r in enumerate(unique_rows)}
    categories = sorted({r.category for r in engine.rules})
    cat_idx = {c: np.asarray([i for i, r in enumerate(engine.rules) if r.category == c],
                             dtype=np.int64) for c in categories}

    reports = {}
    for name, expected in EXPECTED_POOL.items():
        out_path = args.output_dir / f"{name.replace('.npz', '')}_pair_evidence.npz"
        if out_path.exists() and not args.force:
            reports[name] = {"status": "skipped_exists", "output": str(out_path.resolve())}
            continue
        p = np.load(args.pool_dir / name)
        pos_idx = p["positive_idx"].astype(np.int64)
        neg_idx = p["negative_idx"].astype(np.int64)
        # anchors repeat per their positive/negative candidate lists
        pos_anchor = np.repeat(p["anchor_idx"].astype(np.int64),
                               np.diff(p["positive_ptr"].astype(np.int64)))
        neg_anchor = np.repeat(p["anchor_idx"].astype(np.int64),
                               np.diff(p["negative_ptr"].astype(np.int64)))

        n_pos, n_neg = len(pos_idx), len(neg_idx)
        pos_shared_all = np.empty(n_pos, dtype=np.int16)
        pos_conflict_all = np.empty(n_pos, dtype=np.int16)
        neg_shared_all = np.empty(n_neg, dtype=np.int16)
        neg_conflict_all = np.empty(n_neg, dtype=np.int16)
        pos_shared_cat = np.empty((n_pos, len(categories)), dtype=np.int16)
        pos_conflict_cat = np.empty((n_pos, len(categories)), dtype=np.int16)
        neg_shared_cat = np.empty((n_neg, len(categories)), dtype=np.int16)
        neg_conflict_cat = np.empty((n_neg, len(categories)), dtype=np.int16)

        def fill(side: str, n: int, anchors: np.ndarray, neighbors: np.ndarray,
                 shared_all: np.ndarray, conflict_all: np.ndarray,
                 shared_cat: np.ndarray, conflict_cat: np.ndarray) -> None:
            for start in range(0, n, args.edge_chunk):
                end = min(start + args.edge_chunk, n)
                a_loc = np.asarray([global_to_local[int(r)] for r in anchors[start:end]],
                                   dtype=np.int64)
                b_loc = np.asarray([global_to_local[int(r)] for r in neighbors[start:end]],
                                   dtype=np.int64)
                A = np.asarray(labels_mmap[a_loc], dtype=np.uint8)
                B = np.asarray(labels_mmap[b_loc], dtype=np.uint8)
                shared_all[start:end], conflict_all[start:end] = _packed_byte_count(A, B, None)
                for cj, c in enumerate(categories):
                    s, cf = _packed_byte_count(A, B, cat_idx[c])
                    shared_cat[start:end, cj] = s
                    conflict_cat[start:end, cj] = cf
                print(f"  {side} {name} {end:,}/{n:,}", flush=True)

        fill("pos", n_pos, pos_anchor, pos_idx, pos_shared_all, pos_conflict_all,
             pos_shared_cat, pos_conflict_cat)
        fill("neg", n_neg, neg_anchor, neg_idx, neg_shared_all, neg_conflict_all,
             neg_shared_cat, neg_conflict_cat)

        np.savez_compressed(
            out_path,
            pos_shared_all=pos_shared_all, pos_conflict_all=pos_conflict_all,
            neg_shared_all=neg_shared_all, neg_conflict_all=neg_conflict_all,
            pos_shared_cat=pos_shared_cat, pos_conflict_cat=pos_conflict_cat,
            neg_shared_cat=neg_shared_cat, neg_conflict_cat=neg_conflict_cat,
            categories=np.asarray(categories, dtype="U8"),
        )
        reports[name] = {
            "n_pos": int(n_pos), "n_neg": int(n_neg),
            "mean_pos_shared_all": float(pos_shared_all.mean()),
            "mean_neg_shared_all": float(neg_shared_all.mean()),
            "output": str(out_path.resolve()), "sha256": sha256(out_path),
        }
        print(f"  saved {out_path.name}", flush=True)

    save_json(args.output_dir / "pair_evidence.json", {"pools": reports})
    return {"pools": reports}


# --------------------------------------------------------------------------- #
# Stage F — cross-condition same-molecule counts (FN training-data gate)
# --------------------------------------------------------------------------- #

def stage_cross(args: argparse.Namespace, fold: str = "train") -> dict[str, Any]:
    with h5py.File(args.data, "r") as handle:
        folds = decode_array(handle["fold"][:])
        adducts = decode_array(handle["adduct"][:])
        ik14 = np.asarray([v[:14] for v in decode_array(handle["INCHIKEY"][:])])
        inst = decode_array(handle["INSTRUMENT_TYPE"][:])
        ce = np.asarray(handle["COLLISION_ENERGY"][:])
        sim = decode_array(handle["SIMULATION_CHALLENGE"][:])

    sel = np.flatnonzero(folds == fold)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for row in sel:
        groups[(ik14[row], adducts[row])].append(int(row))

    def classify(i: int, j: int) -> str:
        inst_diff = inst[i] != inst[j]
        ci, cj = ce[i], ce[j]
        both_ce = bool(np.isfinite(ci) and np.isfinite(cj))
        ce_diff = both_ce and abs(float(ci) - float(cj)) >= 10
        if inst_diff and ce_diff:
            return "both"
        if inst_diff:
            return "instrument"
        if ce_diff:
            return "ce"
        return "same"

    counters = defaultdict(Counter)   # (adduct, realm) -> Counter(kind)
    mol_sets = defaultdict(set)       # (adduct, realm) -> set of molecules with >=1 cross pair
    for (ik, adduct), members in groups.items():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                kind = classify(i, j)
                realms = ["all"]  # 'real' only when both spectra are non-simulated
                if sim[i] == "False" and sim[j] == "False":
                    realms.append("real")
                for realm in realms:
                    counters[(adduct, realm)][kind] += 1
                    if kind in ("instrument", "ce", "both"):
                        mol_sets[(adduct, realm)].add(ik)

    summary = {
        "fold": fold,
        "n_spectra": int(len(sel)),
        "n_molecule_adduct_groups": len(groups),
        "by_adduct": {},
        "interpretation": (
            "cross = same molecule, same adduct, instrument differs OR |ΔCE|>=10. "
            "real = SIMULATION_CHALLENGE == False on both spectra. "
            "This is the FN training-data gate: enough real cross-condition pairs (>= thousands)?"
        ),
    }
    adducts_of_interest = ["[M+H]+"] + sorted({k[0] for k in counters if k[0] != "[M+H]+"})
    for adduct in adducts_of_interest:
        for realm in ("real", "all"):
            c = counters.get((adduct, realm))
            if c is None:
                continue
            mol = mol_sets.get((adduct, realm), set())
            n_cross = c["instrument"] + c["ce"] + c["both"]
            summary["by_adduct"][f"{adduct}|{realm}"] = {
                "n_same_condition": int(c["same"]),
                "n_cross_total": int(n_cross),
                "instrument_only": int(c["instrument"]),
                "ce_only": int(c["ce"]),
                "both": int(c["both"]),
                "n_molecules_with_cross": int(len(mol)),
            }
    save_json(args.output_dir / "cross_condition_counts.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading rule engine (3,486 rules, P1 code path)...", flush=True)
    RuleEngine = pilot.load_rule_engine_class()
    engine = RuleEngine(tolerance=args.rule_tolerance, use_massbank=True)
    matcher = pilot.FastRuleMatcher(engine, args.rule_tolerance)
    n_rules = len(engine.rules)
    if n_rules != 3486:
        print(f"WARNING: expected 3,486 rules, engine returned {n_rules}", flush=True)
    print(f"Rules: {n_rules}", flush=True)

    gates: dict[str, Any] = {}

    print("\n[G0] matcher consistency vs existing 335-rule cache", flush=True)
    gates["g0"] = stage_g0(args, engine, matcher)
    print(json.dumps(gates["g0"], ensure_ascii=False, indent=2), flush=True)

    if not args.skip_g1:
        print("\n[G1] reproducing P1 validation numbers", flush=True)
        gates["g1"] = stage_g1(args)
        print(json.dumps({k: v for k, v in gates["g1"].items() if k != "recorded"},
                         ensure_ascii=False, indent=2), flush=True)

    print("\n[pool] strict-10ppm pool scale confirmation", flush=True)
    pool_audit, unique = stage_pool_audit(args)
    gates["pool_audit"] = pool_audit
    print(json.dumps(pool_audit, ensure_ascii=False, indent=2), flush=True)

    if not args.skip_labels:
        print("\n[labels] per-spectrum 3,486-rule profiles", flush=True)
        gates["labels"] = stage_labels(args, engine, matcher, unique["union_unique_rows"])
        print(json.dumps({k: v for k, v in gates["labels"].items() if k != "sha256"},
                         ensure_ascii=False, indent=2), flush=True)

    if not args.skip_pairs:
        print("\n[pairs] pair-level shared/conflict evidence", flush=True)
        gates["pairs"] = stage_pairs(args, engine, unique["union_unique_rows"])
        print(json.dumps(gates["pairs"], ensure_ascii=False, indent=2), flush=True)

    if not args.skip_cross:
        print("\n[cross] cross-condition same-molecule counts (FN data gate)", flush=True)
        gates["cross"] = stage_cross(args)
        print(json.dumps(gates["cross"], ensure_ascii=False, indent=2), flush=True)

    verdict = {
        "g0_pass": bool(gates.get("g0", {}).get("pass")),
        "g1_pass": bool(gates.get("g1", {}).get("pass")) if "g1" in gates else None,
        "pool_pass": bool(gates.get("pool_audit", {}).get("pass")),
        "note": (
            "G0/G1/pool are hard gates. The cross-condition count is informational: "
            "the '>= thousands real cross-condition pairs' FN decision is the user's call."
        ),
    }
    gates["verdict"] = verdict
    save_json(args.output_dir / "STEP1_SUMMARY.json", gates)
    print("\n=== STEP1 SUMMARY ===", flush=True)
    print(json.dumps(verdict, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
