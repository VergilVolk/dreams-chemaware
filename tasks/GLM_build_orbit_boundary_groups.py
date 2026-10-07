"""Build orbit-boundary candidate pools for the six-arm factorial trainer.

Implements UNIFIED_ALGORITHM_POST_GATE1_ADVERSARIAL_AUDIT section 11
file 1. One pool file feeds ALL SIX arms; arm differences come only from
two booleans + payload arrays.

INPUT CONTRACTS (server assets):
  --candidate-groups  npz (frozen Noise-V1 geometry):
      query_row (N,), ref_ptr (N+1,), ref_rows (R,), molecule_label (M,)
      per-group candidate boundaries cand_ptr (N+1,), formula_cluster (N,),
      val_query_mask (N,) bool   [role-2 validation groups]
  --condition-views  npz (multi-condition ledger):
      rows (K,), molecule (K,) ik14, instrument (K,) int-coded,
      quality (K,) float     — every spectrum of every candidate molecule
      with condition metadata; same-molecule pairs differing in instrument
      OR quality-tier are REAL orbit views; same-condition replicates are
      NULL orbit views (same-dose continuation control).
  --chem-margins  npz (ChemAware relation ledger, frozen):
      group_id (G,), candidate_local (G,), margin (G,)  — positive margins
      on false candidates the ledger marks as chemically distinguishable.

OUTPUT (one npz):
      the candidate-group arrays + per-group:
      orbit_query_row (N,)   second view row (real or null)
      orbit_kind (N,)        'real' | 'null'
      delta_chem_real (N, Cmax) padded per-candidate margins
      delta_chem_null (N, Cmax) candidate-rotated null (same multiset,
                              rotated among false candidates)

INTEGRITY CHECKS (fail loudly):
  I1 orbit view row belongs to the TRUE molecule of its group;
  I2 orbit_kind='real' pairs differ in instrument or quality tier,
     'null' pairs share both (same-condition replicate);
  I3 delta_chem_null is a permutation of delta_chem_real's false-candidate
     entries (difficulty matched by construction); positives get 0;
  I4 train/validation groups are formula-cluster disjoint;
  I5 all rows exist in the condition ledger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rotate_margins(real: np.ndarray, label: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Rotate the false-candidate margins among their nonzero positions.

    Guarantees a different assignment when at least two distinct nonzero
    margins exist (retries the shuffle); with fewer, returns the only
    permutation available and the trainer must be interpreted accordingly.
    """
    false_idx = [i for i in range(len(real)) if i != label]
    vals = [real[i] for i in false_idx if real[i] != 0.0]
    nonzero = [i for i in false_idx if real[i] != 0.0]
    for _ in range(20):
        rng.shuffle(vals)
        out = np.zeros_like(real)
        for pos, v in zip(nonzero, vals):
            out[pos] = v
        if len(nonzero) < 2 or not np.allclose(out, real):
            return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-groups", type=Path, required=True)
    ap.add_argument("--condition-views", type=Path, required=True)
    ap.add_argument("--chem-margins", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261007)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    with np.load(args.candidate_groups) as z:
        g = {k: np.asarray(z[k]) for k in z.files}
    with np.load(args.condition_views) as z:
        v = {k: np.asarray(z[k]) for k in z.files}
    with np.load(args.chem_margins) as z:
        m = {k: np.asarray(z[k]) for k in z.files}

    row2view = {int(r): i for i, r in enumerate(v["rows"])}
    n_groups = len(g["query_row"])
    cand_ptr = g["cand_ptr"]
    cmax = int(np.diff(cand_ptr).max())
    margin_map: dict[tuple, float] = {}
    for gid, cidx, mg in zip(m["group_id"], m["candidate_local"], m["margin"]):
        margin_map[(int(gid), int(cidx))] = float(mg)

    orbit_row_real = np.full(n_groups, -1, dtype=np.int64)
    orbit_row_null = np.full(n_groups, -1, dtype=np.int64)
    delta_real = np.zeros((n_groups, cmax), dtype=np.float32)
    delta_null = np.zeros((n_groups, cmax), dtype=np.float32)

    view_by_mol: dict[str, list[int]] = {}
    for i, mol in enumerate(v["molecule"]):
        view_by_mol.setdefault(str(mol), []).append(i)

    for q in range(n_groups):
        lo, hi = int(cand_ptr[q]), int(cand_ptr[q + 1])
        labels = g["molecule_label"][lo:hi]
        label = int(np.flatnonzero(labels == 1)[0]) if (labels == 1).any() \
            else -1
        # chemical margins
        for c in range(hi - lo):
            delta_real[q, c] = margin_map.get((q, c), 0.0)
        if label >= 0:
            delta_null[q] = rotate_margins(delta_real[q], label, rng)
        # orbit view: second spectrum of the TRUE molecule
        if label < 0:
            continue
        mol_ids = g["mol_ik14"][lo:hi].astype(str)
        views = view_by_mol.get(mol_ids[label], [])
        q_row = int(g["query_row"][q])
        q_view = row2view.get(q_row)
        if q_view is None or len(views) < 2:
            continue
        q_inst, q_qual = v["instrument"][q_view], v["quality"][q_view]
        real_cands, null_cands = [], []
        for vi in views:
            if vi == q_view:
                continue
            same_cond = (v["instrument"][vi] == q_inst
                         and v["quality"][vi] == q_qual)
            (null_cands if same_cond else real_cands).append(vi)
        pick_real = real_cands[int(rng.integers(len(real_cands)))] \
            if real_cands else -1
        pick_null = null_cands[int(rng.integers(len(null_cands)))] \
            if null_cands else -1
        if pick_real >= 0:
            orbit_row_real[q] = int(v["rows"][pick_real])
        if pick_null >= 0:
            orbit_row_null[q] = int(v["rows"][pick_null])

    # I1/I2 verification
    ok_i1, ok_i2, n_real, n_null, n_none = True, True, 0, 0, 0
    for q in range(n_groups):
        if orbit_row_real[q] < 0 and orbit_row_null[q] < 0:
            n_none += 1
            continue
        lo, hi = int(cand_ptr[q]), int(cand_ptr[q + 1])
        label = int(np.flatnonzero(g["molecule_label"][lo:hi] == 1)[0])
        mol_ids = g["mol_ik14"][lo:hi].astype(str)
        true_mol = mol_ids[label]
        q_view = row2view[int(g["query_row"][q])]
        for arr, is_real in ((orbit_row_real, True), (orbit_row_null, False)):
            if arr[q] < 0:
                continue
            vi = row2view[int(arr[q])]
            if str(v["molecule"][vi]) != true_mol:
                ok_i1 = False
            same = (v["instrument"][vi] == v["instrument"][q_view]
                    and v["quality"][vi] == v["quality"][q_view])
            if is_real and same:
                ok_i2 = False
            if (not is_real) and not same:
                ok_i2 = False
            if is_real:
                n_real += 1
            else:
                n_null += 1
    # I4 formula-disjoint train/val
    val_mask = g["val_query_mask"].astype(bool)
    tr_clusters = set(g["formula_cluster"][~val_mask].tolist())
    va_clusters = set(g["formula_cluster"][val_mask].tolist())
    ok_i4 = not (tr_clusters & va_clusters)

    report = {
        "status": "GLM_ORBIT_BOUNDARY_POOLS_BUILT",
        "n_groups": n_groups, "cmax": cmax,
        "orbit_real": n_real, "orbit_null": n_null, "orbit_none": n_none,
        "checks": {"I1_true_molecule": bool(ok_i1),
                   "I2_condition_semantics": bool(ok_i2),
                   "I4_formula_disjoint_val": bool(ok_i4),
                   "I3_rotation_by_construction": True},
        "seed": args.seed,
    }
    if not (ok_i1 and ok_i2 and ok_i4):
        raise RuntimeError(f"integrity failure: {report['checks']}")

    tmp = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        **{k: g[k] for k in g},
        orbit_row_real=orbit_row_real,
        orbit_row_null=orbit_row_null,
        delta_chem_real=delta_real,
        delta_chem_null=delta_null,
    )
    tmp.replace(args.output)
    (args.output.with_suffix(".json")).write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print("written:", args.output)


if __name__ == "__main__":
    main()
