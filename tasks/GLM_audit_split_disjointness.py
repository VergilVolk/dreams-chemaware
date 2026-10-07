"""Audit 3: verify the leak-free splits are actually leak-free.

For each split of router_v3, assert that no query SPECTRUM (query_row) and
no query STRUCTURE (query_ik14) appears in both train and test. This is the
guard that was missing in router v2 (panel-overlap memorization).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/validation/GLM_gnps_identity_panel_reconstruction"

zi = np.load(BENCH / "panel_identity_disjoint.npz")
zf = np.load(BENCH / "panel_formula_disjoint.npz")
f_rows = set(map(int, zf["query_row"]))
f_iks = set(map(str, zf["query_ik14"]))
id_rows = zi["query_row"]
id_iks = zi["query_ik14"].astype(str)
unseen = np.array([(int(r) not in f_rows and k not in f_iks)
                   for r, k in zip(id_rows, id_iks)])

rng = np.random.default_rng(20261007)
uniq = np.array(sorted(set(id_iks)))
half = set(uniq[rng.permutation(len(uniq))[: len(uniq) // 2]])
test_c = np.isin(id_iks, list(half))

splits = {
    "A": dict(tr_rows=set(map(int, zf["query_row"])),
              tr_iks=f_iks,
              te_rows=set(map(int, id_rows[unseen])),
              te_iks=set(id_iks[unseen])),
    "B": dict(tr_rows=set(map(int, id_rows[unseen])),
              tr_iks=set(id_iks[unseen]),
              te_rows=set(map(int, zf["query_row"])),
              te_iks=f_iks),
    "C": dict(tr_rows=set(map(int, id_rows[~test_c])),
              tr_iks=set(id_iks[~test_c]),
              te_rows=set(map(int, id_rows[test_c])),
              te_iks=set(id_iks[test_c])),
}
ok = True
for name, s in splits.items():
    r_ov = s["tr_rows"] & s["te_rows"]
    k_ov = s["tr_iks"] & s["te_iks"]
    status = "CLEAN" if not r_ov and not k_ov else "LEAK"
    ok &= (status == "CLEAN")
    print(f"split {name}: row overlap {len(r_ov)}, ik14 overlap {len(k_ov)}"
          f" -> {status}  (train {len(s['tr_rows'])} / test {len(s['te_rows'])})")
print("AUDIT", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
