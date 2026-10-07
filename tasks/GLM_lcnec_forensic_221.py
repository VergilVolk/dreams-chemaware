"""Forensic check of the gate-passed dark module (m/z 221.9846).

Question: is the module a blank/contaminant or a real sample constituent?
Uses the frozen dark-feature EIC matrix (85 samples x 221 family rows).
Steps:
  1. Locate the family whose m/z matches 221.9846 (via the priority-module
     mgf precursors -> EIC family index if derivable; here we scan the
     module mgf and match by family ordering if documented, else by
     correlation of presence patterns with the module list).
  2. Break down area/maximum by sample-type prefix (blank / QC / tumor /
     normal naming convention in sample ids).
  3. Report presence ratio per group.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EIC = ROOT / "data/validation/lcnec_hsst3n_dark_eic_gate/dark_feature_eic_matrix.npz"
MGF = ROOT / "data/validation/lcnec_hsst3n_priority_ms2/priority_dark_modules.mgf"

z = np.load(EIC, allow_pickle=True)
area, fam = z["area"], z["family_id"]
sids = [str(s) for s in z["sample_id"]]
prefixes = sorted({s[:3] for s in sids})
print("n samples:", len(sids), "prefixes:", prefixes)
print("sample examples:", sids[:8], "...", sids[-4:])

# parse module precursors
prec = []
cur = None
with MGF.open(encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if line.startswith("PEPMASS="):
            cur = float(line.split("=")[1].split()[0])
        elif line == "END IONS" and cur is not None:
            prec.append(cur)
            cur = None
print("modules in mgf:", len(prec))

# how do families map to modules? count families and check any index file
import collections
c = collections.Counter(fam.tolist())
print("n families:", len(c), "rows per family (top5):",
      sorted(c.values(), reverse=True)[:5])

# try: family order == module order (30 modules, 221 EIC rows)
fam_ids = sorted(set(fam.tolist()))
print("family ids: first/last:", fam_ids[:5], fam_ids[-5:])

# group samples by prefix
groups = {}
for p in prefixes:
    groups[p] = [i for i, s in enumerate(sids) if s.startswith(p)]
for p, idx in groups.items():
    print(f"prefix {p}: {len(idx)} samples")

# If family_id enumerates modules 0..29, examine family nearest 221.9846 by
# matching mgf order; report presence per group for ALL families is too much,
# so first report the family whose row count matches and defer mapping.
target = 221.9846
print("target precursor:", target)
np.save(str(ROOT / "data/validation/GLM_track2_census/"
             "forensic_sample_ids.npy"), np.asarray(sids))
print("saved sample ids for the mapping step")
