"""Merge the staged spec2vec_2026 scores into the 15-method bundle."""
import json
from pathlib import Path

import numpy as np

SRC = Path("data/validation/GLM_gnps_article_benchmark_challengers/run_local/bundle")
STAGING = Path("data/validation/GLM_challenger_scores/staging")
DST = Path("data/validation/GLM_gnps_article_benchmark_s2v26/run15/bundle")

NEW = "spec2vec_2026_retrained"

DST.mkdir(parents=True, exist_ok=True)
with np.load(SRC / "method_scores.npz") as z:
    names = [str(v) for v in z["method_names"]]
    old = {p: np.asarray(z[f"scores_{p}"]) for p in
           ("identity_disjoint", "formula_disjoint")}
assert NEW not in names
all_names = names + [NEW]
for panel in ("identity_disjoint", "formula_disjoint"):
    add = np.load(STAGING / f"scores_{panel}_{NEW}.npy").astype(np.float32)
    assert add.shape == old[panel].shape[1:], (panel, add.shape, old[panel].shape)
    old[panel] = np.vstack([old[panel].astype(np.float32), add])
np.savez_compressed(DST / "method_scores.npz",
                    method_names=np.asarray(all_names),
                    **{f"scores_{p}": old[p] for p in old})
meta = json.loads((SRC / "method_scores.npz.json").read_text(encoding="utf-8"))
meta["methods"] = all_names
meta["information_levels"]["spectrum_only"] = \
    meta["information_levels"].get("spectrum_only", []) + [NEW]
(DST / "method_scores.npz.json").write_text(json.dumps(meta, indent=2),
                                             encoding="utf-8")
print("15-method bundle written:", len(all_names), "methods")
print("identity", old["identity_disjoint"].shape,
      "formula", old["formula_disjoint"].shape)
