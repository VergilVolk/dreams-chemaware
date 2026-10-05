"""Static and numerical utility tests for M2 current-geometry crossfit."""
from __future__ import annotations
import ast
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from build_noise_final_dynamic_direct_m2_crossfit import permute_query_features, strict_bool


def main() -> None:
    for filename in ("build_noise_final_dynamic_direct_m2_crossfit.py", "validate_noise_final_dynamic_direct_m2_crossfit.py"):
        ast.parse((ROOT / "tasks" / filename).read_text(encoding="utf-8"))
    x = np.arange(60, dtype=np.float32).reshape(20, 3)
    folds = np.repeat(np.arange(5), 4)
    first = permute_query_features(x, folds, 17)
    second = permute_query_features(x, folds, 17)
    if not np.array_equal(first, second) or np.array_equal(first, x):
        raise RuntimeError("formula-fold permutation is not deterministic/effective")
    for fold in range(5):
        if set(map(tuple, first[folds == fold])) != set(map(tuple, x[folds == fold])):
            raise RuntimeError("permutation crossed formula-fold boundaries")
    values = strict_bool(pd.Series([True, False, "1", "0"]), "fixture")
    if values.tolist() != [True, False, True, False]:
        raise RuntimeError("strict boolean parsing failed")
    print("[test_noise_final_dynamic_direct_m2_crossfit] PASS")


if __name__ == "__main__":
    main()
