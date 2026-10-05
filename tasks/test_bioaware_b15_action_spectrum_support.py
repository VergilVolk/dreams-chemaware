#!/usr/bin/env python
"""Small unit checks for B15 identifiers and spectrum preprocessing."""
from __future__ import annotations

import numpy as np
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.audit_bioaware_b15_action_spectrum_support import (
    KGM_SOURCE,
    ST_SOURCE,
    fixed_tensor,
    normalise_query_id,
)


def main() -> None:
    st = "ST001154:HILICNEG:x:controllerType=0 controllerNumber=1 scan=1"
    assert normalise_query_id(ST_SOURCE, ST_SOURCE + "::" + st) == (st, st)
    assert normalise_query_id(KGM_SOURCE, KGM_SOURCE + "::M100T2::repeat=7") == (
        "M100T2", KGM_SOURCE + "::M100T2"
    )
    internal = "M3NEG:BV2cell__hilic:1:scan=2"
    assert normalise_query_id("BV2cell", internal) == (internal, internal)
    raw = np.asarray([[10, 20, 30], [1, 3, 2]], dtype=np.float32)
    tensor = fixed_tensor(raw, 100.0, 2)
    assert tensor.shape == (3, 2)
    assert np.allclose(tensor[0], [100.0, 1.1])
    assert np.allclose(tensor[1:, 0], [20.0, 30.0])
    assert np.isclose(tensor[1:, 1].max(), 1.0)
    print("[test_bioaware_b15_action_spectrum_support] PASS")


if __name__ == "__main__":
    main()
