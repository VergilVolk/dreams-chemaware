#!/usr/bin/env python3
"""Small contracts for the CALMS cross-study application adapters."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from score_crossstudy_identity_multiencoder import load_embedding


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="calms_application_adapter_") as raw:
        root = Path(raw)
        legacy = root / "official_legacy.npz"
        np.savez_compressed(
            legacy,
            entity_id=np.asarray(["e1", "e2"]),
            embedding=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        loaded = load_embedding(legacy)
        assert set(loaded) == {"e1", "e2"}
        assert np.array_equal(
            loaded["e1"], np.asarray([1.0, 0.0], dtype=np.float32)
        )

        ambiguous = root / "ambiguous.npz"
        matrix = np.asarray([[1.0, 0.0]], dtype=np.float32)
        np.savez_compressed(
            ambiguous,
            entity_id=np.asarray(["e1"]),
            embedding=matrix,
            embeddings=matrix,
        )
        try:
            load_embedding(ambiguous)
        except RuntimeError:
            pass
        else:
            raise AssertionError("ambiguous embedding caches must fail closed")

        nonfinite = root / "nonfinite.npz"
        np.savez_compressed(
            nonfinite,
            entity_id=np.asarray(["e1"]),
            embeddings=np.asarray([[np.nan, 0.0]], dtype=np.float32),
        )
        try:
            load_embedding(nonfinite)
        except RuntimeError:
            pass
        else:
            raise AssertionError("non-finite embedding caches must fail closed")

    print("PASS: CALMS application adapter contracts", flush=True)


if __name__ == "__main__":
    main()
