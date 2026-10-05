#!/usr/bin/env python
from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import extend_gnps_article_score_bundle as extension
from gnps_pair_score_cache import PANELS, write_pair_score_cache
from test_gnps_pair_score_benchmark import synthetic_benchmark


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        benchmark = synthetic_benchmark(root)
        pd.DataFrame({"row": np.arange(8, dtype=np.int64)}).to_csv(
            benchmark / "manifest.csv.gz", index=False, compression="gzip",
        )
        base = root / "base.npz"
        base_values = np.asarray([[0.8, 0.9, 0.1, 0.8, 0.7, 0.2]], dtype=np.float32)
        with base.open("wb") as handle:
            np.savez(
                handle,
                method_names=np.asarray(["official_dreams"]),
                scores_identity_disjoint=base_values,
                scores_formula_disjoint=base_values,
            )
        base.with_suffix(".npz.json").write_text("{}", encoding="utf-8")

        rows = np.arange(8, dtype=np.int64)
        embeddings = np.eye(8, dtype=np.float32)
        embedding_path = root / "embedding.npz"
        np.savez(embedding_path, rows=rows, embeddings=embeddings)
        pair_cache = root / "pair_cache"
        pair = np.asarray([0.95, 0.9, 0.1, 0.7, 0.8, 0.2], dtype=np.float32)
        write_pair_score_cache(
            pair_cache, benchmark, {"name": "external"},
            {name: pair for name in PANELS},
        )
        output = root / "extended.npz"
        previous = extension.arguments
        extension.arguments = lambda: SimpleNamespace(
            benchmark=benchmark,
            base_bundle=base,
            embedding_method=[f"chemaware={embedding_path}"],
            pair_score_method=[f"public_pair_model={pair_cache}"],
            output=output,
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                extension.main()
        finally:
            extension.arguments = previous
        with np.load(output, allow_pickle=False) as body:
            assert list(map(str, body["method_names"])) == [
                "official_dreams", "chemaware", "public_pair_model",
            ]
            for name in PANELS:
                assert body[f"scores_{name}"].shape == (3, 6)
                assert np.array_equal(body[f"scores_{name}"][2], pair)
        assert output.with_suffix(".npz.json").is_file()
    print("PASS: ChemAware GNPS article-bundle extension contracts")


if __name__ == "__main__":
    main()
