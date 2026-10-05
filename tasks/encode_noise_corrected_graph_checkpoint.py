#!/usr/bin/env python
"""Encode every row used by the corrected Noise graph with one checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from evaluate_noise_dreams_native import encode_rows, sha256_file
from noise_final_core import CandidateGraph
from train_e1_identity import load_base_model
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("corrected-graph encoding requires an allocated GPU")
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (args.graph, args.data, args.checkpoint, args.architecture_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    graph = CandidateGraph(args.graph)
    rows = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row)))
    model, checkpoint_kind = load_base_model(
        args.checkpoint, args.architecture_checkpoint, torch.device("cuda"), 100,
    )
    preprocessor = SpectrumPreprocessor(
        dformat=DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    embeddings = encode_rows(
        model, rows, args.data, preprocessor,
        batch_size=args.batch_size, device=torch.device("cuda"), label="stage1-rrf",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{args.output.name}.", suffix=".npz", dir=args.output.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez(temporary, rows=rows, embeddings=embeddings)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "status": "noise_corrected_graph_checkpoint_encoding_complete",
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "graph_sha256": sha256_file(args.graph),
        "rows": int(len(rows)),
        "embedding_dimension": int(embeddings.shape[1]),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
