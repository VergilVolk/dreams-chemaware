"""Standalone fold-1 relation diagnostics for an already-trained checkpoint.

The v2 trainers measure fold-1 diagnostics before and after training, but
the averaged arm reuses run 2347055's checkpoint, which predates that
instrument.  This scorer rebuilds the native model from the Stage-1 warm
start (the shared "before" for every arm), optionally overlays a slim
continuation checkpoint (the "after"), and runs the identical fold-1
molecule-max diagnostic, so all three arms compare on the same scale.

Author: GLM-5.3 via DeepSeek Harness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from e1_checkpoint_io import torch_load_compat
from train_noise_dreams_native_residual_stage2 import construct_native_model
from train_noise_relation_t1_t3_v2 import fold1_relation_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--candidate-checkpoint", type=Path, default=None,
        help="slim continuation checkpoint; omit to probe the warm start alone",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--triplet-loss-margin", type=float, default=0.1)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (
        args.data, args.corpus / "validation.npz", args.corpus / "report.json",
        args.warm_start_checkpoint, args.architecture_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if "noise_relation_t1_t3_corpus_v2_complete" not in (
        args.corpus / "report.json"
    ).read_text(encoding="utf-8"):
        raise RuntimeError("standalone scorer requires the v2 corpus")

    torch.manual_seed(3407)
    model, initialization_kind = construct_native_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.backbone.unfreeze()
    model.to(device).eval()
    preprocessor = SpectrumPreprocessor(
        dformat=DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    with np.load(args.corpus / "validation.npz", allow_pickle=False) as body:
        validation = {name: np.asarray(body[name]) for name in body.files}
    with h5py.File(args.data, "r") as handle:
        before = fold1_relation_diagnostics(
            model, preprocessor, handle, validation, device,
        )
        after = before
        if args.candidate_checkpoint is not None:
            package = torch_load_compat(args.candidate_checkpoint, map_location="cpu")
            if (
                not isinstance(package, dict)
                or package.get("format") != "official_embedding_slim_v1"
            ):
                raise RuntimeError("candidate checkpoint is not a slim continuation")
            model.backbone.load_state_dict(package["backbone_state_dict"], strict=True)
            model.head.load_state_dict(package["head_state_dict"], strict=True)
            del package
            after = fold1_relation_diagnostics(
                model, preprocessor, handle, validation, device,
            )
    report = {
        "status": "noise_relation_t1_t3_fold1_standalone_complete",
        "candidate_checkpoint": (
            str(args.candidate_checkpoint) if args.candidate_checkpoint else None
        ),
        "warm_start_checkpoint": str(args.warm_start_checkpoint),
        "initialization_kind": initialization_kind,
        "fold1_diagnostics": {"before": before, "after": after},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
