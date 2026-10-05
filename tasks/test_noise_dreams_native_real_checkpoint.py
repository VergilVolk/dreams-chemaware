"""Allocated-GPU smoke test of the real official native DreaMS training path."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from noise_dreams_native_spectrum import (
    make_action_spectrum,
    native_action_model_input,
)
from train_noise_dreams_native import load_official_continuation, make_hdf5_spectrum


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-finetuned-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    paths = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("real native DreaMS smoke requires an allocated GPU")
    args = argparse.Namespace(
        official_finetuned_checkpoint=paths.official_finetuned_checkpoint,
        architecture_checkpoint=paths.architecture_checkpoint,
        lr=5e-6,
        weight_decay=0.0,
        triplet_loss_margin=0.1,
    )
    model = load_official_continuation(args).cuda().train()
    model.backbone.unfreeze()
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("epoch-zero native smoke did not unfreeze all parameters")
    preprocessor = SpectrumPreprocessor(
        DataFormatA(),
        prec_intens=1.1,
        n_highest_peaks=100,
        spec_entropy_cleaning=False,
        precision=32,
        mz_shift_aug_p=0,
        mz_shift_aug_max=0,
    )
    spectra = []
    with h5py.File(paths.data, "r") as handle:
        for row in range(16):
            spectrum = make_hdf5_spectrum(
                handle["spectrum"][row], float(handle["precursor_mz"][row])
            )
            spectra.append(preprocessor(
                spectrum.get_peak_list(),
                prec_mz=spectrum.get_precursor_mz(),
                high_form=False,
            ))
    candidates = torch.from_numpy(np.stack(spectra).astype(np.float32)).cuda()
    # The unmodified native preprocessor cannot normalize an all-zero action.
    # It must be routed before dataset construction, never patched or sent as NaN.
    zero_action = np.zeros((101, 2), dtype=np.float32)
    zero_action[0] = spectra[0][0]
    real_mz = spectra[0][1:, 0]
    real_mz = real_mz[real_mz > 0][:16]
    zero_action[1:1 + len(real_mz), 0] = real_mz
    try:
        make_action_spectrum(zero_action)
    except RuntimeError as error:
        if "not representable" not in str(error):
            raise
    else:
        raise RuntimeError("all-zero action bypassed native representability routing")
    partial_zero_action = spectra[0].copy()
    partial_zero_action[2, 1] = 0.0
    partial_replay = native_action_model_input(partial_zero_action, preprocessor)
    if not np.array_equal(partial_replay, partial_zero_action):
        raise RuntimeError("representable partial-zero action did not replay exactly")
    model.eval()
    model.train()
    with torch.no_grad():
        embeddings = F.normalize(model(candidates).float(), dim=1)
        score = embeddings @ embeddings.T
        best = None
        for action_index in range(1, 16):
            for positive_index in range(1, 16):
                if positive_index == action_index:
                    continue
                for negative_index in range(1, 16):
                    if negative_index in {action_index, positive_index}:
                        continue
                    clean_boundary = 0.1 - score[0, positive_index] + score[0, negative_index]
                    boundary = (
                        0.1 - score[action_index, positive_index]
                        + score[action_index, negative_index]
                    )
                    value = float(torch.minimum(clean_boundary, boundary))
                    if best is None or value > best[0]:
                        best = (value, action_index, positive_index, negative_index)
        assert best is not None
        _, action_index, positive_index, negative_index = best
    clean = candidates[0:1].detach().clone().requires_grad_(True)
    action = candidates[action_index:action_index + 1].detach().clone().requires_grad_(True)
    positive = candidates[positive_index:positive_index + 1].detach().clone().requires_grad_(True)
    negative = candidates[negative_index:negative_index + 1].detach().clone().requires_grad_(True)
    optimizer = model.configure_optimizers()
    optimizer.zero_grad(set_to_none=True)
    _, clean_boundary_loss = model.step({
        "spec": clean,
        "pos_specs": positive[:, None],
        "neg_specs": negative[:, None],
    }, 0)
    if not torch.isfinite(clean_boundary_loss) or float(clean_boundary_loss.detach()) <= 0:
        raise RuntimeError(
            f"real native clean-boundary triplet has inactive loss: {clean_boundary_loss}"
        )
    clean_boundary_loss.backward()
    for role, tensor in (("clean", clean), ("positive", positive), ("negative", negative)):
        if tensor.grad is None or not torch.isfinite(tensor.grad).all() or not torch.count_nonzero(tensor.grad):
            raise RuntimeError(f"real native clean-boundary loss did not reach {role}")
    if action.grad is not None:
        raise RuntimeError("real native clean-boundary loss unexpectedly reached action")
    head_parameter = next(model.head.parameters())
    backbone_parameter = next(
        parameter for parameter in model.backbone.parameters()
        if parameter.grad is not None and torch.count_nonzero(parameter.grad)
    )
    if head_parameter.grad is None or not torch.count_nonzero(head_parameter.grad):
        raise RuntimeError("real native triplet loss did not reach projection head")
    before_head = head_parameter.detach().clone()
    before_backbone = backbone_parameter.detach().clone()
    optimizer.step()
    if torch.equal(before_head, head_parameter.detach()):
        raise RuntimeError("native Adam clean-boundary step did not update projection head")
    if torch.equal(before_backbone, backbone_parameter.detach()):
        raise RuntimeError("native Adam clean-boundary step did not update backbone")

    # Check the action-anchor triplet in a second ordinary native step. The
    # role audit deliberately keeps clean and action losses independent; the
    # production ledger likewise never sums a custom four-input objective.
    optimizer.zero_grad(set_to_none=True)
    for tensor in (clean, action, positive, negative):
        tensor.grad = None
    _, boundary_loss = model.step({
        "spec": action,
        "pos_specs": positive[:, None],
        "neg_specs": negative[:, None],
    }, 1)
    if not torch.isfinite(boundary_loss) or float(boundary_loss.detach()) <= 0:
        raise RuntimeError(
            f"real native boundary triplet has inactive loss: {boundary_loss}"
        )
    boundary_loss.backward()
    for role, tensor in (("action", action), ("positive", positive), ("negative", negative)):
        if tensor.grad is None or not torch.isfinite(tensor.grad).all() or not torch.count_nonzero(tensor.grad):
            raise RuntimeError(f"real native boundary loss did not reach {role}")
    boundary_backbone = next(
        parameter for parameter in model.backbone.parameters()
        if parameter.grad is not None and torch.count_nonzero(parameter.grad)
    )
    before_head_boundary = head_parameter.detach().clone()
    before_backbone_boundary = boundary_backbone.detach().clone()
    optimizer.step()
    if torch.equal(before_head_boundary, head_parameter.detach()):
        raise RuntimeError("native Adam boundary step did not update projection head")
    if torch.equal(before_backbone_boundary, boundary_backbone.detach()):
        raise RuntimeError("native Adam boundary step did not update backbone")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("real native optimizer path left FP32")
    print(
        "[test_noise_dreams_native_real_checkpoint] PASS "
        f"clean_boundary={float(clean_boundary_loss.detach()):.6f} "
        f"boundary={float(boundary_loss.detach()):.6f} "
        "all_zero=routed_clean_only partial_zero=bitwise_exact sequential_native_steps=2 "
        "roles=4 head=updated backbone=updated",
        flush=True,
    )


if __name__ == "__main__":
    main()
