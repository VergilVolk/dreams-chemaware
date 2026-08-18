"""Self-contained DreaMS inference engine.

These are the exact, previously-validated inference routines (originally split
across ``tasks/e1_checkpoint_io.py`` and ``tasks/pilot_paired_layer_cka.py``),
consolidated here so the annotation package does not depend on the tasks/
scratch directory. Only the ``dreams`` package and torch/numpy/h5py are required.

The embedding definition matches the official DreaMS formula
(Bushuiev et al., Nat Biotechnol 2025, DOI 10.1038/s41587-025-02663-3):

    precursor = model(batch, None)[:, 0]
    embedding = L2_normalize( linear(precursor, weight, bias) )

i.e. the position-0 precursor token of the frozen backbone, projected through the
official linear head, then L2-normalized.
"""
from __future__ import annotations

import os
import pathlib
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------- #
# Checkpoint loading (weights-only-safe on Windows)
# --------------------------------------------------------------------------- #
def _install_weights_only_compatibility() -> None:
    """Allow the two harmless legacy classes stored in the official file.

    The official checkpoint pickled a Linux Path and a full DreaMS object inside
    hyperparameters. We only consume ``state_dict``; stand-ins avoid
    reconstructing that duplicate model and turn a multi-minute load into seconds.
    """
    if os.name == "nt":
        path_class = type("PosixPath", (pathlib.WindowsPath,), {"__module__": "pathlib"})
        pathlib.PosixPath = path_class
    else:
        path_class = pathlib.PosixPath
    dummy_dreams = type(
        "DreaMS", (object,), {"__module__": "msml.models.dreams.dreams"}
    )
    try:
        torch.serialization.add_safe_globals([path_class, dummy_dreams])
    except AttributeError:  # PyTorch 2.2 and older
        import torch._weights_only_unpickler as weights_unpickler

        allowed = weights_unpickler._get_allowed_globals()
        allowed["pathlib.PosixPath"] = path_class
        allowed["msml.models.dreams.dreams.DreaMS"] = dummy_dreams


def torch_load_compat(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load Linux-authored Lightning checkpoints safely on Windows (mmap=True)."""
    _install_weights_only_compatibility()
    path = str(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True, mmap=True)
    except Exception:
        try:
            return torch.load(path, map_location=map_location, weights_only=False, mmap=True)
        except TypeError:
            return torch.load(path, map_location=map_location, weights_only=False)


def checkpoint_kind(package: dict[str, Any]) -> str:
    if package.get("format") == "chemaware_multitask_head_v1":
        return "chemaware_multitask_head"
    if package.get("format") == "causal_chemmask_head_v1":
        return "causal_chemmask_head"
    if package.get("format") == "counterfactual_dreams_v1":
        return "counterfactual_dreams"
    if package.get("format") == "e1_identity_v1":
        return "e1_identity"
    if package.get("format") == "official_embedding_slim_v1":
        return "official_embedding_slim"
    state = package.get("state_dict")
    if isinstance(state, dict):
        keys = state.keys()
        if any(key.startswith("backbone.") for key in keys) and any(
            key.startswith("head.") for key in keys
        ):
            return "official_embedding"
        if "args" in package:
            return "raw_ssl"
    raise ValueError(f"Unsupported checkpoint keys: {list(package)[:12]}")


def official_backbone_state(package: dict[str, Any]) -> dict[str, torch.Tensor]:
    if package.get("format") in {
        "counterfactual_dreams_v1",
        "official_embedding_slim_v1",
        "e1_identity_v1",
    }:
        return package["backbone_state_dict"]
    return {
        key.removeprefix("backbone."): value
        for key, value in package["state_dict"].items()
        if key.startswith("backbone.")
    }


def official_head_state(package: dict[str, Any]) -> dict[str, torch.Tensor]:
    if package.get("format") in {
        "causal_chemmask_head_v1",
        "chemaware_multitask_head_v1",
        "counterfactual_dreams_v1",
        "official_embedding_slim_v1",
        "e1_identity_v1",
    }:
        return package["head_state_dict"]
    return {
        key.removeprefix("head."): value
        for key, value in package["state_dict"].items()
        if key.startswith("head.")
    }


# --------------------------------------------------------------------------- #
# Spectrum preprocessing (DreaMS / MassSpecGym convention)
# --------------------------------------------------------------------------- #
def preprocess_spectrum(
    raw_2_n: np.ndarray, precursor_mz: float, n_highest: int
) -> torch.Tensor:
    """Top-N peaks by intensity (stable), intensity normalized to [0,1], then a
    synthetic precursor peak [mz, 1.1] prepended. Returns (n_highest+1, 2)."""
    raw = np.asarray(raw_2_n)
    highest = np.argsort(raw[1], kind="stable")[-n_highest:]
    highest = np.sort(highest)
    peaks = raw[:, highest].T.astype(np.float32, copy=True)
    if len(peaks) < n_highest:
        peaks = np.pad(peaks, ((0, n_highest - len(peaks)), (0, 0)))
    maximum = float(peaks[:, 1].max())
    if maximum > 0:
        peaks[:, 1] /= maximum
    precursor = np.asarray([[precursor_mz, 1.1]], dtype=np.float32)
    return torch.from_numpy(np.vstack((precursor, peaks)))


class SpectrumRows(Dataset):
    """Row-sliced view over an hdf5 file written by dreams.utils.data.MSData."""

    def __init__(self, path: Path, rows: np.ndarray, n_highest_peaks: int):
        self.path = str(path)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.n_highest_peaks = n_highest_peaks
        self._handle = None

    def __len__(self) -> int:
        return len(self.rows)

    def _h5(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, item: int) -> torch.Tensor:
        handle = self._h5()
        row = int(self.rows[item])
        return preprocess_spectrum(
            handle["spectrum"][row],
            float(handle["precursor_mz"][row]),
            self.n_highest_peaks,
        )

    def __del__(self):
        if self._handle is not None:
            self._handle.close()


# --------------------------------------------------------------------------- #
# Lightweight backbone (inference-only)
# --------------------------------------------------------------------------- #
class PositionFeedForward(torch.nn.Module):
    """Inference-only copy of the DreaMS positional feed-forward module (same
    ``ff.*`` keys so checkpoint state loads exactly)."""

    def __init__(
        self, in_dim: int, out_dim: int, hidden_dim: int, depth: int,
        dropout: float, bias: bool,
    ):
        super().__init__()
        layers: list[torch.nn.Module] = []
        hidden = [hidden_dim] * depth
        for layer in range(depth):
            d_in = hidden[layer - 1] if layer else in_dim
            d_out = hidden[layer] if layer != depth - 1 else out_dim
            layers.append(torch.nn.Linear(d_in, d_out, bias=bias))
            if layer != depth - 1:
                layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.ReLU())
        self.ff = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class LightweightDreaMS(torch.nn.Module):
    """Exact inference path of the official checkpoint, without Lightning."""

    def __init__(self, args: Namespace):
        super().__init__()
        if args.vanilla_transformer:
            raise NotImplementedError("Audited checkpoints use the custom encoder")
        if not args.d_fourier or args.d_mz_token:
            raise NotImplementedError("Validated for the Fourier DreaMS model only")
        if args.charge_feature:
            raise NotImplementedError("Audited checkpoints do not use charge features")

        from dreams.models.layers.fourier_features import FourierFeatures
        from dreams.models.dreams.layers import TransformerEncoder

        self.n_layers = int(args.n_layers)
        self.d_model = int(args.d_fourier + args.d_peak + args.d_mz_token)
        self.max_mz = float(args.dformat.max_mz)
        self.graphormer_mz_diffs = bool(args.graphormer_mz_diffs)
        args.d_model = self.d_model
        args.d_graphormer_params = 0

        self.fourier_enc = FourierFeatures(
            strategy=args.fourier_strategy,
            num_freqs=args.fourier_num_freqs,
            x_min=(
                args.dformat.max_tbxic_stdev
                if not args.fourier_min_freq else args.fourier_min_freq
            ),
            x_max=args.dformat.max_mz,
            trainable=args.fourier_trainable,
        )
        self.ff_fourier = PositionFeedForward(
            in_dim=self.fourier_enc.num_features(),
            out_dim=args.d_fourier,
            hidden_dim=args.ff_fourier_d,
            depth=args.ff_fourier_depth,
            dropout=args.dropout,
            bias=not args.no_ffs_bias,
        )
        self.ff_peak = PositionFeedForward(
            in_dim=2,
            out_dim=args.d_peak,
            hidden_dim=args.d_peak,
            depth=args.ff_peak_depth,
            dropout=args.dropout,
            bias=not args.no_ffs_bias,
        )
        self.transformer_encoder = TransformerEncoder(args)

    def forward(self, spec: torch.Tensor, charge=None) -> torch.Tensor:
        padding_mask = spec[:, :, 0] == 0
        normalizer = torch.tensor(
            [self.max_mz, 1.0], device=spec.device, dtype=spec.dtype
        )
        peak_embeddings = self.ff_peak(spec / normalizer)
        fourier_features = self.ff_fourier(self.fourier_enc(spec[..., [0]]))
        tokens = torch.cat([peak_embeddings, fourier_features], dim=-1)
        graphormer_dists = None
        if self.graphormer_mz_diffs:
            graphormer_dists = (
                fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)
            )
        return self.transformer_encoder(tokens, padding_mask, graphormer_dists)


def reconstruct_backbone(
    architecture_package: dict,
    state_dict: dict[str, torch.Tensor],
    n_highest_peaks: int,
    device: torch.device,
):
    """Rebuild the frozen backbone from (raw ssl package, slim backbone state)."""
    started = time.time()
    model_args = Namespace(**architecture_package["args"])
    model_args.dformat = Namespace(
        max_mz=float(architecture_package["args"]["max_mz"]),
        max_tbxic_stdev=float(architecture_package["args"]["max_tbxic_stdev"]),
    )
    backbone = LightweightDreaMS(model_args)
    ignored_prefixes = ("ff_out.", "ro_out.")
    forward_state = {
        key: value for key, value in state_dict.items()
        if not key.startswith(ignored_prefixes)
    }
    incompatible = backbone.load_state_dict(forward_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    backbone.eval().to(device)
    return backbone
