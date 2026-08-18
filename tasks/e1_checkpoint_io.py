"""Checkpoint compatibility helpers shared by E0/E1 scripts."""
from __future__ import annotations

import pathlib
import os
from pathlib import Path
from typing import Any

import torch


def _install_weights_only_compatibility() -> None:
    """Allow the two harmless legacy classes stored in the official file.

    The official checkpoint unnecessarily pickled a Linux Path and a full
    DreaMS object inside hyperparameters.  We only consume ``state_dict``;
    stand-ins avoid importing/reconstructing that duplicate model and reduce a
    many-minute Windows load to seconds.
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


def torch_load_compat(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load Linux-authored Lightning checkpoints safely on Windows.

    ``mmap=True`` is important for the 1.2 GB official embedding checkpoint:
    optimizer states stay on disk unless accessed, which substantially lowers
    startup memory and I/O.
    """
    _install_weights_only_compatibility()
    path = str(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True, mmap=True)
    except Exception:
        # Raw server checkpoints contain argparse Namespace objects that older
        # weights-only loaders reject. These project-owned files are trusted.
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
    if package.get("format") in {"counterfactual_dreams_v1", "official_embedding_slim_v1", "e1_identity_v1"}:
        return package["backbone_state_dict"]
    return {
        key.removeprefix("backbone."): value
        for key, value in package["state_dict"].items()
        if key.startswith("backbone.")
    }


def official_head_state(package: dict[str, Any]) -> dict[str, torch.Tensor]:
    if package.get("format") in {"causal_chemmask_head_v1", "chemaware_multitask_head_v1"}:
        return package["head_state_dict"]
    if package.get("format") in {"counterfactual_dreams_v1", "official_embedding_slim_v1", "e1_identity_v1"}:
        return package["head_state_dict"]
    return {
        key.removeprefix("head."): value
        for key, value in package["state_dict"].items()
        if key.startswith("head.")
    }
