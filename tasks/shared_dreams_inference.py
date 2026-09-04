"""Strict loading helpers for official or fine-tuned shared DreaMS encoders."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from e1_checkpoint_io import torch_load_compat  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402
from dreams.models.chem_aware.shared_embedding_v2 import (  # noqa: E402
    ChemAwareEmbeddingInference, ChemAwareSharedEncoder, SignedPeakResidualAdapter,
)
from dreams.models.chem_aware.peft_v3 import (  # noqa: E402
    DreaMSPEFTConfig, install_dreams_peft, load_peft_state_dict,
)


ALLOWED_SHARED_STATUSES = {
    "noise_final_e4a_direct_shared_dreams_encoder",
}
CHEMAWARE_SHARED_STATUSES = {
    "chemaware_shared_v2_clean_listwise",
    "chemaware_shared_v2_molecule_teacher",
}
CHEMAWARE_PEFT_STATUSES = {
    "chemaware_shared_v3_clean_peft",
    "chemaware_shared_v3_molecule_teacher_peft",
}


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def load_inference_model(
    official_checkpoint: Path,
    architecture_checkpoint: Path,
    device: torch.device,
    n_highest_peaks: int,
    shared_checkpoint: Path | None = None,
):
    """Load one shared query/reference encoder and fail closed on bad metadata."""
    model, initialization = load_base_model(
        official_checkpoint, architecture_checkpoint, device, n_highest_peaks
    )
    metadata = {
        "kind": "official_dreams",
        "checkpoint": str(official_checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(official_checkpoint),
        "initialization": initialization,
    }
    if shared_checkpoint is not None:
        if not shared_checkpoint.is_file():
            raise FileNotFoundError(shared_checkpoint)
        package = torch_load_compat(shared_checkpoint, map_location="cpu")
        status = package.get("status")
        if status not in ALLOWED_SHARED_STATUSES | CHEMAWARE_SHARED_STATUSES | CHEMAWARE_PEFT_STATUSES:
            raise RuntimeError(
                f"unsupported shared-encoder checkpoint status: {status!r}"
            )
        if package.get("P2b_used") is not False:
            raise RuntimeError("shared-encoder checkpoint violates the P2b-free contract")
        if status in CHEMAWARE_PEFT_STATUSES:
            if package.get("format") != "chemaware_shared_v3_peft_v1":
                raise RuntimeError("ChemAware PEFT checkpoint has an unsupported format")
            if package.get("query_reference_encoder_shared") is not True:
                raise RuntimeError("ChemAware PEFT checkpoint is not a shared query/reference encoder")
            if package.get("candidate_inputs_at_inference") is not False:
                raise RuntimeError("ChemAware PEFT checkpoint requires forbidden candidate inputs")
            expected_chemical = status == "chemaware_shared_v3_molecule_teacher_peft"
            if package.get("chemical_supervision") is not expected_chemical:
                raise RuntimeError("ChemAware PEFT chemical-supervision status is inconsistent")
            if package.get("training_only_projector_used") is not False:
                raise RuntimeError("ChemAware PEFT checkpoint used a discardable projector")
            frozen_probe_used = package.get("training_only_frozen_probe_used", False)
            if frozen_probe_used not in {True, False}:
                raise RuntimeError("ChemAware PEFT frozen-probe metadata is malformed")
            if frozen_probe_used:
                if not expected_chemical:
                    raise RuntimeError("a frozen chemical probe requires chemical supervision")
                if package.get("chemical_gradient_absorber_trainable") is not False:
                    raise RuntimeError("ChemAware frozen probe was not demonstrably non-trainable")
                forbidden_probe_keys = {
                    "frozen_probe_state", "chemical_probe_state", "molecule_teacher_state"
                }
                if forbidden_probe_keys & set(package):
                    raise RuntimeError("ChemAware inference checkpoint carries training-only chemistry")
            config = package.get("peft_config")
            state = package.get("peft_state")
            if not isinstance(config, dict) or not isinstance(state, dict):
                raise RuntimeError("ChemAware PEFT checkpoint lacks state/config")
            provenance = package.get("provenance", {})
            if provenance.get("official_checkpoint_sha256") != sha256_file(official_checkpoint):
                raise RuntimeError("ChemAware PEFT checkpoint belongs to different official weights")
            if provenance.get("raw_checkpoint_sha256") != sha256_file(architecture_checkpoint):
                raise RuntimeError("ChemAware PEFT checkpoint belongs to a different DreaMS architecture")
            peft_config = DreaMSPEFTConfig(**config)
            install_dreams_peft(model, peft_config)
            load_peft_state_dict(model, state)
            metadata.update({
                "kind": "experimental_chemaware_shared_peft_embedding",
                "checkpoint": str(shared_checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(shared_checkpoint),
                "training_seed": int(package.get("seed", -1)),
                "training_outer_fold": int(package.get("outer_fold", -1)),
                "chemical_supervision": expected_chemical,
                "teacher_control": package.get("teacher_control"),
                "training_only_frozen_probe_loaded": False,
                "formal_training": bool(package.get("formal")),
                "peft_config": config,
            })
        elif status in CHEMAWARE_SHARED_STATUSES:
            if package.get("query_reference_encoder_shared") is not True:
                raise RuntimeError("ChemAware checkpoint is not a shared query/reference encoder")
            if package.get("candidate_inputs_at_inference") is not False:
                raise RuntimeError("ChemAware checkpoint requires forbidden candidate inputs")
            if package.get("training_only_projector_used") is not False:
                raise RuntimeError("ChemAware checkpoint used a discardable training-only projector")
            if package.get("molecule_projector_state") is not None:
                raise RuntimeError("ChemAware checkpoint unexpectedly contains a molecule projector")
            config = package.get("adapter_config", {})
            required_config = {"embedding_dim", "hidden_dim", "delta_bound"}
            if required_config - set(config) or "adapter_state" not in package:
                raise RuntimeError("ChemAware checkpoint lacks adapter state/config")
            if int(config["embedding_dim"]) != int(model.head.out_features):
                raise RuntimeError("ChemAware adapter dimension disagrees with official head")
            provenance = package.get("provenance", {})
            official_sha256 = sha256_file(official_checkpoint)
            if provenance.get("official_checkpoint_sha256") != official_sha256:
                raise RuntimeError("ChemAware checkpoint belongs to different official weights")
            adapter = SignedPeakResidualAdapter(
                int(config["embedding_dim"]), int(config["hidden_dim"]),
                float(config["delta_bound"]),
                float(config.get("gate_temperature", 1.0)),
                int(config.get("gate_topk", 0)),
                bool(config.get("contextual_gate", False)),
                bool(config.get("global_branch", False)),
            ).to(device)
            adapter.load_state_dict(package["adapter_state"], strict=True)
            model = ChemAwareEmbeddingInference(
                ChemAwareSharedEncoder(model, adapter)
            ).to(device)
            metadata.update({
                "kind": "experimental_chemaware_shared_embedding",
                "checkpoint": str(shared_checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(shared_checkpoint),
                "training_seed": int(package.get("seed", -1)),
                "training_outer_fold": int(package.get("outer_fold", -1)),
                "chemical_supervision": bool(package.get("chemical_supervision")),
                "teacher_control": package.get("teacher_control"),
                "formal_training": bool(package.get("formal")),
                "training_only_molecule_projector_loaded": False,
            })
        else:
            if package.get("inference_clean_only") is not True:
                raise RuntimeError("shared-encoder checkpoint is not clean-spectrum inference")
            if "model_state" not in package:
                raise RuntimeError("shared-encoder checkpoint has no model_state")
            model.load_state_dict(package["model_state"], strict=True)
            metadata.update({
                "kind": "experimental_noise_shared_embedding",
                "checkpoint": str(shared_checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(shared_checkpoint),
                "training_seed": int(package.get("seed", -1)),
                "training_outer_fold": int(package.get("outer_fold", -1)),
                "training_policy": str(package.get("policy", "")),
                "training_action_scope": str(package.get("action_scope", "")),
            })
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metadata
