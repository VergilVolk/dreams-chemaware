"""Small real-DreaMS numerical test for the dependency-minimal prefix cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from chemaware_frozen_prefix_cache import FrozenPrefixSpectrumStore
from train_e1_identity import load_base_model
from train_noise_final_e4a_direct_augmentation import unfreeze_last_blocks
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    device = torch.device("cpu")
    rows = np.asarray([0, 1, 2], dtype=np.int64)
    store = SpectrumStore(
        ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5", rows, 100,
    )
    model, initialization = load_base_model(
        ROOT / "data/e1/official_embedding_slim.pt",
        ROOT / "dreams/models/pretrained/ssl_model_server.pt",
        device,
        100,
    )
    unfreeze_last_blocks(model, 1)
    expected = encode_rows(model, store, rows, device, 3, False, "real_cache_expected")
    cache = FrozenPrefixSpectrumStore(model, store, device, batch_size=3, last_blocks=1)
    with torch.no_grad():
        observed = cache.forward(model, rows, device, batch_size=3, amp=False).cpu().numpy()
    cosine = np.einsum("ij,ij->i", expected, observed)
    if not np.all(np.isfinite(cosine)) or float(np.min(cosine)) < 0.99999:
        raise AssertionError(f"real DreaMS prefix-cache replay drifted: {cosine.tolist()}")
    print(
        f"PASS: real DreaMS prefix cache matches full forward; "
        f"initialization={initialization}; min_cosine={float(np.min(cosine)):.8f}"
    )


if __name__ == "__main__":
    main()
