"""Smoke test for the #1+#2 augmentation fixes in train_causal_chemmask_head.

Verifies, on synthetic preprocessed tensors:
  - mask_noise: pretraining-style intensity-proportional masking sets BOTH m/z and
    intensity to MASK_VAL (-1), protects the precursor (row 0), the base peak
    (intensity 1.0), and padding (intensity 0.0).
  - mask_unique_peaks: masks the highest-intensity peaks with no m/z match in the
    reference to MASK_VAL, leaves matched peaks at their ORIGINAL normalized
    intensity (no re-normalization confound).

Usage: D:\\dreams_env\\python.exe tasks\\smoke_causal_chemmask_augment.py
"""

from __future__ import annotations

import numpy as np
import torch

from train_causal_chemmask_head import MASK_VAL, mask_noise, mask_unique_peaks


def make_tensor(mzs, ints):
    peaks = np.stack([mzs, ints], axis=1).astype(np.float32)
    precursor = np.array([[200.0, 1.1]], dtype=np.float32)
    return torch.from_numpy(np.vstack((precursor, peaks)))


def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {detail}")
    return 0 if cond else 1


def main():
    rng = np.random.RandomState(0)
    fails = 0

    # --- mask_noise ---
    t = make_tensor([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
                    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.0])
    out = mask_noise(t, 0.3, rng).numpy()
    fails += check("noise: precursor intact", np.allclose(out[0], [200.0, 1.1]))
    fails += check("noise: base peak protected", np.allclose(out[1], [100.0, 1.0]))
    fails += check("noise: padding untouched", np.allclose(out[8], [170.0, 0.0]))
    masked_rows = np.flatnonzero(out[1:, 0] == MASK_VAL)
    fails += check("noise: some peaks masked", len(masked_rows) > 0, f"({len(masked_rows)} masked)")
    for r in masked_rows:
        both = out[1 + r, 0] == MASK_VAL and out[1 + r, 1] == MASK_VAL
        fails += check(f"noise: masked row {r} has -1 on both dims", bool(both))

    # --- mask_unique_peaks ---
    # source rows: [100,1.0],[110,0.9],[120,0.8],[130,0.7],[140,0.6]
    # reference matches 100,120,130 -> unique = 110 (0.9) and 140 (0.6)
    src = make_tensor([100.0, 110.0, 120.0, 130.0, 140.0], [1.0, 0.9, 0.8, 0.7, 0.6])
    ref = make_tensor([100.0, 120.0, 130.0], [1.0, 0.9, 0.8])
    out, count = mask_unique_peaks(src, ref, 0.02, 0.3, 12)
    out_np = out.numpy()
    src_np = src.numpy()
    fails += check("unique: masked exactly the 2 unique peaks", count == 2, f"(got {count})")
    fails += check("unique: matched 100 intact", np.allclose(out_np[1], [100.0, 1.0]))
    fails += check("unique: matched 120 intact", np.allclose(out_np[3], [120.0, 0.8]))
    fails += check("unique: 110 masked both dims", np.allclose(out_np[2], [MASK_VAL, MASK_VAL]))
    fails += check("unique: 140 masked both dims", np.allclose(out_np[5], [MASK_VAL, MASK_VAL]))
    # No re-normalization: matched peaks are bitwise identical to the source
    # (a re-normalizing implementation would rescale them up).
    fails += check(
        "unique: no renormalization (matched rows bitwise unchanged)",
        out_np[3, 1] == src_np[3, 1] and out_np[4, 1] == src_np[4, 1],
    )

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
