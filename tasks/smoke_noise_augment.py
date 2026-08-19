"""
Step 2 smoke：验证四轴噪声增强器在真实 MassSpecGym 谱上的输出 sanity。

检查：
  1. 输出形状 (2, m)，m/z 升序，无 NaN，intensity ≥ 0。
  2. 删峰后有效峰数不增、通常减少。
  3. m/z 位移轴精确校验：只开 do_shift 时，Δm/z ≈ N(0, σ=0.01)（std≈0.01，mean≈0）。
  4. 全关开关 = 有效峰数不变。

用法（本机 conda，CPU）：
  python tasks/smoke_noise_augment.py
  python tasks/smoke_noise_augment.py --n 200 --seed 0
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

from noise_augment import NoiseConfig, apply_noise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HDF5_PATH = os.path.join(ROOT, "data/models/MassSpecGym_MurckoHist_split.hdf5")


def decode(x):
    return x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)


def valid_peaks(spec):
    return int((spec[0] > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    with h5py.File(HDF5_PATH, "r") as f:
        folds = np.array([decode(x) for x in f["fold"][:]])
        train_idx = np.where(folds == "train")[0]
        pick = train_idx[: args.n]
        specs = [np.asarray(f["spectrum"][int(i)]) for i in pick]

    # ---- 1+2) 全开：输出合法性 + 峰数变化 ----
    cfg = NoiseConfig()
    bad = 0
    before, after = [], []
    for raw in specs:
        out = apply_noise(raw, rng, cfg)
        ok = (out.shape[0] == 2 and not np.isnan(out).any()
              and (out[1] >= 0).all()
              and np.all(np.diff(out[0]) >= -1e-12))
        if not ok:
            bad += 1
            continue
        before.append(valid_peaks(raw))
        after.append(valid_peaks(out))

    print(f"=== Step 2 噪声增强器 smoke（{len(specs)} 谱，seed={args.seed}）===")
    print(f"  输出非法样本: {bad}（应为 0）")
    print(f"  有效峰数 before: mean={np.mean(before):.1f}  after: mean={np.mean(after):.1f} "
          f"(Δ={np.mean(after) - np.mean(before):+.1f}，删峰≤30%+加峰≤10 的净变化，正常在 [-30%,+10] 区间)")

    # ---- 3) m/z 位移轴精确校验（只开 do_shift）----
    cfg_shift = NoiseConfig(do_delete=False, do_jitter=False, do_add=False, do_shift=True)
    rng2 = np.random.default_rng(args.seed)
    shifts = []
    for raw in specs[:100]:
        out = apply_noise(raw, rng2, cfg_shift)
        in_mz = np.sort(raw[0][raw[0] > 0])
        out_mz = np.sort(out[0][out[0] > 0])
        if len(in_mz) == len(out_mz):
            shifts.append(out_mz - in_mz)
    if shifts:
        shifts = np.concatenate(shifts)
        print(f"  m/z 位移轴（仅 shift）：Δm/z mean={shifts.mean():+.5f} std={shifts.std():.5f} "
              f"（期望 mean≈0, std≈0.01）")

    # ---- 4) 全关 = 峰数不变 ----
    cfg_off = NoiseConfig(do_delete=False, do_jitter=False, do_add=False, do_shift=False)
    raw0 = specs[0]
    out_off = apply_noise(raw0, np.random.default_rng(0), cfg_off)
    print(f"  全关开关：有效峰数不变 = {valid_peaks(raw0) == valid_peaks(out_off)}（应为 True）")

    print("\n=== SMOKE DONE ===")


if __name__ == "__main__":
    main()
