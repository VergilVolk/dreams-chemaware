"""
Step 2: 四轴合成噪声增强器（作用于原始谱 m/z + intensity，走 spec_preproc 之前）。

四轴（已定案 2026-08-18，见 docs/NOISE_TASK_PLAN_20260818.md §5.1）：
  1. 删峰     随机删 p% 峰（含强峰，非只弱峰）   p ~ U[0, 0.3]
  2. 抖动     强度 ×U[1-α, 1+α]                 α = 0.4
  3. 加峰     加 0-10 个弱峰                    强度 U[0, 0.05]，m/z 在谱自身范围内随机
  4. m/z 位移 δ ~ N(0, 0.01 Da)                质量校准 / 仪器漂移（关键，旧 pilot 缺失）

与旧 pilot 的 apply_ms2deepscore_noise 区别：
  - 旧：只删弱峰(intensity<0.4) 0-20%；新：随机删含强峰 0-30%。
  - 旧：加峰强度 U[0,0.01]；新：U[0,0.05]。
  - 新增：m/z 位移轴。

输入 raw_2_n: (2, n) float，raw[0]=m/z、raw[1]=intensity（MassSpecGym spectrum 字段）。
输出: (2, m) float，m/z 升序。

噪声施加在原始谱上、再走 spec_preproc，让 trim→top-n→归一化与噪声交互，
真实模拟"峰进入/掉出 top-n"，而不是在 token 上做表面扰动。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseConfig:
    """四轴噪声参数 + 开关（开关供消融与预留接口用）。"""
    delete_frac: tuple[float, float] = (0.0, 0.3)   # 删峰比例 p ~ U[a,b]（含强峰）
    jitter_alpha: float = 0.4                        # 抖动 ±40%（×U[1-α, 1+α]）
    add_n: tuple[int, int] = (0, 10)                 # 加峰个数（闭区间，均匀整数）
    add_intens: tuple[float, float] = (0.0, 0.05)    # 加峰强度（max 归一化尺度上的弱峰）
    mz_shift_sigma: float = 0.01                     # m/z 位移 σ（Da）

    do_delete: bool = True
    do_jitter: bool = True
    do_add: bool = True
    do_shift: bool = True


def apply_noise(raw_2_n: np.ndarray, rng: np.random.Generator,
                cfg: NoiseConfig = NoiseConfig()) -> np.ndarray:
    """对原始 (2, n) 谱施加四轴合成噪声，返回 (2, m) m/z 升序谱。

    rng 由调用方传入并控制种子，保证可复现；正例每次调用独立抽噪声。
    """
    raw = np.asarray(raw_2_n, dtype=np.float64)
    mz = raw[0].copy()
    intens = raw[1].copy()
    valid = mz > 0
    if not valid.any():
        return raw

    # 先 max 归一化：让"加峰强度 U[0,0.05]"定义在统一尺度（= 基峰的相对强度）。
    # （preprocess_spectrum 之后还会再归一化，这里只为加峰尺度一致。）
    peak_max = intens[valid].max()
    if peak_max > 0:
        intens[valid] = intens[valid] / peak_max

    n_valid = int(valid.sum())

    # 1) 删峰：随机删 p% 峰（含强峰），至少保留 1 个有效峰
    if cfg.do_delete and n_valid > 1:
        frac = float(rng.uniform(*cfg.delete_frac))
        n_rm = min(int(round(frac * n_valid)), n_valid - 1)
        if n_rm > 0:
            rm = rng.choice(np.where(valid)[0], size=n_rm, replace=False)
            keep = np.ones(len(mz), dtype=bool)
            keep[rm] = False
            mz, intens, valid = mz[keep], intens[keep], valid[keep]

    # 2) 强度抖动 ±40%（仅有效峰，padding 强度保持 0）
    if cfg.do_jitter:
        j = rng.uniform(1.0 - cfg.jitter_alpha, 1.0 + cfg.jitter_alpha, size=int(valid.sum()))
        intens[valid] = intens[valid] * j

    # 3) 加峰：0-10 个弱峰，m/z 在谱自身范围内随机
    if cfg.do_add and valid.any():
        n_add = int(rng.integers(cfg.add_n[0], cfg.add_n[1] + 1))
        if n_add > 0:
            lo, hi = float(mz[valid].min()), float(mz[valid].max())
            if hi > lo:
                new_mz = rng.uniform(lo, hi, size=n_add)
                new_intens = rng.uniform(*cfg.add_intens, size=n_add)
                mz = np.concatenate([mz, new_mz])
                intens = np.concatenate([intens, new_intens])
                valid = np.concatenate([valid, np.ones(n_add, dtype=bool)])

    # 4) m/z 位移：仅施加于有效峰（含新增峰），padding(m/z=0) 不动。
    #    ——否则 padding 的 0 会被 δ 变成 ±0.01 的小正数，冒充假峰（smoke 已抓到此 bug）。
    if cfg.do_shift and valid.any():
        mz[valid] = mz[valid] + rng.normal(0.0, cfg.mz_shift_sigma, size=int(valid.sum()))

    order = np.argsort(mz, kind="stable")
    return np.stack([mz[order], intens[order]], axis=0)
