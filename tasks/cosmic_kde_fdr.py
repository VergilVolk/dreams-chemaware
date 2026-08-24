"""COSMIC 式 KDE 得分建模 —— 用核密度估计替代诱饵 FDR（实验验证版）。

方法（映射自 Hoffmann et al. Nat Biotechnol 2022, DOI 10.1038/s41587-021-01045-9）：
  不生成诱饵，改用"真实候选得分分布"当 null，用 KDE 拟合，对每个 hit 得分估
  P-value/E-value。我们映射：query 的 top-1 cosine 得分 vs 自检索里"错误匹配"
  （top-1 是不同 InChIKey）的 cosine 分布。

流程：
  1. 库内自检索（留一法 + m/z 硬约束），得每个 query 的 top-1 cosine + correct 标签。
  2. null 分布 = 错误匹配 query 的 top-1 cosine（logit 变换后 KDE 拟合，处理 cosine 边界）。
  3. 对每个 query：P-value = P(null top-1 >= 该 query top-1)。
  4. BH 校正 -> q-value。
  5. 校准验证：各 q 阈值下，估计 FDR 是否 ≈ 实际 FDR（错误比例）。

关键检验点：正确匹配 vs 错误匹配的 top-1 cosine 分布是否可分（KDE 能否起作用）。
若两者几乎重合（同 m/z 异构体余弦不可分），KDE 也区分不了——诚实结论。

数据: library = data/models/mona_neg_dreams_emb/{embeddings.npy,manifest.csv}
用法 (conda dreams_env, CPU):
    python tasks/cosmic_kde_fdr.py --n-query 5000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation.params import DEFAULT  # noqa: E402

LIB_DIR = ROOT / "data/models/mona_neg_dreams_emb"
PPM = DEFAULT.ppm_tolerance
EPS = 1e-6


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    n = len(p)
    ranked = np.argsort(p)
    bh = p[ranked] * n / (np.arange(n) + 1)
    q_ranked = np.minimum.accumulate(bh[::-1])[::-1]
    q = np.empty(n)
    q[ranked] = q_ranked
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-query", type=int, default=5000)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    lib = np.load(LIB_DIR / "embeddings.npy")
    man = pd.read_csv(LIB_DIR / "manifest.csv")
    l_mz = man["precursor_mz"].to_numpy(dtype=np.float64)
    inchikey = man["inchikey"].tolist()
    n = lib.shape[0]
    libn = lib / np.linalg.norm(lib, axis=1, keepdims=True)

    idxs = list(range(args.start, min(args.start + args.n_query, n)))
    qn = libn[idxs]
    q_mz = l_mz[idxs]
    q_ik = [inchikey[i] for i in idxs]

    # 自检索 top-1（留一 + m/z）
    top1 = np.full(len(idxs), -np.inf)
    top1_ik = [""] * len(idxs)
    for k in range(len(idxs)):
        i = idxs[k]
        sim = qn[k] @ libn.T
        dppm = np.abs(q_mz[k] - l_mz) / np.maximum(np.abs(l_mz), 1e-9) * 1e6
        ok = (dppm <= PPM)
        ok[i] = False
        sm = np.where(ok, sim, -np.inf)
        j = int(np.argmax(sm))
        top1[k] = sm[j]
        top1_ik[k] = inchikey[j]

    top1 = np.array(top1)
    correct = np.array([top1_ik[k] == q_ik[k] for k in range(len(idxs))])

    # 排除无 m/z 匹配（top1=-inf）的 query —— 那不是"错误匹配"，是"无注释"
    finite = np.isfinite(top1)
    top1 = top1[finite]
    correct = correct[finite]

    # 分正确/错误两组
    cos_correct = top1[correct]
    cos_wrong = top1[~correct]
    print(f"=== 正确 vs 错误匹配 top-1 cosine 分布 (n={len(idxs)}) ===", flush=True)
    for name, arr in (("正确", cos_correct), ("错误", cos_wrong)):
        pct = np.percentile(arr, [0, 25, 50, 75, 90, 100]).round(4)
        print(f"  {name:4s} n={len(arr):5d}  percentiles[0,25,50,75,90,max] = {pct}",
              flush=True)

    # null = 错误匹配 top-1 cosine，logit 后 KDE
    if len(cos_wrong) < 10:
        print("错误匹配样本太少，无法拟合 KDE", flush=True)
        return
    kde = gaussian_kde(logit(cos_wrong))
    print(f"\n=== KDE null 分布 (错误匹配 top-1, logit 空间, n={len(cos_wrong)}) ===",
          flush=True)

    # 对每个 query 的 top-1 算 P-value（CDF 用逐元素 integrate_box_1d，兼容旧 scipy）
    ltop = logit(top1)
    pvals = np.array([1.0 - kde.integrate_box_1d(-np.inf, x) for x in ltop])
    qvals = bh_qvalues(pvals)

    print(f"\n=== KDE P-value / BH q-value 校准验证 ===", flush=True)
    print(f"  {'q<阈值':>8s} {'n_pass':>7s} {'n_wrong':>8s} {'实际FDR':>8s}", flush=True)
    for qth in (0.01, 0.05, 0.10, 0.20):
        mask = qvals < qth
        np_ = int(mask.sum())
        nw = int((~correct[mask]).sum())
        fdr = nw / max(np_, 1)
        print(f"  {qth:8.2f} {np_:7d} {nw:8d} {fdr:8.4f}", flush=True)

    # 对比：正确匹配的中位 P-value vs 错误匹配的中位 P-value（分离度）
    print(f"\n  正确匹配中位 P-value = {np.median(pvals[correct]):.4f}", flush=True)
    print(f"  错误匹配中位 P-value = {np.median(pvals[~correct]):.4f}", flush=True)
    print(f"  （两者差异越大，KDE 判别力越强）", flush=True)


if __name__ == "__main__":
    main()
