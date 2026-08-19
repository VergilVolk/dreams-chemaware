"""共享评估指标：空间结构（坍缩探测器）+ 检索 + AUC。

三套指标对应三个验证目标（2026-08-19 定案）：
  1. 拉近：pos_cos = cos(anchor_clean, anchor_noise)   —— 噪声不变性，应 ↑
  2. 拉远：neg_cos = cos(anchor_clean, isomer)          —— 异构体区分，应 ↓
  3. 全局：space_structure_metrics + retrieval          —— 表示空间健康 + 检索能力，不应崩

space_structure_metrics 是坍缩探测器：
  - pairwise_cos_mean  → 高维空间里不同分子应近似正交(≈0)，坍缩则 →1
  - participation_ratio (1/Σp²)  —— SVD 有效秩，健康高，坍缩 →1
  - eff_rank            —— 谱熵有效秩，同义
  - mean_dimension_std  —— 某些维方差→0 = 维坍缩
"""
from __future__ import annotations

import numpy as np


def space_structure_metrics(E: np.ndarray) -> dict:
    """E: (N, d) L2 归一化嵌入（应已去重到每分子一条）。返回空间结构指标。"""
    E = np.asarray(E, dtype=np.float32)
    N, d = E.shape
    if N < 2:
        return {
            "pairwise_cos_mean": float("nan"), "pairwise_cos_std": float("nan"),
            "participation_ratio": float("nan"), "eff_rank": float("nan"),
            "mean_dimension_std": float("nan"), "n_points": int(N),
        }
    S = E @ E.T
    off = S[~np.eye(N, dtype=bool)]
    pairwise_cos_mean = float(off.mean())
    pairwise_cos_std = float(off.std())

    # 不中心化：中心化会减掉"所有点挤向同一方向"的主导方向（恰是坍缩信号），
    # 导致点坍缩被误判为健康（Part A 自检抓到过此 bug）。未中心化时：
    #   健康随机方向 → 高有效秩；点坍缩 → 秩≈1。
    _, s, _ = np.linalg.svd(E, full_matrices=False)
    s2 = (s.astype(np.float64) ** 2)
    s2 = s2[s2 > 1e-14]
    p = s2 / s2.sum()
    participation_ratio = float(1.0 / (p * p).sum())
    eff_rank = float(np.exp(-(p * np.log(p)).sum())) if len(p) else float("nan")

    mean_dimension_std = float(E.std(axis=0).mean())

    return {
        "pairwise_cos_mean": pairwise_cos_mean,
        "pairwise_cos_std": pairwise_cos_std,
        "participation_ratio": participation_ratio,
        "eff_rank": eff_rank,
        "mean_dimension_std": mean_dimension_std,
        "n_points": int(N),
    }


def query_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float((np.count_nonzero(diff > 0) + 0.5 * np.count_nonzero(diff == 0)) / diff.size)


def retrieval_metrics(emb: np.ndarray, iks, pmzs, adducts, ppm_tol: float) -> dict:
    """严格 10ppm 同 adduct 检索（分子聚合），同旧 pilot 协议。"""
    pmzs = np.asarray(pmzs, dtype=float)
    iks = np.asarray(iks)
    adducts = np.asarray(adducts)
    aucs, recalls1, mrrs = [], [], []
    for qi in range(len(iks)):
        ppm_da = ppm_tol * 1e-6 * pmzs[qi]
        cand = (np.abs(pmzs - pmzs[qi]) <= ppm_da) & (np.arange(len(iks)) != qi) & (adducts == adducts[qi])
        idx = np.where(cand)[0]
        if len(idx) == 0:
            continue
        labels = (iks[idx] == iks[qi]).astype(int)
        if labels.sum() == 0 or (labels == 0).sum() == 0:
            continue
        scores = (emb[qi:qi + 1] * emb[idx]).sum(axis=1)
        aucs.append(query_auc(labels, scores))
        best = {}
        for j, sc in zip(idx, scores):
            ik = iks[j]
            if ik not in best or sc > best[ik]:
                best[ik] = float(sc)
        order = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        ranks = [ik for ik, _ in order]
        recalls1.append(1.0 if (iks[qi] in ranks and ranks.index(iks[qi]) == 0) else 0.0)
        mrrs.append(1.0 / (ranks.index(iks[qi]) + 1) if iks[qi] in ranks else 0.0)
    return {
        "n_queries": len(aucs),
        "macro_auc": float(np.mean(aucs)) if aucs else 0.5,
        "recall1": float(np.mean(recalls1)) if recalls1 else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
    }
