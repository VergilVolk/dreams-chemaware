"""验证 fdr.py 的 n_target_ge 方向 bug，并对比修复前后 q 值。

合成例子（手工可算）+ 真实数据（6915 query vs 36663 库/诱饵）双验证。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation.fdr import top1_scores  # noqa: E402


def qvalues_buggy(target_scores, decoy_scores):
    """fdr.py 原实现（n_target_ge 反了）。"""
    order = np.argsort(-target_scores, kind="stable")
    sorted_scores = target_scores[order]
    n_target_ge = np.arange(len(target_scores), 0, -1).astype(np.float64)
    sorted_decoy = np.sort(decoy_scores)
    n_decoy_ge = len(decoy_scores) - np.searchsorted(sorted_decoy, sorted_scores, side="left")
    fdr = (n_decoy_ge + 1.0) / (n_target_ge + 1.0)
    q = np.minimum.accumulate(fdr)
    out = np.empty_like(q)
    out[order] = q
    return out, fdr, order


def qvalues_fixed(target_scores, decoy_scores):
    """修复版：n_target_ge 随降序分数从 1 递增到 N。"""
    order = np.argsort(-target_scores, kind="stable")
    sorted_scores = target_scores[order]
    n_target_ge = np.arange(1, len(target_scores) + 1).astype(np.float64)
    sorted_decoy = np.sort(decoy_scores)
    n_decoy_ge = len(decoy_scores) - np.searchsorted(sorted_decoy, sorted_scores, side="left")
    fdr = (n_decoy_ge + 1.0) / (n_target_ge + 1.0)
    q = np.minimum.accumulate(fdr)
    out = np.empty_like(q)
    out[order] = q
    return out, fdr, order


def main() -> None:
    # ---- 合成例子 ----
    target = np.array([0.99, 0.95, 0.80, 0.66, 0.60])
    decoy = np.array([0.92, 0.70, 0.65, 0.60, 0.55])
    print("=== 合成例子 ===", flush=True)
    print("target:", target, " decoy:", decoy, flush=True)

    qb, fdrb, _ = qvalues_buggy(target, decoy)
    qf, fdrf, _ = qvalues_fixed(target, decoy)
    print(f"buggy  q = {qb}", flush=True)
    print(f"fixed  q = {qf}", flush=True)
    # 手工正确值：fdr(0.99)=1/2=0.5, fdr(0.95)=1/3, fdr(0.80)=2/4, fdr(0.66)=3/5, fdr(0.60)=5/6
    print("手工正确 fdr(s) 从高到低 = [0.5, 0.333, 0.5, 0.6, 0.833]", flush=True)
    print("手工正确 q(min累积)       = [0.5, 0.333, 0.333, 0.333, 0.333]", flush=True)

    # ---- 真实数据 ----
    query = np.load(ROOT / "data/mtbls13729/smoke/neg_emb/embeddings.npy")
    library = np.load(ROOT / "data/models/mona_neg_dreams_emb/embeddings.npy")
    decoy = np.load(ROOT / "data/models/mona_neg_decoy_emb.npy")
    target_scores = top1_scores(query, library)
    decoy_scores = top1_scores(query, decoy)

    print("\n=== 真实数据 (6915 query) ===", flush=True)
    qb, _, _ = qvalues_buggy(target_scores, decoy_scores)
    qf, _, _ = qvalues_fixed(target_scores, decoy_scores)
    print(f"buggy q  percentiles[0,25,50,75,90,max] = "
          f"{np.percentile(qb, [0,25,50,75,90,100]).round(6)}", flush=True)
    print(f"fixed q  percentiles[0,25,50,75,90,max] = "
          f"{np.percentile(qf, [0,25,50,75,90,100]).round(6)}", flush=True)
    print(f"buggy 唯一值 {len(np.unique(qb))} 个；fixed 唯一值 {len(np.unique(qf))} 个",
          flush=True)

    # 修复后，低分 query 的 q 应回到 fdr(0.66)≈0.84 量级，而不是塌到 0.037 以下
    low = target_scores < 0.70
    print(f"\ncos<0.70 低分 query 的 q 中位数: buggy={np.median(qb[low]):.6f}  "
          f"fixed={np.median(qf[low]):.6f}", flush=True)
    # 高分 query 的 q 应接近 fdr(top)=1/27≈0.037
    high = target_scores >= 0.95
    print(f"cos>=0.95 高分 query 的 q 中位数: buggy={np.median(qb[high]):.6f}  "
          f"fixed={np.median(qf[high]):.6f}", flush=True)


if __name__ == "__main__":
    main()
