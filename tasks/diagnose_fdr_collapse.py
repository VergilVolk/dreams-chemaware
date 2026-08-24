"""诊断 shuffle-decoy TDA q-value 塌缩 —— 用全库拿精确数字，坐实机制。

背景：`annotation/fdr.py` 的 target-decoy q 值塌缩（q 全≈1/(N_decoy+1)），
"top1 全过"。本脚本不做任何修补，只把 target/decoy top1 的分布、各阈值的
fdr(s)、以及 q(s) 的塌缩过程逐项打印出来，用真实数字定位根因。

数据（全库，都在盘上）:
    query  = data/mtbls13729/smoke/neg_emb/embeddings.npy      (6915 x 1024)
    library= data/models/mona_neg_dreams_emb/embeddings.npy    (36663 x 1024)
    decoy  = data/models/mona_neg_decoy_emb.npy                 (36663 x 1024)

用法 (conda dreams_env, CPU):
    python tasks/diagnose_fdr_collapse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation.fdr import top1_scores, target_decoy_qvalues  # noqa: E402

QUERY = ROOT / "data/mtbls13729/smoke/neg_emb/embeddings.npy"
LIB = ROOT / "data/models/mona_neg_dreams_emb/embeddings.npy"
DECOY = ROOT / "data/models/mona_neg_decoy_emb.npy"


def main() -> None:
    query = np.load(QUERY)
    library = np.load(LIB)
    decoy = np.load(DECOY)
    print(f"[diag] query {query.shape}  library {library.shape}  decoy {decoy.shape}",
          flush=True)

    target = top1_scores(query, library)   # 每 query 对 36663 target 的最高 cosine
    dec = top1_scores(query, decoy)        # 每 query 对 36663 decoy 的最高 cosine
    print(f"\n[diag] target top1  vs  decoy top1  (n={len(target)})", flush=True)
    for name, arr in (("target", target), ("decoy", dec)):
        pct = np.percentile(arr, [0, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]).round(4)
        print(f"  {name:6s} percentiles[0,25,50,75,90,95,99,99.5,99.9,max] = {pct}",
              flush=True)

    # 诱饵与 target 的"竞争差距"：诱饵 max 是否低于 target 尾部
    dec_max = float(dec.max())
    tgt_max = float(target.max())
    n_target_above_dec_max = int((target > dec_max).sum())
    print(f"\n[diag] decoy max = {dec_max:.4f}  target max = {tgt_max:.4f}", flush=True)
    print(f"[diag] target 尾部 cos > decoy_max 的条数 = {n_target_above_dec_max} "
          f"({n_target_above_dec_max/len(target)*100:.2f}%)", flush=True)

    # 逐阈值 fdr(s) = (N_decoy(s)+1)/(N_target(s)+1)
    print(f"\n[diag] 逐阈值 fdr(s) 与 q(s):", flush=True)
    print(f"  {'threshold':>9s} {'N_target>=s':>10s} {'N_decoy>=s':>10s} "
          f"{'fdr(s)':>8s} {'q(s)':>8s}", flush=True)
    for th in (1.00, 0.98, 0.95, 0.92, 0.90, 0.85, 0.80, 0.75, 0.70, 0.66, 0.60):
        nt = int((target >= th).sum())
        nd = int((dec >= th).sum())
        fdr = (nd + 1.0) / (nt + 1.0)
        # q(s) = min_{s'>=s} fdr(s') —— 用正式实现
        qvals = target_decoy_qvalues(target, dec)
        # q 值按 query 索引对应，取"阈值 th 对应的最小 q"近似：直接看该阈值处的 fdr 与全局 min
        print(f"  {th:9.2f} {nt:10d} {nd:10d} {fdr:8.4f}", flush=True)

    # 正式 q 值分布（塌缩验证）
    q = target_decoy_qvalues(target, dec)
    print(f"\n[diag] 正式 q-value 分布（塌缩验证）:", flush=True)
    print(f"  q percentiles[0,25,50,75,90,100] = "
          f"{np.percentile(q, [0, 25, 50, 75, 90, 100]).round(6)}", flush=True)
    n_unique = len(np.unique(q))
    print(f"  q 唯一值个数 = {n_unique}（若≈1 说明全体被拖到同一小值=塌缩）", flush=True)
    print(f"  q 最小值 = {q.min():.6f}  1/(N_decoy+1) = {1.0/(len(dec)+1):.6f}", flush=True)

    # 塌缩的关键证据：fdr(0.66) 本应≈0.98，但 q(0.66) 被 min 拖到全局最小
    q_low = q[target < 0.70]  # 低分 query 的 q
    print(f"\n[diag] 关键证据: cos<0.70 的低分 query 共 {len(q_low)} 张，", flush=True)
    print(f"  它们的 q-value 中位数 = {np.median(q_low):.6f}（若≈全局最小则塌缩坐实）",
          flush=True)


if __name__ == "__main__":
    main()
