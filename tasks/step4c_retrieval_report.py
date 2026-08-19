"""Part C：从 lr_scan.json 提取检索指标，补齐决策表（检索 = 第三套验证"提升模型能力"）。

Part B 的逐 epoch 打印/决策表没带 retrieval，但它其实已经在 Part B 里算好并写进
lr_scan.json 了（compute_metrics 每次都调 retrieval_metrics）。本脚本**零训练、零 GPU**，
只读 JSON：
  1. 打印每个 lr 每个 epoch 的 retrieval（macro_auc / recall1 / mrr）+ pos/neg/sep/cos/rank
  2. 与 baseline 对照，给出「加检索门槛 + 收紧秩门槛」后的 lr 建议

机器时：纯 JSON 读取，CPU 单核 <1 分钟，不需要 GPU。

用法（sbatch 见 run_step4c_retrieval_report.sbatch）：
  python tasks/step4c_retrieval_report.py \
      --scan data/validation/lr_scan/lr_scan.json \
      --baseline data/validation/space_audit/baseline.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, default=ROOT / "data/validation/lr_scan/lr_scan.json")
    ap.add_argument("--baseline", type=Path, default=ROOT / "data/validation/space_audit/baseline.json")
    args = ap.parse_args()

    scan = json.load(open(args.scan, encoding="utf-8"))
    base = json.load(open(args.baseline, encoding="utf-8"))

    b_ret = base["retrieval"]
    b_auc, b_r1, b_mrr = b_ret["macro_auc"], b_ret["recall1"], b_ret["mrr"]
    b_pos = base["pos_cos_noise_consistency"]
    b_neg = base["neg_cos_isomer"]
    b_cos = base["space_structure"]["pairwise_cos_mean"]
    b_rank = base["space_structure"]["participation_ratio"]

    print("=" * 92)
    print("Part C：检索指标补齐（零训练，只读 lr_scan.json）")
    print("=" * 92)
    print(f"[baseline] retrieval: auc={b_auc:.4f} recall1={b_r1:.4f} mrr={b_mrr:.4f}")
    print(f"[baseline] pos={b_pos:.4f} neg={b_neg:.4f} cos={b_cos:.4f} rank={b_rank:.1f}\n")

    print(f"{'lr':>8} {'ep':>2} {'pos':>6} {'neg':>6} {'sep':>6} {'cos':>6} {'rank':>5} "
          f"{'auc':>6} {'r@1':>6} {'mrr':>6}")
    print("-" * 92)
    last = {}
    for r in scan["results"]:
        lr = r["lr"]
        for i, e in enumerate(r["per_epoch"]):
            ret = e["retrieval"]
            sp = e["space"]
            print(f"{lr:>8.0e} {i:>2} {e['pos_cos']:>6.3f} {e['neg_cos']:>6.3f} "
                  f"{e['separation']:>6.3f} {sp['pairwise_cos_mean']:>6.3f} "
                  f"{sp['participation_ratio']:>5.1f} "
                  f"{ret['macro_auc']:>6.3f} {ret['recall1']:>6.3f} {ret['mrr']:>6.3f}")
        last[lr] = r["per_epoch"][-1]
    print()

    # 决策：加检索门槛 + 收紧秩门槛（0.7×baseline，而非 Part B 的 0.5×）
    rank_thr = 0.7 * b_rank
    print("=" * 92)
    print(f"决策（加检索门槛 + 收紧秩门槛 rank>{rank_thr:.1f}）")
    print("=" * 92)
    valid = []
    for r in scan["results"]:
        lr = r["lr"]
        e = last[lr]
        ret = e["retrieval"]
        sp = e["space"]
        ok_learn = e["loss"] < r["loss_first"]
        ok_pos = e["pos_cos"] > b_pos
        ok_neg = e["neg_cos"] < b_neg
        ok_space = sp["pairwise_cos_mean"] < b_cos + 0.10 and sp["participation_ratio"] > rank_thr
        ok_ret = ret["macro_auc"] >= b_auc - 0.01 and ret["recall1"] >= b_r1 - 0.01
        ok = all([ok_learn, ok_pos, ok_neg, ok_space, ok_ret])
        if ok:
            valid.append(lr)
        print(f"  lr={lr:.0e} learn={ok_learn} pos_up={ok_pos} neg_down={ok_neg} "
              f"space={ok_space} ret_ok={ok_ret} -> {'✓' if ok else '✗'}")
    chosen = max(valid) if valid else None
    print(f"\n[建议] 加检索门槛后选 lr = {chosen:.0e}" if chosen else
          "\n[建议] 无 lr 全过（检索或秩门槛卡住，回查配比）")

    orig = scan["decision"].get("chosen_lr")
    print(f"[对照] Part B 原决策（较松门槛，rank>0.5×baseline）选 lr = {orig:.0e}" if orig else
          "[对照] Part B 原决策无选定")


if __name__ == "__main__":
    main()
