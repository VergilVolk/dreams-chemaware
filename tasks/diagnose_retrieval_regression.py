"""诊断：噪声微调后检索下降的真实机制 —— 提取错误实例，回答一个核心问题：

  macro-AUC 下降，到底是
    (a)「同分子被推散」：正确项（同 ik14）cosine 下降 → 排序里正确项掉到错误项下面；
    (b)「异构体被拉近」：错误项（不同 ik14）cosine 上升 → 排序里错误项压过正确项；
    (c) 两者都有。

方法：加载基线 + 训练后模型，对 eval 锚做 10ppm 同 adduct 检索，逐 query 记录：
  - pos_max_cos：正确项（同 ik14）的最大 cosine（同分子聚类程度）
  - neg_max_cos：错误项（不同 ik14）的最大 cosine（最难异构体逼近程度）
  - top1 是否正确（分子聚合，同 step5 recall1 逻辑）
  然后对比基线 vs 训练后的变化，并挑出「翻转错误」（基线对→训练后错）的真实实例。

输出：
  data/validation/retrieval_regression_diag.json  —— 逐 query 明细 + 聚合
  终端打印：聚合统计 + 翻转错误实例 top 若干

用法（CPU 可跑，max-eval 控制锚子集以提速；候选池 = 同一锚子集，机制诊断足够）：
  python tasks/diagnose_retrieval_regression.py \
      --trained data/validation/noise_isomer_infonce/seed_0/best_infonce.pt \
      --max-eval 2000 --device cpu
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402
from step5_gate_eval import load_trained, embed  # noqa: E402

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_MANIFEST = ROOT / "tasks/massspecgym_isomers/dataset_manifest.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    ap.add_argument("--trained", type=Path, required=True)
    ap.add_argument("--n-highest-peaks", type=int, default=100)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-eval", type=int, default=2000, help="eval 锚子集（0=全量）")
    ap.add_argument("--ppm-tol", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--top-n-flips", type=int, default=15, help="打印翻转错误实例数")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/validation/retrieval_regression_diag.json")
    return ap.parse_args()


def per_query(emb, iks, pmzs, adducts, ppm_tol):
    """逐 query 明细：pos_max_cos / neg_max_cos / top1_correct / auc。"""
    pmzs = np.asarray(pmzs); iks = np.asarray(iks); adducts = np.asarray(adducts)
    rows = []
    for qi in range(len(iks)):
        ppm_da = ppm_tol * 1e-6 * pmzs[qi]
        cand = (np.abs(pmzs - pmzs[qi]) <= ppm_da) & (np.arange(len(iks)) != qi) & (adducts == adducts[qi])
        idx = np.where(cand)[0]
        if len(idx) == 0:
            continue
        labels = (iks[idx] == iks[qi]).astype(int)
        n_pos = int(labels.sum()); n_neg = int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        scores = (emb[qi:qi + 1] * emb[idx]).sum(axis=1)
        pos_cos = float(scores[labels == 1].max()) if n_pos else float("nan")
        neg_cos = float(scores[labels == 0].max()) if n_neg else float("nan")
        # top1（分子聚合，同 step5 recall1）：每个候选 ik 取 max cosine，排序
        best = {}
        for j, s in zip(idx, scores):
            ik = iks[j]
            if ik not in best or s > best[ik]:
                best[ik] = float(s)
        order = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        top1_correct = bool(order and order[0][0] == iks[qi])
        # auc（pos vs neg 两两）
        pos_s, neg_s = scores[labels == 1], scores[labels == 0]
        diff = pos_s[:, None] - neg_s[None, :]
        auc = float((np.count_nonzero(diff > 0) + 0.5 * np.count_nonzero(diff == 0)) / diff.size)
        rows.append({
            "qi": int(qi), "ik14": str(iks[qi]), "adduct": str(adducts[qi]), "pmz": float(pmzs[qi]),
            "n_pos": n_pos, "n_neg": n_neg,
            "pos_max_cos": pos_cos, "neg_max_cos": neg_cos,
            "top1_ik": str(order[0][0]) if order else None,
            "top1_correct": top1_correct, "auc": auc,
        })
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    manifest = json.load(open(args.manifest))
    full = manifest["eval"]
    entries = full[: args.max_eval] if (args.max_eval > 0 and len(full) > args.max_eval) else full
    print(f"[diag] eval 锚子集={len(entries)}（候选池 = 同一子集）", flush=True)

    with h5py.File(args.data, "r") as f:
        pmz_all = np.array(f["precursor_mz"][:], dtype=float)
        clean_specs, iks, pmzs, adducts = [], [], [], []
        for e in entries:
            r = e["anchor_row"]
            raw = np.asarray(f["spectrum"][r])
            clean_specs.append(preprocess_spectrum(raw, float(pmz_all[r]), args.n_highest_peaks))
            iks.append(e["ik14"]); pmzs.append(e["precursor_mz"]); adducts.append(e["adduct"])

    print("[1] 加载基线...", flush=True)
    base_model, _ = load_base_model(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks)
    base_model.eval()
    print("[2] 加载训练模型...", flush=True)
    trn_model, _ = load_trained(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks, args.trained)

    t0 = time.time()
    print("[3] 嵌入基线...", flush=True)
    base_emb = embed(base_model, clean_specs, device, args.batch_size).numpy()
    print("[4] 嵌入训练模型...", flush=True)
    trn_emb = embed(trn_model, clean_specs, device, args.batch_size).numpy()
    print(f"[embed] {time.time()-t0:.0f}s", flush=True)

    base_rows = per_query(base_emb, iks, pmzs, adducts, args.ppm_tol)
    trn_rows = per_query(trn_emb, iks, pmzs, adducts, args.ppm_tol)

    # 对齐 qi
    bm = {r["qi"]: r for r in base_rows}
    tm = {r["qi"]: r for r in trn_rows}
    qis = sorted(set(bm) & set(tm))

    # 聚合
    d_pos, d_neg, d_auc = [], [], []
    flip_bad, flip_good = [], []
    for qi in qis:
        b, t = bm[qi], tm[qi]
        d_pos.append(t["pos_max_cos"] - b["pos_max_cos"])
        d_neg.append(t["neg_max_cos"] - b["neg_max_cos"])
        d_auc.append(t["auc"] - b["auc"])
        if b["top1_correct"] and not t["top1_correct"]:
            flip_bad.append((qi, b, t))
        elif not b["top1_correct"] and t["top1_correct"]:
            flip_good.append((qi, b, t))

    d_pos = np.array(d_pos); d_neg = np.array(d_neg); d_auc = np.array(d_auc)
    base_auc = np.mean([bm[qi]["auc"] for qi in qis])
    trn_auc = np.mean([tm[qi]["auc"] for qi in qis])
    base_r1 = np.mean([bm[qi]["top1_correct"] for qi in qis])
    trn_r1 = np.mean([tm[qi]["top1_correct"] for qi in qis])

    print("\n===== 聚合：基线 vs 训练后 =====")
    print(f"n_query={len(qis)}")
    print(f"macro-AUC:   base {base_auc:.4f} -> trn {trn_auc:.4f}  (Δ={trn_auc-base_auc:+.4f})")
    print(f"Recall@1:    base {base_r1:.4f} -> trn {trn_r1:.4f}  (Δ={trn_r1-base_r1:+.4f})")
    print(f"pos_max_cos  Δ (同分子聚类):  mean {d_pos.mean():+.4f}  (负=同分子被推散)")
    print(f"neg_max_cos  Δ (异构体逼近):  mean {d_neg.mean():+.4f}  (正=异构体被拉近)")
    print(f"  pos 下降(<-0.01) 的 query 占比: {(d_pos < -0.01).mean():.1%}")
    print(f"  neg 上升(>+0.01) 的 query 占比: {(d_neg > +0.01).mean():.1%}")
    print(f"翻转错误(基线对→训练错): {len(flip_bad)}   翻转正确(基线错→训练对): {len(flip_good)}")

    # 翻转错误的机制归因：是 pos 降还是 neg 升主导
    if flip_bad:
        fb_pos = np.array([t["pos_max_cos"] - b["pos_max_cos"] for _, b, t in flip_bad])
        fb_neg = np.array([t["neg_max_cos"] - b["neg_max_cos"] for _, b, t in flip_bad])
        print(f"\n===== 翻转错误 {len(flip_bad)} 例的归因 =====")
        print(f"  这些例里 同分子 pos_max_cos 平均 Δ={fb_pos.mean():+.4f}（{np.mean(fb_pos < -0.01):.1%} 例下降）")
        print(f"  这些例里 异构体 neg_max_cos 平均 Δ={fb_neg.mean():+.4f}（{np.mean(fb_neg > +0.01):.1%} 例上升）")
        print(f"  → 主因是 {'同分子被推散' if fb_pos.mean() < fb_neg.mean() else '异构体被拉近'}")
        print(f"\n===== 翻转错误实例 top {min(args.top_n_flips, len(flip_bad))} =====")
        flip_bad_sorted = sorted(flip_bad, key=lambda x: (x[2]['pos_max_cos'] - x[1]['pos_max_cos']))[: args.top_n_flips]
        for qi, b, t in flip_bad_sorted:
            print(f"  ik14={b['ik14'][:13]}  adduct={b['adduct']:9s} pmz={b['pmz']:.4f}  "
                  f"pos_cos {b['pos_max_cos']:.3f}->{t['pos_max_cos']:.3f} (Δ{t['pos_max_cos']-b['pos_max_cos']:+.3f})  "
                  f"neg_cos {b['neg_max_cos']:.3f}->{t['neg_max_cos']:.3f} (Δ{t['neg_max_cos']-b['neg_max_cos']:+.3f})  "
                  f"top1错项={t['top1_ik'][:13]}")

    summary = {
        "trained_checkpoint": str(args.trained),
        "n_query": len(qis),
        "base": {"macro_auc": base_auc, "recall1": base_r1},
        "trained": {"macro_auc": trn_auc, "recall1": trn_r1},
        "delta": {
            "pos_max_cos_mean": float(d_pos.mean()),
            "neg_max_cos_mean": float(d_neg.mean()),
            "frac_pos_down": float(np.mean(d_pos < -0.01)),
            "frac_neg_up": float(np.mean(d_neg > +0.01)),
        },
        "n_flip_bad": len(flip_bad),
        "n_flip_good": len(flip_good),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[diag] 已写 {args.out}", flush=True)

    del base_model, trn_model
    gc.collect()


if __name__ == "__main__":
    main()
