"""
Step 5: G3 门评估 —— 噪声一致性 + 异构体区分(FP 守卫) + 10ppm 检索。

对比对象：InfoNCE 训练后的模型 vs 官方基线（official_embedding_slim.pt）。

三个门（docs/NOISE_TASK_PLAN_20260818.md §7 G3）：
  G3-1 噪声一致性 ↑   : mean cos(anchor_clean, anchor_noise)  trained > baseline
  G3-2 异构体区分(FP) : mean cos(anchor_clean, isomer)        trained < baseline
                        （同 formula 异构体被推远 = 撞脸不再混淆）
  G3-3 检索不降       : 10ppm 同 adduct 检索 macro-AUC / Recall@1 不降 > baseline-0.01

eval 集 = manifest 的 eval 锚（分子与训练集不相交），噪声现场施加（同 Step 4 四轴）。

用法（GPU）：
  python tasks/step5_gate_eval.py --device cuda \
      --trained data/validation/noise_isomer_infonce/seed_0/best_infonce.pt
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
from noise_augment import NoiseConfig, apply_noise  # noqa: E402

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
    ap.add_argument("--trained", type=Path, required=True, help="训练出的 best_infonce.pt")
    ap.add_argument("--n-highest-peaks", type=int, default=100)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-eval", type=int, default=2000, help="eval 锚子集（0=全量）")
    ap.add_argument("--noise-draws", type=int, default=1, help="每锚噪声抽样数（取均值）")
    ap.add_argument("--ppm-tol", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_trained(base_ckpt, arch_ckpt, device, n_highest, ckpt_path):
    model, kind = load_base_model(base_ckpt, arch_ckpt, device, n_highest)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.backbone.load_state_dict(ckpt["backbone_state_dict"])
    model.head.load_state_dict(ckpt["head_state_dict"])
    model.eval()
    return model, kind


def embed(model, spectra_list, device, batch_size):
    """model 前向得到归一化嵌入；spectra_list: list[Tensor(101,2)]。"""
    out = []
    for i in range(0, len(spectra_list), batch_size):
        batch = torch.stack(spectra_list[i:i + batch_size]).to(device)
        with torch.inference_mode():
            out.append(model(batch).cpu())
    return torch.cat(out, dim=0)


def query_auc(labels, scores):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float((np.count_nonzero(diff > 0) + 0.5 * np.count_nonzero(diff == 0)) / diff.size)


def retrieval_metrics(emb, iks, pmzs, adducts, ppm_tol):
    """严格 10ppm 同 adduct 检索（分子聚合），同旧 pilot 协议。"""
    pmzs = np.asarray(pmzs); iks = np.asarray(iks); adducts = np.asarray(adducts)
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
        for j, s in zip(idx, scores):
            ik = iks[j]
            if ik not in best or s > best[ik]:
                best[ik] = float(s)
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


def evaluate_model(model, entries, h5, pmz_all, n_highest, noise_cfg, rng, device, batch_size, noise_draws, ppm_tol):
    """返回 pos_cos / neg_cos / separation / retrieval 指标。"""
    clean_embs, noisy_embs, neg_embs = [], [], []
    neg_counts = []
    iks, pmzs, adducts = [], [], []

    for e in entries:
        r = e["anchor_row"]
        raw = np.asarray(h5["spectrum"][r])
        pmz = float(pmz_all[r])
        clean = preprocess_spectrum(raw, pmz, n_highest)
        # 噪声多抽样取均值（正例一致性）
        noisy = [preprocess_spectrum(apply_noise(raw, rng, noise_cfg), pmz, n_highest)
                 for _ in range(noise_draws)]
        clean_embs.append(clean)
        noisy_embs.extend(noisy)
        neg_embs.append([preprocess_spectrum(np.asarray(h5["spectrum"][n["row"]]),
                                             float(pmz_all[n["row"]]), n_highest)
                         for n in e["neg"]])
        neg_counts.append(len(e["neg"]))
        iks.append(e["ik14"]); pmzs.append(e["precursor_mz"]); adducts.append(e["adduct"])

    clean_emb = embed(model, clean_embs, device, batch_size)
    noisy_emb = embed(model, noisy_embs, device, batch_size)  # len = n_entries * noise_draws

    # pos cos（每锚 noise_draws 次取平均）
    pos_cos = []
    for i in range(len(entries)):
        seg = noisy_emb[i * noise_draws:(i + 1) * noise_draws]
        pos_cos.append(float((clean_emb[i:i + 1] * seg).sum(1).mean()))
    pos_cos = np.array(pos_cos)

    # neg cos（每锚对全部异构体取平均，无异构体则跳过）
    neg_cos = []
    for i, negs in enumerate(neg_embs):
        if not negs:
            continue
        ne = embed(model, negs, device, batch_size)
        neg_cos.append(float((clean_emb[i:i + 1] * ne).sum(1).mean()))
    neg_cos = np.array(neg_cos)

    ret = retrieval_metrics(clean_emb.numpy(), iks, pmzs, adducts, ppm_tol)
    return {
        "noise_consistency_pos_cos": float(pos_cos.mean()) if len(pos_cos) else float("nan"),
        "isomer_neg_cos": float(neg_cos.mean()) if len(neg_cos) else float("nan"),
        "separation": float(pos_cos.mean() - neg_cos.mean()) if len(pos_cos) and len(neg_cos) else float("nan"),
        "n_anchors": len(entries),
        "n_with_isomer": int((neg_counts and np.array(neg_counts) > 0).sum()) if neg_counts else 0,
        "retrieval": ret,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    entries = manifest["eval"]
    if args.max_eval > 0 and len(entries) > args.max_eval:
        entries = entries[: args.max_eval]
    print(f"[eval] 锚: {len(entries)}", flush=True)

    with h5py.File(args.data, "r") as f:
        pmz_all = np.array(f["precursor_mz"][:], dtype=float)

    noise_cfg = NoiseConfig()

    print("[1] 加载基线（official）...", flush=True)
    base_model, _ = load_base_model(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks)
    base_model.eval()

    print("[2] 加载训练模型...", flush=True)
    trn_model, kind = load_trained(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks, args.trained)

    t0 = time.time()
    with h5py.File(args.data, "r") as f:
        print("[3] 评估基线...", flush=True)
        base_res = evaluate_model(base_model, entries, f, pmz_all, args.n_highest_peaks,
                                  noise_cfg, rng, device, args.batch_size, args.noise_draws, args.ppm_tol)
        print("[4] 评估训练模型...", flush=True)
        trn_res = evaluate_model(trn_model, entries, f, pmz_all, args.n_highest_peaks,
                                 noise_cfg, rng, device, args.batch_size, args.noise_draws, args.ppm_tol)

    g3_1 = trn_res["noise_consistency_pos_cos"] > base_res["noise_consistency_pos_cos"]
    g3_2 = trn_res["isomer_neg_cos"] < base_res["isomer_neg_cos"]
    g3_3 = (trn_res["retrieval"]["macro_auc"] >= base_res["retrieval"]["macro_auc"] - 0.01
            and trn_res["retrieval"]["recall1"] >= base_res["retrieval"]["recall1"] - 0.01)

    summary = {
        "kind": kind,
        "trained_checkpoint": str(args.trained),
        "n_eval_anchors": len(entries),
        "noise_draws": args.noise_draws,
        "baseline": base_res,
        "trained": trn_res,
        "G3_1_noise_consistency_up": bool(g3_1),
        "G3_2_isomer_pushed_away": bool(g3_2),
        "G3_3_retrieval_not_worse": bool(g3_3),
        "gate_overall": bool(g3_1 and g3_2 and g3_3),
        "elapsed_seconds": time.time() - t0,
    }
    out = args.trained.parent / "gate_eval.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"\n=== G3 gate: {'PASS' if summary['gate_overall'] else 'FAIL'} ===", flush=True)
    print(f"  噪声一致性 pos_cos: base {base_res['noise_consistency_pos_cos']:.4f} -> "
          f"trained {trn_res['noise_consistency_pos_cos']:.4f}  ({'↑' if g3_1 else '✗'})")
    print(f"  异构体 neg_cos:     base {base_res['isomer_neg_cos']:.4f} -> "
          f"trained {trn_res['isomer_neg_cos']:.4f}  ({'↓' if g3_2 else '✗'})")
    print(f"  separation:         base {base_res['separation']:.4f} -> trained {trn_res['separation']:.4f}")
    print(f"  检索 macro-AUC:     base {base_res['retrieval']['macro_auc']:.4f} -> "
          f"trained {trn_res['retrieval']['macro_auc']:.4f}")
    print(f"  检索 Recall@1:      base {base_res['retrieval']['recall1']:.4f} -> "
          f"trained {trn_res['retrieval']['recall1']:.4f}")

    del base_model, trn_model
    gc.collect()


if __name__ == "__main__":
    main()
