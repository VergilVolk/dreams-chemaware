"""Part A：官方模型 baseline 空间结构 + 任务指标审计（本机 CPU，仅前向，不训练）。

在下一次训练前确立"健康参照"（2026-08-19 定案，LR-空间门的前置）：
  1. 空间结构（坍缩探测器）：pairwise_cos_mean / participation_ratio / eff_rank /
     mean_dimension_std —— 健康 = 高有效秩 + 低两两 cos；坍缩 = 秩≈1 + cos≈1。
  2. 任务指标：pos_cos（干净 vs 噪声，应"拉近"）/ neg_cos（干净 vs 异构体，应"拉远"）/
     separation = pos − neg。
  3. 检索：10ppm 同 adduct macro-AUC / Recall@1。

附带探测器自检，三类合成嵌入验证指标能区分：
  - 健康（随机方向）     → 高秩、低 pairwise_cos
  - 点坍缩（全挤同一方向）→ 秩≈1、cos≈1（pairwise_cos 与 rank 都该抓）
  - 维坍缩（2 维子空间内方向各异）→ cos≈0 但 秩≈2（只有 rank 抓得到）

采样按「不同分子」而非「锚行」：eval 锚按 ik14 去重取前 N 个分子，避免"前 N 行
其实只有 3 个分子"导致空间指标无意义（smoke 已暴露此问题）。

用法（本机 conda，CPU）：
  python tasks/step4a_space_audit.py --device cpu --max-molecules 300 --max-neg 4
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
from metrics_space import retrieval_metrics, space_structure_metrics  # noqa: E402

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_MANIFEST = ROOT / "tasks/massspecgym_isomers/dataset_manifest.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
OUT_DIR = ROOT / "data/validation/space_audit"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    ap.add_argument("--n-highest-peaks", type=int, default=100)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-molecules", type=int, default=300, help="不同分子数（0=全量）")
    ap.add_argument("--max-neg", type=int, default=4, help="每锚最多几个异构体（审计够用，省前向）")
    ap.add_argument("--noise-draws", type=int, default=1)
    ap.add_argument("--ppm-tol", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def embed(model, spectra_list, device, batch_size):
    out = []
    for i in range(0, len(spectra_list), batch_size):
        batch = torch.stack(spectra_list[i:i + batch_size]).to(device)
        with torch.inference_mode():
            out.append(model(batch).cpu())
    return torch.cat(out, dim=0).numpy()


def detector_self_check(d=1024, n=300, seed=0):
    """三类合成嵌入：健康 / 点坍缩 / 维坍缩。返回 dict 便于打印与自检。"""
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    healthy = space_structure_metrics(X)

    base = rng.standard_normal(d).astype(np.float32)
    base /= np.linalg.norm(base)
    C = base[None, :] + 0.01 * rng.standard_normal((n, d)).astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    point_col = space_structure_metrics(C)

    U, _ = np.linalg.qr(rng.standard_normal((d, 2)).astype(np.float32))
    coords = rng.standard_normal((n, 2)).astype(np.float32)
    D = coords @ U.T
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    dim_col = space_structure_metrics(D)
    return healthy, point_col, dim_col


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    print("===== 探测器自检（三类合成嵌入）=====")
    healthy, point_col, dim_col = detector_self_check()
    print(f"  健康(随机)  : pairwise_cos={healthy['pairwise_cos_mean']:.4f} "
          f"rank={healthy['participation_ratio']:.1f} dim_std={healthy['mean_dimension_std']:.4f}")
    print(f"  点坍缩      : pairwise_cos={point_col['pairwise_cos_mean']:.4f} "
          f"rank={point_col['participation_ratio']:.2f} dim_std={point_col['mean_dimension_std']:.4f}")
    print(f"  维坍缩(rank2): pairwise_cos={dim_col['pairwise_cos_mean']:.4f} "
          f"rank={dim_col['participation_ratio']:.2f} dim_std={dim_col['mean_dimension_std']:.4f}")
    ok = (healthy["participation_ratio"] > 50
          and point_col["participation_ratio"] < 5
          and point_col["pairwise_cos_mean"] > 0.9
          and dim_col["participation_ratio"] < 10
          and dim_col["pairwise_cos_mean"] < 0.1)
    print(f"  自检: {'PASS' if ok else 'FAIL'}（健康高秩/点坍缩秩≈1 cos≈1/维坍缩秩≈2但cos≈0）\n")

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    eval_entries = manifest["eval"]

    # 按不同分子采样：ik14 去重（保序）取前 N 个分子，再用其全部锚行。
    seen = set()
    mol_groups = []
    for e in eval_entries:
        if e["ik14"] in seen:
            continue
        seen.add(e["ik14"])
        mol_groups.append(e["ik14"])
        if args.max_molecules > 0 and len(mol_groups) >= args.max_molecules:
            break
    mol_set = set(mol_groups)
    entries = [e for e in eval_entries if e["ik14"] in mol_set]
    n_mol = len(mol_groups)
    print(f"[eval] 不同分子 {n_mol} 个 -> 锚 {len(entries)} 行", flush=True)

    with h5py.File(args.data, "r") as f:
        pmz_all = np.array(f["precursor_mz"][:], dtype=float)

        noise_cfg = NoiseConfig()
        model, kind = load_base_model(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks)
        model.eval()

        t0 = time.time()
        clean_list, noisy_list = [], []
        neg_flat, neg_ptr = [], [0]
        iks, pmzs, adducts = [], [], []
        for e in entries:
            r = e["anchor_row"]
            raw = np.asarray(f["spectrum"][r])
            pmz = float(pmz_all[r])
            clean_list.append(preprocess_spectrum(raw, pmz, args.n_highest_peaks))
            noisy_list.append(preprocess_spectrum(apply_noise(raw, rng, noise_cfg), pmz, args.n_highest_peaks))
            for n in e["neg"][: args.max_neg]:
                neg_flat.append(preprocess_spectrum(np.asarray(f["spectrum"][n["row"]]),
                                                    float(pmz_all[n["row"]]), args.n_highest_peaks))
            neg_ptr.append(len(neg_flat))
            iks.append(e["ik14"]); pmzs.append(e["precursor_mz"]); adducts.append(e["adduct"])

        clean_emb = embed(model, clean_list, device, args.batch_size)
        noisy_emb = embed(model, noisy_list, device, args.batch_size)
        neg_emb = embed(model, neg_flat, device, args.batch_size) if neg_flat else np.zeros((0, clean_emb.shape[1]), dtype=np.float32)

        # 空间结构：每分子一条（去重，避免同分子多 adduct 虚高 pairwise_cos）
        _, first_idx = np.unique(np.array(iks), return_index=True)
        space = space_structure_metrics(clean_emb[np.sort(first_idx)])

        # 拉近（噪声不变性）
        pos_cos = (clean_emb * noisy_emb).sum(axis=1)
        # 拉远（异构体区分），按锚聚合
        neg_cos = []
        for i in range(len(entries)):
            lo, hi = neg_ptr[i], neg_ptr[i + 1]
            if hi > lo:
                neg_cos.append(float((clean_emb[i:i + 1] * neg_emb[lo:hi]).sum(axis=1).mean()))
        neg_cos = np.array(neg_cos)

        ret = retrieval_metrics(clean_emb, iks, pmzs, adducts, args.ppm_tol)

    result = {
        "kind": kind,
        "n_distinct_molecules": n_mol,
        "n_eval_anchors": len(entries),
        "detector_self_check": {
            "healthy": healthy, "point_collapse": point_col, "dim_collapse": dim_col,
            "pass": bool(ok),
        },
        "space_structure": space,
        "pos_cos_noise_consistency": float(pos_cos.mean()) if len(pos_cos) else float("nan"),
        "neg_cos_isomer": float(neg_cos.mean()) if len(neg_cos) else float("nan"),
        "separation": float(pos_cos.mean() - neg_cos.mean()) if len(pos_cos) and len(neg_cos) else float("nan"),
        "retrieval": ret,
        "elapsed_seconds": time.time() - t0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "baseline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"\n===== Part A 完成 -> {OUT_DIR / 'baseline.json'} =====", flush=True)

    del model
    gc.collect()


if __name__ == "__main__":
    main()
