"""Part B：学习率扫描 + 训练中坍缩监控（定 lr，--no-amp）。

在「下一次训练」前限定 lr：对每个 lr 在固定小训练集上短训 N epoch，每 epoch 在
held-out eval 分子集上采一次全指标（三套：拉近 pos_cos / 拉远 neg_cos / 全局
空间结构+检索），对比 Part A baseline 判断该 lr 是否「学会了但不坍缩」。

决策规则（读 baseline.json 作参照，选不坍缩前提下最高的 lr）：
  对每个 lr 的末 epoch：
    - 学习生效      loss_final < loss_init
    - 拉近          pos_cos > baseline.pos_cos
    - 拉远          neg_cos < baseline.neg_cos
    - 空间不坍缩    pairwise_cos_mean < baseline.pairwise_cos_mean + 0.10
                    且 participation_ratio > 0.5 * baseline.participation_ratio
  取同时满足四条的 lr 中最高者。

复用 step4 的 NoiseContrastiveDataset / infonce_forward / 冻结控制，损失与 G2/G3
完全一致；唯一区别是 --no-amp（fp32，避开 fp16 溢出）与固定 seed。

用法（GPU）：
  python tasks/step4b_lr_scan.py --device cuda \
      --lrs 3e-6,1e-5,3e-5,1e-4,3e-4 --epochs 3 --max-train-anchors 4000 \
      --eval-molecules 300 --baseline-json data/validation/space_audit/baseline.json
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_e1_identity import load_base_model, preprocess_spectrum, seed_everything  # noqa: E402
from noise_augment import NoiseConfig, apply_noise  # noqa: E402
from metrics_space import retrieval_metrics, space_structure_metrics  # noqa: E402
from step4_infonce_train import (  # noqa: E402
    NoiseContrastiveDataset,
    collate,
    freeze_all,
    infonce_forward,
    unfreeze_last_layers,
)

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_MANIFEST = ROOT / "tasks/massspecgym_isomers/dataset_manifest.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_OUT = ROOT / "data/validation/lr_scan"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    ap.add_argument("--baseline-json", type=Path, default=ROOT / "data/validation/space_audit/baseline.json")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lrs", type=str, default="3e-6,1e-5,3e-5,1e-4,3e-4")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--unfreeze-layers", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--n-highest-peaks", type=int, default=100)
    ap.add_argument("--max-train-anchors", type=int, default=4000)
    ap.add_argument("--eval-molecules", type=int, default=300)
    ap.add_argument("--max-neg", type=int, default=4)
    ap.add_argument("--ppm-tol", type=float, default=10.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def embed(model, spectra_list, device, batch_size):
    out = []
    for i in range(0, len(spectra_list), batch_size):
        batch = torch.stack(spectra_list[i:i + batch_size]).to(device)
        with torch.inference_mode():
            out.append(model(batch).cpu())
    return torch.cat(out, dim=0).numpy()


def sample_eval_molecules(eval_entries, max_molecules):
    seen = set()
    mols = []
    for e in eval_entries:
        if e["ik14"] in seen:
            continue
        seen.add(e["ik14"])
        mols.append(e["ik14"])
        if max_molecules > 0 and len(mols) >= max_molecules:
            break
    mol_set = set(mols)
    return [e for e in eval_entries if e["ik14"] in mol_set], len(mols)


def prepare_eval_data(entries, f, pmz_all, n_highest, noise_cfg, rng, max_neg):
    clean_list, noisy_list = [], []
    neg_flat, neg_ptr = [], [0]
    iks, pmzs, adducts = [], [], []
    for e in entries:
        r = e["anchor_row"]
        raw = np.asarray(f["spectrum"][r])
        pmz = float(pmz_all[r])
        clean_list.append(preprocess_spectrum(raw, pmz, n_highest))
        noisy_list.append(preprocess_spectrum(apply_noise(raw, rng, noise_cfg), pmz, n_highest))
        for n in e["neg"][:max_neg]:
            neg_flat.append(preprocess_spectrum(np.asarray(f["spectrum"][n["row"]]),
                                                float(pmz_all[n["row"]]), n_highest))
        neg_ptr.append(len(neg_flat))
        iks.append(e["ik14"]); pmzs.append(e["precursor_mz"]); adducts.append(e["adduct"])
    return clean_list, noisy_list, neg_flat, neg_ptr, iks, pmzs, adducts


def compute_metrics(model, clean_list, noisy_list, neg_flat, neg_ptr, iks, pmzs, adducts,
                    device, batch_size, ppm_tol):
    clean_emb = embed(model, clean_list, device, batch_size)
    noisy_emb = embed(model, noisy_list, device, batch_size)
    neg_emb = embed(model, neg_flat, device, batch_size) if neg_flat else np.zeros((0, clean_emb.shape[1]), dtype=np.float32)

    _, first_idx = np.unique(np.array(iks), return_index=True)
    space = space_structure_metrics(clean_emb[np.sort(first_idx)])

    pos_cos = (clean_emb * noisy_emb).sum(axis=1)
    neg_cos = []
    for i in range(len(iks)):
        lo, hi = neg_ptr[i], neg_ptr[i + 1]
        if hi > lo:
            neg_cos.append(float((clean_emb[i:i + 1] * neg_emb[lo:hi]).sum(axis=1).mean()))
    neg_cos = np.array(neg_cos)

    ret = retrieval_metrics(clean_emb, iks, pmzs, adducts, ppm_tol)
    return {
        "pos_cos": float(pos_cos.mean()) if len(pos_cos) else float("nan"),
        "neg_cos": float(neg_cos.mean()) if len(neg_cos) else float("nan"),
        "separation": float(pos_cos.mean() - neg_cos.mean()) if len(pos_cos) and len(neg_cos) else float("nan"),
        "space": space,
        "retrieval": ret,
    }


def train_one_lr(lr, args, train_entries, eval_data, pmz_all, noise_cfg, device):
    """返回 {lr, loss_first, loss_last, per_epoch: [metrics...]}。"""
    seed_everything(args.seed)
    model, kind = load_base_model(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks)
    freeze_all(model)
    unfreeze_last_layers(model, args.unfreeze_layers)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=args.weight_decay)

    dataset = NoiseContrastiveDataset(train_entries, args.data, pmz_all, args.n_highest_peaks, noise_cfg, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=collate, pin_memory=device.type == "cuda", persistent_workers=False)

    clean_list, noisy_list, neg_flat, neg_ptr, iks, pmzs, adducts = eval_data
    loss_first = None
    per_epoch = []
    model.train()
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        losses = []
        for anchors, pos, nc, nn, neg_ptr_t in loader:
            anchors, pos, nc, nn = anchors.to(device), pos.to(device), nc.to(device), nn.to(device)
            neg_ptr_t = neg_ptr_t.to(device)
            loss, _, _ = infonce_forward(model, anchors, pos, nc, nn, neg_ptr_t, args.tau)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if not np.isfinite(losses[-1]):
                print(f"    [lr={lr:.0e}] NaN at epoch {epoch} step {len(losses)}; 提前终止", flush=True)
                break
        if loss_first is None:
            loss_first = losses[0] if losses else float("nan")
        model.eval()
        m = compute_metrics(model, clean_list, noisy_list, neg_flat, neg_ptr, iks, pmzs, adducts,
                            device, args.batch_size, args.ppm_tol)
        m["loss"] = float(np.mean(losses)) if losses else float("nan")
        per_epoch.append(m)
        print(f"    lr={lr:.0e} epoch {epoch}: loss={m['loss']:.4f} pos={m['pos_cos']:.4f} "
              f"neg={m['neg_cos']:.4f} sep={m['separation']:.4f} "
              f"cos_mean={m['space']['pairwise_cos_mean']:.4f} rank={m['space']['participation_ratio']:.1f}",
              flush=True)
        model.train()
        if not np.isfinite(m["loss"]):
            break
    del model
    gc.collect()
    return {"lr": lr, "kind": kind, "loss_first": loss_first, "per_epoch": per_epoch}


def decide(scan_results, baseline_full):
    """返回 {chosen_lr, table}。baseline_full = Part A 的 baseline.json（整份）。"""
    base_sp = baseline_full["space_structure"]
    base_pcos = base_sp["pairwise_cos_mean"]
    base_rank = base_sp["participation_ratio"]
    base_pos = baseline_full["pos_cos_noise_consistency"]
    base_neg = baseline_full["neg_cos_isomer"]
    table = []
    valid = []
    for r in scan_results:
        last = r["per_epoch"][-1] if r["per_epoch"] else {}
        sp = last.get("space", {})
        ok_learn = np.isfinite(r["loss_first"]) and np.isfinite(last.get("loss", np.nan)) and last["loss"] < r["loss_first"]
        ok_pos = last.get("pos_cos", -1) > base_pos
        ok_neg = last.get("neg_cos", 2) < base_neg
        ok_space = (sp.get("pairwise_cos_mean", 1) < base_pcos + 0.10
                    and sp.get("participation_ratio", 0) > 0.5 * base_rank)
        ok = bool(ok_learn and ok_pos and ok_neg and ok_space)
        if ok:
            valid.append(r["lr"])
        table.append({
            "lr": r["lr"], "valid": ok,
            "loss_first": r["loss_first"], "loss_last": last.get("loss"),
            "pos_cos": last.get("pos_cos"), "neg_cos": last.get("neg_cos"),
            "separation": last.get("separation"),
            "pairwise_cos_mean": sp.get("pairwise_cos_mean"),
            "participation_ratio": sp.get("participation_ratio"),
            "checks": {"learn": bool(ok_learn), "pos_up": bool(ok_pos),
                       "neg_down": bool(ok_neg), "space_healthy": bool(ok_space)},
        })
    chosen = max(valid) if valid else None
    return {"chosen_lr": chosen, "table": table}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    lrs = [float(x) for x in args.lrs.split(",")]
    rng = np.random.default_rng(args.seed)

    baseline = json.load(open(args.baseline_json, encoding="utf-8"))
    print(f"[baseline] pos={baseline['pos_cos_noise_consistency']:.4f} "
          f"neg={baseline['neg_cos_isomer']:.4f} "
          f"space.cos={baseline['space_structure']['pairwise_cos_mean']:.4f} "
          f"space.rank={baseline['space_structure']['participation_ratio']:.1f}", flush=True)

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    train_entries = manifest["train"][: args.max_train_anchors]
    eval_entries, n_mol = sample_eval_molecules(manifest["eval"], args.eval_molecules)
    print(f"[data] train anchors {len(train_entries)}; eval 分子 {n_mol} -> 锚 {len(eval_entries)}", flush=True)

    with h5py.File(args.data, "r") as f:
        pmz_all = np.array(f["precursor_mz"][:], dtype=float)
        noise_cfg = NoiseConfig()
        eval_data = prepare_eval_data(eval_entries, f, pmz_all, args.n_highest_peaks, noise_cfg, rng, args.max_neg)

    results = []
    t0 = time.time()
    for lr in lrs:
        print(f"\n===== LR = {lr:.0e} =====", flush=True)
        results.append(train_one_lr(lr, args, train_entries, eval_data, pmz_all, noise_cfg, device))

    decision = decide(results, baseline)
    decision["baseline"] = {
        "pos_cos": baseline["pos_cos_noise_consistency"],
        "neg_cos": baseline["neg_cos_isomer"],
        "pairwise_cos_mean": baseline["space_structure"]["pairwise_cos_mean"],
        "participation_ratio": baseline["space_structure"]["participation_ratio"],
    }
    decision["elapsed_seconds"] = time.time() - t0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "lr_scan.json"
    out.write_text(json.dumps({"args": vars(args), "results": results, "decision": decision},
                              ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n===== LR 扫描决策表 =====", flush=True)
    for row in decision["table"]:
        flag = "✓ 选" if row["lr"] == decision["chosen_lr"] else ("✓" if row["valid"] else "✗")
        print(f"  lr={row['lr']:.0e} {flag} loss {row['loss_first']:.3f}->{row['loss_last']:.3f} "
              f"pos={row['pos_cos']:.3f} neg={row['neg_cos']:.3f} sep={row['separation']:.3f} "
              f"cos={row['pairwise_cos_mean']:.3f} rank={row['participation_ratio']:.1f}", flush=True)
    print(f"\n选定 lr: {decision['chosen_lr']}（{decision['chosen_lr']:.0e}）" if decision["chosen_lr"] else
          "\n无 lr 同时满足四条件 → 全坍缩，回查噪声菜单/负例配比，非 lr 问题。", flush=True)
    print(f"结果 -> {out}", flush=True)


if __name__ == "__main__":
    main()
