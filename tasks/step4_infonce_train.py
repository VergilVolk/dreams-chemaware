"""
Step 4: InfoNCE 对比训练（噪声不变性 + 异构体区分），解冻 backbone last-N 层。

依据 docs/NOISE_TASK_PLAN_20260818.md：
  anchor = 干净谱 A（CLS 经 head 投影 + L2 归一化）
  正例   = 噪声谱 noise(A)（唯一正例，四轴噪声）；--same-mol-as-positive 时额外
           把 batch 内同分子（同 ik14）的干净谱也作为正例（主动拉近，对冲异构体推远的副作用）
  负例   = 异构体谱 {B, noise(B), C, noise(C), ...}（clean + 噪声版都放）
         ∪ batch 内其他 anchor 的干净谱 {A_j} 与噪声谱 {noise(A_j)}（j≠i，且排除同分子 ik14 相等的 j）
  τ = 0.1；噪声每 step 现场重抽。

起点：官方微调模型 official_embedding_slim.pt（head(CLS) 约定，与 M3/M4/噪声 pilot 可比）。
解冻：last-N 层 transformer（atts+ffs+对应 scales）+ head，其余冻结（--unfreeze-layers，默认 2）。

G2 smoke 三件事（严格因果门）：
  1. 解冻生效：首步反传后 last-N 层 grad 范数 > 0（若冻结则 grad=None → 0）。
  2. loss 下降：首步 loss > 末步 loss。
  3. 无 NaN。

用法：
  # 本机 CPU smoke（tiny 子集，验证机制）
  python tasks/step4_infonce_train.py --smoke --device cpu
  # 服务器 GPU 全量（G3，后续接 sbatch）
  python tasks/step4_infonce_train.py --device cuda --epochs 8 --batch-size 32

输入：
  tasks/massspecgym_isomers/dataset_manifest.json   # Step 3 产物
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt                # 起点（官方微调）
  dreams/models/pretrained/ssl_model_server.pt      # 架构（重建 DreaMS args）
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_e1_identity import (  # noqa: E402
    cpu_state_dict,
    load_base_model,
    preprocess_spectrum,
    seed_everything,
)
from noise_augment import NoiseConfig, apply_noise  # noqa: E402

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_MANIFEST = ROOT / "tasks/massspecgym_isomers/dataset_manifest.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_OUT = ROOT / "data/validation/noise_isomer_infonce"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5, help="PartB/C 定案 3e-5；1e-4/3e-4 会坍缩+检索退化")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--preserve-lambda", type=float, default=0.0,
                    help="L_preserve 权重：λ_p·mean(1−cos(z_train, z_official))；0=关闭（默认）。"
                         ">0 时加载冻结官方教师模型锚定表示，防 InfoNCE 把空间推散导致检索退化")
    ap.add_argument("--same-mol-as-positive", action=argparse.BooleanOptionalAction, default=False,
                    help="同分子转正例（显式携带）：每个 anchor 额外把「同完整 SMILES 的另一条谱」作为正例，"
                         "主动拉近，独立于 batch 采样。G7 前审计：batch 内同分子对稀疏 0.318/批，"
                         "batch 内偶遇方案几乎不触发；改用显式携带。判据用完整 SMILES（含 @/@@ 立体标记），"
                         "不用 ik14（InChIKey 前 14 位=连通性，会把立体异构体误并成同分子，实测 8.86% 混并）。")
    ap.add_argument("--cross-condition-positive", action=argparse.BooleanOptionalAction, default=False,
                    help="Require an explicit real-spectrum positive to differ by instrument or |ΔCE|>=10; no fallback to arbitrary replicates.")
    ap.add_argument("--inbatch-negative-mode", choices=("all", "hard_only"), default="all",
                    help="hard_only uses only prebuilt same-formula/same-adduct hard negatives; all reproduces historical global in-batch InfoNCE.")
    ap.add_argument("--real-positive-weight", type=float, default=0.0,
                    help="Direct cosine consistency weight for the explicit real same-molecule positive. 0 preserves historical multi-positive-only behavior.")
    ap.add_argument("--synthetic-positive-weight", type=float, default=1.0,
                    help="Relative numerator weight of raw->synthetic-noise positive in InfoNCE. G8R uses a small auxiliary value because real cross-condition spectra are the primary positive.")
    ap.add_argument("--positive-relation-preserve-lambda", type=float, default=0.0,
                    help="Teacher-relative guard: penalise real-positive cosine falling below official DreaMS.")
    ap.add_argument("--protect-main-peaks-above", type=float, default=0.0,
                    help="For synthetic identity positives only, never delete peaks at/above this base-peak-relative intensity. 0 keeps historical uniform deletion.")
    ap.add_argument("--disable-random-added-peaks", action=argparse.BooleanOptionalAction, default=False,
                    help="Disable arbitrary random m/z additions. Recommended until additions are replaced by empirical condition-specific peaks.")
    ap.add_argument("--n-highest-peaks", type=int, default=100)
    ap.add_argument("--unfreeze-layers", type=int, default=2,
                    help="解冻 last-N 层；0=只训练官方 projection head（G8R 首门推荐）")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="若 seed 目录已有 last.pt 则续训（--no-resume 强制重训）")
    ap.add_argument("--smoke", action="store_true", help="tiny 子集 + 1 epoch，仅验证机制")
    ap.add_argument("--max-anchors", type=int, default=0,
                    help="DEPRECATED: sorted-prefix selection is invalid; use --train-subset from audit_noise_training_pool.py")
    ap.add_argument("--train-subset", type=Path, default=None,
                    help="Locked representative subset JSON written by audit_noise_training_pool.py. Required for non-smoke small pilots.")
    ap.add_argument("--max-steps", type=int, default=0, help="每 epoch 最多步数（0=全 epoch）")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
class NoiseContrastiveDataset(Dataset):
    """每个 item = 一个 anchor（谱行）；__getitem__ 现场施加四轴噪声。"""

    def __init__(self, entries, h5_path, precursor_mz_all, n_highest, noise_cfg, seed,
                 smiles_all=None, instrument_all=None, collision_energy_all=None,
                 cross_condition_positive=False):
        self.entries = entries
        self.h5_path = str(h5_path)
        self.pmz = precursor_mz_all
        self.n_highest = n_highest
        self.noise_cfg = noise_cfg
        self.seed = seed
        self.epoch = 0
        self._h5 = None
        # 显式同分子正例索引：用完整 SMILES（含 @/@@ 立体标记）判同分子。
        # 不能用 ik14（InChIKey 前 14 位=连通性）：会把立体异构体误并成同分子（实测 8.86% 混并），
        # promote 会把对映体/非对映体错误拉近。SMILES 字段含立体标记，可区分。
        self.smiles = smiles_all
        self.instrument = instrument_all
        self.collision_energy = collision_energy_all
        self.cross_condition_positive = cross_condition_positive
        self.mol_rows = defaultdict(list)
        if smiles_all is not None:
            # The positive relation must match the locked retrieval label:
            # same connectivity (IK14) and same adduct.  Routine MS2 cannot
            # reliably separate stereoisomers, whereas the retrieval protocol
            # deliberately aggregates them at connectivity level.
            for e in entries:
                self.mol_rows[(e["ik14"], e["adduct"])].append(e["anchor_row"])

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def _handle(self):
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + idx)
        h = self._handle()

        def make(row: int, noisy: bool) -> torch.Tensor:
            raw = np.asarray(h["spectrum"][row])
            if noisy:
                raw = apply_noise(raw, rng, self.noise_cfg)
            return preprocess_spectrum(raw, float(self.pmz[row]), self.n_highest)

        anchor_clean = make(e["anchor_row"], False)
        anchor_noisy = make(e["anchor_row"], True)
        neg_clean = [make(n["row"], False) for n in e["neg"]]
        neg_noisy = [make(n["row"], True) for n in e["neg"]]
        # 显式同分子正例：同完整 SMILES 的另一条谱（clean），独立于 batch 采样。
        # 单谱分子无候选 → None（loss 里被 sm_mask 屏蔽，不参与）。
        sm_pos = None
        if self.smiles is not None:
            peers = self.mol_rows[(e["ik14"], e["adduct"])]
            cands = [r for r in peers if r != e["anchor_row"]]
            if self.cross_condition_positive and self.instrument is not None and self.collision_energy is not None:
                source = int(e["anchor_row"])
                cands = [r for r in cands if (
                    self.instrument[source] != self.instrument[r]
                    or (np.isfinite(self.collision_energy[source]) and np.isfinite(self.collision_energy[r])
                        and abs(self.collision_energy[source] - self.collision_energy[r]) >= 10)
                )]
            if cands:
                row = int(cands[rng.integers(0, len(cands))])
                sm_pos = make(row, False)
        return anchor_clean, anchor_noisy, neg_clean, neg_noisy, e["ik14"], sm_pos

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


def collate(batch):
    anchors = torch.stack([b[0] for b in batch])
    pos = torch.stack([b[1] for b in batch])
    nc_flat, nn_flat = [], []
    neg_ptr = [0]
    for b in batch:
        nc_flat.extend(b[2])
        nn_flat.extend(b[3])
        neg_ptr.append(len(nc_flat))
    neg_ptr = torch.tensor(neg_ptr, dtype=torch.long)
    zero = torch.zeros(0, anchors.shape[1], anchors.shape[2])
    nc = torch.stack(nc_flat) if nc_flat else zero
    nn = torch.stack(nn_flat) if nn_flat else zero
    # 同分子 mask：batch 内相同 ik14（同分子不同谱）不得充当负例。
    # 否则 InfoNCE 把「本应同点」的同分子谱推远 → 检索 margin 被压缩（G5 λ_p=5 已证）。
    iks = np.array([b[4] for b in batch])
    same_mol = torch.from_numpy(iks[:, None] == iks[None, :])
    # 显式同分子正例：None 位置用自身 clean 谱占位（被 sm_mask 屏蔽，不进 loss）。
    sm_mask = torch.tensor([b[5] is not None for b in batch], dtype=torch.bool)
    sm_pos = torch.stack([b[5] if b[5] is not None else b[0] for b in batch])
    return anchors, pos, nc, nn, neg_ptr, same_mol, sm_pos, sm_mask


# --------------------------------------------------------------------------- #
# 解冻控制
# --------------------------------------------------------------------------- #
def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_last_layers(model, n: int):
    enc = model.backbone.transformer_encoder
    L = enc.n_layers
    n = max(0, min(n, L))
    # head 永远训练
    for p in model.head.parameters():
        p.requires_grad = True
    # last-N 层 att + ff
    for i in range(L - n, L):
        for p in enc.atts[i].parameters():
            p.requires_grad = True
        for p in enc.ffs[i].parameters():
            p.requires_grad = True
    # 对应 scales（layer i 用 scales[2i], scales[2i+1]）+ 末层 norm scales[-1]
    for i in range(L - n, L):
        for p in enc.scales[2 * i].parameters():
            p.requires_grad = True
        for p in enc.scales[2 * i + 1].parameters():
            p.requires_grad = True
    for p in enc.scales[-1].parameters():
        p.requires_grad = True


def last_layer_grad_norm(model, n: int) -> float:
    """Trainable component gradient norm; n=0 means projection head only."""
    enc = model.backbone.transformer_encoder
    L = enc.n_layers
    total = 0.0
    if n == 0:
        for p in model.head.parameters():
            if p.grad is not None:
                total += float(p.grad.float().norm())
        return total
    for i in range(L - n, L):
        for p in list(enc.atts[i].parameters()) + list(enc.ffs[i].parameters()):
            if p.grad is not None:
                total += float(p.grad.float().norm())
    return total


def last_layer_weight_norm(model, n: int) -> float:
    enc = model.backbone.transformer_encoder
    L = enc.n_layers
    total = 0.0
    if n == 0:
        return sum(float(p.detach().float().norm()) for p in model.head.parameters())
    for i in range(L - n, L):
        for p in list(enc.atts[i].parameters()) + list(enc.ffs[i].parameters()):
            total += float(p.detach().float().norm())
    return total


# --------------------------------------------------------------------------- #
# InfoNCE loss
# --------------------------------------------------------------------------- #
def infonce_forward(model, anchors, pos, nc, nn, neg_ptr, tau, same_mol, sm_pos=None, sm_mask=None,
                    inbatch_negative_mode="all", synthetic_positive_weight=1.0):
    """返回 loss 与 (pos_cos, neg_cos_mean)。全部嵌入先归一化（model.forward 已归一化）。

    same_mol: (B,B) bool，True=同分子（同 ik14）。in-batch 负例排除这些对，避免污染。
    sm_pos:   (B, n_peaks) 显式同分子正例谱（同完整 SMILES 的另一条谱）；None=未启用。
    sm_mask:  (B,) bool，True=该 anchor 有显式同分子正例（单谱分子为 False）。
              True 时把 sm_pos[i] 的嵌入作为额外正例（多正例 InfoNCE，主动拉近）。
    """
    a = model(anchors)        # (B, d)
    p = model(pos)            # (B, d)
    nc_emb = model(nc) if nc.shape[0] else torch.zeros(0, a.shape[1], device=a.device)
    nn_emb = model(nn) if nn.shape[0] else torch.zeros(0, a.shape[1], device=a.device)
    sm_emb = model(sm_pos) if sm_pos is not None else None  # (B, d)

    pos_logit = (a * p).sum(1) / tau                       # (B,)
    sim_aa = a @ a.T / tau                                 # (B, B)
    sim_ap = a @ p.T / tau                                 # (B, B)
    sim_anc = a @ nc_emb.T / tau                           # (B, Nc)
    sim_ann = a @ nn_emb.T / tau                           # (B, Nn)
    sm_logit = (a * sm_emb).sum(1) / tau if sm_emb is not None else None  # (B,)

    B = a.shape[0]
    arange = torch.arange(B, device=a.device)
    losses = []
    for i in range(B):
        pos_logits = []
        if synthetic_positive_weight > 0:
            # log(weight) implements a true relative multi-positive weight.
            pos_logits.append(pos_logit[i:i + 1] + float(np.log(synthetic_positive_weight)))
        if sm_mask is not None and sm_mask[i]:
            pos_logits.append(sm_logit[i:i + 1])           # 正例2：显式同分子谱（主动拉近）
        if not pos_logits:
            raise RuntimeError("InfoNCE item has no positive; enable a synthetic view or provide a real same-molecule positive")
        mask = (arange != i) & ~same_mol[i]                # 排除自身 + 同分子（负采样污染修复）
        # Strict-10ppm retrieval is a local task.  In hard_only mode we do
        # not use unrelated in-batch spectra as negatives; the explicit list
        # already supplies same-formula/same-adduct hard negatives.
        negs = [sim_aa[i, mask], sim_ap[i, mask]] if inbatch_negative_mode == "all" else []
        lo, hi = int(neg_ptr[i]), int(neg_ptr[i + 1])
        if hi > lo:
            negs.append(sim_anc[i, lo:hi])                 # 干净异构体
            negs.append(sim_ann[i, lo:hi])                 # 噪声异构体
        all_logits = torch.cat([*pos_logits, *negs])
        losses.append(-torch.logsumexp(torch.cat(pos_logits), dim=0) + torch.logsumexp(all_logits, dim=0))
    loss = torch.stack(losses).mean()
    return loss, a, p


def cpu_optimizer_state(optimizer):
    """返回 optimizer state 的 CPU **深拷贝**（绝不改 live optimizer 的 state）。

    关键坑：optimizer.state_dict() 里 state[...] 的 inner dict 是 live state 的**引用**，
    直接 `param_state[k] = v.cpu()` 会把 live optimizer 的 exp_avg/exp_avg_sq 原地换成
    CPU tensor，下一步 optimizer.step() 就报
    "Expected all tensors to be on the same device (cuda:0 vs cpu)"。
    CPU smoke 测不出（全是 CPU）；GPU 上 epoch 0 存完、epoch 1 第一步必炸。故必须重建 dict。
    """
    sd = optimizer.state_dict()
    out_state = {}
    for param_id, ps in sd["state"].items():
        out_state[param_id] = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in ps.items()
        }
    return {"state": out_state, "param_groups": sd["param_groups"]}


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    """原子写 checkpoint（先 .tmp 再 rename）；写失败=磁盘/配额问题，立即抛出让 job 停下。

    理由：G3 首跑因 0.2T 配额耗尽，.out 写不进、结束时 checkpoint 也存不下，白烧 6 小时。
    此函数 + per-epoch last.pt 保证：只要配额还够写一个文件，进度就不会全丢；
    写不进去就当场报错，而不是沉默跑到结尾才失败。
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(checkpoint, tmp)
        os.replace(tmp, path)
    except OSError as e:
        print(f"[save] FAILED 写入 {path}: {e}", flush=True)
        print("[save] 大概率是磁盘配额/空间不足。先清空间，再用 --resume 续训（进度已存于上个 epoch 的 last.pt）。", flush=True)
        raise


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    train_entries = manifest["train"]
    if args.smoke:
        train_entries = train_entries[:200]
        args.epochs = 1
    elif args.train_subset is not None:
        subset = json.loads(args.train_subset.read_text(encoding="utf-8"))
        train_entries = subset["entries"]
        if not train_entries:
            raise ValueError(f"Empty locked train subset: {args.train_subset}")
        print(f"[data] using locked representative subset: {args.train_subset}", flush=True)
    elif args.max_anchors > 0:
        raise ValueError(
            "--max-anchors uses a sorted manifest prefix and is prohibited for a real pilot. "
            "First run tasks/audit_noise_training_pool.py, then pass --train-subset its locked JSON."
        )
    if (args.real_positive_weight > 0 or args.positive_relation_preserve_lambda > 0) and not args.same_mol_as_positive:
        raise ValueError("real-positive losses require --same-mol-as-positive")
    print(f"[data] train anchors: {len(train_entries)}", flush=True)

    # 载入 precursor_mz + smiles（全量，一次读进内存）。
    # smiles 用于显式同分子正例判据（含立体标记，区分立体异构体；ik14 只区分连通性会混并）。
    import h5py
    with h5py.File(args.data, "r") as f:
        precursor_mz_all = np.array(f["precursor_mz"][:], dtype=float)
        smiles_all = np.array([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                               for x in f["smiles"][:]])
        instrument_all = np.array([x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
                                   for x in f["INSTRUMENT_TYPE"][:]])
        collision_energy_all = np.asarray(f["COLLISION_ENERGY"][:], dtype=float)

    noise_cfg = NoiseConfig(
        protect_above_relative_intensity=(args.protect_main_peaks_above if args.protect_main_peaks_above > 0 else None),
        do_add=not args.disable_random_added_peaks,
    )
    dataset = NoiseContrastiveDataset(
        train_entries, args.data, precursor_mz_all, args.n_highest_peaks, noise_cfg, args.seed,
        smiles_all=smiles_all if args.same_mol_as_positive else None,
        instrument_all=instrument_all if args.same_mol_as_positive else None,
        collision_energy_all=collision_energy_all if args.same_mol_as_positive else None,
        cross_condition_positive=args.cross_condition_positive,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate, pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    model, kind = load_base_model(
        args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks
    )
    freeze_all(model)
    unfreeze_last_layers(model, args.unfreeze_layers)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] init kind={kind}; trainable params: {sum(p.numel() for p in trainable):,}", flush=True)
    if not trainable:
        raise RuntimeError("没有可训练参数（解冻失败）")

    # 教师 = 官方冻结表示（L_preserve 锚点）；student 漂走时把它拉回官方嵌入，防检索退化
    teacher = None
    if args.preserve_lambda > 0 or args.positive_relation_preserve_lambda > 0:
        teacher, _ = load_base_model(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks)
        freeze_all(teacher)
        teacher.eval()
        print(f"[model] frozen teacher loaded: feature λ_p={args.preserve_lambda}, "
              f"positive-relation λ={args.positive_relation_preserve_lambda}", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_path = run_dir / "last.pt"

    model.train()
    w0 = last_layer_weight_norm(model, args.unfreeze_layers)
    losses = []
    grad_flow = 0.0
    nan_detected = False
    elapsed_before = 0.0
    start_epoch = 0

    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location="cpu")
        model.backbone.load_state_dict(state["backbone_state_dict"])
        model.head.load_state_dict(state["head_state_dict"])
        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        except Exception as e:  # optimizer state 非关键，失败也继续
            print(f"[resume] optimizer state 加载失败（改用全新 optimizer，不影响模型权重）: {e}", flush=True)
        start_epoch = int(state.get("epoch_completed", -1)) + 1
        losses = list(state.get("losses", []))
        grad_flow = float(state.get("grad_flow", 0.0))
        w0 = float(state.get("w0", w0))
        nan_detected = bool(state.get("nan_detected", False))
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
        print(f"[resume] 从 {resume_path} 恢复：已完成 {start_epoch}/{args.epochs} epoch，"
              f"loss 记录数={len(losses)}，累计 elapsed={elapsed_before:.0f}s", flush=True)
    else:
        print(f"[resume] 无 last.pt（或 --no-resume），从 epoch 0 全新开始", flush=True)

    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        for step, (anchors, pos, nc, nn, neg_ptr, same_mol, sm_pos, sm_mask) in enumerate(loader):
            if args.max_steps and step >= args.max_steps:
                break
            anchors = anchors.to(device)
            pos = pos.to(device)
            nc = nc.to(device)
            nn = nn.to(device)
            neg_ptr = neg_ptr.to(device)
            same_mol = same_mol.to(device)
            sm_pos = sm_pos.to(device)
            sm_mask = sm_mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss, a_emb, _ = infonce_forward(
                    model, anchors, pos, nc, nn, neg_ptr, args.tau, same_mol,
                    sm_pos if args.same_mol_as_positive else None,
                    sm_mask if args.same_mol_as_positive else None,
                    args.inbatch_negative_mode,
                    args.synthetic_positive_weight,
                )
                # A log-sum-exp multi-positive numerator can be dominated by
                # the easy synthetic view.  The explicit term below gives the
                # audited real cross-condition pair a guaranteed gradient.
                real_positive_loss = a_emb.new_zeros(())
                real_positive_cos = None
                if args.same_mol_as_positive and sm_mask.any() and (
                    args.real_positive_weight > 0 or args.positive_relation_preserve_lambda > 0
                ):
                    sm_emb_direct = model(sm_pos)
                    real_positive_cos = (a_emb[sm_mask] * sm_emb_direct[sm_mask]).sum(1)
                    real_positive_loss = (1.0 - real_positive_cos).mean()
                    loss = loss + args.real_positive_weight * real_positive_loss
                if teacher is not None:
                    with torch.no_grad():  # no_grad 而非 inference_mode：z_off 要作为常数进 loss 反传
                        z_off = teacher(anchors)  # (B, d) 官方冻结嵌入（已 L2 归一化）
                    preserve = (1.0 - (a_emb * z_off).sum(1)).mean()
                    loss = loss + args.preserve_lambda * preserve
                    if args.positive_relation_preserve_lambda > 0 and real_positive_cos is not None:
                        with torch.no_grad():
                            z_off_pos = teacher(sm_pos)
                            official_positive_cos = (z_off[sm_mask] * z_off_pos[sm_mask]).sum(1)
                        relation = F.relu(official_positive_cos - real_positive_cos).mean()
                        loss = loss + args.positive_relation_preserve_lambda * relation
            scaler.scale(loss).backward()

            if epoch == 0 and step == 0:
                grad_flow = last_layer_grad_norm(model, args.unfreeze_layers)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            li = float(loss.detach())
            if not np.isfinite(li):
                nan_detected = True
            losses.append(li)
            if step % 10 == 0 or step == 0:
                print(f"  epoch {epoch} step {step:4d} loss={li:.4f} "
                      f"grad_flow={grad_flow:.4f}", flush=True)

        # ---- per-epoch 断点续训 checkpoint（写完即续训点；写失败=配额/磁盘问题，立即停） ----
        ckpt = {
            "format": "noise_isomer_infonce_v1",
            "seed": args.seed,
            "epoch_completed": epoch,
            "w0": w0,
            "grad_flow": grad_flow,
            "losses": losses,
            "nan_detected": nan_detected,
            "elapsed_seconds": elapsed_before + (time.time() - t0),
            "backbone_state_dict": cpu_state_dict(model.backbone),
            "head_state_dict": cpu_state_dict(model.head),
            "optimizer_state_dict": cpu_optimizer_state(optimizer),
            "config": {"n_highest_peaks": args.n_highest_peaks,
                       "unfreeze_layers": args.unfreeze_layers,
                       "tau": args.tau, "lr": args.lr, "batch_size": args.batch_size,
                       "preserve_lambda": args.preserve_lambda,
                       "same_mol_as_positive": args.same_mol_as_positive,
                       "cross_condition_positive": args.cross_condition_positive,
                       "inbatch_negative_mode": args.inbatch_negative_mode,
                       "real_positive_weight": args.real_positive_weight,
                       "synthetic_positive_weight": args.synthetic_positive_weight,
                       "positive_relation_preserve_lambda": args.positive_relation_preserve_lambda,
                       "protect_main_peaks_above": args.protect_main_peaks_above,
                       "disable_random_added_peaks": args.disable_random_added_peaks,
                       "train_subset": str(args.train_subset) if args.train_subset else None},
        }
        save_checkpoint(resume_path, ckpt)
        print(f"[save] epoch {epoch} 完成 -> {resume_path.name} "
              f"({resume_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    w1 = last_layer_weight_norm(model, args.unfreeze_layers)
    loss_first = losses[0] if losses else float("nan")
    loss_last = losses[-1] if losses else float("nan")

    checks = {
        "seed": args.seed,
        "kind": kind,
        "n_train_anchors": len(train_entries),
        "unfreeze_layers": args.unfreeze_layers,
        "tau": args.tau, "lr": args.lr, "batch_size": args.batch_size,
        "preserve_lambda": args.preserve_lambda,
        "same_mol_as_positive": args.same_mol_as_positive,
        "cross_condition_positive": args.cross_condition_positive,
        "inbatch_negative_mode": args.inbatch_negative_mode,
        "real_positive_weight": args.real_positive_weight,
        "synthetic_positive_weight": args.synthetic_positive_weight,
        "positive_relation_preserve_lambda": args.positive_relation_preserve_lambda,
        "protect_main_peaks_above": args.protect_main_peaks_above,
        "disable_random_added_peaks": args.disable_random_added_peaks,
        "train_subset": str(args.train_subset) if args.train_subset else None,
        "grad_flow_last_layers": grad_flow,
        "gate1_unfreeze_ok": bool(grad_flow > 0),
        "loss_first": loss_first, "loss_last": loss_last,
        "gate2_loss_decreased": bool(np.isfinite(loss_first) and np.isfinite(loss_last) and loss_last < loss_first),
        "nan_detected": nan_detected,
        "gate3_no_nan": bool(not nan_detected),
        "weight_change_last_layer": float(abs(w1 - w0)),
        "gate_weight_moved": bool(abs(w1 - w0) > 1e-6),
        "elapsed_seconds": elapsed_before + (time.time() - t0),
        "resumed_from_epoch": start_epoch,
    }
    gate = checks["gate1_unfreeze_ok"] and checks["gate2_loss_decreased"] and checks["gate3_no_nan"]
    checks["G2_smoke_pass"] = bool(gate)

    (run_dir / "summary.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存最终 checkpoint（供 G3/G4 eval 加载；smoke 也存，便于检查）
    checkpoint = {
        "format": "noise_isomer_infonce_v1",
        "seed": args.seed,
        "architecture_checkpoint": str(args.architecture_ckpt.resolve()),
        "base_checkpoint": str(args.base_ckpt.resolve()),
        "backbone_state_dict": cpu_state_dict(model.backbone),
        "head_state_dict": cpu_state_dict(model.head),
        "config": {"n_highest_peaks": args.n_highest_peaks, "unfreeze_layers": args.unfreeze_layers,
                   "tau": args.tau, "lr": args.lr, "batch_size": args.batch_size,
                   "preserve_lambda": args.preserve_lambda,
                   "same_mol_as_positive": args.same_mol_as_positive,
                   "cross_condition_positive": args.cross_condition_positive,
                   "inbatch_negative_mode": args.inbatch_negative_mode,
                   "real_positive_weight": args.real_positive_weight,
                   "synthetic_positive_weight": args.synthetic_positive_weight,
                   "positive_relation_preserve_lambda": args.positive_relation_preserve_lambda,
                   "protect_main_peaks_above": args.protect_main_peaks_above,
                   "disable_random_added_peaks": args.disable_random_added_peaks,
                   "train_subset": str(args.train_subset) if args.train_subset else None},
        "checks": checks,
    }
    save_checkpoint(run_dir / "best_infonce.pt", checkpoint)

    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    print(f"\n=== G2 smoke: {'PASS' if gate else 'FAIL'} ===", flush=True)
    print(f"  gate1 解冻生效(grad>0): {checks['gate1_unfreeze_ok']}  (grad_flow={grad_flow:.4f})")
    print(f"  gate2 loss 下降: {checks['gate2_loss_decreased']}  ({loss_first:.4f} -> {loss_last:.4f})")
    print(f"  gate3 无 NaN: {checks['gate3_no_nan']}")
    del model
    gc.collect()


if __name__ == "__main__":
    main()
