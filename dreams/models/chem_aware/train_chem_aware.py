"""
化学感知 DreaMS 轻量微调脚本 [阶段二]

功能：
  冻结预训练 DreaMS backbone → 仅训练化学感知模块（A: GateNetwork + B: λ）
  通过 mask prediction 任务驱动 λ 和门控权重的学习。

用法：
  # 服务器：用 MoNA/NIST20 标注数据微调
  python -m dreams.models.chem_aware.train_chem_aware \
      --dataset_path /path/to/mona.hdf5 \
      --ckpt_path /path/to/ssl_model.ckpt \
      --epochs 5 --batch_size 16

  # 本地：用小示例数据测试流程
  python -m dreams.models.chem_aware.train_chem_aware --dry_run

核心设计（阶段二 vs 阶段三）：
  阶段二（当前）：λ = attenuation * attenuation_scale（单标量可学习参数）
                   梯度来自 mask loss → λ 向最优值收敛
  阶段三（未来）：λ = LambdaGenerator(state_vector)（动态 MLP 输出）
                   梯度来自 mask loss + 对抗 loss → 条件自适应 λ

作者：module1-chem-attn 开发分支
"""

import torch
import torch.nn as nn
import argparse
import sys
from pathlib import Path
import numpy as np
from typing import Optional, Dict, List

import dreams.utils.data as du
import dreams.utils.dformats as dformats
from dreams.models.dreams.dreams import DreaMS
from dreams.models.chem_aware.chem_aware_dreams import ChemAwareDreaMS
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine, NEUTRAL_LOSSES as NL_DICT
from dreams.models.chem_aware.gating import HeadGatingNetwork, StateExtractor, LambdaGenerator
from dreams.models.chem_aware.losses import (
    attention_entropy_loss,
    lambda_regularization_loss,
)
from dreams.definitions import PRETRAINED


def parse_args():
    p = argparse.ArgumentParser(description='ChemAwareDreaMS lightweight fine-tuning')
    p.add_argument('--dataset_path', type=str, default=None,
                   help='Path to HDF5 training dataset (MoNA/NIST20)')
    p.add_argument('--ckpt_path', type=str, default=None,
                   help='Path to ssl_model.ckpt')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--entropy_weight', type=float, default=0.01,
                   help='Weight for attention entropy regularization')
    p.add_argument('--lambda_min', type=float, default=0.05,
                   help='Minimum λ to prevent collapse')
    p.add_argument('--dry_run', action='store_true',
                   help='Use synthetic data for local testing')
    p.add_argument('--save_dir', type=str, default='./chem_aware_checkpoints')
    return p.parse_args()


# ==============================================================================
# 从预训练 DreaMS 构建 ChemAwareDreaMS（参数克隆）
# ==============================================================================

def build_chem_aware_from_pretrained(pretrained_model: DreaMS) -> ChemAwareDreaMS:
    """从预训练 DreaMS 克隆 backbone 参数到 ChemAwareDreaMS"""
    from argparse import Namespace

    # 从 hparams 或模型属性重建 args
    old_args = pretrained_model.hparams.get('args', None) if hasattr(pretrained_model, 'hparams') else None
    if old_args is None:
        old_args = Namespace()

    # 确保必要属性存在
    defaults = {
        'scnorm': False, 'pre_norm': True, 'residual_dropout': 0.1,
        'att_dropout': 0.1, 'ff_dropout': 0.1, 'dropout': 0.1,
        'attn_mech': 'dot-product', 'no_transformer_bias': False,
        'fourier_num_freqs': 512, 'fourier_trainable': False,
        'fourier_min_freq': None,
        'no_ffs_bias': False, 'hot_mz_bin_size': 1.0,
        'train_objective': 'mask_peak_hot', 'lr': 1e-4, 'weight_decay': 0.0,
        'batch_size': 32, 'n_warmup_steps': 0,
        'ret_order_loss_w': 0.0, 'cos_reg_alpha': 0.0, 'cos_reg_reduction': 'mean',
        'entropy_label_smoothing': 0.0, 'mask_val': 0.0,
        'fourier_strategy': 'lin_float_int',
        'gains_dir': Path('.'), 'log_figs': False,
    }
    for attr, default in defaults.items():
        if not hasattr(old_args, attr):
            setattr(old_args, attr, default)

    # 从模型实例获取关键结构参数
    for attr in ['dformat', 'n_layers', 'n_heads', 'd_model', 'd_fourier', 'd_peak',
                 'd_mz_token', 'charge_feature', 'graphormer_mz_diffs',
                 'graphormer_parametrized', 'vanilla_transformer',
                 'ff_peak_depth', 'ff_fourier_depth', 'ff_out_depth']:
        if hasattr(pretrained_model, attr):
            setattr(old_args, attr, getattr(pretrained_model, attr))

    if hasattr(pretrained_model, 'spec_preproc') \
            and not hasattr(old_args, 'max_peaks_n'):
        old_args.max_peaks_n = pretrained_model.spec_preproc.n_highest_peaks

    # 化学感知参数
    old_args.chem_attn = True
    old_args.chem_attn_attenuation = -5.0
    old_args.chem_attn_tolerance = 0.02
    old_args.chem_attn_entropy_w = 0.01

    # 构建
    chem_model = ChemAwareDreaMS(old_args, pretrained_model.spec_preproc)

    # 复制共享参数
    pretrained_state = pretrained_model.state_dict()
    chem_state = chem_model.state_dict()
    transferred = 0
    for key in chem_state:
        if key in pretrained_state and chem_state[key].shape == pretrained_state[key].shape:
            chem_state[key] = pretrained_state[key].clone()
            transferred += 1

    chem_model.load_state_dict(chem_state, strict=False)
    print(f'   Weight transfer: {transferred}/{len(chem_state)} parameters matched')

    return chem_model


# ==============================================================================
# 轻量微调循环
# ==============================================================================

def train_chem_aware(
    model: ChemAwareDreaMS,
    dataloader,
    epochs: int = 5,
    lr: float = 1e-4,
    entropy_weight: float = 0.01,
    lambda_min: float = 0.05,
    save_dir: Path = Path('./chem_aware_checkpoints'),
    device: torch.device = torch.device('cpu')
):
    """
    轻量微调：冻结 backbone，仅训练化学感知模块

    训练循环每步：
      1. 前向传播 → chem_bias（含 λ）+ gate_weights 参与 Transformer
      2. 计算 L = L_mask + β_ent * L_entropy + β_λ * L_lambda
      3. 只反向传播 chem_aware 模块参数
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- 冻结/解冻 ----
    n_frozen = model.freeze_backbone()
    n_trainable = model.unfreeze_chem_aware()
    print(f'   Frozen backbone params: {n_frozen}')
    print(f'   Trainable chem_aware params: {n_trainable}')

    model = model.to(device)
    model.train()

    # ---- 优化器（仅 chem_aware 参数） ----
    chem_params = model.get_chem_aware_params()
    optimizer = torch.optim.Adam(chem_params, lr=lr)
    print(f'   Optimizer params: {sum(p.numel() for p in chem_params):,}')

    # ---- 日志 ----
    history = {
        'epoch': [], 'step': [], 'loss_mask': [], 'loss_entropy': [],
        'loss_lambda': [], 'loss_total': [],
        'lambda_val': [], 'gate_mean': [], 'gate_std': [],
    }

    global_step = 0

    for epoch in range(epochs):
        epoch_losses = []
        epoch_entropies = []
        epoch_lambdas = []

        for batch_idx, batch in enumerate(dataloader):
            # 数据准备 — 兼容 dict 和 tuple 两种格式
            if isinstance(batch, dict):
                spec = batch.get('spectrum', batch.get('spec_mask', None))
                spec_real = batch.get('spec_real', None)
                mask = batch.get('mask', None)
                charge = batch.get('charge', None)
            elif isinstance(batch, (tuple, list)):
                spec = batch[0]  # TensorDataset 返回 (tensor,)
                spec_real = None
                mask = None
                charge = None
            else:
                spec = batch
                spec_real = None
                mask = None
                charge = None

            spec = spec.to(device) if isinstance(spec, torch.Tensor) else torch.as_tensor(spec, device=device)
            if charge is not None and isinstance(charge, torch.Tensor):
                charge = charge.to(device)

            # ---- 前向传播 ----
            if spec_real is not None and mask is not None:
                spec_real = spec_real.to(device)
                mask = mask.to(device)
                loss, embs, pred_mz, real_mz = model.spec_ssl_step(
                    spec, spec_real, mask, charge
                )
                loss_mask = loss.sum() / loss.numel()
            else:
                # 无 mask 字段：直接 forward
                embs = model(spec, charge)
                loss_mask = torch.tensor(0.0, device=device, requires_grad=True)

            # ---- 化学感知辅助损失 ----
            # 注意力熵正则化
            loss_entropy = torch.tensor(0.0, device=device)
            if model._last_chem_analysis is not None:
                # 从缓存中提取注意力矩阵
                # 注意：需要从 model 内部获取注意力权重
                # 目前用桩实现（后续从 hook 获取）
                pass

            # λ 正则化：防止 λ 塌缩到 0
            lambda_val = model.chem_rule_engine._effective_attenuation() \
                if model.chem_rule_engine is not None else 0.0
            abs_lambda = abs(lambda_val) if isinstance(lambda_val, (int, float)) \
                else abs(lambda_val.item())
            loss_lambda = torch.tensor(
                max(0.0, lambda_min - abs_lambda / abs(model.chem_attn_attenuation)),
                device=device
            )

            # ---- 总损失 ----
            loss_total = loss_mask + entropy_weight * loss_entropy + 0.001 * loss_lambda

            # ---- 反向传播 ----
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # ---- 记录 ----
            gate_mean = 0.0
            gate_std = 0.0
            if model.gate_network is not None:
                # 获取最近一次的门控权重
                analysis = model.get_chem_attn_analysis()
                if analysis and analysis.get('gate_weights') is not None:
                    gw = analysis['gate_weights']
                    gate_mean = gw.mean().item()
                    gate_std = gw.std().item()

            epoch_losses.append(loss_mask.item())
            epoch_lambdas.append(abs_lambda)

            history['step'].append(global_step)
            history['loss_mask'].append(loss_mask.item())
            history['loss_lambda'].append(loss_lambda.item())
            history['loss_total'].append(loss_total.item())
            history['lambda_val'].append(abs_lambda)
            history['gate_mean'].append(gate_mean)
            history['gate_std'].append(gate_std)

            global_step += 1

            if batch_idx % 10 == 0:
                print(f'   Epoch {epoch+1}/{epochs} | Step {batch_idx} | '
                      f'mask_loss={loss_mask.item():.4f} | '
                      f'lambda={abs_lambda:.2f} | gate_std={gate_std:.3f}')

        # epoch summary
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_lambda = np.mean(epoch_lambdas) if epoch_lambdas else 0.0
        history['epoch'].append(epoch)
        print(f'\n--- Epoch {epoch+1}/{epochs} Summary ---')
        print(f'   Avg mask loss: {avg_loss:.4f}')
        print(f'   Avg |lambda|:   {avg_lambda:.2f}')
        print(f'   Lambda scale:  {model.chem_rule_engine.attenuation_scale.item():.4f}')

        # 保存 checkpoint
        ckpt_path = save_dir / f'chem_aware_epoch{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        }, ckpt_path)
        print(f'   Checkpoint saved: {ckpt_path}\n')

    return history


# ==============================================================================
# 主入口
# ==============================================================================

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- 加载预训练 DreaMS ----
    ckpt_path = args.ckpt_path or (PRETRAINED / 'ssl_model.ckpt')
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    print(f'Loading pretrained model: {ckpt_path}')
    pretrained = DreaMS.load_from_checkpoint(ckpt_path, map_location=device)
    pretrained.eval()

    # ---- 构建 ChemAwareDreaMS ----
    print('Building ChemAwareDreaMS...')
    model = build_chem_aware_from_pretrained(pretrained)
    print(f'   chem_attn_enabled: {model.chem_attn_enabled}')
    print(f'   Initial lambda: {model.chem_rule_engine._effective_attenuation():.2f}')

    # ---- 准备数据 ----
    if args.dry_run or args.dataset_path is None:
        print('\n*** DRY RUN: Creating synthetic dataset for testing ***')
        from torch.utils.data import DataLoader, TensorDataset
        # 创建 20 张伪谱图（每张 30 个峰），格式 (n, 2) = [mz, intensity]
        dummy_specs = []
        for _ in range(20):
            mz = torch.rand(30) * 1000.0
            mz = mz.sort().values
            intens = torch.rand(30)
            intens = intens / intens.max()
            spec = torch.stack([mz, intens], dim=-1)  # (30, 2)
            dummy_specs.append(spec)
        # Pad to same length
        max_len = max(s.shape[0] for s in dummy_specs)
        specs_padded = torch.zeros(20, max_len, 2)
        for i, s in enumerate(dummy_specs):
            specs_padded[i, :s.shape[0]] = s
        dummy_dataset = TensorDataset(specs_padded)
        dataloader = DataLoader(dummy_dataset, batch_size=args.batch_size, shuffle=True)
        print(f'   Synthetic dataset: {len(dummy_dataset)} spectra, {max_len} peaks each')
        # Dry run: 只做 1 个 epoch 的 5 步
        actual_epochs = 1
    else:
        print(f'\nLoading dataset: {args.dataset_path}')
        # 使用 DreaMS 的 MaskedSpectraDataset
        from dreams.training.train_argparse import parse_args as train_parse
        # 注：此处需根据实际数据格式调整
        # 简化为直接加载 MSData + to_torch_dataset
        msdata = du.MSData.load(args.dataset_path)
        spec_preproc = model.spec_preproc
        dataset = msdata.to_torch_dataset(spec_preproc)
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print(f'   Dataset: {len(dataset)} spectra')
        actual_epochs = args.epochs

    # ---- 训练 ----
    print(f'\nStarting fine-tuning ({actual_epochs} epochs)...')
    print('=' * 60)
    history = train_chem_aware(
        model=model,
        dataloader=dataloader,
        epochs=actual_epochs,
        lr=args.lr,
        entropy_weight=args.entropy_weight,
        lambda_min=args.lambda_min,
        save_dir=Path(args.save_dir),
        device=device,
    )

    print('=' * 60)
    print('Fine-tuning complete!')
    print(f'   Initial lambda: {history["lambda_val"][0]:.2f}')
    print(f'   Final lambda:   {history["lambda_val"][-1]:.2f}')
    print(f'   Checkpoints saved in: {args.save_dir}')


if __name__ == '__main__':
    main()
