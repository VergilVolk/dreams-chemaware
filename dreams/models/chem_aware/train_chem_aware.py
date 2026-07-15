"""
化学感知 DreaMS 轻量微调脚本 [v3 简化版]

功能：
  冻结预训练 DreaMS backbone → 仅训练化学规则引擎的 5 维权重向量
  通过 mask prediction 任务驱动各规则权重的学习。

核心改动（v2 → v3）：
  - 移除 LambdaController / 课程调度 / 协同损失
  - 化学偏置仅注入最后一层
  - 训练仅最小化 mask prediction loss
  - 规则权重通过梯度自然选择：有用的 ↑，无用的 ↓

用法：
  # 服务器：用 MoNA/NIST20 标注数据微调
  python -m dreams.models.chem_aware.train_chem_aware \
      --dataset_path /path/to/mona.hdf5 \
      --ckpt_path /path/to/ssl_model.ckpt \
      --epochs 5 --batch_size 16

  # 本地：用小示例数据测试流程
  python -m dreams.models.chem_aware.train_chem_aware --dry_run

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
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from dreams.definitions import PRETRAINED


def parse_args():
    p = argparse.ArgumentParser(description='ChemAwareDreaMS v3 lightweight fine-tuning')
    p.add_argument('--dataset_path', type=str, default=None,
                   help='Path to HDF5 training dataset (MoNA/NIST20)')
    p.add_argument('--ckpt_path', type=str, default=None,
                   help='Path to ssl_model.ckpt')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--chem_attn_layer', type=int, default=-1,
                   help='Layer index to inject chem bias (-1 = last layer)')
    p.add_argument('--dry_run', action='store_true',
                   help='Use synthetic data for local testing')
    p.add_argument('--save_dir', type=str, default='./chem_aware_checkpoints_v3')
    p.add_argument('--output_dir', type=str, default=None,
                   help='Alias for --save_dir')
    p.add_argument('--num_devices', type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument('--accelerator', type=str, default='gpu', help=argparse.SUPPRESS)
    return p.parse_args()


# ==============================================================================
# 从预训练 DreaMS 构建 ChemAwareDreaMS（参数克隆）
# ==============================================================================

def build_chem_aware_from_pretrained(pretrained_model: DreaMS, chem_attn_layer: int = -1) -> ChemAwareDreaMS:
    """从预训练 DreaMS 克隆 backbone 参数到 ChemAwareDreaMS [v3]"""
    from argparse import Namespace

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

    # ---- [v3] 化学感知参数 ----
    old_args.chem_attn = True
    old_args.chem_attn_tolerance = 0.02
    old_args.chem_attn_layer = chem_attn_layer

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
# 轻量微调循环 [v3 简化版]
# ==============================================================================

def train_chem_aware(
    model: ChemAwareDreaMS,
    dataloader,
    epochs: int = 5,
    lr: float = 1e-4,
    save_dir: Path = Path('./chem_aware_checkpoints_v3'),
    device: torch.device = torch.device('cpu')
):
    """
    轻量微调 [v3]：冻结 backbone，仅训练化学规则引擎的逐规则权重向量

    每步：
      1. 前向传播（化学偏置自动注入最后一层）
      2. 计算 L_mask（掩码峰预测损失）
      3. 反向传播 → 仅更新 rule_weights_raw

    规则权重通过梯度自然选择：
      - 对 mask prediction 有帮助的规则 → 权重 ↑
      - 对 mask prediction 无帮助的规则 → 权重 ↓（自动边缘化，不拖累好规则）
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

    # ---- 优化器（仅规则权重） ----
    chem_params = model.get_chem_aware_params()
    optimizer = torch.optim.Adam(chem_params, lr=lr)
    n_params = sum(p.numel() for p in chem_params)
    n_rules = n_params  # 每个规则一个权重
    print(f'   Optimizer params: {n_params} ({n_rules} rule weights)')

    # ---- 日志 ----
    rule_names = model.chem_rule_engine.get_rule_names()  # 逐规则名称列表
    history = {
        'epoch': [], 'step': [], 'loss_mask': [],
        'rule_weights': [],  # List of (n_rules,) tensors per step
    }

    global_step = 0

    for epoch in range(epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(dataloader):
            # ---- 数据准备 ----
            if isinstance(batch, dict):
                spec = batch.get('spectrum', batch.get('spec_mask', None))
                spec_real = batch.get('spec_real', None)
                mask = batch.get('mask', None)
                charge = batch.get('charge', None)
            elif isinstance(batch, (tuple, list)):
                spec = batch[0]
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

            # ---- 动态 mask 生成（如无预置 mask） ----
            if not (isinstance(spec_real, torch.Tensor) and isinstance(mask, torch.Tensor)):
                bs, n_peaks = spec.shape[0], spec.shape[1]
                n_mask = max(1, int(n_peaks * 0.15))
                mask_bool = torch.zeros(bs, n_peaks, dtype=torch.bool, device=device)
                spec_mask = spec.clone()
                for b in range(bs):
                    idx = torch.randperm(n_peaks - 1, device=device)[:n_mask] + 1
                    mask_bool[b, idx] = True
                    spec_mask[b, idx, :] = 0.0
                spec_real = spec
                spec = spec_mask
                mask = mask_bool

            if isinstance(spec_real, torch.Tensor) and isinstance(mask, torch.Tensor):
                spec_real = spec_real.to(device)
                mask = mask.to(device)

            # ---- [有限差分测试] step0 验证 rule_weights 对 loss 是否有因果影响 ----
            if global_step == 0:
                print('[FINITE DIFF] Testing if rule_weights affect loss...')
                with torch.no_grad():
                    orig_raw = model.chem_rule_engine.rule_weights_raw.clone()
                    orig_w = model.chem_rule_engine.get_rule_weights().clone()

                    # 权重→0（softplus(-100)≈0，即无化学偏置）
                    model.chem_rule_engine.rule_weights_raw.fill_(-100.0)
                    loss0 = model.spec_ssl_step(spec, spec_real, mask, charge)[0].sum().item()

                    # 权重→大值（softplus(10)≈10，即强化化学偏置）
                    model.chem_rule_engine.rule_weights_raw.fill_(10.0)
                    loss10 = model.spec_ssl_step(spec, spec_real, mask, charge)[0].sum().item()

                    # 恢复
                    model.chem_rule_engine.rule_weights_raw.copy_(orig_raw)

                diff = loss10 - loss0
                print(f'[FINITE DIFF] loss(weights→0)={loss0:.6f}  loss(weights→10)={loss10:.6f}  diff={diff:.6f}')
                if abs(diff) < 1e-5:
                    print('[FINITE DIFF] *** FAIL: rule_weights 对 loss 无因果影响! chem_bias 未到达 loss! ***')
                else:
                    print(f'[FINITE DIFF] PASS: rule_weights 影响 loss，diff={diff:.6f}，梯度应能回传')
                print()

            # ---- 前向传播 ----
            loss, embs, pred_mz, real_mz = model.spec_ssl_step(
                spec, spec_real, mask, charge
            )
            loss_mask = loss.sum() / loss.numel()

            # ---- 反向传播 ----
            optimizer.zero_grad()
            loss_mask.backward()

            # [DEBUG] 诊断梯度是否到达 rule_weights_raw
            if global_step < 5:
                rw_param = model.chem_rule_engine.rule_weights_raw
                scale_param = model.chem_residual_scale
                print(f'[DEBUG step {global_step}] rule_weights_raw.grad is None: {rw_param.grad is None}')
                print(f'[DEBUG step {global_step}] chem_residual_scale={scale_param.item():.4f}')
                # 诊断：逐级检查中间张量梯度
                ctx = getattr(model, '_diag_chem_context', None)
                cw2 = getattr(model, '_diag_cw2', None)
                if cw2 is not None:
                    g2 = cw2.grad
                    print(f'[DEBUG step {global_step}] cw2.grad is None: {g2 is None}')
                    if g2 is not None:
                        print(f'[DEBUG step {global_step}] cw2.grad norm: {g2.norm().item():.6f}')
                    else:
                        print(f'[DEBUG step {global_step}] *** cw2 has NO grad — 断在 bmm→cw2 ***')
                if ctx is not None:
                    gctx = ctx.grad
                    print(f'[DEBUG step {global_step}] chem_context.grad norm: {gctx.norm().item() if gctx is not None else 0:.6f}')
                if rw_param.grad is not None:
                    print(f'[DEBUG step {global_step}] grad norm: {rw_param.grad.norm().item():.6f}')
                    print(f'[DEBUG step {global_step}] grad abs sum: {rw_param.grad.abs().sum().item():.6f}')
                    print(f'[DEBUG step {global_step}] scale grad: {scale_param.grad.item() if scale_param.grad is not None else 0:.6f}')
                    # 打印前5个有梯度的规则
                    grad_abs = rw_param.grad.abs()
                    top5 = grad_abs.topk(min(5, len(grad_abs)))
                    for idx, g in zip(top5.indices, top5.values):
                        rn = rule_names[idx]
                        print(f'[DEBUG step {global_step}]   {rn}: grad={g.item():.8f}')
                else:
                    print(f'[DEBUG step {global_step}] *** GRAD IS NONE — 计算图断裂! ***')
                # 同时检查 chem_bias 的统计
                analysis = model.get_chem_attn_analysis()
                if analysis:
                    cb = analysis['chem_bias']
                    print(f'[DEBUG step {global_step}] chem_bias requires_grad: {cb.requires_grad}')
                    print(f'[DEBUG step {global_step}] chem_bias sum: {cb.sum().item():.4f}, '
                          f'min={cb.min().item():.4f}, max={cb.max().item():.4f}')
                    print(f'[DEBUG step {global_step}] chem_bias.grad_fn: {cb.grad_fn}')
                # 规则匹配统计
                stats = model.chem_rule_engine.get_rule_stats()
                print(f'[DEBUG step {global_step}] rule stats: {stats}')
                print()

            optimizer.step()

            # ---- 记录 ----
            analysis = model.get_chem_attn_analysis()
            rw = analysis['rule_weights'].clone() if analysis else torch.zeros(n_rules)

            epoch_losses.append(loss_mask.item())
            history['step'].append(global_step)
            history['loss_mask'].append(loss_mask.item())
            history['rule_weights'].append(rw)

            global_step += 1

            # 每 5000 步保存 checkpoint
            if batch_idx > 0 and batch_idx % 5000 == 0:
                ckpt_path = save_dir / f'chem_aware_v3_step{global_step}.pt'
                torch.save({
                    'epoch': epoch, 'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                }, ckpt_path)
                print(f'   [Checkpoint] Step {global_step} saved to {ckpt_path}')

            if batch_idx % 10 == 0:
                # 每 10 步打印摘要：top-3 最高权重 + scale + 平均值
                rw_np = rw.cpu().numpy()
                top3_idx = np.argsort(rw_np)[-3:][::-1]
                bot3_idx = np.argsort(rw_np)[:3]
                top_str = ' | '.join(f'{rule_names[i].split(":")[-1]}={rw_np[i]:.3f}'
                                    for i in top3_idx)
                bot_str = ' | '.join(f'{rule_names[i].split(":")[-1]}={rw_np[i]:.3f}'
                                    for i in bot3_idx)
                scale_val = model.chem_residual_scale.item()
                print(f'   Epoch {epoch+1}/{epochs} | Step {batch_idx} | '
                      f'mask_loss={loss_mask.item():.4f} | scale={scale_val:.3f} | '
                      f'avg_w={rw_np.mean():.4f} | '
                      f'top:[{top_str}] | bot:[{bot_str}]')

        # ---- Epoch 总结 ----
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        history['epoch'].append(epoch)

        # 最终权重（按类别分组显示）
        final_analysis = model.get_chem_attn_analysis()
        final_rw = final_analysis['rule_weights'] if final_analysis else torch.zeros(n_rules)
        w_by_cat = model.chem_rule_engine.get_rule_weights_by_category()

        print(f'\n{"=" * 60}')
        print(f'Epoch {epoch+1}/{epochs} Summary')
        print(f'  Avg mask loss: {avg_loss:.4f}')
        print(f'  Rule weights by category:')
        for cat in ChemicalRuleEngine.CATEGORY_NAMES:
            if cat not in w_by_cat:
                continue
            cat_rules = w_by_cat[cat]
            sorted_rules = sorted(cat_rules.items(), key=lambda x: x[1], reverse=True)
            print(f'  [{cat}] {len(sorted_rules)} rules, '
                  f'mean={np.mean([w for _, w in sorted_rules]):.4f}, '
                  f'max={sorted_rules[0][1]:.4f}')
            # 显示前 3 和后 2
            shown = sorted_rules[:3]
            if len(sorted_rules) > 5:
                shown += [('...', -1)] + sorted_rules[-2:]
            for name, w in shown:
                if name == '...':
                    print(f'        ...')
                else:
                    bar = '█' * int(w * 40) + '░' * (40 - int(w * 40))
                    short_name = name.split(':')[-1] if ':' in name else name
                    print(f'        {short_name:30s}: {w:.4f} |{bar}|')
        print(f'{"=" * 60}\n')

        # 保存 checkpoint
        ckpt_path = save_dir / f'chem_aware_v3_epoch{epoch+1}.pt'
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
    print(f'ChemAwareDreaMS v3 — Reward-based, last-layer-only')
    print(f'  chem_attn_layer: {args.chem_attn_layer} '
          f'({"last" if args.chem_attn_layer == -1 else args.chem_attn_layer})')

    # ---- 加载预训练 DreaMS ----
    ckpt_path = args.ckpt_path or (PRETRAINED / 'ssl_model.ckpt')
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    print(f'Loading pretrained model: {ckpt_path}')

    pkg = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'args' in pkg and 'state_dict' in pkg:
        print('   Detected server-optimized format (state_dict + args)')
        state_dict = pkg['state_dict']
        from argparse import Namespace
        recon_args = Namespace(**pkg['args'])
        from dreams.utils.dformats import DataFormatA
        recon_args.dformat = DataFormatA()
        for da in ['max_mz', 'max_peaks_n', 'max_tbxic_stdev', 'min_peaks_n',
                   'min_charge', 'max_charge', 'max_prec_mz', 'high_intensity_thld',
                   'min_intensity_ampl', 'max_ms_level']:
            if da in pkg['args']:
                setattr(recon_args.dformat, da, pkg['args'][da])
        recon_args.d_graphormer_params = 0

        from dreams.utils.data import SpectrumPreprocessor
        spec_preproc_recon = SpectrumPreprocessor(
            dformat=recon_args.dformat,
            n_highest_peaks=recon_args.max_peaks_n)
        pretrained = DreaMS(recon_args, spec_preproc_recon)
        pretrained.load_state_dict(state_dict, strict=False)
        pretrained.eval()
        n_args = len(pkg['args'])
        print(f'   Loaded {len(state_dict)} params, {n_args} args')
    elif 'pytorch-lightning_version' in pkg:
        print('   Detected Lightning checkpoint format')
        pretrained = DreaMS.load_from_checkpoint(ckpt_path, map_location=device)
        pretrained.eval()
    else:
        raise ValueError(f'Unknown checkpoint format. Keys: {list(pkg.keys())[:5]}')

    # ---- 构建 ChemAwareDreaMS [v3] ----
    print('Building ChemAwareDreaMS v3...')
    model = build_chem_aware_from_pretrained(pretrained, chem_attn_layer=args.chem_attn_layer)
    print(f'   chem_attn_enabled: {model.chem_attn_enabled}')
    rw = model.chem_rule_engine.get_rule_weights()
    print(f'   Initial rule weights: {rw.tolist()}')

    # ---- 准备数据 ----
    if args.dry_run or args.dataset_path is None:
        print('\n*** DRY RUN: Creating synthetic dataset for testing ***')
        from torch.utils.data import DataLoader, TensorDataset
        dummy_specs = []
        for _ in range(20):
            mz = torch.rand(30) * 1000.0
            mz = mz.sort().values
            intens = torch.rand(30)
            intens = intens / intens.max()
            spec = torch.stack([mz, intens], dim=-1)
            dummy_specs.append(spec)
        max_len = max(s.shape[0] for s in dummy_specs)
        specs_padded = torch.zeros(20, max_len, 2)
        for i, s in enumerate(dummy_specs):
            specs_padded[i, :s.shape[0]] = s
        dummy_dataset = TensorDataset(specs_padded)
        dataloader = DataLoader(dummy_dataset, batch_size=args.batch_size, shuffle=True)
        print(f'   Synthetic dataset: {len(dummy_dataset)} spectra, {max_len} peaks each')
        actual_epochs = 1
    else:
        print(f'\nLoading dataset: {args.dataset_path}')
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
        save_dir=Path(args.save_dir),
        device=device,
    )

    # ---- 最终结果 ----
    print('=' * 60)
    print('Fine-tuning complete!')
    initial = history['rule_weights'][0]
    final = history['rule_weights'][-1]
    print(f'   Initial mean weight: {initial.mean().item():.4f}')
    print(f'   Final mean weight:   {final.mean().item():.4f}')
    print(f'   Checkpoints saved in: {args.save_dir}')

    # 打印权重变化摘要（按变化量排序，top-10）
    rule_names = model.chem_rule_engine.get_rule_names()
    print('\n   Weight change summary (top changes):')
    changes = []
    for i, name in enumerate(rule_names):
        delta = final[i].item() - initial[i].item()
        changes.append((name, initial[i].item(), final[i].item(), delta))
    changes.sort(key=lambda x: abs(x[3]), reverse=True)
    for name, init_val, fin_val, delta in changes[:10]:
        direction = '↑' if delta > 0 else '↓' if delta < 0 else '→'
        print(f'     {name:35s}: {init_val:.4f} → {fin_val:.4f} '
              f'({direction} {abs(delta):.4f})')

    # 最终 top-5 和 bottom-5
    print('\n   Top-5 active rules:')
    w_dict = model.chem_rule_engine.get_rule_weight_dict()
    sorted_rules = sorted(w_dict.items(), key=lambda x: x[1], reverse=True)
    for name, w in sorted_rules[:5]:
        print(f'     {name:35s}: {w:.4f}')
    print('\n   Bottom-5 (near-zero) rules:')
    for name, w in sorted_rules[-5:]:
        print(f'     {name:35s}: {w:.4f}')


if __name__ == '__main__':
    main()
