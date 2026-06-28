"""
化学感知注意力对比实验脚本

功能：
  在同一份预训练 DreaMS 权重上，对比"原版注意力"与"注入化学偏置后的注意力"：
    1. 化学对齐率 (Chemical Alignment Score) — 高注意力峰对匹配已知中性丢失的比例
    2. 注意力熵 — 化学感知后注意力是否更聚焦
    3. 注意力热力图 — 可视化化学偏置矩阵与原版注意力的差异
    4. 化学偏置矩阵 — 展示模型"看到"了哪些化学规则

用法：
  cd D:\DreaMS
  python compare_chem_attn.py

输出：
  - 控制台：定量对比指标
  - 图片文件：attention_comparison.png（保存在当前目录）

作者：module1-chem-attn 开发分支
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sys

# DreaMS 导入
import dreams.utils.data as du
import dreams.utils.dformats as dformats
from dreams.models.dreams.dreams import DreaMS, get_embeddings
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine, NEUTRAL_LOSSES as NEUTRAL_LOSSES_DICT
from dreams.models.chem_aware.gating import HeadGatingNetwork
from dreams.definitions import PRETRAINED


# ==============================================================================
# 工具函数
# ==============================================================================

def load_pretrained_dreaMS() -> DreaMS:
    """加载预训练的 DreaMS 模型（ssl_model.ckpt，含完整 Transformer 参数）"""
    ckpt_path = PRETRAINED / 'ssl_model.ckpt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'预训练权重未找到: {ckpt_path}')
    model = DreaMS.load_from_checkpoint(
        ckpt_path, map_location=torch.device('cpu')
    )
    model.eval()
    return model


def load_example_spectra(model, n_samples: int = 3):
    """
    加载示例谱图数据，预处理后返回 torch dataset。
    直接返回预处理后的 torch dataset，取前 n_samples。
    """
    pth = Path('data/examples/example_5_spectra.mgf')
    msdata = du.MSData.load(pth)
    spec_preproc = du.SpectrumPreprocessor(
        dformat=model.dformat,
        n_highest_peaks=model.spec_preproc.n_highest_peaks
    )
    dataset = msdata.to_torch_dataset(spec_preproc)
    # 取子集
    from torch.utils.data import Subset
    n = min(n_samples, len(dataset))
    return Subset(dataset, list(range(n)))


def extract_raw_attention(
    model: DreaMS,
    spec: torch.Tensor,
    charge: Optional[torch.Tensor] = None,
    layer_idx: int = -1
) -> List[torch.Tensor]:
    """
    从 DreaMS 模型各层提取注意力权重矩阵（通过 forward hook）

    返回：
        attn_matrices: List[Tensor] — 每层的注意力权重，形状 (n_heads, n_peaks, n_peaks)
    """
    attn_matrices = []

    def attn_hook(module, input, output):
        # output 是 (output_tensor, attn_weights)
        if isinstance(output, tuple) and len(output) == 2:
            attn_matrices.append(output[1].detach().cpu())  # (bs*n_heads, n, n)

    # 注册 hook
    hooks = []
    for att in model.transformer_encoder.atts:
        hooks.append(att.register_forward_hook(attn_hook))

    # 前向传播
    with torch.inference_mode():
        _ = model(spec, charge)

    # 移除 hook
    for h in hooks:
        h.remove()

    return attn_matrices


def compute_chemical_alignment(
    attn_weights: torch.Tensor,     # (batch, n_heads, n, n) 或 (n_heads, n, n)
    mz_diffs: torch.Tensor,         # (n, n)
    neutrals: Dict[str, float],
    tolerance: float = 0.02,
    top_k: int = 10
) -> Dict:
    """
    计算注意力与化学规则的"对齐率"：
    从每个注意力头中取 top_k 高分峰对，统计其中匹配已知中性丢失的比例。

    返回：
        dict: {'alignment_rate': float, 'matched_pairs': int, 'total_pairs': int}
    """
    # 处理可能的 batch 维度
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0]  # (n_heads, n, n)
    n_heads, n, _ = attn_weights.shape
    total_high_attn = 0
    matched_high_attn = 0

    for h in range(n_heads):
        # 取上三角（排除对角线 + 对称冗余）
        attn_h = attn_weights[h].clone()
        attn_h.fill_diagonal_(-float('inf'))
        triu_indices = torch.triu_indices(n, n, offset=1)

        # 选 top_k 高分峰对
        triu_vals = attn_h[triu_indices[0], triu_indices[1]]
        if triu_vals.numel() < top_k:
            continue
        _, top_indices = torch.topk(triu_vals, k=top_k)
        top_i = triu_indices[0][top_indices]
        top_j = triu_indices[1][top_indices]

        # 检查这些峰对的质量差是否匹配已知中性丢失
        for idx in range(top_k):
            i, j = top_i[idx].item(), top_j[idx].item()
            diff = mz_diffs[i, j].item()
            matched = False
            for name, mass in neutrals.items():
                if abs(diff - mass) < tolerance:
                    matched = True
                    break
            total_high_attn += 1
            if matched:
                matched_high_attn += 1

    alignment_rate = matched_high_attn / total_high_attn if total_high_attn > 0 else 0.0

    return {
        'alignment_rate': alignment_rate,
        'matched_pairs': matched_high_attn,
        'total_pairs': total_high_attn
    }


def compute_attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算注意力权重矩阵的逐行熵，返回 (n_heads,) 的平均熵"""
    log_attn = torch.log(attn_weights + eps)
    entropy = -(attn_weights * log_attn).sum(dim=-1)  # (n_heads, n)
    return entropy.mean(dim=-1)  # (n_heads,)


def simulate_chem_biased_attention(
    attn_weights: torch.Tensor,     # (n_heads, n, n) — 原版注意力
    chem_bias: torch.Tensor,        # (1, n, n) — 化学偏置
    temperature: float = 1.0
) -> torch.Tensor:
    """
    模拟化学感知注意力：将 chem_bias 加到 logit 空间（softmax 前），重新计算 softmax。

    由于我们从 hook 获取的是 softmax 后的注意力权重，需要逆向到 logit 空间再加偏置。
    简化做法：在 softmax 后的权重上施加温度缩放 * chem_bias 指数调制。
    """
    n_heads = attn_weights.shape[0]
    eps = 1e-8

    # 逆向到 logit 空间（假设 softmax 是可逆的，取 log）
    logits = torch.log(attn_weights + eps)  # (n_heads, n, n)

    # 注入化学偏置
    logits_biased = logits + chem_bias / temperature  # 广播 chem_bias (1, n, n) → (n_heads, n, n)

    # 重新 softmax
    attn_biased = torch.softmax(logits_biased, dim=-1)

    return attn_biased


# ==============================================================================
# 可视化
# ==============================================================================

def plot_comparison(
    spec_idx: int,
    mz_values: np.ndarray,
    orig_attn: np.ndarray,        # (n_heads, n, n)
    biased_attn: np.ndarray,      # (n_heads, n, n)
    chem_bias: np.ndarray,        # (1, n, n)
    mz_diffs: np.ndarray,         # (n, n)
    alignment_orig: Dict,
    alignment_biased: Dict,
    entropy_orig: np.ndarray,
    entropy_biased: np.ndarray,
    neutrals: Dict[str, float],
    save_path: str = 'attention_comparison.png'
):
    """
    绘制 4 面板对比图：
      (a) 化学偏置矩阵 heatmap
      (b) 原版注意力 vs 化学感知注意力（最佳头）
      (c) 化学对齐率对比柱状图
      (d) 注意力熵对比（逐头）
    """
    n_heads = orig_attn.shape[0]
    n_peaks = orig_attn.shape[1]

    # 找到原版注意力与化学偏置最相关的头（展示）
    chem_bias_flat = chem_bias.squeeze().flatten()  # (n*n,)
    best_head = np.argmax([
        np.corrcoef(orig_attn[h].flatten(), chem_bias_flat)[0, 1]
        for h in range(n_heads)
    ])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f'Chemical-Aware Attention Comparison (Spectrum #{spec_idx}, {n_peaks} peaks)',
        fontsize=14, fontweight='bold'
    )

    # ---- (a) 化学偏置矩阵 ----
    ax = axes[0, 0]
    im = ax.imshow(chem_bias.squeeze(), cmap='RdBu_r', aspect='equal',
                   vmin=-5, vmax=0, interpolation='nearest')
    ax.set_title('(a) Chemical Bias Matrix\n(Blue=chemically plausible, Red=attenuated)')
    ax.set_xlabel('Peak index j')
    ax.set_ylabel('Peak index i')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Bias value')

    # 标注 m/z 值
    if len(mz_values) <= 15:
        for i in range(n_peaks):
            ax.text(i, -1, f'{mz_values[i]:.1f}', ha='center', va='bottom',
                    fontsize=6, rotation=45)

    # ---- (b) 注意力对比（最佳头） ----
    ax = axes[0, 1]
    ax.imshow(orig_attn[best_head], cmap='YlOrRd', aspect='equal', interpolation='nearest')
    ax.set_title(f'(b) Original Attention (Head {best_head})')
    ax.set_xlabel('Peak index j')
    ax.set_ylabel('Peak index i')

    ax = axes[0, 2]
    ax.imshow(biased_attn[best_head], cmap='YlOrRd', aspect='equal', interpolation='nearest')
    ax.set_title(f'(c) Chem-Aware Attention (Head {best_head})')
    ax.set_xlabel('Peak index j')
    ax.set_ylabel('Peak index i')

    # ---- (c) 化学对齐率 ----
    ax = axes[1, 0]
    bars = ax.bar(
        ['Original', 'Chem-Aware'],
        [alignment_orig['alignment_rate'], alignment_biased['alignment_rate']],
        color=['#3498db', '#e74c3c'], width=0.5
    )
    ax.set_title('(d) Chemical Alignment Rate\n(% high-attn pairs matching neutral losses)')
    ax.set_ylabel('Alignment Rate')
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, [alignment_orig['alignment_rate'], alignment_biased['alignment_rate']]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', fontsize=12, fontweight='bold')

    # ---- (d) 注意力熵对比 ----
    ax = axes[1, 1]
    x = np.arange(n_heads)
    width = 0.35
    ax.bar(x - width/2, entropy_orig, width, label='Original', color='#3498db')
    ax.bar(x + width/2, entropy_biased, width, label='Chem-Aware', color='#e74c3c')
    ax.set_title('(e) Attention Entropy per Head\n(lower = more focused)')
    ax.set_xlabel('Head index')
    ax.set_ylabel('Entropy (nats)')
    ax.set_xticks(x)
    ax.legend()

    # ---- (e) 中性丢失匹配热图 ----
    ax = axes[1, 2]
    # 创建中性丢失匹配矩阵
    loss_match = np.zeros((n_peaks, n_peaks))
    for i in range(n_peaks):
        for j in range(n_peaks):
            diff = mz_diffs[i, j]
            for name, mass in neutrals.items():
                if abs(diff - mass) < 0.02:
                    loss_match[i, j] = 1
                    break
    ax.imshow(loss_match, cmap='Greens', aspect='equal', interpolation='nearest')
    ax.set_title('(f) Known Neutral Loss Matches\n(Green = matches known rule)')
    ax.set_xlabel('Peak index j')
    ax.set_ylabel('Peak index i')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n对比图已保存至: {save_path}')
    plt.close()


# ==============================================================================
# 主比较流程
# ==============================================================================

def main():
    print('=' * 70)
    print('化学感知注意力对比实验 (Phase 1)')
    print('=' * 70)

    # ------------------------------------------------------------------
    # Step 1: 加载模型与数据
    # ------------------------------------------------------------------
    print('\n[1/5] 加载预训练 DreaMS 模型...')
    model = load_pretrained_dreaMS()
    print(f'   模型: {model.__class__.__name__}')
    print(f'   层数: {model.n_layers}, 头数: {model.n_heads}')

    print('\n[2/5] 加载示例谱图...')
    torch_data = load_example_spectra(model, n_samples=3)
    print(f'   谱图数量: {len(torch_data)}')

    # ------------------------------------------------------------------
    # Step 2: 初始化化学规则引擎
    # ------------------------------------------------------------------
    print('\n[3/5] 初始化化学规则引擎...')
    engine = ChemicalRuleEngine(attenuation=-2.0, tolerance=0.02)
    print(f'   中性丢失规则: {len(engine.neutral_masses)} 条')

    # ------------------------------------------------------------------
    # Step 3: 逐谱图对比
    # ------------------------------------------------------------------
    print('\n[4/5] 执行对比分析...')
    all_results = []

    for spec_idx in range(len(torch_data)):
        sample = torch_data[spec_idx]
        # 转为 torch tensor
        spec = torch.as_tensor(sample['spectrum'], dtype=torch.float32).unsqueeze(0)  # (1, n, 2)
        charge = sample.get('charge', None)
        if charge is not None:
            charge = torch.as_tensor(charge).unsqueeze(0) if np.ndim(charge) == 0 else torch.as_tensor(charge)

        # 提取原始 m/z 值
        raw_mz = spec[0, :, 0].clone()  # (n,)
        n_peaks = (raw_mz > 0).sum().item()  # 有效峰数量
        raw_mz_valid = raw_mz[:n_peaks]

        # 计算化学偏置
        mz_diffs = ChemicalRuleEngine.compute_peak_pair_mz_diffs(
            raw_mz_valid.unsqueeze(0)
        )[0]  # (n_valid, n_valid)
        chem_bias = engine(mz_diffs.unsqueeze(0))[0]  # (1, n_valid, n_valid)

        # 提取原版注意力
        orig_spec = spec[:, :n_peaks, :]  # 裁掉 padding
        attn_matrices = extract_raw_attention(model, orig_spec, charge)

        # 选最后一层注意力（最深层的语义表示）
        orig_attn = attn_matrices[-1]  # (1, n_heads, n_valid, n_valid)
        if orig_attn.dim() == 4:
            orig_attn = orig_attn[0]  # (n_heads, n_valid, n_valid)

        # 模拟化学感知注意力
        biased_attn = simulate_chem_biased_attention(orig_attn, chem_bias, temperature=1.0)

        # 计算评估指标
        alignment_orig = compute_chemical_alignment(
            orig_attn, mz_diffs, NEUTRAL_LOSSES_DICT
        )
        alignment_biased = compute_chemical_alignment(
            biased_attn, mz_diffs, NEUTRAL_LOSSES_DICT
        )
        entropy_orig = compute_attention_entropy(orig_attn).numpy()
        entropy_biased = compute_attention_entropy(biased_attn).numpy()

        # 记录结果
        result = {
            'spec_idx': spec_idx,
            'n_peaks': n_peaks,
            'alignment_orig': alignment_orig['alignment_rate'],
            'alignment_biased': alignment_biased['alignment_rate'],
            'entropy_orig_mean': entropy_orig.mean(),
            'entropy_biased_mean': entropy_biased.mean(),
            'entropy_delta': entropy_orig.mean() - entropy_biased.mean(),
        }
        all_results.append(result)

        # 打印单谱图结果
        print(f'\n   --- Spectrum #{spec_idx} ({n_peaks} peaks) ---')
        print(f'   化学对齐率: 原版 {alignment_orig["alignment_rate"]:.1%} → '
              f'化学感知 {alignment_biased["alignment_rate"]:.1%} '
              f'(+{alignment_biased["alignment_rate"] - alignment_orig["alignment_rate"]:+.1%})')
        print(f'   平均注意力熵: 原版 {entropy_orig.mean():.3f} → '
              f'化学感知 {entropy_biased.mean():.3f} '
              f'({entropy_biased.mean() - entropy_orig.mean():+.3f})')

        # 绘图（仅第一张谱图）
        if spec_idx == 0:
            plot_comparison(
                spec_idx=spec_idx,
                mz_values=raw_mz_valid.numpy(),
                orig_attn=orig_attn.numpy(),
                biased_attn=biased_attn.numpy(),
                chem_bias=chem_bias.numpy(),
                mz_diffs=mz_diffs.numpy(),
                alignment_orig=alignment_orig,
                alignment_biased=alignment_biased,
                entropy_orig=entropy_orig,
                entropy_biased=entropy_biased,
                neutrals=NEUTRAL_LOSSES_DICT,
                save_path='attention_comparison.png'
            )

    # ------------------------------------------------------------------
    # Step 4: 汇总结果
    # ------------------------------------------------------------------
    print('\n[5/5] 汇总结果')
    print('=' * 70)
    print(f'{"指标":<30} {"原版 DreaMS":<15} {"化学感知":<15} {"变化":<10}')
    print('-' * 70)

    avg_orig_align = np.mean([r['alignment_orig'] for r in all_results])
    avg_biased_align = np.mean([r['alignment_biased'] for r in all_results])
    print(f'{"平均化学对齐率":<30} {avg_orig_align:<15.1%} {avg_biased_align:<15.1%} {avg_biased_align - avg_orig_align:+.1%}')

    avg_orig_ent = np.mean([r['entropy_orig_mean'] for r in all_results])
    avg_biased_ent = np.mean([r['entropy_biased_mean'] for r in all_results])
    print(f'{"平均注意力熵":<30} {avg_orig_ent:<15.3f} {avg_biased_ent:<15.3f} {avg_biased_ent - avg_orig_ent:+.3f}')

    print('=' * 70)
    print('\n结论:')
    if avg_biased_align > avg_orig_align:
        print(f'  化学感知注意力使高注意力峰对的化学对齐率提升了 {avg_biased_align - avg_orig_align:+.1%}')
    if avg_biased_ent < avg_orig_ent:
        print(f'  注意力熵降低了 {avg_orig_ent - avg_biased_ent:.3f} nats，注意力分布更聚焦')
    print('  详细可视化请查看 attention_comparison.png')


if __name__ == '__main__':
    main()
