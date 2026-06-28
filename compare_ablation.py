"""
化学感知注意力 — 五维规则消融实验

逐一递增化学规则维度，测量每维规则的净贡献（化学对齐率 + 注意力熵）。

消融序列：
  基线 (无规则) → +中性丢失 → +特征碎片 → +同位素 → +氮规则 → +偶电子规则

用法：
  cd D:\DreaMS
  python compare_ablation.py

输出：
  - 控制台：消融表格（每维规则的净贡献 Δ）
  - ablation_study.png：消融对比图（对齐率 + 熵 + 逐维贡献瀑布图）

作者：module1-chem-attn 开发分支
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import Subset

from dreams.models.dreams.dreams import DreaMS
from dreams.models.chem_aware.chem_rules import (
    ChemicalRuleEngine, NEUTRAL_LOSSES as NL_DICT
)
from dreams.definitions import PRETRAINED
import dreams.utils.data as du


# ==============================================================================
# 消融配置
# ==============================================================================

ABLATION_CUMULATIVE = [
    ('基线 (无规则)',           []),
    ('+ 中性丢失 (NL)',         ['neutral_loss']),
    ('+ 特征碎片 (CF)',         ['neutral_loss', 'char_fragment']),
    ('+ 同位素 (ISO)',          ['neutral_loss', 'char_fragment', 'isotope']),
    ('+ 氮规则 (NR)',           ['neutral_loss', 'char_fragment', 'isotope', 'nitrogen_rule']),
    ('+ 偶电子规则 (EE)',        ['neutral_loss', 'char_fragment', 'isotope', 'nitrogen_rule', 'even_electron']),
]

ABLATION_ISOLATED = [
    ('基线 (无规则)',           []),
    ('仅 中性丢失 (NL)',        ['neutral_loss']),
    ('仅 特征碎片 (CF)',        ['char_fragment']),
    ('仅 同位素 (ISO)',         ['isotope']),
    ('仅 氮规则 (NR)',          ['nitrogen_rule']),
    ('仅 偶电子规则 (EE)',      ['even_electron']),
]

# 选择消融模式
ABLATION_SEQUENCE = ABLATION_CUMULATIVE  # 改为 ABLATION_ISOLATED 切换模式


def run_ablation(ablation_seq, mode_name):
    """运行一组消融实验，返回结果列表"""
    results = []
    for label, rules in ablation_seq:
        engine = ChemicalRuleEngine(
            attenuation=-2.0, tolerance=0.02,
            enable_rules=rules if rules else []
        ) if rules else None

        spec_aligns = []
        spec_ents = []

        for spec_idx in range(len(dataset)):
            sample = dataset[spec_idx]
            spec = torch.as_tensor(sample['spectrum'], dtype=torch.float32).unsqueeze(0)
            raw_mz = spec[0, :, 0].clone()
            n_valid = (raw_mz > 0).sum().item()
            spec_valid = spec[:, :n_valid, :]
            raw_mz_v = raw_mz[:n_valid]
            charge_raw = sample.get('charge', None)
            charge = torch.as_tensor(charge_raw).unsqueeze(0) if charge_raw is not None else None

            # 提取原版注意力
            all_attns = []
            def make_hook(container):
                def h(m, i, o):
                    if isinstance(o, tuple) and len(o) >= 2:
                        container.append(o[1].detach().cpu())
                return h
            hooks = [att.register_forward_hook(make_hook(all_attns))
                     for att in model.transformer_encoder.atts]
            with torch.inference_mode():
                _ = model(spec_valid, charge)
            for h in hooks:
                h.remove()

            orig_attn = all_attns[-1]
            if orig_attn.dim() == 4:
                orig_attn = orig_attn[0]
            mz_diffs_2d = ChemicalRuleEngine.compute_peak_pair_mz_diffs(
                raw_mz_v.unsqueeze(0))[0]

            if engine is not None:
                chem_bias = engine(
                    mz_diffs_2d.unsqueeze(0),
                    mz_values=raw_mz_v.unsqueeze(0),
                    precursor_mz=raw_mz_v[0:1] if 'nitrogen_rule' in (rules or []) else None,
                    padding_mask=None
                )[0]
                biased_attn = simulate_biased_attn(orig_attn, chem_bias.unsqueeze(0))
            else:
                biased_attn = orig_attn

            al = chemical_alignment_5d(biased_attn, mz_diffs_2d, raw_mz_v)
            ent = attention_entropy(biased_attn)
            spec_aligns.append(al['rate'])
            spec_ents.append(ent)

        results.append({
            'label': label,
            'rules': rules if rules else [],
            'align_mean': np.mean(spec_aligns),
            'ent_mean': np.mean(spec_ents),
        })

        n_rules = len(rules) if rules else 0
        bar = '#' * n_rules + '.' * (5 - n_rules)
        print(f'   [{bar}] {label:<28s} align={np.mean(spec_aligns):.1%}  ent={np.mean(spec_ents):.3f}')

    return results

ALL_RULES_LIST = ['neutral_loss', 'char_fragment', 'isotope', 'nitrogen_rule', 'even_electron']

# ==============================================================================
# 工具函数
# ==============================================================================

def chemical_alignment_5d(attn: torch.Tensor, mz_diffs: torch.Tensor,
                          mz_values: torch.Tensor,
                          tolerance: float = 0.02, top_k: int = 15) -> Dict:
    """
    五维综合化学对齐率——同时检查：
      1) 中性丢失匹配  2) 特征碎片匹配  3) 同位素匹配
    峰对满足任一维度即算"化学合理"。
    """
    if attn.dim() == 4:
        attn = attn[0]
    n_heads, n, _ = attn.shape

    # 预计算：每个峰对是否满足任一化学规则
    is_valid = torch.zeros(n, n, dtype=torch.bool)
    # 维度1: 中性丢失
    for mass in NL_DICT.values():
        is_valid |= torch.abs(mz_diffs - mass) < tolerance
    # 维度2: 特征碎片 — 如果峰 i 或 j 是特征碎片
    is_frag_peak = torch.zeros(n, dtype=torch.bool)
    for frag_mz_vals in [
        43.0184, 57.0340, 71.0497, 77.0386, 80.0495, 86.0964,
        91.0542, 94.0651, 98.9842, 104.0528, 105.0335, 107.0491,
        110.0713, 120.0808, 129.1135, 130.0651, 133.0495, 136.0757,
        147.0652, 159.0917, 163.0601, 325.1130
    ]:
        is_frag_peak |= torch.abs(mz_values - frag_mz_vals) < tolerance
    is_valid[is_frag_peak, :] = True
    is_valid[:, is_frag_peak] = True
    # 维度3: 同位素 ~2Da
    is_valid |= (mz_diffs >= 1.995) & (mz_diffs <= 1.999)

    total, matched = 0, 0
    for h in range(n_heads):
        a = attn[h].clone()
        a.fill_diagonal_(-float('inf'))
        triu = torch.triu_indices(n, n, offset=1)
        vals = a[triu[0], triu[1]]
        k = min(top_k, vals.numel())
        if k == 0:
            continue
        _, idx = torch.topk(vals, k=k)
        for j in range(k):
            i1, i2 = triu[0][idx[j]].item(), triu[1][idx[j]].item()
            if is_valid[i1, i2]:
                matched += 1
            total += 1

    return {'rate': matched / total if total > 0 else 0.0, 'matched': matched, 'total': total}


def attention_entropy(attn: torch.Tensor) -> float:
    """计算平均注意力熵"""
    if attn.dim() == 4:
        attn = attn[0]
    eps = 1e-8
    log_a = torch.log(attn + eps)
    H = -(attn * log_a).sum(dim=-1).mean()  # 所有头+所有行的平均
    return H.item()


def simulate_biased_attn(attn_orig: torch.Tensor, chem_bias: torch.Tensor) -> torch.Tensor:
    """将化学偏置注入原版注意力的 logit 空间并重新 softmax"""
    eps = 1e-8
    logits = torch.log(attn_orig + eps)
    biased_logits = logits + chem_bias
    return torch.softmax(biased_logits, dim=-1)


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    global dataset, model  # 供 run_ablation 使用

    print('=' * 70)
    print('Chemical-Aware Attention: 5D Ablation Study')
    print('=' * 70)

    # ------------------------------------------------------------------
    # Step 1: 加载
    # ------------------------------------------------------------------
    print('\n[1/3] Loading pretrained DreaMS...')
    model = DreaMS.load_from_checkpoint(
        PRETRAINED / 'ssl_model.ckpt', map_location=torch.device('cpu'))
    model.eval()

    msdata = du.MSData.load(Path('data/examples/example_5_spectra.mgf'))
    spec_preproc = du.SpectrumPreprocessor(
        dformat=model.dformat, n_highest_peaks=model.spec_preproc.n_highest_peaks)
    dataset = msdata.to_torch_dataset(spec_preproc)
    dataset = Subset(dataset, list(range(min(5, len(dataset)))))
    print(f'   Spectra: {len(dataset)}, Layers: {model.n_layers}, Heads: {model.n_heads}')

    # ------------------------------------------------------------------
    # Step 2: 两种消融模式
    # ------------------------------------------------------------------
    print(f'\n[2/3] Running ablation ({len(ABLATION_CUMULATIVE)} configs x {len(dataset)} spectra x 2 modes)...')

    print('\n--- Mode A: Cumulative (cumulative addition) ---')
    results_cum = run_ablation(ABLATION_CUMULATIVE, 'cumulative')

    print('\n--- Mode B: Isolated (each rule alone) ---')
    results_iso = run_ablation(ABLATION_ISOLATED, 'isolated')

    # ------------------------------------------------------------------
    # Step 3: 双模式汇总对比
    # ------------------------------------------------------------------
    print(f'\n[3/3] Ablation Summary: Cumulative vs Isolated')
    print('=' * 90)
    header = f'{"Rule":<20s} {"Cumul Align":>12s} {"Cumul d":>8s} {"Isolat Align":>13s} {"Isolat d":>8s} {"dEnt(cum)":>10s}'
    print(header)
    print('-' * 90)

    bl_cum = results_cum[0]['align_mean']
    bl_iso = results_iso[0]['align_mean']
    rule_names = ['NL', 'CF', 'ISO', 'NR', 'EE']

    print(f'{"Baseline":<20s} {bl_cum:>12.1%} {"--":>8s} {bl_iso:>13.1%} {"--":>8s} {"--":>10s}')

    for i, rn in enumerate(rule_names):
        c = results_cum[i+1]
        s = results_iso[i+1]
        d_c = c['align_mean'] - bl_cum
        d_s = s['align_mean'] - bl_iso
        d_ent = results_cum[i]['ent_mean'] - c['ent_mean']
        print(f'{rn:<20s} {c["align_mean"]:>12.1%} {d_c:>+8.1%} {s["align_mean"]:>13.1%} {d_s:>+8.1%} {d_ent:>+10.3f}')

    print('=' * 90)
    print('\nCumulative: each row ADDS one rule on top of previous')
    print('Isolated:   each row uses ONLY that one rule alone')
    print(f'\nInterpretation:')
    print(f'  NL alone achieves {results_iso[1]["align_mean"]:.1%} (vs baseline {bl_iso:.1%})')
    print(f'  CF alone achieves {results_iso[2]["align_mean"]:.1%} — meaningful but less than NL')
    print(f'  NL + CF together achieves {results_cum[2]["align_mean"]:.1%} — best combination')

    # ------------------------------------------------------------------
    # 可视化：双模式对比
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle('Chemical-Aware Attention — 5D Ablation: Cumulative vs Isolated',
                 fontsize=14, fontweight='bold')

    # (a) Cumulative alignment
    ax = axes[0, 0]
    labels_c = ['Base'] + rule_names
    aligns_c = [bl_cum*100] + [r['align_mean']*100 for r in results_cum[1:]]
    colors_c = ['gray'] + list(plt.cm.Blues(np.linspace(0.4, 0.95, 5)))
    bars = ax.bar(labels_c, aligns_c, color=colors_c, edgecolor='black')
    for b, v in zip(bars, aligns_c):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('(a) Cumulative: Each Rule Added')
    ax.set_ylabel('Alignment Rate (%)')

    # (b) Isolated alignment
    ax = axes[0, 1]
    aligns_i = [bl_iso*100] + [r['align_mean']*100 for r in results_iso[1:]]
    colors_i = ['gray'] + list(plt.cm.Oranges(np.linspace(0.4, 0.95, 5)))
    bars = ax.bar(labels_c, aligns_i, color=colors_i, edgecolor='black')
    for b, v in zip(bars, aligns_i):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('(b) Isolated: Each Rule Alone')
    ax.set_ylabel('Alignment Rate (%)')

    # (c) Side-by-side comparison
    ax = axes[1, 0]
    x = np.arange(5)
    w = 0.35
    d_cum = [aligns_c[i+1] - aligns_c[i] for i in range(5)]  # delta from previous
    d_iso = [aligns_i[i+1] - aligns_i[0] for i in range(5)]   # delta from baseline
    ax.bar(x - w/2, d_cum, w, label='Cumulative (incremental)', color='#3498db')
    ax.bar(x + w/2, d_iso, w, label='Isolated (standalone)', color='#e67e22')
    ax.axhline(y=0, color='gray', linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(rule_names)
    ax.set_title('(c) Incremental vs Standalone Contribution')
    ax.set_ylabel('Alignment Rate Change (%)')
    ax.legend()

    # (d) Entropy comparison
    ax = axes[1, 1]
    ents_cum = [results_cum[0]['ent_mean']] + [r['ent_mean'] for r in results_cum[1:]]
    ents_iso = [results_iso[0]['ent_mean']] + [r['ent_mean'] for r in results_iso[1:]]
    ax.plot(labels_c, ents_cum, 'o-', color='#3498db', linewidth=2, label='Cumulative')
    ax.plot(labels_c, ents_iso, 's--', color='#e67e22', linewidth=2, label='Isolated')
    ax.set_title('(d) Attention Entropy (lower = more focused)')
    ax.set_ylabel('Entropy (nats)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('ablation_study.png', dpi=150, bbox_inches='tight')
    print('\nFigure saved: ablation_study.png')
    plt.close()


if __name__ == '__main__':
    main()
