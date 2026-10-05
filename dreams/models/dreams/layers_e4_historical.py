"""
多头注意力与 Transformer 编码器模块

来源：
  基础实现参考 https://github.com/tnq177/transformers_without_tears/tree/master
  Fairseq MultiheadAttention: https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py

改动说明（模块一：化学感知注意力）：
  MultiheadAttention.forward() 新增两个可选参数：
    - chem_bias: (batch, n_heads, n, n) 或 (batch, 1, n, n) — 化学规则偏置矩阵
      由 ChemicalRuleEngine 根据峰对质量差计算，在 softmax 前注入，
      使化学合理的碎裂路径获得正常注意力，不合理路径被衰减。
    - gate_weights: (batch, n_heads) — 注意力头门控权重
      由 HeadGatingNetwork 根据当前层输入动态生成，用于调节各注意力头的贡献。

  对比开关：
    当 chem_bias=None 且 gate_weights=None 时，行为与原版 DreaMS 完全一致，
    可直接用于 A/B 对比实验。
"""

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from typing import Optional


class MultiheadAttention(nn.Module):
    """
    多头注意力模块（化学感知扩展版）

    在原版 Graphormer 多头注意力基础上，新增两个可选输入：
      - chem_bias: 化学规则偏置（来自模块 B：ChemicalRuleEngine）
      - gate_weights: 注意力头门控权重（来自模块 A：HeadGatingNetwork）

    参数：
        args: Namespace，需包含以下字段
            d_model (int): 模型总维度
            n_heads (int): 注意力头数量
            att_dropout (float): 注意力 dropout 率
            no_transformer_bias (bool): 是否禁用 QKV 投影偏置
            attn_mech (str): 注意力机制类型 ('dot-product' / 'additive_v' / 'additive_fixed')
            d_graphormer_params (int): Graphormer 距离编码维度（0 表示禁用）
    """

    def __init__(self, args):
        super(MultiheadAttention, self).__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.dropout = args.att_dropout
        self.use_transformer_bias = not args.no_transformer_bias
        self.attn_mech = args.attn_mech
        self.d_graphormer_params = args.d_graphormer_params

        if self.d_model % self.n_heads != 0:
            raise ValueError('Required: d_model % n_heads == 0.')

        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim ** -0.5

        # Q、K、V、O 的线性投影参数（合并为一个大矩阵以提升效率）
        self.weights = Parameter(torch.Tensor(4 * self.d_model, self.d_model))
        if self.use_transformer_bias:
            self.biases = Parameter(torch.Tensor(4 * self.d_model))

        # Graphormer 距离编码：将峰对 m/z 差异投影为注意力偏置
        if self.d_graphormer_params:
            self.lin_graphormer = nn.Linear(self.d_graphormer_params, self.n_heads, bias=False)

        # 参数初始化
        # Xavier normal 初始化 std = sqrt(2/(2D)) 在 PostNorm 中过大导致不稳定，
        # 此处使用 FFN 模块的较小 std = sqrt(2/(5D))
        mean = 0
        std = (2 / (5 * self.d_model)) ** 0.5
        nn.init.normal_(self.weights, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.biases, 0.)

        if self.attn_mech == 'additive_v':
            self.additive_v = Parameter(torch.Tensor(self.n_heads, self.head_dim))
            nn.init.normal_(self.additive_v, mean=mean, std=std)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        graphormer_dists: Optional[torch.Tensor] = None,
        do_proj_qkv: bool = True,
        chem_bias: Optional[torch.Tensor] = None,
        gate_weights: Optional[torch.Tensor] = None
    ):
        """
        多头注意力前向传播（化学感知扩展版）

        参数：
            q, k, v: (batch, n_peaks, d_model) — 查询/键/值
            mask: (batch, n_peaks) — padding 掩码（True = 填充位）
            graphormer_dists: (batch, n_peaks, n_peaks, d_graphormer_params) — Graphormer 距离编码
            do_proj_qkv: bool — 是否执行 QKV 线性投影
            chem_bias: (batch, n_heads|1, n_peaks, n_peaks) — [模块一新增] 化学规则注意力偏置
                - 形状 (batch, 1, n, n) 时自动广播到所有头
                - 化学合理的峰对 → bias ≈ 0（不衰减）
                - 化学不合理的峰对 → bias < 0（衰减）
                - 为 None 时行为与原版完全一致
            gate_weights: (batch, n_heads) — [模块一新增] 注意力头门控权重
                - 值域 [0, 1]，由 HeadGatingNetwork 产生
                - 用于动态调节各注意力头的贡献
                - 为 None 时所有头等权重

        返回：
            output: (batch, n_peaks, d_model) — 注意力输出
            att_weights: (batch * n_heads, n_peaks, n_peaks) — 注意力权重矩阵（用于可视化分析）
        """
        bs, n, d = q.size()

        def _split_heads(tensor):
            """将 d_model 维度拆分为 (n_heads, head_dim) 并调整维度顺序"""
            bsz, length, d_model = tensor.size()
            return tensor.reshape(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)

        if do_proj_qkv:
            q, k, v = self.proj_qkv(q, k, v)

        q = _split_heads(q)  # (bs, n_heads, n, head_dim)
        k = _split_heads(k)
        v = _split_heads(v)

        # --- 注意力分数计算 ---
        if self.attn_mech == 'dot-product':
            att_weights = torch.einsum('bhnd,bhdm->bhnm', q, k.transpose(-2, -1))
        elif self.attn_mech == 'additive_v' or self.attn_mech == 'additive_fixed':
            att_weights = (q.unsqueeze(-2) - k.unsqueeze(-3))
            if self.attn_mech == 'additive_v':
                att_weights = (att_weights * self.additive_v.unsqueeze(0).unsqueeze(2).unsqueeze(3))
            att_weights = att_weights.sum(dim=-1)
        else:
            raise NotImplementedError(f'"{self.attn_mech}" attention mechanism is not implemented.')
        att_weights = att_weights * self.scale  # (bs, n_heads, n, n)

        # --- Graphormer 距离偏置（原版 DreaMS 已有） ---
        if graphormer_dists is not None:
            if self.d_graphormer_params:
                # (bs, n, n, dists_d) -> (bs, n_heads, n, n)
                att_bias = self.lin_graphormer(graphormer_dists).permute(0, 3, 1, 2)
            else:
                # (bs, n, n, dists_d) -> (bs, 1, n, n)，广播到所有头
                att_bias = graphormer_dists.sum(dim=-1).unsqueeze(1)
            att_weights = att_weights + att_bias

        # --- [模块一 遗留] 注意力头门控权重（v3 不再使用） ---
        if gate_weights is not None:
            # gate_weights: (bs, n_heads) -> (bs, n_heads, 1, 1) 广播到 (n, n) 维度
            att_weights = att_weights * gate_weights.unsqueeze(-1).unsqueeze(-1)

        # --- Padding 掩码 ---
        if mask is not None:
            att_weights.masked_fill_(mask.unsqueeze(1).unsqueeze(-1), -1e9)

        # --- Softmax + Dropout ---
        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)

        # --- 加权求和 ---
        _att_weights = att_weights.reshape(-1, n, n)  # (bs * n_heads, n, n)
        output = torch.bmm(_att_weights, v.reshape(bs * self.n_heads, -1, self.head_dim))
        output = output.reshape(bs, self.n_heads, n, self.head_dim).transpose(1, 2).reshape(bs, n, -1)
        output = self.proj_o(output)

        return output, att_weights

    def proj_qkv(self, q, k, v):
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr()
        kv_same = k.data_ptr() == v.data_ptr()

        if qkv_same:
            q, k, v = self._proj(q, end=3 * self.d_model).chunk(3, dim=-1)
        elif kv_same:
            q = self._proj(q, end=self.d_model)
            k, v = self._proj(k, start=self.d_model, end=3 * self.d_model).chunk(2, dim=-1)
        else:
            q = self.proj_q(q)
            k = self.proj_k(k)
            v = self.proj_v(v)

        return q, k, v

    def _proj(self, x, start=0, end=None):
        weight = self.weights[start:end, :]
        bias = None if not self.use_transformer_bias else self.biases[start:end]
        return F.linear(x, weight=weight, bias=bias)

    def proj_q(self, q):
        return self._proj(q, end=self.d_model)

    def proj_k(self, k):
        return self._proj(k, start=self.d_model, end=2 * self.d_model)

    def proj_v(self, v):
        return self._proj(v, start=2 * self.d_model, end=3 * self.d_model)

    def proj_o(self, x):
        return self._proj(x, start=3 * self.d_model)


class FeedForward(nn.Module):
    """FeedForward"""
    def __init__(self, args):
        super(FeedForward, self).__init__()
        self.dropout = args.ff_dropout
        self.d_model = args.d_model
        self.ff_dim = 4 * args.d_model
        self.use_transformer_bias = not args.no_transformer_bias

        self.in_proj = nn.Linear(self.d_model, self.ff_dim, bias=self.use_transformer_bias)
        self.out_proj = nn.Linear(self.ff_dim, self.d_model, bias=self.use_transformer_bias)

        # initializing
        mean = 0
        std = (2 / (self.ff_dim + self.d_model)) ** 0.5
        nn.init.normal_(self.in_proj.weight, mean=mean, std=std)
        nn.init.normal_(self.out_proj.weight, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.in_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x):
        # my preliminary experiments show all RELU-variants
        # work the same and slower, RELU FTW!!!
        y = F.relu(self.in_proj(x))
        y = F.dropout(y, p=self.dropout, training=self.training)
        return self.out_proj(y)


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class TransformerEncoder(nn.Module):
    """
    Transformer 编码器（化学感知扩展版）

    由 N 层 (MultiheadAttention + FeedForward) 堆叠而成，每层使用 Pre-Norm 或 Post-Norm。
    支持梯度检查点以节省显存。

    [模块一 新增] chem_bias 和 gate_weights 透传：
      - chem_bias: 单张偏置矩阵，所有层共享（因为 m/z 值不随层变化）
      - gate_weights: 每层可独立传入，为 None 时等于原版行为

    参数：
        args: Namespace，需包含
            residual_dropout (float): 残差 dropout 率
            n_layers (int): Transformer 层数
            pre_norm (bool): True = Pre-Norm, False = Post-Norm
            scnorm (bool): True = ScaleNorm, False = LayerNorm
    """

    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.residual_dropout = args.residual_dropout
        self.n_layers = args.n_layers
        self.pre_norm = args.pre_norm
        self._gradient_checkpointing = False

        self.atts = nn.ModuleList([MultiheadAttention(args) for _ in range(self.n_layers)])
        self.ffs = nn.ModuleList([FeedForward(args) for _ in range(self.n_layers)])

        num_scales = self.n_layers * 2 + 1 if self.pre_norm else self.n_layers * 2
        if args.scnorm:
            self.scales = nn.ModuleList([ScaleNorm(args.d_model ** 0.5) for _ in range(num_scales)])
        else:
            self.scales = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(num_scales)])

    def gradient_checkpointing_enable(self):
        """启用梯度检查点以节省训练显存"""
        self._gradient_checkpointing = True

    def _layer_forward(self, i, x, src_mask, graphormer_dists, chem_bias=None, gate_weights=None):
        """
        单层 Transformer 前向传播

        参数：
            i: int — 层索引
            x: (batch, n_peaks, d_model) — 当前层输入
            src_mask: (batch, n_peaks) — padding 掩码
            graphormer_dists: (batch, n, n, d) — Graphormer 距离编码
            chem_bias: (batch, n_heads|1, n, n) — [模块一] 化学规则偏置，为 None 时原版行为
            gate_weights: (batch, n_heads) — [模块一] 本层门控权重，为 None 时所有头等权

        返回：
            x: (batch, n_peaks, d_model) — 本层输出
        """
        pre_norm = self.pre_norm
        post_norm = not pre_norm
        att = self.atts[i]
        ff = self.ffs[i]
        att_scale = self.scales[2 * i]
        ff_scale = self.scales[2 * i + 1]

        # --- 注意力子层 ---
        residual = x
        x = att_scale(x) if pre_norm else x
        x, _ = att(q=x, k=x, v=x, mask=src_mask, graphormer_dists=graphormer_dists,
                    chem_bias=chem_bias, gate_weights=gate_weights)
        x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
        x = att_scale(x) if post_norm else x

        # --- FeedForward 子层 ---
        residual = x
        x = ff_scale(x) if pre_norm else x
        x = ff(x)
        x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
        x = ff_scale(x) if post_norm else x
        return x

    def forward(self, src_inputs, src_mask, graphormer_dists=None, chem_bias=None, gate_weights_per_layer=None):
        """
        Transformer 编码器前向传播

        参数：
            src_inputs: (batch, n_peaks, d_model) — 嵌入后的谱图 token 序列
            src_mask: (batch, n_peaks) — padding 掩码
            graphormer_dists: (batch, n, n, d) — Graphormer 距离编码
            chem_bias: (batch, 1, n, n) 或 List[Optional[Tensor]] — [模块一 v3]
                单张 tensor → 所有层共享同一偏置（向后兼容）
                列表 → 每层独立偏置，chem_bias[i] 给第 i 层，可为 None
                整体为 None → 所有层不加化学偏置（原版行为）
            gate_weights_per_layer: List[Tensor] 或 None — [模块一] 每层门控权重列表，
                每个元素形状为 (batch, n_heads)，长度为 n_layers，为 None 时所有头等权

        返回：
            x: (batch, n_peaks, d_model) — 编码器输出
        """
        x = F.dropout(src_inputs, p=self.residual_dropout, training=self.training)

        for i in range(self.n_layers):
            # 提取当前层的门控权重（如有）
            gw = gate_weights_per_layer[i] if gate_weights_per_layer is not None else None
            # 提取当前层的化学偏置（支持列表或单张 tensor）
            if isinstance(chem_bias, (list, tuple)):
                cb = chem_bias[i]
            else:
                cb = chem_bias

            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self._layer_forward, i, x, src_mask, graphormer_dists, cb, gw,
                    use_reentrant=False
                )
            else:
                x = self._layer_forward(i, x, src_mask, graphormer_dists, cb, gw)

        x = self.scales[-1](x) if self.pre_norm else x
        return x
