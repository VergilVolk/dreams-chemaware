# Contrast Atlas v10 完整候选轴：Full23 执行就绪记录（2026-10-31）

## 科学目的

当前系统级 unresolved 增量实验只使用 WSE reciprocal edges，结果未通过。下一步不是继续调整 WSE，而是在同一冻结的完整候选轴上，让每个算法独立评分、独立选择和独立形成 reciprocal edges，再重算完全相同的研究级终点。

冻结轴包含：

- 16 个 study sources；
- 55,127 个 observed entities；
- 826,905 个 query × target-source blocks；
- 141,332 个非空候选块；
- 294,236 条 directed candidate pairs。

目前正式完整轴交付覆盖为：WSE `294,236/294,236`；Official、Noise、Full23 均为 `0/294,236`。因此现在不能宣称统一算法在 Atlas 中成功或失败。

## 新的唯一执行入口

```bash
sbatch tasks/run_contrast_atlas_v10_complete_axis_full23_1gpu.sbatch
```

### 发布级 provenance 加固

- Full23 release manifest 必须逐字节匹配 SHA-256
  `5c4c879de220f67e632c2f15c9d186dacd4d22d7f26eb2efd24e474ab510114e`；
  preflight 不再从一个未冻结的 manifest 接受任意新 checkpoint 哈希。
- 正式作业先校验并在 node-local 目录安装官方 `ms_entropy==1.5.2`
  wheel（SHA-256
  `f0a9a83359118f0fdb3260e1a6ba891ba51f76523bbfa0310e065c9ee5b9e7a7`），
  再执行 preflight；服务器共享 conda 环境中已有的其他版本不会被静默使用。
- release 冻结的 `enveda_unified_candidate_integrator_core.py` 与当前 runtime
  字节不同。该差异仅为未被 Full23 v1 推理调用的训练期正则函数增加一个默认关闭的选项；
  其两端 SHA、适用范围和强制 golden replay 已冻结在
  `deploy/unified_additive_full23_v1/runtime_equivalence_addendum_20261031.json`。
  这是一项明确登记的推理等价例外，不再表述为“所有 runtime 字节与 release 完全相同”。
- encoder、完整轴 scorer、packager、score contract 和 golden replay test
  均由 preflight 钉死 SHA，并写入 `preflight.json`。这些记录只证明运行字节和输入完整，
  不证明算法表现或生物学有效性。

该入口在任何 GPU 推理前执行 fail-closed preflight：

- 校验候选轴、16 个 MGF、source manifest 和全部模型工件的 SHA-256；
- 要求 `ms_entropy==1.5.2`，保持 Full23 冻结 WSE 语义；
- 对 55,127 个冻结实体生成 Official、Noise V1、Stage-1、Phase-A、FullMSG embedding；
- 在全部 294,236 pairs 上计算 WSE、P2b、MSG、各 encoder cosine 和 Full23；
- 每个方法在完整候选块上独立决策，不能继承 WSE 已选边；
- 保存 raw score，不截断为概率；任何必要证据缺失使整个块弃权；
- 运行 Full23 golden replay，再输出逐方法 pair scores、block decisions 与 reciprocal edges。

ChemAware V2 在缺少候选结构时按冻结合同为 unavailable no-op；BioAware 在没有 event-level 样本上下文时不伪造。Full23 是 closed-set 排序器，singleton 或 unique Top-1 仍只是身份假说，不是开放集身份概率。

## 当前核验状态

- 新脚本均通过 `py_compile`。
- 完整轴打包与独立决策测试：6 tests passed。
- Full23 release golden replay：PASS。
- 本机 `ms_entropy==1.5.3` 且缺少多项冻结 checkpoint，因此本机 preflight 应失败；正式运行只能在服务器 `dreams` 环境中、且所有工件哈希完全匹配时开始。
- 本任务与所有后续结果均为 `development/consumed`，不是独立外部验证。

服务器运行结束后，生物端只复用其冻结 score packages，在相同 known split、response operator、eligibility 和分母上重算 cross-fit endpoint。不得修改终点来迎合 Full23 结果。
