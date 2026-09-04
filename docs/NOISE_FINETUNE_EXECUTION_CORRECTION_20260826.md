# 噪声微调执行纠偏与最终决战方案

**日期：** 2026-08-26  
**状态：** 唯一主线重新锁定为“原始谱图 -> 同一编码器 -> 新 embedding”  
**适用范围：** 仅讨论 embedding-space 噪声微调；P2b、C2-C、模块2暂时封存

---

## 1. 本次纠偏结论

过去一轮把 D1b clean-only adapter 当成了噪声微调主结果。这个执行口径错误。

- D1b 确实输出了新的 query embedding，但没有读取任何噪声视图、C1 正例教师、A4 动作教师或化学规则。
- D1b 只能作为“adapter 能否安全承接 clean 候选组监督”的中期桥接实验。
- P2b 与 C2-C 均属于 embedding 后的候选重排/残差模块，不得再占用当前噪声微调预算。
- 现在唯一主任务是把已经验证的 P-arm 与 N-arm 训练期教师迁移进 clean 原始谱图的 embedding。

不得再把以下三类数字混为模型性能：

1. 动作/教师上限；
2. 下游重排器增益；
3. 真正更新编码器后、clean query 上的检索增益。

只有第 3 类能够回答噪声微调是否成功。

---

## 2. 立即封存的旁路线

### 2.1 P2b

定位：官方或新 embedding 后的候选组专家。  
用途：最终作为正交组合和峰级谱学证据模块。  
当前决定：保留工件、结果和代码；在噪声 embedding 冻结前不再优化。

### 2.2 C2-C / P2b + RAW/token residual

定位：P2b 后的残差修正。  
当前决定：负结果归档；不得用于判断峰 token 能否改善 embedding。

### 2.3 D1b clean-only query adapter

定位：安全桥接基线，不是噪声模型。  
当前初步结果：三个 seed 总体净修正分别为 +19、+19、+10；平均 Recall@1 约 +0.067 pp；第三个 seed 的 near 净减少 6 个查询，因此未通过三 seed near 一致性门。  
当前决定：保留为 clean 对照和初始化参考；不把它包装成最终噪声微调。

---

## 3. 七项强制核查

### 3.1 协议是否偷换

D1b 是 strict-10ppm、same-adduct、已知正确候选存在的 formula-OOF 检索开发实验，不是最终未知代谢物注释。正式噪声模型仍需独立封存评价。

### 3.2 模型是否真正改变 embedding

正式模型必须对任何单张原始谱图输出候选无关的新 embedding。查询谱和谱库谱必须使用同一套新编码器，禁止只移动查询、继续使用旧候选空间后宣称形成统一 embedding space。

### 3.3 是否真的使用噪声

正式训练必须至少包含一个经过 OOF 审计的 P-arm 或 N-arm 教师。仅使用 clean cross-entropy、preservation 或 distillation 的实验统一标记为 clean baseline。

### 3.4 优化设置是否有依据

adapter、最后一层 Transformer 使用分离学习率；所有学习率只能在 inner formula folds 选择。不得把峰动作效应大小称为优化器“最佳学习率”。

### 3.5 动作空间是否被真正消费

动作矩阵不是图表终点。C1 的支持互斥正例、role-confounder、candidate-gradient 50% 动态动作和 A4 OOF 安全动作必须变成教师分布或训练视图，才能算进入微调。

### 3.6 训练了什么层

每个实验必须报告：总参数、adapter 参数、解冻 Transformer block、projection head 状态。若七层全部冻结，只能称 adapter 训练。

### 3.7 修正与新增错误是否可解释

每个 OOF query 必须保存旧/新正例分数、最难负例分数、负候选身份、MCES、候选谱数量、仪器/CE、峰权重变化、规则证据和动作来源。没有逐 query 转换表，不允许解释新增错误原因。

---

## 4. 噪声微调的唯一科学问题

给定原始谱图 `q`，训练期可以使用真实身份、同身份支持谱和经过审计的峰干预；推理期只能输入 `q`：

\[
q \xrightarrow{E_\theta} z_\theta(q).
\]

目标是让 `z_theta(q)` 同时满足：

1. 跨仪器/碰撞能的同分子谱仍然接近；
2. strict-10ppm 近质量负候选与 near 异构体更容易分开；
3. 原本正确的 query 不被噪声训练破坏；
4. 训练期改错视图的候选分布被迁移到 clean query；
5. 查询和谱库由同一个编码器产生 embedding。

噪声的角色是训练期干预，不是结构标签，也不是推理时必须执行的预处理。

---

## 5. 最终模型结构

### 5.1 对称编码器

\[
z_\theta(x)=\operatorname{Norm}[z_0(x)+A_\theta(H_x,mz_x,I_x)].
\]

- `z0`：官方 fine-tuned DreaMS embedding；
- `Hx`：官方最后一层 peak tokens；
- `Aθ`：零初始化 peak-token adapter；
- 查询、正谱和负谱统一经过 `Eθ`；
- 候选分子多谱聚合协议在训练与评价中保持一致。

第一阶段只训练 adapter。若真实噪声教师能够改善 inner folds、但 adapter 容量成为明确瓶颈，才允许解冻最后一个 Transformer block；不得直接全量解冻。

### 5.2 P-arm：正证据恢复

使用 C1 的 80,250 个支持谱互斥样本：

- evaluation positive row 不得出现在 prototype 支持谱中；
- clean query 为学生输入；
- 同身份支持谱形成 stop-gradient prototype/候选分布；
- 重点覆盖 positive-deficit、cross-condition 与 near 查询。

### 5.3 N-arm：混淆证据抑制

按固定优先级使用动作矩阵：

1. role-confounder 高精度动作；
2. candidate-gradient 50% 动态动作；
3. A4 formula-OOF、低伤害且正效用动作；
4. 25/50/75% 软衰减；
5. identity-only 峰永久保护；
6. shared-only 与全局 100% 删除禁止作为默认动作。

被干预谱只负责生成 stop-gradient 改错分布；部署学生仍接收原始 clean query。

### 5.4 Safety replay

必须显式过采样：

- baseline-correct query；
- A4 中被动作损害的正确对照；
- D1b 的 introduced queries；
- near 异构体和小 margin 查询；
- 跨条件同身份正对。

---

## 6. 训练目标

\[
L=L_{clean-group}
+\lambda_P KL(\operatorname{sg}\pi_T^P\Vert\pi_\theta(q))
+\lambda_N KL(\operatorname{sg}\pi_T^N\Vert\pi_\theta(q))
+\lambda_S KL(\pi_0\Vert\pi_\theta(q))
+\lambda_C L_{cross-condition}
+\lambda_E L_{preserve}.
\]

关键约束：

- `pi_theta(q)` 必须由 clean 原始谱图产生；
- P/N 教师分布只在训练期存在；
- safety 使用原本正确查询的 margin 与候选分布，而不只约束 embedding cosine；
- formula OOF、identity-equal weighting 和 ties-against-positive 保持不变；
- P2b 分数不得进入任何损失、教师或标签。

---

## 7. 最终执行顺序

### F0：协议与实现锁定

一次性核验：

- P3/新测试 query 身份与所有训练 query/正谱/负谱身份关系；
- 查询与候选使用同一 encoder；
- zero-init 时逐 query 完全复现官方排名；
- D1b introduced/corrected 逐 query 错误图输出；
- 训练/评价多谱聚合一致；
- 不含 P2b 字段。

### F1：P-arm 真正噪声教师迁移

不再运行新的 clean-only 主实验。以官方 zero-init 为起点，单独加入 C1 P-arm。

通过门：

- positive-deficit 与 cross-condition 风险净收益 CI 为正；
- overall 不劣；
- near 不显著下降；
- 三 seed 至少方向一致；
- clean query 的新 embedding，而非教师视图本身，产生改善。

### F2：N-arm 真正峰动作迁移

固定 F1 结构，分别运行：

1. confounder-only；
2. confounder + candidate-gradient 50%；
3. 加入 A4 OOF 安全扩展动作。

每一步都必须比较 corrected、introduced、negative-excess、near 和 safety replay。高修正但高新增的动作不得进入下一步。

### F3：P+N 双臂

仅组合已经单独过门的 P/N 分支。固定权重，不在同一 outer fold 大规模搜索。

### F4：容量升级

只有在教师方向有效、adapter 学生迁移不足且证据表明容量受限时，解冻最后一个 Transformer block。使用 adapter 较高学习率、last block 低学习率和强 preservation；禁止全量七层直接解冻。

### F5：ChemAware 规则辅助

噪声模型通过后，才比较：

- 无规则；
- 规则用于峰保护/动作风险；
- 规则概念解码辅助。

规则不得定义正负样本或 embedding 距离。

### F6：冻结模型与独立评价

冻结单一模型和统一谱库 embedding，评价：overall、near、跨条件、错误臂、corrected/introduced、MRR、macro query AUC、formula/scaffold 泛化。之后才与 P2b 做正交组合。

---

## 8. 立即禁止事项

- 禁止继续优化 P2b、RAW/token residual 或候选后处理；
- 禁止把 clean-only adapter 称为噪声微调；
- 禁止只变换查询、不变换谱库后宣称统一 embedding space；
- 禁止用动作 oracle 或身份教师上限冒充学生模型成绩；
- 禁止因为单 seed 或单折小幅提升就进入封存测试；
- 禁止继续训练却不输出逐 query 新增错误原因；
- 禁止同时加入 P、N、规则、概念头和最后层解冻，导致无法归因。

---

## 9. 当前唯一下一步

先完成 F0 工程合同和 D1b 逐 query 转换审计；随后直接实现 F1 的 C1 P-arm 教师蒸馏。F1 必须输入 clean 原始谱图并输出能够同时编码查询和谱库的新 embedding。完成 F1 前，不再开展任何 embedding 后专家模块。
