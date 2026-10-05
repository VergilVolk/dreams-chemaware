# ChemAware 跨谱边界事件 PEFT：现有证据、失败归因与执行决策

日期：2026-09-13  
状态：本地数据重放、数学单元测试、五臂预检和单 GPU 提交契约通过；GPU 性能结果尚未产生。  
结论边界：不得把教师动作性能、oracle 上限或训练损失下降表述为共享 embedding 已提升。

## 1. 目前真正成立的结果

### 1.1 单谱候选化学动作已经成立

冻结产物：`data/validation/chemaware_orthogonal_rule_residual_policy_v4/report.json`

- 当前代码重复运行 v3/v4 得到相同结果，排除了随机漂移。
- 分子式留出开发折上，correct arm 的 Recall@1 相对官方 embedding 提升 2.6439 pp。
- 61 个查询被纠正，10 个查询被引入错误。
- 分子式簇 bootstrap 的 95% CI 为 `[+1.7442, +3.6045] pp`。
- correct 相对 reversed 提升 3.7325 pp，相对 alignment-permuted 提升 1.8144 pp；两者区间下界均大于 0。
- outer fold 4 未使用。

这证明化学规则能够在候选层产生有方向、有因果对照支持的纠错动作；它不等于共享 embedding 已经获得同等提升。

### 1.2 跨重复谱图的一致动作已经成立

冻结产物：`data/validation/chemaware_crossview_consensus_teacher_v2/report.json`

- 教师阈值只用 formula roles 0/1 选择，role 2 用于确认。
- role 2 共 2,413 个查询；correct arm 为 58 个纠正、11 个引入错误。
- Recall@1 提升 1.9478 pp，分子式簇 95% CI 为 `[+1.021, +2.911] pp`。
- Recall@3、Recall@5、Recall@10 分别提升 0.5387、0.1658、0.0829 pp。
- reversed arm 为 -0.7045 pp，说明动作方向不是可任意替换的相关性信号。

这证明同一分子不同谱图之间存在可重复的化学动作证据，可作为训练期 privileged teacher；部署时不能输入分子身份、候选或其它谱图。

### 1.3 已有动作仍有未释放的互补潜力

冻结产物：`data/validation/chemaware_parent_crossview_consensus_v1/report.json`

- 父级共享核本身：+1.7406 pp，88 个纠正、46 个引入错误。
- 父核加跨谱正确动作的确定性组合：+2.4865 pp，116 个纠正、56 个引入错误。
- 用真值事后选择父核或跨谱动作的 oracle：+4.7659 pp，116 个纠正、1 个引入错误。

4.7659 pp 只是错误集合的 oracle 可分上限，不是可部署方法的结果。它说明后续核心问题是学习“何时执行、何时拒绝”，而不是继续枚举更多无约束加分核。

## 2. 已被证伪的注入方式

### 2.1 后置全局残差适配器

`chemaware_orthogonal_event_shared_adapter_smoke_v2` 把训练纠错约束满足率做到 75.93%，但 held fold 增益只有 +0.7258 pp，且比 clean-only 少 0.0518 pp。

### 2.2 显式规则条件后置适配器

`chemaware_orthogonal_event_rule_conditioned_smoke_v1` 同样满足 75.93% 训练事件，held fold 只有 +0.3110 pp，低于 clean-only 的 +0.5184 pp。

### 2.3 更多重复动作加后置适配器

`chemaware_crossview_event_shared_adapter_smoke_v1` 的事件损失从 0.12909 降到 0.01649，满足率 63.53%，但 role-2 Recall@1 仅 +0.0829 pp，置信区间跨 0。

共同结论：动作可拟合但不迁移。单一后置映射只能改变已经压缩后的 1,024 维向量，无法重新组织峰 token、碎片关系和注意力路径。继续调后置 MLP 的学习率、轮数或维度没有科学依据。

## 3. 当前方法：多视图、候选中心的边界事件 PEFT

令共享谱图编码器为

\[
z_\theta(x)=\frac{f_\theta(x)}{\lVert f_\theta(x)\rVert_2}.
\]

一个候选分子可能有多张参考谱。对冻结官方 embedding 选出的最多 4 张竞争性参考谱，训练时重新计算当前学生的平滑最大分数：

\[
S_\theta(q,c)=\tau\left[
\log\sum_{r\in R_c^{(4)}}\exp\left(\frac{z_\theta(q)^\top z_\theta(r)}{\tau}\right)
-\log |R_c^{(4)}|
\right],\quad \tau=0.002.
\]

减去 `log |R|` 防止参考谱较多的分子仅靠数量获得优势；低温使它接近评估时的 exact max，同时允许微调后的获胜参考谱发生切换。旧实现固定一张官方获胜参考谱，训练边界会随参数更新而过期。

对于 correct teacher 识别的事件：

- correction：官方 top-1 错误，化学动作把真值候选推到 top-1；
- protection：官方 top-1 正确，化学动作提出了会破坏正确结果的挑战者；
- 其余 selected 但不直接触碰真值-vs-top1 边界的动作，化学梯度严格为 0。

定义当前候选边界

\[
m_\theta(q)=S_\theta(q,c^+)-S_\theta(q,c^-).
\]

目标边界为

\[
t_i=\begin{cases}
\epsilon,&\text{correction},\\
\max(\epsilon,m_0-\delta),&\text{protection},
\end{cases}
\]

其中 `epsilon=0.005`，`delta=0.002`。化学损失为单侧 Huber：

\[
L_{chem}=\frac{1}{|A|}\sum_{i\in A}
H_\beta\left([t_i-m_\theta(q_i)]_+\right).
\]

因此一个官方 margin 为 -0.11 的纠错事件需要约 +0.115 的真实位移；不会再被旧实现的固定 0.05 action cap 截断。

基础目标仍为 identity-balanced clean listwise retrieval、官方正确边界保护和 embedding preservation。化学梯度 `g_c` 相对基础梯度 `g_0` 做单侧投影和范数限制：

\[
\tilde g_c=\begin{cases}
g_c-\frac{g_c^\top g_0}{\lVert g_0\rVert^2}g_0,&g_c^\top g_0\le 0,\\
g_c,&g_c^\top g_0>0,
\end{cases}
\]

并限制 `||g_c|| <= 0.25 ||g_0||`，最终更新为 `g_0 + g_c`。这只是一次更新的一阶安全约束，不能替代 held-formula 评估。

## 4. 五臂归因设计

所有臂使用同一官方初始化、同一 clean schedule、同一事件采样随机数、同一 PEFT 容量和同一冻结前缀缓存。

1. `zero_contrast`：只有 clean 目标，测普通 PEFT 自身。
2. `matched_random`：85 correction + 16 protection，与 correct 完全同事件剂量；按官方 margin、候选数、身份重复谱数和 baseline rank 无放回匹配，并排除 correct teacher 身份。当前最大 SMD 为 0.1449，小于 0.20。
3. `reversed_contrast`：化学方向反转。
4. `alignment_permuted`：破坏 query-candidate 化学对齐。
5. `correct`：正确化学动作。

若 correct 只胜过 zero，却不能胜过 matched-random，则结论只能是“通用难例课程有效”，不能归因于化学规则。若它不能胜过 reversed/alignment，则化学方向或对齐也未转移。

## 5. 评估与通过标准

- formula roles 0/1 参与优化；role 2 完全不参与梯度；outer fold 4 保持封存。
- 所有臂保存 Recall@1/3/5/10/20/50、MRR、micro-AUC、macro-query AUC、macro-formula AUC、top-1 margin。
- Recall@1 和 AUC 使用分子式簇 bootstrap。
- correct 的 Recall@1 绝对增益 CI 下界必须大于 0。
- correct 的纠正数必须大于两倍引入错误数。
- correct 相对 zero、matched-random、reversed、alignment 的 Recall@1 差值 CI 下界都必须大于 0。
- MRR、micro-AUC、macro-formula AUC 必须增加；Recall@3/5/10/20/50 不得下降。
- 当前同分子式候选组最大为 21，官方 Recall@20/50 已为 1.0；这两个饱和指标只能检验“不下降”，不能证明增效。真正的 R@20/50 增益需要另建宽候选或全库检索评估。

## 6. 资源与故障恢复

- 唯一入口：`sbatch tasks/run_chemaware_crossview_event_peft.sbatch`
- Slurm 请求严格为一个 GPU，不手工指定内存。
- 五个臂在一个进程中依次运行，只构建一次 24,812 谱图的冻结前缀缓存。
- 执行顺序为 clean、difficulty-matched random、correct；若 correct 的绝对分子式簇区间不为正，或不能以正区间胜过前两者，作业立即形成可审计 FAIL 产物并跳过 reversed/alignment，避免继续消耗两个无决策价值的 GPU 臂。
- PEFT 使用仓库既有 G1 容量审计支持的最后一层 rank-4/alpha-4、`lr=2e-4` 和 3 epochs；同学习率 rank-8 的既有审计未选中任何非零 epoch，因此不采用。
- 每个臂完成后立即保存 `result.json`、`held_predictions.npz` 和 `peft.pt` 到 `.partial` 目录；节点故障或超时不会抹掉已经完成的臂。
- 五臂全部完成后，才原子重命名为正式 `run_<jobid>` 目录。
- 对每个非零化学更新都复用 Noise direct-v3 的 virtual-AdamW 反事实审计；要求化学可归因更新占完整更新的 p10 不低于 0.10、可归因更新与保留化学梯度的余弦 p10 不低于 0.05、全局裁剪触发率不高于 0.10，且虚拟步复现真实 AdamW 的相对误差不高于 `1e-3`。

## 7. 文献与现有代码给出的边界

- DreaMS 官方微调路径允许使用预训练谱图编码器进行下游微调；本实验保留单谱输入和共享 query/reference 编码器。[DreaMS fine-tuning documentation](https://dreams-docs.readthedocs.io/en/latest/tutorials/fine_tuning.html)
- 多视图质谱学习支持同一分子的多张谱图共同提供监督，而不是把一张固定谱图永久当作代表。[MVP](https://pmc.ncbi.nlm.nih.gov/articles/PMC12642559/)
- 训练期 privileged information 可以通过蒸馏约束学生，而部署时丢弃教师输入；这正是候选化学动作在本方法中的角色。[Phuong and Lampert, LUPI/KD](https://proceedings.mlr.press/v97/phuong19a/phuong19a.pdf)
- 可微排序与 top-k surrogate 支持直接围绕检索决策边界优化，而非回归一个无尺度保证的教师 utility。[Adaptive Rankmax](https://proceedings.neurips.cc/paper/2020/file/070dbb6024b5ef93784428afc71f2146-Paper.pdf)
- 梯度冲突处理只能作为优化层安全手段，不能修复教师动作、训练目标或评估目标的错位。[Conflict-Averse Gradient Descent](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)

## 8. 当前决策

先运行这一个五臂 PEFT 归因实验，不同时启动父核并集、重排器或新的超参数矩阵。原因不是其它路线没有潜力，而是当前实验第一次同时解决了：动作已验证、边界量纲对齐、多参考谱动态聚合、保护事件、matched-random 对照、共享原始谱图编码器和故障恢复。

若 correct 通过全部门槛，再将父核的独有纠错事件加入同一 PEFT 框架，并做 `correct-crossview`、`correct-parent`、`correct-union` 的增量归因；若 correct 不胜 matched-random，下一步应改进化学动作的可识别路由，而不是扩大 PEFT 容量；若所有事件臂都不胜 clean，则共享表示的可迁移性仍是瓶颈，应转向训练期关系蒸馏或部署期重排，但不能宣称化学 embedding 成功。
