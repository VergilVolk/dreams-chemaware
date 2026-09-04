# Noise E4-B2 偏航审计与路线纠偏

日期：2026-09-03  
状态：冻结审计；取代“由 E4-B2 直接继续梯度手术或训练”的解释  
范围：只讨论共享 DreaMS embedding 的噪声微调；P2b、ChemAware 和 P3 不参与

## 一、裁决

E4-B2 的数值结果可保留为探索性零更新诊断，但不得继续标作对原预注册
E4-B2 协议的严格独立确认，也不得据此训练 gradient-surgery student。

本轮没有证明噪声路线失败。它只证明：

1. 对现有 candidate-gradient / role-confounder 动作，加入 clean anchor 可以机械地
   改善动作梯度与 clean 梯度的夹角；
2. 这种处理没有在 B1 未见 formula 上形成可靠的跨 formula 公共方向；
3. B2 没有测试 clean spectrum 是否能够识别动作适用性，也没有测试处理后的更新
   是否改善完整候选图上的 clean retrieval；
4. 因此 B2 不能回答主计划中的 Experiment B（clean-input learnability）或
   Experiment C（full-list counterfactual training）。

立即停止从 B2 派生新的 PCGrad、anchor、scope 或 pooled-gradient 扫描。

## 二、冻结结果的正确读取

### 2.1 数据规模

- screen：复用 B1 的 576 actions，即 9 cells × 2 baseline states × 32 formulas；
- confirm：624 actions；
- 合计：1,200 actions、screen 322 个 unique formulas、confirm 238 个 unique formulas；
- screen / confirm formula overlap 为 0；
- role-confounder 的 confirm error strata 仍然很小，step 2–5 分别只有
  11、9、9、8 formulas。

扩大后的面板改善了计算覆盖，但它是看到原始可用性不足后的协议修订，不再等同于
2026-09-02 文档中“排除 B1 后全局 hash 分 screen/confirm、error/correct 同 formula
配对”的原预注册设计。

### 2.2 三个锁定候选

1. candidate-gradient step 6 + anchor 0.25 + head：
   clean alignment 和 action retention 为正，但 multiplicity-adjusted action margin
   下界为 -0.00276，gradient consensus 下界为 -0.00105；
2. role-confounder step 1 + anchor 0.25 + backbone：
   action margin 下界 +0.00221、clean alignment 下界 +0.00858、retention 下界
   +0.9422；唯一失败项是 consensus，下界 -0.00552；
3. candidate-gradient step 3 + anchor 0.25 + backbone：
   action margin 下界 +0.00315、clean alignment 下界 +0.09486、retention 下界
   +0.9637；唯一失败项是 consensus，下界 -0.00307。

三项均未独立确认。step 3 candidate-gradient 是三项中最完整的局部信号，
role-confounder step 1 是次级信号；step 6 head 同时缺 margin 与 consensus，
不应继续。

## 三、B2 相对冻结总计划的偏航

### 3.1 在 B1 零 cell 通过后继续强制修梯度

根因审计的顺序是：

1. E4-A 三臂因果归因；
2. clean-input action learnability；
3. 通过后才进入完整候选 listwise counterfactual training。

E4-A 已显示 targeted 相对 matched-random 只有 +0.0338 pp、CI 下界为 0。
B0/B1 随后定位出动作 forward advantage 存在，但 pooled paired-advantage gradient
缺少 clean alignment / cross-formula consensus；B1 有 0 个 cell 同时通过三门。

此时正确动作应是检验“哪些作用能由 clean-visible features 预测”，或重写动作机制。
B2 转向 PCGrad/anchor，相当于在监督是否可识别尚未成立时先加工梯度，顺序错误。

### 3.2 post-hoc 改动仍被写成 formal

原 B2 因 24-formula 门失败后，代码改为：

- 复用 B1 formula 作为 screen；
- B1 未见 formula 作为 confirm；
- error/correct 不再要求同 formula；
- confirm 每 stratum 最多 64，最低只要求非空。

这些改动本身可以构成新的探索性设计，但发生在看到样本可用性之后。生成报告仍写
formal=true，没有显式记录 protocol amendment，因此形式声明过强。现有结果应
降级为 exploratory amended audit；原始 report 和 hash 保留，不回写篡改。

### 3.3 B2 优化的是局部 surrogate，不是最终任务

B2 每个 query 只使用 4 个 positive spectra 和 8 个 negative molecules。
forward margin 是 action view 相对 selected matched-random 的局部 margin，
不是经过梯度更新后的 clean query full-list margin。

完整候选图中的候选切换、新 hardest negative、同分子多参考谱聚合和 introduced
errors 均不在 B2 梯度终点中。因此 B2 没有完成总计划要求的 full-list transfer gate。

### 3.4 anchor 指标存在部分构造性通过

anchor 0.25 按原 action-gradient 范数显式加入 clean gradient。随后又以
clean alignment 作为筛选指标，所以 alignment 变正部分来自变换定义本身，而不是
动作语义突然变得可迁移。三个入选配置全部为 anchor 0.25 正是这一现象的证据。

action-descent retention 接近 1 只说明 anchor 没有抹掉原 action gradient；
它不证明一步参数更新会改善 clean retrieval。

### 3.5 scope 不是可执行的完整训练更新

B2 的 head/backbone scope 是把扁平梯度切片后分别计算几何量。它没有组装一个完整
optimizer update，也没有规定未被选中参数块接受 clean、action 还是 zero gradient。
因此 anchor 0.25 + backbone 目前只是诊断标签，不是已定义完整的训练算法。

### 3.6 consensus 门的含义被扩大

leave-one-formula-out cosine 接近 0 说明不存在简单的单一 pooled mean direction。
它是反对“把所有动作压成一个全局向量”的证据，但不是 SGD 成功的必要条件：
高维网络中不同样本梯度近似正交很常见，输入条件化的共享网络也不要求每个 formula
梯度同向。

因此不得从 consensus 失败推导“噪声微调不可学习”；同时也不得忽略 E4-A 已经显示
targeted student 没有显著优于 matched-random。真正缺失的是 clean-input
predictability 和 full-list functional transfer。

## 四、当前应永久停止的低性价比操作

1. 继续扫描 PCGrad、anchor beta、head/backbone scope；
2. 继续降低公式数门槛或用新 hash 重切 screen/confirm；
3. 在 B2 上直接加学习率、epoch、解冻层数；
4. 把 gradient consensus 当成最终模型性能代理；
5. 把局部 4-positive/8-negative margin 当成完整谱库检索；
6. 再训练一个使用正确候选、错误候选或 action outcome 作为推理输入的专家。

## 五、恢复到冻结计划的下一步

### L0：全量 action-learnability 标签账本

对 R0 成熟 action 和其两条 frozen matched controls，在同一 checkpoint、同一完整
候选图上计算：

- target action full-list margin gain；
- 两条 matched-random full-list margin gain；
- paired advantage = target gain - random mean gain；
- action 后 candidate switch；
- positive / zero / harmful；
- corrected、introduced 与 unique coverage。

outcome 只作为监督标签和最终审计终点，不能进入 clean-visible feature。

### L1：clean-input 可识别性审计

输入只允许原始 clean spectrum 可见信息：

- contextual peak tokens；
- m/z、intensity、谱稀疏度、precursor/adduct；
- 原谱中实际存在的 neutral-loss / peak-pair motif；
- acquisition metadata。

禁止输入 identity、正确候选、hardest wrong candidate、target path、action outcome、
P2b 或 P3。使用 formula-crossfit，同时报告 action-family-only 和 label-permutation
负对照。

主终点不是普通分类 accuracy，而是高置信选择后的 held-formula：

1. target-minus-random full-list margin CI 下界 > 0；
2. corrected > introduced 且 risk-net > 0；
3. 足够的 identities/formulas 和 mature-union unique coverage；
4. 每个 false-positive action 的 candidate-switch / introduced-error 来源。

若 L1 不通过，该 action 只保留为反事实解释和 headroom，不进入共享 encoder。

### L2：只有 L1 通过后才训练

训练一个 shared clean-spectrum encoder；selector 只用于训练期 conditional sampler /
no-op routing，不是 embedding 后面的部署专家。每个 microbatch 同时包含：

- clean full-list rank；
- target action full-list rank；
- target-vs-matched-random paired advantage；
- harmful/no-op margin protection；
- 当前 student geometry 下刷新但不 outcome-mine 的 hard negatives。

主比较仍是 clean inference 上 targeted conditional student 相对 matched-random
student，而不是相对 official 的单臂增益。

## 六、下一项授权

只授权先实现 L0 的只读全量账本 preflight 与容量审计。L0 不更新模型、不消费 P3、
不训练 selector。必须先报告可执行 action 数、完整候选覆盖、各 family/step 的
positive/zero/harmful 数量及 formula/identity 覆盖，再决定 L1 的模型与阈值。

在 L0 完成前，不启动 B3 梯度聚合、不训练新 encoder。
