# ChemAware `SIMULATION_CHALLENGE` 字段勘误与工件处置

**日期**：2026-09-02  
**范围**：只影响 ChemAware / G8R 中对 MassSpecGym 字段的解释，不改变谱图本身和已记录的数值结果  
**核心勘误**：`SIMULATION_CHALLENGE` 是谱图是否满足 MassSpecGym spectrum-simulation benchmark 子集要求的布尔标记，不是“实验谱/模拟谱”的来源字段

## 1. 禁止继续使用的解释

以下推论全部无效：

- `False = 真实实验谱`；
- `True = in-silico 模拟谱`；
- 只保留 `False` 可以提高谱图真实性；
- `True→False` 可以称为 sim-to-real；
- 某个结果含较多 `True` 谱就说明存在 simulation-specific 增益。

该字段仍可作为 benchmark membership 分层变量报告，但不得用于来源、真实性或数据质量判断。当前 HDF5 没有可替代它的原始来源库 provenance 字段。

## 2. 工件分级处置

### A. 已修正并可继续使用

- `data/reference/unified_v3/`：两类 membership 均纳入，MassSpecGym 代表谱保留仪器、碰撞能、fold 和 membership；
- `data/validation/g8r_p3_allow_recovered_corrected_v3_20260902/`：使用全部 train 身份减去已消费 P3 身份；
- `data/e1/chemaware_control_train_mh_triplet_pool_10ppm_p3disjoint_v3.npz`：修正后的 `[M+H]+` identity-continuation 控制池；
- `data/e1/chemaware_control_train_mna_triplet_pool_10ppm_p3disjoint_v3.npz`：修正后的 `[M+Na]+` identity-continuation 控制池；
- `data/validation/chemaware_training_lineage_audit_v3/report.json`：当前训练谱系审计。

### B. 数值可复现，但只能改名为 restricted membership cohort

- 已消费 P3 中原名含 `real` 的主面板：应解释为 `SIMULATION_CHALLENGE=False` membership 子集；
- 原名 `P3-sim-to-real-secondary`：应解释为 membership=True query 对 membership=False gallery 的压力面板；
- `condition_invariance_benchmark`、`cross_condition_m3` 和旧 noise/positive-pair 工件中写明 `False only` 的结果；
- 本地目录名含 `cached_real` / `local_real` 的图和训练诊断。

这些结果不能据此声称实验来源纯度，但谱图、身份、加合物、质量窗口和条件差异本身的数值结果不因字段勘误自动失效。

### C. 不得作为新 formal 入口

- `data/reference/unified_v2/`：构建时误删 119,029 条 membership=True 合法谱，只能做迁移回归；
- 旧 `p3_p2_allowed_training_ik14.json` 中的 `real_train_primary` / `simulation_train_optional`；
- 从旧 `real_train_primary` 构建的 23,876-query `g8r_error_atlas_listwise_cache.npz`；
- 绑定该图的 G0 rule cache、token cache、shared-v2/v3 preflight 和 formal 提交链。

`tasks/preflight_chemaware_shared_v2.py`、`tasks/build_g8r_p2_listwise_cache.py` 和 `tasks/submit_chemaware_shared_v3_formal.sh` 已默认 fail-closed。旧 cohort 只能显式用于历史复现，不得输出新的 formal 主张。

## 3. 已消费 P3 的处理原则

不重建、不替换、不假装没有消费过。旧 P3 仍是一个冻结的 membership=False 子集评价，其既有结果按该限定重新解释。训练身份排除继续取所有 P3 面板中已经暴露的身份并集；修正后的本地重建得到：

- P3 已暴露：7,606 queries / 4,219 identities；
- P3-disjoint train：137,830 rows / 19,403 identities；
- membership=True：80,556 rows；
- membership=False：57,274 rows；
- 与 P3 身份交集：0。

该 allow-list 是从本地已消费审计恢复的语义修正版，不宣称与缺失的服务器 seal 字节一致。

## 4. 新工件的强制合同

任何新 ChemAware 数据构建器必须：

1. 明文声明 `SIMULATION_CHALLENGE` 只作 benchmark membership；
2. 默认不按该字段筛谱；
3. 若为了 benchmark 分层使用它，名称中必须写 `member/nonmember`，不得写 `real/simulated`；
4. 来源或跨来源主张必须来自独立 provenance 字段；字段缺失时必须报告“来源未知”；
5. P3-disjoint 训练入口只接受 `train_primary_all` schema，遇到 `real_train_primary` 必须失败关闭；
6. 旧图、旧缓存和旧 preflight 哈希不得带入新 formal 训练。

## 5. 相关审计

- 化学库合同：`data/validation/chemaware_chemical_library_contract_audit_v3/report.json`
- v2→v3 迁移：`data/validation/chemaware_reference_library_migration_v2_to_v3/report.json`
- 训练谱系：`data/validation/chemaware_training_lineage_audit_v3/report.json`
- 科学问题重审：`docs/CHEMAWARE_CHEMICAL_LIBRARY_REASSESSMENT_20260902.md`
