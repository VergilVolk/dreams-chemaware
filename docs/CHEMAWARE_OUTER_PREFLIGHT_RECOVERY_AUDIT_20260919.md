# ChemAware one-time outer: seal 后故障恢复审计

**日期：** 2026-09-19  
**审计性质：** 纯只读 preflight；未创建或删除 lock，未读取 fold 4 真值，未提交 SBATCH，未运行 outer inference。  
**审计对象：**

- `tasks/run_chemaware_truthblind_outer_once.sbatch`
- `tasks/evaluate_chemaware_truthblind_outer_once.py`
- `tasks/test_chemaware_truthblind_outer_once_contracts.py`
- canonical development policy：`data/validation/chemaware_truthblind_candidate_policy/run_2338337/policy/`

## 结论

**不建议现在提交 one-time outer。**

当前实现已经正确冻结了以下科学顺序：canonical/provenance 检查 → role 3 exact replay → 原子创建 seal → truth-blind outer inference → 写出预测 → 首次读取 outer truth → 评价 → 临时目录原子发布。现有四项 ChemAware contract test 也全部通过，canonical 三个关键产物和六个运行时代码文件的固定 SHA256 与入口一致；当前本地未发现 `OUTER_FOLD_4_OPENED.lock` 或 outer 结果目录。因此，fold 4 在当前可见仓库中仍未打开。

但是，当前实现没有 seal 后的正式恢复状态机。seal 写入之后，任何 Python 异常都会删除整个临时目录，而 seal 永久保留；SBATCH 的重入又会因 lock 已存在而退出。尤其是在“预测已计算但尚未发布”或“truth 已读取但 report 尚未发布”时，一次磁盘、内存、节点或序列化故障即可造成：外层已被消费、结果未留下、且不能合法重跑。这个风险在唯一一次外层评价中不可接受。

在补齐并测试 seal 后恢复合同以前，只能登记为：

> `OUTER_READY_SCIENTIFICALLY / BLOCKED_BY_POST_SEAL_RECOVERY`

## 当前写入与揭盲顺序

现有 evaluator 的实际顺序如下：

1. 拒绝覆盖已经存在的 outer output。
2. 检查 release policy、canonical policy、manifest、token cache 和 rule library。
3. 核对 canonical 三件套 SHA、输入 provenance、policy metadata 和两个注册 baseline。
4. 用已消费的 inner role 3 做 exact replay；任一 rank/slot 不一致或 utility 最大误差超过 `1e-12` 时停止，此时不写 seal。
5. 用 `O_CREAT | O_EXCL` 创建 `OUTER_FOLD_4_OPENED.lock`，写入 policy hash、job id、query 数和目标输出路径，并对 lock 文件执行 `fsync`。
6. 在 outer output 的父目录内创建随机临时目录。
7. 执行 truth-blind inference；此函数不接受 `molecule_label`、`query_formula` 或 rank。
8. 将预测写为临时目录中的 `predictions_truthblind.npz`，随后在内存中计算其 SHA256。
9. 第一次调用 `evaluate_ledger`，此时才读取 outer label 和 formula。
10. 写 `evaluation_with_truth.npz` 和 `report.json`。
11. 将整个临时目录以 `Path.replace()` 发布为 outer output。
12. SBATCH 再检查三个文件非空及 report status。

步骤 5、7、8、9 的相对顺序满足 truth-blind 合同；问题不在“提前看真值”，而在 seal 后缺少持久状态与合法恢复路径。

## 故障矩阵

| 故障阶段 | 当前遗留状态 | 当前是否可恢复 | 科学解释 | 事故处置 |
|---|---|---:|---|---|
| seal 前：依赖、SHA、import、contract test、release refit失败 | 新 job 的 run root可能残留；无seal | **可恢复** | outer仍未打开 | 保留日志；修复纯工程问题；用新job重新做全部preflight。不得改变policy、threshold、dose、feature或gate。 |
| seal 前：role 3 exact replay失败 | 无seal、无outer预测 | **可恢复但不得直接提交** | release candidate未精确重放canonical | 只调查环境、版本、数值与artifact漂移；恢复exact replay前不得开outer。 |
| seal创建过程中失败 | lock可能不存在、为空或部分写入；父目录未fsync | **必须人工裁决** | 不能仅凭“文件存在/不存在”判断是否完成开封 | 冻结现场，记录inode/size/hash/mtime和日志；不得删除或重建lock；由独立审计决定是否仍可视为未开封。 |
| seal完成后、outer inference开始前失败 | 有seal；无持久预测 | **当前不可重入** | policy已承诺但truth尚未读取 | 不得删除lock，不得普通重跑。当前代码无合法恢复入口。 |
| outer inference中途失败 | 有seal；预测只在内存；普通异常会删除temp | **当前不可恢复** | 未读取truth，但没有可验证的断点或完整预测 | 保留seal和日志；禁止调参或更换模型；等待预先审计过的same-attempt恢复工具，当前不要人工拼接。 |
| 预测文件写完后、truth读取前发生普通Python异常 | 有seal；`except`删除含预测的temp | **不可恢复** | 本可作为冻结预测的关键证据被主动销毁 | 这是提交前必须修复的首要缺陷。 |
| 预测文件写完后进程被SIGKILL/节点掉线 | 有seal；随机temp可能残留 | **理论上可取证，当前无合同** | 可能存在完整预测，但seal不记录temp路径/预测hash；无法自动证明写完且先于truth | 不得自行移动、评价或重算；只做只读取证，交由独立恢复审计。 |
| truth已读取、评价或bootstrap中失败 | 有seal；普通异常删除temp；无正式report | **科学上最严重，当前不可恢复** | outer truth已经消费，但结果和prediction receipt可能丢失 | 永久停止该outer的常规重跑；不得删除seal、改gate或回到inner调参。只能报告技术事故，除非事前批准的恢复合同能证明使用原冻结预测。 |
| 三个临时产物已写、目录发布前失败 | 普通异常时temp被删除；硬失败时temp可能残留 | **当前无正式恢复** | 可能已有完整结果，但没有稳定路径和完成标记 | 只读取证；不能以人工复制代替原子发布。 |
| `temporary.replace(output)`成功，随后SBATCH文件检查或grep失败 | seal和正式output均存在 | **结果可只读恢复** | evaluator已完成，通常是尾部校验/调度故障 | 禁止重跑；重新计算文件hash并运行独立只读validator。只有三个文件、report status及内部provenance均通过时才登记complete。 |
| evaluator完整成功、节点在最终echo前失败 | seal和正式output均存在 | **可恢复** | 科学结果已经原子发布 | 按上一行只读验证，不因scheduler状态为FAILED而重跑。 |
| 主机/文件系统在rename附近掉电 | 状态可能为temp或output；文件本身未显式fsync，父目录也未fsync | **不保证耐久** | `replace`保证同文件系统命名原子性，但不等于掉电耐久性 | 必须做文件完整性和hash审计；不能仅看目录名。 |

## 当前实现中可靠的部分

1. **seal 使用排他创建。** `O_EXCL` 防止两个任务同时合法开封同一outer。
2. **临时目录与正式output位于同一父目录。** 正常文件系统条件下，最终目录替换是同文件系统原子发布，不会逐文件暴露半成品。
3. **预测先于truth。** `inference_ledger` 的接口和静态合同均排除了label、formula、baseline rank；预测文件在 `evaluate_ledger` 之前写出。
4. **inner replay先于seal。** 环境或release policy不能精确复现canonical时，不会消费outer。
5. **output拒绝覆盖。** evaluator不会静默替换已有结果。
6. **baseline已经冻结。** outer同时评价official、same-feature-direct、nuisance-only和ChemAware；恢复工作不得借机增加模型或改变门。

## 必须在提交前补齐的恢复合同

以下项目全部通过以前，不提交正式SBATCH：

1. **持久attempt manifest**
   - 在seal中固定 attempt id、canonical/release/input/code hash、正式output路径和持久staging路径。
   - seal和attempt manifest都必须原子写入并验证可读；记录创建阶段。

2. **seal后不删除证据**
   - seal成功以后，异常处理不得 `rmtree` staging。
   - staging必须保留只追加的阶段状态与错误记录；不得覆盖已经写出的预测。

3. **预测冻结receipt**
   - prediction必须先写临时文件、flush/fsync、计算hash，再原子改名为持久的 `predictions_truthblind.npz`。
   - 写独立receipt，至少记录policy/input/code hash、query namespace、shape/dtype、prediction hash、完成时间和 `truth_opened=false`。
   - receipt持久化成功以前，禁止读取truth。

4. **truth-open receipt**
   - 首次读取label/formula之前，以排他方式写入 `TRUTH_OPEN_STARTED` 状态，并绑定唯一prediction hash。
   - 评价完成后追加final evaluation/report hash；不得修改已有prediction receipt。

5. **受限恢复模式**
   - 恢复必须由已有seal指定的同一attempt触发，而不是创建新实验。
   - 若冻结prediction及receipt完整，只允许从该prediction继续评价，禁止再次运行policy inference。
   - 若truth已经开始读取而冻结prediction不完整，必须判定为不可恢复事故，不能重跑。
   - 若truth从未读取而prediction不完整，是否允许以完全相同hash的policy重算，必须在首次提交前书面预注册并由独立测试覆盖；不能事故后临时决定。

6. **幂等只读validator**
   - validator只读检查seal、attempt、prediction、evaluation和report之间的hash闭包。
   - scheduler失败但正式output存在时，由validator裁决complete，不重新提交outer。

7. **故障注入测试**
   - 至少在以下位置强制失败并验证状态机：seal前、seal后inference前、预测写一半、prediction receipt后、truth-open receipt后、evaluation写入中、rename前、rename后。
   - 测试只能使用合成数据；不得使用fold 4。

## 提交前检查表

### A. 科学与冻结边界

- [ ] `OUTER_FOLD_4_OPENED.lock` 不存在；检查只允许测试存在性，不读取任何outer真值。
- [ ] 不存在已发布的 `outer_fold_4`；若存在，立即停止并转事故审计。
- [ ] canonical policy、report、inner ledger SHA与注册值一致。
- [ ] 六个runtime文件SHA与SBATCH固定值一致。
- [ ] manifest、official embeddings、rule library provenance与policy report一致。
- [ ] release metadata的所有frozen key与canonical一致。
- [ ] baseline集合严格为 `same_feature_direct` 和 `nuisance_only`；不添加新baseline。
- [ ] exact inner replay在目标服务器环境通过，rank/slot位级一致，utility误差 `<=1e-12`。

### B. 恢复与持久化

- [ ] 上述seal后状态机、prediction receipt、truth-open receipt和只读validator已经实现并用合成数据通过故障注入测试。
- [ ] 已书面确定“seal后、truth前、prediction不完整”是否允许same-attempt deterministic recomputation；事故后不得再选择规则。
- [ ] 持久staging与正式output位于同一文件系统。
- [ ] seal、staging、output和日志目录权限正确；排他创建与原子rename在目标文件系统上经过测试。
- [ ] 服务器剩余空间和inode足以同时容纳release refit、中间kernel/cache、完整staging和正式output，并保留安全余量。
- [ ] 不依赖shell trap作为恢复记录；进程被SIGKILL或节点掉线后仍能从持久状态判定阶段。

### C. 运行环境

- [ ] 使用唯一单GPU SBATCH；无array、无手工memory覆盖。
- [ ] 目标环境的Python、NumPy、scikit-learn、PyTorch、joblib版本被记录。
- [ ] 四个contract test和compile检查在allocation内通过。
- [ ] `run_2338356/release_policy`若复用，先经过同一provenance、metadata和exact replay；不存在时仅允许按现有代码在roles 0--2重拟合。
- [ ] 18小时walltime经过无outer真值的资源估算确认充分；不要用正式outer测试时间预算。
- [ ] 指定一名独立事故裁决人；运行agent无权删除seal、修改gate或决定重跑。

## 事故处置硬规则

1. **永不删除、改名、覆盖或伪造seal。** seal存在即表示one-time outer流程已经承诺启动，不以Slurm最终状态为准。
2. **永不因技术失败重新训练、改threshold、改dose、改feature、换baseline或看outer后修改代码。**
3. **先冻结现场再诊断。** 保存stdout/stderr、scheduler状态、job id、进程退出原因、seal、run root、staging/output目录清单、大小、mtime和SHA；所有调查先只读。
4. **正式output存在时不重跑。** 只运行预注册的只读validator；通过则登记完成，不通过则登记事故。
5. **seal存在而正式output不存在时不自动重跑。** 根据持久阶段receipt裁决；当前版本没有receipt，因此此状态一律视为阻塞事故。
6. **truth-open以后不得重新生成prediction。** 只能评价seal绑定的原冻结prediction；若该预测不存在或无法验证，实验不可恢复。
7. **科学失败与技术失败分开。** 六个注册科学门未全过是有效阴性结果；缺文件、hash不闭合、半写入或无法证明预测先于truth是技术事故，不能解释为科学阴性。
8. **outer只裁决一次。** 即使 `scientific_pass=false`，也不得以“修bug”为名在同一fold重新选择方法；真正实现错误须作为限制报告，并转向新外部数据。

## 提交建议

**当前建议：NO-GO。**

阻塞原因只有一个，但属于致命级：seal后的预测与评价没有持久、可验证、幂等的恢复路径；正常Python异常还会主动删除唯一临时证据。科学合同本身已经足够完整，不需要增加baseline，也不应继续改模型。下一步应仅由独立工程agent在合成数据上补齐恢复状态机和故障注入测试；完成后再次进行纯只读GO/NO-GO审计。只有恢复检查表全部通过，才建议提交现有one-time outer。

本审计没有评价fold 4性能，也没有改变任何已冻结科学门。
