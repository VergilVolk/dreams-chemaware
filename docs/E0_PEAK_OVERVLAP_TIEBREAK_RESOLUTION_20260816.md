# E0 峰重叠：B/A 矛盾已解析（top_overlap 的 tie-breaking 假象）

## 结论（一句话）
`top_overlap`（raw，离散分数）**不能单独当排序信号**——它在 B（有向 pair）里
"赢 cos +4.6pp" 是 **tie-breaking 假象**；套到官方全量检索协议后 **输给 cos
-1.5pp**。真正有效的峰纠错是 **cos+4 峰特征的逻辑回归（连续分数）**，+2.04pp，
这个是真的。

## 矛盾现象（当时无法解释）
- B（npz 有向 pair，query-clustered CV）：`topov_only` R@1 = **0.9482**，cos = 0.9020 → topov 大赢。
- A（官方 10ppm 全量检索）：`topov` R@1 = **0.8857**，cos = 0.9003 → topov 反输。
- 同一批 21,163 eligible query，topov 差了 6.25pp，cos 只差 0.17pp。不可能。

## 定位过程（逐 query 对比，非猜测）
写 `tasks/debug_b_vs_a.py`，对 500 个随机 query 同时用两种方式算 topov rank：
- 候选集 **完全相同**（`b_set == a_set`，sets same=True）；
- topov 函数完全相同（A 的 `top_overlap` == B 的 `pair_peak_features(...)[4]`）；
- 唯一差异：rank 不同（31/500 = 6.2%），全部是 B=1、A=2。

进一步验证（对 A 侧 query 不是 top-1 的 54 例）：
- **30 例是精确同分**（query topov 与第一名完全相等，如 1.0=1.0、0.9=0.9）；
- 24 例是真实更低（topov 本身噪声大）。

## 根因：`sorted()` 是稳定排序，同分靠「插入顺序」破平
- `top_overlap = top_matches / min(10, n_a, n_b)`，top_matches ∈ {0..10}，分母 ≤10 →
  **最多几十个离散取值**，同一 query 下大量候选同分。
- B 的 npz 拍平（eval_e0_baseline.py `evaluate_pairs`）把 **pos（同 IK）候选先写入、
  neg 后写**，所以 query 自己的 IK **永远第一个插入 dict** → 同分时稳定排序把它留在 rank 1。
- A 的官方协议按 **manifest 索引升序**枚举候选，pos/neg 交错 → 同分时破平任意，
  一部分 tie 判给错误 IK → rank 2。
- 结果：B 系统性高估 topov（0.9482），A 才是 honest 值（0.8857）。6.25pp ≈ 6.2% 的
  查询存在「top-1 tie」，完全吻合。

## 为什么 cos 不受影响
cos 是**连续**分数（float 点积），同分几乎为零 → B/A 只差 0.17pp（罕见 float tie）。

## A 的诚实结论（官方全量检索，来自 peak_full_retrieval.json）
| adduct | cos R@1 | topov R@1 | Δ |
|---|---|---|---|
| [M+H]+ | 0.9003 | 0.8857 | **-1.46pp** |
| [M+Na]+ | 0.9027 | 0.8882 | **-1.45pp** |

raw top_overlap 在官方协议上**输给 cos**，不是提升。

## 真正的峰纠错（连续分数，不受 tie 影响）
`eval_e0_peak_correction.py` 的 LR（cos + n_matched + matched_frac + shared_intensity
+ top_overlap，`predict_proba` 连续分）：
- baseline cos R@1 = 0.9020 → LR R@1 = **0.9224（+2.04pp）**
- 1237 例修正 / 805 例新增错误（净 +432）
- 5/5 fold 全正，n_pairs = 1,697,022（全部对，非 dry-run）
- 这是连续分数，**不是 tie 假象**。

## 修正后的结论边界
1. B 里 `topov_only 0.9482` **作废**——是 pos-first 插入顺序造成的 tie 假象，不代表真实排序力。
2. 负 cos 系数（collinearity）的解释仍成立：cos 与 topov corr 0.618、与 shared_int 0.770，
   raw 峰信号已经更强，cos 被"去偏"出局。但"raw 峰信号更强"只对**连续拟合后的分数**成立，
   对 raw top_overlap 这个离散值不成立（它 tie 太多，排序力弱）。
3. 峰特征的价值是**作为 LR 的连续输入**（+2.04pp），不是**单独当离散排序键**。

## 一句话
"top_overlap 单独能赢 cos" 是测量假象；"cos+峰特征的 LR 能 +2.04pp" 是真的。二者不矛盾——
矛盾来自把离散分数直接当排序键时的稳定排序破平偏向。
