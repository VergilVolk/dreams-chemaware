# E4-A 高学习率直接噪声微调：5-fold x 3-seed 结果

日期：2026-08-27  
状态：开发阶段确认通过；P3 未使用。

## 固定训练配置

- 一个共享 DreaMS query/reference encoder；推理只输入原始 clean spectrum；
- 解冻最后 1/7 Transformer block 和官方 projection head；
- backbone LR 2e-6，head LR 1e-5，gradient clip 1.0，4 epochs；
- action scope=all，identity-equal，4 views/identity；
- candidate-gradient 50% step 3-6；role-confounder 100% step 1-5；
- 不使用 teacher、P2b、下游重排分数或 held-fold action outcome。

## 完整结果

| seed | 五fold查询 | Recall@1增益 | near增益 | corrected / introduced | corrected - 2xintroduced | 最低fold保持度 |
|---:|---:|---:|---:|---:|---:|---:|
| 20260828 | 23,876 | +0.611 pp | +0.508 pp | 183 / 37 | 109 | 0.995156 |
| 20260829 | 23,876 | +0.662 pp | +0.537 pp | 191 / 33 | 125 | 0.995140 |
| 20260830 | 23,876 | +0.632 pp | +0.522 pp | 185 / 34 | 117 | 0.995245 |

三seed均值约为：

- Recall@1：+0.635 pp；
- near Recall@1：+0.522 pp；
- 每seed平均修正约186个、引入约35个；
- 风险加权净收益 `corrected - 2*introduced` 每seed约117；
- 15/15 fold 的 formula-cluster CI 下界均大于0；
- 15/15 run 均通过当前E4-A安全门。

## fold异质性

- overall最弱fold仍为正：约+0.425 pp；最强约+0.874 pp；
- near最弱fold约+0.119 pp，但没有fold出现near退化；
- fold 3/4 的near收益明显较小，说明固定N-arm对不同化学空间覆盖不均；
- 每个fold平均保持度均超过0.995，但p01约0.980-0.982，提示少量谱图仍存在较大漂移，后续必须继续报告尾部保持与新增错误类型。

## 结论

该结果确认：固定、预注册的峰级噪声动作可以通过直接微调迁移进共享 DreaMS embedding，
且改进跨 formula fold 与训练seed稳定。它不再是单fold偶然现象，也不是下游重排器收益。

但+0.635 pp仍明显低于3.35-3.85 pp逐查询事后动作oracle。固定N-arm主要处理错误候选过近，
不能补回占主体的同分子正例不足。因此下一阶段不是继续提高删峰强度，而是加入真实跨条件P-arm，
并保留当前N-arm和clean safety作为三流联合训练。

## 下一阶段硬门

P/N联合模型必须相对当前N-arm模型同时满足：

1. overall Recall@1进一步提高，且formula-cluster CI下界大于0；
2. positive-deficit和cross-condition子集明确改善；
3. near不退化；
4. corrected > introduced，且corrected - 2*introduced > 0；
5. 平均保持度 >=0.995，并单独报告p01；
6. 完成开发fold后再一次性进入封存P3。

## 声明边界

这是开发图上的多fold、多seed证据，不是封存P3或外部谱库最终结果。目前可声明
“共享embedding在固定噪声增强下稳定改善”，不能声明全面超过DreaMS或达到3.85 pp。
