# Noise-v3 C1：支持谱互斥的正证据空间扩展

## 结果

C1覆盖完整23,876个训练图查询，构造80,250个“教师支持谱—排名正例谱”严格互斥的交叉拟合样本，涉及1,217个身份、627个分子式，其中49,667个为near样本。

| 指标 | 结果 |
|---|---:|
| Baseline accuracy | 0.5664 |
| Cross-fit teacher accuracy | 0.5912 |
| Delta | +2.47pp |
| Corrected / introduced | 2,382 / 396 |
| Risk net（corrected - 2×introduced） | 1,590 |
| Near delta | +2.34pp |
| Formula-cluster risk net / example | 0.0198 [0.0168, 0.0232] |
| Teacher-vs-wrong formula-cluster delta | 0.1562 [0.1449, 0.1681] |

所有预注册门通过，可以进入候选感知的峰token学生阶段。

## 与B0的关系

B0教师上限约+9.62pp；严格排除教师谱与排名正例谱的直接重叠后，C1降为+2.47pp。这证明早期上限受到直接正例对齐放大，但同身份正证据梯度并未消失：其在更大化学空间、near子集和公式簇bootstrap中仍稳定为正。

80,250是交叉拟合样本数，不是独立分子数。后续训练必须按query/identity采样并按formula隔离，严禁把同一查询的多个holdout当作独立证据。

## 下一步

冻结官方DreaMS，缓存全部23,876个查询的最终层上下文化峰token。下一代学生由三个部分构成：救援资格门、峰token方向专家、安全干预门。第一阶段只验证峰token是否能显著提高教师方向对齐与净修正；未通过公式隔离门时不得进入封存P3。
