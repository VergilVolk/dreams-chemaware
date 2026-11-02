# Full23 v1.3 cumulative-repair result (2026-11-02)

## Decision

The returned six-arm run is complete and internally consistent. The selected
arm is `v13_01`: formal Full23 v1.3 retrained with both of the following
changes:

1. the historical anchor-correct training multiplier is reduced from 2 to 1;
2. the regularizer is applied to the query-centred **raw** residual rather than
   to the already bounded residual.

The selected checkpoint is
`data/validation/full23_v13_cumulative_repairs_run_2434084/modeling/v13_01.pt`
(SHA-256
`6603013a04fdc26946445a6ba1ed566d768bac49423046ce3922daac79fe637e`).
It is byte-identical to the run's `selected_model.pt`.

This is a consumed-development selection, not an untouched external
confirmation. Enveda-180 supplied the fixed training folds; GNPS selected the
arm after all six arms had been fit.

## Data and denominators

| Resource | Role in this run | Denominator |
|---|---|---:|
| Enveda-180 identity-disjoint | consumed meta-development; five-fold formula-OOF fitting | 25,000 queries |
| Enveda-180 formula-disjoint | consumed meta-development; five-fold formula-OOF fitting | 9,435 queries |
| GNPS identity-disjoint | consumed development and arm selection | 10,995 queries |
| GNPS formula-disjoint | consumed development and arm selection | 5,261 queries |
| GNPS query-balanced joint endpoint | consumed development and arm selection | 11,019 unique queries |

The two GNPS panels overlap. Their query counts and corrected/introduced counts
must not be naively added. The joint endpoint gives each unique query total
weight one and averages its outcomes when it occurs in both panels.

## Fixed six-arm result

| Model | Identity R@1 | Formula R@1 | Joint unique-query Top-1 | Delta from formal v1.3 |
|---|---:|---:|---:|---:|
| formal v1.3 | 88.2947% | 89.6408% | 88.3429% | - |
| `0` | 88.2310% | 89.5837% | 88.2975% | -0.0454 pp |
| **`01`** | **88.3311%** | 89.5647% | **88.3928%** | **+0.0499 pp** |
| `012` | 88.0491% | 89.2986% | 88.1205% | -0.2223 pp |
| `0123` | 87.9309% | 89.2796% | 87.9889% | -0.3539 pp |
| `01234` | 88.0582% | 89.1846% | 88.0978% | -0.2450 pp |
| `012345` | 88.1492% | 89.5457% | 88.2113% | -0.1316 pp |

For `v13_01` versus formal v1.3:

- identity-disjoint: 55 corrected / 51 introduced, net +4 correct queries,
  `+0.0364 pp` R@1;
- formula-disjoint: 28 corrected / 32 introduced, net -4 correct queries,
  `-0.0760 pp` R@1;
- query-balanced joint endpoint: effective 56.5 corrected / 51.0 introduced,
  net +5.5 weighted correct queries, `+0.0499 pp` Top-1;
- query-balanced joint MRR: 92.4555% to 92.4911%, `+0.0356 pp`.

The per-panel paired intervals cross zero. The result is therefore a small
development improvement under the already fixed point-estimate selection rule,
not evidence of a stable external effect. It must not be described as improving
both panels.

## What was actually fixed

The raw-space penalty repaired a concrete saturation failure:

| Panel | Formal v1.3 candidates at at least 99% of bound | `v13_01` |
|---|---:|---:|
| identity-disjoint | 57.78% | 9.11% |
| formula-disjoint | 66.32% | 12.23% |

The fraction of queries using at least 90% of the maximum residual span also
fell from 95.09% to 70.96% (identity) and from 92.15% to 62.25% (formula).
This is the useful mechanistic result: the old objective let the readout live on
the tanh boundary, while penalizing the query-centred raw residual largely
restored a graded correction.

Arm `0` alone was worse than formal v1.3. Therefore the result does **not** show
that removing the multiplier by itself is beneficial; the two changes in `01`
are retained as one empirically selected recipe.

## What must not be carried forward

The widened residual bound (`012`) is the main damaging intervention. Relative
to `01`, it loses 0.2820 pp on identity, 0.2661 pp on formula and 0.2723 pp on
the joint endpoint. Adding anchor context, a wider pair bottleneck and a signed
query-minus-reference channel after that damage only recovers part of the loss.
The lower Enveda training losses of the larger arms do not imply better
transfer.

The next experiment must therefore use `v13_01` as its parent and test the
remaining representation changes directly, one at a time, while keeping the
original bound. It must not continue the harmful `012 -> 0123 -> ...` chain.

## Claim boundary

- Enveda-180: consumed meta-development.
- GNPS Gold/Silver: consumed development and model selection.
- MassSpecGym: previously consumed and not used as a new confirmation set.
- No untouched external confirmation was produced in this run.
- Historical `corrected - 2*introduced` and confidence intervals remain
  descriptive; neither is used to override the fixed query-balanced Top-1
  selection endpoint.
