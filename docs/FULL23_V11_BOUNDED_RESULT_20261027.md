# Full23 v1.1 bounded multi-reference result

Date: 2026-10-27. Run: `full23_multireference_readout_run_2420264`.
Implementation lineage: `fd1575d -> d05f555 -> 5d17fc1`.

## Decision

The bounded implementation ran as specified, but neither representation arm
qualified for promotion. The released closed-set ranker remains **Unified
Additive Full23 v1**, checkpoint
`7f1309e2a89ef3640d9b6871a821df6866f2528e629547075d5305ab6707d717`.

The exact development adjudication is:

- `selected = full23_v1`;
- `promotion_from_full23_v1 = false`;
- `multi_reference_content_supported = false`.

The job-completion label records successful execution of the experiment; it
does not denote a v1.1 release.

## Contract actually executed

Both independently trained, capacity-matched arms used the same fixed bound:

`final_score = Full23_v1 + B * tanh(raw_residual / B)`

with `B = 0.6255066394805908` and maximum pairwise movement
`2B = 1.2510132789611816`. The bound was calculated from Enveda-180 five-fold
formula-OOF Full23 parent scores only; the calibration report records
`gnps_accessed = false`. Top1 and Top3 have separate fitted weights and the
same 71,809-parameter budget. The candidate-reference permutation control was
also evaluated.

Enveda-180 was consumed meta-development fitting data: 25,000 identity-panel
queries and 9,435 formula-panel queries. No Enveda performance claim is made.
GNPS Gold/Silver was consumed development adjudication: 10,995 identity-panel
queries and 5,261 formula-panel queries. These results are not untouched
external confirmation.

## GNPS result

| Panel / method | Recall@1 | Delta vs Full23 | MRR | Corrected / introduced | Formula-cluster CI, pp | Near Recall@1 |
|---|---:|---:|---:|---:|---:|---:|
| identity Full23 v1 | 88.2492% | -- | 92.3792% | -- | -- | 83.6934% |
| identity bounded Top1 | 88.1128% | -0.1364 pp | 92.2962% | 31 / 46 | [-0.2892, +0.0092] | 83.5090% |
| identity bounded Top3 | 88.0673% | -0.1819 pp | 92.2687% | 39 / 59 | [-0.3644, -0.0090] | 83.4299% |
| formula Full23 v1 | 89.5267% | -- | 93.6815% | -- | -- | 82.2011% |
| formula bounded Top1 | 89.3366% | -0.1901 pp | 93.5815% | 14 / 24 | [-0.4176, +0.0194] | 81.8954% |
| formula bounded Top3 | 89.2416% | -0.2851 pp | 93.5393% | 15 / 30 | [-0.5582, -0.0191] | 81.6916% |

Macro and micro AUROC, micro AUPRC and MRR did not rescue either arm. For the
predeclared Top3 arm, Full23 v1 was better on every reported ranking and near
metric in both panels.

Top3 did not beat the parameter-matched Top1 control: `-0.0455 pp` in identity
and `-0.0950 pp` in formula, with both cluster intervals crossing zero. Top3
also did not beat candidate-reference permutation: `-0.0182 pp` and
`+0.0570 pp`, again with both intervals crossing zero. The experiment
therefore provides no support for a multi-reference-content mechanism.

Reference-count stratification reinforces that result. Top3 minus Top1 was
positive for true-candidate K=1, but negative for K=2 and K=3 in both panels;
the K=3 loss was the largest (`-0.6957 pp` identity, `-0.7628 pp` formula).

## What the bound repaired, and what it did not

Relative to the unbounded `run_2419582`, the bound substantially reduced
ranking exchanges and lambda-2 risk. For Top3, identity changed from 102/128
corrected/introduced to 39/59, and formula from 43/56 to 15/30. This confirms
that the corrected implementation is materially bounded, not a relabelled
rerun.

It did not create a transferable representation increment. Candidate-level
residuals remained strongly clipped, predominantly on negative candidates:
Top3 had `|residual| >= 0.99B` for 52.31% of identity candidates and 42.31% of
formula candidates. The head was therefore using the bound heavily while
still introducing more Full23 errors than it corrected.

## Scientific boundary and next action

This result rejects only the tested **post-Full23 full-vector/multi-reference
residual**. It does not remove Noise V1, Stage-1, Phase-A or FullMSG from the
released system: their candidate-level evidence is already present in Full23
v1, and prior strict-refit analyses address the self-trained evidence families.

No further GNPS-conditioned search over `B`, pooling, reference count or a new
readout head is authorized. The productive main line is now:

1. deploy Full23 v1 on complete, model-independent application candidate
   blocks;
2. add task-compatible released competitors with explicit training-overlap
   accounting;
3. evaluate no-match/coverage-risk and runtime separately from input-missing
   abstention;
4. reserve any final performance claim for a newly frozen, unconsumed external
   resource.

Primary local artifacts:

- `data/validation/full23_multireference_readout_run_2420264/modeling/report.json`
  (`e7cf2613021eb780fa29127e19a435cc231c2717e7f82c8f5bbd311a6daadbf0`)
- `data/validation/full23_multireference_readout_run_2420264/gnps_development_adjudication/report.json`
  (`adbcbb5fcbdc78a2980c96e894173e6c5afdbb65a6b21689871ba7b9cd9da457`)
