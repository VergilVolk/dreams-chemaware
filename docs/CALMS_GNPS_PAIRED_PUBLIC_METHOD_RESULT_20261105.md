# CALMS paired comparison with public MS/MS methods

Date: 2026-11-05

## Scope

This is a re-expression of the existing 18-method common-axis GNPS result with
frozen CALMS as the paired reference.  It does not refit a score, select a
model or open a new dataset.  GNPS is **consumed development evidence**, so the
result supports a method comparison but not unopened external confirmation or
a field-wide SOTA claim.

The strict common domain contains 10,983 identity-disjoint queries and 5,253
formula-disjoint queries.  Every method sees the same query and candidate
blocks.  Paired intervals are formula-cluster bootstrap intervals with the
existing familywise correction.

## Frozen CALMS metrics

| Panel | R@1 | R@3 | R@5 | MRR | Micro AUROC | Micro AP |
|---|---:|---:|---:|---:|---:|---:|
| Identity-disjoint (n=10,983) | 88.400% | 95.939% | 97.505% | 0.9247 | 0.9587 | 0.8445 |
| Formula-disjoint (n=5,253) | 89.644% | 97.658% | 98.972% | 0.9376 | 0.9539 | 0.8744 |

## Paired Recall@1 advantage

Positive values favor CALMS.  Corrected/introduced are oriented from each
listed comparator to CALMS.

| Comparator | Identity delta, pp (familywise CI) | Corrected / introduced | Formula delta, pp (familywise CI) | Corrected / introduced |
|---|---:|---:|---:|---:|
| DreaMS | **+3.014** [2.121, 3.870] | 532 / 201 | **+2.779** [1.512, 4.083] | 251 / 105 |
| WSE | **+0.983** [0.432, 1.541] | 221 / 113 | **+1.333** [0.596, 2.165] | 116 / 46 |
| Noise V1 | **+1.812** [1.094, 2.603] | 388 / 189 | **+1.542** [0.489, 2.657] | 181 / 100 |
| MS2DeepScore 2.x | **+9.242** [8.029, 10.548] | 1,233 / 218 | **+8.110** [6.370, 9.908] | 533 / 107 |
| Spec2Vec (public) | **+2.085** [1.338, 2.893] | 421 / 192 | **+2.170** [1.051, 3.333] | 208 / 94 |
| MSBERT | **+6.501** [5.455, 7.619] | 912 / 198 | **+6.073** [4.530, 7.724] | 424 / 105 |
| SpecEmbedding | **+13.138** [11.763, 14.557] | 1,636 / 193 | **+10.356** [8.459, 12.337] | 648 / 104 |

The proposal-safe statement is therefore narrower than a field-wide SOTA claim:

> On the consumed GNPS common-domain development panels, frozen CALMS v1.0
> (internal Full23 v1.3.2) has
> higher Recall@1 than the evaluated public checkpoints and spectral-search
> baselines, including official fine-tuned DreaMS and the auxiliary
> MS2DeepScore 2.x structural-similarity model; the within-comparator paired
> cluster intervals for the comparisons above remain positive.

These methods have different native objectives and training sources.
MS2DeepScore 2.x is primarily a structural-similarity model, so its exact
identity result is a useful same-axis comparator rather than a judgment of its
native task.

The displayed intervals are familywise-adjusted over 20 endpoints within each
single CALMS-versus-comparator analysis. They are not simultaneous across the
17 comparators or the two overlapping panels, and they do not include the
uncertainty introduced by prior CALMS development and residual-scale selection
on GNPS. The common availability domain retains 10,983/10,995 identity queries
and 5,253/5,261 formula queries; the complete 17-comparator table is in the TSV.

## Reproducible artifacts

- Original CALMS-referenced reports:
  `data/validation/gnps_run18_full23_reference_20261102`
- Machine-readable summary and forest plot:
  `data/validation/calms_gnps_paired_superiority_v3_20261105`
- Summary script:
  `tasks/summarize_calms_paired_superiority.py`
