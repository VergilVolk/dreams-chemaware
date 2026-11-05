# CALMS correction landscape and closed-set selective ranking

Date: 2026-11-05

## Scope

This analysis characterizes the already frozen **CALMS v1.0** predictions
(internal Full23 v1.3.2). It does not fit a new model, tune a threshold, route
queries, or change the released score. The two GNPS Gold/Silver panels are
consumed development data:

- identity-disjoint: 10,995 queries, 4,204 formula clusters;
- formula-disjoint: 5,261 queries, 2,210 formula clusters.

The panels overlap: 5,237/5,261 formula-panel query rows also occur in the
identity panel. Their query and correction counts must not be added. Intervals
in the new diagnostic tables use 10,000 formula-cluster bootstrap resamples and
are descriptive unless explicitly stated otherwise. The selective-ranking
analysis is strictly closed-set: the true candidate is present for every query.

## Where CALMS changes the decision

| Comparator | Panel | Comparator R@1 | CALMS R@1 | Delta, pp (descriptive 95% formula-cluster CI) | Corrected / introduced |
|---|---|---:|---:|---:|---:|
| WSE | Identity | 87.367% | 88.367% | **+1.000** [0.637, 1.361] | 223 / 113 |
| WSE | Formula | 88.272% | 89.603% | **+1.331** [0.812, 1.874] | 116 / 46 |
| DreaMS | Identity | 85.357% | 88.367% | **+3.010** [2.461, 3.559] | 533 / 202 |
| DreaMS | Formula | 86.809% | 89.603% | **+2.794** [1.942, 3.628] | 252 / 105 |

Relative Top-1 error reduction versus official fine-tuned DreaMS is 20.6% on
identity and 21.2% on formula. The existing run17 multiplicity-adjusted WSE
intervals also remain positive: [0.464, 1.555] pp and [0.521, 2.154] pp.

The strongest mechanism observation is selective low-margin correction:

- Every CALMS-versus-WSE Top-1 exchange (336 identity and 162 formula) lies in
  WSE's lowest Top1-Top2 gap quartile. Gains within that quartile are +4.001 pp
  [2.570, 5.407] and +5.323 pp [3.315, 7.338]. Q2-Q4 contain no WSE Top-1
  exchanges.
- In the predefined near-candidate subset, CALMS exceeds WSE by +1.317 pp
  [0.810, 1.825] on 7,592 identity queries and +2.208 pp [1.317, 3.112] on
  2,944 formula queries.
- Against DreaMS, 94.1%/95.5% of decision exchanges occur in the lowest gap
  quartile. Near-subset gains are +3.886 pp [3.121, 4.667] and +4.314 pp
  [2.926, 5.749].

These are fixed post-freeze diagnostics, not a deployable router. Margin
quartiles are comparator-specific ranks, not calibrated probabilities. The
near label uses truth/structure metadata and therefore cannot be used at
deployment. Candidate count, peak count, instrument, and near status are
overlapping strata and do not constitute independent experiments.

## Closed-set selective ranking

CALMS confidence is the frozen Top-1 minus Top-2 candidate-score gap. Keeping
only the largest gaps gives the following development operating points:

| Coverage | Identity retained | Identity R@1 (95% CI) | Formula retained | Formula R@1 (95% CI) |
|---:|---:|---:|---:|---:|
| 50% | 5,497 | **99.745%** [99.594, 99.873] | 2,630 | **99.810%** [99.620, 99.962] |
| 80% | 8,796 | **96.703%** [95.999, 97.320] | 4,208 | **97.006%** [95.991, 98.039] |
| 90% | 9,895 | **93.108%** [92.201, 93.980] | 4,734 | **93.874%** [92.642, 94.937] |
| 95% | 10,445 | **90.857%** [89.965, 91.759] | 4,997 | **91.875%** [90.575, 93.045] |
| 100% | 10,995 | **88.367%** [87.458, 89.248] | 5,261 | **89.603%** [88.407, 90.801] |

At matched coverage, the CALMS high-confidence prefix exceeds the WSE prefix
most consistently at high coverage:

| Coverage | Identity CALMS-WSE, pp (95% CI) | Formula CALMS-WSE, pp (95% CI) |
|---:|---:|---:|
| 90% | **+0.677** [0.327, 0.979] | **+0.634** [0.131, 1.124] |
| 95% | **+0.661** [0.349, 1.016] | **+0.800** [0.223, 1.212] |
| 100% | **+1.000** [0.638, 1.366] | **+1.331** [0.806, 1.856] |

CALMS has a lower descriptive area under the risk-coverage curve than WSE:
0.01791 versus 0.01956 on identity and 0.01563 versus 0.01730 on formula
(relative reductions of approximately 8.5% and 9.6%). No bootstrap interval
was computed for the AURC difference.

The CALMS margin does **not** dominate WSE as a confidence statistic: its
correctness AUROC is 0.915/0.912 versus 0.921/0.923 for WSE. The improved
selective accuracy therefore partly reflects CALMS's stronger base ranking,
not universally superior confidence ordering or calibration. Several
low/mid-coverage CALMS-WSE intervals also cross zero.

## Claim boundary

The defensible conclusion is:

> On consumed GNPS closed-set development panels, frozen CALMS improves the
> full-coverage candidate ranking and concentrates its decision changes in
> low-margin, near-competitor cases. Its score gap supports high-accuracy
> selective ranking, with the most consistent advantage over WSE at 90-100%
> coverage.

This is not unopened external confirmation, no-match/open-set evaluation,
false-discovery-rate control, calibrated match probability, or a deployable
threshold. GNPS was used during CALMS development and 0.9 residual-scale
selection. No threshold from this analysis may be promoted without independent
calibration.

## Reproducible artifacts

- Correction analysis:
  `data/validation/calms_correction_landscape_20261105`
- Selective-ranking analysis:
  `data/validation/calms_v132_closed_set_selective_ranking_20261105`
- Scripts:
  `tasks/analyze_calms_correction_landscape.py`
  and `tasks/evaluate_calms_closed_set_selective_ranking.py`
