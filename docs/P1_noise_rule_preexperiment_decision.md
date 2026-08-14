# P1 rule and peak-masking pre-experiment: decision

**Date:** 2026-08-11  
**Status:** rule-side intervention and two validation cohorts complete; DreaMS
re-embedding under masking awaits a CUDA environment.

## Questions

1. Do the P0 failure directions correspond to reproducible chemical-evidence
   conflicts?
2. Are the 3,486 rules stable when part of the spectrum is unavailable?
3. Can a compact rule subset directly correct DreaMS errors?
4. Is DreaMS-style peak masking safe for retrieval fine-tuning?

## Intervention

The primary perturbation reproduces the raw DreaMS checkpoint metadata:
intensity-proportional fragment selection, precursor protected, m/z replaced by
-1, with 10%, 20% and 30% mask rates. Physical token deletion is a matched
control. Identity and strict 10-ppm candidates remain unchanged.

The rule-side pilot used 36 P0 error queries and 24 MCES-stratified correct
controls, three random masks per rate: 1,080 interventions over 3,486 rules.

## Rule stability and direction

At 30% masking:

| Group | Rule retention | True minus wrong Rule-Jaccard |
|---|---:|---:|
| Correct controls | 0.747 | +0.285 |
| MCES 0-2 errors | 0.739 | -0.096 |
| Cross-instrument errors | 0.697 | -0.105 |
| High-rule conflicts | 0.783 | -0.213 |

The rules survive missing peaks reasonably well, but selected DreaMS errors
retain evidence that is ambiguous or wrong-directed.

## Compact-panel screen

Predefined exploratory gates required at least three independent error
molecules, masked true-minus-wrong frequency >=0.08 and mask stability >=0.60.

| Assignment | Rules |
|---|---:|
| Corrective candidate | 0 |
| Conflict-mining only | 165 |
| Generic high coverage | 49 |
| Noise fragile | 48 |
| Insufficient or nonspecific | 3,224 |

The largest positive error-set margin was only 0.052 and covered at most two
error molecules. The absence of corrective candidates is therefore not caused
only by the 0.08 threshold.

## Validation outside the screening molecules

All 54 molecules represented in the 60-query screen were excluded.

### Different-molecule validation

One spectrum from each of 400 error molecules and 400 never-error control
molecules:

| Panel | Error-detection ROC-AUC (95% CI) |
|---|---:|
| 165-rule conflict panel | 0.608 (0.571-0.646) |
| 335 core rules | 0.612 (0.573-0.652) |
| 3,486 total rules | **0.647 (0.608-0.686)** |

### Within-molecule paired validation

For 300 molecules, one Top-1 error spectrum was paired with one correctly
retrieved spectrum from the same molecule:

| Panel | Error-query ROC-AUC (95% CI) |
|---|---:|
| 165-rule conflict panel | 0.583 (0.537-0.628) |
| 335 core rules | 0.580 (0.534-0.624) |
| 3,486 total rules | **0.599 (0.557-0.644)** |

The signal therefore is not solely a molecular-family artifact. Within the same
molecule, spectra with weaker true-directed rule evidence are more likely to be
DreaMS failures. The effect is real but moderate.

## Decision

1. The P0 pain point is supported: DreaMS errors are associated with loss of a
   clear true-candidate rule advantage, especially in local analogues,
   high-rule-overlap conflicts and cross-instrument comparisons.
2. Rules are useful as a conflict/uncertainty signal, not as an identity label.
3. The 165-rule compact panel does not outperform all 3,486 rules on held-out
   molecules. It must not be presented as an optimized panel.
4. No rule has yet earned direct corrective supervision. Do not add a rule
   distance loss or a rule-decode loss from this screen.
5. Rule-side robustness does not prove DreaMS-side augmentation safety. The next
   experiment is the small CUDA re-embedding test using native masking only.

## Next executable gate

```powershell
& 'D:\dreams_env\python.exe' tasks\pilot_rule_noise_stress.py `
  --device cuda --n-per-stratum 6 --n-controls 12 `
  --mask-rates 0.10 0.20 0.30 --modes native_mask --n-seeds 3
```

Pass conditions for correct controls:

- Top-1 loss <=10 percentage points in this pilot;
- clean-to-masked embedding cosine >=0.90;
- deterioration or reproducible recovery in P0 strata differs from controls.

Only after this gate passes should physical peak dropout or consistency
fine-tuning be attempted.
