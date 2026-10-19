# Contrast Atlas first structure-hidden cross-study response result

Date: 2026-10-14  
Evidence role: development / consumed  
Leakage status: the two public-study relations were used during method development; neither is unopened external validation.

## Scientific question

Can a study-level response object use signed effect size and uncertainty on MS2-linked features to recover a prespecified biological response family across independent studies, without requiring exact structures for all contributing entities?

This is a Contrast-to-Contrast test. It is not ordinary spectrum retrieval: all candidate contrasts within a study share the same chemical measurement universe, so spectrum composition alone cannot distinguish them. Candidate ranking must use study-internal effect direction, magnitude and uncertainty.

## Frozen response objects

### Query: MSV000097426 mouse urine HFD minus NFD

- 12 biological samples: 6 HFD and 6 NFD; 3 pooled QC and 2 blanks.
- 4,635 recurrent MS2 anchors; 2,163 passed label-blind measurement QC.
- Effect: Welch HFD minus NFD on PQN-normalized log2 MS1 peak height.

### Candidate panel 1: MSV000088023 mouse plasma

- Candidate universe: female and male HFD-versus-normal responses plus two sex contrasts.
- Prespecified target family: the two HFD-versus-normal contrasts.
- Candidate universe and target family were fixed before scoring.

### Candidate panel 2: MSV000084112 human urine

- 48 independent subjects: NH, OH, non-obese diabetes and obese diabetes, 12 per group; each group contains 6 women and 6 men.
- 12,116 recurrent MS2 anchors; 5,501 passed label-blind measurement QC.
- Four HC3 contrasts: two obesity effects and two diabetes effects, adjusted for sex and centered age.
- Prespecified target family: obesity; distractor family: diabetes.

## Result 1: exact HFD response retrieval abstained for lack of chemical coverage

MSV000097426 to MSV000088023 used a fixed relation graph of 10 ppm precursor tolerance, 0.02 Da fragment tolerance, at least six matched fragments, WSE at least 0.70 and reciprocal-best matching.

- 13 candidate relations were found, but only 9 were reciprocal.
- The frozen minimum was 12 reciprocal relations.
- All-entity world: abstain, 9/12 required relations.
- Known same-library-identity world: abstain, 0/12.
- Structure-hidden-at-both-ends world: abstain, 8/12.

No rank is scientifically admissible. This is a chemical-coverage failure, not evidence that the HFD responses are dissimilar.

## Result 2: HFD to human obesity gave the correct descriptive rank but failed calibration

The second test used the same fixed spectral relation contract. Both masses in the previously reported m/z 396.1684/220.1366 relation were excluded before primary scoring.

- 629 candidate relations; 223 reciprocal before the reported-pair exclusion; 221 primary relations.
- All-entity world: unique Top-1 was `obesity_without_diabetes`; target-minus-best-distractor margin = +0.125760.
- Structure-hidden-at-both-ends: 207 relations; the same target ranked first; margin = +0.135112.
- Known same-library identity: only 10 relations, below the frozen minimum of 12, so it abstained.
- Unsigned-effect margins were only +0.005957 and +0.007786. The observed ranking therefore came from response direction, not from chemical composition alone.
- The target remained Top-1 under all four RT-sensitivity cutoffs and all ten leave-one-mass-decile-out checks; the minimum leave-decile-out margin was +0.079145.

However, robustness to feature subsets is not calibration. Across all 924 exact 6-of-12 mouse diet-label assignments:

- all-entity observed rank = 296/924; one-sided exact p = 0.320346;
- structure-hidden observed rank = 280/924; one-sided exact p = 0.303030;
- the mass-stratified response-link shuffle gave p = 0.269546.

Formal verdict: `FAIL_NO_CALIBRATED_CROSS_STUDY_RESPONSE_RETRIEVAL`.

The result is a real, direction-dependent descriptive signal, but it is not exceptional under the legal sample-label null. It cannot be called a confirmed retrieval or biological discovery.

## What this experiment establishes

1. The implemented query is a biological-response query: signed study-internal effects and their uncertainty change the rank, whereas spectrum-only composition cannot rank the four shared-universe candidates.
2. Structure-hidden relations can carry the descriptive signal: 207/221 primary relations were unresolved at both ends, while the strict known-identity world lacked coverage.
3. Correct Top-1 is insufficient. Exact sample-label calibration can overturn an apparently stable and direction-dependent result.
4. The current relation graph should not be loosened post hoc to rescue either experiment. A new independent source with larger participant-level denominators is the appropriate next test.

## Result 3: independent human obesity-to-obesity query failed in the wrong direction

The pre-frozen MSV000083077 experiment has now been executed. The current public
Qiita/MassIVE files yielded 62 independent participants after taking one 10:00
sample per person, plus eight blanks. The four groups were:

- non-obese / low depressive symptoms: 22;
- obese / low depressive symptoms: 8;
- non-obese / high depressive symptoms: 17;
- obese / high depressive symptoms: 15.

Across these 70 files, 156,800 eligible MS2 observations produced 15,184
recurrent anchors. Label-blind measurement QC retained 7,132 anchors using
detection in at least 31/62 biological samples and mean biological abundance at
least five times the blanks. Of the original anchors, 309 had no positive MS1
measurement in the frozen quantification window and were explicitly rejected;
they were not assigned an artificial effect.

The saliva HC3 model used obesity, depressive-symptom status, their interaction,
sex, z-scored age, acquisition plate and race. One deposited race code contained
only one participant and would have given that sample leverage exactly one,
making HC3 undefined. Before opening any chemical effect, race codes with fewer
than five participants were therefore pooled into the modal reference level.
The resulting 62 x 11 design had rank 11, condition number 8.4562 and maximum
leverage 0.3370. Raw race counts were 35/15/11/1 and model counts were 36/15/11.

The query was the equal-weight mean of the two MSV000084112 urine obesity simple
effects, with and without diabetes. Candidate targets were the two saliva obesity
simple effects; the two saliva depressive-symptom simple effects were frozen
distractors. Results were:

| chemical world | edges | Top-1 | target minus distractor | response-link p |
|---|---:|---|---:|---:|
| WSE all entities | 152 | `depression_obese` | -0.208911 | 0.998800 |
| unresolved at both ends | 135 | `depression_obese` | -0.208447 | not separately tested |
| same-library identity | 12 | `depression_obese` | -0.199284 | not separately tested |
| reciprocal mass-only, no MS/MS | 795 | `depression_obese` | -0.037682 | 0.957009 |

Both urine component views also failed: obesity without diabetes ranked
`depression_nonobese` first, whereas obesity with diabetes ranked
`depression_obese` first. In the primary WSE world the four aggregate signed
scores were `+0.060371` (depression-obese), `-0.102153`
(depression-nonobese), `-0.148540` (obesity-high-depression) and `-0.198495`
(obesity-low-depression).

Formal verdict: `FAIL_WRONG_PRESPECIFIED_TARGET_TOP1`.

The target-family Top-1 gate and the response-link null both failed decisively.
The 24,000 legal saliva sample-label assignments and 2,000 urine Monte Carlo
assignments had been checked in advance for full-rank, finite-HC3 designs, but
were not scored after this decisive failure because no sample-label p-value
could reverse the mandatory wrong-Top-1 gate. This fail-fast action is recorded
in the audit report rather than being presented as a completed calibration.

This experiment does not support transport of a generic human obesity response
from urine to saliva under the current response representation. WSE supplied a
much smaller and chemically stricter graph than mass alone, but did not rescue
the biological target; unresolved entities and strict known identities agreed
on the wrong family. The route is closed for post-hoc threshold, subgroup or
contrast retuning on these two studies.

## Experiment that was frozen before raw scoring

MSV000083077 will provide an independent human saliva panel rather than another evaluation component:

- one 10:00 sample per participant to remove repeated-measure pseudoreplication;
- objective BMI >= 30 and BDI total >= 3 definitions;
- two obesity contrasts as the target family and two depressive-symptom contrasts as distractors;
- HC3 effects adjusted for sex, z-scored age, acquisition plate and race; these nuisance terms were frozen before any saliva effect or cross-study score was computed after an outcome-blind metadata audit found plate and demographic imbalance;
- the public deposited cohort is analyzed as a newly frozen cohort because its current Qiita/MassIVE denominators do not match the paper's reported 60 participants and 261 samples.

That experiment is Result 3 above and failed. The next valid advance is not
another obesity threshold search. It must use a new design with a genuinely
stronger pre-existing positive relation, preferably repeated intervention or
genetic perturbation across independent sources, while retaining the same
identity/relation distinction, mass-only control and fail-closed calibration.

## Immutable boundaries

- MSV000097426, MSV000088023, MSV000084112 and MSV000083077 are development / consumed assets.
- Chemical relation edges are not entity identity unless supported by the strict same-library-identity partition.
- A candidate-study Top-1 does not imply a shared molecular mechanism.
- A structure-hidden gain is reportable only if it increases correct answered coverage at fixed error risk and passes sample-level calibration.
- These results do not authorize an Atlas-scale biological claim or any final external-validation claim.
