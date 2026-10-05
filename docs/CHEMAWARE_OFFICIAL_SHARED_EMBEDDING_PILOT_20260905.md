# ChemAware official shared-embedding pilot — 2026-09-05

## Decision

The official DreaMS embedding has measurable held-formula headroom under the
complete molecule-candidate retrieval objective.  The admitted cheap screen is
the spectrum-only, train-error-conditioned, identity-equal shared residual
adapter.  It is an admission test for direct encoder fine-tuning, not evidence
that a chemical teacher has contributed.

## Frozen data and evaluation boundary

- manifest: `data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz`
- manifest SHA256: `504ce0a570ac1bc461e76cad81acb3ca3560e58b008ec2d8ff14036e944599f5`
- official checkpoint SHA256: `8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245`
- 83,619 queries; 392,229 query-candidate molecule nodes; 6,220,661
  query-reference spectrum edges; 87,848 unique reachable spectra
- training pool after formula exclusion: 47,352 queries
- inner selection/evaluation: 1,929 identities in 1,210 formula clusters
- outer: 16,198 queries untouched
- query and references always use the same transform; candidates and molecular
  structures are absent from deployment inference

## Full-cache results

| Arm | Recall@1 | Delta | Corrected / introduced | Formula-cluster 95% CI | Preservation | Gate |
|---|---:|---:|---:|---:|---:|---|
| frozen official | 90.461% | — | — | — | 1.000000 | reference |
| uniform spectrum-only | 90.721% | +0.259 pp | 7 / 2 | [-0.092, +0.517] pp | 0.999386 | FAIL |
| error-conditioned 0.50, identity mass | **90.824%** | **+0.363 pp** | **9 / 2** | **[+0.096, +0.785] pp** | 0.998208 | **PASS** |
| error-conditioned 0.50, formula-identity mass | 90.721% | +0.259 pp | 7 / 2 | [+0.062, +0.689] pp | 0.998387 | PASS, weaker |

The selected main arm used hidden dimension 256, batch size 64, 2 sampled real
reference spectra per candidate molecule, 2,048 identities per epoch, learning
rate `3e-5`, preservation weight 20, margin-floor weight 2, and gradient norm
limit 5.  The selected checkpoint was step 128 of the 500-step budget.

## Interpretation boundary

1. The strict-positive result establishes that training-time conditional
   routing plus the complete candidate objective can improve the official
   shared spectrum space.
2. Uniform continuation did not pass its formula-cluster interval, so the main
   result is not explained by merely adding optimizer steps.
3. The gain is **0.363 percentage points**, not multiple percentage points.
4. No chemical semantic contribution has yet been established.  A chemical
   claim requires a correct teacher arm to beat spectrum-only and a
   candidate-swapped/permuted dose-matched arm on the same held identities.
5. Formula-equal training mass, although useful in the latest Noise protocol,
   is weaker here and is not admitted to the first direct run.

## Direct official-DreaMS stage

`tasks/train_chemaware_full_candidate_direct.py` now performs the capacity
upgrade from the frozen official checkpoint:

- unfreezes the projection head and final Transformer block only;
- freezes the complete four-epoch query/candidate schedule before training;
- materializes only scheduled, replay, and inner-evaluation spectra;
- uses complete candidate-molecule boundaries with sampled real replicate
  spectra;
- applies the same encoder to query and reference spectra;
- uses initialization margin floors and embedding preservation;
- evaluates 1,929 held identities against every real reference spectrum of
  every candidate molecule;
- saves a clean-spectrum-only deployable shared encoder.

The end-to-end CPU smoke reconstructed the 117,101,029-parameter official
model, passed official replay, updated the final block/head, evaluated complete
candidates, and saved a 468 MB checkpoint.  It is labelled `SMOKE_ONLY` and is
not a performance result.

The server dependency chain is:

1. `tasks/run_chemaware_full_candidate_official_cache.sbatch`
2. `tasks/run_chemaware_full_candidate_official_pilot.sbatch`
3. `tasks/run_chemaware_full_candidate_direct_official.sbatch`

Submit through `tasks/submit_chemaware_full_candidate_official_pilot.sh`.
The direct job is `afterok`-gated by the passed adapter result.  As of this
freeze, both known SSH paths time out before authentication, so no Slurm job ID
has been fabricated or reported.

## ICEBERG candidate-distribution attribution gate

The full-manifest ICEBERG teacher had large diagnostic headroom, but its
candidate-relative distribution did not transfer into a better held-formula
shared spectrum embedding.  Four arms used the same frozen query/reference
schedule, optimizer budget, and fixed step-400 endpoint.  Candidate molecules
were used only to define the training loss; deployment inference remained a
single clean spectrum to a shared embedding.

| Arm | Recall@1 | Delta | Corrected / introduced | Preservation |
|---|---:|---:|---:|---:|
| spectrum-only | 90.617% | +0.156 pp | 7 / 4 | 0.995548 |
| correct ICEBERG distribution | 90.617% | +0.156 pp | 7 / 4 | 0.993097 |
| candidate-swapped distribution | 90.617% | +0.156 pp | 7 / 4 | 0.992475 |
| peak-permuted distribution | 90.721% | +0.259 pp | 9 / 4 | 0.992863 |

Paired formula-cluster comparisons for the correct arm were:

- versus spectrum-only: +0.000 pp Recall@1; 95% CI
  [-0.083, +0.124] pp;
- versus candidate-swapped: +0.000 pp; 95% CI
  [-0.124, +0.165] pp;
- versus peak-permuted: -0.104 pp; 95% CI
  [-0.331, +0.083] pp.

Decision: `DEVELOPMENT_FAIL`.  Do not move this KL-distribution injection into
the official DreaMS Transformer.  The result does not refute chemical priors in
general; it specifically rejects this candidate-distribution target and dose.
The admitted server experiment remains the passed error-conditioned,
spectrum-only complete-candidate route described above.

## ICEBERG sampling-router attribution gate

A second development gate tested whether ICEBERG should choose which official
DreaMS errors receive repeated real-spectrum hard-label training.  Each arm
selected exactly 320 of the same 619 training-formula official errors and used
the same 128-step optimizer budget.  The correct arm selected 320 teacher-Hit@1
errors; the candidate-swapped and peak-permuted controls selected 85 and 178
teacher-Hit@1 errors respectively.  No teacher score entered the loss.

| Router | Recall@1 | Delta | Corrected / introduced | Preservation |
|---|---:|---:|---:|---:|
| seeded random error subset | 90.617% | +0.156 pp | 6 / 3 | 0.996749 |
| correct ICEBERG ranking | 90.461% | +0.000 pp | 7 / 7 | 0.996259 |
| candidate-swapped ranking | 90.617% | +0.156 pp | 8 / 5 | 0.996742 |
| peak-permuted ranking | 90.565% | +0.104 pp | 7 / 5 | 0.996449 |

The correct router was -0.156 pp versus the random router and -0.156 pp versus
the candidate-swapped router.  Its paired formula-cluster intervals crossed
zero, and it failed all five attribution gates.  Decision:
`DEVELOPMENT_FAIL`.  Teacher-solvable errors are not automatically the errors
whose repeated training best transfers to unseen formulas.  Do not submit this
router to the official-encoder GPU stage.

## Clean-visible mass-kernel shared embedding gate

The failed structure-teacher routes motivated a stricter question: can a
chemical prior that is fully observable from each clean spectrum improve the
shared geometry without a candidate-conditioned inference function?  A fixed
CountSketch embedding was built independently for every spectrum from its 32
highest-intensity fragment masses and precursor-minus-fragment neutral losses.
Concatenating it with official DreaMS is a genuine 3,072-dimensional shared
embedding; its dot product is rank-equivalent to
`official_dot + beta * mass_kernel_dot`.

Beta was selected on 2,048 identity-equal training-formula queries at their
natural 11.9% official error prevalence.  The previously used 50/50
error/correct sample is retained only as a headroom diagnostic and is not used
for hyperparameter selection.  The selected beta was 0.10.

On all 1,929 held inner identities:

| Embedding | Recall@1 | Delta | Corrected / introduced | Formula-cluster 95% CI |
|---|---:|---:|---:|---:|
| frozen official | 90.461% | - | - | - |
| official + mass kernel | **91.343%** | **+0.881 pp** | **20 / 3** | **[+0.389, +1.291] pp** |

Decision: `DEVELOPMENT_KERNEL_PASS`.  Exact peak mass paired with its observed
intensity is a deployable signal.  The earlier +0.778 pp figure and its
fragment/uniform/control ablations are withdrawn because that audit took a
separate molecule-level maximum in each channel before fusion.  Such an order
can combine evidence from two different reference spectra and is not a valid
evaluation of one shared spectrum embedding.  All figures retained here fuse
each query-reference pair first and only then take the molecule maximum.

## Explicit chemical-rule response extension

The repository rule library was next converted into a spectrum-only response
embedding: 214 curated neutral-loss masses and 102 characteristic-fragment
masses are matched with a 0.02 Da triangular tolerance.  The response vector
is computed independently for every query and reference spectrum.  Combining
it with the mass kernel therefore remains a valid shared embedding and never
uses candidate structures or candidate-list statistics at inference.

Hyperparameters were again selected only on the 2,048 training-formula
identities.  Fusion is performed for every query-reference spectrum pair
before taking the maximum reference score for a candidate molecule; taking
separate per-channel maxima is invalid because they may select different
references.  On the same 1,929 held inner identities, the correctly aggregated
rule-mass embedding reached **91.706% Recall@1**, or **+1.244 pp** over official
DreaMS (32 corrected / 8 introduced; formula-cluster 95% CI **[+0.649,
+1.785] pp**).  It exceeded the corrected mass kernel by **+0.363 pp**, with
paired formula-cluster 95% CI **[+0.055, +0.764] pp**.  The rule-mass embedding
beat the independently tuned mass-shifted rule control by **+1.037 pp**, with
paired formula-cluster 95% CI **[+0.317, +1.143] pp**.  Rule response alone tied
the combined embedding at Recall@1 but had one more correction and one more
introduced error; their paired interval crossed zero.  The selected beta for
both deployable rule variants is 0.20.

## Direct rule-kernel transfer into official DreaMS

**Formal update (Slurm 2331449):** all eight arms completed.  Ordinary
continuation and the target `rule_mass` arm both reached 90.9279% Recall@1
(+0.4666 pp versus official), so Phase A did not establish any
ChemAware-specific increment.  The locked result and the replacement
rank-equivalent paired-margin transfer design are recorded in
`docs/CHEMAWARE_RULE_KERNEL_DIRECT_PHASE_A_RESULT_20260905.md`.  The planning
text below is retained as the pre-run contract.

The stronger combined rule-mass kernel is now used as a frozen clean-visible
teacher for the same
official DreaMS encoder.  For every training candidate list, the student is
optimized against the teacher distribution induced by the training-formula
selected shared-kernel score, while retaining hard-label
full-list ranking, in-batch spectrum contrast, initialization margin floors,
and geometry preservation.  Only the final Transformer block and official
projection head are trainable.  The teacher is discarded after training; the
deployed model remains one clean spectrum to one 1,024-dimensional embedding.

The development direct-transfer experiment contains eight schedule-matched
arms in one Slurm array:

1. ordinary continuation (`none`);
2. the corrected mass teacher (`mass`, beta 0.10);
3. rule response alone (`rule_response`, beta 0.20);
4. the target combined teacher (`rule_mass`, beta 0.20);
5. mass-shifted rules (`rule_mass_shifted`, beta 0.20);
6. error-routed mass only (`mass_error`, beta 0.10);
7. error-routed target (`rule_mass_error`, beta 0.20);
8. error-routed shifted-rule control (`rule_mass_shifted_error`, beta 0.20).

The final three arms use candidate retrieval outcomes only to route the
training-time teacher loss.  Hard-label candidate ranking, the margin floor,
and geometry preservation still cover every scheduled query; the deployed
encoder receives only a clean spectrum and has no router.  The mass and
shifted-rule arms are repeated under the same scope so any routed-target claim
has dose- and scope-matched controls.

The direct Slurm entrypoint remains
`tasks/run_chemaware_mass_kernel_direct_arm.sbatch`; it now launches the
eight-task array.  The local unit tests and real official-checkpoint
`rule_mass` end-to-end CPU smoke passed, including one optimizer step, replay,
evaluation, and checkpoint serialization.  No formal trained-model result
or Slurm job ID exists yet because the server currently accepts TCP connections
but rejects both available local keys before command execution.

A real-model two-query gradient audit was also run before cluster submission.
The correct and mass-shifted rule teachers used the same candidates, official
initialization, and 13,638,656 trainable parameters.  Their gradient cosine was
0.5860 at the corrected beta 0.20, the differential gradient norm was 2.1770,
and all 11 trainable
parameter tensors had nonzero differential gradients.  This establishes that
the final DreaMS block can receive a rule-specific optimization direction; it
does not establish retrieval improvement, which remains the role of the
eight-arm GPU experiment.

A locked-inner routing audit supplies the reason for the three routed arms.
Unrouted pair-first `rule_mass` corrected 32 official errors and introduced 8
new errors (+1.244 pp).  Applying its frozen diagnostic outcome only to the 184
official-error queries retained all 32 corrections with no introduced errors
(+1.659 pp; formula-cluster 95% CI +0.812 to +1.951 pp).  Extending the route
to baseline-correct low-margin queries monotonically reduced risk utility.
This is teacher-routing headroom, not a trained-embedding result, and the outer
fold remains untouched.

The exact formal schedule was frozen locally with the same Slurm arguments:
4 epochs, 4,096 optimizer steps, and 46,955 raw spectra materialized across
training and full inner evaluation.  Two reference spectra are sampled per
candidate molecule, reducing the variance of the molecule-level maximum while
keeping the schedule practical.  A batch contains at most 80 unique reference
spectra (95th percentile 39), and a query contains at most 23 legal candidate
molecules after formula-disjoint filtering.  A real official-checkpoint
two-reference forward/backward smoke also passed.  Four references passed the
same smoke but increased the one-step pre-clip gradient norm from 1.25 to 5.18,
so it is reserved for a later capacity arm instead of silently increasing the
first formal run's optimization risk.
