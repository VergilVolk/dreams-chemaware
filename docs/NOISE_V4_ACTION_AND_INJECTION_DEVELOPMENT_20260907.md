# Noise v4 direct-action and injection-loss development record

Date: 2026-09-07

## Scope and immutable objective

The objective remains one shared clean-spectrum DreaMS encoder embedding.  E4/E8
are direct-fine-tuning foundations.  N, P, A4, E10B, E11 and E12B are real
spectrum actions and boundary supervision, not embedding teachers.  No
distillation route is introduced here, and the currently running server v3 arm
is not modified in place.

The desired 4--5 percentage-point broad improvement is the experiment target,
not a guaranteed result.  Only the complete held graph and registered metric
suite may establish it.

The frozen outer-train geometry contains 65,286 eligible queries and 4,448 E8
Top-1 errors.  Existing routed actions actually reach Top-1 on 3,450 errors, an
upper-bound headroom of 5.2844 pp.  A 4 pp clean-embedding gain therefore needs
at least 2,612 net corrections, or 75.7% of that executable headroom; 5 pp needs
3,265, or 94.6%, before accounting for introductions.  This is why merely adding
more actions cannot substitute for reducing the clean-encoder injection loss.

## What the preceding formal epoch actually established

The supplied formal routed epoch completed 3,475 optimizer steps.  Every step
was action-active.  All 4,186 corrective queries produced 13,894 query
exposures; the 61,163 selected corrective actions produced 202,798 action
exposures, with every action seen three or four times at equal cumulative dose.
N/P/A4 corrective epoch mass was equalized.

The former hard 90% injection outage is therefore no longer present:

- action-gradient retention after risk projection and clipping: 0.999947;
- corrective-direction retention after risk projection and clipping: 1.01077;
- inner auxiliary projection p10 retention: 0.999801;
- pooled active transfer edge fraction: 0.779628.

The remaining losses are different and must not be mislabeled as the old
duty-cycle failure:

1. The actual optimizer update has only 0.138 mean cosine with the action
   direction.
2. The AdamW counterfactual attributes only 0.164449 of the update norm to the
   action residual, although that residual itself has 0.907593 alignment with
   the action gradient and 0.695992 with the corrective gradient.
3. A hard 0.10 transfer cap affects 22.505% of corrective edges overall and
   40.565% of `recurrent_peak_graft` edges, erasing strength ordering among many
   high-value actions.
4. The prior virtual-step reproduction error of 0.001244 came from omitting
   parameter-dtype writeback rounding in the audit implementation.  The virtual
   AdamW code now mirrors the in-place update and passes a 65,536-dimensional
   FP32 `foreach=True` exactness test.  This is an audit repair, not a performance
   claim.

## Real current-E8 multi-peak action experiment

Output:
`data/validation/noise_corrected_action_expansion_v4_local_e8_panel128_20260907`

The panel was selected before executing any new action, excluded outer formula
fold 0, contained 104 formulas, and used the current E8 checkpoint.  After
recomputing the full candidate geometry, 47 selected queries were current-E8
errors and 81 were correct.  Every action used real fragment tokens and was
compared with a disjoint same-role, intensity/mz-matched composite control.

| action | errors corrected | correct introduced | risk-net | target-control Top1 mean | formula-cluster 95% CI |
|---|---:|---:|---:|---:|---:|
| quad attenuation 4x50% | 5/45 | 0/74 | +5 | +0.0571 | [+0.0183, +0.1048] |
| supported boost 2x50% | 5/47 | 0/81 | +5 | +0.0345 | [+0.0083, +0.0714] |
| quad attenuation 4x25% | 3/45 | 0/74 | +3 | +0.0381 | [+0.0091, +0.0777] |
| dual attenuation 2x50% | 4/47 | 1/80 | +2 | +0.0336 | [0.0000, +0.0776] |
| conservative exchange | 6/47 | 3/80 | 0 | 0.0000 | [-0.0481, +0.0451] |
| signed multiplicative | 4/47 | 2/80 | 0 | +0.0097 | [-0.0306, +0.0526] |
| single attenuation 1x50% | 1/47 | 2/81 | -3 | -0.0080 | [-0.0417, +0.0250] |

The quad-50 and supported-boost corrections currently overlap completely (five
queries); they are independent corroborating views, not ten-query headroom.
Both add four corrections beyond the new single-peak action.  Quad-25 is a
lower-dose robust view.  Dual, exchange, signed and single variants are not
qualified as new corrective actions by this panel.

Those table counts are raw action-space outcomes.  Applying the formal router's
stricter requirements (complete matched control and target-minus-control margin
at least 0.01) removes query 80439, whose control could not be constructed.  The
strict fixed quad/boost union is therefore four queries, not five.

The same formal three-semantics thresholds also show why a mixed family must
not be accepted or rejected wholesale.  Among rows with complete controls,
quad-50 routes as 25 corrective / 55 robust / 6 harmful / 19 uncertain;
adaptive up as 14 / 54 / 9 / 35; adaptive down as 23 / 58 / 12 / 25; and
adaptive joint as 18 / 53 / 17 / 33.  Only the exact routed instances may enter
their corresponding objective branch.

## Developed v4 injection candidates

### Mass-neutral monotone transfer allocation

The v3 0.10 hard cap was replaced in an isolated scalar and tensor-level v4
candidate by within-query, mechanism/source/family water filling.  It preserves
the exact old capped target mass and maximum optimizer dose while retaining the
ordering of strong actions.

On the local v3 training ledger:

- 466 distinct positive targets become 638 (+36.91%);
- 74 strong targets remain distinguishable above the old cap;
- maximum target delta is 0.129903 under a hard 0.20 ceiling;
- query/source/family and source mass errors are at floating-point zero;
- no ordering inversion occurs.

Tensor tests also verify that aggregate transfer gradient mass is unchanged in
v3's linear Huber region.  This attacks strength compression without increasing
learning rate, epochs, action count or total transfer target mass.

### Optimizer-space action restoration

An isolated pure v4 helper decomposes one virtual AdamW step into a risk-only
counterfactual plus its action-attributable residual.  If the attributable
fraction is below a registered target, it removes only direct opposition to the
risk update, applies a capped residual gain, and rescales the complete update to
the original combined-step norm.  CPU tests verify target attainment, exact
norm neutrality, risk non-conflict, gain-cap behavior and dtype/layout
preservation.

This has not yet been wired into a formal trainer.  A full-tensor, memory-bounded
implementation and real gradient panel are required before any server arm.

A real AdamW unit experiment with 20 momentum-dominated risk steps drove the
unmodified action-attributable update fraction below 0.10.  The bounded
restoration raised it above 0.25 with unchanged total update norm and a maximum
gain of four.  This validates the mechanism, not encoder performance.

### Lightweight directional update repair

A second pure helper addresses the directly observed 0.138 update/action
cosine without constructing a risk-only AdamW update for every training step.
It minimally rotates the realized AdamW descent update toward the already
risk-projected action gradient, preserves the exact original update norm, and
backs off the rotation whenever it would reduce the original first-order
protective descent component.  On a vector with the observed 0.138 cosine, a
coefficient below 0.20 is sufficient to reach 0.25 alignment.  Tests cover
exact norm neutrality, an already-sufficient no-op, conflicting-action
projection, protective non-degradation and a risk-limited no-op.  This remains
an unwired v4 candidate.

Both counterfactuals are now attached to the existing 16 registered optimizer
audit positions in metric-only mode.  They do not change parameters and do not
materialize another full 117M-parameter update tensor, avoiding a new GPU-memory
spike.  A future run will therefore report feasibility on real action/risk
gradients before either repair can be enabled.

### Minimal-dose trust-region actions

A second isolated action family replaces fixed 50% edits with the minimum
first-order dose predicted to close the current clean-margin deficit.  It:

- attenuates only non-identity adverse peaks;
- boosts only existing identity/shared supported peaks;
- enforces limits on edited peaks, per-peak fraction and total fraction;
- freezes its recipe before observing the action outcome;
- remains a direct spectrum intervention, not a teacher target.

Down-only, up-only and joint variants were evaluated on the same 128-query E8
panel with the same matched-control protocol.

The completed result qualifies adaptive up as the broad safe variant: 5/47
errors corrected, 0/81 correct queries introduced, risk-net +5, matched-control
Top1 mean +0.0357 and formula-cluster 95% CI [+0.0086, +0.0714].  Its five
corrections exactly match fixed supported-boost and quad-50, so it improves dose
discipline but does not add headroom on this panel.  Down-only and joint have
family-level risk-net -1 and +1 with CIs crossing zero, so they are not broad
corrective families.  They nevertheless both produce one additional exact,
matched-control-specific correction (query 32906, formula C15H12O6) beyond the
four-query strict fixed-action union, raising the strict route-eligible union to
five.  They may therefore enter a three-semantics router as mixed sources, but
never as globally corrective views.

### Strong-dose stress test

Output:
`data/validation/noise_corrected_action_expansion_v4_local_e8_panel128_strong_20260907`

The predeclared strong variants increased the desired margin from 0.05 to 0.15,
the per-peak bound from 0.75 to 0.90, the total fractional-dose bound from 2.0
to 3.0 and the edited-peak bound from six to eight.  They do not qualify as
broad safe families:

| action | errors corrected | correct introduced | risk-net | target-control Top1 mean | formula-cluster 95% CI |
|---|---:|---:|---:|---:|---:|
| adaptive strong down | 8/47 | 4/81 | 0 | +0.0531 | [-0.0090, +0.1182] |
| adaptive strong up | 6/47 | 1/81 | +4 | +0.0381 | [0.0000, +0.0857] |
| adaptive strong joint | 6/47 | 4/81 | -2 | +0.0273 | [-0.0275, +0.0841] |

Formal per-instance routing is more informative than those family aggregates.
Strong down has seven strict Top-1 corrections and strong up has five.  Relative
to the five-query strict union from the fixed and mild adaptive variants,
strong down adds queries 33669, 77301 and 80302; strong up adds only 77301.
The complete strict union therefore rises from five to eight queries.  These
incremental rows may be retained as routed corrective instances, while the
introduced and adverse rows must enter only the harmful boundary branch.

The stress test also falsifies the assumption that simply raising a one-shot
dose solves the action bottleneck.  Median total fractional dose is the maximum
3.0 for every strong direction, while only 26.6% of up, 30.5% of down and 40.6%
of joint actions reach their first-order target.  The clean-state gradient is
therefore stale after large multi-peak edits.  The next action candidate must
re-encode the spectrum, refresh the active hard negative and reselect a
supported peak after every direct edit.  This is action-path refinement within
E4/E8 direct fine-tuning, not distillation.

### Sequential re-encoded supported boosting

Output:
`data/validation/noise_corrected_action_expansion_v4_local_e8_panel128_sequential_boost_20260907`

The follow-up implements that refresh literally.  After every 50% boost it
re-encodes the edited spectrum, recomputes the positive-versus-current-hard-
negative boundary, refreshes the input gradient and chooses a new supported
peak.  The complete target path is frozen before constructing a disjoint
same-role control path.  It is direct spectrum fine-tuning evidence; no teacher
embedding or distillation target is present.

| sequential prefix | errors corrected | correct introduced | risk-net | complete controls | target-control Top1 mean | formula-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 1 x 50% | 3/47 | 1/81 | +1 | 118 | +0.0169 | [-0.0163, +0.0522] |
| 2 x 50% | 5/45 | 0/80 | +5 | 103 | +0.0583 | [+0.0189, +0.1068] |
| 3 x 50% | 3/38 | 0/77 | +3 | 88 | +0.0455 | [+0.0109, +0.0941] |
| 4 x 50% | 4/33 | 0/75 | +4 | 79 | +0.0759 | [+0.0244, +0.1392] |
| 5 x 50% | 3/26 | 0/63 | +3 | 68 | +0.0735 | [+0.0147, +0.1389] |
| 6 x 50% | 3/26 | 0/55 | +3 | 65 | +0.0462 | [-0.0156, +0.1129] |

Two steps are the broad candidate: zero introductions, positive clustered CI,
the largest safe eligible panel and risk-net +5.  Four steps are a routed
extension rather than a global recipe because they have a larger paired effect
but substantially lower path/control availability.  Later prefixes cannot be
compared by raw correction rate because their eligible denominators shrink.

The strict sequential-prefix union is queries 70624, 71661, 76144, 77301 and
80439.  Query 80439 is genuinely new relative to the prior complete eight-query
fixed/mild/strong union: its one-step target changes rank 2 to rank 1, its
matched control remains rank 2, target-minus-control margin is +0.02263 and it
passes the exact corrective route.  The complete local strict union is now nine
queries.  This expands action coverage by one on this fixed panel; it is not a
claim of nine additional clean-input encoder corrections.

## Runtime correction and bounded two-GPU canary

The first formal-v3 submission is computationally mis-sized.  Its smoke run
used 163.17 seconds for eight optimizer steps, while the formal schedule has
3,475 steps per epoch and four epochs per arm.  At the observed rate that is
about 19.7 hours per epoch and 78.8 hours per arm.  Two workers serializing four
and five arms cannot finish the nine-arm design within the 96-hour allocation;
the task ordering also prevents any complete same-seed routed/shuffled/clean
triad before the limit.  Silence after the two 87,848-row initial encodes is not
evidence of an exception: the submitted trainer had no calibration or in-epoch
heartbeat.

`tasks/run_noise_corrected_direct_v3_canary_2gpu.sbatch` is the corrected
decision experiment.  It runs routed and source/family/exact-recipe shuffled
arms concurrently on two GPUs, then the clean continuation control, using 512
corrective, 512 harmful, 256 robust, 4,096 clean and 4,096 outer-held queries
for one epoch.  It requests no manual memory amount.  The trainer now emits the
first step and every 25 canary steps, and the canary summary compares the exact
same initial E8 rows across all three arms with the complete registered metric
family and formula-cluster paired intervals.  It cannot promote a checkpoint;
it decides whether a full run is worth the multi-day cost.

## Next gates

1. Carry sequential two-step boosting as the broad candidate and route the
   four-step prefix only per exact instance; do not treat either family as
   globally corrective.
2. Evaluate new actions on residuals of the complete N/P/A4/E10B/E11/E12B
   routed union, not merely residuals of historical single-peak A4.
3. Run a real-gradient, memory-bounded optimizer-space attribution pilot before
   wiring update restoration into a training arm.
4. Run the bounded two-GPU v3 canary before another multi-seed formal job; do
   not interpret the overlong v3 job's silence as a model result.
5. Judge any trained checkpoint only on the full registered held metrics,
   including formula-cluster paired confidence intervals and the MassSpecGym
   10-ppm pooled pairwise AUROC.  Action-space corrections are not encoder
   improvement.
