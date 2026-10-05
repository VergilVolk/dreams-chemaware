# Noise best-action direct V5: frozen canary specification

Date: 2026-09-07

## Decision

The next runnable experiment is one direct-fine-tuning canary that combines the
already completed formal N/P66/A4 routed bank with the qualified V4 raw-spectrum
actions and the mass-neutral monotone transfer allocator.  It starts from the
mature E8 shared clean-spectrum encoder, trains that same encoder, and performs
clean-spectrum inference.  It does not distil a teacher embedding.

The 4--5 percentage-point target is a strict experimental gate, not a result
that code can guarantee.  The bounded canary can reject this implementation or
authorize a larger experiment; it cannot by itself establish a full-graph 4 pp
gain.

## Correction of the previous interpretation

The failed v3 canary did not literally omit every historical action.  Its
ledger contained `N_mature`, `P_guided_original`, `E10B`, `E11`, `E12B`, and
`A4_exact`.  E4 and E8 are training/initialization foundations rather than two
additional spectrum-action banks, while E13 reused E12B actions and is not a
new bank.  Calling all of those assets absent would be another provenance
error.

The real defects were narrower and actionable:

1. The canary predated the qualified V4 multi-peak, adaptive-dose, and
   sequential re-encoded paths.
2. The old hard 0.10 target cap compressed strong-action ordering on 22.3% of
   canary transfer edges, despite the action gradient reaching every optimizer
   step.
3. A historical `S3A+A4` label did not prove A4 consumption: that path had null
   `query_row` wiring.  V5 instead asserts the selected ledger source rows
   immediately before training.
4. Action-space headroom was repeatedly confused with trained clean-embedding
   performance.  V5 keeps them separate.

The original 90% duty-cycle outage is not present in the repaired trainer.  The
completed v3 evidence showed 100% action-active optimizer steps, approximately
0.99995 post-projection/clip retention, and approximately 0.78 active transfer
edges.  V5 therefore preserves that injection kernel and changes only the
qualified action content plus the target-allocation compression identified by
the audit.

## Frozen executable action union

The mature route remains:

- `N_mature`: candidate-gradient and role-confounder paths;
- `P_guided_original`: the original six guided intensity cells;
- `E10B`: 19 registered cells;
- `E11`: 16 registered cells;
- `E12B`: 25 relaxed-recurrence cells;
- `A4_exact`: exact-peak paths.

The new `V4_gradient_path` source contains exactly nine frozen recipes:

- quad attenuation at 4x25% and 4x50%;
- supported boost at 2x50%;
- adaptive trust up;
- exact-routed adaptive trust down and strong down;
- sequential re-encoded supported boost prefixes 1x, 2x, and 4x.

Single attenuation, dual attenuation, signed multiplication, conservative
exchange, strong up/joint, adaptive joint, and sequential 3x/5x/6x remain
excluded because their frozen development evidence was harmful, redundant,
uncertain, or inferior.  Mixed qualified families are not globally labelled
corrective: each target is compared with a disjoint same-role matched-neutral
control and routed into corrective, harmful, robustness-only, or uncertain.
Uncertain rows have exactly zero training weight.

V4 is assigned to the N top-level mechanism because it is a direct
input-gradient spectrum path.  N/P/A4 receive equal corrective mechanism mass;
within N, `N_mature` and V4 receive source/family opportunities before recipe
multiplicity.  This prevents the 66 P recipes from starving either source.

## Low-loss direct injection

The clean query and live clean candidate references remain the deployed
objects being updated.  The real action spectrum is also encoded so its payload
and live symmetric consistency reach the same encoder.  Action and matched
control margins are detached only when they define a conservative clean-boundary
target; this is direct boundary supervision, not embedding imitation.

The V5 allocator uses within-query, mechanism/source/family mass-neutral
water-filling.  It preserves the old total target mass and the maximum optimizer
dose, allows a maximum individual target of 0.20, and preserves strength
ordering.  On the frozen local ledger this changed 466 positive targets to 638,
kept 74 informative targets above the old cap, and introduced no ordering
inversion or mass error.

The optimizer-space restoration helpers remain audit-only.  The bounded v3
canary's measured optimizer action fraction was already above the registered
0.10 floor, and applying a 117M-parameter post-AdamW rewrite at every step would
be a new unvalidated intervention.  V5 does not silently add it to a supposedly
mature combined run.

### V4 clean-geometry correction after job 2332693

The first V5 attempt completed its full V4 route but stopped in the prospective
plan audit before training.  V4 had selected its panel from the batch-256 cached
E8 geometry, then mistakenly wrote each action's `clean_rank` and
`clean_margin` from a second live gradient-batch forward.  N/P/A4 consistently
write the immutable cached geometry.  Combining those records therefore made
some queries appear to have two clean ranks even though every route used the
same checkpoint hash.

V4 now uses the exact `initial_ranks` and `initial_margins` arrays that selected
the panel; the live forward is used only for the input-gradient action
direction.  The combined-ledger builder independently rejects any future
cross-source rank disagreement before it streams and materializes the selected
spectrum tensors.  Job 2332693 performed no optimizer step and its staging
artifact is not a model result.

## Fail-closed server sequence

The single SBATCH entry point:

1. compiles and runs the routing, injection, evaluation, and summary tests;
2. reuses the completed formal N/P66/A4 route artifact from job 2332161;
3. mines all current-E8 outer-train errors plus the registered vulnerable
   correct boundary panel for the nine V4 recipes;
4. combines four route directories and asserts all seven executable sources,
   a non-empty V4 corrective subset, explicit control semantics, no held-formula
   leakage, a lossless selector frontier, and at least 4 pp training-geometry
   action headroom;
5. runs routed and exact-stratum shuffled arms concurrently on two GPUs, then
   the matched clean continuation arm;
6. writes a unique atomic output directory and never overwrites a prior run.

The canary uses 512 corrective, 512 harmful, 256 robust, 4,096 protective-clean,
and 4,096 outer-held queries for one epoch.  These bounds make it a fast causal
decision experiment.  A later formal experiment must remove all query limits,
train the complete 65,286-query fold-0 outer-train graph for the registered
schedule, and evaluate all 18,333 held queries.

Manual submission from the server repository root is exactly:

```bash
sbatch tasks/run_noise_corrected_best_v5_canary_2gpu.sbatch
```

The script requests exactly two GPUs and contains no manual memory request.

## Simultaneous evaluation contract

Every arm reports, on the identical clean-input held query set:

- Recall@1/2/3/5/10/20;
- MRR and mean/median rank;
- macro-query AUROC/AUPRC;
- micro-candidate AUROC/AUPRC;
- positive-vs-best-negative margin and Top1--Top2 gap;
- corrected, introduced, risk-net, and the corresponding near-subset outcomes;
- near-subset retrieval/macro/margin metrics;
- formula-cluster paired confidence intervals;
- MassSpecGym all-adduct and `[M+H]+` strict-10-ppm pooled pairwise
  AUROC/AUPRC.

The pooled MassSpecGym metric is explicitly not called an exact reproduction of
the NIST20 paper value near 0.85.  A strict follow-up gate requires routed to
beat initial E8, shuffled action, and clean continuation on Recall@1 and
risk-net, have a strictly positive formula-cluster Recall@1 lower bound, and
show no regression in any registered metric direction.  Saturated Recall@k or
median rank may tie but may not worsen.
