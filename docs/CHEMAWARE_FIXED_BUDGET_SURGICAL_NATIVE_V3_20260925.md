# ChemAware fixed-budget surgical native curriculum

## Decision

The next embedding experiment is a joint retraining from the official DreaMS
embedding, not another continuation from the +1.8144 pp stage-1 checkpoint.
It retains the successful stage-1 native triplet curriculum and replaces a
small number of its events with frozen counterfactual-specific,
reference-aligned events. It never appends events.

This directly addresses the two observed failures:

- pair expansion increased hard-event count but traded corrections for new
  errors;
- post-stage-1 specific replay produced a weak local Recall@1 gain while
  reducing micro-AUC and positive-margin geometry.

## Identifiable intervention

Let the frozen stage-1 pool contain `N` events `B_i`, and let `S_d` be the
replacement positions for dose `d`. For arm `a`, the optimized empirical loss
is

`L_a(theta) = N^-1 [sum_(i not in S_d) loss(B_i; theta) + sum_(j in S_d) loss(C_a,j; theta)]`.

The event count, background events, replacement positions, initialization,
optimizer, step count, and random seed are identical across the correct arm and
three semantic-null arms. Therefore

`grad L_correct - grad L_null = N^-1 sum_(j in S_d) [grad loss(C_correct,j) - grad loss(C_null,j)]`.

The background gradient cancels exactly. This was not true when each arm used
its own full stage-1 pool, because thousands of non-intervention triplets could
differ.

## Event-quality rules

1. Keep the total number of events unchanged.
2. Use the same successful stage-1 correct-arm pool as the background for all
   four arms.
3. Replace the same event positions in all arms.
4. Use only explicit reference-aligned positive and negative spectra at an
   intervention position. Union with the old molecule-wide reference set is
   forbidden because it is usually a no-op.
5. Do not create duplicate query-candidate pairs.
6. Keep at least 95% of the stage-1 events untouched.
7. Copy the stage-1 validation pool unchanged; select checkpoints only by the
   external formula-role-2 retrieval graph.

## Frozen experiment

- Initialization: official DreaMS embedding.
- Doses: one and two replacements per eligible query.
- Optimizer/loss/preprocessing: native DreaMS, unchanged.
- Checkpoints: 1000, 2000, and 3000 optimizer steps.
- Role 2 gate against stage 1: strictly positive Recall@1, positive
  formula-cluster CI lower bound, `corrected - 2 * introduced > 0`, nonnegative
  MRR, Recall@3, micro-AUC, and macro-AUC.
- Only the winning role-2 dose/step may train the three matched null arms and
  run role 3.
- Role 4 remains untouched.

## Five-point target

On 1,929 role-3 queries, a +5 percentage-point Recall@1 gain over the official
embedding requires about 97 additional net top-1 corrections. The target is
therefore a release gate, not an assumed outcome. A result below +5 pp may be
retained as mechanistic evidence only if it still passes every stage-1 safety
and null-specificity gate.

## Entry point

`sbatch tasks/run_chemaware_surgical_native.sbatch`
