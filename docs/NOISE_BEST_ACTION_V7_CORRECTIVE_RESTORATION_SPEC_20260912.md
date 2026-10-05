# Noise best-action V7 corrective-only direct fine-tuning contract

Date: 2026-09-12  
Status: executed on fold 0 / seed 20260911; rejected by the completed three-arm
result.  See `NOISE_BEST_ACTION_V7_RESULT_AND_ROOT_CAUSE_AUDIT_20260913.md`.

## Purpose

V7 preserves the mature E4/E8 direct shared-encoder route and the complete V6
strict best-action bank.  It does not introduce a teacher, embedding target or
distillation loss.  Its only purpose is to stop robust/harmful auxiliary updates
from being counted and amplified as if they were clean-boundary corrections.

The running V6R job is immutable because it executes its own source snapshot.
V7 uses seed `20260911` so a later V7 execution can be compared with V6R without
adding a seed change.

## Frozen action and graph boundary

The registered input remains:

- 32,127 strict Top-1 corrective rows before the `5e-6` replay floor;
- 32,114 admitted corrective rows after the floor;
- 3,482 unique outer-training queries;
- all seven executable sources: `N_mature`, `P_guided_original`, `E10B`,
  `E11`, `E12B`, `A4_exact`, and `V4_gradient_path`;
- no one-best-per-query compression;
- 85,959 harmful rows and 179,289 robust rows retained in separate branches;
- 18,333 outer-held queries used only for final evaluation.

The `3482 / 65286 = 5.333456 pp` value remains training action headroom.  It is
not a learned checkpoint result or a promise of held performance.

## Corrective-only optimizer counterfactual

For every active optimizer step V7 evaluates three virtual AdamW updates from
the same pre-step model, moments and step number:

1. `U_protect`: protective clean/full-graph gradient only;
2. `U_noncorrective`: protective plus projected robust/harmful auxiliary
   gradients;
3. `U_combined`: the actual protective, auxiliary and corrective gradient.

Only

```text
U_corrective = U_combined - U_noncorrective
```

is eligible for restoration.  V7 never uses `U_combined - U_protect` as the
restored signal.  The latter contains auxiliary action content and was the
scientific defect in V6R.

Restoration is independent in the projection head and last unfrozen backbone
block.  It is norm-neutral within each group, has gain at most `4.0`, targets a
corrective-attributable update fraction of `0.25`, and retains at least `0.90`
of the projection on the independent `U_protect` safety reference.

The formal end-to-end gate is the same `0.25`, not the legacy V6 `0.10` gate.
It is applied globally and separately to head and backbone at p10.  In addition,
both optimizer groups must reach the restoration target together on at least
`90%` of all 14,032 action-active steps.  Target coverage is therefore a
promotion condition rather than an unreported diagnostic mean.

If a group has a zero corrective residual, or the residual vanishes after the
protective projection, V7 applies the original combined update and records a
failed restoration target.  It does not terminate the run before held
evaluation.

## AdamW state reconciliation

After the ordinary AdamW step and corrective-only parameter materialization,
V7 solves the AdamW update equation for the stored first moment so that the
moment, current second-moment denominator and step number reproduce the update
actually applied.  The stable update-scale AdamW equation reconstruction error
must be at most `1e-6` on every active step.  FP32 replay through a
parameter-scale subtraction is reported separately as numerical quantization
and is never substituted for the stable state-consistency check.

The second moment remains the second moment of the real combined gradient and
is reported exactly as such.  V7 therefore does not claim to be an unmodified
AdamW optimizer; it is an explicit, audited direct-update arbitration rule.

Protective-gradient reach is checked for every trainable parameter in both
optimizer groups on every active step.  Missing protective reach fails the
signal contract instead of silently labelling weight decay or momentum as
corrective signal.

## Transfer allocation

V7 uses the already developed `mass_neutral_monotone` within-query allocator.
It preserves per-query target mass and strength ordering while avoiding the
recurrent individual-target compression of `hard_cap`.  This is a registered
best-bundle change, not retrospectively described as a one-variable replication
of V6R.

## Causal arms and resources

The formal entrypoint remains
`tasks/run_noise_corrected_best_v6_full_2gpu.sbatch` for server compatibility,
but the job and output directories carry the V7 identity.  It requests exactly
two GPUs, no manual memory amount, and runs:

- routed best actions;
- source/family/exact-recipe matched shuffled actions;
- matched clean continuation.

Routed and shuffled receive the same optimizer rule.  The clean arm is an
explicit no-op for restoration.

## Decision boundary

V7 is not successful merely because the signal diagnostic reaches `0.25`.
Success still requires, on the complete held graph:

- Recall@1 gain at least `+4.0 pp` versus the exact E8 initialization;
- strictly positive formula-cluster and near formula-cluster confidence bounds;
- routed strictly better than shuffled and clean controls;
- registered non-regression/improvement checks for Recall@1/2/3/5/10/20, MRR,
  mean/median rank, macro-query AUROC/AUPRC, micro-candidate AUROC/AUPRC,
  positive-vs-best-negative margin, Top1-Top2 gaps, corrected/introduced,
  risk-net, near-subset metrics, and MassSpecGym pairwise metrics;
- every schedule, provenance, protective-reach, corrective-only restoration and
  AdamW reconciliation contract passing.

One seed can only produce `pending_multiseed`, never final promotion.  No code
path or action-headroom calculation guarantees a 4-5 pp encoder gain.

The completed job-2336335 result confirms this boundary: routed Recall@1 gained
only `+0.130911 pp` versus exact E8, was `-0.060001 pp` below shuffled, and the
end-to-end corrective update fraction remained only `0.089913` at p10.  V7 must
not be expanded unchanged to another seed.

## 2026-09-12 report-closure repair

Audit of the still-running seed-20260908 V6 log exposed two fail-open reporting
gaps before V7 submission.  The real trainer's configuration exporter selected
the V6 registry even for V6R/V7, which omitted restoration scope, materialization,
continuation and AdamW-reconciliation fields required by the final summarizer.
It now exports the registry selected by the actual direct contract.  The same
audit showed the old non-materialized V4 candidate reached its `0.25` target on
only `87.5%` of sampled steps; V7 now records and gates target coverage as
specified above.  These repairs change validation/report closure only; they do
not alter the frozen action bank, training schedule, optimizer dose, causal arms
or held graph.

## 2026-09-12 FP32 reconciliation repair

Job `2336248` passed the complete source/test/preflight closure and reached the
first optimizer step, then the first-moment audit reported relative error
`3.629975e-4` with only a `1.0169%` first-moment change.  The old audit formed a
small AdamW displacement by subtracting two parameter-scale FP32 tensors.  That
is catastrophic cancellation, not optimizer-state divergence.  Reconciliation
now targets the displacement actually materialized in the parameter tensor and
solves/verifies decay plus adaptive descent directly at update scale.  The
strict `1e-6` state-equation gate is retained; parameter-scale FP32 replay and
requested-versus-materialized rounding are recorded as separate diagnostics.
