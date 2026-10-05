# Noise best-action V6 full direct fine-tuning specification

> Historical boundary: the running V6R source snapshot remains immutable, but
> this local submission route is superseded on 2026-09-12 by
> `NOISE_BEST_ACTION_V7_CORRECTIVE_RESTORATION_SPEC_20260912.md`.  V6R restores
> a composite action residual; V7 restores only the conditional corrective
> residual and must be reported under its separate contract name.

Date: 2026-09-08

## 2026-09-12 V6R optimizer-boundary repair

The live two-GPU entrypoint now runs the registered
`best_action_v6_restored` contract at seed `20260911`.  It supersedes the
failed seed-20260908 V6 execution for submission purposes, while preserving
that run and its diagnosis as immutable evidence.

The new server evidence localized the remaining loss after four complete
epochs.  Gradient-space transmission was already healthy: action retention
p10 was `0.9998`, attributable direction alignment p10 was `0.9354`, and
corrective alignment p10 was `0.2061`.  The failure occurred after AdamW:
action-attributable update fraction p10 was only `0.0538` (head `0.0546`,
backbone `0.0525`).  The existing V4 candidate reported possible restored
fractions but was invoked with `materialize_updates=False`; it never changed a
single encoder parameter.  Thus V6 measured the correct repair without using
it.

V6R changes only this optimizer boundary.  For every action-active optimizer
step and separately for projection head and backbone, it:

1. computes exact combined-gradient and risk-only AdamW counterfactual
   displacements from the same pre-step moments;
2. removes only direct action-versus-risk opposition;
3. increases the action-attributable residual toward fraction `0.25`, with
   gain capped at `4.0`;
4. preserves the original update norm independently in each parameter group;
5. preserves at least `90%` of the original protective/risk descent
   projection; and
6. advances ordinary AdamW moments from the real combined gradient, then
   writes the registered displacement into the live shared encoder.

The routed and exact-stratum shuffled arms use the identical restoration.  The
clean arm remains an exact no-action/no-restoration control.  No teacher,
embedding target, distillation, learning-rate increase, additional epoch,
additional optimizer step or new action mining is introduced.  The seven
source action bank, strict Top-1 admission, 877-batch/3508-step schedule, all
loss branches, E8 initialization and full held graph remain unchanged.

Counterfactual attribution is now exhaustive over all `4 x 3508 = 14,032`
active steps per causal arm, rather than sampled at 64 positions.  If a signal
gate nevertheless fails, the worker records the failure but still writes the
diagnostic checkpoint and complete frozen held evaluation.  Such a result
cannot pass the summary or promotion gate; this change only prevents another
long run from losing all scientific evidence before evaluation.

The immutable execution snapshot contains a 60-file SHA-256 source closure;
the added file is the optimizer-restoration contract test executed inside the
allocated job.

The `+4.0 pp` threshold remains a required empirical outcome, not a guarantee.
V6R fixes the proven 94.6% optimizer-boundary attenuation; whether the fixed
action signal generalizes to a 4--5 pp clean shared-embedding improvement is
decided only by the registered held metrics and causal controls below.

## 2026-09-09 deterministic schedule-geometry repair

Every 2026-09-08 V6 delivery archive, including
`noise_best_v6_exact_splice_20260908_repaired_final.tar.gz`, is superseded and
must not be resubmitted.  Only the post-incident schedule-repaired release may
be used.

Job 2333423 failed before its first optimizer update with
`4.0276 > 4.0000`.  This was an implementation/preflight omission, not a data,
GPU, action-bank or numerical-training failure.  Strict admission gives 3,482
corrective queries, whose natural size-four packing is 871 batches.  The full
39,165-query robust and 16,955-query harmful panels produce 14,031 auxiliary
microbatches and therefore require 3,508 optimizer steps at the frozen maximum
of four auxiliary microbatches per step.  The old implied recycle factor was
thus exactly `3508 / 871 = 4.027554535...`; 24 batches would have been exposed a
fifth time.  The prior preflight checked the separate counts but failed to check
their joint schedule geometry before model construction.

The repair does not raise the 4.0 cap, remove any action or safety example,
reduce the 3,508 optimizer steps, or change the injector, objective, calibration,
learning rates or four-epoch budget.  After the existing identity-stratified
ordering, it deterministically and without further RNG use repartitions the
same complete corrective query panels into the smallest cap-safe count:

```text
3,482 queries = 851 batches of 4 + 26 batches of 3 = 877 batches
3,508 steps   = 877 batches x 4 exact exposures
```

The 26 smaller batches are evenly dispersed rather than placed in a tail.
No `BoundaryExample` is split; flattened query order, query/action coverage and
all action/reference contents are byte-logically preserved.  The existing
`actual_queries / registered_batch_queries` cardinality factor makes a size-3
mean equal to its registered size-4 mass, so every corrective query/action has
exactly four physical exposures and weighted dose four.  The illegal 4.0275545
dose is reduced by 0.6843% to the already registered legal limit; preserving
that illegal excess and preserving the 4.0 cap are mathematically incompatible.

One shared integer geometry implementation now drives the allocated-job
preflight, the trainer and the standalone plan audit.  Formal V6 requires the
exact `871 -> 877`, 3,508-step geometry before `SpectrumStore`, model loading,
initial GPU encoding or calibration.  Every epoch rechecks the same geometry,
and the final three-arm summarizer rejects a missing, altered or cross-arm
inconsistent schedule report.

## 2026-09-08 fail-closed release repair

The first V6 delivery archive is superseded and must not be submitted.  The
repaired release keeps the scientific action-to-V3 training path unchanged,
but closes four deployment and decision-integrity gaps:

- the archive and compute-node source manifest contain the complete recursively
  resolved 59-file local Python/provenance closure, including package initializers, every test invoked by the
  Slurm entrypoint and the DreaMS backbone/runtime modules;
- the frozen E8 `decision.json` is pinned in addition to its checkpoint;
- every arm binds `held_per_query.csv.gz` to `decision.json` by byte SHA-256 and
  row count, and V6 requires exactly 18,333 actual rows with all rank-derived
  metrics recomputed from the CSV;
- this one-seed job reports `single_seed_gate_pass`; the compatibility field
  `promote` is always false until a separate registered multi-seed gate exists.

The registered seed, action admission, `hard_cap` objective, calibration,
optimizer, learning rates, four epochs and signal thresholds are unchanged.

## Immutable diagnosis of job 2333096

Job 2333096 completed; it did not crash.  It preserved all 32,114
numerically stable strict Top-1 action rows and therefore fixed the earlier
action-input outage.  The scientific result nevertheless failed:

- targeted versus initial E8 Recall@1: `+0.120002 pp`, formula-cluster CI
  `[-0.020440, +0.263916] pp`, 68 corrected and 46 introduced;
- targeted versus shuffled Recall@1: `-0.027273 pp`, 31 corrected and 36
  introduced;
- micro-candidate AUPRC versus E8: `-0.025065 pp`;
- all-adduct 10-ppm pooled pairwise AUROC versus E8: `-0.325679 pp`;
- `[M+H]+` 10-ppm pooled pairwise AUROC versus E8: `-0.360880 pp`;
- every promotion gate was false.

The 99.57% targeted action-view pass rate is not a learned result.  Those rows
were admitted precisely because the frozen E8 action view was already strict
Top-1.  The deployed clean view passed only 37.56% of its training boundaries.
Thus action materialization succeeded while clean-embedding transfer failed.

The failed run connected the best action bank to the old E4 symmetric cosine
interface.  It optimized each four-action physical batch immediately, used a
fixed 0.16 pre-scale followed by action weights and a success gate, and had no
candidate-boundary transfer branch.  This made repeated correlated views easy
to overwrite and did not directly encode the positive-versus-hard-negative
margin that made an action useful.  It was the wrong injection interface for
the right action bank.

## Numeric truth boundary

The current corrected-graph `+5.334988 pp` number is `3483 / 65286` before the
numerical replay floor.  After excluding the single numerical-boundary query,
the registered input is `3482 / 65286 = 5.333456 pp`.  Both numbers are the
fraction of outer-train queries having at least one materialized action that is
strict Top-1 in the frozen E8 geometry.  They are direct-action training
capacity, not a learned checkpoint result or a held-test promise.

The older `+4.929934 pp` number was also an action-oracle union on an already
consumed 5,923-query historical fold.  The best historical shared clean
encoder result was approximately `+0.6352 pp` on a now-retired candidate graph.
No prior checkpoint has established a 4--5 pp clean-embedding gain on the
current corrected graph.

V6 therefore retains 4 pp as a strict promotion gate.  It does not relabel
action headroom as model performance or claim that code can guarantee the
gate.

The single formal run is registered at seed `20260908`; a different seed is a
new replication and cannot be labelled as this formal V6 result.

## Frozen best-action input

V6 reuses, without re-mining, the immutable ledger from:

```text
data/validation/noise_corrected_best_v5_canary_fold_0_run_2332784/ledger
```

The preflight requires exactly:

- 327,678 routed rows: 62,430 corrective, 85,959 harmful and 179,289 robust;
- all seven executable sources: `N_mature`, `P_guided_original`, `E10B`,
  `E11`, `E12B`, `A4_exact` and `V4_gradient_path`;
- 32,127 strict Top-1 corrective rows before the frozen numerical margin
  floor;
- 32,114 rows and 3,482 unique queries after the `5e-6` floor;
- every admitted strict action retained; no one-best-per-query compression;
- common current-E8 clean ranks and zero cross-route clean-rank disagreement.

The formal entrypoint additionally pins the exact files, not merely their
names or counts:

- ledger report SHA-256 `246e7e871e6669fec9f2c62330a69bf9eb563a1b97b72b89732cecd682844349`;
- action CSV SHA-256 `93f0785a69b5e323490a0b543059fa213fabb6ff848697831efba5b3fed667aa`;
- action NPZ SHA-256 `6d57615aebbfb7bd6327bb7edb143c6837d2586221c61aa2ff45e833f5e1512a`;
- candidate graph/report SHA-256 `8a57bb3a9cccdf69a738f2b093bfad8f6fc4393b23f1342c7a035ac54ce58fa1`
  and `1a69db8f14d271adfa454900a206b5bc3cc4d1c51e91703e499ca74837bf557f`;
- E8 initialization SHA-256 `8047b3f58c6808c86b320ac94b9e610610384040fa3a03e8d70550eb438a24af`;
- E8 initialization decision SHA-256 `0a7de098701e9ef5f6c3fa2ee62fcc2e1b301dab44878286bd0b61bc31e3e9d9`;
- official checkpoint SHA-256 `8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245`;
- MassSpecGym HDF5 SHA-256 `ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f`;
- raw architecture checkpoint SHA-256 `9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2`.

The allocated job also checks a separately pinned manifest containing the
complete 59-file recursive local source/provenance closure: the selector, registered V3
trainer and objective stack (including the actual `hard_cap` implementation),
all transitive task imports, DreaMS backbone/runtime modules, causal shuffle,
evaluator, summarizer, and every formal test invoked by the job.  It verifies
all byte-exact SHA-256 values before loading the model.  The compact admitted-ID
and semantic-boundary hashes are deterministic derivatives of those pinned
inputs and sources; every arm records them and the final summarizer requires
exact cross-arm equality.

E4 is the direct shared-training foundation and E8 is the initialization; they
are not omitted spectrum-action sources.  E13 reused E12B and contributes no
new action bank.  E14's failed trainer is not reused.

## V6 direct injection contract

V6 changes no scientific route and uses no teacher embedding or distillation
target.  It trains the shared DreaMS encoder directly from clean, real action,
matched-control and candidate spectra.

1. **Strict corrective admission.** Only frozen outer-train rows with clean
   rank not equal to one, action rank equal to one and action margin above
   `5e-6` carry the corrective gradient.  Harmful and robust rows remain in
   their separate safety branches.
2. **Admission precedes every control transform.** The 32,114 corrective rows
   are combined with all 85,959 harmful and 179,289 robust rows, giving one
   297,362-row admitted table.  Action and paired-control tensors are subset by
   the identical original index vector and then continuously reindexed.  Only
   after this operation may the shuffled arm choose donors.  Therefore a
   rejected corrective row, held row or query-limited row cannot become a
   shuffled donor.
3. **Query-complete aggregation.** Every selected action for one query is
   encoded in the same `BoundaryExample`; the query receives one aggregated
   optimizer contribution.  Action multiplicity cannot create extra optimizer
   steps.
4. **Candidate-relative transfer.** The existing v3 objective transfers the
   bounded action-versus-clean and action-versus-control improvement on the
   union of clean/action/control hard candidate boundaries.  Targets are
   detached scalars, while clean query and live shared candidate references
   remain trainable.  This preserves the empirically superior E8 shared
   geometry and is direct margin supervision, not embedding imitation.
5. **Registered V3 transfer allocation unchanged.** The allocator remains
   `hard_cap`, exactly as in the signal-tested V3 injector.  The later
   `mass_neutral_monotone` proposal is excluded because it has only local
   tensor evidence and would introduce a second scientific change.
6. **Post-shuffle actual-bank calibration.** Each arm is calibrated on the
   exact action bank it will use in optimization.  Routed and clean therefore
   calibrate on the admitted targeted bank; shuffled calibrates after the
   admitted bank has been shuffled.  Coefficients may differ, but each active
   arm must independently hit the same registered branch and dense
   corrective-to-protect gradient ratios without a scale cap.  Reusing routed
   coefficients on shuffled data is forbidden because it does not equalize
   realized optimizer dose.
7. **Low-loss safety path.** Corrective, robust, harmful and full clean-protect
   gradients remain separate through calibration.  Auxiliary gradients are
   projected so they cannot erase corrective direction, the combined action
   gradient is projected against clean risk, and post-PCGrad/clip plus realized
   AdamW attribution are recorded separately for head and backbone.
   Counterfactual AdamW attribution is sampled uniformly at the registered 16
   steps per epoch and is explicitly labelled non-exhaustive whenever the epoch
   has more steps.  The registered `>=0.10` gates are unchanged: a value exactly
   equal to 0.10 still passes that historical threshold, while the separate
   descriptive field `legacy_90pct_signal_boundary_observed` flags the exact
   90%-loss boundary.  That boundary flag is not evidence that every optimizer
   step is safe, because attribution is explicitly sampled rather than exhaustive.
8. **Full coverage.** Four registered epochs use the complete 65,286-query
   fold-0 outer-train clean graph.  Evaluation uses all 18,333 formula-held
   queries.  There are no development query limits.
9. **Fail-closed forward-memory bound.** A query is never split or truncated.
   Before calibration, the trainer calculates the exact number of spectrum
   views required by every query-complete logical batch and refuses any batch
   above 512 views.  The bound only prevents an OOM-prone forward; it does not
   rescale, repack or change the V3 loss.

The selective query-only stop-gradient idea is deliberately excluded.  The
historical E8 factor experiment found fixed candidate references about
0.2195 pp worse than shared current references, and no held result establishes
that a selective stop-gradient improves this route.

## Causal arms and server execution

The formal job runs one complete seed (`20260908`):

- routed best actions on GPU 0;
- exact-stratum shuffled actions concurrently on GPU 1;
- matched clean continuation on GPU 0 after both action arms finish.

All three arms start from the same E8 checkpoint and use the same query
schedule, clean protection, optimizer budget and seed.  Routed and shuffled
are independently calibrated on their actual post-transform banks to the same
registered realized gradient-ratio targets.
The first action-arm failure terminates the peer immediately.  Outputs are
written to a unique staging directory and moved atomically only after complete
evaluation.

The job requests exactly two GPUs and no manual memory allocation.  Python
tests and preflight run inside the Slurm allocation.  Nothing in this protocol
authorizes Python execution on the login node.

The concurrent-arm fail-fast path is compatible with Bash 4.2 and does not use
`wait -n`.  The source snapshot preserves repository-relative paths; its
manifest remains directly verifiable after the staging directory is moved to
the final atomic output directory.  Every preflight, trainer and summarizer
Python process executes from that immutable snapshot, so a live-checkout edit
during the long job cannot change either executed code or recorded provenance.

Manual submission from the server repository root is exactly:

```bash
sbatch tasks/run_noise_corrected_best_v6_full_2gpu.sbatch
```

## Simultaneous metric and promotion gate

On the identical 18,333-query clean-input held graph, the summary reports:

- Recall@1/2/3/5/10/20;
- MRR and mean/median rank;
- macro-query AUROC/AUPRC;
- micro-candidate AUROC/AUPRC;
- positive-versus-best-negative margin and signed/unsigned Top1--Top2 gap;
- corrected, introduced and risk-net (`corrected - 2 * introduced`);
- the same retrieval, margin and risk outcomes on the near subset;
- multiplicity-corrected formula-cluster paired confidence intervals;
- MassSpecGym all-adduct and `[M+H]+` 10-ppm pooled pairwise AUROC/AUPRC.

The MassSpecGym pooled value is not labelled an exact NIST20 paper-0.85
replication.  The single-seed gate requires at least `+4.0 pp` Recall@1 versus the exact
initial E8 checkpoint, routed superiority over both shuffled and clean arms,
strictly positive paired formula-cluster CI lower bounds, positive overall and
near risk-net, no regression across the registered metric family, and every
signal-transmission/configuration gate passing.  A non-promoted result remains
an immutable result and is never silently reinterpreted as success.

The summarizer itself hard-locks seed `20260908`, threshold `4.0`, 10,000
bootstrap resamples and Bonferroni family size four.  It requires each actual
held CSV to contain 18,333 unique queries, verifies the CSV byte hash and row
provenance, closes graph/panel counts, and recomputes every rank-derived metric,
margin and risk outcome before using the JSON metric panel.  Each arm also writes
a hashed `held_metric_evidence.npz` containing the exact held molecule labels,
10-ppm spectrum-pair labels, `[M+H]+` mask and official/E8/candidate scores.  The
summarizer independently recomputes micro-candidate and both pooled pairwise
AUROC/AUPRC panels from that evidence and refuses a JSON-only metric claim.
Passing this job
sets `single_seed_gate_pass=true` and `promotion_status=pending_multiseed`;
`promote` remains false.  Fold-0 promotion authorization still requires the
separate registered multi-seed decision described by the V3 protocol.
