# Noise best-action V10 hard-safe exact-dose direct specification

Date: 2026-09-13  
Status: implemented bounded canary; not yet a performance result

## Objective and frozen scientific route

The only objective remains a better shared clean-spectrum DreaMS encoder. V10
is direct fine-tuning from the existing E8 initialization. It introduces no
teacher embedding, distillation target, new action bank, peak gate, or changed
downstream task.

The action bank remains the frozen union of `A4_exact`, `E10B`, `E11`, `E12B`,
`N_mature`, `P_guided_original`, and `V4_gradient_path`. Strict current-E8
Top-1 corrective admission, mass-neutral monotone transfer, query/action-only
corrective locality, E4/E8 protective continuation, robust/harmful branches,
one-block/head capacity, learning rates, and candidate construction remain.

V9 established that the full action branch retained 99.9868% of its gradient
through projection/clipping and delivered approximately 25% optimizer-space
action fraction, yet its formula-held Recall@1 was 0.0732 pp below clean
continuation. The missing causal boundaries were:

1. the configured 0.90 protective floor was audited but not enforced; and
2. full/scalar arms matched pre-optimizer corrective:risk norms, not their
   realized AdamW corrective dose.

## V10 optimizer composition

This optimizer-boundary mechanism is now frozen behind
`tasks/noise_action_injector_v1.py`; its versioned interface and anti-drift
rules are specified in `NOISE_ACTION_INJECTOR_V1_SPEC_20260913.md`. The
mathematics below remains the golden V10 definition.

For each head/backbone parameter group and action-active optimizer step, let:

- `U` be the ordinary combined AdamW descent update;
- `B` be the noncorrective `protective + projected robust/harmful` AdamW
  counterfactual;
- `P` be the independent protective-only AdamW update; and
- `C = U - B` be the observed corrective AdamW residual.

V10 performs the following bounded parameter-space composition:

1. If `<B,P>/||P||^2 < 0.90`, add only the minimum multiple of `P` needed to
   put the noncorrective counterfactual inside the 0.90 protective half-space.
2. Remove only a negative projection of `C` onto `P`. The surviving corrective
   direction therefore cannot consume the guaranteed protective component.
3. Solve analytically for the positive coefficient `a` satisfying
   `||aC|| / ||B + aC|| = 0.25` after materialization.
4. Use the safety-projected `B` as the attribution counterfactual, so the
   protective repair is never counted as action signal.
5. Fail the worker before applying the update unless both head and backbone
   attain the 0.25 fraction, retain at least 0.90 protective component, and
   remain below a 1.50 update-norm ratio relative to ordinary combined AdamW.
6. Materialize the composed update once and reconcile AdamW's first moment to
   the realized displacement while leaving its real combined-gradient second
   moment and step number intact.

Unlike V7-V9, an unsafe gain-one update cannot be returned unchanged. Zero
corrective residual, failed exact dose, failed protective floor, excessive norm
growth, or failed first-moment reconstruction is a fail-closed runtime error.

## Common calibration and causal arms

All four arms calibrate once on the same targeted action bank with the complete
`full_action_view` policy. The resulting branch coefficients, corrective
scale, global gradient scale, and calibration tensor hash must be identical.
Only after calibration does the scalar arm set payload and consistency scales
to exactly zero. The exact optimizer composer then gives each active arm the
same realized 0.25 corrective fraction.

The bounded job trains four arms in two waves on exactly two allocated GPUs:

1. `full_action_view`: scalar transfer + payload + live consistency;
2. `scalar_transfer_only`: scalar transfer, payload/consistency zero;
3. `matched_shuffled`: full policy with source/family/exact-recipe matched
   shuffled action tensors; and
4. `clean_control`: the same continuation/protective schedule without action
   updates.

## Formula-disjoint development boundary

Before action admission, the outer-training action ledger is split by a stable
formula hash into five folds. Fold 0 is never used for calibration, action
training, shuffle donors, robust/harmful training, or protective query
selection. Its clean queries form `inner_held_corrective`.

The canary uses at most 512 training corrective queries, 1,024 risk queries,
2,048 robust queries, 4,096 clean-continuation queries, 4,096 outer-formula-held
queries, and one epoch. The training-corrective panel remains diagnostic only.
An apparent 4-5 pp gain there cannot authorize promotion unless it transfers to
the formula-disjoint inner panel and the outer held graph.

## Required evaluation and advancement gate

Every arm reports Recall@1/2/3/5/10/20, MRR, mean/median rank, macro-query
AUROC/AUPRC, micro-candidate AUROC/AUPRC, positive-vs-best-negative margin,
Top1-Top2 gaps, corrected/introduced/risk-net, near-subset metrics, formula-
cluster confidence intervals, and MassSpecGym all-adduct and `[M+H]+` 10-ppm
pooled pairwise AUROC/AUPRC. The pooled metric is not an exact NIST20-paper
replication.

A larger pilot is allowed only when all conditions are true:

- calibration signatures are byte/number identical across all arms;
- full, scalar, and shuffled attain 0.25 realized optimizer fraction within
  `2e-6` on every parameter group;
- all action-active steps enforce the 0.90 protective floor and pass every
  signal/reconciliation/norm gate;
- full beats scalar and clean on the formula-disjoint inner panel with positive
  risk-net and the registered strictly positive formula-cluster lower bounds;
- the same causal direction holds on the outer formula-held graph;
- full beats matched shuffle; and
- every required outer metric improves over clean without a hidden regression.

The bounded result is not itself a 4-5 pp claim and cannot promote a shared
encoder. Its purpose is to prove or reject actual cross-formula action-to-clean
transfer after eliminating the two remaining injection confounders.

## Server entrypoint

Run only through Slurm:

```bash
sbatch tasks/run_noise_corrected_v10_safe_exact_canary_2gpu.sbatch
```

Do not execute any Python preflight or summarizer on the login node. The job
requests exactly two GPUs and intentionally contains no manual memory request.

## Job 2336553 storage-failure recovery

Job 2336553 completed both wave-1 training/evaluation passes, but the
`full_action_view` worker encountered a filesystem write failure while saving
`final_shared_encoder.pt`.  The independently completed
`scalar_transfer_only` directory and its digest-verified checkpoint remain
valid; the original fail-fast prevented both wave-2 arms from starting.

After storage capacity/quota has been cleared, resume only through:

```bash
sbatch tasks/run_noise_corrected_v10_safe_exact_resume_2336553_2gpu.sbatch
```

The recovery job verifies the original frozen source snapshot and the scalar
checkpoint digest, refuses invalid or overwritten arm directories, checks
filesystem headroom, runs only the three missing arms, and then performs the
registered four-arm summary. It retains the original job-2336553 result root
and never retrains the completed scalar arm.

The first recovery submission completed all three missing arms but exposed a
shell-control defect: `run_arm_async` uses `exec`, while the final clean arm was
called in the foreground.  The clean Python process therefore replaced the
batch shell; after it exited successfully, no summary or final directory move
could execute, and Slurm correctly produced an empty stderr file.  All four
atomic arm directories remain valid in the hidden staging root.  Do not
retrain them.  Finish only with:

```bash
sbatch tasks/run_noise_corrected_v10_summary_recovery_2336553.sbatch
```

This one-GPU Slurm job performs no model training. It digest-verifies the four
checkpoints, runs the frozen 10,000-resample summarizer, and atomically promotes
the original hidden staging root to the visible job-2336553 result directory.
