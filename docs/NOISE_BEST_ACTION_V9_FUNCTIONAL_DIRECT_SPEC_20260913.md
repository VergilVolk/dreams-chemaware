# Noise V9 Functional Direct-Transfer Specification (2026-09-13)

## Decision being tested

V8 removed the corrective reference-gradient bypass, transmitted essentially
the requested optimizer action fraction, and still did not beat clean
continuation on held clean queries. Therefore V9 does not invent a teacher, a
new action bank, or a new scientific task. It tests the unresolved direct-
finetuning bottleneck inside the existing best-action injector:

> Does the live action-view gradient add transferable clean-query information
> beyond the scalar candidate-margin transfer when total corrective gradient
> budget, data, schedule, initialization, optimizer restoration, and risk
> arbitration are held fixed?

The best-action bank remains the frozen union of A4_exact, E10B, E11, E12B,
N_mature, P_guided_original, and V4_gradient_path with strict current-geometry
Top-1 admission. This experiment does not distill a teacher.

## Two newly trained causal arms

Both arms use the same 512 corrective queries, 1,024 risk queries, 2,048 robust
queries, 4,096 clean-continuation queries, 4,096 outer-formula-held evaluation
queries, one epoch, E8 initialization, query-action-only corrective locality,
and corrective-only optimizer restoration.

1. `full_action_view`: scalar margin transfer plus payload rank/safety and
   symmetric live clean/action consistency.
2. `scalar_transfer_only`: the same scalar margin transfer, while payload and
   live consistency have exactly zero effective branch scale.

Calibration is performed after the branch mask. Each arm's combined corrective
gradient is normalized to the same 1:1 corrective-to-risk gradient ratio. This
prevents a larger raw full-branch norm from masquerading as semantic value.

## Honest injection gates

- Inclusive minimum gates use only a tiny numerical tolerance. A measured
  0.24999998 satisfies a target of 0.25; a material 0.249 does not.
- Optimizer restoration coverage is gated whenever restoration was actually
  materialized. It cannot be bypassed by naming the run `v3` instead of a V7
  contract.
- The signal report must distinguish an actual <=10% retained signal from a
  floating-point comparison artifact.
- Payload and consistency scales must be exactly zero in the scalar arm and
  strictly positive in the full arm.

## Frozen V8 controls and exact bridge

To avoid spending another full two-phase job on controls that V8 already
computed, V9 reruns `full_action_view` and compares it to the frozen V8
query-local routed arm. V8 shuffled and clean controls are admitted only if:

- all 4,096 held query ranks are identical;
- all 512 train-corrective clean query ranks are identical; and
- the maximum train-corrective clean margin difference is <=1e-6.

If this bridge fails, V8 controls are suppressed and cannot support any causal
claim. The new full-versus-scalar comparison remains valid because both arms
are trained from the same V9 source snapshot.

## Required evaluation

For the primary full-versus-scalar comparison, and for bridged comparisons
against shuffled and clean controls, report:

- Recall@1/2/3/5/10/20;
- MRR, mean rank, median rank;
- macro-query AUROC/AUPRC;
- micro-candidate AUROC/AUPRC;
- positive-vs-best-negative margin and Top1-Top2 gap;
- corrected, introduced, and risk-net with lambda=2;
- near-subset metrics;
- paired formula-cluster 95% CI and three-comparison familywise CI;
- MassSpecGym 10-ppm pooled pairwise AUROC/AUPRC, including the [M+H]+ subset.

Recall@1, MRR, macro-query AUROC/AUPRC, micro-candidate AUROC/AUPRC,
margin/gap, near-subset primary metrics, and both pooled pairwise AUROC/AUPRC
must improve strictly. Higher Recall@k and rank summaries may tie but may not
regress in this bounded canary; a later full promotion run must adjudicate any
ties at adequate scale.

The pooled pairwise metric is explicitly not an exact replication of the
paper's NIST20 0.85 result.

The train-corrective clean-query ledger is also compared arm-by-arm. Action
success inside materialized action spectra is not counted as functional
conversion; only improvement of the clean query embedding counts.

## Advancement boundary

This bounded canary is not a 4-5 pp claim and cannot promote an encoder. A
larger direct-finetuning pilot is permitted only if:

- the corrective gradient budgets match;
- full action view beats scalar transfer with positive risk-net and a strictly
  positive formula-cluster 95% lower bound;
- the same direction is present on train-corrective clean queries;
- the frozen V8 bridge passes; and
- all optimizer transmission and restoration-safety gates pass.

If full does not beat scalar, the live action-view branch is not converting. If
scalar beats full, action-view aggregation is destructive. If both beat clean
but full does not beat scalar, scalar transfer—not raw action-view imitation—is
the supported direct route. None of these outcomes authorizes a 4-5 pp claim
without a larger outer-formula-held run.
