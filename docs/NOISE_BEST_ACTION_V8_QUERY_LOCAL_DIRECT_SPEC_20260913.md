# Noise best-action V8 query-local direct canary

Date: 2026-09-13  
Status: bounded four-arm canary implementation; no formal full-run authority.

## Immutable scientific boundary

V8 retains the mature E4/E8 shared encoder, the exact V7 seven-source action
ledger, strict Top-1 admission, all multi-action query blocks, shared clean
continuation, shared protective/robust/harmful objectives, candidate graph,
initialization, optimizer schedule and complete clean-input evaluator.  It does
not create a teacher embedding or use distillation.

The sole semantic change is the action-specific corrective residual.  V7 lets
that residual update clean/action views and positive/negative candidate
references.  V8 keeps identical forward scores but detaches candidate
references only inside that residual, forcing its parameter gradient through
the clean query and real action views.  References remain fully trainable in
the E4/E8 continuation and safety objectives.

## Why this is the next causal test

Job 2336335 already records the missing role attribution.  In the routed arm:

- transfer embedding-gradient norm: clean query `0.244449`, references
  `0.380897`; the reported median role fractions imply approximately 27.5%
  versus 70.6% squared energy;
- payload embedding-gradient fraction: action `0.262603`, references
  `0.961531`; squared energy is approximately 6.9% versus 92.5%;
- routed action semantics are active, but held Recall@1 is 0.060001 pp below
  the exact shuffled arm.

Thus V7 primarily writes action-specific rank pressure through the candidate
reference side.  Its pairwise AUROC can improve while query-local Top-1 barely
changes.  The existing V7 implementation already measures these role norms;
V8 promotes them to an explicit locality gate instead of inventing a new
action family.

## Four matched arms

All arms use one epoch, the same seed, 512 corrective queries, 2,048 robust
queries, 1,024 harmful queries, at most 4,096 clean queries, and the same 4,096
outer-formula-held clean queries.

1. query-local routed: true best actions, corrective references detached;
2. query-local shuffled: exact source/family/recipe matched shuffle with the
   same locality and dose;
3. shared routed: V7 reference-live corrective residual;
4. clean control: matched continuation without corrective action updates.

The first pair identifies action semantics, the first versus third identifies
gradient locality, and the first versus fourth identifies gain beyond generic
continuation.  The optimizer restoration remains identical so this canary does
not confound semantic localization with another optimizer rewrite.

## Independent gates

The summary reports four independent decisions:

- role gate: corrective reference gradients are exactly zero while clean and
  action paths remain live;
- semantic gate: query-local routed beats query-local shuffled on paired held
  Recall@1 and lambda-2 risk-net;
- locality gate: query-local routed beats both shared routed and clean control;
- optimizer gate: the existing end-to-end signal gate passes.

Only all four together permit development of a full candidate.  This canary
cannot establish or guarantee a 4 pp result.  If locality and semantic gates
pass while the optimizer gate fails, the next change is restricted to the
optimizer boundary.  If role locality passes but routed does not beat shuffled,
the admitted action directions lack transferable clean-query causality and
must not be amplified.
