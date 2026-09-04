# Noise final CPG: counterfactual residual teacher and peak-gated shared encoder

Date: 2026-09-03  
Status: implementation contract; supersedes the scalar-action objective used by L2

## Objective

Train one candidate-independent, shared clean-spectrum DreaMS encoder. Candidate
graphs and action outcomes are training-only supervision. Inference receives one
ordinary spectrum and emits one embedding. P2b, ChemAware and P3 are outside this
experiment.

## Architecture A: mature-action candidate-residual teacher

The teacher replays the complete fixed mature grid under one frozen mature
shared encoder:

- N arm: candidate-gradient attenuation 0.50 steps 3-6 and role-confounder
  attenuation 1.00 steps 1-5, each against its two frozen matched controls;
- P intensity arm: matched-intensity transport, prevalence attenuation and
  consensus projection at doses 0.25, 0.50, 0.75 and 1.00, each against the
  same operation driven by the hardest-wrong reference;
- P recurrent arm: recurrent peak graft, balanced peak exchange and recurrent
  union mix at doses 0.10, 0.25 and 0.50, again with the hardest-wrong
  reference as direction control.

For every action and every negative candidate molecule, store

`[(target positive - target candidate) - mean(control positive - control candidate)]`.

This ragged vector is the supervision. A single positive-vs-hardest-negative
scalar is retained only as an audit field. No cell is removed using held-formula
outcomes. Ineffective payloads are omitted and counted. Signed/harmful residuals
remain in the ledger and cannot be relabelled as corrective examples.

## Architecture B: counterfactual peak-gated shared encoder

The student contains the official DreaMS backbone and projection head plus a
candidate-independent contextual peak gate. At initialization every valid peak
has gate value one, so the emitted embedding is exactly the initialization.
The gate can only reweight peaks already present in the clean spectrum; it does
not hallucinate missing peaks. Real recurrently observed missing-peak actions
are training augmentations and residual teachers, not inference inputs.

The primary transfer loss is defined on the clean query and the full molecule
candidate list. It fits the change in each clean positive-vs-candidate margin
relative to the frozen initialization to a bounded, cross-fitted teacher
residual. It is not an action-embedding cosine loss and is not an action-view
ranking loss in disguise.

Separate losses are required for:

1. clean full-list ranking;
2. signed candidate-residual transfer;
3. no-op / harmful-action protection;
4. acquisition-view consistency;
5. initialization preservation.

The matched-random teacher is a frozen numerical counterfactual. It is never a
second trainable branch whose score can be deliberately degraded.

## Sampling and optimization contract

- split unit: formula; identity equalization is applied inside outer-train;
- sampling hierarchy: identity, query, mechanism, action; no replacement within
  an epoch and no action recycling;
- all mature steps remain available; repeated queries do not gain unbounded
  weight;
- branch gradient norms are calibrated on at least 32 stratified microbatches;
- auxiliary gradients that oppose clean/safety gradients are projected before
  summation;
- train gate/head first, then compare unfreezing the final one and two blocks;
- learning-rate search is downstream of target replay and non-zero-gradient
  gates, never a substitute for them.

## Required audits after each architecture

Architecture A must pass source/provenance checks, exact action/control replay,
ragged-pointer integrity, all-candidate coverage, signed-residual arithmetic,
formula-fold isolation, and bounded GPU batching.

Architecture B must pass exact zero-initialization reproduction, non-zero peak
gate gradient, candidate/reference gradient flow, no missing-peak fabrication,
no-replacement exposure, branch-gradient accounting, and a tiny overfit test
before any formula-held pilot.

## Causal evaluation

Every pilot has same-initialization arms: clean continuation, matched-random
residual, N-only, P-only, N+P, and N+P+acquisition. The primary endpoint is
clean full-list formula-held retrieval. Required comparisons report Recall@1,
MRR, near Recall@1, corrected, introduced, risk-net, candidate switches and
initialization preservation. No action headroom is reported as learned embedding
gain.

