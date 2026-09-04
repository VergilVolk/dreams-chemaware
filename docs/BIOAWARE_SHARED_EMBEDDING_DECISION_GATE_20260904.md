# BioAware shared-embedding: evidence-first decision gate

## Decision now

Do **not** start a BioAware shared-embedding training run yet.  The existing
small-cache B0 probes do not show that frozen DreaMS embedding contains a
reaction-specific signal beyond chemical metadata.  Their strictest result had
only 17 matched groups; the conditional incremental AUC was negative and the
paired groupwise result was 1 corrected versus 3 introduced.  More importantly,
the current decoy construction had large chemical imbalance (Tanimoto SMD
1.23).  A positive result under that imbalance would not establish a biological
signal.

This is not a rejection of biological context.  It is a rejection of the
unsafe shortcut: treating a Rhea neighbour as a spectral positive or forcing
reaction-neighbour cosine to increase.  Reaction-neighbour molecules are
different identities and must remain different in the retrieval space.

## What is genuinely novel enough to test

The target is **reaction-aware but identity-preserving shared spectral
representation learning**:

1. one clean-spectrum encoder is shared by query and reference at inference;
2. same-identity cross-condition spectra are the only retrieval positives;
3. same-formula / MCES-near candidates remain hard negatives;
4. Rhea relation type is an auxiliary *decodability* task, never an embedding
   attraction target;
5. an auxiliary reaction gradient reaches the encoder only when it is
   compatible with the primary retrieval gradient and bounded by preservation;
6. no phenotype, sample-group outcome, P2b score, candidate truth, or network
   path is available at inference.

The innovation is therefore not generic network propagation (MetDNA and KGMN
already do that).  It is a falsifiable constraint: biochemical relation may
shape directions readable from the common spectral embedding **only when it
does not sacrifice identity discrimination**, particularly the MCES-near
margin.  A relation head may use pair features `abs(z_a-z_b)` and `z_a*z_b`,
but reaction neighbours are never labelled as same-molecule pairs.

## The non-negotiable experiment chain

### B0: frozen-representation reaction-signal test

Run the formal B0 only after the following hard conditions are met:

- P3-disjoint official DreaMS embeddings and a frozen Rhea participant cache;
- formula-community outer folds, with all cross-fold reaction edges omitted;
- degree-preserving, chemical-property-matched non-edge controls;
- every mass delta, heavy-atom delta, Tanimoto, target-degree and
  same-formula imbalance has SMD <= 0.10;
- conditional test: embedding-plus-metadata must outperform metadata-only by
  at least 0.02 AUC, survive within-candidate-group embedding permutation,
  and improve groupwise Top-1 with a positive bootstrap lower bound.

If chemical balance fails, the output is a **matching failure**, not a negative
statement about biology.  Repair the control construction before interpreting
the encoder.  If balance passes but the conditional test fails, stop B1:
there is no evidence that current DreaMS geometry carries usable reaction
information.

### B1: shared embedding pilot (only if B0 passes)

Use a zero-initialized bounded token adapter and a frozen official DreaMS
initialization.  Optimize, in priority order:

\[
L = L_{\mathrm{same>near/control}}
  + \lambda_{\mathrm{safe}}L_{\mathrm{correct-margin}}
  + \lambda_{\mathrm{pres}}L_{\mathrm{preserve}}
  + \lambda_{\mathrm{rel}}L_{\mathrm{typed-relation}}.
\]

`L_typed-relation` classifies relation types; it never minimizes reaction-pair
cosine.  Project or discard its encoder gradient when its cosine with the
primary retrieval gradient is non-positive.  Its post-projection norm must be
at most 25% of the main retrieval-gradient norm.  The relation head can still
train during warm-up, but an uninformative or conflicting reaction loss cannot
move the encoder.

Pilot gates, all on untouched outer-formula rows:

- primary Recall@1 and MRR non-negative;
- MCES-near Recall@1 and hard-negative margin non-negative;
- corrected > introduced;
- preservation >= 0.995;
- post-projection relation gradient is non-zero but capped;
- B0 relation readout remains better than its chemical metadata control.

### B2: formula-OOF shared-embedding result

Five outer formula folds, then three seeds only after formula-OOF passes.  The
P3 locked panel is not used to choose architecture, loss weights, seed,
epochs, gate, or relation vocabulary.  Final evaluation must report overall,
near, cross-instrument, formula/scaffold/dataset isolation, corrected and
introduced, formula-cluster confidence intervals, and the no-context
official-DreaMS fallback.

### Separate B-context branch

Sample-context candidate embeddings, reaction-factor graphs, MetDNA-style
propagation, KGMN-style peak correlation, and NetID-like global consistency
may be useful but are **not** shared embeddings.  They belong to a separately
named contextual candidate expert and must have target-decoy/FDR and
leave-sample/study-out evaluation.  Its result may not be substituted for B1
or B2.

## Why this addresses the state of the art

- MetDNA3 and KGMN establish that metabolic-reaction and data-layer networks
  can expand annotations, but their propagation evidence is contextual and
  cannot be used as an identity label for a universal spectral encoder.
- JESTR demonstrates that candidate-aware hard negatives belong in the core
  representation objective, rather than being appended only at inference.
- METASPACE-ML demonstrates the value of context-stratified, target-decoy
  evaluation and explicitly reports the conditions in which contextual evidence
  is reliable; this motivates our strict balance, fallback and risk reporting.

## Immediate action

The B0 script and server sbatch now treat decoy balance as a pass/fail gate
(`--maximum-standardised-imbalance 0.10`).  The next permissible server action
is the B0 formal probe, not an adapter training run:

```bash
sbatch tasks/run_bioaware_b0_reaction_embedding_signal.sbatch
```

Interpretation is precommitted: B1 proceeds only if `report.json` has
`pass_to_b1: true`.  A failure identifies whether to improve control matching
or to stop the shared-reaction-embedding claim; it must not be bypassed by
loosening a threshold after observing the result.
