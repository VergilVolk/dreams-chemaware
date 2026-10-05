# BioAware B47-M1 truth-blind spectral seed contract

## Scope

B47-M1 converts the frozen official DreaMS query/reference embeddings into a
candidate score table and a sample-local seed table. It does not open sealed
annotation truth, evaluate accuracy, fit BioAware, use phenotype labels, or use
P2b. Its sole purpose is to freeze the biological context that will later be
available to every candidate of a query.

## Candidate score

For query spectrum `q` and candidate IK14 `c`:

\[
s(q,c)=\max_{r\in R(c)}\cos(z_q,z_r).
\]

Both `z_q` and `z_r` come from the same frozen official DreaMS encoder. The
candidate set remains the already frozen 10 ppm positive-mode graph. A tied
Top-1 is adverse and is never accepted as a seed.

## Why a single high-scoring spectrum is insufficient

A single Top-1 can be confidently wrong and would propagate that error to many
other features. The primary seed therefore requires both an absolute spectral
gate and agreement of the same consensus feature across independent samples.

Primary policy, fixed before truth is opened:

- unique Top-1;
- Top-1 cosine at least `0.80`;
- Top1-Top2 margin at least `0.05`;
- the same study/feature has at least two sample spectra supporting one modal
  candidate identity;
- the modal identity accounts for at least `80%` of all MS2-observed sample
  events for that feature;
- at least two modal events independently pass the absolute score/margin gate;
- the identity is non-currency, occurs in Rhea, and has reaction degree at most
  `250`.

A fixed strict sensitivity policy uses score `0.90`, margin `0.10`, at least
three supporting samples and `90%` modal agreement. It is not selected after
looking at truth and cannot replace the primary policy merely because its
eventual performance is better.

## Independent support and ion forms

Within one sample, multiple features resolving to the same candidate identity
do not count as independent evidence. Only the event with the strongest score,
then margin, is retained for each `(study, sample, candidate identity)`. This
conservatively collapses duplicated adduct/isotope/in-source manifestations at
the identity-support level. B47-M2 must still retain explicit feature IDs so an
ion-family graph can identify and report the physical forms that were merged.

## Leakage rule for the sealed evaluator

The original sample seed set is frozen without truth. When truth is eventually
opened by the separate evaluator, a query is eligible only when its truth
identity was absent from that original same-sample seed set. It is not enough
to delete the truth seed after seeing it. This restriction changes the
evaluation denominator and must be reported explicitly.

## Required outputs before B47-M2

- one score row per query/candidate IK14;
- one query-level Top-1/margin record;
- one cross-sample feature-consensus record;
- one primary and one strict-sensitivity seed table;
- exact input/output hashes;
- at least 500 primary seed rows, 200 seed identities, and 100 primary seed
  rows from each external source before constructing reaction events.

These are context-readiness gates, not evidence of accuracy or BioAware gain.
