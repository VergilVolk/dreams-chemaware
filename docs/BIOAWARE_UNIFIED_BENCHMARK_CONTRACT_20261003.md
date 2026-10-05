# BioAware unified benchmark contract

Date: 2026-10-03  
Status: preregistered benchmark design; no new outcome has been opened

## Decision

BioAware now has two scientifically distinct assets and they must not be
reported as one result:

1. **B30/B35 candidate-risk expert.**  This is an opened, cross-fitted
   engineering result.  It improves complete real-library ranking but its
   dominant mechanism is spectral boundary plus catalogue familiarity and a
   cross-source sink veto.  It is not evidence that exact biochemical
   reactions caused the gain.
2. **B47 exact-event branch.**  This is a prospective, truth-blind sample-local
   event construction.  U3-v3 produced more real candidate-specific events
   than its degree-rewired and seed-context-permuted structural nulls, but the
   evidence was concentrated and the frozen authorization gate did not permit
   opening ranking truth.

The next BioAware experiment is therefore a benchmark programme, not another
uncontrolled model fit.  It has three tracks with explicit information
budgets.

## Track A: spectrum-reference retrieval

This track uses the same query spectra, reference library, candidate molecules,
reference-spectrum aggregation and strict tie policy for every method.

Mandatory methods:

- modified cosine;
- spectral entropy;
- Spec2Vec;
- MS2DeepScore 2.0;
- official fine-tuned DreaMS;
- any new shared encoder, reported as a separate contender.

The local 83,619-query corrected graph is development-only.  Public benchmark
claims require the official MassSpecGym v1.5 test split/candidate resources or
an independently sealed external library panel.

## Track B: spectrum-structure retrieval

This track permits candidate molecular structure as an explicit input and is
not merged with Track A without an information-budget label.  Mandatory
reproducible methods are MIST, JESTR, FLARE and ICEBERG 2.0.  Released frozen
weights and models retrained on an equal training denominator are two separate
leaderboards.

## Track C: truth-blind sample context

This is the only track that can establish the BioAware biological claim.
Every arm receives identical query spectra and candidate sets.  The minimum
comparison is:

1. frozen unary ranking;
2. static catalogue membership/degree only;
3. a MetDNA3/KGMN-compatible network arm;
4. BioAware candidate-specific exact events;
5. degree/component rewires, intact seed-context permutations and matched
   non-neighbour controls.

Context-absent queries remain in the full denominator and force abstention.
Eligible-only results are secondary and must report coverage.

The existing B47 U3-v3 `pass_to_frozen_event_ranking_evaluation=false` is
binding.  No B47 outcome may be opened until a prospectively documented repair
addresses all failed concentration and identity-quality gates without using
truth or performance outcomes.

## Canonical candidate-ranking schema

Candidate manifest, one row per query/candidate:

- `query_id`;
- `candidate_id`;
- `is_truth` (exactly one per query);
- `formula_cluster`;
- `source`;
- `polarity`;
- optional `near_query` and other frozen strata.

Each method produces exactly:

- `query_id`;
- `candidate_id`;
- `score`.

Missing candidates, extra candidates, duplicate keys and non-finite scores are
fatal.  A method may abstain only by returning the frozen unary ordering, never
by dropping a query.

## Metrics

Primary candidate-ranking metrics:

- Recall@1/2/5/10/20;
- MRR and median rank;
- nDCG@5/10/20;
- macro-query AUROC;
- truth-versus-best-negative margin;
- corrected, introduced, net correction and `corrected - 2*introduced`;
- exact McNemar p-value;
- paired formula-cluster and source-cluster confidence intervals;
- context coverage, intervention rate and risk-coverage curve.

The DreaMS-paper-style 10-ppm pooled spectrum-pair AUROC is a separate task. It
must never be labelled macro-query AUROC or candidate Recall@k.  Unless the
licensed NIST20 data, MoNA-disjoint identities and original pair construction
are reproduced, the result is called `MassSpecGym 10-ppm pooled pairwise
AUROC`, not an exact reproduction of the paper's approximately 0.85 value.

## Integrated BioAware claim gates

A BioAware sample-context claim requires all of:

- untouched external or outer-fold Recall@1 gain at least +3.0 percentage
  points;
- formula- and source-cluster CI lower bounds above zero;
- corrected greater than twice introduced;
- at least 50 corrected identities;
- nonnegative point estimate in every major source;
- superiority to catalogue/degree-only and every structural null;
- no degradation in near-isomer and context-unmapped strata;
- complete per-query ledger, frozen method versions and SHA256 provenance.

The +3 pp rule is a decision gate, not a promised outcome.

## Immediate execution order

1. Use `tasks/bioaware_unified_benchmark_core.py` as the sole metric engine.
2. Freeze MassSpecGym v1.5 formula and mass candidate manifests separately.
3. Export official DreaMS and classical spectral baselines first; their exact
   replay is the implementation gate for every later method.
4. Add structure-aware methods only after candidate-key identity is proven.
5. Keep B47 sealed until its failed U3 concentration/identity gates have a
   truth-blind, preregistered remedy.
6. Run B30/B35 only as opened-development engineering baselines and retain the
   B44 transfer reversal in the main limitation table.

## Claim limit

This contract and metric implementation do not establish BioAware gain, shared
embedding gain or SOTA.  They define the first common ruler capable of testing
those claims without mixing tasks or denominators.
