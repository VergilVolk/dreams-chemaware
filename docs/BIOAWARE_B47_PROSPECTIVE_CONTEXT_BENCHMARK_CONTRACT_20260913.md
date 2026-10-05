# BioAware B47 prospective sample-context benchmark contract

## Why B47 is necessary

B44, B45 and B46 close three shortcuts:

1. Static KEGG/Rhea catalogue coverage produced a large opened-development
   gain, but failed on MassBank when the truth-versus-wrong coverage imbalance
   reversed.
2. After forcing candidates to have exactly the same catalogue signature, real
   topology added only `+0.2326 pp` and did not beat degree-preserving
   permutations.
3. A fixed conjunction of reaction-transform and coabundance summaries reduced
   Recall@1 by `0.7714 pp` and was worse than its matched-random coabundance
   control.

The current six context sources also do not provide a prospective unknown
annotation task. Four are synthetic held-identity rotations, one is a standards
repeat, and the sample-local leave-one-seed-out source previously contained
every evaluation identity in its seed table. The current ledgers contain zero
event-specific spectral rows and zero event-specific coabundance rows.

Therefore, another model fit on these ledgers cannot answer the BioAware
question. B47 first constructs a benchmark in which the biological context is
observable without access to the query truth.

## Primary scientific question

For a query feature whose true identity was not used as a seed, does exact
sample-local evidence on a candidate--seed--reaction event improve candidate
ranking beyond:

- the frozen DreaMS spectral score;
- static database membership and degree;
- formula, mass, structure and candidate-count effects;
- matched randomized networks and seed sets?

Only a positive answer licenses a BioAware reranker or context-conditioned
embedding experiment.

## Candidate public benchmark

The primary acquisition target is the 2026 MSMICA head-to-head/external
validation deposit (`10.5281/zenodo.21574649`). It is attractive because the
deposit reports inputs and outputs for MetDNA2, MetDNA3, MSMICA and
xMSannotator on the same CHDWB HILIC+ data, plus external validation material.

This is a candidate, not an automatic benchmark. B47-M0 must establish that the
archive exposes enough raw or processed information to reconstruct every
context edge without reading validation truth. If the archive only contains
method outputs that already encode truth or post-selected annotations, B47-M0
fails and no model is fitted.

MetDNA3 NIST urine data already used by this project remain development-only
and cannot be relabelled as an independent B47 test.

## B47-M0: acquisition and immutable sealing

Before any score is computed:

1. Download the public archives and verify the deposited MD5/SHA256 values.
2. Inventory every file without merging algorithm outputs into experimental
   inputs.
3. Separate three namespaces:
   - `observable_input`: MS1 feature table, MS/MS, RT, sample metadata and
     acquisition polarity;
   - `context_construction`: seed library matches and reaction database;
   - `sealed_truth`: standard-backed or otherwise independent validation
     identities.
4. Write immutable file hashes and an identity/formula/source overlap report.
5. Deny the modeling process read access to `sealed_truth`; evaluation runs in
   a separate one-time process.

No phenotype, case/control label or downstream biological endpoint may enter
candidate scoring.

## B47-M1: candidate and seed construction

For every query:

- construct the same formula/adduct/mass-tolerance candidate set for every arm;
- compute frozen DreaMS unary scores once;
- derive high-confidence seeds without using the query truth;
- remove the query feature and every spectrum or ion-family member of its truth
  identity from the seed context;
- require the truth identity to have been absent from the original seed set,
  not merely deleted after selection;
- collapse isotope, adduct and in-source-fragment forms before counting
  independent seed support.

Required M1 gates:

- at least 1,000 evaluable queries;
- at least 200 independent truth formulas;
- at least 200 frozen-DreaMS Top-1 errors;
- at least 100 reaction-reachable frozen-DreaMS errors;
- zero truth-identity seed leakage;
- all candidate arms have identical candidate sets and tie handling.

## B47-M2: exact event evidence

The atomic unit is not a candidate-level average. It is

`query feature -- candidate identity -- seed feature -- seed identity -- reaction`.

Each event stores independently auditable evidence:

- reaction source, identifier, direction and participant side;
- candidate/seed mass transform and its residual;
- MS/MS relation conditioned on the expected transformation;
- RT and CCS compatibility when available;
- sample-level abundance evidence with shrinkage for small sample counts;
- ion-family reconciliation;
- network degree and catalogue membership as nuisance variables, never as an
  unqualified positive vote.

Positive coabundance is not assumed universally. Directional substrate/product
relations, enzyme state and study design determine whether positive, negative
or absent association is expected. If direction cannot be justified, the
abundance term is marked unknown rather than forced positive.

## B47-M3: factor-graph action and controls

The first deployable action is a conservative factor graph, not another scalar
sum of weak features:

\[
  S(c)=S_{DreaMS}(c)+\sum_e \psi_{event}(c,e)-\psi_{conflict}(c).
\]

Candidate support is aggregated only across explicit event IDs. Competing
candidates share the same seed context. Contradictory edges reduce confidence,
and the system abstains when calibrated utility is non-positive.

Every real-network result is compared with:

- static catalogue membership/degree only;
- degree- and component-preserving reaction-edge rewires;
- within-sample seed permutations;
- wrong-sign or wrong-mass-transform events;
- matched non-neighbour events;
- spectral-only DreaMS.

Model selection is nested by source/study and truth formula. No query, truth
identity, truth formula group, seed derived from it, or validation outcome may
cross from an outer test fold into training.

## B47-M4: decision gates

The context action passes only when all of the following hold on untouched
outer folds or the one-time external evaluation:

- Recall@1 gain is at least `+3.0 pp`;
- formula-cluster and study/source-cluster confidence-interval lower bounds are
  greater than zero;
- `corrected > 2 * introduced`;
- at least 50 distinct truth identities are corrected;
- every major source has nonnegative point-estimate gain;
- real event evidence beats each structural null with a positive clustered CI;
- catalogue coverage/degree alone does not explain the gain;
- near-isomer and unmapped-truth strata do not degrade.

The `+3 pp` threshold is a gate, not a promised outcome.

## Embedding boundary

BioAware context is candidate- and sample-specific; a clean spectrum alone does
not contain that information. Consequently, context evidence may enter a
shared clean-spectrum encoder only if an additional spectrum-only
recoverability test shows that the action leaves a reproducible signal in the
spectrum. Otherwise, the scientifically correct deployment is a context expert
after the shared DreaMS embedding, not forced distillation into the encoder.

If the action passes M4, the first embedding experiment must compare:

1. direct candidate-group ranking with shared query/reference encoder;
2. the same loss with context removed;
3. the same dose with rewired/seed-permuted context;
4. frozen DreaMS preservation and near-isomer safety.

No embedding claim is allowed unless the trained encoder improves clean,
candidate-independent retrieval on an identity- and formula-isolated panel.

## Relation to prior methods

MetDNA2/KGMN uses a knowledge reaction network, an MS2 similarity network and a
global peak-correlation network. MetDNA3 further pre-maps experimental features
onto a curated/predicted reaction network and propagates annotations through a
two-layer topology. B47 adopts the central lesson that knowledge topology must
be constrained by experimental data, but adds the controls exposed by our
B44--B46 failures: coverage-neutral comparisons, exact event provenance,
prospective seed exclusion, explicit abstention and structural nulls.

## Claim limit

This contract defines a falsifiable acquisition and evaluation route. It is
not evidence that BioAware improves annotation, changes the DreaMS embedding,
or reaches SOTA.
