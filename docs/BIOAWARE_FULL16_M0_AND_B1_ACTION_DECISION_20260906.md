# BioAware Full16 M0 and B1 action decision (2026-09-06)

## Current evidence boundary

The Full16 support audit independently replayed the strict 10 ppm, same-adduct
candidate protocol.  Among 6,004 Level-1 rows, 2,558 rows (501 identities and
351 formulas) are ambiguity-bearing; 329 identities and 211 formulas have
Rhea-or-KEGG seed support.  Full-source leave-one-out training support is only
104--178 identities per source after identity/formula purging.  This supports a
low-dimensional preregistered action test, not a high-capacity neural model.

The earlier approximately six percentage-point development gain is not yet a
reaction-specific result.  The degree-only/catalog-opportunity control matched
or exceeded the full network model, and the degree/path-conditioned null did
not reject.  It is therefore an engineering comparator, not an embedding
teacher and not a SOTA claim.

## Single next experiment

`develop_bioaware_b1_multisource_action.py` combines all already-computed
positive- and negative-ion candidate evidence (1,426 queries) without spectrum
re-encoding.  It evaluates four nested models under leave-one-biological-source
out validation, purging every held truth identity and formula:

1. DreaMS spectral score only;
2. catalog opportunity (reference count, network membership/degree and mass
   candidate opportunity);
3. reaction-path availability;
4. reaction-path strength plus raw-MS2 bottleneck evidence.

The scientific endpoint is model 4 minus model 2, not model 4 minus DreaMS.
Passing requires at least +3 pp incremental Recall@1, positive identity- and
formula-cluster confidence bounds, more than twice as many corrections as
introductions, at least 25 corrected identities, nonnegative results in at
least three of four sources, and no source below -1 pp.

If this gate passes, the next stage is a degree/reference/path-opportunity
conditioned candidate-assignment falsification followed by a frozen external
test.  If it fails, network degree remains a catalog-popularity prior and the
next BioAware action must use reaction-transform-to-fragment/neutral-loss
concordance; it must not be injected into a shared spectrum embedding.

## Claim boundary

This B1 experiment is opened development.  It does not update the DreaMS
embedding, does not use P2b or phenotype labels, and cannot establish SOTA.

## Local implementation audit result

The complete local replay passed the independent validator on 1,426 queries,
5,384 candidate rows, 361 truth identities and 310 truth formulas.  The
catalog/opportunity control improved gated Recall@1 by 1.753 pp (31 corrected,
6 introduced).  The full reaction-strength model produced the same aggregate
gain, but relative to the opportunity control it corrected 6 queries and
introduced 6, for exactly zero net change.  Formula-cluster and
identity-cluster intervals both crossed zero.  Mouse brain and NIST plasma
were negative in the reaction-specific comparison.

Therefore the available shortest-path/bottleneck evidence is not an eligible
teacher for shared-embedding training.  The approximately 4--6 pp historical
negative-ion result remains a useful high-precision engineering action, but
its dominant transferable signal is catalog opportunity rather than verified
reaction chemistry.  The next scientific BioAware action must encode a
specific reaction transformation and test concordance with candidate-specific
fragment or neutral-loss evidence; more tuning of the current path-score
ranker is not justified.

## B2 preregistered successor

The successor is implemented in
`tasks/audit_bioaware_b2_reaction_transform_action.py`.  It uses only direct
Rhea/KEGG relations and, within each identity-held-out seed rotation, requires
the query and seed spectrum to have the same polarity and exact adduct.  For
each candidate--seed relation it computes the formula-derived precursor shift,
its residual from the observed precursor difference, direct fragment matches,
formula-shifted fragment matches, their one-to-one modified-cosine union, and
two controls: an observed-precursor-shift score and a wrong-sign formula shift.

Four nested source-LOSO recipes are frozen before outcomes are inspected:
catalog opportunity, generic direct-edge spectral evidence, wrong-shift
control, and reaction-transform evidence.  The primary endpoint is reaction
transform minus generic edge.  Passing requires at least +3 pp, positive
identity- and formula-cluster confidence bounds, corrected greater than twice
introduced, at least 25 corrected identities, nonnegative transfer in at least
three biological sources, and superiority to the wrong-shift control.  Failure
forbids use of this action as either a reranker signal or shared-embedding
teacher; success permits a separately frozen external validation, not an
immediate SOTA or embedding claim.

## B2 observed result and stop decision

The complete 1,426-query local B2 run passed its independent implementation
validator, but failed the scientific gate.  Reaction-transform scoring reached
80.8555% Recall@1 versus 80.6452% for the nested generic-edge comparator: only
+0.210 pp (5 corrected, 2 introduced).  The identity-cluster 95% interval was
[-0.140, +0.601] pp and the formula-cluster interval was
[-0.137, +0.567] pp.  It also failed to beat the wrong-sign transformation
control reliably (+0.140 pp; formula interval [-0.135, +0.426] pp).  Only 3 of
7 preregistered gates passed.

This result is informative rather than an implementation failure.  Under the
strict same-adduct, precursor-mass-matched candidate protocol, the candidate
formula-derived precursor shift was essentially identical to the observed
query--seed precursor difference for both true and wrong candidates.  Thus the
proposed mass-shift feature does not identify isomers.  The theoretical hybrid
fragment score consequently collapsed onto the observed-shift control.  B2 is
therefore stopped: it is neither an eligible reaction-specific reranker nor an
embedding teacher.

## B3 reaction-coabundance successor

The next experiment tests a different, sample-specific BioAware signal rather
than retuning B2.  For every query feature and candidate, it measures stable
log-abundance correlation across the six biological replicates to direct
Rhea/KEGG seed neighbours.  Each reaction neighbour is compared with three
non-neighbour seed controls matched on polarity, graph degree, mean abundance
and abundance dispersion.  The nested comparison is therefore reaction
coabundance versus catalog opportunity plus degree/abundance-matched random
coabundance, not versus DreaMS alone.

The experiment is source-LOSO with held truth identities and formulas purged.
The action is preregistered only for negative-ion queries because the B1
positive-ion stratum showed no benefit; positive-ion queries must remain
untouched as a safety stratum.  Passing requires at least +3 pp reaction-
specific gain, positive identity- and formula-cluster confidence bounds,
corrected greater than twice introduced, at least 20 corrected identities,
nonnegative transfer in all four biological sources, and zero positive-ion
interventions.  Even a pass establishes a sample-context candidate action,
not a universal spectrum-only embedding improvement: the same spectrum can
occur in different biological contexts, so this information can enter only a
context reranker or an explicitly context-conditioned representation.

## B3 observed result and protocol repair

The full 37,688-rotation B3 run passed the implementation validator but failed
the reaction-specific gate.  Relative to the degree/abundance-matched random
coabundance comparator it corrected 3 and introduced 4 queries (net -1,
-0.070 pp).  Identity- and formula-cluster intervals both crossed zero, and
Mouse brain was negative.  The absolute reaction-neighbour correlation had
descriptive headroom on 39 of 163 negative-ion DreaMS errors, but only 24
retained the expected ordering after subtracting the matched non-neighbour
control.  Hence raw correlation headroom cannot be reported as reaction-
specific performance.

One protocol weakness prevents immediate termination: the first B3 ranker was
fitted on 878 positive-ion and 548 negative-ion queries, then deployed only on
negative ion.  This mixed-polarity fit is misaligned with the intended action
and can dilute or reverse a polarity-specific coefficient.  B3b is therefore
the single allowed repair: fit only the negative-ion rows and add exactly one
reaction-specific excess-correlation coordinate to an otherwise identical
low-dimensional matched-random comparator.  No new feature family, threshold
sweep, or positive-ion intervention is permitted.  Failure of B3b terminates
coabundance as a BioAware action on this six-replicate benchmark.

## B3b final result

B3b completed with the exact negative-only protocol.  The matched-random
comparator improved DreaMS by 3.467 pp (20 corrected, 1 introduced).  Adding
the single reaction-neighbour excess-correlation coordinate changed three
wrong decisions to correct and three correct decisions to wrong: net zero,
with identity and formula intervals of approximately [-1.06, +0.95] pp.
Coefficient signs were not stable across held biological sources.  Therefore
the six-replicate coabundance family is terminated as a reaction-specific
BioAware action on this benchmark.  It must not be used as an embedding target.

The highest surviving engineering action remains the historical mixed-polarity
catalog/opportunity model deployed only on negative ion: +4.562 pp, 26
corrections and 1 introduction.  B4 must now decompose that action into
reference-library multiplicity versus graph membership/degree.  Only an
increment beyond the reference-only comparator can be called BioAware.

## B4 corrected opportunity-decomposition result

The denominator-corrected B4 result is
`data/validation/bioaware_b4_opportunity_decomposition_local_v2_20260906/report.json`.
The earlier directory without the `v2` suffix is invalid for the primary
comparison because it divided a negative-only intervention by all 1,426
mixed-polarity queries.  It must not be quoted.

On the 548 negative-ion queries, the reference-spectrum-count model made no
gated intervention and gave zero gain over DreaMS.  The graph-only recipe
(`spectral_score`, graph membership, graph degree and graph-covered mass-
candidate fraction) improved Recall@1 by 4.927 pp: 28 corrections, 1
introduction, risk-weighted net 26 at lambda=2, 14 corrected identities and 13
corrected formulas.  Identity-cluster and formula-cluster 95% confidence
intervals were [2.272, 8.205] pp and [1.890, 8.394] pp.  The held-source gains
were positive in BV2 cells (+5.263 pp), mouse brain (+5.344 pp), mouse liver
(+5.682 pp) and NIST plasma (+3.425 pp).

This changes the surviving-action diagnosis.  The gain is not explained by
reference-library multiplicity; it is attributable to graph coverage and
degree under the fixed low-margin gate.  It is therefore a real BioAware
graph-prior action.  It is not, however, evidence for a particular reaction
path or transformation: B2 and B3b already showed that the tested transform
and six-replicate co-abundance coordinates add no reliable reaction-specific
increment.  B4 remains an opened-development, negative-ion candidate reranker
result.  It does not change the shared DreaMS embedding and is not yet an
external or SOTA claim.

## B5 external-transfer contract

B4 may advance only as a frozen low-dimensional graph-prior action.  One model
is fitted on the Full16 development candidates after purging the union of all
external truth IK14s and formulas; its four feature names, scaler, coefficients,
gate constants and input hashes are then frozen before any external outcome is
scored.  ST001154 and KGM-200STD are already opened historical resources, so
they are transfer-development panels rather than new blind confirmation.

The external audit must report both the ungated graph-ranking headroom and the
frozen gated result.  This distinction is mandatory: it separates failure of
the graph ordering from failure of absolute confidence calibration.  No gate
may be retuned on these external outcomes, and any successor calibration rule
developed with them requires a fifth, previously unused negative-ion Level-1
collection for confirmation.  Reaction-specific claims remain forbidden even
if the graph prior transfers.

## B5 observed transfer result

The frozen graph-only ordering model was evaluated on three already-opened
transfer-development panels after purging the union of their truth identities
and formulas from the Full16 fit.  The serialized scaler and coefficients were
reloaded before outcomes were computed.  On the ST001154 same-formula/10-ppm
panel it improved Recall@1 by 11.333 pp (18 corrected, 1 introduced; identity-
and formula-cluster interval lower bounds about +1.9 and +2.0 pp).  On the
KGMN-200STD hidden-seed rotations it improved by 9.259 pp (18 corrected, 3
introduced), although its cluster intervals crossed zero because only 39 truth
identities were available.  On the differently constructed ST001154 author-
candidate panel it changed Recall@1 by -0.621 pp (6 corrected, 7 introduced).

The preregistered absolute probability gate intervened zero times on all three
panels.  The defensible diagnosis is therefore: graph ordering signal transfers
on the two mass-matched candidate protocols, while absolute cross-domain
calibration does not.  This is not three independent confirmations and not a
SOTA claim, because all panels are now opened and one candidate protocol is
negative.  It nevertheless justifies testing whether graph-selected retrieval
boundaries contain spectrum-visible signal that a shared encoder can learn.

## B7 direct shared-embedding injection contract

B7 does **not** distil graph probabilities or a candidate reranker into the
encoder.  The graph-only model is used once, in nested truth-formula OOF, to
mine two sets: official DreaMS errors for which its unique direct proposal is
the true molecule, and official-correct cases that it would harm.  The first
set is the corrective hard-example stream; the second is forced into the
safety stream.  The actual optimization target is the unique true molecular
identity under a candidate-group listwise loss plus a hardest-negative margin,
with official-margin floors and representation preservation on safety cases.

Both query and candidate spectra are passed through the same trainable encoder.
Only the final transformer block and official projection head are unfrozen;
dropout remains off.  P2b, phenotype labels, graph logits and graph features
are absent from the loss and from inference.  At deployment, one clean spectrum
is mapped directly to the new embedding.

Execution is fail-closed in three stages.  First, the nested action router must
retain at least +3 pp direct headroom, a positive `corrected - 2*introduced`
risk net, at least 25 corrected identities, and nonnegative gain in every outer
formula fold.  Second, an eight-identity one-step audit must reproduce official
embeddings, produce a finite nonzero gradient, reduce the direct truth-ranking
objective, increase the exact truth-versus-baseline-wrong margin for at least
60% of cases, and retain cosine preservation of at least 0.999.  Only then may
the fold-0 GPU pilot run.  The pilot is an opened development result; it cannot
be called an embedding improvement until its held-formula Recall@1/MRR,
corrected/introduced counts and preservation are read from the frozen report.
