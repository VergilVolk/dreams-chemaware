# BioAware B47-U3 truth-blind event-yield protocol amendment

Date: 2026-09-21  
Status: reaction-signature-corrected v3 frozen on 2026-09-22 before annotation
truth, phenotype, event outcome, model fitting, or performance evaluation is
opened. The v1 executor was rejected before formal execution. The v2 server run
was interrupted after its first truth-blind null preview because it recomputed
the expected transformation from rewired identities and treated wrong-identity
intervention counts as a directional null. Its event/null counts are invalid
for scientific claims and its output namespace is retired.

## Why the protocol is amended

The original B47-M1 contract required at least 200 distinct seed identities before constructing reaction events. U1c and U2 show 238 spectrally consensual identities, 127 Rhea-safe identities, 141 provisional Rhea/strict-KEGG identities, and 3,243 independent sample-collapsed seed events.

The 200-seed-identity threshold measures seed diversity. It is not the statistical denominator of the intended endpoint. The endpoint is candidate ranking on independent query events, and one seed identity may create candidate-specific typed events for many queries. Therefore the old threshold cannot by itself establish that the event model is underpowered.

This amendment is permitted because no B47 annotation truth, event gain, or model outcome has been opened. It does not lower an outcome-dependent threshold after seeing performance.

## The missing innovative step

U3 constructs an auditable candidate-specific event layer with atomic unit:

`query feature -- candidate identity -- seed feature -- seed identity -- reaction source/id/direction`.

The method is not static catalogue membership and not multi-hop diffusion. It asks whether an observed, truth-blind, high-confidence metabolite event in the same sample supplies a typed biochemical transformation that distinguishes one candidate from its actual competitors.

Each event must retain:

- study, sample, query feature and candidate identity;
- seed feature, seed identity, seed score and independent-support key;
- exact source label, reaction identifier, participant sides and direction status;
- expected precursor transformation and mass residual;
- candidate-specificity against every competing candidate;
- ion-family reconciliation status when available;
- catalogue degree/membership as nuisance variables, never as unqualified positive votes.

An event is eligible only after adduct-aware neutral-mass replay against the
frozen candidate and seed molecular formulas.  Query and selected reference
neutral masses must each independently match their assigned formula within 10
ppm before the candidate-minus-seed transformation residual is tested; matching
only the mass difference is forbidden because two identity errors could cancel.
In addition, the reaction endpoint's original non-hydrogen elemental-delta
signature is frozen before any graph rewiring.  A rewired or identity-permuted
edge may not redefine its expected transformation from the reassigned
identities; doing so would make every random edge chemically self-consistent by
construction.  Hydrogen and terminal charge are excluded from this signature
because Rhea participant protonation and positive-mode neutral candidates need
not use the same ionic representation.
Participants with zero net stoichiometric
change are removed as catalysts/carriers.  Candidate specificity is computed
across every candidate supported by the observed seed, not separately per
reaction record.  Multiple Rhea records for one observed seed--candidate pair
are collapsed before noisy-OR so catalogue duplication cannot manufacture
independent biological evidence.

## Structured controls

The same event table must generate, before truth is opened:

1. degree/component-preserving reaction rewires;
2. between-sample, within-study permutation of intact seed events, matched on
   seed score and degree while preserving each target sample's seed-row count;
3. within-query, same-adduct candidate-identity permutations that preserve the
   spectral score and reference multiplicity, providing mass-matched wrong
   transformations;
4. static catalogue membership/degree as a reported nuisance baseline.

Reactome-consensus direction is a diagnostic stratum, not a negative control.
In a steady-state sample, observing a product as the high-confidence seed can
still support the identity of a substrate feature.  Treating every reverse
orientation as false would therefore encode an invalid biological assumption.
Canonical Rhea left/right serialization is never interpreted as physiological
direction.

## U3 decisions

U3 reports event yield rather than accuracy:

- exact-event-supported queries and candidates;
- queries where real events distinguish at least one candidate from all competitors;
- independent seed identities, seed features, reaction IDs, query features and candidate formulas;
- per-study coverage and concentration;
- real-versus-null event-support distributions;
- maximum possible intervention coverage without using truth.

Candidate-specific evidence is not automatically an intervention opportunity.
If the unique event winner is already the unique DreaMS winner, it cannot alter
Recall@1.  The actionable denominator is limited to exclusive event winners
that either disagree with a unique spectral winner or resolve a spectral tie
without using candidate order.  With 51,976 frozen queries, a +3 percentage-
point final target is arithmetically impossible unless at least
`ceil(0.03 * 51,976) = 1,560` such query decisions can change.  U3 must clear
this headroom gate before any one-time outcome evaluation.

U3 may authorize a one-time frozen event-ranking evaluation only if event
support is not dominated by one seed, reaction, candidate, study, catalogue
degree or reference multiplicity; both studies contribute material support;
all three control families have at least 20 repeats. Real candidate-specific
event opportunity must exceed the graph-rewired and seed-context-permuted
structural nulls with empirical one-sided p <= 0.05 and at least 10% lift over
the worst null both overall and within each study. The within-query
wrong-identity permutation is retained as a non-directional identity--spectrum
misalignment diagnostic: random wrong identities can create more disagreements
than a conservative biological graph, so a larger intervention count is not
interpreted as stronger evidence. Candidate/formula concentration and
event-winner reference
multiplicity/catalogue degree are frozen report fields rather than post-hoc
explanations. The final
performance gate remains unchanged: at
least +3 pp, positive formula/source clustered intervals, corrected greater
than twice introduced, at least 50 corrected identities, and superiority to
catalogue-only and all nulls.

Fail-fast is mandatory when any completed structural-null repeat equals or
exceeds the real actionable-opportunity count.  At that point the
real-greater-than-every-null gate is mathematically unrecoverable, so remaining
repeats may not consume compute merely to refine a result that is already a
formal `NO-GO`.  The executor must still write and validate the partial null
ledger, record requested versus completed repeats, and force the authorization
decision to false.

## Claim boundary

U3 construction is the prospective innovation test. It does not promise +3--5 pp and does not convert inferred or EMRN edges into exact biochemical reactions. If real typed events do not create sufficient candidate-specific opportunity beyond nulls, the exact-event branch stops without opening an embedding experiment.
