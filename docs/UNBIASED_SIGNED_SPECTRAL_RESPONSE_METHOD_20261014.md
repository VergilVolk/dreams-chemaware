# Unbiased signed spectral response method

Date: 2026-10-14  
Status: active biology-method development contract; all datasets named below are development/consumed unless separately sealed

## 1. Scientific decision

The biology method is not an obesity classifier, a disease-specific network, a website, or a requirement to build an Atlas.  Its testable objective is:

> Can every QC-qualified LC-MS/MS entity enter a study-level biological response without requiring a structure name, and can the unresolved part add correct, independently verifiable information beyond known-only, MS1-only, and the strongest frozen spectral baseline?

`Unbiased` refers only to entry into discovery: entities are not selected by a library hit, named pathway, expected molecule, or target disease story.  It does not imply unbiased ionization, chromatography, DDA sampling, cohort recruitment, or public-data availability.

The HFD/obesity queries executed on 2026-10-14 do not support the biological innovation.  They are retained only as failed method falsifications.  The coarse biological truth, cross-species and cross-biofluid differences, and consumed post-hoc status prevent them from serving as a flagship application.

## 2. Core object: a signed spectral response measure

For study `d`, entity `i` has a study-internal effect and uncertainty,

\[
r_{di}=g(\hat\beta_{di},SE_{di}),
\]

where `g` is fixed before cross-study truth is opened.  A bounded example is `beta / sqrt(beta^2 + SE^2)`.  QC, blank, missingness, ion-family redundancy, and MS2 acquisition weights are represented by `q_di`.

The complete contrast is represented by two finite measures over frozen spectral evidence `z_i`:

\[
\mu_d^+=\sum_i q_{di}\max(r_{di},0)\delta_{z_i},\qquad
\mu_d^-=\sum_i q_{di}\max(-r_{di},0)\delta_{z_i}.
\]

Study comparison is performed with a calibrated spectral kernel, not an arbitrary k-nearest-neighbour component and not a forced one-to-one feature match.  The four signed comparisons (`+:+`, `-:-`, `+:-`, `-:+`) remain visible so that concordant and inverted responses cannot cancel invisibly.

This object has three advantages that must be demonstrated rather than assumed:

1. all eligible known and unresolved entities can enter under the same QC rule;
2. partially overlapping chemical coverage can contribute without claiming identical molecular structure;
3. effect direction and uncertainty are part of retrieval, rather than being consulted only during final scoring.

## 3. Evidence semantics remain separate

| Evidence level | Permitted conclusion | Forbidden upgrade |
|---|---|---|
| calibrated entity evidence | compatible with the same observable entity, with no-match | exact structure, pathway, or flux |
| spectral relation kernel | shared fragmentation/chemical evidence at measured resolution | identity, biochemical reaction, or function |
| signed response concordance | two designs show reproducibly concordant or inverted chemical responses | common mechanism or causality |
| isotope/perturbation evidence | source incorporation or intervention dependence | exact reaction or enzyme without the required orthogonal evidence |
| standard/in-vitro/rescue evidence | the specifically tested identity or causal step | generalisation beyond the tested system |

Identity and relation channels are reported separately.  A model may assign zero weight or abstain from either channel.  The method is not allowed to obtain power by counting multiple ions, highly connected common fragments, or several derivatives of the same spectral score as independent evidence.

## 4. Required control worlds

Every decisive experiment uses one fixed entity universe and reports:

1. `known_only`;
2. measurement-matched known controls;
3. `unresolved_MS1_only`;
4. unresolved entities with MS2 links shuffled within acquisition/mass/information strata;
5. strongest eligible frozen spectral baseline;
6. the full signed spectral response method;
7. unresolved-only as a diagnostic, not a substitute for incremental utility.

The primary unknown increment is performance of world 6 minus the best of worlds 1--5 at the same candidate universe and selective-risk/coverage operating point.  Increasing answered coverage while increasing false biological links is a failure.

## 5. Development ladder

### Gate A: hidden relation truth

MSV000097764 (IsoNet) is used only as a development/consumed hidden-truth benchmark.  Compound names, formulae, annotation confidence, author metabolite pairs, atom differences, reaction classes, KEGG distance, and validated target names are withheld.  Observable inputs are anonymous entity IDs, m/z, retention time, isotope-label fractions, ion mode, and fragment peaks.

Passing Gate A means that frozen spectral evidence preserves a low-resolution relation independently supported by isotope patterns.  It does not establish unbiased full-feature discovery, LPS biology, or a new discovery by this project.

### Gate B: hidden intervention retrieval

The same signed-response representation must retrieve held-out interventions or targets from a multi-intervention collection under source/instrument/intervention holdout.  Labels may define the final endpoint but may not select entities, spectral edges, kernel scale, or abstention thresholds.

ChemProp is already consumed and negative as a cross-design endpoint; it cannot be tuned further to rescue this gate.  TT7 is a same-study genetic positive control, not an independent source.  Additional sources must therefore supply real intervention contrasts and verifiable MS1--MS2 linkage.

### Gate C: unresolved conditional increment

Across at least two biological sources, `known + unresolved` must exceed every matched control world with a cluster-level confidence-interval lower bound strictly above zero, without a higher wrong-link rate.  A pooled result driven by one source, one chemical family, or one ion family fails.

### Gate D: new biological discovery

Only after Gates A--C are frozen may a disease or organismal study query the perturbation evidence.  A flagship candidate must be chosen without using its eventual validation result and must receive independent support from a second cohort/source plus one causal/source experiment such as perturbation-rescue, time/dose, isotope tracing, or a specific in-vitro reaction.  A public co-occurrence, spectral neighbour, or known pathway story is insufficient.

## 6. MSV000097764 readiness result

The Zenodo demo archive was downloaded from record `14911858`:

- bytes: `7,726,594`;
- deposited and observed MD5: `f510080c207db9501a835e42c06f7970`;
- role: development/consumed, not sealed external validation.

Six structure-masked views were built:

| View | MS/MS spectra | coordinate rows | isotopologue rows | labeled entities | replicates |
|---|---:|---:|---:|---:|---:|
| 293T negative | 5,888 | 4,470 | 16,952 | 984 | 6 |
| 293T positive | 7,143 | 4,947 | 17,927 | 1,005 | 6 |
| iBMDM control negative | 4,375 | 3,366 | 10,544 | 686 | 6 |
| iBMDM control positive | 4,816 | 3,494 | 12,864 | 716 | 6 |
| iBMDM LPS negative | 4,416 | 3,385 | 11,085 | 731 | 6 |
| iBMDM LPS positive | 4,870 | 3,547 | 13,581 | 751 | 6 |

Leakage audit: forbidden semantic columns are absent and cross-view anonymous entity-ID overlap is `0`.  View-scoped hashing prevents an original feature name from becoming a free cross-condition identity key.

The MassIVE accession contains 211 files totalling `3,742,694,632` bytes.  Raw LPS and control, labeled and unlabeled iBMDM measurements plus pooled-QC ddMS2 are present.  At least three negative-mode mzXML entries are zero-byte in the public listing (`L_control_5`, `L_LPS_6`, `U_LPS_4`) and must remain explicit missing replicates.  The processed demo is prefiltered for labeled metabolites, so it cannot replace the raw unlabeled full-feature universe for an unbiased LPS-response test.

## 7. Current falsifications

1. MSV000101150 positive/negative hidden-individual retrieval is invalid as an unbiased validation because both polarities use the identical 15-injection order; run order alone can recover sample identity.
2. The all-edge degree-normalized WSE response kernel did not rescue cross-study response retrieval.  On the consumed MSV000097426-to-MSV000084112 panel, the target ranked first but node-level mass-stratified permutation was `p=0.0828` for all 626 edges and `p=0.1976` for 544 unresolved-both edges.  This is a failed post-hoc method screen, not evidence for obesity biology.
3. The strict HFD cross-study panel had only 13 qualified spectral candidates and 9 reciprocal edges, below the frozen minimum of 12.  Coverage, not ranking, blocks that panel.

These results close threshold relaxation, additional obesity-specific weighting, and run-order-confounded sample retrieval as high-value routes.

## 8. Nissle genetic positive-control decision

MSV000086206 is not independent of MSV000083387.  The official file listings
contain the same six biological RAW files (`S_N1--3`, `S_P1--3`) with identical
names and exact byte sizes in both accessions, and their metadata assign the
same three WT and three yersiniabactin-KO samples.  MSV000086206 is therefore a
small processed republication for an LC-MS viewer, not a second source family.

The small processed panel contains 1,987 features and exactly 1,987 linked
MS/MS spectra.  A structure-hidden WT-minus-KO analysis was scored before the
paper-supported yersiniabactin masses were opened.  The two commercial-standard
supported chromatographic peaks map to four ion features (two singly charged
and two doubly charged forms).  All four changed in the expected WT-positive
direction and attained the minimum possible one-sided exact p value (`1/20 =
0.05`) under the 3-versus-3 design.

This is not selective discovery.  There were 363/1,987 WT-positive features at
the same minimum exact p value, while the four truth ions ranked only 254--291
by the frozen response scores.  Only 25/1,987 features passed the consumed GNPS
Gold/Silver contract (10 ppm, WSE >=0.70, at least six matched peaks), and all
four truth ions remained unresolved.  Frozen WSE nevertheless linked the two
same-charge truth peaks strongly (`0.987905` with 14 matches and `0.943793` with
37 matches), showing that spectral relation evidence preserves real chemistry
that response magnitude alone cannot prioritise.

Formal decision: retain this panel as one `development/consumed` genetic and
spectral-relation positive control.  It does not estimate unknown-over-known
increment, cross-source transfer, or a new discovery; it cannot carry the
biology innovation.  The original accession's blanks, QCs and other metal
branches would be needed for a stronger reanalysis, but their acquisition is
not justified until the general relation gate passes.

## 9. IsoNet structure-hidden relation endpoint

An orthogonal relation endpoint was constructed without reading compound names,
formulae, author metabolite-pair output, reaction classes or KEGG fields.  For
each anonymous entity, the positive endpoint is its top 1% neighbours by the
stable-isotope incorporation vector after removing M+0, so shared absence of
labelling cannot create a positive.  Every eligible entity is scored against
the same complete within-view candidate universe.

The six views retained 598--874 entities with at least six cleaned MS/MS peaks,
for 357,006--763,002 directed candidate pairs per view.  The fixed WSE baseline
was weakly informative in all views (AUROC 0.5222--0.5518); its AP was 1.45--2.47
times the approximately 1% positive prevalence.  This was not a spectral-method
win.  Median WSE AUROC was 0.5366 versus 0.6185 for truth-blind mass-plus-RT
proximity, and WSE beat that baseline on AUROC in 0/6 views.  WSE AP exceeded
mass-plus-RT in 4/6 views, showing a limited selective tail rather than broad
relation recovery.

Formal decision: the endpoint is ready, but WSE alone fails to qualify the
chemically informed relation layer needed by the biology method.  The identical
anonymous pair universe must now be scored once by the qualified unified
Noise/ChemAware/P2b model.  It may advance only if it improves WSE and the MS1
baseline consistently across views without using isotope patterns for tuning.
Local qualified checkpoints are currently unavailable, so that comparison is
`NOT_RUN`, not silently replaced by an unqualified model.

## 10. Immediate experiment

The next experiment is not another disease query and not another generic
evaluator.  The six masked spectra sets and their exact entity orders are now a
fixed scoring bundle.  The algorithm branch must return one similarity matrix
per view from its qualified frozen unified model, without receiving isotope
vectors or pair labels.  The biology branch then applies the already executed
top-1% isotope endpoint and reports the same AUROC, AP enrichment and query
Recall@10 beside WSE and mass-plus-RT.

Only if that comparison passes may we acquire the minimum non-zero
`U_control/U_LPS` raw subset and test an all-feature signed LPS response.  Nissle
remains a positive-control shard, not a substitute second source.  A new
biological discovery still requires the later independent intervention and
prospective/orthogonal validation gates.

Failure to exceed WSE and mass-plus-RT on the frozen isotope endpoint ends the
relation-kernel route.  Success advances only to Gate B; it does not itself
count as the paper's biological discovery.
