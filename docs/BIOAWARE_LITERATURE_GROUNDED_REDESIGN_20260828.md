# BioAware redesign grounded in existing methods (2026-08-28)

## Decision

Retire BioAware v1 as a ranking algorithm. Its global one-hop Rhea score is
kept only as an audited negative baseline and an explanation source.

BioAware v2 will not be another independently invented graph score. It will
implement a data--knowledge two-layer annotation design grounded in MetDNA3,
with DeepMet-style continuous metabolite-likeness, DreaMS as the experimental
spectral layer, and conservative evidence fusion. Biological module discovery
is separated from identity ranking and follows the TidyMass2/NetID principle.

## What the strongest relevant methods actually do

### DeepMet (Nature, 2026)

- Learns a metabolite-like structure distribution from known metabolites.
- Uses sampling frequency as a continuous structure prior, not a binary
  reaction-neighbour label.
- Retrieves structures within precursor-mass tolerance and combines the prior
  with experimental-vs-predicted MS/MS similarity.
- Adds orthogonal MS1 isotope and retention-time evidence in a cross-validated
  meta-model.
- Validates selected discoveries with standards and matched LC-MS/MS/RT.

DeepMet therefore supports a candidate prior and candidate expansion. It does
not justify overriding a spectral top hit merely because a candidate is one
reaction away from any seed.

### MetDNA and MetDNA3 (Nature Communications, 2019 and 2025)

- Initial seeds are identified using standard MS/MS and RT libraries.
- A knowledge layer contains reaction-paired metabolites.
- A data layer contains experimental features linked by MS/MS similarity.
- Candidate propagation requires agreement between reaction-neighbour search,
  precursor m/z, predicted/experimental RT, and feature-level MS/MS similarity.
- MetDNA3 pre-maps knowledge and data layers and retains only cross-network
  links supported by experimental constraints.
- Propagated annotations are reported as lower-confidence annotations rather
  than silently promoted to authentic-standard identity.

The missing element in BioAware v1 is the experimental feature layer. A Rhea
edge alone is not an annotation.

### NetID (Nature Methods, 2021)

- Builds one global peak network rather than scoring every query independently.
- Uses mass, RT, isotopes, adducts, in-source fragments, MS/MS, and feasible
  biochemical transformations.
- Optimizes assignments jointly so that one local decision must remain
  consistent with the rest of the observed LC-MS feature graph.

This directly addresses v1's failure to distinguish a plausible biochemical
neighbour from the identity of the current feature.

### MetGenX (Nature Communications, 2026)

- Retrieves structural templates by spectral similarity.
- Encodes template fingerprints, molecular formula, and formula differences.
- Generates candidate structures and then re-ranks them.
- Uses structure-disjoint evaluation and explicit candidate-generation metrics.

MetGenX is relevant to candidate generation for genuinely unknown structures;
it is not the first fix for v1's unsafe network override.

### MS-Net (Analytical Chemistry, 2026)

- Combines spectral-network similarity, molecular-structure similarity, and
  taxonomic knowledge.
- Propagates from high-confidence seeds and uses a composite link score to
  rescue lower-ranked candidates.

Its useful lesson is that network propagation requires simultaneous structural
and spectral consistency plus domain context. Its plant/taxonomy validation is
not directly transferable to human colorectal tissue without revalidation.

### TidyMass2 (Nature Communications, 2026)

- Separates metabolite-origin inference and biological module analysis from
  strict identity assignment.
- Uses a multi-database metabolic network and feature-level modules to derive
  biological information even when many features remain unannotated.

This supports a separate downstream module for MTBLS13729 biology; phenotype
labels must not leak into the identity-ranking model.

## Why BioAware v1 failed

The v1 score omitted every safeguard that the methods above rely on:

1. no experimental feature--feature network;
2. no requirement that seed and candidate MS/MS spectra are similar;
3. no sample-wise co-detection or abundance consistency;
4. no RT/isotope/adduct consistency in the network score;
5. no human/tissue/enzyme restriction on universal Rhea reactions;
6. no reaction-complete/co-substrate evidence;
7. no global assignment constraint;
8. a forced Top-1 override instead of an ambiguity/abstention state.

The GABA/2-aminobutanoate failure is exactly predicted by this omission: two
chemically valid transamination paths make 2-aminobutanoate plausible, but do
not establish that the observed feature is 2-aminobutanoate.

## BioAware v2 architecture

### Layer A: candidate structures

- Existing spectral-library/database candidates remain the first source.
- DeepMet frequency is an optional continuous prior for candidates present in
  its released structure table.
- DeepMet/MetGenX candidate expansion is evaluated separately from ranking so
  coverage gains cannot be confused with ranking gains.

### Layer B: experimental feature graph

Nodes are aligned MS1 features. Edges are created only from label-free
experimental evidence:

- DreaMS similarity between linked MS2 spectra;
- classical/neutral-loss spectral similarity;
- isotope/adduct/in-source-fragment relationships;
- RT compatibility and cross-panel consistency;
- sample-wise co-detection and abundance covariance, estimated without tumor/
  normal labels.

### Layer C: biochemical knowledge graph

- Human-relevant reactions are preferred over the unrestricted universal graph.
- Currency compounds and highly promiscuous edges are downweighted.
- Reaction pairs must preserve meaningful structural subgraphs/atom mappings.
- Direction is not claimed unless supported; undefined Rhea directions remain
  undirected evidence.
- Candidate origin (human, microbial, dietary, drug, environmental) is retained
  as context, not silently equated with truth.

### Cross-layer consistency gate

A network contribution is permitted only when all of the following hold:

1. the candidate matches precursor mass/formula/adduct;
2. a reaction-paired metabolite is supported by an independent seed;
3. the corresponding observed features are neighbours in the experimental
   feature graph;
4. spectral/RT/isotope evidence does not contradict the candidate;
5. leave-query-out and leave-truth-identity-out conditions hold;
6. at least two independent evidence families support an override.

Otherwise BioAware emits `supported`, `conflicted`, or `insufficient`, without
changing Top-1.

### Fusion and uncertainty

The first implementation uses a preregistered monotone/listwise fusion over
orthogonal candidate evidence:

- DreaMS candidate score;
- DeepMet prior when available;
- experimental feature-edge score;
- biochemical cross-layer score;
- RT, isotope, adduct, and neutral-loss evidence;
- peak-token explanation/conflict indicators.

Formula-/scaffold-disjoint out-of-fold training selects the model and gate.
Calibration and abstention are evaluated explicitly. Phenotype, differential
abundance, and pathway enrichment are forbidden features for identity ranking.

## Minimal sequence of experiments

### V2-0: reproduce established mechanics

On a large identity-known benchmark, reproduce three ablations:

1. DreaMS alone;
2. DreaMS plus unrestricted one-hop Rhea (the v1 negative control);
3. DreaMS plus MetDNA3-style cross-layer consistency.

The cross-layer model must beat both controls in formula-cluster confidence
intervals and must reduce, not merely exchange, introduced errors.

### V2-1: MTBLS13729 feature graph

Build the full neg-RP and pos-RP feature graphs from the MS1 matrices and linked
MS2 spectra. The graph construction is phenotype-blind. Verify coverage,
degree distribution, edge reproducibility across patients, and known adduct/
isotope recovery before candidate ranking.

### V2-2: GABA ambiguity replay

Replay feature 1705. A correct v2 implementation should not force the
2-aminobutanoate override unless the feature graph supplies independent
experimental evidence. The expected safe output under current evidence is
`ambiguous aminobutanoate isomer`, not a new definitive identity.

### V2-3: external accuracy benchmark

Use hundreds to thousands of identity-known queries with enough baseline
errors and graph coverage. Freeze the model before final testing. Report
Recall@1, MRR, corrected/introduced transitions, calibration, coverage, and
formula/scaffold-stratified confidence intervals.

### V2-4: biological application

After identity scoring is frozen, use paired tumor/adjacent abundance changes
and a TidyMass2-style feature module analysis. This downstream stage can use
phenotype labels and addresses metabolic reprogramming; it must not alter the
annotation benchmark retrospectively.

## Project-specific innovation after reproduction

Only after V2-0 reproduces established behavior do we add our contributions:

1. DreaMS supplies cross-instrument/condition-tolerant edges in the data layer.
2. Contextual peak tokens and the double mapping provide peak-level evidence
   for why a cross-layer link is supported or contradicted.
3. The noise/ChemAware embedding projects can distill only cross-layer-confirmed
   relations into a shared embedding, rather than distilling raw Rhea adjacency.
4. Risk-calibrated abstention prevents a biologically plausible edge from
   silently overriding stronger spectral evidence.

This preserves the original project direction while replacing the unsupported
one-hop heuristic with a literature-grounded and falsifiable annotation system.

## Primary sources

- DeepMet: https://doi.org/10.1038/s41586-025-09969-x
- MetDNA: https://doi.org/10.1038/s41467-019-09550-x
- MetDNA3: https://doi.org/10.1038/s41467-025-63536-6
- NetID: https://doi.org/10.1038/s41592-021-01303-3
- MetGenX: https://doi.org/10.1038/s41467-026-72149-6
- TidyMass2: https://doi.org/10.1038/s41467-026-68464-7
- MS-Net: https://doi.org/10.1021/acs.analchem.6c01026

