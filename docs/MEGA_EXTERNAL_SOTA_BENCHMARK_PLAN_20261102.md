# Mega external identity benchmark and SOTA ladder plan

Date: 2026-11-02

Status: strict evaluator, public-model adapters and a 16-method GNPS smoke
ladder are materialized locally; no claim of an unopened large external result

## Executive decision

The project will build one evaluation system, not one falsely inflated sample
count.  It will report five denominators everywhere:

- `N_Q`: labelled query spectra;
- `N_I`: independent identity clusters;
- `N_F`: independent formula clusters;
- `N_E`: directed query--candidate-molecule edges;
- `N_P`: actual query--reference-spectrum comparisons.

The target of more than ten million operations applies to `N_P`.  It must
never be described as ten million independent labelled queries.  No verified
public, bulk-downloadable small-molecule MS/MS corpus currently provides ten
million experimental spectra with reliable identity truth.  Unlabelled public
spectra, propagated annotations, predicted spectra and database-search hits do
not become ground truth by being numerous.

The final SOTA claim therefore has three tiers:

1. **Primary independent confirmation:** NIST26 identities genuinely new
   relative to NIST23, after exclusion of all consumed identities, formulas
   and available exact spectrum hashes.  This requires a commercial licence
   and is not present locally.
2. **Independent condition stress:** MultiMS2 CID60-to-CID40, with EAD16-to-
   CID40 as a separate OOD endpoint.  The exact 77,183,510-byte LF-normalized
   source is now local and matches SHA-256
   `4ef86f75a12ad2fdadbd63732e2d6690f4e506a6ff08f171758c1808d1249673`.
   Its competitive-query denominator is expected to be hundreds, so it cannot
   be the sole SOTA test.
3. **Large aggregation/throughput stress:** StructureMASST/FASSTrecords and
   other public aggregates after source-aware de-duplication.  These measure
   throughput and robustness; aggregation labels and training overlap prevent
   them from automatically becoming an independent final test.

If licensed NIST26 cannot be obtained, the final replacement is a MassBank
release first published after the model freeze date or a prospective blind
multi-instrument standards experiment.  GNPS, Enveda, MassSpecGym, MassBank
2026.03, local MoNA and the MSnLib source family are already consumed.

The machine-readable boundary is
`tasks/mega_external_sources_v1.json`.

## What has been completed locally

### Frozen Full23 comparison

Full23 v1.3.2 has been added to the existing 15-method GNPS same-axis ladder.
The two panels are consumed development, not external confirmation:

| Endpoint | Identity-disjoint (`N_Q=10,995`) | Formula-disjoint (`N_Q=5,261`) |
|---|---:|---:|
| Recall@1 | **88.3674%** | **89.6027%** |
| MRR | **92.4522%** | **93.7402%** |
| near Recall@1 | **83.8251%** | **82.3030%** |
| macro query AUROC | **94.9251%** | **94.6532%** |
| micro candidate AUROC | **95.8538%** | **95.3840%** |
| micro candidate average precision | **84.4136%** | **87.4153%** |

On these exact axes Full23 v1.3.2 leads all six endpoints shown above.  It is
not first on every high-k endpoint: identity R@20 and formula R@10/R@20 are
led by Noise V1; formula R@5 is led by P2b-on-Noise.  This is the correct
meaning of a multi-metric ladder.

Relative to WSE, Full23 v1.3.2 changes identity Recall@1 by `+1.0005 pp`
(`223 corrected / 113 introduced`) and formula Recall@1 by `+1.3305 pp`
(`116 / 46`).  Relative to official fine-tuned DreaMS the changes are
`+3.0105 pp` and `+2.7941 pp`.  Full details are in
`docs/FULL23_V132_SOTA_COMPARISON_AND_MANUSCRIPT_POSITION_20261102.md` and
`deliverables/GLM_gnps_article_ladder/run16_full23_v132/`.

### New data and competitors materialized

- MultiMS2 pinned source commit `659bd9b...` and exact MGF are local.
- MSBERT official code is pinned at `4b1e85e...`; the historical Git-LFS
  checkpoint is materialized locally (486,361,781 bytes, SHA-256
  `f6a50e1a...f064605`).
- SpecEmbedding official code is pinned at `c16205c...`.  The authors'
  official Hugging Face Space supplies the executable 31,572,706-byte
  checkpoint (SHA-256 `0ca0aa00...f0f5cca`), which passes a true 512-D
  finite/unit-norm forward test.  The Figshare archive still returns HTTP 403
  from this network, but it is no longer a checkpoint blocker.
- MS2Query and MS2Query2 official repositories are pinned locally.  They are
  not silently inserted into the fixed candidate graph because their native
  retrieval/neighbourhood task is different.
- Spec2Vec, MS2DeepScore, MatchMS and MS-Entropy are installed locally.

Exact method roles and contamination boundaries are in
`tasks/mega_sota_methods_v1.json`.

### Large evaluator

`tasks/evaluate_mega_identity_benchmark.py` and
`tasks/plot_mega_identity_benchmark.py` implement a streamed candidate-
molecule evaluation.  The formal contract requires:

- Recall@1/2/3/5/10/20;
- MRR, mean rank and median rank;
- near-subset Top-k and rank endpoints;
- macro-query AUROC and average precision;
- micro-candidate AUROC and average precision;
- corrected/introduced and paired cluster confidence intervals;
- exact ROC and precision--recall curve artefacts;
- ladder, Top-k, ROC and PR figures in PNG and PDF;
- explicit `N_Q/N_I/N_F/N_E/N_P` accounting and candidate-axis provenance.

The evaluator streams methods and queries and memory-maps NPZ members.  It
does not materialize a method-by-edge matrix.  Exact pooled curves still
require sorting one `N_E`-length index per method; this memory cost is reported
rather than hidden.  Score archives are rejected unless they bind the complete
candidate axis with the panel SHA-256 or canonical axis digest and the exact
`N_Q/N_E/N_P` counts.

### Completed 16-method evaluator smoke test

The strict evaluator has now been run on both consumed GNPS development
panels.  This is a systems test and an updated development ladder, not a new
external result.

| Panel | `N_Q` | `N_I` | `N_F` | `N_E` | `N_P` | near queries |
|---|---:|---:|---:|---:|---:|---:|
| identity-disjoint | 10,995 | 5,534 | 4,204 | 87,518 | 175,171 | 7,592 |
| formula-disjoint | 5,261 | 2,662 | 2,210 | 24,401 | 47,724 | 2,944 |

All 16 existing score streams were evaluated for Recall@1/2/3/5/10/20,
MRR, rank, near-subset metrics, macro-query ROC/AP and pooled candidate ROC/AP.
The evaluator also emitted exact compressed ROC/PR coordinates, corrected and
introduced ledgers, and 10,000-resample formula- and identity-cluster paired
intervals.  Full23 v1.3.2 exactly reproduces the earlier run-16 metrics.  Its
paired Recall@1 improvement over WSE is `+1.0005 pp` on identity-disjoint
(formula-cluster familywise CI `[+0.4637,+1.5548] pp`) and `+1.3305 pp` on
formula-disjoint (`[+0.5207,+2.1539] pp`).

Local outputs are under
`data/validation/gnps_run16_mega_evaluation_v2_20261102/`; curated reports and
figures are copied to the versioned deliverable directory for the commit.

## Comparable algorithms

The primary table contains only methods that can score the identical
query--reference axis:

1. Full23 v1.3.2;
2. weighted spectral entropy;
3. official fine-tuned DreaMS;
4. SpecEmbedding;
5. MSBERT;
6. Spec2Vec;
7. MS2DeepScore 2.0 as an explicitly labelled auxiliary comparator because it
   is trained for chemical similarity rather than exact identity.

MS2Query is evaluated only in a separate native-system table unless its whole
library-neighbourhood chain is retrained and frozen without benchmark
exposure.  LSM-MS2 has no public reproducible checkpoint/complete reference
library.  JESTR, MIST, ESP and MVP are spectrum-to-structure systems and their
published Recall@k values cannot be mixed with experimental spectrum-to-
spectrum identity retrieval.

GNPS-trained published checkpoints (MSBERT, SpecEmbedding, Spec2Vec and other
public models) will be shown on GNPS only as contaminated/consumed development
comparators.  Their final status is decided on a genuinely new identity-
disjoint source.

### Current 2023--2026 frontier and table eligibility

The main ladder is intentionally task-specific.  A method is not admitted
merely because it emits one scalar for a spectrum pair.

| Method | Native target | Main-ladder treatment |
|---|---|---|
| DreaMS | general MS/MS representation and fine-tuned similarity | same-axis identity comparator |
| MSBERT (2024) | identity and analogue spectral retrieval | same-axis identity comparator on its strict supported domain; GNPS is training-contaminated |
| SpecEmbedding (2025) | identity plus structural-similarity retrieval | official checkpoint run on the complete frozen axis; GNPS is training-contaminated |
| MS2DeepScore 2.0 (2026) | molecular fingerprint-similarity prediction | same-axis **auxiliary** comparator, not relabelled as identity training |
| TransExION (2024) | pairwise structural similarity and analogue interpretation | same-axis **auxiliary** comparator; official gradient-hook inference and its resource cost are preserved |
| MS2Query 1.5.4 (2023) | whole-library retrieval plus neighbourhood-feature reranking | native-system table only; official positive/negative model and library assets are public, but inserting its RF score into the frozen small candidate graph would change the method |
| MS2Query 2 | prototype library/retrieval framework | prototype only; no paper-grade frozen reranker yet |
| LSM-MS2 (2025 preprint) | large-library spectrum-to-spectrum identity retrieval | same-task reported-only frontier because its 1.8M mixed public/internal library and checkpoint are not publicly reproducible |
| PRISM | proprietary large foundation model | reported-only; model, training corpus and executable code are unavailable |
| MSAlign / MSAlign+ (2026) | spectrum-to-molecule representation alignment | official MIT code and MassSpecGym formula/MCES checkpoint archives are public; run only in a separate structure-retrieval table |
| GLMR (AAAI 2026) | generative spectrum-to-molecule retrieval and reranking | separate structure-retrieval table; no reproducible same-axis package located |
| MS2-SMILES AlignNet / MSAN (2026) | spectrum-to-SMILES alignment on CASMI | separate structure-retrieval table; a matching candidate public release exists but its provenance is not yet verified |

MIST, CSI:FingerID/SIRIUS, JESTR, ESP, MVP, MSAlign, GLMR, MSAN and related systems rank molecular
structures conditioned on a query spectrum.  De-novo generators answer yet
another task.  Their published Recall@k values therefore go in separate task
tables rather than being mixed with experimental-spectrum identity retrieval.

The first newly executed official checkpoint, SpecEmbedding, is recorded in
`docs/GNPS_SPECEMBEDDING_SAME_AXIS_RESULT_20261102.md`.  Its same-axis GNPS
Recall@1 is 75.252% on identity-disjoint and 79.282% on formula-disjoint,
versus 88.367% and 89.603% for Full23 v1.3.2.  This is consumed development
evidence under a different candidate protocol than the published
SpecEmbedding table, not a contradiction of that paper and not an external
claim.

## Mega benchmark construction

For every source and polarity/adduct domain:

1. normalize structures and retain a unique identity key and molecular
   formula;
2. remove invalid or conflicting identities before any model score exists;
3. exclude consumed identity, formula and available exact-spectrum hashes;
4. choose query/reference acquisition conditions before scoring;
5. construct the complete 10-ppm candidate graph without injecting a missing
   positive;
6. aggregate all reference spectra belonging to one candidate molecule with
   each method's prespecified aggregation rule;
7. bind every score artifact to a canonical digest of the complete query,
   molecule and reference-spectrum axes;
8. record `N_Q/N_I/N_F/N_E/N_P`, source, polarity, adduct, instrument and
   acquisition-condition strata.

Reference multiplicity is part of the benchmark, not hidden preprocessing.
Molecule-level ranking and ROC/PR use `N_E`; optional spectrum-pair curves use
`N_P` and are reported as a different endpoint.

## Statistical and plotting contract

- The primary ladder order is prespecified Recall@1, not an average rank over
  correlated metrics.
- Macro and micro ROC/PR are both reported; average precision is named
  explicitly even when a historical table calls it AUPRC.
- Formula/source/identity-cluster paired bootstrap intervals use at least
  10,000 replicates.
- Every promoted method is compared query-by-query with Full23, WSE and
  official DreaMS using corrected/introduced counts.
- Top-k and AUC curves are rendered from the same immutable score axis.
- Open-set/no-match is a separate panel with risk--coverage, FDR--coverage,
  matchability ROC/PR and OSCR.  It is not disguised as closed-set Recall@1.
- Runtime, peak host memory and peak GPU memory are reported for the same
  hardware and candidate graph.

## Claim boundary

The current defensible statement is:

> Full23 v1.3.2 is benchmark-leading among the reproducible same-axis methods
> currently evaluated on the consumed GNPS Gold/Silver 10-ppm development
> panels.

It is not yet defensible to write field-wide SOTA or externally validated
SOTA.  That statement requires the primary independent source above and the
missing published-model runs on the exact same candidate graph.

## Immediate execution order

1. build MultiMS2 without opening ranking truth and score the frozen Full23
   package plus all reproducible comparators once;
2. run the materialized MSBERT and SpecEmbedding checkpoints on the frozen
   axes; their GNPS values remain contaminated development comparators;
3. obtain NIST26/NIST23 licence assets or freeze the next eligible MassBank
   release;
4. run the full external ladder once, render Top-k/ROC/PR plots and publish all five
   denominators next to every result.
