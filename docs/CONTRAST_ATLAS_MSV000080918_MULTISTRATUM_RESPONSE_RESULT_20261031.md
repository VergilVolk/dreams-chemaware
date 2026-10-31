# MSV000080918 / MSV000080558 multistratum response result

Date: 2026-10-31
Evidence status: **development / consumed**
Result directory: `data/validation/contrast_atlas_msv000080918_multistratum_response_v1_20261031`

## Outcome

This work adds one large, auditable public LC-MS/MS source family to the
Contrast Atlas development set. It does **not** establish an Atlas discovery,
an external validation, unknown-information gain, or an in-house algorithm
gain.

The source contains 451 biological samples, 176 raw cage labels (175 after
whitespace normalization), and 8,515 abundance features. The raw-label
duplication occurs only in the unused six-month B6 panel and does not affect
either retained contrast. The complete GNPS archive contains 9,869 MGF records, but 3,984 have
no usable fragment peaks. After the fixed minimum-three-peak QC, 5,885 spectra
remain. A truth-blind 10-ppm / 12-second / source-sample-overlap join links
1,037 abundance features to 1,934 candidate spectra.

Only two contrasts are retained for response estimation:

- an IgA-knockout versus intact MUP-uPA HFD contrast as the single primary
  development positive control;
- a CD8-knockout versus intact MUP-uPA HFD contrast as a sensitivity-only
  positive control.

Both are already consumed by the source publication. They are not independent
discoveries. MSV000080918 and MSV000080558 are two accessions from the same
parent study, not two independent sources.

## Fixed data and spectrum linkage

| Item | Count |
|---|---:|
| Biological samples | 451 |
| Raw cage labels / whitespace-normalized labels | 176 / 175 |
| Abundance features | 8,515 |
| Raw MGF records | 9,869 |
| MGF spectra with >=3 fragment peaks | 5,885 |
| Empty / below-QC MGF records | 3,984 |
| Abundance features with >=1 spectral candidate | 1,037 |
| Feature--spectrum candidate rows | 1,934 |
| Reciprocal-unique unresolved features | 693 |
| Reciprocal-unique library-supported features | 22 |
| Single-candidate but nonreciprocal unresolved features | 26 |
| Ambiguous unresolved features | 280 |
| Ambiguous features with a library candidate | 16 |

The feature--spectrum join reads precursor mass, retention time, raw abundance
presence, and GNPS constituent-sample membership. It does not read treatment
labels, effect estimates, significance, or the downstream response result.
Calling a feature `reciprocal_unique` additionally requires that its sole
candidate cluster maps back to only that abundance feature. A sole candidate
is not enough by itself.

The raw archive and all analysis inputs are checksum-bound in `report.json`.
The abundance input SHA256 is
`580dbbbd4979b7ec2438178e7bf612813d3ee1147b2096f000acf845eadf1e2f`;
the MGF SHA256 is
`2c414c7ec0dcf2df1130e34bfed4f6f5a148de63f297333af6549b1d53d9c544`.

## Contrast qualification

| Contrast | Role | Qualification |
|---|---|---|
| B6 6-month HFD vs NC | X-Align direct-competitor context only | Rejected: one shared date, one cage per arm after date matching, and nonoverlapping ages. It is not an exact reproduction of X-Align's reported 5-vs-6 subset. |
| B6 10--11-month HFD vs NC | Exploratory candidate | Rejected from response inference: three cages per arm are insufficient for the date-plus-age-adjusted model (one residual degree of freedom). |
| MUP-uPA HFD IgA-KO vs intact | Primary development positive control | Retained. Five raw shared dates; the age-unsupported 2015-09-02 stratum is removed before analysis, leaving four dates, 11 control cages, 15 KO cages, and 26 independent cages. |
| MUP-uPA HFD CD8-KO vs intact | Sensitivity positive control | Retained only as sensitivity evidence: four dates, 4 control arm-cages, 11 KO arm-cages, 12 independent cages, and 3 shared cages. One date has a ten-week arm age difference; a linear age term cannot make this contrast fully identified. |
| CD8-KO antibiotic vs untreated | Failed design | Rejected: one shared date and nonoverlapping age ranges. |
| Oxaliplatin vs untreated | Excluded | Rejected: one date, only four independent cages, and no verified treatment provenance in the source paper. |
| Oxaliplatin + anti-PD-L1 vs untreated | Excluded | Rejected: one date, three independent cages, and no verified treatment provenance in the source paper. |

The original paper and X-Align already consume this source family. X-Align
uses a B6 six-month HFD/NC subset for cross-platform alignment; the present
valid response objects are not an unopened external X-Align benchmark.

## Response model

The source-wide abundance matrix is median-ratio normalized and transformed as
`log2(normalized abundance + 1)`. Mouse rows are aggregated within cage-by-arm
cells. The response model contains a treatment/genotype indicator, shared-date
fixed effects, and centered cage-cell mean age. Standard errors are CR1
finite-sample-corrected and clustered by cage. The t reference degrees of
freedom are the smaller of `cages - 1` and the model residual degrees of
freedom.

Feature eligibility is label blind: a feature must be detected in at least
`max(3 cages, 50% of independent cages)` after collapsing detection across
arms within cage. Arm-specific prevalences are descriptive only. BH correction
is then performed over **all** label-blind eligible abundance features before
the spectral-evidence inner join. This is why the reported BH family is larger
than the 1,037 features with MS/MS candidates.

Collection-date leave-one-out fits are reported separately while retaining the
source-wide normalization learned before the fold split. They are therefore a
fixed-normalization direction-robustness diagnostic, not another p-value or an
independent validation. All four date-omission fits were estimable for both
retained contrasts.

## Development results

| Contrast | Spectrally linked and response-eligible | BH family over all abundance features | q<=0.05 | q<=0.05 and date-LOSO direction stable | Reciprocal-unique unresolved among stable q<=0.05 | Reciprocal-unique library-supported among stable q<=0.05 |
|---|---:|---:|---:|---:|---:|---:|
| IgA-KO vs intact | 372 | 1,489 | 32 | 32 | 22 | 3 |
| CD8-KO vs intact, sensitivity only | 357 | 1,292 | 5 | 5 | 3 | 0 |

These counts are genotype-associated, source-internal development responses.
They do not demonstrate that unresolved information improves a retrieval task,
and they do not validate any molecular identity.

Across the full abundance-level BH families, 186 IgA and 56 CD8 features have
`q<=0.05`. The 32 and 5 values above are the subset remaining after the
precomputed MS/MS-evidence inner join. BH controls each contrast's eligible
abundance family; it does not control multiplicity across contrasts, panel
selection, or the project as a whole.

Two unresolved reciprocal-unique features are q<=0.05, date-LOSO direction
stable, and positive in both contrasts:

| Feature | IgA-KO beta / q | CD8-KO beta / q | Spectrum boundary | Interpretation |
|---|---|---|---|---|
| `280.1374_21.0918` | +9.250 / 0.0121 | +16.249 / 1.11e-9 | 12 fragment peaks | Very early RT; solvent-front or source artefact risk makes it unsuitable as a lead biological result. |
| `580.3417_443.7585` | +13.903 / 9.11e-7 | +12.117 / 0.00144 | 13 fragment peaks | Higher-priority within-source hypothesis, but still lacks structure, independent source replication, and algorithm-dependent identity evidence. |

The latter is a candidate for targeted follow-up, not a discovery claim. Its
large log effect also requires raw-trace and peak-shape review before any
biological interpretation.

## What was explicitly withdrawn during development

- The apparent shared response at `m/z 159.9691` was removed. Its GNPS MGF
  record had zero fragment peaks, so it was not unresolved MS/MS evidence.
- The initial B6 HFD q<=0.05 list was removed after using the correct
  date-plus-age model degrees of freedom. The retained B6 design is too small
  for that inferential model.
- A feature is no longer called unique merely because it has one candidate;
  reciprocal cluster-to-feature uniqueness is required.
- The FDR family no longer depends on treatment-arm-specific prevalence.

These withdrawals are corrections, not negative evidence against the Atlas
question.

## Allowed and prohibited claims

Allowed:

> In the development/consumed MSV000080918--MSV000080558 source family,
> metadata-defined genotype groups show cage-clustered, date- and age-adjusted
> normalized LC-MS feature associations. A subset has nonempty MS/MS evidence
> and remains direction-stable after leaving out each collection date.

Not allowed:

- independent or sealed external validation;
- exact X-Align reproduction;
- causal HFD, IgA, CD8, antibiotic, or treatment effects;
- metabolite identity or pathway flux;
- unknown-information increment;
- Noise, ChemAware, Full23, BioAware, or unified-algorithm increment;
- disease mechanism or treatment-response discovery.

## Scientific use in the Contrast Atlas

The primary IgA response object can now enter the development Atlas. The CD8
object must retain its sensitivity tier and cannot be promoted by a favorable
retrieval result. The useful next experiment is not another source-specific
threshold search. It is to ask, on a fixed cross-study candidate universe,
whether the unresolved response evidence changes a correct research-level
answer relative to known-only, and whether an in-house spectrum model changes
that answer relative to WSE/Official under the same denominator.

`580.3417_443.7585` is the only current source-internal unresolved lead worth
manual raw-signal review. It needs an independent source or prospective assay
before it can support the Atlas biology story.

## Primary sources

- Source study: [Shalapour et al., Nature 2017](https://doi.org/10.1038/nature24302)
- Public raw LC-MS/MS: [MSV000080918](https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000080918)
- Public feature table: [MSV000080558](https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000080558)
- Direct alignment competitor: [X-Align, Chemical Science 2026](https://doi.org/10.1039/d6sc04644d)
