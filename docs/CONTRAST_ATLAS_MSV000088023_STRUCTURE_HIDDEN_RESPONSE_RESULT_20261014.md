# Contrast Atlas: MSV000088023 structure-hidden HFD response result

Date: 2026-10-14  
Status: **development / consumed; within-study biological-unit qualification**

## Scientific question

Can MS2-linked features without requiring structure annotation form a real,
directed biological response object?  In the public mouse-plasma study
MSV000088023, the female HFD-versus-normal response was queried against the
male HFD-versus-normal response and two real sex-effect distractors; the query
was then reversed.  Female and male responses use disjoint animals.

This is a component qualification for Metabolomic Contrast Atlas.  It is not
cross-study validation and it does not test cross-platform spectral transport.

## Data and denominators

Official dataset: MSV000088023, *DIO_mouse_high_fat_diet_plasma_study*.

- 39 mapped biological samples: 20 HFD and 19 normal diet.
- Cells: female HFD 10, female normal 10, male HFD 10, male normal 9.
- The deposited sample expected as `22_344962_M_normal.mzML` is absent.
- 1,701 XCMS features and 971 unique MGF feature IDs were available.
- A label-blind QC contract retained 212 MS2-linked features: pooled-QC CV
  <=30%, mean biological/single-blank >=5, at least ten original detections in
  all biological samples combined, finite abundance in every biological
  sample, and at least six cleaned MS2 peaks.
- The quantitative object used log2 peak areas followed by per-sample median
  centering over this fixed feature universe.

The MS2 spectra came from three pooled DDA acquisitions, not from individual
animals.  MS1 abundance supplies the animal-level effect; pooled MS2 supplies
the chemical evidence attached to each feature.

## Frozen library-supported versus unresolved partition

The local GNPS Gold/Silver benchmark is already consumed development data.  A
feature was called library-supported only when a positive-mode spectrum within
10 ppm achieved weighted spectral entropy >=0.70 with at least six matched
fragments at 0.02 Da tolerance.

- library-supported: 5/212;
- unresolved under this contract: 207/212.

The five hits are only putative Level-2 support.  Their names are not treated
as confirmed identities.  With fewer than the prespecified 12 features,
`library_supported` abstained rather than producing an unstable known-only
score.

## Retrieval result

The candidate universe was fixed at three contrasts for each query:

1. opposite-sex HFD-versus-normal response, the exact target;
2. female-versus-male under HFD;
3. female-versus-male under normal diet.

Uniform chance for an exact Top-1 is 1/3 per query.  The two query orientations
are correlated views, not independent experiments.

| world | response representation | exact Top-1 | minimum target margin | mean target margin |
|---|---:|---:|---:|---:|
| all, n=212 | standardized signed beta | 2/2 | +0.8241 | +0.8284 |
| unresolved, n=207 | standardized signed beta | 2/2 | +0.8266 | +0.8328 |
| all, n=212 | magnitude only | 2/2 | +0.6016 | +0.6050 |
| unresolved, n=207 | magnitude only | 2/2 | +0.6000 | +0.6046 |
| library-supported, n=5 | any | abstain | NA | NA |

A composition-only method cannot distinguish candidates because all candidates
use the same fixed feature set.  The successful retrieval therefore comes from
the biological effect vectors, not feature presence or raw spectral
composition.

## Animal-level specificity and uncertainty

The focused audit used seed 20261014 and retained the fixed feature universe.

- 5,000 null draws independently permuted diet labels within female and male
  animals while preserving cell sizes.
- 5,000 bootstrap draws resampled animals within the four observed
  sex-by-diet cells.

For standardized signed beta:

| world | empirical p, mean margin | empirical p, minimum margin | bootstrap 95% interval, minimum margin | bootstrap both-query Top-1 |
|---|---:|---:|---:|---:|
| all | 0.000400 | 0.000400 | [0.6545, 0.8499] | 1.000 |
| unresolved | 0.000400 | 0.000400 | [0.6580, 0.8536] | 1.000 |

The discrete `2/2 Top-1` endpoint by itself was not rare under the small
three-candidate design (`p≈0.25`).  Evidence therefore rests on the continuous,
prespecified target margin and its animal-level stability, not on counting two
successful ranks.

The signed-minus-magnitude-only mean-margin difference was +0.2234 for all
features and +0.2282 for unresolved features.  Stratified-bootstrap 95%
intervals were [0.1016, 0.2760] and [0.1040, 0.2797], respectively.  Thus
response direction contributes information beyond change magnitude in this
study.

## Ion-redundancy sensitivity

The 207 unresolved rows are features, not 207 proven analytes.  A post-hoc,
deliberately conservative sensitivity greedily suppressed coeluting features
with abundance correlation 0.90, 0.95, or 0.98 across all biological samples.
Across RT windows of 2, 5, and 10 seconds:

- 153--179 representatives remained;
- all 9 signed-response settings retained both exact Top-1 results;
- minimum signed target margins ranged from +0.7688 to +0.7851.

This argues that the result is not created solely by duplicate coeluting ions.
It is not definitive ion-family annotation: correlation suppression can also
merge distinct co-regulated analytes.

## What is now established

Within this consumed public study, structure-hidden MS2-linked features form a
directed HFD response that transfers across disjoint female and male animals,
survives animal-level resampling and diet-label permutation, and is not
explained by feature composition alone.  Nearly all usable features were
outside the strict library-supported partition, while known-only had
insufficient coverage to answer.

## What is not established

- No cross-study or cross-instrument response retrieval has been demonstrated.
- The unresolved features are not asserted to be novel molecules.
- Exact feature IDs supplied entity correspondence inside one XCMS table; this
  experiment did not require Noise, ChemAware, P2b, WSE, or another spectral
  transport algorithm.
- The experiment does not establish an unknown-information increment over a
  matched known-only panel because known-only correctly abstained.
- The study is consumed development evidence and cannot serve as the final
  sealed external test.

## Next falsifiable experiment

Use two independently acquired urine LC-MS/MS studies with a prespecified
biological relation: mouse HFD/NFD MSV000097426 and human obesity/non-disease
MSV000084112.  The source paper independently reports a cross-study match for
an unknown glucuronide at m/z 396.1684.  That reported anchor must be excluded
from scoring.  The test is whether the remaining response pattern retrieves the
obesity relation against fixed real distractors, and whether all-entity evidence
adds correct coverage over known-only and shuffled-MS2 controls.

This next experiment changes the scientific question from within-study response
qualification to genuine Contrast-to-Contrast transport.

## Reproducible artifacts

- `tasks/analyze_msv000088023_hfd_unknown_response.py`
- `data/validation/msv000088023_hfd_unknown_response_20261014_v3/report.json`
- `tasks/audit_msv000088023_hfd_unknown_response_specificity.py`
- `data/validation/msv000088023_hfd_unknown_specificity_audit_20261014_v1/report.json`
- `tasks/audit_msv000088023_ion_family_sensitivity.py`
- `data/validation/msv000088023_hfd_ion_family_sensitivity_20261014_v1/report.json`

