# Contrast Atlas sparse local response alignment: development result

Date: 2026-10-23

Evidence role: development / consumed

Verdict: **the current 0.1-Da sparse-fragment Top-1 local-alignment route does not establish response-specific retrieval or an unresolved-entity increment.**

## Question

The experiment tested a narrow but necessary method hypothesis: can local MS/MS
correspondences preserve cross-study biological response information that is lost
by a global mean-spectrum representation?

This is not an annotation-count experiment.  Every candidate relation combines a
phenotype-blind chemical graph with study-internal effect estimates.  Registered
condition-family relations were opened only after graphs, observed candidate
scores, and response-profile null scores were frozen.

## Fixed data and denominators

- 12 independent studies and 27 study-level contrasts;
- 91,295 response rows and 38,247 study-specific entities;
- 59,682 sparse 0.1-Da fragment dimensions;
- 198 source-pair/world graphs and 150,440 frozen directional edges;
- 1,944 fixed candidate rows per analysis arm;
- 10 directional registered relation rows, but only 5 canonical relations and
  2 connected biological relation components (HFD and IBD);
- no registered relation has signed metabolite-level truth, and the 638
  unregistered directional contrast pairs are unadjudicated rather than negative.

All, common-policy library-supported, and unresolved worlds use the same fixed
query/candidate universe.  Missing edges remain zero contributions in the full
source-entity denominator.

## Method

For each source pair, square-root-intensity, L2-normalized sparse fragment spectra
were connected to their Top-1 cross-source neighbor when cosine was at least 0.70.
Graphs were frozen before beta, standard error, contrast definitions, or registered
relations were opened.

Each response weight was

`beta / sqrt(beta^2 + SE^2)`.

The first score averaged cosine-weighted products of locally aligned response
weights in both directions.  A chemical-composition score used the same graph but
ignored responses.

The decisive control used 256 deterministic-seed random reassignments of whole
entity response profiles within each study and common identity stratum.  This
preserved the graph, candidate universe, response marginals, missingness, and
within-profile cross-contrast dependence.  Each candidate's observed response
score was converted to a z score relative to its own reassignment distribution.
The vectorized null scorer was required to reproduce the frozen observed scores
before any permutation was accepted.

## Results

### Raw local score

The local score looked better than the earlier global sparse-fragment centroid,
but this was not evidence of response specificity.

| Evidence world | Global median unsigned rank | Local median unsigned rank | Local unsigned Top-1 | Local composition Top-1 |
| --- | ---: | ---: | ---: | ---: |
| all | 21 | 9 | 1/5 | 0/5 |
| known | 12 | 18 | 1/5 | 1/5 |
| unresolved | 21 | 8 | 0/5 | 0/5 |

The one fixed cyclic profile-reassignment sensitivity was already unfavorable:
among the five canonical relations, the observed all-world similar-response rank
was better for 0, tied for 2, and worse for 3 relations.

### Response-profile null calibration

| Evidence world | Similar z Top-1 | Unsigned z Top-1 | Median similar-z rank | Median unsigned-z rank | Unsigned empirical p = 1/257 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 0/5 | 0/5 | 12 | 10 | 0/5 |
| known | 1/5 | 0/5 | 20 | 16 | 0/5 |
| unresolved | 0/5 | 0/5 | 10 | 8 | 0/5 |

Thus the apparent raw local advantage did not survive calibration against the
chemical graph's candidate-specific composition expectation.  The unresolved
world did not produce a registered Top-1 response retrieval.

Several highest-ranked calibrated candidates also crossed unrelated study designs:
the mouse HFD query ranked a lanthanophore-overexpression contrast first; the
pediatric VEO-IBD query ranked a *B. subtilis* contrast first; and the adult IBD
query ranked *B. subtilis* and obesity contrasts above its registered IBD
relations.  Because unregistered pairs lack biological adjudication, these cannot
be called false positives.  They nevertheless show that the current score cannot
be interpreted as a qualified biological discovery engine.

## Scientific decision

The implementation and complete-denominator experiment are successful as a
falsification test; the biological method hypothesis is not supported.

Specifically, these results do **not** establish:

- correct open-set cross-study biological response retrieval;
- an advantage of unresolved over known entities;
- a signed similar/opposite response claim;
- generalization beyond two registered biological relation components; or
- a contribution from Noise, ChemAware, P2b, BioAware, or the final unified model.

The result rules out treating raw local sparse-fragment alignment as the central
Contrast Atlas method.  It also explains why reporting the earlier directional
`3/10` Top-1 count would have been misleading: forward/reverse rows are dependent,
and the apparent signal disappears under response-specific calibration.

## Next method gate

Do not tune the 0.70 threshold, fragment bin width, or neighbor count against these
registered relations.  The next admissible test must replace the chemical evidence
with independently frozen algorithm-agent outputs on the same entity and candidate
universe, while retaining this response-profile null calibration.  A method is
interesting only if it adds registered response retrieval beyond both composition
and reassignment controls, with new independent relation components; otherwise it
remains a better spectral matching layer rather than a biological discovery method.

## Reproducible artifacts

- `tasks/run_contrast_atlas_sparse_local_alignment.py`
- `tasks/calibrate_contrast_atlas_sparse_local_response.py`
- `data/validation/contrast_atlas_sparse_local_alignment_v1_20261023/`
- `data/validation/contrast_atlas_sparse_local_response_calibration_v2_20261023/`
