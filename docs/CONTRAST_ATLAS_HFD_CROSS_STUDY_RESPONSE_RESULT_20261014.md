# Contrast Atlas: first positive cross-study HFD response control

Date: 2026-10-14  
Status: positive `development/consumed` known/reported chemical-class control; not external validation and not an unknown-MS/MS increment result

## Scientific question

Can a query study's **directed within-study metabolic response** retrieve the
same intervention family in an independent study, when chemical composition
alone is uninformative?

This is a Contrast-to-Contrast query.  The score consumes within-study effect
vectors.  It does not infer the relation from spectrum similarity alone.

## Studies, biological units and fixed candidate universe

- Query: Workbench `ST002956`, Calico mouse plasma, adult high-fat diet versus
  adult chow at three repeated time points; nominally `10 + 10` mice per
  contrast.  The same mice are reused over time, so the three queries are not
  independent cohorts.
- Candidate: Workbench `ST001336`, QIMR Berghofer mouse serum, a `2 x 2` design
  with diet (HighFat/Normal) and bile-acid exposure (DCA/water), `11` mice per
  cell.
- Each of five ST001336 analytical panels supplies the same four candidates:
  `diet_water`, `diet_dca`, `dca_normal`, and `dca_hfd`.
- The first two are the predeclared HFD family; the latter two are DCA
  distractors.  Uniform random HFD-family Top-1 probability is `2/4 = 0.5`.
- The five panels reuse the same 44 candidate mice.  Thus `3 x 5 = 15`
  query-panel evaluations are correlated analytical views, not 15 independent
  biological confirmations.

Both studies were opened and used during development.  This experiment is
`development/consumed`, `sealed=false`, `formal_external=false`.

## Response representation and leakage boundary

For each reported chemical key, the study-level response is a log2 mean
difference, with a Welch standard error retained as an available uncertainty
field.  The best exploratory score here is Pearson-equivalent cosine similarity
between per-contrast standardized beta vectors on a fixed shared-key universe.

The biological target family was defined from deposited study designs before
score inspection.  Target labels do not enter the score.  Within-study diet and
DCA labels necessarily enter effect estimation; this is legal response input,
not cross-study truth leakage.

Cross-study correspondence uses depositor-reported names collapsed to a
conservative lipid sum-composition key.  It is not exact structure identity:

- isomers/adducts assigned to one key vote once;
- multiple source measurements are aggregated by median beta and conservative
  maximum source SE;
- `180` ST001336 rows and `35` ST002956 measurement units identified as
  isotope-labelled/internal standards are excluded;
- RefMet alternatives such as `TAG 48:2; TAG 14:0-16:0-18:2` are treated as
  two descriptions of one molecule, not summed twice.

After fail-closed parsing, all 15 query-panel comparisons pass the frozen
minimum overlap of 12 keys.  Shared-key counts range from `27` to `111`, with
median `38`.

## Frozen exploratory result

| representation | unique HFD-family Top-1 | exact `diet_water` Top-1 | median HFD-minus-DCA margin |
|---|---:|---:|---:|
| standardized signed beta | **14/15** | 8/15 | +0.2434 |
| standardized absolute beta | 12/15 | 11/15 | +0.0557 |
| signed reliability | 9/15 | 7/15 | +0.0155 |
| signed tanh(t) | 10/15 | 7/15 | +0.0561 |
| direction only | 5/15 | 4/15 | -0.1053 |
| composition only | 0/15; all tied | 0/15 | 0 |

The sole standardized-beta family failure is query `hfd_t1` against panel
`AN002229`, where `dca_hfd` ranks first.  The exact water-arm HFD contrast is
not expected to win every panel because `diet_dca` is also a legitimate HFD
response in the factorial design; exact-target results are reported but are
secondary.

The unsigned result shows that part of the retrieval signal lies in **which
lipids change strongly**, not solely in sign agreement.  Signed standardized
effects recover two additional query-panel views.  Therefore this experiment
supports response-pattern retrieval, but does not establish that direction is
the only useful component.

## Focused falsification

A single chemical-key permutation is reused synchronously across all three
query times and all five analytical panels, preserving their dependence.  With
5,000 permutations:

- global key permutation: empirical `p = 0.00020` for mean margin and
  `p = 0.0194` for at least 14/15 positive margins;
- within-reported-class permutation: empirical `p = 0.00020` for mean margin
  and `p = 0.00720` for at least 14/15 positive margins.

Thus the result is not reproduced by the marginal distribution of query effect
sizes or by exchanging sum-composition identities within broad reported lipid
classes.

Leave-one-reported-class-out analysis tests 13 classes.  Removing any one
class retains HFD-family Top-1 in 75.0%--93.3% of still-eligible query-panel
views.  `LPC` is the most influential class: removing it reduces the result to
`9/12` eligible views.  The signal is therefore partly LPC-driven but not
strictly dependent on one class.

## Scientific decision

This is the first positive control in the project showing that a real
cross-study biological response object contains retrieval information that a
composition/presence-only representation cannot provide.  It establishes a
necessary premise for Metabolomic Contrast Atlas: study effects can be a
searchable object across independent laboratories.

It does **not** establish:

- unresolved chemical information increment;
- structure-agnostic inference;
- cross-species, cross-tissue or broad disease generalization;
- formal external validation;
- superiority of the final unified Noise/ChemAware/P2b ranker.

The public datasets, key parser and score family were all consumed during
development.  No independent p-value is claimed from the correlated 15 views.
The permutation audit tests chemical-key specificity of this fixed observation;
it does not turn the result into an unopened validation.

## Next experiment

`MSV000088023` is an independent mouse plasma HFD versus normal-diet study with
40 biological samples and pooled DDA deep scans.  One deposited pooled-DDA
mzML has been retrieved (`58,136,220` bytes; SHA-256
`69e4540d757cc1b138b733d365aa6683d343ce4c0543d9c097b05855c8a46191`) and
contains `1,497` MS1 plus `7,462` precursor-resolved MS2 spectra.  The deposit
also exposes sample metadata, an XCMS feature table and a consensus MGF.

The next gate is not another HFD score variant.  It is to build one
phenotype-blind, fixed MS/MS entity graph for MSV000088023 and test whether its
unresolved entities add correct HFD-family retrieval beyond the known/reported
chemical-class control.

## Machine-readable evidence

- `data/validation/st001336_st002956_hfd_cross_study_query_20261014_v4/report.json`
- `data/validation/st001336_st002956_hfd_cross_study_query_20261014_v4/query_panel_summary.csv`
- `data/validation/st001336_st002956_hfd_specificity_audit_20261014_v2/report.json`
- `data/validation/st001336_st002956_hfd_specificity_audit_20261014_v2/leave_one_class_out.csv`
- `data/validation/st001336_st002956_hfd_specificity_audit_20261014_v2/synchronized_key_permutation_null.csv`

