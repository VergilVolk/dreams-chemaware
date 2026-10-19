# First real cross-study Contrast-to-Contrast query

Date: 2026-10-13  
Status: primary exact-intervention gate failed; frozen development/consumed result

## Question

Can directed, uncertainty-aware study contrasts recover the same genetic
intervention across two independent Arabidopsis LC-MS studies when exact
structures are not required for every source feature?

This is the first execution in this project that actually ranks real study
contrasts using within-study effects.  It is not a synthetic evaluator.

## Studies, denominators and leakage boundary

- Query: `MSV000096214`, seed DDA LC-MS/MS, author-processed structure-hidden
  abundance table; three independent TT7 alleles and one TT4 line versus WT;
  `5 + 5` biological samples per query contrast.
- Candidates: `MTBLS528`, flower LTQ LC-MS, eight flavonoid reference mutants
  versus Col-0; `3 + 3` raw biological replicates per candidate contrast.
- Candidate universe: eight candidates for every query; per-query random Top-1
  is `1/8 = 12.5%`.
- Query denominator: four contrasts, but only two independent intervention
  genes.  The three TT7 alleles are correlated biological replication, not
  three independent genes.
- Shared coordinate: 30 negative-mode nominal-mass bins.  Within-bin feature
  effects are aggregated by median and the standard error includes both
  replicate uncertainty and between-entity dispersion.
- Role: both studies are public, opened and used for method development;
  `development/consumed`, `sealed=false`, `formal=false`.
- Within-study intervention labels are required to estimate each contrast.
  The cross-study same-gene mapping was not used in score construction and was
  opened only for evaluation.  Nevertheless, the whole method family is
  post-development on public data.

MTBLS528's 68 coordinates are an author-selected known-metabolite panel whose
annotations were aided by the same mutant series.  This makes the experiment a
known-MS1 response control, not an unbiased-discovery result.

## Frozen results

| representation | Top-1 | Top-3 | MRR | true ranks |
|---|---:|---:|---:|---|
| signed response + uncertainty | 1/4 | 3/4 | 0.5083 | TT7: 2,1,3; TT4: 5 |
| unsigned response + uncertainty | 0/4 | 4/4 | 0.4167 | TT7: 2,2,3; TT4: 3 |
| terminal chemical state | 1/4 | 4/4 | 0.6250 | TT7: 2,2,2; TT4: 1 |

The primary gate required all three TT7 alleles to rank `f3h/TT7` first and
the TT4 query to rank `tt4` first.  Both clauses failed.  In particular, the
TT4 response ranked the independent `f3h/TT7` contrast first and its exact TT4
counterpart fifth.  This may reflect real shared-pathway response, but the
predefined exact-intervention question is still answered incorrectly.

All candidate contrasts share the same 68-metabolite measurement universe, so
a composition/presence-only retrieval ties all eight candidates.  Directed
effects therefore do create distinguishable scores, but those scores do not
yet provide reliable exact-intervention retrieval and do not beat the terminal
state baseline.

## Scientific decision

The current nominal-mass response transport is **not** a positive Contrast
Atlas result.  It shows that a real contrast object can be constructed and
queried across sources, but also that low-specificity entity alignment
collapses nearby pathway perturbations.  No mass window, precision floor,
feature subset, target panel or relevance definition will be retuned on these
opened ranks.

MTBLS528 cannot test Noise, ChemAware, P2b or WSE because its fragmentation
channel is source-induced full MS1 without isolated precursors.  It also cannot
test unknown-MS/MS increment.  It remains a useful consumed negative/control
shard and an independent MS1 biological-response replication source.

The next positive experiment requires a second study with real biological
replicates, a pre-established comparable intervention, and precursor-resolved
MS2 that permits higher-specificity chemical alignment.  The ongoing
`MSV000096214` raw chain remains useful for qualifying the query-side response,
but cannot rescue this already opened MTBLS528 test.

Machine-readable evidence:

- `data/validation/mtbls528_raw_modality_gate_20261013/report.json`
- `data/validation/mtbls528_flavonoid_mutant_known_response_20261013/report.json`
- `data/validation/msv096214_to_mtbls528_gene_response_query_20261013/report.json`
- `data/validation/msv096214_to_mtbls528_gene_response_query_20261013/candidate_scores.csv`

