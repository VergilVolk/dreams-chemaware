# BioAware B47 B1 / U0 formal result

Date: 2026-09-21  
Scope: truth-blind denominator, provenance and nuisance audit only

## 1. Provenance result

The B1 registry closed 18 files (638,372,014 bytes) across the frozen candidate
graph, official shared embeddings, truth-blind seed artifact and U0 output.
All hashes and cross-stage provenance links passed; forbidden truth/phenotype
headers were absent.

- `pass_b1_provenance = true`
- `pass_to_u0 = true`
- `pass_to_b2_exact_event = false`

The B2 block is scientific rather than engineering: 3,243 primary seed rows did
not reach the preregistered minimum of 200 distinct seed identities.

## 2. U0 reference-multiplicity result

Across 51,976 queries, 216,793 query-candidate identities and 2,078,709
query-reference edges:

- reference spectra per candidate: median 3, p90 24, maximum 287;
- `max - expected-single` score lift: median 0.0401, p90 0.1429;
- Spearman correlation between log reference count and max lift: 0.8297;
- max aggregation versus expected-single changed the unique Top-1 identity for
  11,517 / 51,780 both-unique queries (22.24%);
- the identity-flip fraction replicated by source: 22.16% in ST001122 and
  22.39% in ST003356;
- 47.11% of queries had a maximum/median candidate reference-count ratio at
  least 2, and 28.42% had a ratio at least 4;
- the top 1% most exposed candidate identities accounted for 10.74% of all
  query-candidate pairs.

These are large nuisance effects. They do not establish that mean or another
aggregator is more accurate, because U0 did not open truth.

## 3. U0 adduct-pooling result

- 12,178 / 51,976 queries (23.43%) exposed both `[M+H]+` and `[M+Na]+`
  neutral-mass hypotheses;
- among those queries, 23.85% had an absolute best-branch score gap no greater
  than 0.05;
- max versus expected-single changed the winning adduct for 1,435 / 51,780
  both-unique queries (2.77%).

The pooled adduct protocol is therefore materially exposed to branch
competition. U0 cannot identify the true branch.

## 4. Scientific decision

1. B1 provenance is complete.
2. The original candidate-max seed artifact remains immutable and auditable,
   but it is not licensed for B2: it failed identity diversity and its unary
   score is materially reference-count sensitive.
3. Do not lower the 200-identity gate or retune seed thresholds on coverage.
4. The next admissible step is U1: build and freeze a strong spectrum-only
   unary on non-B47 labelled development data, explicitly controlling reference
   multiplicity and adduct branch. Only then may seeds be regenerated
   truth-blind on B47.
5. B47 truth, phenotype and BioAware outcomes remain unopened.

## Claim boundary

Allowed: the B47 denominator is provenance-complete and the current candidate
max protocol has large, source-replicated multiplicity/adduct sensitivity.

Forbidden: any statement that BioAware improves external annotation, that mean
aggregation is superior, that B47 seed precision is known, or that sample-aware
reaction evidence has passed.
