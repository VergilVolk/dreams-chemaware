# BioAware NetID external validation contract (2026-08-28)

## Why this source was initially locked

MTBLS13729 contains only 21 currently evaluable identity queries. It exposed a
real failure mode, but it is too small for model selection. The dependency-
corrected reaction-context score was therefore frozen before opening any NetID
manual-curation outcome.

NetID v1.0 is used because its public release contains a yeast negative-mode
peak table, reference libraries, network outputs, and a manual-curation table.
The associated paper deposits the raw LC-MS data under MassIVE MSV000087434.

## Feasibility amendment (before outcomes were opened)

The source preflight found that the yeast-negative NetID benchmark and the
DreaMS structure-retrieval task are not the same endpoint. The 314 manual
curations evaluate MS1 peak/formula/ion-relation assignments. The public yeast
collection is explicitly MS1 and contains no named yeast targeted-MS2 files;
the large targeted-MS2 collection in MSV000087434 is liver data.

Therefore NetID remains a valid external benchmark for the separate MS1 global
formula/feature-graph module, but it is blocked as the primary DreaMS/BioAware
structure-ranking benchmark. This amendment was made without reading values in
`manual_curate.csv`. The structure-ranking benchmark is moved to the MetDNA3
Exploris release, which provides 20 MS2 panels and Level-1 standard annotations.

## Frozen method

The BioAware increment is not unrestricted graph diffusion. For each candidate:

1. construct leave-query-out reaction paths from independently supported seeds;
2. inspect all non-currency compounds on the seed side of each Rhea reaction;
3. group complete paths by seed compound;
4. group incomplete paths by their shared missing-source signature;
5. count only the strongest path in each dependency group, then combine groups;
6. permit an override only at a low DreaMS spectral margin and otherwise abstain.

This rule was fixed after it removed the MTBLS13729 alpha-ketobutyrate
convergence error while retaining the separate guanine-to-guanosine development
correction. Neither example is external evidence.

## Blindness and leakage rules

`FDR_example/manual_curate.csv` is the held-out outcome table. During source
installation and contract lock, code may record only its path, byte size, and
SHA-256; it must not read cell values. Candidate generation, seed construction,
thresholds, and dependency grouping are frozen before the table is opened.

Author NetID assignments cannot be used as BioAware seeds for the same query.
Phenotype, pathway enrichment, and differential abundance are forbidden from
identity ranking. Every query must be leave-query-out and leave-truth-identity-
out at evaluation.

## Primary decision gate

The external result passes only if all are true:

- at least 200 identity-evaluable queries;
- corrected Top-1 errors exceed introduced errors;
- the clustered 95% CI for Recall@1 improvement is entirely above zero;
- MRR is non-decreasing;
- the real reaction graph beats the 95th percentile of degree-preserving decoys.

Formula/scaffold strata, intervention coverage, and all transitions must be
reported. Failure is reported as failure; no post-outcome threshold adjustment
is permitted on this benchmark.

## Stages after locking

1. Install and hash-seal the exact Zenodo v1.0 release.
2. Build the candidate graph without reading manual-curation outcomes.
3. Freeze candidate rows, seeds, scores, and all thresholds.
4. Open the manual-curation table once and run the paired external evaluation.
5. Only after the external decision, download raw MSV000087434 spectra if a
   DreaMS MS2 reconstruction is needed for the manuscript-grade comparison.

The external test concerns a downstream annotation expert. It does not claim a
new DreaMS embedding, human colorectal biology, MSI Level 1 identification, or
metabolic flux.
