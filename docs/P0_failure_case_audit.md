# P0: DreaMS strict-E0 failure-case audit

**Status:** first-pass audit complete  
**Protocol:** MassSpecGym validation fold; `[M+H]+`; same-adduct candidates
within 10 ppm; exact peak-hash duplicates excluded; candidate spectra are
aggregated by the maximum cosine similarity for each 14-character InChIKey.

## 1. Reproduction check

The audit reconstructed the original molecule-level retrieval from cached E0
pair arrays and embeddings.

| Metric | E0 report | P0 reconstruction |
|---|---:|---:|
| Eligible query spectra | 21,163 | 21,163 |
| Top-1 errors | — | 2,109 |
| Recall@1 | 0.90034494 | 0.90034494 |

The exact match required reproducing the original stable tie-breaking rule and
the same floating-point reduction used by the query-level evaluator.

## 2. Main failure concentrations

All enrichment values below are univariate associations. They identify where
errors concentrate but do not by themselves establish causal mechanisms.

| Stratum | Queries | Errors | Error rate | Enrichment vs 9.97% overall |
|---|---:|---:|---:|---:|
| Rule Jaccard >=0.75 | 870 | 244 | 28.05% | 2.81x |
| Morgan Tanimoto >=0.75 | 1,134 | 286 | 25.22% | 2.53x |
| Only one positive spectrum | 675 | 170 | 25.19% | 2.53x |
| Query and best positive from different instruments | 1,735 | 412 | 23.75% | 2.38x |
| More than ten candidate molecules | 385 | 79 | 20.52% | 2.06x |
| Same formula and same Murcko scaffold | 5,796 | 923 | 15.92% | 1.60x |
| Query instrument is QTOF | 5,153 | 749 | 14.54% | 1.46x |

Instrument metadata is missing for 431 queries, of which 150 are errors. This
3.49x enrichment should be treated as a dataset/source-quality warning, not as
evidence that a literal "missing instrument" causes an error.

Of the 2,109 errors, 1,934 (91.7%) select a same-formula wrong molecule. This
large fraction is partly induced by the 10-ppm same-adduct candidate protocol:
75.7% of all queries already have a same-formula strongest negative. The
same-formula error-rate enrichment is therefore modest (1.21x), whereas the
same-scaffold and high-fingerprint-similarity enrichments are stronger.

## 3. MCES case-control analysis

MCES was computed for every Top-1 error and an equal-sized random sample of
correct queries. There were 1,243 unique molecule pairs after deduplication.
Distances above ten may be lower-bound results from thresholded myopic-MCES.

| MCES bin | Error cases | Correct controls | Case-control odds ratio (95% CI) |
|---|---:|---:|---:|
| 0-2 | 889 (42.2%) | 448 (21.2%) | 2.70 (2.36-3.09) |
| 3-5 | 302 (14.3%) | 246 (11.7%) | 1.27 (1.06-1.52) |
| 6-10 | 638 (30.3%) | 702 (33.3%) | 0.87 (0.76-0.99) |
| >10 or lower bound >10 | 246 (11.7%) | 692 (32.8%) | 0.27 (0.23-0.32) |

The dominant error is local structural confusion. DreaMS errors are strongly
enriched when the strongest wrong candidate is only MCES 0-2 from the query;
structurally remote candidates are under-represented among errors.

## 4. Rule interpretation

High Rule Jaccard is enriched among errors rather than guaranteeing identity.
This directly supports the revised role of the rule library:

- Rule overlap must not define positive labels.
- High-rule-overlap, different-identity pairs are useful hard cases.
- MCES and identity determine the target ordering.
- Rules describe why two close structures may produce confusing fragmentation.

The audit has 95.8% coverage for the cached 335-rule vectors. Rule Jaccard is a
diagnostic feature only; no rule value is used to change ground truth.

## 5. Replicate-balanced check

The 21,163 eligible query spectra correspond to 2,159 unique query molecules.
There are 846 molecules with at least one Top-1 error. The mean per-molecule
spectrum error rate is 17.25%, which is higher than the spectrum-weighted 9.97%
because molecules with many easy replicate spectra otherwise dominate the
query-weighted score.

Formula and scaffold outputs are therefore provided twice:

1. query-spectrum-weighted tables for exact correspondence to Recall@1;
2. unique-molecule-weighted tables for chemical interpretation.

## 6. Architecture decision from P0

P0 does not support broad global remapping of the DreaMS space. The observed
failures are concentrated in local, genuinely similar structures and are also
affected by acquisition-domain mismatch and candidate-set size.

The next training pool should prioritize:

1. identity-labelled MCES 0-2 wrong candidates;
2. same-formula, same-scaffold competitors;
3. high Rule-Jaccard identity negatives;
4. cross-instrument positive pairs and controlled peak corruption;
5. formulas and molecules with reproducible errors across multiple spectra.

P0.2-P0.6 have now assigned candidate structural categories to all 736 unique
error pairs and generated a stratified 30-case visual review pack. The automated
categories remain screening labels, not chemical ground truth. The four expert
columns in `manual_review_30.csv` are deliberately blank and constitute the
human sign-off gate before the cases enter training.

The completed method decision is recorded in
`data/validation/e0_failure_audit/P0_FINAL_DECISION.md`. The immediate trainable
baseline is identity supervision with MCES-local hard negatives and
cross-instrument positives. Rule overlap remains a mining/diagnostic feature;
rule decoding and peak-level feedback require the expert review and intervention
tests defined in that report.

## 7. Reproduction command

```powershell
& 'D:\dreams_env\python.exe' tasks\audit_e0_failures.py `
  --compute-mces --mces-time-limit 5
```

Outputs are in `data/validation/e0_failure_audit/`.
