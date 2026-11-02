# GNPS2 Standard mega-benchmark: score-blind preflight (2026-11-02)

## Scientific status

This is a metadata-only capacity audit. No model, embedding, prediction, or
evaluation score was read. The current status is
`GNPS2_STANDARD_MEGA_PREFLIGHT_BLOCKED_NOT_EXTERNAL` because three required
consumed-source exclusion assets are absent locally:

- MassSpecGym development;
- MoNA positive consumed;
- MoNA negative consumed.

Consequently, none of the counts below is an external-test result. GNPS
Gold/Silver overlaps that are locally available are recorded, but the missing
required exclusions prevent release of a post-exclusion panel.

The terms used throughout are intentionally distinct: `N_Q` is the number of
labelled query spectra, `N_E` is the number of query-to-candidate-molecule
edges, `N_P` is the number of natural Top-k query-to-reference-spectrum
comparisons, and `N_pairs` is the number of labelled pairs on the separate
verification axis. A large `N_pairs` value must never be reported as an
equally large query count or Top-k comparison count.

## Source integrity and identity usability

| Source | MGF records | Strict eligible before overlap | Identity use | MGF SHA-256 |
|---|---:|---:|---|---|
| BMDMS-NP | 162,764 | 81,648 | Q1 identity source | `33bdf36f84880fe92649359af97a836bbe23dd0e710692fdea356178eeff0090` |
| PNNL-LIPIDS-POSITIVE | 30,574 | 2 | annotation-only; two sparse verifiable records | `c7346f0a3ccba08edea4a1f37e51ce4aa16a36b965c81cc2783e66cf436d9c36` |
| PNNL-LIPIDS-NEGATIVE | 16,142 | 0 | annotation-only; no verifiable IK14 | `a8be531ec22dfe4601231150a94723f1165cecb72ca02998d29e752c60fcd04d` |
| MCE-DRUG | 20,955 | 17,684 | Q3-only identity source | `6c134cebb1aff50282b4e4cf7a3c6206b2264632c735d66594556b3433472829` |

The BMDMS-NP file is complete at exactly 168,670,066 bytes. Official GNPS2
slim metadata were joined one-to-one for both PNNL libraries. They contain no
usable structure coverage for PNNL-negative and only 2/30,574 structure-labelled
records for PNNL-positive, so PNNL is not promoted to the identity benchmark.

After structure, precursor/adduct, polarity, peak-count, identity/formula
consistency, exact-spectrum conflict, and duplicate controls, the joint strict
inventory contains 98,939 spectra, 4,923 IK14 identities, and 4,098 formulas.
The cleaning removed 28 cross-identity exact-spectrum conflicts, 282
same-identity exact duplicates, and 85 identity/formula conflicts.

## Unopened Top-k capacity forecast

These use complete 10-ppm candidate windows. `N_P` is the number of
query-reference spectrum comparisons, never the number of labelled queries.

| Quality stratum | N_Q | N_I | N_F | N_E | N_P | N_P >= 10M |
|---|---:|---:|---:|---:|---:|---|
| Q1/Q2 primary | 17,406 | 863 | 361 | 56,135 | 1,139,056 | no |
| Q3 separate | 2,251 | 652 | 502 | 5,162 | 16,969 | no |
| Q1/Q2/Q3 expanded | 21,828 | 1,776 | 1,086 | 68,301 | 1,147,641 | no |

The four GNPS2 sources therefore do **not** support a ten-million-comparison
Top-k identity panel. MCE Q3 is kept separate from the Q1/Q2 primary panel.

## Separate pairwise identity-verification axis

To study representation quality at larger scale without changing the Top-k
task, a second score-blind pair axis was constructed. Positives are distinct
query/reference spectra with the same IK14. Negatives always share polarity
and normalized adduct and are prioritized by precursor proximity. Per query,
at most 8 positives and 512 negatives are retained with seed `20261102`.

| Quality stratum | N_Q | N_I | N_F | Positive pairs | Negative pairs | Total pairs | >= 10M |
|---|---:|---:|---:|---:|---:|---:|---|
| Q1/Q2 primary | 40,750 | 2,089 | 1,546 | 296,959 | 6,990,415 | 7,287,374 | no |
| Q3 separate | 9,138 | 2,644 | 2,494 | 33,363 | 654,817 | 688,180 | no |
| Q1/Q2/Q3 expanded | 51,666 | 4,663 | 3,909 | 336,251 | 10,410,751 | 10,747,002 | yes |

For the expanded axis, negative pairs comprise 585,891 within 10 ppm,
1,299,441 outside 10 ppm but within 0.05 Da, and 8,525,419 outside 0.05 Da
but within 2 Da. Thus, the ten-million scale is a **pairwise verification**
resource, not ten million queries and not ten million exact-mass Top-k
comparisons.

The index-only NPZ contains no spectra and is 17,352,856 bytes:

- path: `pre_exclusion_q123_pairwise_verification_indices.npz`;
- SHA-256: `1fc260c761191c1753154e275d13e2d2b90e678aa794ed1aa40cf2df10346882`;
- 10,747,002 unique pairs;
- zero self-pairs;
- disjoint query/reference axes;
- zero positive identity mismatches or exact-spectrum duplicates;
- zero negative identity, polarity, adduct, or 2-Da-window violations.

This artifact is explicitly pre-exclusion and remains blocked from any external
claim until all required consumed sources are supplied and the indices are
rebuilt after exclusion. Locally available exclusions already flag 11,791
identity-overlap records, 61,485 formula-overlap records, and 1,705 exact-spectrum
overlap records, so the final independent panel will be smaller.

## Reproducibility

Canonical local output:
`data/validation/gnps2_standard_mega_preflight_v4_20261102/`.

Implemented entry points:

- `tasks/preflight_gnps2_standard_mega_benchmark.py`;
- `tasks/gnps2_standard_mega_sources_v1.json`;
- `tasks/test_preflight_gnps2_standard_mega_benchmark.py`.

Validation completed:

- Python compilation passed;
- synthetic contracts passed;
- source files were checked for mutation during scan and hash;
- the full materialized pair index was independently checked for uniqueness,
  label consistency, domain matching, mass-window compliance, and checksum.
