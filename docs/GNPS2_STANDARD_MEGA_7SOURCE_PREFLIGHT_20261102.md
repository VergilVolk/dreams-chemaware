# GNPS2 seven-source mega-benchmark preflight

Date: 2026-11-02

## Decision

The seven-source score-blind inventory is materially larger than the original
four-source inventory. It supports a natural 10-ppm Top-k task with 29,332
labelled queries and a separate pairwise identity-verification task with
14,950,709 labelled pairs.

It remains `GNPS2_STANDARD_MEGA_PREFLIGHT_BLOCKED_NOT_EXTERNAL`. The complete
MassSpecGym and MoNA positive/negative consumed-source exclusions are absent
locally, so neither pre-exclusion nor locally available-exclusion counts may be
reported as a final independent benchmark result. No model, embedding,
prediction or evaluation score was read during construction.

## Source inventory

| Source | MGF records | Bytes | SHA-256 | Identity role |
|---|---:|---:|---|---|
| BMDMS-NP | 162,764 | 168,670,066 | `33bdf36f84880fe92649359af97a836bbe23dd0e710692fdea356178eeff0090` | Q1 identity source |
| PNNL-LIPIDS-POSITIVE | 30,574 | 96,509,122 | `c7346f0a3ccba08edea4a1f37e51ce4aa16a36b965c81cc2783e66cf436d9c36` | annotation-only; 2 verifiable structures |
| PNNL-LIPIDS-NEGATIVE | 16,142 | 37,351,288 | `a8be531ec22dfe4601231150a94723f1165cecb72ca02998d29e752c60fcd04d` | annotation-only; no verifiable identity truth |
| MCE-DRUG | 20,955 | 31,168,496 | `6c134cebb1aff50282b4e4cf7a3c6206b2264632c735d66594556b3433472829` | Q3-only identity source |
| BERKELEY-LAB | 24,992 | 36,479,098 | `0b115429997a15313d33c0918453a842f44abc2bf3dd231d97d3bb78f29ba400` | newly added unused source |
| BIRMINGHAM-UHPLC-MS-NEG | 7,198 | 8,656,218 | `aa45a01477f6c6911ea5f2f55c59d38ee6638d4d5f83fa97ec54ed98b8fce311` | newly added unused source |
| BIRMINGHAM-UHPLC-MS-POS | 7,040 | 9,934,405 | `04e5ee42b55bd588da71e041937da8f22c6e03c68408829bd373d1fed3958ef3` | newly added unused source |

After structure, polarity/adduct, precursor, peak-count, identity/formula,
exact-spectrum conflict and duplicate controls, 129,058 spectra remain before
consumed-source exclusion, spanning 8,625 IK14 identities and 6,363 formulas.
Cleaning removed 113 cross-identity exact-spectrum conflicts, 1,000
same-identity exact duplicates and 271 identity/formula conflicts.

## Natural Top-k ranking axis

`N_P` is the number of actual query-to-reference-spectrum comparisons. It is
not the number of queries.

| Stratum | N_Q | N_I | N_F | N_E | N_P | Near queries |
|---|---:|---:|---:|---:|---:|---:|
| Q1/Q2 primary | 19,703 | 1,117 | 477 | 62,617 | 1,127,864 | 18,605 |
| Q3 separate | 7,729 | 3,086 | 1,818 | 26,110 | 49,681 | 4,959 |
| Q1/Q2/Q3 expanded | 29,332 | 4,567 | 2,546 | 108,037 | 1,228,074 | 23,328 |

The natural exact-mass ranking task does not reach ten million comparisons.

## Separate pairwise verification axis

| Stratum | N_Q | N_I | N_F | Positive pairs | Negative pairs | N_pairs |
|---|---:|---:|---:|---:|---:|---:|
| Q1/Q2 primary | 48,272 | 2,619 | 1,914 | 346,544 | 8,092,592 | 8,439,136 |
| Q3 separate | 17,689 | 6,203 | 4,915 | 44,389 | 1,572,348 | 1,616,737 |
| Q1/Q2/Q3 expanded | 60,925 | 8,284 | 6,139 | 356,255 | 14,594,454 | **14,950,709** |

Expanded-axis negatives consist of 711,438 pairs within 10 ppm, 1,884,276
pairs outside 10 ppm but within 0.05 Da, and 11,998,740 pairs outside 0.05 Da
but within 2 Da. Each query contributes at most eight positives and 512
negatives under seed `20261102`.

Known locally available exclusions reduce the expanded identity-disjoint
capacity to 54,555 labelled queries and 11,721,176 pairs. The corresponding
formula-disjoint upper bound is 23,325 queries and 1,684,299 pairs. These
remain incomplete upper bounds until all required exclusions are loaded.

## Reproducibility

- source manifest: `tasks/gnps2_standard_mega_sources_v2.json`;
- preflight entry point: `tasks/preflight_gnps2_standard_mega_benchmark.py`;
- score-blind report: `data/validation/gnps2_standard_mega_7source_preflight_v1_20261102/report.json`;
- materialized pair artifact: `data/validation/gnps2_standard_mega_7source_preflight_v2_20261102/pre_exclusion_q123_pairwise_verification_indices.npz`.

The materialized pair artifact contains only integer row indices, labels and
pair-type codes, not copied spectra. It is 24,571,375 bytes with SHA-256
`dffce092d30890cf7ad6c4485b94077d6802f4b1e2763703274fb59c1e8e0462`.
An independent full-array check confirmed 14,950,709 rows, 356,255 positives,
14,594,454 negatives, zero self-pairs, zero query/reference-axis overlap and
zero duplicate query/reference pairs.

## Unit boundary

- `N_Q`: labelled query spectra;
- `N_I`: identity clusters;
- `N_F`: formula clusters;
- `N_E`: query-to-candidate-molecule edges;
- `N_P`: natural Top-k query-to-reference-spectrum comparisons;
- `N_pairs`: labelled verification pairs.

`N_pairs=14,950,709` must never be written as 14.95 million queries or 14.95
million natural Top-k comparisons.
