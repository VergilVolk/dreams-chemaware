# CALMS v1.0 reproducibility checklist

Date: 2026-11-05

This checklist distinguishes reproducible frozen inference from reconstruction
of every upstream score and statistical analysis from raw MS/MS files.

## 1. Verify the two archives

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `deliverables/CALMS_v1.0_20261105.zip` | 2,386,163 | `7305d38b74dad4edf74c1313e5a3989c72df1cfffd030781336a11dce2be6deb` |
| `deliverables/CALMS_v1.0_evidence_bundle_20261105.zip` | 3,258,704 | `f6995e2acaa746d3a808d066e6129b71e1d692c8fe429c96b12bb2cc1a431648` |

The standalone release contains 21 archive files, including its checksum
ledger. The compact evidence archive contains 62 entries; its internal
`checksums.sha256` registers the other 61 payloads.

Frozen model identities:

```text
candidate-evidence parent
7f1309e2a89ef3640d9b6871a821df6866f2528e629547075d5305ab6707d717

embedding corrector
6603013a04fdc26946445a6ba1ed566d768bac49423046ce3922daac79fe637e

rounded six-decimal golden prediction signature
d4236ddb2da67902aae7668d017d0f04a10a167e63a410d2c78d9a414cea71df
```

## 2. Validate the standalone release

After extracting `CALMS_v1.0_20261105.zip`:

```bash
python -m pip install -r calms_v1/requirements-runtime.txt
python calms_v1/test_calms_frozen_release.py
python calms_v1/examples/run_smoke.py --output-root calms_smoke
```

Expected terminal statuses:

```text
CALMS_FROZEN_RELEASE_PASS
CALMS_RETAINED_SYNTHETIC_SMOKE_PASS
```

The release test verifies package hashes, frozen golden predictions, CPU batch
invariance, candidate-axis failures, embedding-bank provenance and exact parent
fallback when representation evidence is unavailable. The smoke run retains
deterministic synthetic inputs and actual CLI predictions for three queries and
four candidate rows; expected statuses are `scored`, `scored` and
`no_candidates`. The smoke fixture tests release I/O, not scientific
performance.

The runtime requirements are lower bounds (`numpy>=1.24`, `torch>=2.0`) rather
than a complete environment lock. The audited environment used Python 3.11.9,
NumPy 2.4.6 and PyTorch 2.14.0+cpu, so the frozen release test must be run in a
new environment.

## 3. Reproduce the manuscript tables from compact evidence

After extracting `CALMS_v1.0_evidence_bundle_20261105.zip`:

```bash
cd CALMS_v1.0_evidence_bundle_20261105
python -m pip install -r deploy/calms_v1/requirements-runtime.txt
python deploy/calms_v1/test_calms_frozen_release.py
python deploy/calms_v1/examples/run_smoke.py --output-root calms_smoke
python tasks/build_calms_manuscript_main_tables.py \
  --evidence-root . \
  --output reproduced_tables
```

Expected table identities:

| File | SHA-256 |
|---|---|
| `table1_primary_results.csv` | `dd6f46f45282319a2499da41623f33f750966752de33e9b48d92992f613ee6a6` |
| `table2_mechanism_selective.csv` | `90af5f0201b96244b81a27efc39115bc92fcf29ed95f87ebde18acfef71d9479` |
| `table3_two_layer_ablation.csv` | `8b7fc3eb6be022b1d95d9a98c4b18d405e4c64acb5609fed592dd4dddee2c52f` |
| `STATISTICAL_NOTES.md` | `d2e6dc7368c15299eebffce3207dc3a76be181bf13998ceacc58a32b0ed345e2` |

The table command validates and reformats frozen compact reports. It does not
rerun candidate generation, model inference or bootstrap resampling.

## 4. What the compact evidence bundle omits

The compact archive deliberately excludes four large per-query correction
ledgers:

```text
query_ledger_identity_disjoint_official_dreams.csv.gz
query_ledger_identity_disjoint_weighted_spectral_entropy.csv.gz
query_ledger_formula_disjoint_official_dreams.csv.gz
query_ledger_formula_disjoint_weighted_spectral_entropy.csv.gz
```

It also excludes the original raw MS/MS libraries, full candidate-axis
construction inputs, complete score caches and a generic raw-MGF upstream
producer. Those assets are required to rerun every score, bootstrap and
stratified analysis from raw spectra. The portable release itself is a scorer
over caller-provided aligned candidates, seven evidence families and four
embedding banks.

## 5. Reporting boundary checklist

- [ ] Label GNPS and Enveda as consumed development resources.
- [ ] Do not describe GNPS as unopened external confirmation.
- [ ] Do not add the two overlapping GNPS panel denominators or error counts.
- [ ] State that the final 0.9 embedding scale was chosen after GNPS inspection.
- [ ] Keep parent strict-refit contributions separate from the final embedding
      increment.
- [ ] State that selective ranking is closed-set and truth-present.
- [ ] Do not claim open-set false-acceptance control, calibrated match
      probability or FDR control.
- [ ] Do not describe the portable scorer as a raw-MGF-to-annotation workflow.
- [ ] Keep BioAware outside CALMS v1.0. Any future sample-event extension
      requires a separate versioned interface and evaluation with genuine
      candidate-specific events.
- [ ] Require a genuinely unused source and a frozen contamination-exclusion
      protocol before an external-confirmation or field-wide SOTA claim.

## 6. Audit performed for this release

- The standalone archive was independently built twice and was byte-identical.
- The compact evidence archive was independently built twice and was
  byte-identical.
- A fresh extraction of the evidence archive passed the frozen release test and
  retained CLI smoke run.
- All three tables and the statistical notes were regenerated inside that
  extracted directory and matched the bundled files byte-for-byte.
- Ten focused unit/contract tests passed for the release builder, table builder,
  correction analysis and closed-set selective-ranking analysis.
