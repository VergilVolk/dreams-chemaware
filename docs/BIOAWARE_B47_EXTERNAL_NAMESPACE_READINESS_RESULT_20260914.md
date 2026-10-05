# BioAware B47 external namespace readiness result (2026-09-14)

## Decision

The MSMICA Zenodo deposit is useful, but it is not yet a complete independent
DreaMS/BioAware benchmark.

- The CHDWB head-to-head archive contains a complete development input stack:
  an MS1 feature-abundance table, sample information, and an MGF containing
  2,094 MS/MS spectra. It may proceed only after physical extraction of the
  observable-input namespace, with all validation and algorithm-output members
  excluded.
- The external-validation archive contains six studies/polarity panels with
  m/z, RT and sample-abundance columns, but contains zero raw or standardized
  `.mgf`, `.msp`, `.mzML`, or `.mzXML` inputs. The MS/MS-search CSV files under
  validation are evaluation products and are sealed, not query spectra.
- Therefore no BioAware model is fitted and no performance metric is computed
  at this stage. `pass_to_model_fitting=false` is the scientifically correct
  result.

## Frozen archive identity

| Archive | Bytes | Deposited MD5 | Local SHA256 |
|---|---:|---|---|
| Head-to-head comparison | 392,864,268 | `97510e6a680e7d42d5ca3ca0a360196d` | `67887553720068e18394fb6fb011f8365bc6714f5f3ce99c693fd13dc742dafa` |
| External validation | 95,445,300 | `6adabc70d429e75fac75555bab3cca5e` | `2d6ed0ae51b385b99025398e62939d8902096ba798e603dc5bd0acc8c9b28898` |

The authoritative machine-readable report is
`data/validation/bioaware_b47_namespace_readiness_20260914_v3/report.json`.
The v1 and v2 reports are retained as an audit trail for two corrected header
semantics: Boston's combined `m/z_RT(sec)` column and CHDWB's non-abundance
`name`/`rt` columns.

## Observable-input inventory

### CHDWB HILIC positive

- Feature table: 1,242 columns, including m/z, RT and 1,239 sample-abundance
  columns.
- Sample-information table: present.
- MS/MS: 2,094 spectra in the canonical MetDNA2 input MGF.
- Validation namespace: identifiable by path and never opened by this audit.
- Algorithm-output namespace: identifiable by path and never opened by this
  audit.

This establishes input availability only. It does not yet establish the number
of independent validated identities, DreaMS errors, reaction-reachable errors,
or absence of seed leakage.

### Six external sources

| Source | m/z | RT | Abundance columns | Raw/standardized MS/MS in deposit | Ready for DreaMS retrieval |
|---|---|---|---:|---|---|
| ST003356 HILIC+ | yes | yes | 26 | no | no |
| ST001122 HILIC+ | yes | yes | 43 | no | no |
| ST001236 HILIC+ | yes | yes | 271 | no | no |
| ST002903 HILIC+ | yes | yes | 43 | no | no |
| ST002576 C18+ | yes, combined column | yes, combined column | 176 | no | no |
| ST002576 C18- | yes, combined column | yes, combined column | 176 | no | no |

The absence is archive-specific, not necessarily study-specific: the official
Metabolomics Workbench pages advertise raw archives for all five study
accessions. Their reported sizes make a staged acquisition necessary:

| Study | Official raw archive | Reported size | Reported MD5 |
|---|---|---:|---|
| ST001122 | `ST001122_rawdata.zip` | 860.2 MB | `eedbf0071f945b4d6e01b4b836d3f5ca` |
| ST003356 | `ST003356_Rawdata.zip` | 1.2 GB | `2498d32330e1dbdfd7f06fd793319215` |
| ST002903 | `ST002903_rawdata.zip` | 12.8 GB | `593faba2e270f7bf8fb03ed64167b5cf` |
| ST001236 | `ST001236_rawdata.zip` | 35.3 GB | `703063713804e93f7ecd484c988314be` |
| ST002576 | `ST002576_rawdata.zip` | 131.5 GB | `a8f88b93d4bfad5dc1ee2dc60d966ed6` |

## Next actions

1. Download and checksum ST001122 and ST003356 first. They provide the best
   information-per-byte test of whether the source raw data contain usable
   data-dependent MS/MS and whether raw filenames join to the deposited feature
   tables.
2. Inventory scan levels, precursor metadata, polarity and file-to-sample joins
   without opening the sealed validation tables.
3. Physically extract only the three CHDWB observable inputs into a new hashed
   namespace. Do not extract validation or algorithm outputs into the modeling
   tree.
4. Re-run B47-M0. Proceed to candidate/seed construction only if at least one
   independent external source supplies joinable query MS/MS.
5. Retain the pre-registered B47 gates: at least 1,000 queries, 200 formulas,
   200 DreaMS errors, 100 reaction-reachable errors, zero truth-seed leakage;
   final gain at least +3 pp with positive formula/source CIs and
   `corrected > 2 * introduced`.

## ST001122 raw-source gate (completed after the archive audit)

The first staged raw acquisition passed every truth-blind feasibility gate:

- `ST001122_rawdata.zip` downloaded completely and reproduced the official MD5
  `eedbf0071f945b4d6e01b4b836d3f5ca`.
- The archive contains 43 Thermo `.raw` files. Their basenames match all 43
  sample-abundance columns in the deposited ST001122 feature table exactly,
  with no missing, extra, or duplicate sample.
- A deterministic probe conversion of `IC1_22.raw` produced 5,834 MS2 spectra.
  The extracted raw file is byte-identical to its archive member.
- At 10 ppm, requiring the MS2 RT to lie inside the empirical feature peak
  boundary and the feature abundance to be positive in `IC1_22`, 2,762/5,834
  MS2 spectra join to a detected feature; 2,747 are unique joins covering 1,583
  feature IDs. At 5 ppm, 2,730 match and 2,729 are unique, so the result is not
  being created by a permissive mass window.
- No confirmed-metabolite, validation, or MSMICA-output payload was opened.

Machine-readable reports:

- `data/validation/bioaware_b47_st001122_raw_source_readiness_20260914_v1/report.json`
- `data/validation/bioaware_b47_st001122_probe_join_20260914_v1/report.json`

This changes the decision from “external MS/MS absent from the Zenodo bundle”
to “external MS/MS is recoverable from the official study repository with an
exact sample namespace.” It licenses full truth-blind conversion/join for
ST001122, but still does not license model fitting or a performance claim.

## ST003356 raw-source gate (completed)

The second staged raw acquisition independently passed the same truth-blind
source-feasibility gate:

- `ST003356_Rawdata.zip` downloaded completely and reproduced the official MD5
  `2498d32330e1dbdfd7f06fd793319215`.
- The archive contains 52 Thermo `.raw` files: 26 HILIC-positive and 26
  HILIC-negative. The 26 positive basenames match all 26 sample-abundance
  columns in the deposited ST003356 HILIC-positive feature table exactly, with
  no missing, extra, or duplicate sample.
- A deterministic probe conversion of
  `SEVO001_Pos_128_QE1_HILIC_022.raw` produced 7,164 MS2 spectra. The extracted
  raw file is byte-identical to its archive member.
- No confirmed-metabolite, validation, or MSMICA-output payload was opened.

Machine-readable report:

- `data/validation/bioaware_b47_st003356_raw_source_readiness_20260914_v1/report.json`

Together, ST001122 and ST003356 show that the raw-MS2 absence was a Zenodo
supplement packaging limitation rather than a lack of external query spectra.
The next permitted step is full truth-blind conversion and feature joining of
these two sources; sealed truth remains unopened until the observable query
graphs and all evaluation denominators are frozen.

## Full truth-blind conversion and observable query graph (completed)

Both staged sources were converted in full and joined without opening any
confirmed-metabolite, validation, phenotype, or algorithm-output table:

| Source | Real samples | MS2 spectra | Any feature join | Unique feature join | Frozen queries with at least 2 candidate identities |
|---|---:|---:|---:|---:|---:|
| ST001122 HILIC+ | 43 | 265,453 | 129,729 | 129,103 | 33,829 |
| ST003356 HILIC+ | 20 | 186,704 | 62,260 | 61,525 | 18,147 |
| **Total** | **63** | **452,157** | **191,989** | **190,628** | **51,976** |

ST003356 blank and pooled-QC feature events were excluded before the query
denominator was frozen. The query unit is a real-sample MS1 feature with one
truth-blind representative MS2 spectrum. Ambiguous feature joins are excluded.
For an event with multiple joined spectra, the representative is selected by
peak TIC, then peak count, then original spectrum index; no annotation outcome
enters the choice.

The candidate graph uses the deployment-observable feature m/z only. All
positive-mode MassSpecGym reference spectra within 10 ppm compete, with
`[M+H]+` and `[M+Na]+` grouped by IK14 after retrieval. It contains 51,976
queries and 2,078,709 query-reference rows. Candidate identities per query are
2 minimum, 3 median, 8 at the 90th percentile and 30 maximum. This denominator
is large enough to proceed to frozen DreaMS encoding and truth-blind seed
construction, but it still does not reveal truth coverage or performance.

Frozen graph artefacts:

- `queries.csv.gz`: `c9551bc5d32fcec93405bd80cbe62880e3ac309c3df18e120123fb5fbf4133be`
- `candidate_references.csv.gz`: `64f5940669876db01d17a4b6cd36977d043a930ac471e53d7af3c5018bf67c56`
- `queries.mgf`: `302d3ff12beaa4b09ad3555b64c2682e82336597135be34031359c5f9f80a4a0`

The next execution step encodes both query spectra and candidate reference
spectra with the same frozen official DreaMS encoder. MGF `TITLE` values are
joined explicitly to query IDs; physical file order is never assumed. The
encoder stage is prohibited from computing candidate scores, selecting seeds,
opening truth, or fitting a model.

## Scientific boundary

The deposit proves that a rigorous CHDWB development reconstruction is
possible. It does not yet prove that BioAware has independent external signal.
Using MSMICA's identified-metabolite outputs, validation tables, or MS/MS search
results as model inputs would leak the answer and invalidate the benchmark.
