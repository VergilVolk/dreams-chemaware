# MSV000101150 gene-by-diet interaction readiness

Date: 2026-10-14  
Status: COMPLETED dual-polarity development analysis; see result document  
Evidence role if opened: development / consumed

## Scientific role

MSV000101150 is an untargeted Q Exactive lipidomics study of *C. elegans*
wild-type N2 and `tmem-120(hj50)` under normal and high-fat diets. The public
description and associated paper independently state that loss of TMEM-120
alters long-chain lipid abundance and makes animals resistant to HFD-induced
TAG synthesis.

This makes the study useful for a general Contrast Atlas question that is
different from disease classification and from the TT7 epistasis experiment:

> Can structure-unresolved MS2-linked entities recover a prespecified
> gene-by-environment response interaction rather than merely separate four
> sample groups?

It is not a cross-study validation shard. Its value is independent biological
design breadth: a factorial environmental perturbation with a genetic modifier.

## Observable file design

The public MassIVE file index exposes, in each polarity:

- N2 HFD: 3 biological files;
- N2 normal diet: 3 biological files;
- `hj50` HFD: 3 biological files;
- `hj50` normal diet: 3 biological files;
- pooled QC: 3 files.

Thus there are 12 biological sample identities and three QCs, each measured in
positive and negative mode: 30 RAW files total. Negative files are about
72--77 MB each and positive files about 115--119 MB each. The complete download
is approximately 2.9 GB, but it is not authorized by readiness alone.

Primary public sources:

- MassIVE accession: `https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000101150`
- file index: `https://massive.ucsd.edu/ProteoSAFe/dataset_files.jsp?task=db9126cd887d4e709e1dd34fbb302529`
- associated article: `https://pubmed.ncbi.nlm.nih.gov/42098142/`

## Frozen cost gate

Download exactly one negative biological RAW file first:
`N2-high-fat-diet-1.raw` (reported size 73,533,997 bytes). Convert it with the
existing local ThermoRawFileParser without reading any abundance outcome.

Proceed to the remaining files only if all conditions hold:

1. the downloaded byte count equals 73,533,997;
2. the RAW file contains at least 500 precursor-resolved MS2 scans;
3. at least 80% of those MS2 scans have finite positive precursor m/z and at
   least six raw fragment peaks;
4. the file polarity and sample name agree with the public index;
5. the spectra are not full-scan source-induced fragmentation or another
   precursor-free modality.

Failure of any condition stops the MS2 route. Do not download the other 29
files to rescue a failed modality gate.

## Analysis boundary if the gate passes

Before opening feature effects, construct one label-blind feature/MS2 object
using pooled QC for feature coordinates and representative spectra wherever
possible. The fixed biological model is the 2x2 interaction:

`(hj50_HFD - hj50_normal) - (N2_HFD - N2_normal)`.

The published lipid result supplies a positive-control direction for reported
TAG-related chemistry, but the primary structure-agnostic endpoint must use the
complete QC-qualified MS2-linked feature universe. Required worlds are all
MS2-linked, strict library-supported, unresolved, and pooled-QC unresolved.
Known-only must abstain if it lacks the frozen minimum denominator; library
thresholds must not be relaxed.

The experiment may establish that unresolved response objects encode a
gene-by-diet interaction. It cannot by itself establish cross-study transport,
human relevance, a TMEM-120 molecular mechanism, or superiority of Noise,
ChemAware or P2b. Frozen algorithm embeddings enter only if their checkpoint
artifacts and hashes are locally available before effect inspection.

## Modality gate result

The exact frozen file was recovered through the accession's official FTPS
location and matches 73,533,997 bytes (SHA256
`3f81ccf4186bd0505d77baa866c28b45637df55be73a6337a75121479249e2ee`).
ThermoRawFileParser converted it without warnings to a 65,158,678-byte mzML
(SHA256
`9116d67864b6fc6eaad3a997acb131b2fd3ad23a3ac6b30f0e2cea2de65cbd9c`).

The truth-blind scan audit found:

- 11,459 total scans: 1,055 MS1 and 10,404 MS2;
- 10,404 / 10,404 MS2 scans with finite positive precursor m/z;
- 10,302 / 10,404 valid MS2 scans (99.02%) with at least six raw peaks;
- negative polarity for 11,459 / 11,459 scans;
- precursor-resolved `Full ms2 ... @hcd25.00` filter strings, not
  precursor-free source fragmentation.

All six frozen conditions passed. The negative-mode panel was opened first;
after it passed the prespecified value gate, positive mode was opened once under
the same rules as a same-sample analytical replication.

## Negative-arm acquisition and pre-effect freeze

The MassIVE file page was parsed rather than transcribed by hand. All 15 / 15
negative RAW files were recovered and individually byte-verified, totalling
1,120,490,267 bytes. Of these, 12 are biological samples (three independent
files per cell of the 2x2 design) and three are pooled QCs. The original gate
file was hard-linked into the panel and was not downloaded twice. Conversion
to mzML completed for all 15 files. This is acquisition-only evidence;
genotype-by-diet effects were still unopened at this freeze.

The following analysis rules were fixed before effect inspection:

1. Calibrate OpenMS feature detection using the three pooled QCs only. Scan the
   predeclared intensity grid 3,200, 1,600, 800, 400, 200, 100, 50 from strict
   to permissive and choose the highest threshold whose median QC feature count
   is at least 2,000.
2. Define chemical coordinates only from features directly detected in at
   least two of the three QCs, using 5 ppm and a 10-second warped-RT grouping
   window.
3. Quantify every frozen coordinate in all 15 files using the same 5-ppm,
   plus-or-minus-15-second EIC rule. Require EIC evidence in at least two QCs
   and QC CV no greater than 30%. No blank filter can be claimed because this
   deposition contains no blank file.
4. Only after the QC-qualified matrix is written may filenames be decoded into
   N2 / `hj50` and normal / HFD groups. The primary response is the prespecified
   interaction above, estimated on features present in all 12 biological
   samples. Global inference uses a Freedman--Lane residual permutation under
   the additive genotype-plus-diet null (fixed seed 20261014; 4,999 draws),
   with feature-level parametric FDR, permutation FDR and max-T FWER reported.
5. Link MS2 without using effect magnitude. Primary representatives come from
   pooled QCs, using 10-ppm precursor and 15-second warped-RT windows; spectra
   must retain at least six peaks after the frozen cleaning rule. Strict GNPS
   Gold/Silver partitioning remains development / consumed and uses 10 ppm,
   0.02 Da fragments, WSE at least 0.70 and at least six matched peaks.

The QC calibration has now frozen `noise_threshold_int=3200`: QC1, QC2 and QC3
yielded 10,024, 10,314 and 10,419 detected features, respectively (median
10,314; 3/3 QCs; development / consumed; effect-blind). No lower threshold was
opened because the highest predeclared value already passed.

This remains a development / consumed factorial application, not a sealed
external validation result. It can test whether structure-unresolved chemical
entities carry a real gene-by-diet response, but cannot by itself establish
cross-study transport or superiority of an unavailable learned embedding.

Machine-readable report:
`data/validation/msv000101150_thermo_modality_gate_20261014_v1/report.json`.

Additional machine-readable provenance:

- `data/external/msv000101150/neg_download_report.json`;
- `data/external/msv000101150/neg_conversion_all_report.json`;
- `data/external/msv000101150/neg_calibration.json`;
- `data/external/msv000101150/pos_download_report.json`;
- `data/external/msv000101150/pos_conversion_all_report.json`;
- `data/external/msv000101150/pos_calibration.json`.

The completed result, including the positive arm and the failed entity-level
cross-polarity concordance audit, is documented in
`docs/CONTRAST_ATLAS_MSV000101150_GENE_DIET_RESULT_20261014.md`.
