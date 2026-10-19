# MSV000085505 ↔ MTBLS1866 cross-source qualification audit

**Audit date:** 2026-10-19  
**Scope:** metadata and small public tables only; no large spectrum download, feature extraction, response fitting, or retrieval.  
**Evidence role:** public, outcome-open, **development/consumed**. This pair is not an unopened validation asset.

## Decision

**Current verdict: NO-GO for signed-truth registration and NO-GO for an all-entity beta/SE pair.**

The two accessions are genuinely independent source families and both contain positive-mode DDA MS/MS. MSV000085505 is unusually analysis-ready: its public 1,031-feature abundance table and 1,031-spectrum MGF have exact feature-ID correspondence. MTBLS1866 has a clean ISA sample-to-mzML crosswalk for 155 positive-mode DDA files. Those are real strengths.

The pair nevertheless fails admission now for four reasons:

1. **Same disease name is not signed truth.** Neither accession supplies a pre-registered feature-level expected direction shared between studies.
2. **The MSV cohort is not the full randomized trial.** Of 33 deposited COVID files, the public metadata marks 6 as randomized-trial participants, 24 as not randomized, and 3 as unspecified; it contains no colchicine/placebo arm and no usable subject identifier.
3. **MTBLS1866 does not expose its full unknown-feature matrix in the audited stable locations.** The positive MAF has 477 identity-oriented rows, whereas the original paper reports 2,075 total positive-mode identified and unidentified lipid species. The XCMS table used by Wieder et al. is described but is not present in the accession root or their GitHub `Studies/MTBLS1866` directory.
4. **The 109-versus-96 denominator is not reconciled.** The official positive assay has 83 non-critical COVID plus 26 healthy files (109), while the 2026 network paper reports 96 samples. The current archived MAFs/code do not reproduce 96 as the final three-assay intersection.

Accordingly, do not download the entire study and do not fit the primary all-entity response yet. First obtain two small provenance assets listed below.

## Qualification matrix

| Criterion | MSV000085505 | MTBLS1866 | Pair decision |
|---|---|---|---|
| Independent source family | University of Sao Paulo, Brazil; Bruker micrOTOF-Q II | Novara cohort, Italy; Thermo Q Exactive Plus | **PASS** |
| Biological material | Plasma | Plasma | **PASS** |
| Positive-mode DDA | Top-5 DDA in the source-method paper; 68 mzML runs | Top-10 ddMS2 in the original paper; 155 assay-mapped mzML runs | **PASS** |
| Primary groups | 33 hospitalized COVID, 13 healthy | 83 non-critical COVID, 26 healthy | **PASS WITH SCOPE**; severity/treatment populations differ |
| One biological unit per file | 46 distinct biological filenames, but subject IDs are absent/constant | 157 unique ISA Source/Sample Names; 155 unique positive assay rows | MSV **UNRESOLVED** |
| Full abundance table | 1,031 features, all 46 biological files | 477-row positive MAF is not the full 2,075-feature universe | **FAIL** for unbiased all-entity analysis |
| Feature-to-MS2 link | 1,031/1,031 exact `FEATURE_ID` link to MGF | No deposited all-feature spectrum link table found | **FAIL** |
| Pre-defined signed relation | None | None | **FAIL** |

## Exact evidence and denominator reconciliation

### 1. MSV000085505

The official [MassIVE record](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=16326d112dfc42b7b4502e275167bbc9) is Partial Public, DOI `10.25345/C52988`, and reports 144 files. The repository snapshot records the same accession, title, PI/institution, instrument and file count at `data/validation/massive_index.json:9436`. The local Pan-ReDU snapshot begins this accession at `data/reference/pan_redu_snapshot_20261014.tsv:1065800`.

#### Positive DDA file inventory

The [MassIVE file manifest](https://massive.ucsd.edu/ProteoSAFe/dataset_files.jsp?task=16326d112dfc42b7b4502e275167bbc9) contains:

- `ccms_peak/`: 68 mzML files;
- `peak/`: the same 68 runs as mzXML;
- the 68 runs comprise 33 COVID patient, 13 healthy control, 19 solvent blank, and 3 standard-mixture files;
- 136 open-format spectrum files plus 8 parameter/update/metadata assets explain the displayed total of 144 files.

The source-method paper states positive-mode top-5 data-dependent MS/MS acquisition. Pan-ReDU labels the 68 files `Unclassified`, so the DDA assignment comes from the acquisition method, not from that derived classifier. Pan-ReDU contains 158,124 MS2 spectra in total: 80,444 in the 33 COVID files, 31,633 in the 13 healthy files, 39,478 in blanks, and 6,569 in standards.

#### Public feature table and MS2 crosswalk

The update folder `updates/2020-06-05_mwang87_b131d026/other/` contains:

- `COVID19_Brazil_only-plasma-sample_quant.csv`: 1,031 feature rows and 53 non-empty abundance columns (33 COVID, 13 healthy, 4 blanks, 3 standards); every one of the 46 biological files is present;
- `COVID19_Brazil_only-plasma-sample.mgf`: 1,031 MS2 records;
- `gnps_metadata.tsv`: 68 file-level metadata rows;
- `COVID19_Brazil_mzmine-batch.xml`: processing parameters.

Read-only cross-checks found identical feature-ID sets between the CSV and MGF (1,031/1,031), maximum precursor mismatch 0.468 ppm, and maximum RT mismatch 0.0005 s. This side is therefore ready for a feature-level response once its biological-unit and treatment semantics are resolved. One quantification-only blank, `Branco 23_1-50_01_11245.mzXML`, is absent from the 144-file manifest; it does not affect the 46 biological columns but must not be treated as an available raw QC run.

#### Why 46 is not the clinical-trial denominator

The [registered trial](https://ensaiosclinicos.gov.br/rg/RBR-8jyhxh/) assessed 131 people, randomized 75, and analyzed 72; the [trial paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7868202/) reports enrollment from 11 April to 30 August 2020. The deposit instead contains 33 COVID spectra plus 13 external healthy controls. In `gnps_metadata.tsv`:

- randomized-trial flag among the 33 cases: 6 yes, 24 no, 3 unspecified;
- chloroquine flag: 14 yes, 16 no, 3 unspecified;
- there is no colchicine/placebo-arm field;
- `SubjectIdentifierAsRecorded` is unspecified and `UniqueSubjectID` is the same placeholder for every row.

Thus the 46 biological files are a metabolomics subset/panel, not the 75 randomized or 72 analyzed trial participants. A COVID-versus-healthy response could only be named **hospitalized COVID under the observed 2020 treatment mix versus external healthy controls**. It cannot be called a randomized treatment effect or an unconfounded SARS-CoV-2 effect.

### 2. MTBLS1866

The official [MetaboLights accession root](https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS1866/) exposes the ISA sample and assay tables, three MAFs, and mzML directories. The audited API revision was 2 (`revisionDatetime=2025-09-06T16:40:33.410035`; modified 2026-04-14). The local Pan-ReDU snapshot begins this accession at `data/reference/pan_redu_snapshot_20261014.tsv:617937`.

The [original study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7696386/) describes 161 subjects: 84 non-critical COVID, 19 ICU-COVID, 20 symptomatic non-COVID, 12 ICU non-COVID, and 26 healthy. It specifies positive and negative top-10 ddMS2 acquisition.

#### Positive DDA file inventory and crosswalk

The current `s_MTBLS1866.txt` has 157 unique Source Names and 157 unique Sample Names:

- 84 non-critical COVID;
- 26 healthy;
- 19 ICU-COVID;
- 16 symptomatic non-COVID;
- 12 ICU non-COVID.

The difference from the paper's 161 is four symptomatic non-COVID subjects (20 in the paper versus 16 in the deposited sample sheet).

The positive assay table has 155 unique one-to-one assay rows, with `Raw Spectral Data File` blank and `Derived Spectral Data File` pointing to mzML:

- 83 non-critical COVID, 837,843 MS2 spectra;
- 26 healthy, 272,167 MS2 spectra;
- 18 ICU-COVID, 192,570 MS2 spectra;
- 16 symptomatic non-COVID, 171,915 MS2 spectra;
- 12 ICU non-COVID, 116,521 MS2 spectra.

The two sample-sheet rows without a positive assay file are `13` (non-critical COVID) and `DA 13` (ICU-COVID). The primary case-control universe is therefore **109 assay-mapped files = 83 COVID + 26 healthy**, totaling 1,110,010 MS2 spectra. The two primary FTP directories physically contain 111 objects because `AI_p.mzML` and `IRA_p.mzML` are each cross-copied into both directories; the ISA assay table, not directory enumeration, is authoritative. Its 109 primary paths total approximately 9.76 GiB from the FTP index sizes.

#### Public matrices and the 109-versus-96 discrepancy

The positive MAF has 477 feature rows and 155 abundance columns; 217 rows have a non-empty `database_identifier` and 260 do not. It therefore carries some unresolved rows, but it is still not the full all-entity matrix: the original paper reports 2,075 total positive-mode identified and unidentified lipid species and 467 identified lipids. Its identified subset can support a known-focused control only after alias/sample reconciliation.

Wieder et al. 2026 report [96 MTBLS1866 samples](https://pmc.ncbi.nlm.nih.gov/articles/PMC13372841/) and state that a positive-mode XCMS table from the MetaboLights Galaxy API was used for LAMP. However, at GitHub commit [`17b9fd6`](https://github.com/cwieder/MetaNetworkIntegration/tree/17b9fd63c948da5fd754c65003f9308e33a89070/Studies/MTBLS1866), only the sample sheet and three MAFs are present for this accession; no XCMS matrix or 96-row inclusion ledger is deposited there.

The complete public notebooks at that same immutable commit were also inspected through the GitHub blob API. Their `MTBLS1866` object reads the three submitter MAFs, retains `HEALTHY` and `COVID-19`, joins the processed assay tables by their common sample index, and explicitly drops `SA 8`. It contains no Galaxy history ID, dataset ID, XCMS download URL, or separate 96-sample ledger. Replaying that declared membership rule on the deposited tables yields the 94/93 counts below, confirming that the notebook itself cannot reconstruct the LAMP positive-mode denominator of 96.

The live MetaboLights Labs Galaxy API was queried anonymously on 2026-10-19. It exposed zero published or importable histories. Its library registry exposed a public-facing library named `MTBLS Spectral Matrices`, but anonymous library-content enumeration returned HTTP 403, and no `MTBLS1866` history or dataset identifier was discoverable. This is evidence that the matrix is not anonymously recoverable through the currently exposed public API, not evidence that the private Galaxy artifact has been deleted.

Exact current-file matching gives:

- GCxGC MAF: 96 case-control columns matching the sample sheet (76 COVID + 20 healthy);
- LC-MS positive MAF: 97 (74 + 23);
- LC-MS negative MAF: 97 (74 + 23);
- three-assay exact-name intersection: 94 (74 + 20), or 93 after the notebook's declared `SA 8` outlier removal.

Therefore the published 96 is a downstream analysis denominator, not the official positive-DDA file denominator, and it is not reproducible as the final current three-assay intersection. The exact 96 inclusion list and XCMS output must be obtained before using that denominator. For a new positive-mode analysis, the default denominator remains the official assay-mapped 109, followed by a prospectively logged QC exclusion ledger.

## Source independence and truth boundary

Source-family independence passes: the studies differ in country, recruiting hospital, investigator group, instrument vendor/model, chromatography system, and acquisition method details. No shared participants or acquisition batch is plausible from the available provenance.

Biological equivalence does not pass as truth. MSV cases are hospitalized moderate/severe/critical patients under a heterogeneous medication mix; the primary MTBLS cases are explicitly non-critical. Both have age/severity/treatment confounding, and the original MTBLS paper itself notes that COVID participants were older than healthy controls. Consequently:

- `same COVID label` may register a **candidate related contrast**, never a signed ground-truth edge;
- feature-level signs must be evaluated after independent entity/relation matching, not assumed;
- unknown features cannot define their own truth by post-hoc Top-1 agreement;
- any use of this pair is development/consumed, not final external confirmation.

## Minimum acquisition plan

Do these in order; stop before large downloads if either small asset cannot be obtained.

1. **MSV small crosswalk request:** filename → stable subject ID, draw relative to treatment, randomized status, colchicine/placebo arm, and whether the draw was pre-dose/baseline. A de-identified key is sufficient.
2. **MTBLS small processed-asset request:** the exact positive-mode XCMS/MS-DIAL all-feature abundance matrix plus sample mapping and the exact 96-sample inclusion/exclusion ledger used by Wieder et al. A stable Galaxy history/dataset ID also suffices.
3. **Freeze the contrasts before fitting:**
   - MSV: hospitalized COVID under observed treatment mix versus external healthy, with medication/trial sensitivity analyses;
   - MTBLS: non-critical COVID versus healthy; exclude ICU-COVID, symptomatic non-COVID, and ICU non-COVID from the primary contrast.
4. **Only if the MTBLS matrix cannot be recovered:** acquire the 109 assay-listed positive mzML paths only (~9.76 GiB), not all 467 accession files, and create a new all-feature matrix with a frozen QC and feature-to-MS2 linkage contract.
5. Treat the resulting pair as a development query. A signed relation requires a separately pre-registered entity-level direction set or a third independent held-out study; it cannot be manufactured from this pair's disease labels.

## STOP conditions

Stop this route, rather than repair it post hoc, if any of the following holds:

- the MSV biological files cannot be mapped to independent subjects or repeat draws cannot be identified;
- treatment timing/arm cannot be recovered and the proposed claim remains a pure infection or randomized-treatment effect;
- the MTBLS all-feature matrix cannot be linked one-to-one to the 109 assay-listed files and their MS2 evidence;
- the 96 inclusion ledger remains unavailable while a method attempts to use 96 as its denominator;
- fewer than 10 independent biological units remain in either arm after prospectively defined QC;
- the only proposed truth is shared disease naming, spectral similarity, or post-hoc retrieval rank;
- a model-selection or threshold decision reads the cross-study relation outcome before freezing.

Machine-readable adjudication: `tasks/contrast_atlas_covid_plasma_qualification_v1.json`.
