# MSV000101166 factorial response gate

Date: 2026-10-14  
Status: STOP; untouched-replicate precursor gate failed  
Evidence role: development / consumed; not sealed external validation

## Scientific question

MSV000101166 is the relative untargeted lipidomics arm of the published
*C. elegans* `tmem-120(hj50)` study. It provides a prespecified 2x2 factorial
design rather than another case-control or drug-retrieval example:

`genotype (WT / tmem-120) x diet (normal / high-fat)`.

If the raw spectra pass the modality gate, the study will test whether a
research-level response object can represent a gene-by-environment interaction
and whether structure-unresolved MS2-linked entities retain that interaction.
This is a design-breadth experiment for the Metabolomic Contrast Atlas method;
it is not a cross-study validation result and it is not a TMEM-120 mechanism
claim.

## Observable design, before opening feature effects

The deposited 411-byte sample map contains exactly 12 biological samples:

- WT, normal diet: 3;
- WT, high-fat diet: 3;
- `tmem-120(hj50)`, normal diet: 3;
- `tmem-120(hj50)`, high-fat diet: 3.

The MassIVE file index contains 24 Bruker `.d` directories, two technical
injections for each biological sample. All are positive-mode acquisitions. The
full indexed deposit is 9,274,527,271 bytes. Only `analysis.tdf` and
`analysis.tdf_bin` are required for the intended raw analysis: 48 files and
6,309,171,200 bytes (671,596,544 bytes of SQLite metadata plus 5,637,574,656
bytes of detector data). Other Bruker visualization/cache files will not be
downloaded.

The public sample map and acquisition metadata were inspected before any
abundance effect. No feature table, interaction effect, or endpoint result has
been opened at this stage.

## Frozen single-injection modality gate

The gate file is the first injection of mutant HFD sample `L10-5-1-2%`:

- `analysis.tdf`: expected 27,860,992 bytes;
- `analysis.tdf_bin`: expected 242,532,352 bytes.

Proceed to the other 23 injections only if all conditions hold:

1. both files exactly match the indexed byte counts;
2. the SQLite sample name is `L10-5-1-2%` and all frames are positive polarity;
3. at least 500 PASEF isolation events have finite positive precursor m/z;
4. at least 80% of the isolation events contain at least six decoded detector
   peaks in their deposited scan interval;
5. the acquisition is precursor-resolved PASEF, not DIA, source-fragmentation,
   or precursor-free full scan.

Failure stops this route. Thresholds must not be relaxed to rescue the study.

## Metadata-only gate result

The downloaded `analysis.tdf` exactly matches 27,860,992 bytes. It contains:

- 10,773 total frames: 3,704 MS1 and 7,069 PASEF MS/MS frames;
- 109,864 PASEF isolation events and 67,361 distinct precursor records;
- 109,864 / 109,864 events with finite positive isolation m/z, spanning
  110.8426--1,354.8379;
- positive polarity for 10,773 / 10,773 frames;
- zero DIA rows and zero ordinary frame-MS/MS rows;
- sample name `L10-5-1-2%`, instrument `timsTOF Pro`.

All 7,069 PASEF frames contain at least 37 frame-level detector peaks. As a
capacity check, 6,699 / 7,069 frames (94.77%) have at least six frame-level
peaks per isolation event on average. This does not prove the required
per-event peak count; the binary gate remains pending until `analysis.tdf_bin`
is decoded.

## Binary result and unit adjudication

AlphaTims 1.1.1 decoded 64,506,210 detector events. Two independent index
calculations (the SQLite frame/scan intervals and AlphaTims quadrupole
segments) agreed exactly for all 109,864 isolation events.

The frozen event-level peak rule failed:

- 60,646 / 109,864 events (55.20%) had at least six detector peaks;
- 1,339 / 109,864 (1.22%) had zero peaks;
- the median was six peaks per event.

This failure is retained and cannot be rewritten as a pass. It also exposed a
modality-specific unit error in the gate: AlphaTims defines a ddaPASEF MS/MS
spectrum by `Precursor`, while a precursor in this file has one or two PASEF
isolation events. Aggregating events by precursor, as a post-result diagnostic,
gave 55,741 / 67,361 spectra (82.75%) with at least six detector peaks. That
diagnostic cannot validate the same file.

One adjudication is frozen before opening another raw file. The untouched
second technical injection of the same biological sample,
`L10-5-1-2%_13_2_4667.d`, will be evaluated using the correct precursor-level
spectrum unit. Its expected files are 27,885,568 and 239,026,176 bytes. It must
have at least 500 precursor spectra, at least 80% with six or more aggregated
detector peaks, at least 95% with one or more peaks, and otherwise satisfy the
same sample, polarity, precursor and ddaPASEF checks. Failure stops the route;
no third-file adjudication or threshold change is allowed.

## Untouched-replicate confirmation result

Both second-injection files matched their indexed byte counts and hashes. The
file contained 10,789 frames (3,710 MS1 and 7,079 PASEF), 109,917 isolation
events, 67,319 precursor-level spectra and 63,307,116 decoded detector events.
All isolation m/z values were finite and positive, all frames were positive
polarity, and the two independent peak-index calculations agreed exactly.

The precursor-level confirmation failed its only quality threshold:

- 46,722 / 67,319 spectra (69.40%) had at least six aggregated detector peaks,
  below the frozen 80% requirement;
- 66,845 / 67,319 (99.30%) were nonempty, above the frozen 95% requirement;
- the median spectrum had eight detector peaks.

Therefore the first file's post-result 82.75% diagnostic did not reproduce in
the untouched technical injection. The route is stopped without downloading a
third file or the remaining 5.77 GB of core raw data. This is a raw-modality
capacity negative for the current structure-unresolved MS2 interaction test,
not a negative result about the underlying TMEM-120 biology.

Machine-readable reports:

- `data/validation/msv000101166_pasef_event_gate_20261014_v1/report.json`
- `data/validation/msv000101166_pasef_precursor_confirmation_20261014_v1/report.json`

## Frozen biological analysis if the modality gate passes

Technical injections will be combined within biological sample before
inference. The feature-level interaction is

`(tmem_HFD - tmem_normal) - (WT_HFD - WT_normal)`.

The associated paper supplies only a broad positive-control expectation: loss
of TMEM-120 attenuates HFD-induced TAG accumulation and alters chain-length
dependence. It does not authorize selecting features by the published result.
Primary analysis must use the complete quality-qualified MS2-linked feature
universe and exact or blocked label permutations over the 12 biological
samples.

Required evidence worlds are all MS2-linked, strict library-supported,
unresolved, and a measurement-matched unresolved control. `Known-only` must
abstain if its denominator is below the frozen minimum. The experiment may
show response sufficiency in an independent factorial design. It cannot by
itself show cross-study transfer, algorithm superiority, human relevance, or a
specific molecular mechanism.

## Primary sources

- MassIVE: `https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000101166`
- article: `https://pmc.ncbi.nlm.nih.gov/articles/PMC13365615/`
- deposited data DOI: `10.25345/C50K26R4D`
