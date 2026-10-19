# MTBLS528 as an independent TT7/TT4 response-replication source

Date: 2026-10-13  
Status: MS1 response-replication source only; precursor-resolved MS/MS gate failed

## Why this dataset matters

`MTBLS528` is an independent Arabidopsis experiment from a different group,
tissue and instrument than `MSV000096214`.  It contains 72 genotypes or natural
accessions with three biological replicates each (`216` samples).  The final
eight genotype groups are flavonoid reference mutants.  In particular:

- `Col-0`: samples `14_1`--`14_3`;
- `f3´h` (the TT7 enzymatic step): samples `66_1`--`66_3`;
- `tt4`: samples `71_1`--`71_3`.

This supplies two relations fixed by intervention identity before any Atlas
score is computed: TT7-versus-WT and TT4-versus-WT.  The other mutant and
accession contrasts can provide mechanistic and unrelated decoys.  It is a much
better candidate than ChemProp for asking whether a response survives source,
instrument and tissue change.

## Current observable data

- Public RAW and CDF files exist for all 216 samples.
- The nine targeted RAW files for Col-0, f3´h and tt4 are about 66 MB each,
  approximately 594 MB in total.
- Acquisition used a Thermo LTQ-XL linear ion trap with alternating polarity.
- The paper states that RAW contains negative scans at collision energy 0 and
  30 meV plus positive scans.  It also states that the public CDF files contain
  positive and negative data without the in-source-fragmentation channel.
- The authors used standards, fragmentation/RT profiles and mutant comparisons
  to annotate 68 reported metabolites, but the study is not restricted to
  those annotations.

The local metadata snapshot contains 216/216 sample and assay rows and has
official SHA-256 entries for 432 RAW/CDF files.  Its role is now
`development/consumed`; it is not a sealed external test.

## Fragmentation-modality gate: failed for tandem-MS use

One official RAW file, `66_1.RAW`, was downloaded before any cross-study score
was computed.  Its local size is `69,292,990` bytes and its SHA-256 is
`c9fc62e6249b9cb9b1a50882d232c18e1815469418cdcf2cba4a09dc847a1003`, an
exact match to the deposited checksum.  ThermoRawFileParser v2.0.0.0 reports:

- `4,398` scans;
- `4,398` MS1 spectra;
- `0` precursor-resolved MS2 spectra.

The first 300 consecutive scan headers form 100 repeated three-scan cycles:

- negative full MS1, `ITMS - c ESI Full ms [193.00-2000.00]`;
- positive full MS1, `ITMS + c ESI Full ms [193.00-2000.00]`;
- negative source-induced-dissociation full MS1,
  `ITMS - c ESI sid=35.00 Full ms [100.00-2000.00]`.

Thus the fragmentation channel is source-induced full scanning without an
isolated precursor.  It is not a legal input to Noise, ChemAware, P2b, WSE or
any other precursor-conditioned tandem-spectrum comparison.  The remaining
files will not be downloaded for an MS/MS claim.

MTBLS528 remains scientifically useful as an independent **MS1 response**
source.  Its raw biological replicates can estimate TT7-versus-WT and
TT4-versus-WT effects and uncertainty, while `MSV000096214` supplies a distinct
DDA-MS/MS study.  A cross-study result using this pairing must be labelled an
MS1/known-metabolite response replication unless the entity linkage on the
`MSV000096214` side uses additional qualified MS/MS evidence.  It cannot by
itself demonstrate unknown-MS/MS increment.

No cross-study retrieval claim is made until the `MSV000096214` feature-table
and effect-stage gates pass.

Sources: official MetaboLights metadata snapshot and Tohge, Borghi & Fernie,
Scientific Data 2018, DOI `10.1038/sdata.2018.51`.
