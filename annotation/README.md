# Chem-aware DreaMS annotation platform (`annotation/`)

A reference-free LC-MS/MS spectrum annotation pipeline for non-targeted
metabolomics in medical/biological research, built on the DreaMS self-supervised
MS/MS encoder (Bushuiev et al., *Nat Biotechnol* 2025,
DOI 10.1038/s41587-025-02663-3).

The pipeline turns raw DDA MS/MS spectra into **library annotations with a
measurable false-positive rate and a calibrated probability**, then mines the
unannotated mass ("dark matter") for candidates. Every numeric threshold and
every method traces to a published source; nothing is fabricated.

---

## Pipeline (modules M0–M9, each independently usable and ablatable)

| Module | File | What it does | Literature basis |
|---|---|---|---|
| M0 | `params.py` | Central thresholds, each with a DOI | see `SOURCES` dict |
| M1 | `_inference.py`, `embed.py`, `retrieve.py` | peak preprocessing (top-100, max-norm) → DreaMS official embedding → cosine top-k + precursor-m/z hard constraint | Bushuiev 2025 |
| M2 | `confidence.py` | Schymanski confidence levels 1–5 (platform ceiling: 2a + 3) | Schymanski 2014 |
| M3 | `fdr.py` | target-decoy FDR → per-annotation q-value | Elias & Gygi 2007; Scheubert 2017 |
| M4 | `calibrate.py` | Platt / isotonic calibration of score → P(correct) | Platt 1999; Zadrozny & Elkan 2002; Hoffmann 2022 |
| M5 | `diff.py` | differential abundance (Fisher exact + BH on DDA spectral counts) | Fisher 1925; Benjamini & Hochberg 1995 |
| M6 | `pathway.py` | hypergeometric ORA + mummichog-style m/z enrichment | Li 2013 |
| M7 | `darkmatter.py` | dark-spectrum clustering + candidate leads | Cao 2025; Schymanski 2014 |
| M8 | `ablation.py` | module on/off ladder to quantify each stage's effect | — |
| M9 | `cli.py` | end-to-end command line | — |

## Quick start (CPU)

```bash
# 1. embed query spectra (hdf5, as written by dreams.utils.data.MSData)
python -m annotation.cli embed --kind query \
    --hdf5 data/msv100574/Metabolomics/neg/PF_1.hdf5 \
    --hdf5 data/msv100574/Metabolomics/neg/HF_1.hdf5 \
    --out data/msv100574/embeddings/met_neg

# 2. embed a reference library (MGF with SMILES/INCHIKEY/PEPMASS)
python -m annotation.cli embed --kind library \
    --mgf data/models/mona_neg_full.mgf \
    --out data/models/mona_neg_dreams_emb

# 3. annotate (retrieve + confidence + optional FDR + calibration)
python -m annotation.cli annotate \
    --query-dir data/msv100574/embeddings/met_neg \
    --library-dir data/models/mona_neg_dreams_emb \
    --out data/msv100574/annotation/met_neg \
    --fdr --library-mgf data/models/mona_neg_full.mgf \
    --calibrate
```

Each stage writes its own artifacts (`embeddings.npy`, `manifest.csv`,
`annotations.csv`, `report.json`) so the pipeline is auditable and ablatable.

## Verified results (Met/neg, MONA-neg library)

**Ablation ladder** (same query set, 13,770 spectra):

| Stage | annotation rate | FP proxy (m/z mismatch) |
|---|---|---|
| raw cosine (cos ≥ 0.7) | 0.248 | 0.764 |
| + m/z constraint (±20 ppm) | 0.059 | 0.000 |
| + target-decoy FDR (q ≤ 0.01) | *pending — needs 1:1 decoys (68 min CPU)* | 0.000 |

The m/z constraint removes 76% of raw-cosine top-1 hits whose precursor m/z
disagrees — the raw cosine over-annotates ~4×.

**Calibration** (library leave-one-out self-retrieval, 36,663 spectra):
70.5% of spectra recover a same-compound (InChIKey14) hit; correct hits average
cosine 0.884 vs. 0.781 for incorrect — the score separates, but raw cosine is
*not* a probability, hence M4.

**Dark matter**: 94.1% of Met/neg spectra have no confident annotation — the
majority are candidates for M7 mining, never claimed as identified.

## Honest limitations

- **Level 1 / Level 4 are never emitted** (require a reference standard / a
  formula predictor like SIRIUS, which this platform does not consume).
- **M6 ships no compound→pathway database** (HMDB/KEGG/Reactome is external;
  must be supplied as a DataFrame — avoids fabricated mappings).
- **M5 differential analysis is semi-quantitative** (DDA MS2 spectral counts,
  less precise than MS1 peak-area / DIA), and requires per-group replicates.
- **FDR decoy generation requires re-embedding the whole library.** Measured
  throughput is ~9 spectra/s on CPU, so 36,663 decoys ≈ 68 min. Decoy embeddings
  are cached to `data/models/mona_neg_decoy_emb.npy` (see
  `tasks/run_fdr_met_neg.py`). The FDR logic and full link are verified; only the
  full 1:1 decoy run is deferred. A subset of decoys is *not* a valid substitute —
  with 500 decoys every q-value collapses to the floor (~1e-4), so the filter has
  no resolution.
