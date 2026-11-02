# Mega identity ladder: implemented result and external-benchmark boundary

Date: 2026-11-02

## Result status

The strict mega evaluator has been implemented and run on the two consumed
GNPS development panels.  It evaluates an immutable candidate-molecule axis,
binds every score cache to the panel SHA-256 or canonical axis digest, and
reports query, identity, formula, candidate-edge and spectrum-pair
denominators separately.  This run is a reproducibility/system test and an
updated development ladder; it is not a new external confirmation.

| Panel | Role | `N_Q` | `N_I` | `N_F` | `N_E` | `N_P` | near queries |
|---|---|---:|---:|---:|---:|---:|---:|
| GNPS identity-disjoint | development / consumed | 10,995 | 5,534 | 4,204 | 87,518 | 175,171 | 7,592 |
| GNPS formula-disjoint | development / consumed | 5,261 | 2,662 | 2,210 | 24,401 | 47,724 | 2,944 |

`N_Q` is labelled queries, `N_E` is query--candidate-molecule edges and `N_P`
is actual query--reference-spectrum comparisons.  These quantities are not
interchangeable.

## Full23 v1.3.2 metrics

| Metric | Identity-disjoint | Formula-disjoint |
|---|---:|---:|
| Recall@1 | **88.3674%** | **89.6027%** |
| Recall@2 | **94.0155%** | **95.7613%** |
| Recall@3 | **95.9436%** | **97.6620%** |
| Recall@5 | **97.5080%** | 98.9736% |
| Recall@10 | **98.8813%** (tie) | 99.6959% |
| Recall@20 | 99.7544% | 99.9620% |
| MRR | **92.4522%** | **93.7402%** |
| macro-query AUROC | **94.9251%** | **94.6532%** |
| micro-candidate AUROC | **95.8538%** | **95.3840%** |
| micro-candidate average precision | **84.4136%** | **87.4153%** |
| near Recall@1 | **83.8251%** | **82.3030%** |

Full23 ranks first among the 16 actually evaluated methods for Recall@1, MRR,
near Recall@1, macro/micro AUROC and micro AP on both panels.  It does not lead
every high-k endpoint: Noise V1 leads identity Recall@20 and formula
Recall@10/20; P2b-on-Noise leads formula Recall@5.

Relative to weighted spectral entropy:

- identity: `+1.0005 pp`, 223 corrected and 113 introduced; formula-cluster
  familywise interval `[+0.4637,+1.5548] pp`, identity-cluster interval
  `[+0.4410,+1.5646] pp`;
- formula: `+1.3305 pp`, 116 corrected and 46 introduced; formula-cluster
  interval `[+0.5207,+2.1539] pp`, identity-cluster interval
  `[+0.5148,+2.1568] pp`.

The intervals use 10,000 cluster bootstrap replicates and a familywise
multiple-endpoint adjustment.  Relative to official fine-tuned DreaMS,
Recall@1 improves by `+3.0105 pp` and `+2.7941 pp`.

## Recall@1 ladder for the 16 completed methods

| Method | Identity | Formula |
|---|---:|---:|
| Full23 v1.3.2 | **88.367%** | **89.603%** |
| weighted spectral entropy | 87.367% | 88.272% |
| entropy raw | 87.312% | 88.139% |
| P2b-on-Noise frozen | 87.131% | 88.253% |
| denoising search | 87.131% | 88.025% |
| P2b sqrt cosine | 87.040% | 88.158% |
| P2b official frozen | 86.976% | 88.025% |
| P2b unweighted entropy | 86.976% | 88.158% |
| neutral-loss sqrt cosine | 86.812% | 87.873% |
| Noise V1 | 86.558% | 88.063% |
| Spec2Vec GNPS public | 86.294% | 87.436% |
| Spec2Vec 2026 retrained | 86.085% | 87.246% |
| official fine-tuned DreaMS | 85.357% | 86.809% |
| cosine greedy | 85.039% | 87.170% |
| modified cosine | 85.030% | 87.151% |
| MS2DeepScore 2.x public | 79.118% | 81.467% |

MS2DeepScore is an auxiliary learned chemical-similarity score, not an
exact-identity-specific model; its row must be interpreted accordingly.

## Curves and exact artifacts

The versioned deliverable contains PNG/PDF ladder, Top-k, ROC and PR plots,
plus machine-readable reports, summary tables and paired comparisons:

`deliverables/mega_sota_ladder/gnps_run16_20261102/`

Exact compressed ROC/PR coordinates and per-query ledgers remain in the local
validation output:

`data/validation/gnps_run16_mega_evaluation_v2_20261102/`

The PR figure reports non-interpolated average precision and labels the older
project term “historical AUPRC” as an alias; it does not silently substitute a
trapezoidal area.

## Latest-method completion state

- SpecEmbedding official code and the authors' official 512-D checkpoint are
  local and pass a real finite/unit-norm forward test.  Full GNPS encoding is
  in progress.  Its model used GNPS/MoNA/MTBLS1572, so its GNPS row will be
  labelled contaminated development evidence.
- MSBERT official code and the complete historical Git-LFS 512-D checkpoint
  are local and forward-tested.  It will run after SpecEmbedding to remain
  within the 16-GB host-memory budget.  It was trained on filtered GNPS.
- MS2Query 1.5.3 changes retrieval and uses full-library neighbourhood
  features, so it belongs in a separate native-system table rather than being
  inserted into this frozen small candidate graph.
- MS2Query2 remains a prototype.  LSM-MS2 is a direct recent competitor but
  has no public reproducible checkpoint/complete reference library; its
  published numbers are reported-only and cannot be mixed into this ladder.
- TransExION is a possible auxiliary spectrum-pair similarity comparator, not
  an exact-identity-specific method.

The machine-readable method registry is
`tasks/mega_sota_methods_v1.json` and the standard public-checkpoint adapter is
`tasks/public_sota_embedding_adapter.py`.

## Why a “10-million labelled-query benchmark” is not claimed

No verified public, bulk-downloadable small-molecule experimental MS/MS
resource currently supplies more than ten million independently labelled
identity queries.

- NIST26 contains 3,219,115 tandem spectra and 68,635 compounds but is sold
  through licensed distributors: <https://www.nist.gov/programs-projects/tandem-mass-spectral-library>.
- METLIN 960K represents more than 960,000 authentic standards and roughly six
  million experimental spectra, but the bulk local database is separately
  licensed: <https://pubs.acs.org/doi/10.1021/acs.analchem.5c08031>.
- FASSTrecords exposes billion-scale raw spectra and spectral matches, but the
  matches are algorithmic retrieval results rather than independent identity
  truth: <https://www.nature.com/articles/s41587-026-03082-8> and
  <https://zenodo.org/records/18199544>.
- MultiMS2 is genuinely useful for independent multi-energy/multi-fragmentation
  stress, but has 43,728 spectra and 2,899 compounds:
  <https://github.com/zamboni-lab/MultiMS2>.

The scientifically valid large-scale target is therefore more than ten
million `N_P` comparisons over a substantial number of independently labelled
queries, never a relabelled claim of ten million ground-truth queries.  The
preferred final source is NIST26 identities genuinely new relative to NIST23,
after exclusion of every consumed identity, formula and available exact
spectrum hash.  The open independent supplement is MultiMS2.  FASSTrecords is
reserved for throughput/coverage stress rather than identity-accuracy truth.

## Defensible claim today

> Full23 v1.3.2 is benchmark-leading among the reproducible same-axis methods
> actually evaluated on the consumed GNPS Gold/Silver 10-ppm development
> panels.

This is not yet a field-wide or independently externally confirmed SOTA claim.
That requires the pending SpecEmbedding/MSBERT rows and a genuinely new final
identity source such as the licensed NIST26 release increment or a prospective
blind standards panel.
