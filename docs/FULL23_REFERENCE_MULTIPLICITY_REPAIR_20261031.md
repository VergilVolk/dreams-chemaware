# Full23 reference-multiplicity repair (2026-10-31)

## Structural defect

Full23 v1 first takes a separate maximum over all reference spectra of each
candidate molecule for Noise, WSE, P2b, MSG, Stage-1, Phase-A and FullMSG. A
candidate with more reference spectra therefore receives more opportunities
to produce an extreme score. The resulting seven maxima can also be supported
by different spectra, so the final candidate vector need not correspond to any
single observed reference spectrum.

This is distinct from ordinary expert fusion and was not addressed by the
previous query-level reliability or embedding-residual experiments.

## Local GNPS mechanism experiment

Dataset role: consumed development. The residual was trained with five strict
formula folds on the formula panel and applied to both panels by the model that
did not train on the query formula. It is not external evidence.

Reference structure:

| Quantity | Identity | Formula |
|---|---:|---:|
| Queries | 10,995 | 5,261 |
| Candidate molecules | 87,518 | 24,401 |
| Candidates with more than one reference spectrum | 59.87% | 57.97% |
| Seven experts choose the same reference, multi-reference candidates | 11.28% | 15.03% |
| Same-reference support among positive multi-reference candidates | 38.18% | 36.74% |
| Same-reference support among negative multi-reference candidates | 7.80% | 9.29% |

Two bounded residuals were fitted:

1. query-centred log reference count only;
2. reference agreement plus the identical count feature.

The agreement coefficient was unstable around zero across the five folds and
changed only two Top-1 outcomes relative to the count-only control. Therefore
the supported defect is reference multiplicity, not reference identity
agreement.

| Panel | Method | R@1 | MRR | Near R@1 | Macro AUROC | Micro AUROC | Micro AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| Identity | Full23 v1 | 88.2310 | 92.3766 | 83.6407 | 94.8400 | 95.3995 | 83.6835 |
| Identity | Count-corrected | **88.4766** | **92.5242** | **83.9831** | **94.9657** | **95.5403** | **84.0554** |
| Formula | Full23 v1 | 89.4507 | 93.6454 | 82.0652 | 94.4271 | 94.8373 | 86.1828 |
| Formula | Count-corrected | **89.6027** | **93.7532** | **82.3370** | **94.5680** | **95.0024** | **86.7272** |

Paired Top-1 outcomes versus Full23 v1:

- Identity: `+0.2456 pp`, formula-cluster CI `[+0.0988,+0.3922]`,
  corrected/introduced `49/22`, `corrected - 2*introduced = +5`.
- Formula: `+0.1521 pp`, formula-cluster CI `[-0.0382,+0.3578]`,
  corrected/introduced `19/11`, risk net `-3`.

The five count coefficients had the same sign and similar magnitude. This is
evidence of max-over-reference opportunity bias, but the correction was fitted
on GNPS and is not deployable as the final model.

## Transfer experiment now implemented

`train_full23_reference_count_correction.py` performs the valid next test:

1. read Enveda Full23 five-fold OOF predictions and frozen candidate panels;
2. fit one bounded query-centred log-reference-count coefficient with the
   existing global formula folds and repeated-query weights;
3. produce Enveda OOF estimates;
4. refit once on all Enveda development data;
5. apply the frozen coefficient to GNPS without refitting or selecting there.

No encoder, P2b, MSG, V2 policy or Full23 checkpoint is retrained. The Slurm
entry point is `tasks/run_full23_reference_count_correction.sbatch`.

Promotion requires the Enveda-fitted correction to retain positive direction
on both GNPS panels, improve the query-balanced endpoint, and not create a
material near-subset or AUPRC loss. Otherwise Full23 v1 remains the release.

## Artifacts

- Local diagnostic:
  `data/validation/full23_reference_coherence_residual_local_20261031`
- Diagnostic implementation:
  `tasks/train_full23_reference_coherence_residual.py`
- Transfer implementation:
  `tasks/train_full23_reference_count_correction.py`
- Server entry point:
  `tasks/run_full23_reference_count_correction.sbatch`

