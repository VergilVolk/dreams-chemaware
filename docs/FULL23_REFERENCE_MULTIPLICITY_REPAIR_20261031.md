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
| Identity | Full23 v1 | 88.2492 | 92.3792 | 83.6934 | 94.8493 | 95.7594 | 84.4745 |
| Identity | Count-corrected | **88.4766** | **92.5182** | **83.9963** | **94.9691** | **95.9050** | **84.8462** |
| Formula | Full23 v1 | 89.5267 | 93.6815 | 82.2011 | 94.4717 | 95.2662 | 87.3952 |
| Formula | Count-corrected | **89.6217** | **93.7577** | **82.3709** | **94.5629** | **95.4349** | **87.8228** |

Paired Top-1 outcomes versus Full23 v1:

- Identity: `+0.2274 pp`, formula-cluster CI `[+0.0810,+0.3701]`,
  corrected/introduced `47/22`, `corrected - 2*introduced = +3`.
- Formula: `+0.0950 pp`, formula-cluster CI `[-0.0934,+0.2842]`,
  corrected/introduced `16/11`, risk net `-6`.

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
  `data/validation/full23_v1_reference_coherence_residual_local_20261031`
- Diagnostic implementation:
  `tasks/train_full23_reference_coherence_residual.py`
- Transfer implementation:
  `tasks/train_full23_reference_count_correction.py`
- Server entry point:
  `tasks/run_full23_reference_count_correction.sbatch`
