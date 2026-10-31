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

## Next loss defect: listwise probability fitting does not protect Top-1

The first correction exposed a mirror failure: all corrected cases demoted a
high-reference false top candidate, whereas every introduced case demoted a
high-reference true top candidate in favour of a low-reference false one.
Ordinary listwise cross-entropy has no explicit cost for eroding an already
correct parent margin. Merely doubling the loss of parent-correct queries had
effectively no impact because most of their softmax losses were already
saturated.

`train_full23_reference_support_risk_residual.py` therefore compared the old
listwise objective with a direct boundary objective. The latter trains only on
queries within the bounded residual's reachable region: parent errors are
pushed across the Top-1 boundary, while erosion of a parent-correct margin is
charged twice. It also tests within-candidate support stability for each of the
seven experts (`max-second` and `max-mean`) instead of relying only on the
number of reference spectra.

This remains strict formula-OOF on consumed GNPS development. It is a mechanism
experiment, not external evidence.

| Panel | Arm | R@1 | Delta vs Full23 | 95% formula-cluster CI | corrected / introduced | risk net |
|---|---|---:|---:|---:|---:|---:|
| Identity | count + listwise CE | 88.4857 | +0.2365 pp | [+0.0986,+0.3787] | 45 / 19 | +7 |
| Identity | support + boundary Top-1 | 88.4311 | +0.1819 pp | [+0.0815,+0.2906] | 26 / 6 | **+14** |
| Formula | count + listwise CE | 89.6408 | +0.1140 pp | [-0.0577,+0.2865] | 15 / 9 | -3 |
| Formula | support + boundary Top-1 | 89.6217 | +0.0950 pp | [-0.0192,+0.2235] | 8 / 3 | **+2** |

The result is specific and useful:

- weighting ordinary cross-entropy by two did not change either panel;
- adding support features under the old loss was harmful;
- support features became useful only when the loss directly represented the
  Top-1 repair/preservation problem;
- the boundary arm has positive `corrected - 2*introduced` on both panels and
  materially fewer introduced errors, but its formula CI still crosses zero.

Therefore the deployable next experiment is not a larger fusion network. It is
the same low-capacity support residual trained on Enveda OOF with the boundary
loss and then transferred once to GNPS. Full23 v1 remains the release until
that fixed-transfer result exists.

## Large-source fixed-transfer implementation

`train_full23_reference_support_transfer.py` now implements that experiment
without regenerating any upstream artifact:

- training/development source: Enveda identity 25,000 queries plus formula
  9,435 queries, using its existing five global-formula folds and frozen
  repeated-query weights;
- fixed transfer evaluation: GNPS identity 10,995 plus formula 5,261 queries;
- performance arm: count-only listwise residual;
- quality arm: 15-feature reference-support residual with boundary Top-1 loss;
- unchanged parent and evidence: released Full23 v1, Noise, WSE, P2b, MSG,
  Stage-1, Phase-A, FullMSG and V2 artifacts are only read from existing
  caches.

The Enveda residual OOF estimate conditions on already-created parent OOF
scores, so it is development evidence rather than a newly sealed external
claim. The GNPS result is a fixed application of the all-Enveda residual fit.
Both resources are already consumed; neither can support a final external
claim. Their combined scale is appropriate for deciding whether the repair is
stable enough to include in the next frozen candidate.

The runnable entry point is
`tasks/run_full23_reference_support_transfer.sbatch`; its synthetic variable-
candidate and repeated-query contract is
`tasks/test_full23_reference_support_transfer.py`.

Promotion requires the Enveda-fitted correction to retain positive direction
on both GNPS panels, improve the query-balanced endpoint, and not create a
material near-subset or AUPRC loss. Otherwise Full23 v1 remains the release.

## Artifacts

- Local diagnostic:
  `data/validation/full23_v1_reference_coherence_residual_local_20261031`
- Diagnostic implementation:
  `tasks/train_full23_reference_coherence_residual.py`
- Risk-aligned support-loss implementation:
  `tasks/train_full23_reference_support_risk_residual.py`
- Transfer implementation:
  `tasks/train_full23_reference_count_correction.py`
- Server entry point:
  `tasks/run_full23_reference_count_correction.sbatch`
