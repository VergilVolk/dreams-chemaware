# BioAware B2 direct-context v2 contract (2026-09-05)

## Scientific question

Can sample-local reaction, spectral-network and retention evidence improve a
candidate-specific embedding over frozen DreaMS cosine without relying on
phenotype labels, P2b, or outcome-guided post-hoc routing?

This is not a test of whether reaction neighbours should become the same
universal molecule embedding. The query remains the official DreaMS embedding;
the candidate receives a small, gated, sample-context-dependent update before
cosine scoring.

## Fixed matrix

The primary cell is `full` evidence with leave-one-study-out training. Fixed
attribution controls use reaction+SMN without RT, reaction only, SMN only,
SMN+RT, and RT only. The full model is repeated after removing held-out truth
formulas from every training fold, and a held-out candidate identity is also
forbidden from appearing in a negative slot of any training query. A separate
identity-disjoint run is omitted because, after candidate-list isolation, it
selects exactly the same training queries in all four studies. Hyperparameters
are identical across cells; there is no cell-wise tuning or winner selection.

The strict folds contain as few as 32 context-active training identities. They
are therefore safety/generalisation sensitivities, not independently powered
primary efficacy tests.

The study-level SMN attribution folds contain 95--104 context-active training
identities. The executable uses 90 identities as a computability floor for all
study-level cells and 30 for strict formula sensitivities. These thresholds do
not define efficacy; every actual count and every outer-study effect is
reported, and only `full/study` is the fixed primary.

## Formal recipe

- seeds: 20260830, 20260831, 20260832
- epochs: 40
- batch size: 32
- adapter: hidden 64, update rank 16, delta bound 0.05
- learning rate: 5e-4
- listwise temperature: 0.08
- safety / preservation / gate weights: 4 / 8 / 0.005
- bootstrap: at least 10,000 formula-cluster resamples
- ties count against the positive candidate
- training unit: one randomly cycled query per truth identity per epoch, so
  identities with many spectra cannot dominate the gradient

Only a command carrying `--formal` and matching every frozen value may emit a
formal result. Short local pilots must report `formal=false`.

Every outer-study/seed model is frozen with its evidence scaler, selected
columns, architecture configuration and SHA256. The aggregate report therefore
does not depend on an unrecorded in-memory refit. Before a cell may enter the
matrix, a separate replay process loads only those artifacts and must reproduce
every baseline and adapted rank exactly.

Every cell also emits a frozen flip-mechanism audit. For each corrected or
introduced query it compares the truth with the negative candidate that
actually decided the rank, reports the score changes and all evidence-family
differences, and records whether both identities were exposed in the training
candidate graph. A correction produced by suppressing a candidate with
positive network support is not counted as mechanistic evidence merely because
its Top-1 label improved.

## Gate before direct DreaMS fine-tuning

The frozen-backbone contextual adapter must have a positive formula-cluster CI,
more corrections than introductions, no negative study, nonnegative identity-
and formula-disjoint point effects, and preservation at least 0.995. Only then
is final-block DreaMS fine-tuning justified. Direct fine-tuning, not
distillation, is the preferred next experiment. The final-block encoder and
context adapter must be trained and evaluated together under the same
leave-study-out contract; the universal spectrum-only embedding must also be
reported separately because a sample-specific biological context cannot be
reconstructed from an isolated spectrum at inference.

## Claim boundary

Passing establishes cross-study OOF evidence for contextual candidate
embeddings. It does not establish universal single-spectrum embedding gain,
blind external generalisation, biological mechanism, or SOTA performance.
