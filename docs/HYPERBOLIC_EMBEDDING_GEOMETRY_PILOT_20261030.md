# Hyperbolic embedding geometry pilot (2026-10-30)

## Question

Does a learned Lorentz-space readout recover identity information from the
Noise-started, ChemAware-continued FullMSG embedding that is not recovered by
an equally trained Euclidean readout or the original 1024-dimensional cosine?

This is a geometry experiment, not a new encoder experiment. It does not test
pure Noise V1 because its full GNPS embedding cache is not present locally.

## Scientific contract

- Dataset: GNPS Gold/Silver, already consumed development evidence.
- Identity-disjoint: 10,995 queries, 87,518 molecule candidates, 175,171
  directed spectrum edges.
- Formula-disjoint: 5,261 queries, 24,401 molecule candidates, 47,724 directed
  spectrum edges.
- Five formula-hash folds; the formula panel supplies training queries. Both
  panels are predicted by the model that did not train on the query formula.
- Frozen 1024-dimensional `noise_chemaware_fullmsg` embeddings.
- Same initialization, 16-dimensional learned readout, optimization budget and
  bounded cosine residual for Euclidean and Lorentz arms.
- The direct full-embedding run uses no random compression.
- The fixed-radius no-learning fact remains important: a same-radius mapping
  is monotone in cosine and cannot change candidate order. This experiment
  therefore learns angular and radial coordinates.

## Full 1024-dimensional result

All values below are percentages. `Pair AUPRC` is the strict 10-ppm directed
query-candidate spectrum-edge endpoint.

| Panel | Method | R@1 | R@2 | R@3 | MRR | Near R@1 | Macro AUROC | Micro AUROC | Micro AUPRC | Pair AUROC | Pair AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Identity | Original 1024 cosine | 86.6940 | 92.8604 | 95.2615 | 91.3085 | 81.5727 | 94.1537 | 94.3236 | 75.7222 | 94.0510 | 74.2213 |
| Identity | Anchored Euclidean | 86.5484 | 92.9150 | 95.1432 | 91.2285 | 81.4146 | 94.1301 | 94.2965 | 75.9403 | 94.0220 | 74.4547 |
| Identity | Anchored Lorentz | 86.5302 | 92.8968 | 95.1432 | 91.2163 | 81.3883 | 94.1298 | 94.3698 | 76.1478 | 94.0979 | 74.7008 |
| Formula | Original 1024 cosine | 88.3102 | 95.1720 | 97.3199 | 92.9473 | 80.4008 | 93.9256 | 93.1551 | 78.2079 | 92.9525 | 76.4418 |
| Formula | Anchored Euclidean | 88.3862 | 95.3431 | 97.3009 | 93.0154 | 80.6386 | 93.9651 | 93.2195 | 78.6200 | 93.0288 | 76.9023 |
| Formula | Anchored Lorentz | 88.3862 | 95.2671 | 97.2819 | 93.0031 | 80.6046 | 93.9929 | 93.3308 | 78.9652 | 93.1385 | 77.2778 |

Paired R@1 comparisons:

| Comparison | Identity delta (formula-cluster CI), corrected/introduced | Formula delta (formula-cluster CI), corrected/introduced |
|---|---:|---:|
| Lorentz vs original cosine | -0.164 pp `[-0.565, +0.239]`, 94/112 | +0.076 pp `[-0.512, +0.663]`, 48/44 |
| Euclidean vs original cosine | -0.146 pp `[-0.547, +0.255]`, 90/106 | +0.076 pp `[-0.460, +0.634]`, 48/44 |
| Lorentz vs Euclidean | -0.018 pp `[-0.272, +0.238]`, 43/45 | 0.000 pp `[-0.337, +0.348]`, 19/19 |

Auxiliary candidate discrimination moved consistently in the Lorentz arm:

- Pair AUPRC versus original cosine: +0.480 pp identity, +0.836 pp formula.
- Micro-candidate AUPRC versus original cosine: +0.426 pp identity, +0.757 pp formula.
- Pair AUPRC versus the parameter-matched Euclidean arm: +0.246 pp identity,
  +0.376 pp formula.

These AUPRC changes do not rescue the primary ranking endpoint. No paired
cluster interval was computed for micro/pair AUPRC, so they remain exploratory.

## Decision

The learned Lorentz readout is not promoted as a replacement for cosine or as
a Full23 production component:

1. It does not beat original cosine on R@1 in both panels.
2. It does not beat the parameter-matched Euclidean readout on R@1.
3. Identity corrected/introduced risk is adverse.

The experiment does reveal a bounded secondary signal: Lorentz geometry
improves candidate-level AUPRC in both panels while leaving Top-1 essentially
unchanged. If the geometry route is revisited, the only justified next test is
to add one frozen Lorentz evidence channel to the existing Full23 integrator
and test fixed cross-source transfer. It is not justified to scan curvature,
radius, dimensions or epochs on consumed GNPS.

## Artifacts

- Implementation: `tasks/train_gnps_hyperbolic_embedding_geometry_pilot.py`
- Full-input result:
  `data/validation/gnps_hyperbolic_fullmsg_full1024_anchored_oof_20261030`
- Earlier 256-dimensional compression control:
  `data/validation/gnps_hyperbolic_fullmsg_anchored_oof_20261030`

