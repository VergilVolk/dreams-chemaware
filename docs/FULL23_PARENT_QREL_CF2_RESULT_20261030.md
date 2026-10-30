# Full23 parent + query-relative CF2 result

## Decision

The CF2 residual is **not promoted**.  `Full23 v1` remains the released
candidate-ranking model.

CF2 was a fixed, low-capacity test of whether explicit cross-family products
could recover information that the 23-feature Additive parent had not used.
It improved GNPS development ranking, but it significantly harmed Enveda and
did not beat the simpler linear query-relative residual on the two primary
source-balanced Top-1 endpoints.  It therefore does not establish a useful
cross-family interaction mechanism.

Both Enveda and GNPS are consumed development resources.  None of the numbers
below is an untouched external or final SOTA result.

## Fixed method

The immutable parent was the released `Full23 v1` Additive model.  CF2 added a
zero-initialized linear residual with:

- 19 candidate-varying main effects;
- 8 representation-by-spectral products;
- 16 representation-by-chemical products;
- 8 spectral-by-chemical products;
- no within-family products, query-constant features, source indicator,
  router, gate, or hyperparameter grid;
- exact no-op chemistry semantics whenever ChemAware V2 was unavailable.

There were 52 trainable parameters including the intercept.  Training used
five global formula outer folds, query-balanced source weights, 20 fixed
epochs, and the two-source MGDA loss.  The parent was reconstructed without
the held-out formula and the residual was evaluated only on that held-out
formula fold.

## CF2 versus released Full23 v1

| Consumed development panel | Queries | Full23 v1 R@1 | CF2 R@1 | Delta (pp) | Formula-cluster 95% CI (pp) | Corrected / introduced | `corrected - 2*introduced` | MRR delta (pp) | Micro AUPRC delta (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Enveda identity-disjoint | 25,000 | 77.592% | 76.840% | **-0.752** | [-1.058, -0.445] | 591 / 779 | -967 | -0.694 | +0.058 |
| Enveda formula-disjoint | 9,435 | 86.084% | 86.137% | +0.053 | [-0.404, +0.495] | 242 / 237 | -232 | -0.082 | -2.755 |
| GNPS identity-disjoint | 10,995 | 88.249% | 88.568% | +0.318 | [-0.000, +0.648] | 168 / 133 | -98 | +0.321 | -3.064 |
| GNPS formula-disjoint | 5,261 | 89.527% | 89.793% | +0.266 | [-0.270, +0.784] | 97 / 83 | -69 | +0.280 | -3.393 |

The source-balanced endpoints reach the same decision:

| Endpoint | Independent units | Top-1 delta (pp) | Formula-cluster 95% CI (pp) | Improved / worsened |
|---|---:|---:|---:|---:|
| Enveda unique query | 30,343 | **-0.592** | [-0.860, -0.320] | 778 / 957 |
| GNPS unique query | 11,019 | **+0.349** | [+0.018, +0.680] | 180 / 142 |
| GNPS identity collapsed by IK14 | 5,534 | +0.316 | [-0.018, +0.646] | 150 / 110 |

Thus the apparent GNPS improvement does not transfer back to Enveda, is not
strictly positive after identity collapse, and is accompanied by negative
risk on all four panels.  Candidate-level AUPRC also falls by about 3 pp on
both GNPS panels.

## CF2 versus the simpler linear query-relative residual

| Consumed development panel | CF2 minus linear R@1 (pp) | Formula-cluster 95% CI (pp) | MRR delta (pp) | Macro AUROC delta (pp) |
|---|---:|---:|---:|---:|
| Enveda identity-disjoint | +0.048 | [-0.208, +0.296] | -0.155 | -0.261 |
| Enveda formula-disjoint | -0.265 | [-0.632, +0.096] | -0.245 | -0.230 |
| GNPS identity-disjoint | +0.055 | [-0.192, +0.299] | +0.081 | +0.218 |
| GNPS formula-disjoint | +0.152 | [-0.226, +0.536] | +0.111 | +0.296 |

On unique-query endpoints CF2 minus linear was `-0.038 pp` on Enveda
(`95% CI [-0.257, +0.180]`) and `+0.059 pp` on GNPS
(`[-0.169, +0.289]`).  The explicit cross-family products therefore did not
produce an independently supported Top-1 gain over the simpler main-effect
residual.

## Optimization diagnosis

The result is not explained by a candidate-axis or parent-model mismatch.
The direct CF2-versus-linear evaluator required exact equality of query
metadata and six frozen score vectors on every panel.  All checks passed for:

- the sequential Full23 outer parent;
- released Full23 v1;
- WSE;
- official fine-tuned DreaMS;
- Noise V1;
- frozen P2b-on-Noise.

The unresolved issue is source conflict during residual fitting.  In the
all-data refit, mean Enveda training loss moved from `0.6369` at epoch 1 to
`0.6931` at epoch 20, while GNPS improved from `0.5362` to `0.4489`.  Mean
source-gradient cosine was `-0.708` initially and `-0.743` at the end.  The
same direction occurred in every outer fold.  This supports an optimization
or objective-conflict diagnosis; it does not justify selecting a favorable
epoch after inspecting held-out results.

## Scientific boundary and next action

1. Keep the released `Full23 v1` package unchanged.
2. Do not promote CF2, the linear query-relative residual, or a GNPS-specific
   branch.
3. Do not claim that learned cross-family interaction has been established.
4. Before another expensive fit, verify whether the MGDA update really gives
   a Pareto descent step under AdamW and stochastic source batches.  Any repair
   must be specified from training behavior, not selected using the four
   held-out result panels.
5. A future development winner would still require one genuinely new source
   before an external or SOTA claim.

## Reproducibility artifacts

Run directory:

`data/validation/full23_parent_qrel_cf2_mgda_local_20261029`

Evaluation directories:

- `data/validation/full23_parent_qrel_cf2_endpoint_eval_20261030`
- `data/validation/full23_cf2_vs_linear_endpoint_eval_20261030`

SHA-256:

- training report: `006f7ef081a59273560beacb85509c0dab1159916b6e102336304f57c7863458`
- outer-OOF predictions: `e155537325f3c59b5a569d35d889c5f45183d63a731f677bd9c118f2e37aeadf`
- all-data checkpoint: `d45fe495970bcfde70530ee6b038b68c5df762652a67899812516092a0727dce`
- CF2-versus-v1 endpoint report: `36b53c76a8c1805383501e8ece46e1eac63b7555d2d18fdbc3971d3c00c24c03`
- CF2-versus-linear endpoint report: `864420d33a1c5d472d2f0050e638d25edd15bd188cff4b5da1696ed7bad067be`
