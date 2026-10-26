# Unified candidate integrator: cross-source result and bounded next step

## Scientific status

The unified evidence architecture has produced a real cross-source result.  It
must not be described either as a failed integration or as a final SOTA claim.

The system combines candidate-level evidence from:

- the Noise V1, ChemAware Stage-1, ChemAware Phase-A and Noise-ChemAware
  FullMSG encoders;
- WSE, frozen P2b-on-Noise and the frozen Noise-MSG composite;
- continuous ChemAware V2 evidence relative to its three fixed null controls.

The learned additive arm is a fitted linear candidate ranker over this complete
evidence vector.  It is not an unweighted score average.  BioAware remains a
conditional sample-context stage downstream of this library-ranking model and
is unavailable, rather than zero evidence, when event-level context is absent.

## Results and evidence roles

All numbers below use the fixed candidate axes of the named panels.  Enveda and
GNPS have both been consumed for development; neither can now support a final
external-generalization or SOTA claim.

| Dataset and role | Queries | Official fine-tuned DreaMS | WSE | Additive | Nonlinear | Frozen B |
|---|---:|---:|---:|---:|---:|---:|
| Enveda identity, consumed fusion OOF | 25,000 | 53.100% | 73.028% | **77.616%** | 78.668% | 70.160% |
| Enveda formula, consumed fusion OOF | 9,435 | 67.207% | 85.056% | **86.105%** | 86.900% | 82.830% |
| GNPS identity, consumed fixed transfer | 10,995 | 85.357% | 87.367% | **88.231%** | 87.767% | 88.595% |
| GNPS formula, consumed fixed transfer | 5,261 | 86.809% | 88.272% | **89.451%** | 88.842% | 89.850% |

The official comparator above is the local official **fine-tuned embedding**
checkpoint, not the unfine-tuned SSL backbone.

The main transferable result is the Additive arm:

- Enveda query-balanced unique-query Recall@1: 75.398% WSE to 79.336%
  Additive, **+3.938 pp**.
- Fixed Enveda-to-GNPS transfer: Additive exceeds WSE by **+0.864 pp** on
  identity and **+1.179 pp** on formula.
- Relative to the official fine-tuned DreaMS comparator, Additive is
  **+2.874 pp / +2.642 pp** on the two GNPS panels.

Additive remains slightly below GNPS-trained B on GNPS, but the paired result
does not establish a reliable loss:

| GNPS panel | Additive - B | Corrected / introduced | Formula-cluster 95% CI |
|---|---:|---:|---:|
| identity, n=10,995 | -0.364 pp (-40 queries) | 248 / 288 | [-0.813, +0.091] pp |
| formula, n=5,261 | -0.399 pp (-21 queries) | 142 / 163 | [-1.124, +0.338] pp |

The reverse transfer is decisive in the other direction.  Frozen GNPS full-fit
B, applied without retraining to Enveda, is below WSE by **-2.868 pp** on
identity (95% formula-cluster CI [-3.371, -2.357]) and **-2.226 pp** on formula
([-2.884, -1.554]).  Thus B is the strongest GNPS-domain development ranker,
but it is not a transferable universal anchor.  This comparison is post-hoc on
consumed Enveda and must not be presented as an untouched external test.

## What is and is not established

Established:

1. The multi-module evidence set contains transferable ranking information.
2. A low-capacity learned additive decision layer preserves more of that value
   across Enveda and GNPS than the present nonlinear reliability head.
3. The previous instruction that Additive could never be deployed was an
   implementation restriction, not a scientific result.  The saved checkpoint
   already contains both arms.
4. Replacing the system with GNPS-trained B would sacrifice cross-source
   transfer, despite B's higher GNPS point estimate.

Not established:

1. No untouched final benchmark remains in these two resources; SOTA is not
   yet demonstrated.
2. The nonlinear transfer loss has not been causally attributed to its
   query-level reliability context.
3. The current result proves the joint evidence set, not an independent gain
   for every included module.
4. Library ranking does not by itself establish BioAware or biological-response
   benefit; those require real sample context and independent downstream
   evidence.

## Implemented next step

Two changes are now bounded and executable:

1. `additive_linear_control` can be explicitly selected from the existing
   checkpoint for the fixed application axis.  Omission of the option preserves
   the original nonlinear primary, so the historical experiment is not
   rewritten after seeing results.
2. One nonlinear diagnostic removes exactly 31 columns that are constant for
   all candidates of a query (three query-size fields, seven expert top gaps and
   21 pairwise top-1 agreement fields).  It retains 23 candidate-varying
   evidence columns, including the masked ChemAware V2 top-gap.  Every excluded
   column is checked at runtime to be constant within every query.  No encoder,
   score cache, candidate set, optimization setting or training budget changes.

This diagnostic can test the domain-specific-gating hypothesis.  It cannot by
itself prove that all query context is harmful, and it must not trigger a new
architecture search.

## Execution

Application scoring with the already-fitted Additive arm:

```bash
ENVEDA_INTEGRATOR_ARM=additive_linear_control \
UNIFIED_RANKER_MODEL=data/validation/enveda_unified_candidate_integrator_run_2411772/modeling/unified_candidate_integrator.pt \
sbatch tasks/run_msv000095331_integrated_multiencoder_identity.sbatch
```

The single cached nonlinear diagnostic:

```bash
sbatch tasks/run_enveda_no_query_context_diagnostic.sbatch
```

The diagnostic job reuses the Enveda encoder/MSG caches, continuous chemistry,
GNPS embeddings and GNPS B prediction ledger already present on the server.  It
does not encode spectra, rebuild ChemAware evidence or access a new dataset.

## Promotion rule

Additive is the current transferable application candidate.  The no-context
nonlinear arm replaces it only if the same fixed run shows that it retains the
Enveda gain **and** removes the observed GNPS transfer loss.  Otherwise the
diagnostic ends, Additive remains the decision layer, and the next scientific
test is its effect on fixed entity matches and independently supported response
recovery—not another fusion-head search.
