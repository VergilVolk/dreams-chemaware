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

## Completed bounded diagnostic

The no-query-context diagnostic completed as run `2413878`. It removed exactly
31 query-constant columns and retained 23 candidate-varying evidence columns.
Inputs, candidate axes, folds, upstream scores, optimization settings and seed
were unchanged.

The reduced nonlinear head preserved Enveda OOF Recall@1 while improving fixed
GNPS transfer by `+0.400/+0.570 pp` relative to the old nonlinear head. This
recovered about 81--82% of the old nonlinear-to-Additive transfer gap. It did
not outperform Additive on either GNPS panel, and its error-exchange risk and
micro AUPRC were worse. The 23-feature Additive arm is therefore the current
transferable application candidate. Full results and the leakage/axis audit
are recorded in `UNIFIED_NO_QUERY_CONTEXT_DIAGNOSTIC_RESULT_20261026.md`.

## Execution

Application scoring with the already-fitted Additive arm:

```bash
ENVEDA_INTEGRATOR_ARM=additive_linear_control \
UNIFIED_RANKER_MODEL=data/validation/enveda_no_query_context_diagnostic_run_2413878/modeling/unified_candidate_integrator.pt \
sbatch tasks/run_msv000095331_integrated_multiencoder_identity.sbatch
```

The completed cached nonlinear diagnostic was launched with:

```bash
sbatch tasks/run_enveda_no_query_context_diagnostic.sbatch
```

The diagnostic reused the Enveda encoder/MSG caches, continuous chemistry,
GNPS embeddings and GNPS B prediction ledger. It did not encode spectra,
rebuild ChemAware evidence or access a new dataset.

## Promotion rule

The promotion condition was not met. Additive remains the decision layer, and
the next scientific test is its effect on fixed entity matches and
independently supported response recovery--not another fusion-head search.
