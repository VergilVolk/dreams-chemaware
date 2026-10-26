# Fixed unified Additive: contribution matrix and application handoff

## Decision

The ranking architecture is now fixed. The deployable library-ranking model is
the 23-feature `additive_linear_control` in run `2413878`; no further fusion-head
search is part of this round. The next computation measures which evidence
families make the fitted combination non-replaceable and applies that exact
model to the fixed biological candidate axis.

Enveda-180 and GNPS are both consumed development resources. The work below can
establish component contribution and deployment behavior, but it cannot by
itself establish final external generalization or a field-wide SOTA claim.

## Strict contribution matrix

The full arm is not retrained. Its checkpoint and OOF candidate ledger are
loaded directly from:

`data/validation/enveda_no_query_context_diagnostic_run_2413878/modeling`

The runner validates the input hashes, 23-feature contract, five formula folds,
optimizer budget, checkpoint hash, OOF-ledger hash and every saved candidate
axis before using that reference. It then trains only four reduced variants:

| Reduced refit | Retained evidence | Scientific question |
|---|---|---|
| official fine-tuned DreaMS + WSE | official score and percentile, WSE score and percentile, recomputed reliability summaries | Can a learned combination of existing baselines explain the full result? |
| without ChemAware representation group | Noise V1, WSE, P2b, MSG and V2 | What is added jointly by Stage-1, Phase-A and Noise-ChemAware FullMSG? |
| without ChemAware V2 | all seven spectral/representation experts | What is added by continuous candidate-specific chemical evidence? |
| without P2b/MSG | Noise V1, WSE, the three ChemAware-derived encoders and V2 | What is added by the frozen local-reranker family? |

For every reduced arm, raw percentiles, top-1 vote fraction, mean percentile and
percentile dispersion are recomputed from the retained experts. This is a true
same-fold refit, not inference-time column masking. All four arms use the same
candidate graphs, formula folds, epochs, optimizer, anchor penalty and seed
schedule; no hyperparameter or architecture scan is performed.

The resulting full-minus-reduced differences estimate evidence-family
contribution under this fixed low-capacity ranker. They are not single-module
causal effects, and a redundant group is allowed to have no measurable
increment.

One server submission runs all four Enveda refits and the fixed GNPS transfer:

```bash
sbatch tasks/run_enveda_unified_additive_ablation_matrix.sbatch
```

The expected output is
`data/validation/enveda_unified_additive_ablation_run_${SLURM_JOB_ID}` and
contains the reduced-arm checkpoint, Enveda OOF ledger, GNPS candidate ledgers,
paired comparisons and a checksum manifest. The exact run-2413878 full model is
referenced rather than copied or refitted.

## V2 absence is a different question

Two analyses are deliberately kept separate:

1. `without_chemaware_v2` is a newly fitted reduced model and measures V2 group
   contribution under the fixed training protocol.
2. `full_additive_v2_unavailable_replay` keeps the frozen full checkpoint and
   changes only the six V2 fields to their registered semantic no-op before
   normalization. It measures deployment sensitivity when candidate structures
   are unavailable; it is not a contribution ablation.

The final-refit Enveda replay on already-consumed data produced:

| Panel | Observed V2 | V2 unavailable | Difference | Corrected / introduced | Top-1 changed |
|---|---:|---:|---:|---:|---:|
| identity, 25,000 queries | 77.728% | 77.036% | +0.692 pp | 703 / 530 | 1,618 (6.472%) |
| formula, 9,435 queries | 86.179% | 85.681% | +0.498 pp | 122 / 75 | 237 (2.512%) |

This replay used labels only to summarize a post-result deployment condition.
It is final-refit rather than OOF, accessed no biological application outcome,
and cannot establish V2 causality or external performance. Its reproducible
implementation is `tasks/analyze_enveda_integrator_v2_deployment_condition.py`;
the local report is
`data/validation/enveda_integrator_v2_deployment_condition_local_20261026/report.json`
and its SHA256 is
`e291ff7895608c40bd5c3d72540f9e81e14738210067e34df911b8a154d0bbf9`.

## Fixed application handoff

The application scorer now defaults to the exact run-2413878 checkpoint and
explicitly selects `additive_linear_control`. It never discovers a newer model
by directory timestamp. The fixed application universe remains 38,628 query
blocks and 22,046 candidate pairs, including zero-, one- and multi-candidate
blocks. No model is allowed to add a candidate.

The application candidate table has no legal candidate-structure fields, so V2
is represented by `availability=0` and five zero continuous inputs through the
checkpoint-registered no-op helper. BioAware is not part of this 23-feature
library ranker and is not fabricated; `candidate_pair_id` is retained for a
later join to real, independently constructed sample context.

The application runner reuses the four frozen encoder caches from run `2404230`
when present and encodes only the official fine-tuned DreaMS comparator. If a
cache is absent, it falls back to complete encoding with the named checkpoint;
it never silently switches the ranking model.

```bash
sbatch tasks/run_msv000095331_integrated_multiencoder_identity.sbatch
```

The required Noise-MSG frozen model is expected at
`data/validation/noise_msg_fusion_run_2349961/fusion_model/`. Application
results must be reported as the structure-unavailable deployment profile, not
as evidence that all training-time modules were active.

## Current performance and manuscript boundary

The fixed 23-feature Additive currently has the following consumed-development
results:

| Comparison | GNPS identity, n=10,995 | GNPS formula, n=5,261 |
|---|---:|---:|
| Additive Recall@1 | 88.249% | 89.527% |
| versus official fine-tuned DreaMS | +2.892 pp | +2.718 pp |
| versus WSE | +0.882 pp | +1.255 pp |
| versus GNPS-trained B | -0.346 pp | -0.323 pp |

The article-level claim under development is therefore not “a complicated
fusion head wins.” It is that specialized representation, local spectral and
candidate-chemical training creates transferable evidence that a low-capacity
candidate decision layer can exploit, and that the resulting new entity links
recover independently supported biological information.

The final SOTA/generalization claim still requires a genuinely new resource.
MassSpecGym, GNPS, Enveda, MassBank 2026.03, MoNA and MSnLib are consumed.
MultiMS2 can support a focused cross-condition confirmation but is too small
and already design-inspected to carry the final claim alone. The cleanest
future options are new NIST identities released after the frozen training era,
subject to license, or MassBank records deposited after the project freeze.
Existing same-axis MS2DeepScore 2.x and Spec2Vec artifacts are already cached;
they can be reported as community baselines without another model-download or
architecture-development branch.
