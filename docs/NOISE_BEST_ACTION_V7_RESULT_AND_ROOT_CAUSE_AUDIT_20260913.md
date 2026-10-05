# Noise best-action V7 result and root-cause audit

Date: 2026-09-13  
Status: complete three-arm fold-0 result recovered from the job-2336335 stdout;
the registered summary JSON still requires the no-retraining recovery job.

## Evidence boundary

This audit uses the complete 22,093-line stdout attached from server job
`2336335`, SHA256
`7b4ef4e85a3c121c608c572bb77f8bae2926cf4e010931c4b81bb5af5e40b4ce`.
The log contains completed `decision.json` reports for `routed_direct`,
`shuffled_action_control`, and `clean_control`, including all formal held
metrics.  It reaches stage 3/3 and fails only when the arm summarizer attempts
to encode a NumPy `bool_` into JSON.  Therefore the training checkpoints and
per-query/evidence ledgers are completed assets, but the cross-arm bootstrap
values in the final `report.json` must not be invented from stdout.

The only valid recovery is:

```bash
sbatch tasks/recover_noise_corrected_best_v7_summary_2336335.sbatch
```

That job verifies all three completed arm directories and reruns only tests and
the summarizer.  It never calls the trainer.

## What V7 actually consumed

The prior best-action work was not omitted in this run:

- 32,127 strict Top-1 action rows before the `5e-6` replay floor;
- 32,114 admitted corrective rows;
- 3,482 complete outer-training queries;
- all sources `N_mature`, `P_guided_original`, `E10B`, `E11`, `E12B`,
  `A4_exact`, and `V4_gradient_path`;
- no one-best-per-query compression;
- 85,959 harmful and 179,289 robust rows kept in separate branches;
- exactly four physical exposures and weighted dose four for every admitted
  corrective action in every epoch;
- the exact routed action/control candidate-boundary rows retained without
  truncation.

Thus the result does not support the explanation that good E4/E8/E10/E11/E12
actions were forgotten.  The `3,482 / 65,286 = 5.333456 pp` value is only the
outer-training action-space coverage ceiling.  It is not a clean-encoder result
and was never a held-performance guarantee.

## Formal held result

The primary baseline is the exact initial E8 checkpoint evaluated on the same
18,333-query held graph.  Values below are percentage-point changes versus E8
unless stated otherwise.

| Metric | Initial E8 | Routed V7 | Routed delta | Shuffled V7 | Clean control |
|---|---:|---:|---:|---:|---:|
| Recall@1 | 0.934544 | 0.935853 | +0.130911 pp | 0.936453 | 0.935635 |
| MRR | 0.962770 | 0.963753 | +0.098324 pp | 0.963999 | 0.963459 |
| Mean rank | 1.106584 | 1.101620 | -0.004964 | 1.101075 | 1.103856 |
| Macro-query AUROC | 0.972602 | 0.973808 | +0.120565 pp | 0.974001 | 0.973513 |
| Mean positive-best-negative margin | 0.347332 | 0.367122 | +0.019790 | 0.360669 | 0.357600 |
| Micro-candidate AUROC | 0.959269 | 0.961487 | +0.221812 pp | 0.960867 | 0.960461 |
| Micro-candidate AUPRC | 0.880435 | 0.884207 | +0.377212 pp | 0.882846 | 0.882515 |
| MassSpecGym all-adduct 10-ppm AUROC | 0.835608 | 0.841341 | +0.573391 pp | 0.837314 | 0.837096 |
| MassSpecGym `[M+H]+` 10-ppm AUROC | 0.844662 | 0.850466 | +0.580398 pp | 0.846196 | 0.846119 |
| Near-subset Recall@1 | 0.884088 | 0.884678 | +0.058988 pp | 0.886300 | 0.884678 |

The MassSpecGym pairwise number is not the NIST20 paper's exact `0.85`
replication even though the routed value happens to be `0.850466`.

Routed V7 corrected 67 and introduced 43 formal held Top-1 decisions, giving
only 24 net corrections and a risk-net at lambda 2 of `-19`.  Its formula-cluster
Recall@1 interval versus E8 is `[+0.005625, +0.258435] pp`; the near interval is
`[-0.198611, +0.313798] pp`.  Routed is only `+0.021819 pp` above clean control
(four held queries) and is `-0.060001 pp` below shuffled control (eleven held
queries).  The registered causal and near-risk gates therefore fail regardless
of the pending exact cross-arm bootstrap report.

A `+4.0 pp` held gain on 18,333 queries requires roughly 733 additional net
Top-1 successes.  V7 produced 24 versus E8, four beyond clean continuation, and
eleven fewer than shuffled.  It is not close to the promotion boundary.

## Primary root cause: the corrective gradient bypasses clean/action through references

The completed V7 calibration already contains the decisive role attribution;
it was previously recorded but not promoted to the causal decision.  In the
routed arm:

| Corrective component | Intended live role | Intended norm/fraction | Reference norm/fraction |
|---|---|---:|---:|
| Margin transfer | clean query | `0.244449` / `0.524755` | `0.380897` / `0.840337` |
| Payload rank/safety | real action | fraction `0.262603` | fraction `0.961531` |
| Consistency | clean + action | both live | exactly zero |

Because the role slices are disjoint rows of the encoded tensor, squared norm
fractions expose the approximate energy split.  About 70.6% of transfer energy
is reference-side versus 27.5% clean-query-side; about 92.5% of payload energy
is reference-side versus 6.9% real-action-side.  The primary corrective signal
is therefore not localized to the clean/action response that must generalize
at inference.

This is the missing link between the otherwise puzzling observations: routed
semantic edges are genuinely active and routed improves pooled pairwise AUROC,
yet it does not beat shuffled on formula-held Top-1.  The objective can largely
satisfy action-specific rank pressure by moving positive/negative candidate
references, changing global geometry without learning the desired clean-query
noise response.

The repository had already predicted this exact failure mode in
`NOISE_NPA4_PENDING_RESULT_RETROSPECTIVE_20260906.md`: run query-versus-reference
gradient attribution if training correction is strong but held correction is
weak.  V7 instead repaired allocation and optimizer-state reconstruction while
leaving the shared corrective reference path intact.  That was the central
implementation mistake.

## Secondary root cause: the end-to-end optimizer-tail outage remains

Gradient-space transmission is healthy.  Routed corrective-direction retention
after semantic/risk projection and clipping has p10 `0.999733`; clipping is not
the current loss point.  AdamW-space transmission is not healthy:

| Signal diagnostic | Routed | Shuffled | Contract |
|---|---:|---:|---:|
| Corrective-attributable update fraction p10 | 0.089913 | 0.090689 | at least 0.25 |
| Attributable/corrective alignment p10 | 0.924818 | 0.922576 | at least 0.05 |
| Both groups reach 0.25 target | 0.712229 | 0.697834 | at least 0.90 |
| Minimum observed protective component retention | 0.187369 | 0.146198 | at least 0.90 |
| Maximum corrective gain used | 4.0 | 4.0 | cap 4.0 |
| AdamW first-moment reconstruction error | 8.95e-8 | 8.94e-8 | at most 1e-6 |

The run's own decision reports
`legacy_90pct_end_to_end_loss_reproduced = true`.  V7 raised the median update
fraction to approximately 0.25, but it did not repair the lowest decile.  About
29% of routed steps and 30% of shuffled steps fail the joint head/backbone 0.25
target.  Average epoch values conceal this failed tail.

The implementation explains the result:

1. `U_corrective = U_combined - U_noncorrective` is often small after AdamW's
   existing moments and adaptive denominator.
2. Corrective-only restoration can multiply that residual by no more than 4.
   When this is insufficient, it deliberately returns a below-target update.
3. The protective constraint is not an enforced projection floor in every
   case.  If the gain-1 candidate is already below the 0.90 protective floor,
   `arbitrate_corrective_optimizer_updates` returns the original combined update
   unchanged with `target_reached = false`.  This is why observed retention can
   be 0.146-0.187 despite a configured minimum of 0.90.
4. The independent first-moment reconciliation now passes.  It proves the
   stored first moment reproduces the materialized displacement; it does not
   prove that displacement contains 25% corrective information.

The V7 prose says restoration "retains at least 0.90" of the protective
component.  The code actually detects but preserves an already-unsafe original
update.  The final gate is fail-closed, so no false promotion occurred, but this
is a contract/implementation semantic mismatch and not a solved safety floor.

## Consequence: routed action semantics do not transfer to held Top-1

Routed semantic transfer remains live throughout training: its pooled active
transfer fraction declines only from `0.794290` in epoch 1 to `0.765502` in
epoch 4.  The matched shuffled fractions are only `0.253448` to `0.193071`.
The routed action content is therefore present and distinguishable inside the
loss.

Nevertheless, the clean training-query panel changes as follows:

| Arm | Corrective-panel Recall@1 change | Corrected of 3,482 |
|---|---:|---:|
| Routed | +34.865020 pp | 1,214 |
| Shuffled | +34.147042 pp | 1,189 |
| Clean | +14.618036 pp | 509 |

Only 25 more training errors are corrected by routed semantics than by shuffled
semantics.  Most apparent training recovery comes from the common continuation,
augmentation and optimizer schedule rather than the discovered action identity.
On held formulas the routed semantic arm then loses eleven Top-1 queries to the
shuffled arm.

At the same time, routed exceeds shuffled by about `+0.4270 pp` on the
MassSpecGym `[M+H]+` pairwise AUROC and has a larger mean retrieval margin.
Therefore the signal is neither absent nor completely random.  It changes the
global embedding geometry, but it is not localized to the unseen-formula
query/candidate boundaries that decide Top-1.  Increasing the same global
injection alone would risk increasing the 43 introduced errors.

The scientific bottleneck is consequently action-to-clean boundary
transferability, not raw transformed-view action success.  A spectrum action
that makes its own transformed query Top-1 is not automatically a reliable
clean-gradient direction across formulas.  Strict action outcome is necessary
evidence, not sufficient clean-encoder supervision.

## What must change before another full run

V7 must be frozen as a rejected diagnostic.  It must not be expanded to another
seed by merely raising epochs, learning rate, recycle dose or the gain cap.

The next direct-fine-tuning candidate must pass two independent local gates
before a full server run:

1. **Semantic-locality gate.** Preserve the mature shared E4/E8 continuation,
   protective and auxiliary objectives, but detach positive/negative reference
   embeddings only inside the action-specific corrective residual.  Exact
   forward scores and all candidate identities remain unchanged; the residual
   must backpropagate through clean/action views.  Compare true routed,
   source/family/recipe-matched shuffled, V7 shared-locality and clean-control
   arms on the same held queries.
2. **Optimizer gate.** Use an explicitly separate corrective optimizer-state
   stream, or an equivalent functional direct-update stream, so the current
   corrective gradient is not defined as a small difference between two AdamW
   counterfactuals sharing a momentum-dominated state.  Compose the protective,
   auxiliary and corrective descent updates once in parameter space; enforce,
   rather than merely report, the protective floor and a 0.25 head/backbone
   corrective fraction.  Every infeasible step must have an explicit reason
   (`zero_residual`, `gain_cap`, `protective_infeasible`) and cannot silently
   reuse an unsafe update.
3. **Cross-source direction gate.** Keep all admitted actions in the audit ledger,
   but admit a direction as positive clean supervision only when direct
   clean-boundary gradients agree across independent source/mechanism evidence.
   Idiosyncratic action directions remain robustness/safety evidence with zero
   positive corrective weight.  The local comparison must beat the exact
   matched shuffled arm on clean-query correction and introduced-error risk;
   action-view Top-1 headroom alone cannot pass it.

This remains direct shared-encoder fine-tuning.  It introduces neither a teacher
embedding nor a distillation target.

## Serialization failure and repair

The stage-3 crash comes from a `numpy.bool_` produced by `np.isfinite` in
`shuffled_semantic_transfer_is_measured_not_forced`.  Python's standard JSON
encoder rejects that scalar type.  The repair:

- explicitly casts that comparison to Python `bool`;
- recursively normalizes all NumPy scalars/arrays at the final report boundary;
- performs `json.dumps(report)` before output publication as a transport gate;
- adds a real-NumPy regression for bool, integer, float and array values;
- supplies the job-2336335 summary-only atomic recovery entrypoint.

This repair changes no metric, checkpoint, training schedule, action tensor or
scientific decision.
