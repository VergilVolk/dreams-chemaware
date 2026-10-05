# Noise E4 signal-preserving Hybrid V2 specification

Status: job 2337358 is a completed negative result; the clean-boundary repair is
an unsubmitted successor candidate and must publish to a distinct result root.

Job 2337358 proved that the optimizer bridge delivered the requested residual:
24,576 semantic-active steps reached an exact 0.25 optimizer-space fraction and
the injection gate passed. It did not prove useful action-specific transfer.
Targeted improved only +0.0927 pp over E8, had lambda-2 risk-net -9, and lost
0.0273 pp to matched shuffled with lambda-2 risk-net -13. Its formula-cluster
intervals versus E8 and shuffled crossed zero. This is a scientific failure,
not a signal-transmission or job-completion failure.

## Exact scientific composition

Hybrid V2 is direct fine-tuning of the shared clean-spectrum DreaMS encoder. It
does not distil an embedding or margin teacher and does not add an inference
branch. Its three frozen inputs are:

1. the fold-0 mature E8 checkpoint, SHA256
   `8047b3f58c6808c86b320ac94b9e610610384040fa3a03e8d70550eb438a24af`;
2. the formal corrected-graph full E4 mechanism bank from
   `noise_corrected_npa4_formal_fold_0_run_2332161_v3routes/routes/n_bank`;
3. the seven-source strict corrective union from
   `noise_corrected_best_v5_canary_fold_0_run_2332784/ledger`.

This is not another official-to-E4 rerun. The old R0 table is not used as the
base supplier because it belongs to an earlier query graph. The corrected base
contains the same nine E4 cells, re-mined without outcomes in the exact E8 and
corrected-graph geometry:

- candidate-gradient attenuation 0.50, steps 3--6;
- role-confounder attenuation 1.00, steps 1--5;
- 190,324 rows, 7,624 identities and no held-fold formulas;
- current rebuilt action-table/report hashes `1ce8c412...` and `9db644dc...`;
  the report binds builder `dafffea...` and the registered formal configuration.

The later union contains 32,114 numerically robust strict actions for 3,482
queries and 1,536 identities from `N_mature`, `P_guided_original`, `E10B`,
`E11`, `E12B`, `A4_exact`, and `V4_gradient_path`. Its action tensor, table,
and report remain byte-identical to the previously audited artifacts.

## Loss separation

The corrected E4 base stream retains the complete historical direct objective:

- live symmetric clean/action encoding by the same shared encoder;
- clean and action molecule-ranking terms;
- symmetric consistency with weight 0.25;
- margin floor with weight 2.0;
- clean/reference preservation with weight 5.0;
- one matched safety stream;
- shared live candidate references, batch-actions 4, four views per identity,
  four epochs, final block plus official projection head, backbone LR `2e-6`,
  head LR `1e-5`, AdamW and global norm 1.

Because this is an E8 continuation rather than an official-model restart, the
fixed margin-floor and preservation vectors are the pre-update E8 embeddings.
The E4 loss form, coefficients, live shared references, action cells and
identity-balanced sampler are retained; silently pulling the warm start back to
official embeddings is forbidden.

Later actions do not replace that objective. The job-2337358 implementation used
only action molecule rank plus 0.25 symmetric clean/action consistency. That was
a repeated semantic error: it omitted the selected error query's own clean
molecule boundary and kept rewarding action views after they had already cleared
the registered 0.05 E4 margin.

The repaired residual contains exactly three E4-derived terms on the selected
positive/negative molecule boundary:

- clean molecule rank with weight 1.0;
- action molecule rank with weight 1.0 only while the live action margin is
  below 0.05, and exactly zero action-rank gradient after that margin is met;
- E4's validated symmetric clean/action consistency with weight 0.25. The old
  stop-gradient alternative is not revived because its E8 paired result had no
  Top-1 increment over symmetric/shared.

Candidate reference embeddings are detached only inside this later residual,
closing the V7 reference-gradient bypass. They remain live in the complete E4
base and safety streams. No route score, stored action margin, action embedding,
teacher embedding or teacher margin is a loss target.

## Dose and coverage

Each base identity supplies exactly four E4 views per epoch. This produces
30,496 base action examples and 7,624 optimizer steps per epoch. The 190,324
rows are the complete validated supplier pool, not a claim that all rows enter
the gradient. Historical identity-balanced E4 provides 121,984 physical base
slots over four epochs; the run must report the actual unique base rows,
queries, coverage fraction and unexposed rows. Expanding this into full-row
coverage would be a new training method, not faithful E4 preservation.

Each later-action identity supplies four equal semantic action bags per epoch.
The four bags occupy four distinct optimizer steps; they are never compressed
into one update. Therefore each epoch has 6,144 semantic-active steps and 1,480
exact E4-only steps. Within an identity, every observed `(source, family)` is
present in every bag and receives an equal share of that bag's semantic mass.
Actions are partitioned inside their own source/family; a small family recycles
only after all of its unique actions have appeared. Thus high-row-count P-like
families cannot take more of the fixed-attribution optimizer opportunities than
N/A4/V4. Every one of the 32,114 unique actions is physically exposed, while
every identity still receives exactly four semantic opportunities per epoch.

Each semantic forward is packed without renormalizing solved weights and is
hard-capped at 64 spectra. The E4 base and safety forwards retain their
four-action bound. This addresses the earlier 100 GiB Graphormer replay OOM.

## Optimizer-boundary injection

`E4SignalPreservingInjectorBridgeV2` owns an independent E4-only shadow AdamW
state. Before every step, shadow parameter values are synchronized to the live
model, while its first and second moments are advanced only by the clipped E4
base plus safety gradient. The live combined AdamW counterfactual is compared
with this E4-only counterfactual.

For each head/backbone group, the materialized update must:

- assign exactly 0.25 of update norm to the later clean-boundary semantic
  residual, with
  maximum absolute error `2e-6`;
- retain at least 0.90 of the E4 protective-axis projection;
- retain between 0.90 and 1.50 times the E4-only update norm, both groupwise
  and globally;
- have stable virtual-AdamW error and first-moment reconstruction error no
  larger than `1e-3` and `1e-6` respectively.

On the 1,480 zero-residual steps, the exact shadow E4 displacement is applied.
If previous semantic history has changed the live moments, simply accepting a
normal live AdamW step is forbidden; the shadow update is materialized and the
first moment is reconciled to the realized displacement. Every one of 30,496
steps per arm is audited.

## Causal arms and evaluation

One allocated GPU runs the targeted and source/family/recipe-matched shuffled
action views sequentially. E8 initialization, E4 base actions, safety samples, candidate
references, optimizer schedule and later-action row schedule must be identical.
Only the materialized later action tensor changes.

Each arm reports, on the same complete corrected held-formula graph:

- Recall@1/2/3/5/10/20, MRR, mean and median rank;
- macro-query AUROC/AUPRC and micro-candidate AUROC/AUPRC;
- positive-versus-best-negative margin and Top1--Top2 gaps;
- corrected, introduced and lambda-2 risk-net versus official, initial E8 and
  matched shuffled, both overall and on the near subset;
- the complete near-subset retrieval panel;
- formula-cluster paired Recall@1 confidence intervals with Bonferroni
  familywise alpha 0.05 across the three registered comparisons (official,
  initial E8 and matched shuffled);
- MassSpecGym all-adduct and `[M+H]+` 10-ppm pooled pairwise AUROC/AUPRC.

The 10-ppm result is not the paper's NIST20 0.85 replication. Promotion requires
a strict positive formula-cluster increment over E8 and matched shuffled,
positive overall and near-subset risk-net against all three comparators, no
registered-metric regression, and an observed Recall@1 gain of at least four
percentage points over official. The 5.33 pp action-space headroom is not a
guarantee of encoder gain.

## Submission boundary

The sole entrypoint is:

```bash
sbatch tasks/run_noise_e4_signal_preserving_hybrid_v2_2gpu.sbatch
```

The SBATCH requests exactly one GPU and does not specify memory. All Python,
tests, training and summarization execute inside the Slurm allocation. The
source snapshot and all frozen input hashes are verified before either worker
starts; output publication is atomic.
