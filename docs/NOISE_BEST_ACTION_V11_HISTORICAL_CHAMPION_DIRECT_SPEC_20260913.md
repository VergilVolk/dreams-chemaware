# Noise V11 historical-champion actions + Injector V1 specification

Date: 2026-09-13  
Status: implemented bounded two-GPU canary; shared-encoder result pending

## One permitted change

V11 freezes the complete V10 direct E4/E8 training and Injector V1 numerical
contract.  It changes only the upstream corrective action supplier:

- V10 supplied all 32,114 current-E8 strict Top-1 action views from the seven
  mature sources and averaged heterogeneous views inside a query.
- V11 supplies exactly one deterministic champion per correctable query: the
  strict Top-1 action with maximum current-E8 positive margin, followed by the
  unchanged `5e-6` numerical margin floor.
- Harmful, robust and protective branches are not removed or relabelled.

This is the no-op-aware E12-B / historical E4-native best-action union rule on
the current outer-training geometry.  The registered full ledger has 32,127
strict rows before the floor, 3,483 per-query champions before the floor and
3,482 champions after it.  On 65,286 outer-training queries this is 5.333456 pp
of action-space headroom.  That number is not a trained-encoder gain.

## Frozen action provenance

Every mature source competes for the champion position:

- `N_mature`: candidate-gradient 0.50 steps 3--6 and role-confounder 1.00
  steps 1--5;
- `P_guided_original`;
- `E10B`;
- `E11`;
- `E12B`;
- `A4_exact`;
- `V4_gradient_path`, restricted to its nine already qualified recipes.

No source name is treated as an experiment result.  E4 and E8 remain the
direct-fine-tuning foundation and initialization; E13 reused E12B and is not a
separate action bank.  The selector reads only current training-ledger action
rank and margin.  It does not read outer-held outcomes, historical
`passing_cells`, `best_fixed_cell`, or teacher embeddings, and it does not
mutate spectra.

## Frozen injection contract

`tasks/noise_action_injector_v1.py` remains byte-frozen at SHA-256
`21e881f00942a61c2a192a4358806379215d55026f87c934c04f975da1887931`.
V11 therefore retains exactly:

- one real AdamW step;
- the combined, noncorrective and protective same-state virtual ledgers;
- the hard 0.90 protective floor;
- exact 0.25 optimizer-space action attribution for head and backbone;
- the 1.50 update-norm cap;
- materialization as `before - composed_update`;
- first-moment-only reconciliation, with actual combined-gradient second
  moment and step preserved.

The registered V11 configuration differs from V10 only in
`direct_contract` and `action_bank_contract`.

## Causal canary and evaluation

The same four arms run in two waves on exactly two GPUs:

1. champion `full_action_view`;
2. champion `scalar_transfer_only`;
3. exact source/family/recipe-matched shuffled champion control;
4. clean continuation control.

The action bank is split by formula before selection, calibration, donor
construction or training.  The canary uses 512 corrective training queries,
an 800-query formula-disjoint inner corrective panel when the registered data
reproduce, and 4,096 outer-held queries.  It reports Recall@1/2/3/5/10/20,
MRR, mean/median rank, macro-query and micro-candidate AUROC/AUPRC, margins,
Top1--Top2 gaps, corrected/introduced/risk-net, near metrics, formula-cluster
paired CIs, and all-adduct plus `[M+H]+` 10-ppm pooled pairwise AUROC/AUPRC.

No 4--5 pp encoder claim is made before those held results exist.  The run is
the controlled test of whether the previously strongest per-query action
content, now transmitted by the already verified Injector V1, transfers into
the clean shared embedding.

## Server entrypoint

Run only through Slurm:

```bash
sbatch tasks/run_noise_corrected_v11_historical_best_canary_2gpu.sbatch
```

The job requests exactly two GPUs, makes no manual memory request, executes all
Python tests and action-ledger preflight on allocated compute resources, uses
an immutable source snapshot, fails both workers when either arm fails, and
publishes the result directory atomically.
