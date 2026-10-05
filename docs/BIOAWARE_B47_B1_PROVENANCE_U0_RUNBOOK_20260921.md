# BioAware B47 B1 provenance + U0 runbook

## Scientific scope

B1 closes only the truth-blind spectrum-only denominator. It does not open
annotation truth, use phenotype, score a reaction network, fit a model, or
claim BioAware gain. It freezes four linked stages:

1. the 51,976-query candidate graph;
2. one official shared DreaMS query/reference embedding cache;
3. the truth-blind cross-sample-consensus seed set;
4. U0 reference-multiplicity, candidate-exposure and adduct-pooling nuisance audit.

Only a complete hash-consistent registry can license B2 exact-event work.

## Server dependencies

The server must already contain:

- `data/validation/bioaware_b47_truthblind_candidate_graph_20260914_v1/`;
- `data/validation/bioaware_b47_truthblind_embeddings_20260914_v1/`;
- `data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz`;
- the `dreams` conda environment with CUDA PyTorch, NumPy and pandas.

Files that must be synchronised from this workspace are enumerated in
`tasks/bioaware_b47_b1_upload_manifest.txt`.

The job reuses and validates an existing frozen seed/U0 output. If either is
absent, it creates it through a job-specific partial directory and moves it
atomically only after independent validation. It never deletes or overwrites a
sealed result.

## Single entry point

```bash
sbatch tasks/run_bioaware_b47_b1_provenance_u0.sbatch
```

Logs appear in the repository root as `bioaware_b47_b1_<jobid>.out/.err`.
There is no `logs/` dependency and no explicit memory request; the job requests
exactly one GPU.

## Decision fields

Read:

- `data/validation/bioaware_b47_b1_artifact_registry_20260921_v1/report.json`;
- `data/validation/bioaware_b47_u0_reference_bias_20260921_v2/report.json`.

`pass_b1_provenance=true` means provenance, truth-blind headers, seed artifacts
and U0 are complete. `pass_to_b2_exact_event=true` additionally requires the
frozen seed policy gate; a false value blocks B2 but does not invalidate B1.
Neither flag means sample-aware signal exists. U0 rank
flips identify nuisance sensitivity only; they do not identify the superior
aggregator without a later sealed labelled comparison.
