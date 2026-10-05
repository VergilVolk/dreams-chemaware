# ChemAware direct-action runtime audit (2026-09-06)

## Decision

The previous entrypoint was not safe to resubmit. Three observed late failures and one
additional latent cache-interface failure were reproducible from code inspection. The
current entrypoint is fail-closed before model construction for data-contract failures,
and an arm is consumable only after an atomic completion marker is written.

This audit establishes runtime readiness, not an embedding-performance claim. No new GPU
result was generated during this audit.

## Observed failures now covered

1. Safety query rows were absent from the frozen-prefix cache closure (`KeyError: 144813`).
2. Subset evaluation indexed rows belonging to the entire graph (`KeyError: 2623`).
3. An empty near-query stratum returned `None` and was compared with zero.
4. `cached_final_tokens` still referenced the removed `graphormer_bias` cache field; the
   current cache stores `graphormer_projected` and materializes pairwise bias per batch.
5. The first strengthened action-bank validator incorrectly rejected the generator's legal
   `role_code=-1` sentinel for action-unseen folds. Validation now proves the exact sentinel
   mask: every setting evaluates discovery folds, only the selected setting evaluates the
   confirmation fold, and inner/outer folds remain unseen.

## Fail-closed changes

- Validate CLI bounds, finite scalars, fold roles and positive batch sizes before data load.
- Validate teacher query indices, pointer partition, per-query/per-candidate array shapes,
  rank bounds, formulas and finite predictions before DreaMS construction.
- Validate action-bank schema, selected setting, role matrix, rank/margin cube, exact query
  alignment, frozen eligibility derivation and evaluation-fold outcome absence.
- Validate HDF5 row bounds and the official embedding cache's unique rows, 1024 dimensions,
  float32 type, finiteness, unit norms and complete graph-row coverage.
- Reject empty action, safety, inner-selected and inner-all sets before training.
- Reject non-finite losses, gradient geometry, gradient norms, cached embeddings and final
  metrics.
- Treat a zero-size near stratum as explicitly not applicable; it is not counted as a pass.
- Serialize the report before writing the 117M checkpoint; write CSV, checkpoint and JSON
  atomically; write `COMPLETE.json` last.
- Require `COMPLETE.json`, valid rows, report/gate consistency and row-level metric agreement
  before causal summarization.
- Run syntax, import/dependency, recognized-argument, synthetic runtime and exact production
  preflight checks before the first model is constructed in the SBATCH job.

## Evidence run locally

- Python compilation: pass.
- SBATCH Bash syntax: pass.
- Trainer and summarizer import/argparse: pass.
- All SBATCH trainer options are recognized by the trainer parser: pass.
- Synthetic cache-row, subset-evaluation, empty-near, data-boundary, report and summary
  regressions: pass.
- Real local graph/teacher contract: 700 selected queries and 3,788 teacher candidate rows,
  pass.
- Real local full-graph HDF5/official-cache contract: 32,779 unique reachable spectra, pass.
- Real official DreaMS full-forward versus frozen-prefix-forward numerical replay on three
  HDF5 spectra: minimum cosine 0.99999988, pass.
- GPU entrypoint classification/quarantine audit: pass.

## Remaining boundary

The passed direct-action bank exists on the training server but is not present in the local
workspace, so its newly strengthened schema check was not executed against the real payload
locally. The exact-production `--preflight-only` call is therefore mandatory inside the
SBATCH entrypoint and runs before `load_base_model`. If it fails, no 117M model or training
loop is started. A completed preflight is not evidence of performance; only the matched-arm
reports and formula-clustered causal summary can establish a gain.
