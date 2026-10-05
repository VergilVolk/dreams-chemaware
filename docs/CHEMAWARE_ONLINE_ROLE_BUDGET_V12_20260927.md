# ChemAware V12 Phase-A-role-budget online re-mining

> **Closed negative result (run 2345481).** Do not resubmit this experiment.
> The scheduler strongly reduced errors on the adaptive training graph but
> monotonically degraded formula-held-out role-2 Recall@1 relative to the
> protected Phase-A checkpoint.  See
> `CHEMAWARE_ONLINE_ROLE_BUDGET_V12_RESULT_20260927.md`.

## Scientific change

V11 is closed as a negative incremental result.  It reduced train-graph errors
but starved rare error and chemical roles by sampling all focused queries
uniformly.  V12 changes only the online batch scheduler.

The Phase-A training pool itself defines the immutable role budget.  For every
250 online optimizer steps, largest-remainder allocation gives:

| role | Phase-A fraction | steps per 250 |
|---|---:|---:|
| current-correct safety | 60.53% | 151 |
| current-winner error | 13.05% | 33 |
| same-query chemical error | 9.23% | 23 |
| official native replay | 17.19% | 43 |

Roles are shuffled within each interval.  Every role uses a homogeneous batch
of native one-positive/one-negative DreaMS triplets.  Queries are uniform
within role; error/chemical evidence is not broadcast to another query.

The model, `ContrastiveHead.step`, margin, batch size, Phase-A LR, online LR,
single Adam object, total online steps, re-mine frequency, checkpoints and
role-2/role-3 gates remain identical to V11.  This makes V12 a direct test of
the dose diagnosis rather than another unconstrained method change.

## Historical entry point (archived; do not resubmit)

```bash
sbatch tasks/run_chemaware_online_packet_native.sbatch
```

The entry requested one GPU, specified no manual memory, derived the scheduler
from the rebuilt Phase-A pool, and recorded each re-mine's stream sizes and exact
step allocation.  It is retained only for reproducibility of the closed run.
