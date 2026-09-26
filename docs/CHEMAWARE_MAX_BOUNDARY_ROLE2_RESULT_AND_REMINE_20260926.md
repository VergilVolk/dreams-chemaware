# ChemAware max-boundary role-2 result and dynamic-remine decision

## Frozen result: run 2343896

The max-reference-aligned curriculum completed exactly as specified.  The
role-2 gate stopped before role 3, so the historical `+1.8144 pp` stage-1
checkpoint remains the released model.

The most informative checkpoint is step 2,000:

| role-2 measure | official | stage 1 | max-boundary step 2,000 |
|---|---:|---:|---:|
| Recall@1 | 0.89620 | 0.91139 | 0.91747 |
| corrected / introduced vs official | -- | 54 / 24 | 54 / 12 |
| MRR | 0.93989 | 0.95013 | 0.95353 |
| Recall@3 | 0.98430 | 0.98987 | 0.99139 |
| micro-AUC | 0.93744 | 0.94231 | 0.94608 |
| macro-AUC | 0.95523 | 0.96244 | 0.96689 |
| mean positive margin | 0.34256 | 0.33170 | 0.37009 |
| cosine to official | 1.0 | 0.71643 | 0.79529 |

Thus retrieval alignment plus DreaMS replay materially improves preservation:
introduced top-1 errors versus official are halved and every reported global
metric exceeds stage 1.  But it does not increase gross corrections: both
models correct 54 official errors on role 2.

Relative to stage 1, step 2,000 is `+0.6076 pp` with 37 corrected and 25
introduced outcomes.  Its formula-cluster CI is `[-0.2373,+1.4014] pp` and
`corrected - 2*introduced = -13`; therefore it cannot pass the frozen
incremental gate.  The stop is scientifically correct even though the absolute
point estimate is better.

## Updated bottleneck

The first experiment separates preservation from correction coverage:

- static max-reference targeting and native replay solve much of the geometry
  damage;
- the fixed official-geometry candidate/reference boundary does not supply new
  gross corrections after the weights move;
- the corrected query set churns, showing that the frozen top reference and
  sometimes the frozen hardest molecule cease to be the active competitor.

The next experiment must therefore re-mine the complete training candidate
graph after a short native update.  It must not add a custom loss, retune the
replay dose, or reopen semantic-null attribution.

## Dynamic-remine experiment

1. Reproduce phase A from official for 2,000 native steps.
2. Encode every roles-0..3 reachable spectrum under that checkpoint.
3. On roles 0--1 only, recompute the current max-positive and the current
   highest-scoring false molecule over the complete candidate list.
4. Emit singleton max-boundary triplets for up to three currently active
   negative references and retain up to two active ChemAware candidates.
5. Keep 1,024 untouched official DreaMS replay events.
6. Continue with the unchanged DreaMS model/loss/Adam for at most 1,000 steps
   at LR 2e-6, checking every 250 steps.
7. Require the same strict positive role-2 increment over stage 1 before role 3.

There is deliberately no minimum current-error fraction: that quantity is an
observed consequence of phase A, not a tunable success criterion.  The hard
construction gate is complete coverage of every current training error that
does exist.

Entry point:

```bash
sbatch tasks/run_chemaware_max_boundary_remine.sbatch
```

## Protected +2.1266 pp development artifact

Run `2343962` reproduced the Phase-A result exactly on frozen role 2:

- official Recall@1: `0.8962025316455696`;
- Phase-A Recall@1: `0.9174683544303798`;
- paired gain: `+0.021265822784810127`, or `+2.1266 pp`;
- corrected / introduced top-1 outcomes: `54 / 12`;
- formula-cluster 95% CI: `[+1.2761,+3.0303] pp`;
- checkpoint SHA-256 before the old stop cleanup:
  `0a11e8dd44d18ac3f8500ed7ad48c586f7c61337e4535ad6317883988fb84180`.

This is a real shared-embedding result: the DreaMS backbone and projection
head weights changed, and no reranker or candidate-side inference input was
used.  Its evidence class remains `ROLE2_DEVELOPMENT_RESULT_NOT_ROLE3_CONFIRMED`.
The independently confirmed release therefore remains the `+1.8144 pp`
role-3 checkpoint.

The historical stop branch may already have deleted the `run_2343962`
checkpoint.  The exact low-cost reconstruction and immutable manifest entry is:

```bash
sbatch tasks/run_chemaware_phasea_2pp_protect.sbatch
```

The protection job reuses the original checkpoint if it is still present;
otherwise it rebuilds only the 2,000-step Phase A, repeats role-2 evaluation,
requires the exact `54/12` transition counts and `+2.1266 pp` gain, and stores
the weights, evidence, hashes, and claim boundary under a new protected run.
