# ChemAware V12 result: residual online continuation is closed

## Verdict

Run `2345481` completed the intended V12 experiment.  It used the
`phasea_role_budget` scheduler, one continuous Adam optimizer, native DreaMS
triplets, and no reranker, distillation, custom loss, or custom model.

The result is a clean negative incremental result.  The protected Phase-A
step-2000 checkpoint remains selected.  Role 3 was not opened.

## Exact role-2 result

| checkpoint | Recall@1 | change vs Phase A | corrected / introduced vs Phase A | risk utility |
|---|---:|---:|---:|---:|
| Phase A, step 2000 | 0.917468 | reference | - | 0 |
| V12 step 2250 | 0.916962 | -0.0506 pp | 7 / 8 | -9 |
| V12 step 2500 | 0.915443 | -0.2025 pp | 6 / 10 | -14 |
| V12 step 2750 | 0.915443 | -0.2025 pp | 7 / 11 | -15 |
| V12 step 3000 | 0.913418 | -0.4051 pp | 10 / 18 | -26 |

Every incremental checkpoint failed the predeclared gate.  At step 3000,
Recall@3 and micro-AUC were nonnegative relative to Phase A, but Recall@1, MRR,
macro-AUC, transition utility, and the paired formula-cluster confidence gate
failed.  That checkpoint is not a successor.

The protected Phase-A claim is unchanged: on frozen formula role 2, Recall@1
improved from `0.896203` to `0.917468`, or `+2.1266 pp`, with 54 corrections,
12 introductions, and a strictly positive formula-cluster CI
`[+1.2413, +3.0362] pp`.

## What V12 proved

V12 directly falsified the V11 under-dose hypothesis.  With the Phase-A role
budget, current training-graph errors fell from 151 at step 2000 to 126, 97,
and 73 at the subsequent re-mines.  Over the same interval, held-formula
role-2 Recall@1 moved in the opposite direction.  More error/chemical dose can
therefore optimize the observed training boundary without improving the
transferable shared embedding.

The adaptive support also collapsed during continuation:

| re-mine step | current errors | included chemical queries | distinct chemical negatives |
|---|---:|---:|---:|
| 2000 | 151 | 59 | 44 |
| 2250 | 126 | 46 | 33 |
| 2500 | 97 | 31 | 21 |
| 2750 | 73 | 21 | 14 |

The fixed scheduler nevertheless assigned 33 error batches and 23 chemical
batches in every 250-step interval, at batch size 4.  By the final interval,
92 chemical examples were repeatedly drawn from only 21 eligible chemical
queries and 14 distinct chemical negatives.  This is concentrated replay of a
shrinking, model-conditioned support, not expansion of transferable chemical
coverage.

## Scientific diagnosis

The limiting problem is not a lack of optimizer continuity, hard-negative
pressure, or ChemAware dose.  It is the conditional support of the triplets.
The online events are chosen because the current model makes a particular
decision on a particular training query and candidate graph.  That condition
contains query/formula-specific boundary information.  As optimization fixes
those decisions, the remaining support becomes smaller and more exceptional,
so a fixed role budget increasingly oversamples residual cases.  The falling
training error together with worsening formula-held-out performance is direct
evidence of adaptive hard-negative overfitting.

V11 and V12 therefore close the family of **Phase-A continuation by current
error re-mining**.  A V13 that merely changes the same role weights, adds more
steps, changes the re-mine period, or lowers the learning rate would not test a
new scientific hypothesis and must not be submitted.

## Required successor direction

The next method must improve **only the construction and admission of the
ChemAware triplets**.  It must not replace or modify the mature DreaMS native
triplet sampler, random reference sampling, contrastive head, loss, optimizer,
or the protected official replay population.

The initialization is not forced to start over.  Two valid, separately labelled
uses are retained:

1. initialize from the official fine-tuned embedding checkpoint when testing a
   complete new triplet curriculum against the Phase-A recipe;
2. initialize from the protected `+2.1266 pp` Phase-A checkpoint for a bounded
   incremental experiment using only newly qualified ChemAware triplets plus
   the unchanged official replay/safety triplets.

The second route is the default high-value continuation.  V12 closes only the
construction rule based on repeatedly re-mining the current model's residual
errors; it does not prohibit continued training from the protected Phase-A
checkpoint with a genuinely improved triplet bank.

Before another GPU run, the constructor must report, per chemical rule or
relation:

1. independent formula, query, identity, and candidate support;
2. a formula-level cap that prevents a small relation from dominating batches;
3. leave-one-formula-group-out transfer of the triplet preference inside the
   training roles;
4. matched nonchemical controls at comparable baseline margin and activation;
5. the number of genuinely new chemical relations added, rather than repeated
   spectrum realizations of the same query-specific decision.

Only relations with positive cross-formula support should enter the new
ChemAware pool.  The existing DreaMS sampling and optimization path consumes
that pool unchanged.  This changes the ChemAware triplet evidence from `repair
this model's current query-specific mistake` to `train a chemical ordering that
repeats across formulas`, without changing the official training mechanism.
