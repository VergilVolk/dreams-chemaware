# ChemAware max-boundary DreaMS-native curriculum

## Decision

The immediate problem is not whether ChemAware can move a shared embedding.
That was established by the frozen `+1.8144 pp` stage-1 result.  The problem is
why only 35 net role-3 top-1 errors were removed when 97 net corrections are
needed for a five-point gain.

The previous surgical/null proposal is stopped before GPU submission because
it prioritised attribution over the performance bottleneck.

## Quantified bottlenecks

1. Stage 1 contains 7,688 molecule events, but only 1,084 events (14.1%) come
   from the 427 official-error training queries.
2. DreaMS samples one positive and one negative reference per event.  At the
   3,000-step optimum (about 1.56 visits/event), only about 27.6% of
   official-error events are expected to have sampled the exact positive and
   negative references that define the frozen max-reference boundary.
3. Of the 5,999 action-hard events on official-correct training queries, about
   68.2% have zero initial native hinge.  They consume sampling probability
   without supplying correction gradients.
4. The stage-1 run used the DreaMS runtime but not the official 112,601-anchor
   10-ppm DreaMS triplet pool.  Its narrow update therefore lacks explicit
   global replay, consistent with 22 introduced role-3 top-1 errors and a mean
   cosine of 0.721 to the official embedding.
5. The strongest candidate-side policy has 93 corrected and 17 introduced
   outcomes on role 3 (net 76, +3.94 pp).  The current action bank therefore
   does not itself guarantee five-point shared-embedding headroom.  Both action
   coverage and triplet transfer efficiency must improve.

## First corrective experiment

The model and optimizer are unchanged.  Only the empirical triplet pool is
rebuilt.

- For each official-error training query, retain its official wrong candidate
  and at most two active ChemAware false candidates.
- Freeze the top-scoring true reference and up to two active top-scoring false
  references under the official embedding.  Each native event is a singleton
  reference pair, so DreaMS optimizes the same max-reference boundary used by
  retrieval instead of hoping random sampling reaches it.
- For each official-correct query, keep only the exact nearest false-candidate
  max boundary as a safety sentinel.  ChemAware expansion is not spent on
  already-correct queries.
- Append 1,024 events sampled without replacement from the official DreaMS
  10-ppm triplet pool, with their positive and negative lists copied exactly.
  This is global geometry replay, not a custom preservation loss.

The resulting local audited pool has:

- 4,032/4,032 ChemAware training queries and 2,518 formulas;
- 5,956 total native events;
- 1,327 retrieval-aligned error events (22.28% of the total, versus 14.10% in
  stage 1);
- all 427 official-error queries covered;
- 777 official-error max-boundary events and 550 active chemical
  max-boundary events;
- 3,605 correct-query safety sentinels;
- exactly 1,024 unmodified official DreaMS replay events;
- identity-valid positive and negative edges throughout.

## Frozen runtime

- initialization: official DreaMS embedding;
- native `ContrastiveSpectraDataset` and `ContrastiveHead`;
- cosine triplet margin 0.1, Adam, LR 5e-6, batch size 4;
- checkpoints at 500/1,000/1,500/2,000/2,500/3,000 steps;
- role 2 must beat the frozen stage-1 checkpoint with positive formula-cluster
  CI, positive corrected-minus-two-introduced utility and nonnegative MRR,
  Recall@3, micro-AUC and macro-AUC;
- role 3 is evaluated only for the one role-2 winner;
- formula role 4 remains untouched.

Server entry point:

```bash
sbatch tasks/run_chemaware_max_boundary_native.sbatch
```
