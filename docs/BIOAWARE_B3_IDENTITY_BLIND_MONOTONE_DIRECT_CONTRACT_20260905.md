# BioAware B3: identity-blind monotone direct-context contract

Date frozen: 2026-09-05

## Why B3 is necessary

BioAware B2 is a direct listwise candidate-context embedding adapter, not
distillation.  Its free MLP can read the candidate DreaMS embedding.  In the
leave-study-out RT-only cell it improved Recall@1 by 2.05 percentage points
(19 corrected, 1 introduced; formula-cluster 95% CI 0.72 to 3.68 points).
However, every flipped truth identity and competitor identity had appeared in
the training candidate lists, and every truth identity had appeared as a
training truth while its decisive competitor had not.  Removing held-out truth
identities or formulas collapsed the effect to 0.23 points (3 corrected, 1
introduced; CI crosses zero).  Therefore the large result is treated as a
candidate-identity shortcut, not transferable retention-time evidence.

The preregistered B2 primary (all evidence, leave-study-out) changed only five
Top-1 decisions: 4 corrected and 1 introduced, with a formula-cluster CI lower
bound of zero.  One corrected/introduced pair reversed truth roles across
studies.  The fixed B2 matrix therefore does not unlock DreaMS last-block
fine-tuning.

## B3 scientific question

Can outcome-blind reaction, spectral-network and retention-time evidence
improve candidate ranking when the learnable function is unable to identify a
candidate or query molecule?

This is the minimum causal bridge that must pass before adding a higher-capacity
graph encoder or fine-tuning DreaMS itself.

## Model

For unit-normalized query and candidate embeddings q and c, the learnable model
sees only a non-negative evidence vector e.  It computes

    w_j = theta_j^2
    support(e) = 1 - exp(-sum_j w_j e_j)
    tangent(q,c) = q - (q dot c)c
    c_context = normalize(c + delta_max * support(e) * tangent(q,c))

The parameterized function never receives q, c, candidate identity, formula,
study, phenotype or downstream P2b scores.  Query and candidate embeddings are
used only by the fixed geometric lift.  Increasing an evidence coordinate
cannot turn that evidence into a penalty.  Zero evidence returns c exactly.

This is direct listwise training.  There is no teacher and no distillation.

## Frozen development matrix

All cells use three seeds, 200 identity-balanced epochs, batch size 32,
learning rate 0.01, temperature 0.08, maximum tangent step 0.05, exact safety
slack 0, safety weight 8, preservation weight 8 and gate weight 0.005.

Primary:

- `full_support / truth_formula`: all 12 evidence coordinates can activate the
  model; the outer study, every held-out truth formula and every training query
  containing a held-out candidate identity are excluded.

Attribution controls under the same formula/candidate isolation:

- `reaction_smn`
- `smn_only`
- `rt_only`

The attribution cells cannot replace a failed primary.  They identify what
evidence family should be expanded in a later prospective dataset.

## Gates

The primary passes only if all hold:

1. formula-cluster bootstrap Recall@1 delta CI lower bound is greater than 0;
2. corrected is greater than introduced and corrected - 2*introduced is
   greater than 0;
3. no outer study has negative Recall@1 delta;
4. mean candidate-embedding preservation is at least 0.995;
5. artifact replay has zero rank mismatch;
6. held-out truth/candidate identity overlap is zero;
7. every corrected flip has a positive learned support advantage over its
   decisive competitor.

Failure means the current evidence table is too sparse or weak for embedding
learning.  It does not justify a larger MLP, a DreaMS last-block update, or a
SOTA claim.  The next data intervention would be the MetDNA3/KGMN-style global
two-layer feature graph: MS1 feature nodes, MS2-similarity edges, reaction
edges, ion-family/correlation constraints, explicit unknown states and global
assignment consistency.

## Claim boundary

B3 outputs a query-conditioned candidate contextual embedding.  It is not a
universal single-spectrum embedding.  A biological context cannot be inferred
from one spectrum unless biological context is supplied at inference.  Direct
DreaMS fine-tuning is permitted only after B3 passes and must be evaluated both
with and without context to show that generic spectral retrieval is preserved.
