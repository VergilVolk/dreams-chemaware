# ChemAware exact-boundary residual V9

Date: 2026-09-27

## Decision from the dense V8 result

The protected Phase-A embedding remains the role-2 champion:

- official Recall@1: `0.8962025316455696`;
- Phase-A Recall@1: `0.9174683544303798`;
- paired gain over official: `+2.1266 pp`;
- corrected / introduced at top 1: `54 / 12`;
- formula-cluster 95% CI: `[+1.2413,+3.0362] pp` in the recovered audit.

Dense true-support V8 did not advance this result.  Its best saved checkpoint,
step 500, retained `+2.0759 pp` versus official but was `-0.0506 pp` versus
Phase A, with 10 corrections and 11 introductions relative to Phase A.  Later
checkpoints were worse.  V8 is therefore a valid stopped negative result, not
a second two-point improvement.

## Root-cause audit

V8 established chemical support on one representative query per identity,
converted the map to `support_by_identity`, and then mined current hard
boundaries from every spectrum of that identity.  The local full-graph replay
quantified the resulting disconnect:

| Quantity | Count |
|---|---:|
| V8 correction events | 4,425 |
| Same source query as chemical evidence | 1,565 (35.37%) |
| Identity-broadcast events | 2,860 (64.63%) |
| Same source query and contrasted candidate | 803 (18.15%) |
| Exactly aligned queries / formulas | 343 / 228 |

Even 18.15% is an upper-bound interpretation because multiple reference rows
can represent one candidate boundary.  The failure is therefore not a lack of
raw triplet count.  Most of the extra count did not retain the chemical
evidence unit.

The reproducible audit is
`tasks/audit_chemaware_dense_boundary_alignment.py`; its local output is
`data/validation/chemaware_dense_true_support_official_geometry_localcheck_20260926/boundary_alignment_audit.json`.

## V9 scientific unit

For query `q`, true candidate `c+`, and the current Phase-A highest-scoring
false candidate `c-`, V9 defines a candidate-boundary contrast for each arm:

`B_a(q) = u_a(q,c+) - u_a(q,c-)`.

The chemical specificity statistic is

`S(q) = B_correct(q) - max_k B_null-k(q)`.

For the original baseline false candidate, the already centered
candidate-minus-baseline rule and action-advantage fields are used directly.
For a changed Phase-A false candidate, truth and false slots are explicitly
subtracted before comparison with the three null arms.  An event is admitted
only when:

1. the exact query is currently wrong under Phase A;
2. the proof refers to that same current false candidate;
3. at least the frozen minimum number of metric contrasts is positive;
4. median chemical specificity is positive; and
5. the native DreaMS triplet hinge is active.

No evidence is transferred to another query of the same identity.

## Native residual curriculum

The emitted object is still the standard DreaMS triplet
`(query spectrum, same-identity positive spectrum, different-identity negative spectrum)`.
Only triplet selection changes.

- exact correction: up to three currently active references of the proven
  current false candidate;
- safety: three current-correct max-boundary sentinels per correction event,
  nearest boundaries first and formula-diverse first;
- global replay: 512 unmodified official DreaMS 10-ppm events;
- sampler: ordinary uniform shuffled DataLoader without replacement;
- model/loss/optimizer: unchanged DreaMS shared encoder, `ContrastiveHead`,
  cosine triplet margin `0.1`, and Adam;
- initialization: checksum-verified protected Phase-A checkpoint;
- schedule: LR `1e-6`, at most 500 updates, checkpoints every 100 updates.

The safety set deliberately contains both currently active events and inactive
guards.  Guards become active if continuation moves a correct boundary toward
the margin, reproducing the preservation mechanism that distinguished Phase A
from the earlier stage-1 model.

## Executed local full-data preflight

The local run uses the official embedding cache only as an engineering
stand-in; it is not a Phase-A performance result.

| Quantity | Local preflight |
|---|---:|
| evidence queries | 4,032 |
| current errors under stand-in | 428 |
| exact correction queries | **166** |
| correction formulas | **142** |
| strict boundary queries | 130 |
| exact correction spectrum events | **373** |
| active safety events | 302 |
| inactive safety guards | 817 |
| untouched DreaMS replay | 512 |
| total native events | **2,004** |

All 24,278 positive and 23,760 negative identity edges passed.  Two independent
full constructions were array-identical for all nine CSR and metadata fields.
The result passes query/candidate alignment, coverage, safety budget, replay,
unique-signature, formula-role isolation, and no-custom-sampler gates.

Formal Phase-A counts are intentionally unknown until the server reconstructs
the checksum-matched role-0/1 cache.  Fewer than 100 correction events, 50
independent queries, or 40 formulas stops the job before training.

## Formal Phase-A coverage result: stopped before training

The checksum-matched Phase-A cache changed the engineering conclusion.  The
formal run found 151 remaining role-0/1 errors, but only 23 queries / 23
formulas had a same-query, same-current-false-candidate chemical proof; those
queries emitted 58 correction events.  Of the 23 queries, 21 addressed the
original baseline boundary, only two addressed a changed Phase-A boundary,
and nine passed every specificity contrast.

All provenance, safety, replay, uniqueness and split-isolation gates passed.
The three intentionally frozen coverage gates failed.  This is therefore a
scientific coverage stop, not a Python failure and not evidence of a harmful
trained checkpoint: training never started and no weights were updated.

The ratio `23 / 1929` is **not** a performance ceiling: training boundaries can
generalize to unseen evaluation queries.  The formal conclusion is narrower
and defensible: 23 queries / 23 formulas provide too little independent
chemical and structural diversity for a claimed broad residual curriculum,
and make a formal continuation unusually sensitive to individual examples.
Lowering thresholds or broadcasting the 23 proofs across identities would
recreate the exact V8 failure mode and is prohibited.

The next admissible route must enlarge the *independent query evidence unit*.
The retained candidate is a leave-one-spectrum-out cross-view **triplet
miner**: other spectra of the same training identity may validate
candidate-identity-aligned chemical evidence, while the held query and its
current Phase-A false candidate define an ordinary DreaMS triplet boundary.
No teacher score is distilled, no auxiliary head is trained, and no chemical
input exists at inference.  A read-only coverage audit must show at least the
required independent current-error queries before native continuation is
authorized.

## Frozen evaluation rule

Formula role 2 compares every 100-step checkpoint directly with protected
Phase A.  A checkpoint advances only if it has:

- strictly positive Recall@1 and MRR versus Phase A;
- strictly positive formula-cluster Recall@1 CI lower bound;
- positive `corrected - 2 * introduced`;
- nonnegative Recall@3, micro-AUC, and macro-AUC.

Only an advancing checkpoint reaches independent role 3 and the combined
role-2/3 evaluation.  Formula role 4 remains untouched.

## Single server command

```bash
sbatch tasks/run_chemaware_exact_boundary_residual.sbatch
```

The job requests exactly one GPU and contains no manual memory request.

## Claim boundary

V9 is currently an implemented, deterministic, full-data-audited experiment,
not a retrieval gain.  The established shared-embedding results remain the
protected `+2.1266 pp` role-2 Phase A and the historical `+1.8144 pp` role-3
confirmation until the frozen server evaluation reports otherwise.
