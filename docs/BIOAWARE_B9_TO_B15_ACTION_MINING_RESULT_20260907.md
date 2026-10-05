# BioAware B9-B31 action-mining result (2026-09-07)

## Frozen conclusion

The current BioAware action with the strongest reproducible opened-development
evidence is B17, a fully nested arbitration of the B12 catalogue-opportunity
action and the B16 shallow nonlinear pairwise action. It conditionally switches
a low-margin DreaMS candidate toward a candidate supported by catalogue/network
opportunity features. It is **not** yet a reaction-specific mechanism, a trained
shared embedding, an independent blind result, or a SOTA claim.

| Stage | Purpose | Opened result | Decision |
|---|---|---:|---|
| B9 | reaction-specific spectral transformation | six actions approximately -0.07 to +0.07 pp; none advanced | reject as corrective action |
| B11 | Full16 catalogue interaction, nested OOF | 33/2, +5.657 pp; 17 corrected identities | positive but too narrow |
| B12 | six-domain catalogue action, nested domain LOSO | 58/8, +5.814 pp; 29 identities; 27 formulae; cluster CIs positive | current action baseline |
| B13 | simple hub-degree veto | 56/8, +5.581 pp; risk net 40 | does not beat B12 |
| B14 | reaction availability/strength recipes | 57/8, +5.698 pp; risk net 41 | path context participates but adds no independent gain |
| B15 | exact B12 action-to-spectrum support | 58 rows collapse to 54 physical corrective spectra; all references exact | passed |
| B16 | shallow nonlinear pairwise action | 44/3, +4.767 pp; risk net 38 | safer but too little coverage |
| B17 | nested B12/B16 arbitration | 57/7, +5.814 pp; risk net 43 | current safest action router |
| B18 | query-contextual nonlinear pairwise action | 45/3, +4.884 pp; risk net 39 | fails to beat B17 |
| B19 | residual evidence-action atlas | 150 actions; 106/235 residual errors reachable by some raw action, but 167/568 protected-correct queries vulnerable | headroom only |
| B20 | exact B17 direct-action manifest | 57 rows collapse to 53 physical corrective spectra; 481 exact candidate reference spectra | passed to gradient canary |
| B21 | nested mass-coverage fallback | exactly reproduces B17 (57/7, risk 43) | no independent transportable increment |
| B22 | cross-fit selector over nine residual actions | exactly reproduces B17 (57/7, risk 43) | selector rationally falls back to B17; no independent increment |
| B23 | typed catalogue/reaction/coabundance consensus | 58/12, +5.349 pp; risk net 34 | one extra correction costs five extra harms; reject |
| B24 | fixed high-hub harm veto | nested selection yields exactly B17 (57/7, risk 43) | opened hub enrichment does not transport; reject |
| B25 | B12/B16 branch-disagreement arbitration | 46 disagreements have eight-correction oracle headroom, but nested selection yields exactly B17 | apparent full-pool degree arbitration does not transport; reject |
| B26 | direct shared-encoder engineering canary | implemented; GPU result pending | tests gradient reachability and on-action realisation only |
| B27 | exact candidate-own spectrum veto over B17 actions | nested result exactly B17; full-pool veto can lower harm only by discarding many corrections | raw cosine/neutral-loss evidence is not a transportable B17 veto |
| B28 | candidate-own spectrum arbitration on 46 B12/B16 disagreements | nested result exactly B17 | raw peak evidence does not resolve branch disagreement |
| B29 | linear action-risk router over B17 interventions | 55/6, +5.698 pp; risk net 43 | removes one harm but loses two corrections; does not beat B17 |
| B30 | leave-one-source-out recurrent candidate-sink veto | 54/3, +5.930 pp; risk net 48; all six sources nonnegative | best opened action, with explicit known-candidate-only scope |
| B31 | sink-safe direct training bank | 54 corrective rows (50 physical), all 7 B17 harms as safety, 3 sink-conflicting corrections excluded | passed to shared-encoder canary |

## What the negative results mean

1. The B12 gain is not evidence that one-hop Rhea reaction adjacency is already
   encoded as a useful spectral transformation. B9 directly failed that test.
2. Five of the eight B12 introduced rows repeatedly promoted one high-degree
   glucose-like catalogue identity, so hub harm is real. B13 showed that a
   fixed degree-jump veto is not transportable enough to remove the harm without
   sacrificing useful corrections.
3. B14 allowed reaction availability and path strength to compete with the B4
   linear recipe under nested leave-domain-out selection. It did not improve the
   risk-weighted result. Therefore path features cannot yet be called the causal
   source of the B12 gain.

## B17 evidence that is retained

- 860 negative-mode opened queries across four Full16 biological sources,
  ST001154 and KGMN-200STD.
- 57 corrected and 7 introduced evaluation rows (+5.814 pp).
- Risk net at penalty two: 43, versus 42 for B12.
- 28 corrected truth identities and 26 corrected formulae.
- Identity- and formula-cluster bootstrap lower bounds are positive.
- Every held domain has nonnegative Recall@1 change.
- B12 baseline/final/intervention/corrected/introduced decisions replay exactly
  inside B17 with zero query-wise mismatches.

KGMN hidden-seed repeats reuse the same physical spectra. Evaluation rows must
therefore not be interpreted as independent training gradients. B20 exports
physical-query duplicate weights and identity-equal weights.

## B20 direct-action gate

B20 established all of the following before direct shared-encoder work:

1. all 860 B17 rows map to a real query spectrum;
2. truth, baseline and final candidates map to real MoNA reference spectra;
3. at least 50 physically distinct corrective queries remain after collapsing
   KGMN seed-mask repeats;
4. 28 corrective identities and 26 corrective formulae remain;
5. exactly seven physically distinct known-harm queries remain.

The exact counts are 753 physical query spectra, 53 physical corrective
queries, seven physical safety queries and 481 unique candidate reference
spectra. A direct shared-encoder canary may now be designed. That pilot
must use the true identity as positive, baseline/proposed wrong identities as
negatives, identity-equal sampling, and B17 introduced actions as safety cases.
It must not pull reaction neighbours together, use P2b, distil catalogue scores,
or claim the opened +5.814 pp as expected embedding gain.

## B26 direct shared-encoder canary contract

B26 is deliberately separated from the historically fragile B4 trainer.  Its
input is rebuilt and revalidated from B20 inside the same Slurm job.  It then:

1. collapses the 57 routed rows to 53 unique physical corrective spectra and
   keeps the seven non-overlapping known-harm spectra as safety cases;
2. verifies that every corrective pair is ordered incorrectly and every safety
   pair correctly by a fresh FP32 official DreaMS forward pass;
3. applies the same live encoder to query, true-reference and wrong-reference
   spectra, with dropout disabled and only the last transformer block plus the
   official projection head trainable;
4. optimises a direct true-versus-wrong margin, a one-sided safety margin floor,
   and clean-embedding preservation; and
5. records separate head/backbone gradient norms, parameter deltas, per-action
   margin changes and minimum query/reference preservation.

The canary passes only if the corrective gradient reaches both trainable model
parts, mean corrective margin increases by at least 0.005, at least five of the
53 deliberately hard training actions cross the pairwise boundary, no more
than one of seven known harms flips, and minimum clean preservation remains at
least 0.98.  This is an engineering and memorisation gate, not a held-out
performance result.  Only a subsequent formula-isolated experiment may test
generalisation.

Server entry point:

```bash
sbatch tasks/run_bioaware_b26_direct_shared_embedding_canary.sbatch
```

## B27-B31 candidate-sink result

B27 and B28 tested exact candidate reference spectra rather than the B9
reaction-neighbour spectrum. Neither fixed peak-similarity vetoes nor spectral
arbitration of B12/B16 disagreements improved the fully nested B17 result.
This is mechanistically informative: B17 often corrects candidates for which
ordinary cosine and neutral-loss similarities also favour the wrong molecule,
so a generic raw-spectral veto discards true corrections together with harms.

B29 then tested whether a regularised linear risk router could decide which
of the 109 B17 switches to execute. The outer source and all matching truth
identities/formulae were held out; hyperparameters came from inner
formula-group OOF predictions and physical duplicates carried one vote. B29
reduced introduced rows from seven to six but also reduced corrections from 57
to 55. Its risk net remained 43 and its Recall@1 gain fell to +5.698 pp. It is
therefore rejected rather than tuned further.

The seven B17 harms are not exchangeable. Four are repeated promotions into
the same candidate, `WQZGKKKJIJFFOK`, across five sources. Across its nine B17
actions this identity is correct three times, harmful four times and neutral
twice: a recurrent catalogue-supported annotation sink. B30 uses a fixed
leave-one-source-out memory rule: a proposed candidate is vetoed only when the
other sources provide at least two source observations and three physical
actions, and `corrected - 2*introduced <= 0`. Unseen candidates retain B17.
The nested opened result is 54 corrected / 3 introduced, +5.930 pp and risk net
48, versus B17's 57/7, +5.814 pp and risk net 43. Identity- and formula-cluster
CI lower bounds remain positive and every outer source remains nonnegative.

The three residual harms are promotions into three candidates absent from the
other-source sink history. A source-held audit of every available single
spectral/context threshold found zero transported harm removals: development
sources repeatedly selected a strong negative candidate-spectrum advantage,
but that threshold did not identify the harms in the held source. Therefore
these three cases are retained as safety examples. They are not converted into
an outcome-written veto, and no claim of unseen-candidate harm prediction is
made.

B30 is the current best **opened action**, but its scope is narrower than B17:
it is a known-candidate reliability memory and does not establish behaviour for
unseen proposed candidates. It was designed after inspecting opened harms, so
it must not be called independent validation or SOTA. Its value for direct
embedding training is sharper: the three B17 corrections vetoed as sink
conflicts must not inject positive gradients, while all seven original harms
remain valid one-sided safety pairs. B31 freezes exactly that bank: 54
corrective rows (50 physical), seven safety rows (seven physical), and three
uncertain sink-conflict rows with zero corrective weight. It reuses the B20
query/reference tensors byte-for-byte.

Before any shared-encoder training, B30 must first pass a deterministic
action-only replay. The action job loads one frozen B20 manifest, executes
B30 twice into separate directories, validates each run, then compares the
decompressed query-transition table, candidate-history table and all core
report fields. The only action-validation command is:

```bash
sbatch tasks/run_bioaware_b30_cross_source_sink_veto.sbatch
```

Its log files are written directly to the repository root as
`bioaware_b30_sink_<jobid>.out` and `.err`; no `logs/` directory is required.
The server dependency is the three-file frozen directory
`data/validation/bioaware_b20_direct_action_manifest_localcheck_20260907_v1`;
its action CSV, tensor manifest and report hashes are checked before either
replay. The server no longer rebuilds B20 from unsynchronised B15/B17/ST/KGMN/
MoNA intermediate artifacts.
Only after that replay passes should the single server entry point that
rebuilds and validates B20, B30 and B31 before running the shared-encoder
engineering canary be used:

```bash
sbatch tasks/run_bioaware_b31_sink_safe_embedding_canary.sbatch
```

The Slurm log files are created in the repository root as
`bioaware_b31_canary_<jobid>.out` and `.err`; no pre-existing `logs/` directory
is required. A passing canary establishes only that the direct corrective and
safety gradients reach the shared encoder and are realisable without excessive
drift. It is followed, not replaced, by formula-isolated evaluation.

## B32 complete action bank and direct bridge

An additional cross-ledger audit found a real omission in B31. The older
same-formula `current_v4` action contained 39 corrections and 11 harms. Of its
39 corrections, 27 were already B30/B31 corrections, three were exactly the
B30 cross-source sink conflicts that must retain zero weight, and nine were
non-conflicting corrections absent from B31. Of its 11 harms, only four were
already B31 safety pairs; seven known harmful pairs were unprotected.

The frozen B32 bank therefore contains the non-duplicated union: 63 corrective
rows (59 physical spectra; 31 identities; 28 formulas), 14 safety rows, and
three sink-conflicting rows with zero corrective weight. Four candidate spectra
missing from B31 were materialised from the frozen MoNA library. P2b, phenotype
labels, action probabilities, and catalogue scores are absent from the training
manifest.

The server run consumes the three-file frozen B32 artifact and verifies every
SHA256 before loading the model:

```bash
sbatch tasks/run_bioaware_b32_direct_shared_embedding_canary.sbatch
```

This is a direct shared-encoder bridge, not distillation: query, truth reference,
and wrong reference all pass through the same model; the final transformer block
and official head receive the margin gradient; dropout remains disabled; and
all 14 known harms receive one-sided safety floors. The run is deliberately an
on-action engineering canary. It can establish gradient reachability and local
realisability, but it cannot turn the opened B30 `+5.930 pp` into a claimed
embedding improvement. Formula/source-isolated training and untouched
evaluation remain mandatory after the bridge passes.

The B32 server canary passed. With one shared query/reference encoder, only
transformer block 6 of 7 plus the official projection head trainable, the 59
physical corrective actions moved from mean margin `-0.03300` to `+0.00844`.
Thirty-six of 59 pairwise errors crossed the boundary; none of the 14 known
harm pairs flipped. Minimum query and reference preservation were `0.9962` and
`0.9979`. This establishes direct gradient reachability and local action
realisability; it is not a Recall@1 result because B32 encoded only the 73
supervised physical actions and their selected references.

All 96 B32 optimizer steps hit the global norm cap of 1.0 (raw norms were about
4.3--14.3). Global norm clipping multiplies the complete batch gradient by one
scalar, so it does not delete the action direction or change the within-step
ratio among parameter coordinates. It does mean gradient magnitude differences
are suppressed and the effective step is controlled by the clipping boundary.
The positive B32 margin transfer shows this was not an action-direction failure.
It remains a reported optimization diagnostic and is not itself evidence of
generalisation.

## B33 full-reference safety bridge

B32's safety scope was incomplete: 14 safety actions protect known failures,
but the opened benchmark has 485 official-correct physical queries. A small
shared-embedding drift could introduce errors outside the known-harm set, and
B32 did not encode their candidate groups. B33 therefore adds, without changing
the BioAware action, a one-sided boundary floor for every official-correct
physical query and evaluates the complete candidate graph after training.

The frozen graph contract contains 860 evaluation rows (753 physical queries),
3,314 candidate molecules, 22,737 candidate-to-reference links, and 3,515 exact
MoNA reference spectra. Candidate scores are re-maximised across all reference
spectra after the encoder changes; retaining only the official model's original
best reference is forbidden because the best reference may change under a new
embedding. The graph exactly reproduces the official row-weighted Recall@1
`0.660465` and MRR `0.794199` before training.

B33 keeps the B32 capacity, learning rates, 96-step schedule, and action loss
fixed. It adds 485 physical broad-safety queries, evaluates corrected and
introduced errors on all 860 candidate groups, reports both row-weighted and
physical-query-weighted Recall@1/MRR, and saves a checkpoint only as an opened
engineering artifact. The single server entry point is:

```bash
sbatch tasks/run_bioaware_b33_full_graph_bridge.sbatch
```

Passing B33 requires positive full-graph Recall@1 and nonnegative MRR, corrected
greater than twice introduced in both weighting schemes, no more than one known
harm flip, at least 20 realised corrective actions, and minimum query/reference
preservation of 0.98. Even a five-point opened gain would still require
formula/source-isolated training and untouched external evaluation before any
SOTA claim.

## Residual-action boundary

B19 shows why another unstructured feature sweep is low value. Among the 235
official errors left unresolved by B17, 106 are recoverable by at least one raw
catalogue/reaction/coabundance action, but 167 of 568 official-correct queries
are harmed by at least one such action. The most conservative single residual
action, maximum known mass-window catalogue coverage, adds only three opened
corrections. When its none/0.04/0.05/0.08 gate is selected inside each outer
domain (B21), the transported result is exactly B17: no independent gain.

This means the remaining opportunity is an **action-selection** problem, not a
missing scalar feature. Any successor must learn among a small fixed action
set using inner-domain cross-fitting; it cannot import the B19 truth-ranked
oracle as a performance result.

B22 performed that test with a shallow candidate selector over nine fixed
actions and a third leave-domain-out gate inside every outer fold. It replayed
B17 query by query but selected no additional transportable correction. B23
then tested the lower-capacity hypothesis that independent catalogue, reaction
and co-abundance families must nominate the same candidate. The consensus did
not behave as independent evidence: it produced 58 corrections and 12 harms,
reducing the risk net from 43 to 34. B24 tested the observed high-hub harm
signature as one fixed revert-to-DreaMS veto. Inner-domain selection retained
the veto only in a fold where it changed no outer query, so the nested result
again equalled B17. The apparent opened full-pool 54/3 hub-veto result must not
be reported as transported performance.

Therefore B17 remains the frozen broad-action comparator, while B30 is the
current opened safety action and B31 is the only authorised direct-training
bank. Further tuning on these same six opened domains is no longer an
independent action-discovery test. New action families require either genuinely
new evidence (for example candidate-specific fragment transformations) or new
development cohorts; merely adding selector capacity, voting rules, or hub
thresholds is not justified by the current results.

B25 additionally audited the one candidate set B22 did not explicitly retain:
the alternative B12 and B16 branch proposals. They disagree on 46 queries; the
union contains the truth on 25 while B17 is correct on 17, so the opened oracle
headroom is eight queries. A fixed higher-catalogue-degree arbitration looks
slightly better on the pooled opened table (60/8 versus B17 57/7), but every
outer fold selected the original B17 from its inner-domain results. The formal
nested result therefore changes zero queries. Branch disagreement is useful
diagnostic headroom, not a validated new action.

## Claim boundary

B17 establishes a broad, low-margin catalogue-opportunity routing signal on
opened development cohorts. B20 establishes only whether that signal has exact
spectrum-level training support. A shared encoder must still independently show
positive identity- and formula-cluster confidence intervals, corrected greater
than twice introduced, nonnegative MRR, and strong preservation before the
BioAware embedding claim can advance.
