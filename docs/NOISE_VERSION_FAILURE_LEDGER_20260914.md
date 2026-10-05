# Noise direct-finetuning version and failure ledger

Purpose: prevent a new implementation from silently replacing a previously
validated action supplier, loss, graph, initialization, or optimizer boundary.
Numbers below are evidence labels, not interchangeable performance claims.

## Evidence hierarchy

- `5.33 pp` is corrected outer-train action-space Top-1 headroom. It is not a
  held shared-encoder gain and cannot be promised by an injector.
- Historical E4 was about `+0.574 pp` on its matching old fold-0 graph; the
  five-fold/three-seed mean was about `+0.635 pp`. It proves the E4 direct
  mechanism is useful, not that old R0 row indices are valid on the corrected
  graph.
- V7 achieved only `+0.130911 pp` over E8 on corrected held data, with 67
  corrections, 43 introductions and lambda-2 risk-net `-19`. It was
  `+0.021819 pp` over clean continuation and `-0.060001 pp` below shuffled.
- Hybrid V1 engineering completed, but targeted was about `-0.092729 pp`
  versus official, `-0.005455 pp` versus its initialization, and only
  `+0.010909 pp` versus shuffled. Its action semantics were distinguishable,
  but the final update retained only about 3.6--4.5% of the original absolute
  update magnitude.
- Hybrid V2 job 2337358 completed all 30,496 steps per arm and delivered exact
  0.25 residual dose on all 24,576 active steps. Targeted was only +0.092729 pp
  over E8 and -0.027273 pp below matched shuffled, with lambda-2 risk-net -9
  and -13 respectively. This is negative residual evidence, not a transmission
  outage.

## Version failures and retained lessons

| Stage | Observed failure | Root cause | What remains valid |
|---|---|---|---|
| corrected router tests | `_route` import failure | test imported an obsolete private function | use public routing contracts and run tests only in Slurm |
| early full runs | 20+ hours without result | unbounded full-graph/action work and weak progress reporting | bounded forwards, visible counters, atomic status files |
| V3 action routing | one shuffled recipe mapped to multiple families | exact-recipe key was not source/family scoped | shuffled control must preserve source, family, recipe and change only query/action tensor |
| E4-native replay | 100 GiB CUDA allocation | action replay batched complete candidate blocks rather than only action spectra | encode each action once; reuse cached initial candidates; hard-cap training forwards |
| E4-native replay | clean ranks did not replay | stored and current geometry/boundary rows were mixed | bind checkpoint, graph and exact winning positive/negative rows; tolerate only measured fp32 error |
| V5/V6 | action recycling factor exceeded 4.0 | row multiplicity was treated as fixed optimizer dose | coverage and effective identity dose are separate ledgers |
| V6 | action-attributable optimizer fraction about 5--7% | gradient survived projection but AdamW/base update dominated it | audit realized optimizer displacement, not only pre-optimizer gradient retention |
| V7 | first-moment reconciliation error `3.63e-4` | reconstruction subtracted parameter-scale FP32 tensors | solve with realized small displacement and stable decay-plus-adaptive arithmetic |
| V7 | positive semantic alignment but weak/unsafe result | most payload gradient flowed through candidate references; protective floor was only audited | query-local reference detachment and hard optimizer-space protection are both required |
| summary | NumPy `bool_` not JSON serializable | report values were not converted to Python scalars | final reports must be serialization-tested before long jobs |
| V10 | checkpoint write failure | filesystem exhaustion/write failure after computation | preflight disk headroom, unique staging, atomic publish and result recovery |
| V11 | historical-best supplier did not recover gain | supplier/injector combination did not preserve the full successful E4 scientific stream | do not call a selected strict-action panel “E4” merely because it uses E4-shaped loss |
| Hybrid V1 | good actions plus Injector V1 still negative | later strict panel replaced the E4 base; complete E4 terms were mislabeled as corrective; absolute update shrank ~96% | retain seven-source tensors and matched shuffled control, but separate base and semantic residual |
| Hybrid V2 half-draft | four later bags collapsed into one optimizer step | fixed per-step attribution makes loss-level averaging non-equivalent to four update opportunities | each identity's four bags must occupy four distinct steps |
| Hybrid V2 preflight draft | valid corrected-E4 bank would be rejected before training | loader required invented `configuration`/verification fields absent from the frozen server report | validate the report's actual registered nine cells, counts, contracts and hashes |
| Hybrid V2 source closure draft | malformed manifest hash and placeholder manifest digest | source closure was written before final verification | recompute every entry, then bind the final manifest digest in SBATCH and test it |
| Hybrid V2 final preflight | validator expected the obsolete three-column shuffled receipt | trainer actually uses the separate four-column source/family/recipe producer | bind validator and summarizer to the imported producer schema and execute its routing tests inside Slurm |
| Hybrid V2 artifact preflight | E4 report failed an obsolete raw-byte hash | the same formal bank had been rebuilt by the current configuration-locked builder; report schema and gzip bytes changed while all registered counts/provenance remained aligned | bind the observed rebuilt bytes, then still validate every report contract, row namespace, fold, action cell and count before training |
| Hybrid V2 numerical preflight | 64-step alternating stress test was rejected by a final/combined-update norm ratio | destructive gradient cancellation can make the combined AdamW displacement arbitrarily small, so that denominator is not a stable safety reference | remove only that redundant gate; keep the 0.90--1.50 groupwise/global cap against the independent E4-only shadow update |
| Hybrid V2 final audit | three ordinary 95% formula-cluster CIs were gated independently | the joint promotion decision did not control its familywise error rate and could pass a boundary result by chance | use Bonferroni alpha 0.05/3 for all three registered comparisons and gate only the adjusted lower bounds |
| Hybrid V2 final audit | 190,324 base supplier rows could be misread as full-row training coverage | faithful historical E4 samples four views per identity per epoch, giving 121,984 physical slots over four epochs | preserve the historical sampler, but persist actual unique exposed/unexposed base rows and never claim all 190,324 were injected |
| Hybrid V2 final audit | all 32,114 later rows were present but high-row-count families still received more fixed-attribution opportunities | identity-only bags averaged rows without equalizing `(source, family)`, repeating the earlier P-like dilution failure | make every source/family present with equal mass in every identity bag; recycle only within family after its unique rows are covered; keep four steps per identity and the complete action union |
| Hybrid V2 job 2337358 | exact 0.25 injection still lost to matched shuffled | later residual omitted clean molecule rank and kept rewarding already-satisfied action views | keep the proven injector and full E4 base; restore selected clean-boundary rank and gate action rank at margin 0.05; preserve E4 symmetric/shared consistency |
| live-shared V3 failed run | all live E4 roles were present but performance still regressed | the new V3 bridge changed exact 0.25 delivery into a maximum-0.25 ceiling, forbade amplification, ignored the registered 0.90 retention argument, and accepted any non-zero transmission | retain the complete live E4 objective and independent states, but require exact 0.25 on every active group with the registered 0.90 projection floor and 1.50 norm ceiling |
| exact-injection V3 preflight | ranking-only clean probe was non-zero on 31/32 queries although the complete loss reached all four roles on 32/32 | the audit incorrectly required a non-zero clean rank-anchor derivative even when one query's winning positive and negative embeddings were identical, making that derivative mathematically zero | require complete-loss gradients for every query; for the ranking probe require all non-degenerate anchors plus all action/positive/negative paths, and persist the degenerate query indices |

## Current implementation rules (supersede the obsolete V2 residual rules)

1. Start Hybrid V2 from the frozen E8 checkpoint; do not restart from official.
2. Use the current corrected-graph, E8-mined 190,324-row E4 mechanism bank as
   the base. Never reinterpret old R0 row indices on the corrected graph.
3. Preserve all 32,114 seven-source strict actions and their exact tensors.
4. The E4 base and later-action ledgers are separate, but every later action
   uses the complete live E4 relation: clean rank, action rank, symmetric
   consistency, floor and preservation.
5. Clean, action, positive and negative inputs share the same encoder graph and
   all receive ranking gradients. Candidate references are not detached.
6. Action rank is never hard-gated; its softplus term supplies smooth decay as
   the live margin grows.
7. Dose later actions by query. One query contributes one action to an action
   opportunity, and two actions of the same query never share an optimizer
   step. Coverage cycles across source/family rows before recycling.
8. The injector sees gradients and optimizer state only; it never selects or
   mutates actions.
9. Maintain independent E4 and action AdamW states. On every active group the
   action-attributable optimizer displacement is exactly 0.25, the E4
   projection is at least 0.90, and the final/E4 norm ratio is at most 1.50.
10. Targeted and shuffled arms differ only in the later action tensor, and their
    parameter-gradient directions are compared before the injector.
11. Run Python only inside the one-GPU SBATCH allocation; never on the login node.
12. Never manually specify Slurm memory.
13. Never claim 4--5 pp until the complete held metric panel and paired CIs say so.
