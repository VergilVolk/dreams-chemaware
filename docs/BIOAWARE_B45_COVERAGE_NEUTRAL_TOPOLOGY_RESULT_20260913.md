# BioAware B45 coverage-neutral topology result

## Decision

B45 is a decisive negative mechanism result. It removes the catalogue-
membership shortcut identified by B44 and asks whether Rhea/KEGG topology
degree alone can still improve DreaMS when the baseline and challenging
candidate have exactly the same catalogue-coverage signature.

It cannot. Static catalogue topology is not admitted to another external
evaluation, a shared-embedding target, or a biological-mechanism claim. This
result supersedes the earlier B42 interpretation that mapped competition by
itself established an independent topology benefit.

## Protocol

- Development universe: the six already opened B42 sources; B44 is never read.
- Outer evaluation: leave one source out in full.
- Inner selection: leave one remaining source out; no outer-source outcome is
  used to choose the gate.
- Identity and formula belonging to the held source are purged from training by
  the frozen B42 splitting code.
- A candidate may replace the DreaMS baseline only when both have the exact
  same `(independent_member_count, independent_member_intersection)` signature.
- Training uses only truth-negative pairs with the same signature.
- Comparator arms use spectral score alone, real within-coverage topology, and
  three deterministic within-query/within-signature topology permutations.
- All arms use the same learner, folds, gate grid, weighting and tie rules.

The audit covers 860 negative-mode queries. There are 515 queries with a
coverage-neutral alternative and 153 baseline errors that are in principle
recoverable inside that restricted action set.

## Result

| Arm | Recall@1 delta vs DreaMS | Corrected / introduced | Interventions |
|---|---:|---:|---:|
| Spectral only | 0.0000 pp | 0 / 0 | 0 |
| Real topology degree | **+0.2326 pp** | **4 / 2** | 10 |
| Permuted degree 0 | 0.0000 pp | 1 / 1 | 4 |
| Permuted degree 1 | 0.0000 pp | 5 / 5 | 22 |
| Permuted degree 2 | -0.2326 pp | 5 / 7 | 22 |

For real topology versus DreaMS:

- formula-cluster 95% CI: `[-0.3922, +1.0181] pp`;
- identity-cluster 95% CI: `[-0.3803, +1.0158] pp`;
- `corrected - 2 * introduced = 0`;
- five of six sources selected no operation; only NIST plasma had a positive
  point estimate, while Mouse liver had equal corrections and harms.

For real topology versus the three matched permutations, all formula- and
identity-cluster confidence intervals cross zero. The observed advantage is
therefore indistinguishable from arbitrary reassignment of degree values among
candidates with the same coverage signature.

## What this changes

1. The B42/B43 approximately 6--7 pp opened-development result is real as an
   engineering replay but is not evidence of a portable reaction-topology
   mechanism.
2. B44 showed the catalogue coverage prior reverses under external catalogue
   ascertainment shift: unmapped truths are systematically displaced by mapped
   wrong candidates.
3. B45 shows that after removing this coverage shortcut, topology degree does
   not retain a material or statistically supported effect.
4. The earlier B42 statement that mapped competition established an
   independent topology benefit is superseded by this stricter audit.

## Frozen boundary

Do not:

- retune the B42/B44 threshold;
- use catalogue membership or degree as a BioAware shared-embedding teacher;
- call B42/B43 reaction propagation, sample context, or cross-database gain;
- run another external catalogue test of the same static feature family.

The next BioAware experiment must use query-specific observed sample context
and must beat degree/coverage controls and structural nulls before any context
adapter or encoder fine-tuning. Existing four-hop diffusion (B40), aggregated
direct-path features (B38), and strict atomic path intersections (B39) are
already negative and must not simply be rerun with more hyperparameters.

## Reproducible artifacts

- `tasks/audit_bioaware_b45_coverage_neutral_topology.py`
- `docs/BIOAWARE_B45_COVERAGE_NEUTRAL_TOPOLOGY_CONTRACT_20260913.md`
- `data/validation/bioaware_b45_coverage_neutral_topology_local_20260913_v1/report.json`
- `data/validation/bioaware_b45_coverage_neutral_topology_local_20260913_v1/nested_loso_transitions.csv.gz`
- `data/validation/bioaware_b45_coverage_neutral_topology_local_20260913_v1/inner_gate_ledger.csv.gz`

## Claim limit

B45 is an opened-development mechanism audit. It establishes neither a new
external performance estimate nor absence of all biological signal. It
specifically rejects static catalogue coverage/degree as a portable BioAware
mechanism under the tested protocol.
