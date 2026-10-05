# BioAware B47 current-context readiness result

## Decision

The existing BioAware ledgers are **not ready** to establish a prospective
sample-context annotation benefit. No additional reranker, adapter or shared
embedding may be trained from these context labels.

This is an identifiability decision, not a claim that metabolic context is
biologically useless.

## Evidence closure

| Requirement | Current evidence | Gate |
|---|---:|---:|
| Prospective-unknown sources | 0 | fail |
| Event-specific spectral rows | 0 | fail |
| Event-specific coabundance rows | 0 | fail |
| Sample-local truth absent from original seed set | 0/162 | fail |
| B40 prospective reconstruction | false | fail |
| B45 coverage-neutral topology | +0.2326 pp, 4/2, CI crosses 0 | fail |
| B46 exact-enough dual evidence | -0.7714 pp, 10/21 | fail |
| B44 independent catalogue transport | -0.5599 pp, 8/13 | fail |

The source semantics are four synthetic held-identity rotations, one hidden
standard repeat and one sample-local leave-one-seed-out reconstruction. In the
last source, all 162 evaluated identities had already appeared in the sample
seed table. It is a counterfactual recovery experiment, not prospective
unknown annotation.

## Why this matters

The B42 development gain cannot be rescued by arguing that the network was
merely underfit:

- B44 showed that static catalogue coverage reverses from favoring truth to
  favoring wrong candidates externally.
- B45 removed catalogue-coverage differences while retaining 153 recoverable
  errors; real topology then became negligible and indistinguishable from
  topology permutations.
- B46 retained 144 coverage-neutral errors and combined two experimental
  summaries; the action became significantly harmful at the formula level.

The bottleneck is therefore neither learning rate nor neural capacity. The
present supervision does not identify the proposed causal object: a specific
sample-observed seed supporting a specific candidate through a specific
reaction event.

## Next executable route

Proceed only through the frozen B47 prospective-context contract:

1. acquire a public dataset with linked MS1, MS/MS, RT, sample abundance and
   independent validation identities;
2. seal validation truth before context construction;
3. require each evaluation truth identity to be absent from the original seed
   set;
4. construct event-level candidate--seed--reaction evidence;
5. compare against catalogue coverage, rewired edges, permuted seeds,
   wrong-transform events and matched non-neighbours;
6. require at least `+3 pp`, positive formula/source clustered confidence
   intervals, and `corrected > 2 * introduced`.

The 2026 MSMICA head-to-head and external-validation Zenodo deposit is the
primary acquisition candidate, subject to a metadata-only M0 audit proving
that experimental inputs and independent truth can be separated.

## Artifacts

- `tasks/audit_bioaware_b47_current_context_readiness.py`
- `tasks/validate_bioaware_b47_current_context_readiness.py`
- `data/validation/bioaware_b47_current_context_readiness_20260913_v1/report.json`
- `docs/BIOAWARE_B47_PROSPECTIVE_CONTEXT_BENCHMARK_CONTRACT_20260913.md`

## Claim limit

This result closes inappropriate reuse of the current context ledgers. It does
not evaluate a new BioAware method, establish external gain, change a shared
embedding or support a SOTA claim.
