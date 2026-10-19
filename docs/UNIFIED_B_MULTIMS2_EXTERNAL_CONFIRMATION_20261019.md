# Unified-B external confirmation on MultiMS2

Date: 2026-10-19

Status: benchmark builder implemented; full six-source exclusion not yet run

Dataset role: genuinely new external confirmation candidate, not model development

## Fixed algorithm entering external confirmation

The fixed candidate is the full-coverage `B_joint_residual` selected on the
already consumed GNPS development panels.  The recovered artifact is
`data/validation/wse_unified_B_candidate_run_2388498/unified_B_development_candidate.pt`
with SHA-256
`4b2d7e28f82a8cfca158e7b4ab6e110ef110ccdd37e07b6e506be2cd78c28baf`.

GNPS development results, which are not external claims:

| Panel | Queries | B Recall@1 | Delta vs WSE | Formula-cluster paired CI | Corrected / introduced | corrected - 2*introduced |
|---|---:|---:|---:|---:|---:|---:|
| identity-disjoint | 10,995 | 88.5948% | +1.2278 pp | [0.7384, 1.7265] pp | 390 / 255 | -120 |
| formula-disjoint | 5,261 | 89.8498% | +1.5776 pp | [0.7946, 2.3655] pp | 215 / 132 | -49 |

This fixes the claim boundary.  B is the development ranking winner, but it
failed the earlier lambda=2 risk gate.  B also did not significantly beat the
capacity-matched additive control C, so conditional interaction is not an
established mechanism.  No GNPS threshold, loss, feature, or residual bound
will be changed after this point.

## Why MultiMS2

MultiMS2 is an independent standards library deposited as MSV000099369 and
released separately from the consumed MassSpecGym, GNPS, MoNA, MassBank,
MSnLib, and Enveda assets.  The pinned source is commit
`659bd9b608408a75ecbb4b03295ed4c03ba52539`; the Linux checkout of
`data/multims2_spectra.mgf` must have SHA-256
`4ef86f75a12ad2fdadbd63732e2d6690f4e506a6ff08f171758c1808d1249673`.

The source contains 43,728 validated spectra from 2,899 compounds, including
CID at 20/40/60 eV and EAD at 12/16/24 eV.  This permits a fixed-condition
test of the exact problem the unified system is intended to address:
condition change plus local candidate competition.

## Frozen benchmark definition

Primary endpoint:

- query: CID 60 eV;
- reference library: CID 40 eV;
- ion domain: positive `[M+H]+` only, matching the GNPS development domain;
- candidates: every eligible reference identity inside the complete 10-ppm
  precursor window;
- one representative per identity and condition, chosen by maximum peak
  count and then minimum source row before any model score is computed;
- at least two candidate identities and exactly one naturally eligible
  positive; the positive is never injected after candidate construction.

Prespecified secondary endpoint:

- EAD 16 eV query against the same CID 40 eV reference rule;
- reported as an OOD stress test, never pooled with the CID primary endpoint.

Each endpoint has an identity-disjoint and a formula-disjoint version.  These
names describe exclusion against consumed sources; they are not deployment
routes and do not mean that candidates share a formula.

All six required consumed sources are loaded fail-closed from
`tasks/unified_consumed_sources_v1.json`.  Candidate spectra are excluded for
consumed identity or formula as appropriate, and for either supported exact
spectrum hash.  Missing any required source aborts construction.

The builder writes sanitized spectra and observable candidate panels under
`observable/`.  Query identities, labels, and near-subset truth are written
under `sealed_truth_do_not_open_during_scoring/`.  Model scoring must finish
for WSE, A, B, C, and the old four-module system before the sealed directory
is opened once for paired evaluation.

## Pre-score feasibility only

A local three-source preliminary pass was used only to check feasibility; it
is not the final denominator.  After structure validation, at least 10 peaks,
positive `[M+H]+`, 30-ppm precursor/structure consistency, and conservative
within-source hash conflict removal, it produced:

| Panel | Competitive queries | Candidate identities | Near queries |
|---|---:|---:|---:|
| CID identity-disjoint | 448 | 1,162 | 322 |
| CID formula-disjoint | 90 | 206 | 53 |
| EAD identity-disjoint stress | 375 | 960 | 258 |
| EAD formula-disjoint stress | 83 | 187 | 50 |

These counts used only locally available GNPS, MassBank, and MSnLib
exclusions.  They must not be quoted as final because MassSpecGym and both
MoNA files were unavailable locally.  The server job determines the only
valid external denominators using all six sources.

## Implemented entrypoint

- Builder: `tasks/build_multims2_external_benchmark.py`
- Executable contracts: `tasks/test_build_multims2_external_benchmark.py`
- Server job: `tasks/run_build_multims2_external_benchmark.sbatch`

Submit only:

```bash
sbatch tasks/run_build_multims2_external_benchmark.sbatch
```

This job downloads the exact pinned source if needed, validates its hash,
loads every required consumed source, builds the observable and sealed
artifacts, and stops.  It does not load B, compute method scores, read Enveda,
or open external truth.  The next implementation is authorized only after the
full report shows a viable primary denominator.
