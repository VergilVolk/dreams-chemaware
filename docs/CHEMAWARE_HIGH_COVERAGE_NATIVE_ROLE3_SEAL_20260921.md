# ChemAware high-coverage native DreaMS role-3 result seal

Date: 2026-09-21
Status: `ROLE3_CONFIRMATION_POSITIVE / OUTER_NOT_OPENED / CODE_FROZEN`

## Frozen run identity

- Evidence and triplet construction run: `run_2340524`
- Resumed training run: `run_2340721_resume_2340524`
- Frozen evaluation policy: `run_2340524/evidence/confirmation_triplet_evidence.npz`
- Selected model: `run_2340721_resume_2340524/training/best.ckpt`
- Rejected late model: `run_2340721_resume_2340524/training/last.ckpt`
- Evaluation output: `run_2340721_resume_2340524/role3_evaluation_after_stop.json`
- Evaluation queries: 1,929
- Official frozen-ledger rank mismatches: 0

## Frozen role-3 result

| Metric | Official DreaMS | ChemAware best | Absolute delta |
|---|---:|---:|---:|
| Recall@1 | 0.9046137895 | 0.9227579057 | +0.0181441161 (+1.8144 pp) |
| Recall@3 | 0.9875583204 | 0.9927423536 | +0.0051840332 (+0.5184 pp) |
| Recall@5 | 0.9974079834 | 0.9979263867 | +0.0005184033 (+0.0518 pp) |
| Recall@10 | 0.9994815967 | 1.0000000000 | +0.0005184033 (+0.0518 pp) |
| Recall@20 | 1.0000000000 | 1.0000000000 | 0 |
| Recall@50 | 1.0000000000 | 1.0000000000 | 0 |
| MRR | 0.9458699507 | 0.9566592198 | +0.0107892691 |
| Micro AUC | 0.9417011886 | 0.9497803994 | +0.0080792108 |
| Macro AUC | 0.9560965221 | 0.9654386006 | +0.0093420785 |
| Mean positive margin | 0.3327025473 | 0.3268384933 | -0.0058640540 |

Paired Recall@1 evidence:

- corrected at rank 1: 57
- introduced at rank 1: 22
- corrected/introduced ratio: 2.5909
- formula-cluster bootstrap 95% CI: `[+0.0081547640, +0.0287032893]`
- strict-positive CI: yes
- `corrected > 2 * introduced`: yes (`57 > 44`)

The valid frozen claim is therefore:

> On the 1,929-query formula-role-3 confirmation graph, the ChemAware hard-negative triplet model selected by training loss improves Recall@1 over the exactly replayed official DreaMS embedding by 1.8144 percentage points, with a formula-cluster bootstrap 95% CI of +0.8155 to +2.8703 percentage points and 57 corrected versus 22 introduced rank-1 outcomes.

## Rejected late checkpoint

The stopped run's `last.ckpt` is not a release candidate:

- Recall@1 delta: +0.4147 pp
- formula-cluster bootstrap 95% CI: `[-0.8887, +1.8005]` pp
- corrected/introduced: 71/63
- Recall@3 delta: -0.1555 pp
- Recall@5 delta: -0.2074 pp
- mean cosine to official embedding: 0.3057, versus 0.7212 for `best.ckpt`

This is evidence of late-training shared-space drift. Do not replace `best.ckpt` with `last.ckpt`, and do not continue the same 301-epoch schedule as if more optimization were necessarily better.

## Scientific boundary

- This is a positive role-3 confirmation result, not a sealed outer result.
- Formula role 4 remains untouched by this run.
- The checkpoint callback monitored `Train loss` every 1,000 train steps; it did not select checkpoints using role-3 Recall@1.
- The role-3 pool was nevertheless attached as the trainer validation loader, so the final paper-level generalization claim must still come from the untouched outer protocol.
- No claim of a 3--5 pp gain is permitted from this result. The demonstrated gain is +1.8144 pp on this frozen role-3 protocol.
- The result supports direct ChemAware-to-shared-embedding transfer through chemistry-selected, identity-supervised DreaMS-native triplets. It does not establish that every chemical action or every dataset benefits.

## Code seal

The exact active source set is recorded in `CHEMAWARE_HIGH_COVERAGE_NATIVE_CODE_SHA256_20260921.txt`. The Git commit containing this document and that hash ledger is the canonical local code seal. Unrelated dirty-worktree files are excluded from the seal.

The three contract tests that must remain green are:

```text
PASS: ChemAware high-coverage triplet mining contracts
PASS: ChemAware uses native DreaMS training runtime
PASS: ChemAware V2 direct-triplet evaluation contracts
```

## Server artifacts that must be recovered verbatim

Before any server cleanup, copy the following without renaming or editing:

```text
data/validation/chemaware_high_coverage_native/run_2340524/evidence/
data/validation/chemaware_high_coverage_native/run_2340524/triplets/
data/validation/chemaware_high_coverage_native/run_2340524/official_role3_preflight.json
data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt
data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/last.ckpt
data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/role3_evaluation_after_stop.json
data/validation/chem_native_hc_2340524.out
data/validation/chem_native_hc_2340524.err
data/validation/chem_native_resume_2340721.out
data/validation/chem_native_resume_2340721.err
```

Also recover the evaluation job's `chem_native_eval_<job-id>.out` and `.err`. After recovery, compute local SHA-256 hashes and append them to a separate artifact ledger; do not edit this code/result seal retroactively.
