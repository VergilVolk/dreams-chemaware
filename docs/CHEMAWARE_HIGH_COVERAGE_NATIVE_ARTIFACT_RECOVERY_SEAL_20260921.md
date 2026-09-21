# ChemAware high-coverage native artifact recovery seal

Date: 2026-09-21

Status: `CORE_ARTIFACTS_RECOVERED_AND_HASHED / READ_ONLY_LOCAL_SEAL`

## Recovered roots

```text
data/validation/chemaware_high_coverage_native/run_2340524/
data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/
```

The complete local inventory contains 19 files. Every file's byte size and SHA-256 digest is frozen in `CHEMAWARE_HIGH_COVERAGE_NATIVE_ARTIFACT_SHA256_20260921.txt`.

## Transfer repair and integrity checks

Three downloaded checkpoints retained the transfer-client suffix `.filepart` even though transfer activity had stopped:

```text
run_2340524/training/best.ckpt.filepart
run_2340721_resume_2340524/training/best.ckpt.filepart
run_2340721_resume_2340524/training/last.ckpt.filepart
```

Before renaming, each file was opened as a PyTorch ZIP container and all 272 archive members passed CRC validation with `bad_member=None`. Only after that validation were the files renamed to their canonical `.ckpt` names. No checkpoint content was rewritten.

## Content-level validation

- `role3_evaluation_after_stop.json` parsed successfully.
- Evaluation status: `CHEMAWARE_V2_DIRECT_TRIPLET_SHARED_EMBEDDING_EVALUATION_COMPLETE`.
- Formula role: 3.
- Outer role 4 accessed: false.
- Queries: 1,929.
- Unique spectrum rows encoded: 24,218.
- Official frozen-ledger rank mismatches: 0.
- ChemAware best Recall@1 delta: +0.0181441161.
- Formula-cluster bootstrap 95% CI: `[+0.0081547640, +0.0287032893]`.
- Corrected/introduced at rank 1: 57/22.
- `confirmation_triplet_evidence.npz` parsed successfully with 1,929 query rows and four arms.
- `train_pool.npz` parsed successfully with 7,688 anchor triplets.
- `val_pool.npz` parsed successfully with 3,709 anchor triplets.

## Canonical release candidate

```text
data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt
SHA-256: 09c419dcabf2ff424fe38bb33eaefcd25ccab0b6d125daf03b7959c25fff5838
Bytes: 1241317018
```

The corresponding `last.ckpt` remains preserved as negative late-training evidence and must not replace the release candidate.

## Relationship to the code seal

- Code/result commit: `af37d73`
- Code/result tag: `chemaware-role3-positive-20260921`
- Code ledger: `CHEMAWARE_HIGH_COVERAGE_NATIVE_CODE_SHA256_20260921.txt`
- Scientific result record: `CHEMAWARE_HIGH_COVERAGE_NATIVE_ROLE3_SEAL_20260921.md`

This recovery seal adds artifact hashes without changing the prior code/result seal.

## Remaining non-core provenance item

The Slurm `.out` and `.err` logs were not present in the recovered directories or the local `data/validation` root. Their absence does not invalidate the checkpoint/result/evidence hashes, but they should be copied later if a complete operational log archive is required.
