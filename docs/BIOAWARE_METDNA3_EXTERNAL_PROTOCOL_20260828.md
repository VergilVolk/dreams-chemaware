# BioAware MetDNA3-aligned external protocol (2026-08-28)

## Decision

Use the public MetDNA3 Exploris 480 release (MSV000097913) as the primary
external benchmark for DreaMS plus BioAware. It directly matches the intended
task: feature-level MS2, standard-supported seeds, a knowledge layer, a data
layer, and held-out propagation targets.

The release contains 239 open mzML files (10,700,428,871 bytes). Its file
layout yields 20 panels: five sample types, HILIC/RPLC, and positive/negative
ionization. The associated paper reports 1,652 unique Level-1 metabolites and
evaluates network propagation by using 30% of Level-1 annotations as seeds and
70% as held-out validation targets.

## Frozen split before source-data outcomes are opened

- Development: NIST urine HILIC positive and negative.
- Internal validation: NIST urine RPLC positive and negative.
- Untouched external test: the 16 BV2, mouse brain, mouse liver, and NIST
  plasma panels.

This split prevents repeated tuning on all 20 panels while preserving matrix
and instrument diversity for the final claim.

## What is reproduced versus what is ours

Reproduced from the published design:

- Level-1 seeds require MS1, RT, and MS2 standard agreement;
- 30% seeds and 70% held-out targets in ten folds;
- MS1/formula pre-mapping before network propagation;
- knowledge-layer reaction links must agree with data-layer MS2 links.

Project-specific increment:

- DreaMS provides the representation-level data-layer similarity;
- classical peak and neutral-loss evidence remains an orthogonal control;
- Rhea hyperedges are dependency corrected so several paths sharing one absent
  cosubstrate are not treated as independent evidence;
- low-confidence candidates may be overridden, while conflicts abstain;
- every transition is exposed as corrected, introduced, or unchanged.

## Claim gate

No result is called an improvement unless the untouched 16-panel test has a
positive pooled clustered Recall@1 confidence interval, non-decreasing MRR,
more corrections than introductions, and at least 12 of 16 panels are
nonnegative. Degree-preserving graph decoys must also be beaten.

The benchmark evaluates a downstream candidate expert. It must never be
reported as a new DreaMS embedding-space result.
