# Metabolomic Contrast Atlas: factorial HFD execution decision (2026-10-23)

## Decision

The next biology experiment is the already frozen three-study factorial HFD
wave, not another small disease-pair rescue.  No performance result is claimed
yet.  Raw abundance, MS/MS response scores and the registered reciprocal result
remain unopened at the time of this decision.

This panel is more informative than the corrected IBD handoff slice and the
two small cisplatin studies because it contains a repeated **genotype-by-diet
interaction**, not merely two studies with similarly named disease groups.
The biological response itself is the query object.

## Frozen denominators and evidence role

- 3 public studies and 3 registered source families;
- 67 independent animals;
- 11 analytical LC-MS/MS panels;
- 244 panel-sample mzML files;
- 3,586,894,955 deposited bytes (3.340556 GiB);
- 5 directed response objects per study, hence 15 study-response objects;
- 10 external-study candidates per query;
- 450 frozen score rows across `all`, `known` and `unresolved` worlds once the
  pre-truth index is built.

All three studies are development/consumed.  `MTBLS135` and `MTBLS3316` are
separate experiments using the same Hi-MYC model and research lineage; they
are not independent-laboratory confirmation.  `MTBLS2085` is an NQO1-by-HFD
out-of-family challenge and is not assigned a favorable or unfavorable truth.

## Scientific test

Within each panel, recurrent MS2-supported anchors and their MS1 abundance are
constructed without genotype or diet labels.  Conservative coelution,
mass-gap, abundance-correlation and ratio-stability rules collapse likely
redundant ions.  A 2 x 2 factorial model then estimates:

1. HFD minus control in the reference genotype;
2. HFD minus control in the modified genotype;
3. modified minus reference under control diet;
4. modified minus reference under HFD;
5. genotype-by-HFD interaction.

The primary test asks whether the Hi-MYC interaction from each of
`MTBLS135` and `MTBLS3316` retrieves the other experiment's interaction from
all ten external-study candidates.  Both reciprocal queries must be rank 1,
must beat every wrong response type in the counterpart study, must pass the
pre-frozen maximum-candidate permutation null with family-wise one-sided
`P <= 0.025`, and must retain the result in at least 80% of relevant
single-panel deletions.

Chemistry-only and magnitude-only scores are required falsifications.  A
favorable all-entity result would establish only a replicated-model proof of
principle.  It would not establish an unresolved-entity increment; that needs
a complete frozen identity partition and matched `known`/`unresolved`
analysis.  It would also not establish a general external biology result.

## Why the other immediately available candidates were not promoted

- Corrected IBD: 662 response entities but only 10 public-response validation
  opportunities and zero fully validated unresolved candidates in the current
  slice.  It remains an interface/error-analysis panel.
- ST002390/ST002532 cisplatin kidney: 5 versus 5 animals per arm, 25 versus
  20 mg/kg cisplatin and shared institutional/authorship lineage.  This is a
  dose-tolerant development comparison, not an exact independent repeat.
- ST001336/ST002956 HFD: useful known/reported-metabolite positive control with
  44 and 90 animals, but the existing cross-study join is name-based and does
  not test unresolved MS/MS evidence.
- Existing two-accession HFD and three-source IBD edge tests: neither passed
  the frozen cross-project unresolved response-replication null.

## Execution and leakage protection

`tasks/submit_factorial_hfd_full_chain.sh` submits one dependency chain:

1. resume-safe download of all 244 mzML files;
2. exact byte, mzML-envelope and SHA-256 verification against the frozen
   manifest and per-batch receipts;
3. phenotype-blind panel construction, ion-family collapse and factorial
   effect estimation;
4. polarity-matched phenotype-blind identity partitioning: positive panels use
   the frozen GNPS Gold/Silver `[M+H]+` control and negative panels use only
   InChIKey-bearing `[M-H]-` entries in the consumed GNPS-source negative
   library; absence of a hit is never called novelty;
5. frozen official-DreaMS spectral encoding;
6. construction of all 15 response objects and all-query scores;
7. separate 1,999-permutation maximum-candidate nulls and single-panel
   deletion analyses for `all`, `known` and `unresolved` evidence worlds;
8. 16 pre-truth unresolved resamples matched to each panel's strict-known
   ion-family representative count, with the same sampled entities reused
   across all five response types;
9. hash freeze of every pre-truth artifact;
10. automated opening of only the two registered reciprocal Hi-MYC queries.

Because the final opening is dependency-bound to the hash-freeze job, there is
no opportunity to inspect scores and then change the response method, candidate
universe, null or robustness threshold.  A failed upstream job prevents all
downstream opening.

## Current implementation verification

On 2026-10-23, the original factorial-HFD method tests plus the new download,
polarity-aware identity, three-world calibration and count-matched gates passed
`15/15`.  This verifies executable contracts only;
it is not a biological success result.
