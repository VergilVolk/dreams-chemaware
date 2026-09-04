# BioAware v1 on MTBLS13729: failure audit (2026-08-28)

## Verdict

The evaluation pipeline completed correctly, but this pilot cannot support a
BioAware accuracy claim.  The frozen spectral baseline is correct for 20/21
queries.  BioAware changes one query, correcting none and introducing one
error.  The only baseline error has no reaction-network evidence for either
candidate.

This is a negative result for the current global one-hop override rule.  It is
not evidence that biochemical context is generally useless.

## Formal result

- Queries: 21 (18 truth-formula clusters).
- Baseline Recall@1: 20/21 = 0.95238.
- BioAware Recall@1: 19/21 = 0.90476.
- Corrected / introduced: 0 / 1.
- Intervention rate: 1/21 = 0.04762.
- Formula-cluster bootstrap delta CI: [-0.1667, 0].
- Degree-preserving decoy delta: 0 in all 10 repeats.
- Formal gate: failed.

With only one baseline error, the maximum possible improvement is one query
(+4.76 percentage points).  Even a perfect 1/0 corrected/introduced outcome
cannot establish significance with a two-sided exact McNemar test.  The pilot
was therefore underpowered by construction for an improvement claim.

## The sole introduced error: feature neg_rp:1705

The frozen Level-2a-supported spectral annotation is 4-aminobutanoic acid
(GABA; IK14 `BTCSSZJGUNDROE`).  BioAware changes it to (2S)-2-aminobutanoate
(`QWCKQJZIFLGMSD`).

| candidate | role | spectral score | support spectra | network support |
|---|---:|---:|---:|---:|
| 4-aminobutanoic acid | frozen pseudo-truth / spectral top | 0.928094 | 17 | 0 |
| 2-aminobutanoic acid | BioAware final top | 0.902365 | 3 | 0.186671 (2 paths) |
| (S)-3-aminobutyric acid | third candidate | 0.824318 | 3 | 0 |

The spectral margin is 0.025728.  Adding `0.15 * 0.186671` raises the raw fused
score of 2-aminobutanoate to 0.930366, just 0.002272 above the GABA score.
The override is therefore numerically fragile and depends completely on the
two reaction paths below.

### Reaction paths

1. RHEA:66116 connects the high-confidence glutamine seed (`neg_rp:705`) to
   (2S)-2-aminobutanoate through
   `2-oxobutanoate + L-glutamine = 2-oxoglutaramate + (2S)-2-aminobutanoate`.
2. RHEA:66124 connects the high-confidence histidine seed (`neg_rp:1075`) to
   (2S)-2-aminobutanoate through
   `2-oxobutanoate + L-histidine = 3-(imidazol-5-yl)pyruvate + (2S)-2-aminobutanoate`.

The paths are chemically coherent, but neither demonstrates that feature 1705
is 2-aminobutanoate.  Rhea is location-, tissue-, and species-independent, and
these two entries have undefined physiological direction in the cache.  The
current global seed pool also does not require co-detection, abundance
covariation, the shared co-substrate 2-oxobutanoate, enzyme presence, or a
sample-specific module.  Global one-hop adjacency is therefore insufficient
for an identity override.

The reference label itself is not authentic-standard truth.  Feature 1705 may
contain unresolved/co-eluting aminobutanoate isomers, but the present data do
not establish that interpretation.  This row must be treated as an ambiguity
case, not used to tune a post-hoc threshold.

## The sole baseline error: feature neg_rp:1119

The frozen pseudo-truth is hypoxanthine (`FDGQSTZJBFJUBT`, spectral score
0.867746; 18 supporting spectra).  DreaMS ranks threonic acid
(`JPIJQSOTBSSVTP`, 0.885019; 8 supporting spectra) first by a margin of
0.017273.  Neither candidate receives a leave-query/leave-truth network path,
so BioAware cannot alter this error.

This establishes a coverage failure, not a gating failure: the only baseline
error lies outside the evidence support of BioAware v1.

## Evidence-state audit

- No network evidence: 13/21 queries.
- Spectral/network agreement: 6/21.
- Strong spectral conflict, abstained: 1/21.
- Network-supported override: 1/21 (the introduced error).
- Queries with any supported candidate: 8/21.

## Required redesign

1. Freeze this result; do not optimize thresholds on these 21 queries.
2. Keep one-hop Rhea paths as explanations, not sufficient override evidence.
3. Require sample-context evidence before identity intervention: independent
   neighboring seeds, reaction-complete support where possible, co-detection or
   abundance covariation, and human/tissue-relevant reaction evidence.
4. Separate an abstention/ambiguity output from a forced Top-1 replacement.
5. Evaluate correction on a much larger identity-known benchmark containing at
   least several dozen baseline errors with graph coverage.  Use MTBLS13729 for
   biological-context application and orthogonal evidence aggregation, not as
   the sole accuracy benchmark.

