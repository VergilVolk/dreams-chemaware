# Enveda-180 scientific-validity audit (2026-10-12)

## Verdict

The repeatedly failing v1/v2 construction should not be rescued and used as the
final external result.  Its main defects were scientific, not checksum-related:
the reference library contained only molecules that could themselves become
queries, the true molecule was forced into a candidate set even when it failed
the candidate rule, reference collision energies were allowed to vary, and a
256-candidate truncation was applied after protecting the truth.  Each choice
can make retrieval artificially easy or give the positive privileged access.

The corrected v3 builder is an acceptable **model-blind component benchmark**.
It is not yet an eligible final test of a “grand unified ranker,” because no
single frozen ranking and abstention function currently combines Noise V1,
WSE, P2b, ChemAware and BioAware.  Enveda labels must remain unopened until the
paper either (a) freezes such a function and its primary contrast, or (b)
explicitly narrows the claim to component and layered-workflow validation.

## Corrected experimental object

- Source: Enveda-180 filtered MGF, 1,175,202 spectra, 285,632,591 bytes, MD5
  `6d06a2c916aefe1c0b2d86bf40628441`, SHA-256
  `04036a4c735c9cb2605623e14c77afc108059559f24299e279a64c5b0dedba34`.
- Query condition: one 60 eV spectrum per eligible identity/adduct.
- Reference condition: up to three 20 eV spectra per identity/adduct.
- Reference library: every eligible identity with a 20 eV spectrum, including
  identities without a 60 eV query.
- Candidate tasks: complete 10-ppm precursor window and strict same-formula.
- Candidate cap: none.  The truth must pass the same rule as every negative and
  is never inserted after candidate generation.
- Leakage exclusions: consumed identities and spectra for identity panels;
  consumed identities, formulas and spectra for formula panels; cross-identity
  collisions under both primary and secondary spectrum hashes.
- Query sampling: fixed hash order, at most 25,000 queries per panel, constructed
  before model scores exist.

The two disjointness levels answer different questions.  Identity-disjoint is
the larger external transfer panel but allows formulas seen during development.
Formula-disjoint is the stricter confirmation panel.  Neither name means that
Enveda is independent of every possible unlabeled spectrum ever seen during
foundation-model pretraining; the defensible claim is disjointness from the
enumerated development/fine-tuning assets.

## What Enveda can and cannot validate

Enveda-180 is a large targeted library of drug-like compounds measured under
controlled collision energies.  It is well suited to cross-condition spectral
retrieval, candidate ranking, near-isomer competition and transported open-set
calibration.  It is not an untargeted biological sample, so it cannot validate
BioAware event context, entity abundance reproducibility, phenotype association
or a biological-discovery claim.

The automatic score bundle currently contains official DreaMS, Noise V1, WSE
and frozen P2b.  ChemAware V2 is evidence-only in the present workflow and
BioAware is unavailable on this dataset.  Therefore a table comparing those
four methods is a component ladder, not evidence that all modules form one
algorithm.  Historical module gains must not be added arithmetically.

## Open-set interpretation

GNPS thresholds calibrated at nominal 5% FDR may be transported unchanged.
Their Enveda result must be reported as observed risk and coverage at a
GNPS-calibrated threshold.  They may only be called “5% FDR on Enveda” if the
observed Enveda estimate and its uncertainty support that statement.  No
threshold may be selected after Enveda labels are opened.

## Required pre-opening decision

Before any Enveda metric is calculated or viewed, record one of these paths:

1. **Unified-algorithm path:** define one deterministic candidate score and one
   abstention rule, specify how missing ChemAware/BioAware evidence behaves,
   name one primary panel, endpoint and baseline, then freeze them.
2. **Layered-workflow path:** state that WSE is the primary identity output,
   Noise is a transferable coordinate, P2b is a local-competition analysis,
   ChemAware is explanatory evidence and BioAware is out of scope.  Enveda then
   validates the layers separately, not a single unified ranker.

The current contract deliberately sets `opening_authorized=false`.  Generating
score-blind embeddings/scores is permitted; evaluating them against labels is
not.  This is the one scientifically meaningful stop condition in the current
pipeline.

## Remaining limitations to disclose

- The planned scaffold-disjoint hard subset has not been implemented and must
  not be claimed.
- “Independent positive” means a distinct spectrum at a different collision
  energy, not an independent compound or biological replicate.
- Primary-adduct filtering narrows the deployment domain.
- A broad SOTA claim requires runnable comparators on the identical candidate
  graph; literature numbers from different libraries are not comparable.
- The final biological increment still requires real samples and event-level
  context outside Enveda.

## Implementation status

The v3 builder, six score-blind panels (four closed, two open), all-panel
encoding/scoring, and scientific contract tests are implemented in `tasks/`.
The old v1/v2 benchmark directories must remain historical artifacts and must
not be opened as the final test.
