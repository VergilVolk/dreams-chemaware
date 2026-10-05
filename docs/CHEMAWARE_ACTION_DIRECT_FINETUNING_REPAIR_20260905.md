# ChemAware action library and direct fine-tuning repair — 2026-09-05

## Outcome

ChemAware now has two explicitly separated knowledge layers:

1. **Empirical structure–spectrum associations.** They must have an executable
   parent SMARTS predicate, an acquisition context, a chemically specified
   observation, within-formula structure-negative controls, formula-disjoint
   confirmation, a positive clustered interval, and multiplicity control.
2. **Spectrum actions.** They may change intensities of peaks already observed
   in the training spectrum, but may not synthesize a new m/z. An association
   is not automatically an action, and an action is not automatically allowed
   to update DreaMS.

The deployable model is still the official candidate-independent DreaMS
encoder. At inference it receives one clean spectrum and no structure,
candidate list, rule match, ICEBERG score, or action.

## Rule-library result

- 335 legacy core entries remain quarantined because they have no executable
  parent-structure prerequisite.
- 3,151 MassBank record-derived entries remain quarantined because support=1
  record memories are not transferable chemical rules.
- 120 mass observations remain an observation dictionary, not a rule library.
- 85 RDKit functional-group SMARTS were frozen as parent-predicate hypotheses;
  they were not labelled fragmentation mechanisms.
- The complete 85 × 120 discovery screen used 83,619 spectra, 9,854 molecular
  identities, and 6,220 formulas. Formula folds 0–1 were used for discovery,
  fold 2 for one confirmation, and folds 3–4 were not inspected.
- Twenty-one discovery hypotheses entered confirmation. Two empirical
  associations passed the fixed support, sign-flip, BH-FDR, effect, and
  formula-cluster interval gates. Both concern the observed C2H6N diagnostic
  ion at m/z 44.0495.
  - no-hydrogen nitrogen predicate `[N&H0,n&H0]`: confirmation effect +0.0726,
    70 matched formulas, 95% CI [+0.0279, +0.1226], BH q=0.0252;
  - piperazine predicate `N1CCNCC1`: confirmation effect +0.1588, 24 matched
    formulas, 95% CI [+0.0604, +0.2596], BH q=0.0315.
- These two rules fail the independent action-coverage gate on the 700-query
  panel: discovery has 10 eligible identities / 9 formulas; confirmation has
  6 identities / 5 formulas. They are therefore stored as knowledge but are
  forbidden as a current fine-tuning action source.

## Maximum bounded action search

The remaining action candidate uses the true training structure only to form
ICEBERG true-structure minus hardest-same-formula-negative evidence. It searches
21 global settings:

- conflict-peak attenuation: 3 strengths × 3 top-k values;
- supported-peak amplification: 3 strengths × 3 top-k values;
- continuous signed reweighting: 3 strengths.

Every setting has candidate-swapped and peak-evidence-permuted controls. Only
formula folds 0–1 select a single global setting. Only that setting is then
applied to fold 2 for confirmation. No action is generated for fold 3, which is
the embedding-development evaluation fold, or fold 4, which remains the final
outer fold. A training action must be individually corrective under the correct
structure and beat both controls. Robust/no-op actions are not included merely
to increase sample count.

The local preflight passed on 700 queries / 447 formulas. It plans 17,787 frozen
encoder spectrum evaluations. No weights are updated by this screen.

## No-distillation direct objective

For a qualified training identity, the same trainable DreaMS encoder maps:

- the clean spectrum to `z_clean`;
- its qualified intensity-only action view to `z_action`;
- every real reference spectrum in the full candidate list to `z_reference`.

The loss is exactly:

`L = CE(full_candidates | z_clean) + CE(full_candidates | z_action) + CE(clean_safety_stream)`

All three terms use the same molecular identity label and the same complete
real-spectrum candidate list. The direct mode rejects any non-zero teacher
score loss, teacher embedding loss, clean/action embedding consistency,
official-embedding preservation, official-margin floor, or peak-token teacher
loss. ICEBERG creates a training-only data view; it supplies no numerical loss
target.

## Causal fine-tuning pilot

If and only if the frozen action bank passes, the bounded pilot runs four arms
with exactly matched identities, candidates, optimizer schedule, and update
count:

1. clean duplicate (continuation control);
2. correct-structure action;
3. candidate-swapped action;
4. peak-evidence-permuted action.

The array is sequential (`0-3%1`), one GPU per arm, two epochs, at most 512
action identities, and at most 8 GPU-hours in total. A claimed chemical gain
requires a strict-positive formula-clustered confidence interval for the
correct arm both absolutely and versus every control, while global inner-fold
retrieval and safety gates remain non-negative. The outer fold is not evaluated.

## Current evidence boundary

There is not yet a measured embedding improvement from this repaired route.
The result currently established is methodological and data-level:

- two confirmed empirical associations, but insufficient rule-action coverage;
- a passed CPU/static preflight for the 21-setting structure-differential
  action bank;
- a directly optimized shared embedding objective with all distillation paths
  disabled;
- a matched four-arm causal decision rule that cannot convert common continued
  training gains into a chemical-action claim.

## Server order

First run the frozen action qualification:

```bash
sbatch tasks/run_chemaware_direct_action_bank.sbatch
```

Only when its report says `CHEMAWARE_DIRECT_ACTION_BANK_PASS`, run:

```bash
sbatch tasks/run_chemaware_direct_action_views_pilot.sbatch
```

After all four array arms finish, summarize with the actual array run folder:

```bash
sbatch --export=ALL,ACTION_VIEW_RUN=data/validation/chemaware_direct_action_views_pilot/run_JOBID tasks/run_chemaware_direct_action_views_summary.sbatch
```
