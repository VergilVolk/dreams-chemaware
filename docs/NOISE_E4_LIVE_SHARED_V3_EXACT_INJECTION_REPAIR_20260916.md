# Noise E4 live-shared V3 exact-injection repair

## Root cause

The failed V3 run did not establish that the complete live E4 action relation
was ineffective.  The production injector had silently changed the already
validated signal-delivery contract:

- `0.25` was implemented as a maximum action-update fraction, not a target;
- action updates were forbidden from being amplified;
- therefore a natural 5% action displacement remained 5%, while the gate only
  required it to be non-zero;
- the registered `0.90` E4-retention argument from the Slurm command was not
  passed into the V3 injector; V3 substituted an internal projection rule of
  at least 1.0.

Consequently, the run could pass every V3 gate while recreating the original
optimizer-boundary signal loss.  This is a core training implementation error,
not evidence against the action bank or the full live E4 objective.

## Repaired production contract

The action objective remains unchanged:

- clean, action, positive and negative spectra are encoded in one shared graph;
- all four roles receive ranking gradients;
- clean rank, action rank, symmetric consistency, margin floor and preservation
  retain the E4 coefficients;
- action rank is never hard-gated;
- query-equal scheduling and the complete seven-source action panel remain.

Only the optimizer-boundary defect is repaired:

- E4 and action streams keep independent AdamW moments;
- each active head/backbone group must reach an exact action-attributable update
  fraction of `0.25`, whether this requires scaling the natural action update up
  or down;
- the final update must retain at least `0.90` of the E4 projection;
- its norm may not exceed `1.50` times the independent E4 update;
- zero-action steps remain bitwise ordinary E4 AdamW steps.

The regression suite now includes an intentionally tiny natural action update.
It fails the former ceiling-only implementation and passes only when the weak
signal is filled to exactly 0.25.

## Claim boundary

This repair removes a demonstrated signal-delivery defect.  It does not by
itself establish a 4--5 pp held improvement; that claim still requires the
targeted-versus-shuffled full metric panel and formula-cluster paired intervals.
