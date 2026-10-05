# Noise Action Injector V1 — frozen V10 optimizer-boundary contract

Date: 2026-09-13  
Status: implemented, integrated, CPU golden-equivalence tested  
Scope: noise direct fine-tuning only

## Decision

The engineering-successful V10 injection mechanism is now frozen as
`ActionInjectorV1` in `tasks/noise_action_injector_v1.py`. Future action
families must reuse this interface rather than copying or rewriting the
optimizer-boundary code. A numerical or state-transition change requires a
new injector version; it must not be introduced by editing V1 in place.

This freezes an injection mechanism, not a scientific efficacy claim. V10's
held comparison did not establish a 4–5 pp embedding improvement. Action
selection and action-to-clean semantics remain upstream work and must be
tested separately.

## Exact boundary

The injector starts after all upstream action construction and gradient-space
arbitration. It accepts three aligned, already-clipped gradient ledgers at one
unchanged AdamW state:

1. `clipped_combined_gradients`: the actual combined gradient that the real
   optimizer step will consume;
2. `clipped_noncorrective_gradients`: protective plus admitted projected
   robust/harmful auxiliary gradients, without the corrective increment;
3. `clipped_protective_gradients`: the protective-only safety reference.

It is action-family agnostic. It does not receive spectra, action IDs, source
families, recipes, ranks, teacher embeddings, or distillation targets. N, P,
A4, E-series, or future actions are interchangeable at this boundary if they
produce the same three-ledger contract.

The injector does not own action admission, loss construction, dose
scheduling, corrective recycling, inner semantic projection, outer risk
projection, or held evaluation. Those are explicitly outside V1 and cannot be
misrepresented as properties of the injector.

## Frozen sequence

For every action-active optimizer step, V1 performs exactly this sequence:

1. Evaluate combined, noncorrective, and protective virtual AdamW descent
   updates without changing parameters, moments, or step number.
2. For each registered optimizer parameter group independently:
   - define the corrective residual as combined minus noncorrective;
   - repair the noncorrective baseline to the registered protective-component
     floor;
   - assign that safety repair to the counterfactual baseline, never to the
     action-attributable numerator;
   - remove only the part of the corrective residual opposing the protective
     axis;
   - analytically solve the positive corrective coefficient for the exact
     attributable-update fraction;
   - fail closed if the protective floor, exact fraction, or update-norm cap
     fails in any group.
3. Snapshot parameters.
4. Execute exactly one ordinary `torch.optim.AdamW.step()` using the actual
   combined gradient.
5. Measure the ordinary parameter displacement against the virtual combined
   update.
6. Overwrite parameter values with `before - composed_descent_update`.
7. Reconcile only `exp_avg` to the realized materialized displacement.
8. Preserve the advanced step number and `exp_avg_sq` from the actual combined
   gradient; do not create a second optimizer step.

The clean-control arm remains a no-op for injection and takes its ordinary
AdamW step.

## Frozen numerical details

The V1 semantic-contract hash is:

`e7a742a9a8a695a439d2437f502e52a77c9f6bf8e82c793c6e618fd097bce579`

Non-configurable numerical constants:

- protective-floor FP32 guard: `1e-7`;
- protective-floor acceptance tolerance: `1e-6`;
- exact-fraction acceptance tolerance: `2e-6`;
- update-norm-cap comparison tolerance: `1e-7`;
- stable AdamW first-moment reconstruction threshold: `1e-6`;
- existing trainer virtual-AdamW reproduction threshold: `1e-3`.

The three explicit experiment thresholds remain immutable configuration values
on each injector instance. The registered V10 values are target attributable
fraction `0.25`, minimum protective-component retention `0.90`, and maximum
per-group update-norm ratio `1.50`.

## Anti-drift controls

- `ActionInjectorV1Config` is frozen and exposes no tolerance overrides.
- V1 checks that the optimizer parameter set and named group partition are
  complete and exact.
- A prepared receipt records parameter identities and AdamW step numbers; it
  refuses application after an intervening optimizer step or on another model.
- The receipt also pins the injector configuration and exact named AdamW group
  membership; it cannot be applied through another threshold set or a swapped
  head/backbone partition.
- V1 runtime-pins the exact V10 composition and V4 materialization/
  reconciliation source hashes.
- The trainer records the semantic-contract hash, dependency hashes,
  configuration, and V1 source hash in its report.
- `tasks/test_noise_action_injector_v1.py` compares the complete new lifecycle
  against the legacy V10 sequence with `torch.equal` for virtual updates,
  composed updates, final parameters, first moments, second moments, and step
  state.
- The V10 source snapshot manifest includes the injector and golden test.

## Public reuse interface

```python
injector = ActionInjectorV1(ActionInjectorV1Config(
    target_attributable_fraction=0.25,
    minimum_protective_component_retention=0.90,
    maximum_update_norm_ratio_to_original=1.50,
))
prepared = injector.prepare(
    optimizer,
    trainable_parameters,
    clipped_combined_gradients=combined,
    clipped_noncorrective_gradients=noncorrective,
    clipped_protective_gradients=protective,
    parameter_group_positions=parameter_groups,
)
applied = injector.step_and_inject_(optimizer, trainable_parameters, prepared)
```

The caller must not call `optimizer.step()` between `prepare` and
`step_and_inject_`; V1 checks and rejects that stale receipt. The second method
owns the one real optimizer step and the materialization/reconciliation
transition.

## Acceptance evidence

Before this module may be used with a new best-action panel, all of the
following must pass locally:

1. Python compilation of injector and trainer;
2. frozen dependency/hash check;
3. V1 golden tensor/state equivalence suite;
4. existing V10 arbitration tests;
5. existing V4 AdamW materialization/reconciliation tests;
6. existing direct-v3 trainer tests;
7. source-snapshot manifest and Slurm static tests.

Only after these gates pass should a server job reuse prior best actions. A
passing injector gate proves faithful signal delivery; it does not guarantee
that the supplied actions improve held retrieval metrics.
