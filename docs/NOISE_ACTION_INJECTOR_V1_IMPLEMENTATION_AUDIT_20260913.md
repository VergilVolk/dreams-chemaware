# Noise Action Injector V1 implementation audit

Date: 2026-09-13  
Verdict: PASS for faithful V10 engineering reuse  
Scientific promotion verdict: not evaluated by this refactor

## What is now fixed

`tasks/noise_action_injector_v1.py` is the sole versioned implementation
boundary for V10 safe-exact injection. The direct trainer constructs one
`ActionInjectorV1`, sends the three clipped gradient ledgers through
`prepare()`, and delegates the one real AdamW step plus materialization and
moment reconciliation to `step_and_inject_()`.

The reusable boundary is action-type neutral. It has no imports from action
routing, spectrum generation, rank evaluation, or teacher code and exposes no
action-source-specific parameter.

## Exactness audit

The implementation preserves the completed V10 transition:

- same pre-step AdamW state for combined, noncorrective, and protective
  counterfactuals;
- same FP32 in-place AdamW arithmetic in virtual updates;
- same per-parameter-group V10 hard-floor and analytical exact-fraction
  composition;
- same safety-repaired counterfactual baseline for attribution;
- same single ordinary optimizer step;
- same `before - composed_update` materialization;
- same first-moment-only reconciliation;
- same actual-combined-gradient second moment and advanced step number;
- same `1e-7` norm-cap comparison tolerance and existing report gates.

During integration review, an intermediate wiring error assigned the raw
noncorrective update to the attribution baseline. It was corrected before
acceptance: the trainer now uses
`prepared_action_injection.composition.counterfactual_baseline_updates`, so the
protective-floor repair remains counterfactual and is not falsely counted as
corrective signal. A static regression rejects a return to the raw baseline.

## Golden equivalence

`tasks/test_noise_action_injector_v1.py` independently executes the legacy V10
sequence and V1 on identical two-group AdamW states. It requires exact
`torch.equal` equality for:

- all three virtual AdamW update ledgers;
- groupwise composed updates;
- safety-repaired counterfactual updates;
- ordinary AdamW displacements;
- final parameter tensors;
- AdamW `exp_avg`, `exp_avg_sq`, and step tensors.

It also verifies the reconciliation report, semantic-contract hash, frozen
dependency hashes, action-agnostic public signature, exact AdamW group
membership, configuration-bound receipts, and stale-receipt rejection.

## Tests executed locally

- `test_noise_action_injector_v1.py`: PASS, 7 tests
- `test_noise_corrected_update_arbitration_v10.py`: PASS, 5 tests
- `test_summarize_noise_corrected_v10_safe_exact_canary.py`: PASS, 2 tests
- `test_noise_corrected_best_action_v10.py`: PASS, 5 tests
- `test_noise_corrected_best_action_v6.py`: PASS, 11 tests
- `test_noise_corrected_direct_v3_core.py`: PASS, 22 tests
- `test_noise_corrected_direct_v3_trainer.py`: PASS, 26 tests
- `test_noise_corrected_gradient_locality_v8.py`: PASS, 4 tests
- `test_noise_corrected_update_arbitration_v4.py`: PASS, 16 tests
- `test_noise_corrected_fullgraph_evaluation.py`: PASS, 4 tests
- `test_noise_corrected_v10_safe_exact_sbatch.py`: PASS, 6 tests
- Python compilation of injector, trainer, and changed tests: PASS
- complete V10 SHA256 source-closure verification: PASS

## Frozen identifiers

- semantic contract SHA256:
  `e7a742a9a8a695a439d2437f502e52a77c9f6bf8e82c793c6e618fd097bce579`
- injector source SHA256:
  `21e881f00942a61c2a192a4358806379215d55026f87c934c04f975da1887931`
- V10 arithmetic dependency SHA256:
  `00b587506b6ac5755c6ba3f1750c879eca993981ad191555951dd8d0552a3193`
- V4 materialization/reconciliation dependency SHA256:
  `6c11387102eb5edb54783d3a0f0ea1ed7ed3b1d63976249d0cde51e6039c3473`
- V10 source-closure manifest SHA256:
  `5a9b81e5563af20323b75060db387fdd89e61b4e23cdb33004edfdc9ec125bab`

The V10 Slurm entrypoint pins that manifest hash, snapshots the injector spec,
and runs the V1 golden test before GPU work.

## Limit of this verdict

This PASS means the injector has been modularized without changing the V10
optimizer-space mechanism. It does not convert V10's scientific result into a
4–5 pp improvement and does not validate any new action panel. The next stage
must reuse this frozen injector while separately restoring and auditing the
best historical action set under the full held evaluation protocol.
