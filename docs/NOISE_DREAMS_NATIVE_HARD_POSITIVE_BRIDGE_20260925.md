# Noise native DreaMS hard-positive-only continuation

## Scientific correction from run 2344688

Run 2344688 established two different facts that must not be mixed:

- the registered action view is useful as a hard positive: targeted-minus-control
  mean margin was `+0.136544`, and targeted won on `92.463%` of materialized
  action views;
- the added clean-to-action deployment bridge was harmful: targeted-minus-control
  mean margin was `-0.066740`, and targeted won on only `21.201%` of views.

The repaired experiment therefore retains only the relation supported by the
replay. It does not lower the learning rate, invent a new loss, filter actions
using held outcomes, or add another optimization mechanism.

## Frozen training relations

The model, preprocessor, loss and optimizer remain native DreaMS. Noise changes
only the spectra occupying the native triplet roles.

Each representable effective action contributes exactly one triplet:

```text
anchor   = same-query registered action view
positive = independent measured spectrum of the same molecule
negative = exact false candidate identified by that action
```

Each unrepresentable effective action contributes exactly one measured fallback:

```text
anchor   = measured clean query
positive = exact measured true spectrum
negative = exact false candidate identified by that action
```

The harmful relation below is diagnostic only and is forbidden from the
training ledger:

```text
clean query -> action view -> false candidate
```

The ordinary clean-identity stream contributes one event per selected compact
train query. It preserves clean retrieval without pretending to be Noise
signal.

## Identity and provenance boundary

An action view keeps the measured query precursor and its frozen
action-id-to-query-row provenance. Registered intensity changes, fragment
removal, and candidate/positive-guided grafted or shifted peaks are the Noise
counterfactual payload; they are not reclassified as measured peaks. Exact
positive and negative rows are independently checked against the frozen graph.

The causal control is the ledger's registered same-query `control_spectra`.
Cross-query tensor shuffling remains forbidden because it changes molecular
identity rather than action content.

## Native optimization and bounded dose

- repository `ContrastiveSpectraDataset`, sampling one positive and one negative;
- native 1024-dimensional `ContrastiveHead`;
- official slim backbone/head reconstructed with the already validated
  `load_base_model` interface before constructing the native head;
- cosine triplet hinge, margin `0.1`;
- native Adam, learning rate `5e-6`, weight decay `0`;
- batch size `4`, FP32, backbone unfrozen at epoch `0`;
- exactly one pass and exactly one event per effective action;
- one query cannot occur twice in an optimizer batch;
- no query-dose equalization, no action replay, and no gradient-bearing
  equalization fillers;
- only zero-to-three action-free clean events may be repeated to make the final
  batch divisible by four;
- targeted and same-query control use the same seed, relations, order, step
  count and optimizer configuration.

The compact clean base retains every action query plus one deterministic query
from each remaining outer-train formula. With the current frozen artifacts the
expected scale is about `33,575` base events and `8,394` optimizer steps, versus
`29,816` steps in run 2344688. A hard preflight gate rejects any schedule above
`9,000` steps.

## Decision boundary

The registered promotion threshold remains at least `+2 pp` Recall@1 versus
both official DreaMS and mature E8, with strictly positive formula-cluster CIs,
positive risk-net, and a strictly positive targeted-minus-control paired formula
CI. This is a promotion threshold, not a promised result.

The sole server entry point is:

```bash
sbatch tasks/run_noise_dreams_hard_positive_native_2gpu.sbatch
```
