# Noise action-hard native DreaMS reset

Date: 2026-09-22

## Frozen decision

Noise now reuses the exact native training interface that produced the sealed
ChemAware `+1.8144 pp` development result.  The reusable component is the
training runtime, not ChemAware supervision:

- `train_chemaware_dreams_native.py`;
- native `ContrastiveSpectraDataset` dynamic one-positive/one-negative draw;
- native `ContrastiveHead`, shared backbone and official 1024-D projection;
- official spectrum preprocessing (100 peaks, precursor intensity 1.1);
- cosine hinge triplet margin 0.1;
- Adam, learning rate `5e-6`, weight decay 0, batch size 4, FP32;
- complete backbone trainable from epoch 0;
- initialization from `official_embedding_slim.pt` and native Lightning
  checkpoints.

ChemAware rules, formula roles, rule-response features and role-3 evaluation
are not imported.

## Noise-to-triplet translation

The previous action-anchor design was scientifically wrong for deployment:
it optimized `action -> positive / negative` and assumed that relation would
cross an unproven action-to-clean interface.  The replacement uses only
measured spectra:

```text
anchor    = measured clean query spectrum
positive  = every other measured reference spectrum of the true identity
negative  = every measured reference spectrum of the false candidate molecule
            selected by the Noise action
```

Noise therefore changes only hard-negative curriculum content.  An action
tensor, teacher score, action margin or action embedding never enters the
model or loss.

Each action query also contains one official-hard negative molecule, providing
the same stable base used by the successful ChemAware curriculum.  Noise adds
each distinct `action_hard_negative_row` molecule as its own native triplet
event.  All relations are identity-audited against the corrected graph.

## Dose and coverage

The optimizer unit is the same one used by the successful ChemAware run: one
native event per unique `(query, targeted negative molecule)` relation.  The
shared official-hard relation is included once. Every distinct Noise-selected
target molecule is therefore an explicit event presented to the native loss.
Multiple action recipes or sources that rediscover the same relation become
aliases rather than extra dose. Query dose is proportional to distinct
constraints, not raw action rows, and its min/median/max distribution is
reported rather than hidden.
All 32,114 qualified seven-source action IDs remain in an alias ledger mapping
them to the event they support, but rediscovery of the same boundary cannot
multiply optimizer dose. A negative event retains all measured reference
spectra for its false molecule; a positive event retains all available
same-identity references. Dynamic one-positive/one-negative sampling remains
the native DreaMS implementation.

After construction, the job reads the frozen event count `M`, computes
`ceil(M/4)` batches per complete pass and fixes the budget to 1.5 passes.  The
final partial batch is retained, so every relation is exposed in the first
pass.  Both arms are evaluated at the same fixed final-step checkpoint;
neither training loss nor the held retrieval graph selects it.  The epoch
ceiling is two, preventing another 301-epoch drift while matching the roughly
1.56-pass regime of the ChemAware success.

## Entrypoints

- Builder: `tasks/build_noise_action_hard_native_triplets.py`
- Proven trainer: `tasks/train_chemaware_dreams_native.py`
- Full frozen evaluation: `tasks/evaluate_noise_dreams_native.py`
- Two-seed decision: `tasks/summarize_noise_dreams_native_replicates.py`
- Two-GPU job: `tasks/run_noise_dreams_native_2gpu.sbatch`

The two GPUs train the same frozen hard-negative curriculum from the same
initialization with preregistered seeds 3407 and 3408. Both use the same
optimizer and 1.5-pass budget. Each seed independently must deliver at least
`+4 pp` Recall@1 versus official and mature E8, positive formula-cluster
intervals, positive risk nets and no registered metric regression. The two
seed Recall@1 gains must agree within 1 pp.

The ChemAware `+1.8144 pp` number is development evidence for this runtime,
not a guarantee for Noise and not an outer result.
