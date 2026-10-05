# ChemAware frozen qualified-action delta transfer

Date: 2026-09-07

## Correction of research scope

The active action is not rediscovered in this experiment.  Its only source of
truth is the passed action bank.  The frozen setting is
`conflict_attenuate / strength=0.75 / top_k=3`: three already observed peaks
that favour the hardest same-formula negative are attenuated.  The retained E1
evidence is discovery 8 corrected / 2 introduced and confirmation 3 / 2, with
confirmation Recall@1 +0.7519 pp and positive formula-cluster intervals for
absolute and both matched-control margin contrasts.

The recently completed 272-query candidate-residual experiment is not this
action.  It transfers ICEBERG candidate scores and its correct arm did not beat
matched continuation or pseudo-prior controls in the trained shared embedding.
It is therefore closed as a negative injection result and is not the source of
the next training target.  Its four-checkpoint mechanism result and exact
numerical boundary are frozen in
`docs/CHEMAWARE_ICEBERG_RESIDUAL_MECHANISM_AUDIT_RESULT_20260908.md`.

## Injection objective

Let `f0` be frozen official DreaMS, `f_theta` the live shared encoder, `x_i` a
clean spectrum, `T_a(x_i)` the bank-qualified peak action, and `R_i` all real
reference spectra in the same-formula candidate list.  Molecule scores maximise
over all reference spectra for that molecule.  The fixed chemical effect is

```
d_i = C [ score(f0(T_a(x_i)), f0(R_i))
        - score(f0(x_i),      f0(R_i)) ],
```

where `C(v) = v - mean(v)` removes the unidentifiable common candidate-score
offset.  The live clean-query displacement is

```
q_i(theta) = C [ score(f_theta(x_i), stopgrad(f0(R_i)))
               - score(f0(x_i),      stopgrad(f0(R_i))) ].
```

The chemical loss is a candidate-vector Smooth-L1 objective,

```
L_chem = sum_i w_i Huber(q_i(theta) - alpha rho_i d_i).
```

This differs from every rejected route:

- no ICEBERG candidate-score residual is used;
- the modified action spectrum is never forwarded through the trainable model;
- the chemical branch cannot move candidate references;
- no symmetric clean/action consistency is used;
- no scalar hardest-negative PMT is used;
- no correct-minus-control gradient is trained;
- no dynamic gradient-ratio scaling followed by a saturated global cap is used.

The same live encoder still processes clean query and reference spectra in the
ordinary full-list clean retrieval and safety branches.  Deployment therefore
remains one clean spectrum to one normalized 1024-dimensional embedding.  The
chemical target is transparently teacher-derived counterfactual effect
transfer; it must not be described as a pure no-distillation objective.

## Action roles and safety

Only the bank-selected `corrective_rank` and `corrective_margin` actions in
formula folds 0--2 enter the chemical objective.  `corrective_rank` receives
full action dose.  The `corrective_margin` dose is derived from its positive
margin gain relative to the median positive rank-corrective margin gain and is
capped at one.  Harmful, robustness and uncertain roles have exactly zero
chemical weight.  Inverse formula frequency supplies formula-equal scheduled
loss mass.  The action bank, not the CLI, supplies mode, strength and top-k.

The clean branch retains complete candidate listwise training.  Baseline-correct
training queries retain a one-sided official margin floor and embedding
preservation.  Every matched batch uses an operator split: one clean/safety
AdamW step followed by a newly forwarded chemical-effect step.  The two steps
have separate optimizer moments and the chemical optimizer has zero weight
decay.  This prevents the much larger temperature-scaled listwise gradient from
silently diluting the chemical signal and is not PCGrad or dynamic norm-ratio
scaling.  Both steps use a norm cap of 5 instead of the previous fully saturated
cap of 1; their clipping rates are recorded separately and saturation is a
failing gate.

Separate moments do not, by themselves, prove that the requested chemical
dose survives AdamW: its coordinate-wise normalization is approximately
invariant to a common gradient scale.  The trainer therefore audits the first
and last chemical step of every epoch after clipping.  It measures the cosine
and first-order descent between the retained chemical gradient and the actual
parameter update, separately for the projection head and final backbone block,
then immediately re-forwards the same batch and requires the action-delta loss
not to increase.  Missing signal in either parameter group, an opposing update,
or a rising same-batch chemical loss terminates the arm.  This is measurement,
not an optimizer-space rewrite or an unvalidated restoration method.

The complete qualified-action panel is also evaluated before the first update
and after the final update.  The report records formula-weighted candidate
delta Huber error and target/student displacement norms.  Every non-null
action arm must reduce this complete-panel target error.  Therefore a negative
held result can be localized: a frozen source/replay failure, an optimizer
write failure, or a learned-but-nongeneralizing action effect are no longer
collapsed into the same number.

## Evaluation and causal gate

Stage 1 runs identical-schedule clean-duplicate and correct-action-effect arms.
It stops before further GPU work unless the correct effect has a strictly
positive formula-cluster Recall@1 advantage over clean continuation, positive
absolute MRR/Macro-AUC/Micro-AUC, nonnegative Recall@5/10/20/50, preservation
at least 0.995, and nonsaturated clipping.  Only then are candidate-swapped and
peak-permuted effect arms run.  The final gate requires the correct arm's
Recall@1 formula-cluster interval to beat all three comparators and every other
registered metric to be nonnegative relative to them.

The primary development evaluation is every identity represented once in the
broad formula-held fold 3 manifest, with no 2,000-identity cap.  It is not the
older 180-query action graph; that graph remains a diagnostic only.  Fold 4 is
untouched and no action is generated for either evaluation fold.

This experiment can establish a chemically attributable shared-embedding
increment.  It cannot guarantee the requested 3--5 pp before execution, and a
single development seed is never release-eligible.

## Server entry point

```bash
sbatch tasks/run_chemaware_qualified_action_delta_phase_a.sbatch
```

The job requests exactly one GPU, has no manual memory request, and fails before
model construction if the frozen bank, selected action, graph/cache closure,
formula roles, CLI, or CPU contracts drift.
