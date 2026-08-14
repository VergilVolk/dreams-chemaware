# P1 pre-experiment: rule evidence under controlled peak masking

## Scientific question

This is a training-free intervention experiment. It does not ask whether adding
noise improves a trained model. It first asks whether the P0 failure directions
are real and whether peak masking is a safe augmentation candidate.

The three P0 strata are:

1. MCES 0-2 local structural confusion;
2. high Rule-Jaccard but different-identity conflict;
3. cross-instrument mismatch between a query and its best positive spectrum.

## Experimental design

- Model: the same raw DreaMS SSL checkpoint and cached strict-E0 gallery.
- Retrieval: same adduct, precursor mass within 10 ppm, molecule-level maximum
  similarity, exact peak-hash duplicates excluded.
- Rules: all 3,486 rules are the primary analysis; the 335 core rules are a
  nested control.
- Primary intervention: reproduce DreaMS pretraining masking by sampling peaks
  proportional to intensity, protecting the precursor, and replacing masked
  m/z values by -1. Rates 10%, 20%, and the checkpoint's 30% are tested.
- Matched control: delete the exact same selected peak tokens entirely.
- Repeats: three masks per query and condition.
- Evaluation: corrupted queries are searched against the unchanged clean E0
  gallery. Rules are recomputed from the surviving peaks only.

The default sample contains 12 queries from each P0 failure stratum and 24
correct controls stratified by the MCES bin of their strongest negative. A query
can belong to more than one failure stratum.

## What would support the hypotheses?

### The P0 pain point is intervention-sensitive

Compared with correct controls, the relevant error stratum shows a larger
within-query fall in true-minus-wrong similarity margin or embedding cosine
under masking. Existing-error selection alone is not considered causal proof.

### Peak masking is feasible

On clean-correct controls:

- Top-1 falls by no more than 10 percentage points in this small pilot; and
- mean cosine between clean and corrupted embeddings remains at least 0.90.

These are pilot stop/go thresholds, not publication claims. The lowest passing
mask rate should enter a later training ablation.

### Rule evidence is useful

The true-candidate minus wrong-candidate rule margin is positive and remains
stable after masking. High clean-to-noisy rule retention without a positive
candidate margin only proves that generic rules are stable, not that they help
identification.

## Run

CPU pilot:

```powershell
& 'D:\dreams_env\python.exe' tasks\pilot_rule_noise_stress.py --device cpu
```

Minimal smoke test:

```powershell
& 'D:\dreams_env\python.exe' tasks\pilot_rule_noise_stress.py `
  --device cpu --n-per-stratum 1 --n-controls 2 `
  --mask-rates 0.30 --modes native_mask --n-seeds 1 --skip-plot
```

On a CPU-only laptop, first validate the rule intervention without repeating
the expensive DreaMS forward pass:

```powershell
& 'D:\dreams_env\python.exe' tasks\pilot_rule_noise_stress.py `
  --device cpu --n-per-stratum 1 --n-controls 2 `
  --mask-rates 0.30 --modes native_mask --n-seeds 1 `
  --rules-only --skip-plot --output-dir data\validation\rule_noise_pilot_smoke
```

`--rules-only` cannot answer whether masking is safe for DreaMS. It only checks
selection, peak intervention and all 3,486 rule calculations before spending
GPU time on re-embedding.

Outputs are written to `data/validation/rule_noise_pilot/`:

- `selected_queries.csv`: selected cases and clean retrieval results;
- `per_query_perturbation.csv`: paired query-level results;
- `aggregate_results.csv`: group-level means and bootstrap intervals;
- `rule_noise_pilot.png`: summary figure;
- `REPORT.md`: readable interpretation;
- `pilot_summary.json`: machine-readable gates.

## Decision after the pilot

1. Native masking passes, dropout fails: use masked-token consistency first.
2. Both pass: compare them in a small fine-tuning ablation.
3. Both fail: reduce masking to the lowest safe rate; do not train yet.
4. Rules remain stable but favor the wrong candidate: the pain point is real,
   but rule injection cannot solve it directly; retain rules for conflict mining.
5. A compact subset of rules keeps a positive true-candidate margin: those
   rules become candidates for the first concept-decoding head.

## Completed rule-side pilot (2026-08-11)

The rules-only stage completed on 60 queries (36 P0 errors and 24 MCES-stratified
correct controls), three mask rates, two matched perturbation encodings and
three random masks per condition: 1,080 interventions in total.

At 30% masking, the 3,486-rule retention was 0.747 in correct controls, 0.739
in MCES 0-2 errors, 0.697 in cross-instrument errors and 0.783 in high-rule
conflicts. More importantly, the true-minus-wrong Rule-Jaccard margin was
+0.285 in correct controls but -0.096, -0.105 and -0.213 in those three error
strata, respectively. Thus the expanded rules are reasonably stable to missing
peaks, but in the selected DreaMS errors they usually preserve the wrong
chemical preference instead of rescuing the true identity.

This does not yet establish that masking is safe for DreaMS. The GPU follow-up
must re-embed the perturbed query spectra. A budget-conscious first run is:

```powershell
& 'D:\dreams_env\python.exe' tasks\pilot_rule_noise_stress.py `
  --device cuda --n-per-stratum 6 --n-controls 12 `
  --mask-rates 0.10 0.20 0.30 --modes native_mask --n-seeds 3
```

Only if native masking passes should the matched `peak_dropout` control be run.
