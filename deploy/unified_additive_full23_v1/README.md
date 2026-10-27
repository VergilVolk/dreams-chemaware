# DreaMS Unified Additive Full23 v1

This is the frozen candidate-decision package selected from run `2413878`.
It wraps the existing feature builder and checkpoint inference code; it does
not duplicate or alter the learned scoring mathematics.

## What is packaged

- `assets/unified_candidate_integrator.pt`: the frozen integrator checkpoint.
- `manifest.json`: exact arm, feature order, component registry, hashes and
  scientific boundary.
- `model_status.json`: concise current selection, dataset roles, inference
  semantics, successor verdict and application readiness.
- `evidence_manifest_template.json`: provenance sidecar required by the CLI.
- `evidence/*.json`: the four original selection, transfer and strict-refit
  ablation reports, validated by canonical-LF content hashes across platforms.
- `tasks/unified_additive_release.py`: fixed-axis inference wrapper.
- The bundle builder also includes the two original core modules used for
  feature construction and scoring.

The deployed arm is explicitly `additive_linear_control`. The checkpoint also
contains an older nonlinear arm; the wrapper never selects it implicitly.

## Architecture boundary

The package consumes candidate-level evidence already produced by the frozen
upstream experts: Noise V1, WSE, P2b-on-Noise, Noise-MSG, ChemAware Stage-1,
ChemAware Phase-A and Noise-ChemAware FullMSG. ChemAware V2 real-versus-null
evidence is optional and has an exact registered unavailable state. P2b and
MSG each enter once as a frozen composite.

Official fine-tuned DreaMS is not an independent top-level Full23 feature. It
is a comparison baseline and is used indirectly inside the already-frozen
Noise-MSG composite and V2 selector geometry. BioAware is a downstream
sample-context module and is not part of the library-ranking checkpoint.

## Input NPZ

Required fixed-axis arrays:

- `query_ptr`, `query_ids`, `candidate_query_ids`, `candidate_ids`,
  `candidate_pair_ids`;
- `noise_v1`, `weighted_spectral_entropy`, `p2b_noise_v1_frozen`,
  `noise_msg_fusion_stack_v1`, `chemaware_stage1`, `chemaware_phasea`,
  `noise_chemaware_fullmsg`.

Optional ChemAware V2 arrays must be supplied together:

- `chem_valid`, `candidate_utility_correct`, `candidate_utility_zero`,
  `candidate_utility_reversed`, `candidate_utility_rotated`.

Here `candidate_utility_correct` means the original, unperturbed chemical-policy
channel. It is compared with truth-blind zero, reversed and candidate-mismatch
(`rotated`) null channels; the name does not mean that a ground-truth-positive candidate
is supplied at inference.

If all five are absent, the original `unavailable_chemaware()` helper creates
the exact no-op. Labels, molecular formulae and biological response endpoints
are neither required nor read.

The manifest mode must agree with the NPZ. `mode: unavailable` forbids all five
arrays and binds the exact no-op helper/core hash. `mode: real` requires all
five arrays plus the registered frozen V2-policy hash. Mixed or ambiguous
provenance fails before scoring.

The JSON evidence manifest must use schema
`unified_additive_candidate_evidence_v1`, list the required score names in the
frozen order, give the SHA256 of the NPZ and record upstream artifact hashes.

## Run

From the repository root:

```bash
python -u tasks/unified_additive_release.py \
  --evidence candidate_evidence.npz \
  --evidence-manifest candidate_evidence.json \
  --output unified_additive_predictions \
  --device cpu
```

The output preserves the complete candidate axis. A missing required score in
any candidate abstains the whole nonempty query; candidates are never deleted
to make a query easier. Zero-candidate queries remain in the query registry.
Singletons receive a ranking-support score but are marked
`singleton_hypothesis`, not accepted as calibrated identities.

For the frozen MSV000095331 cross-study identity axis, the repository's
`tasks/build_full23_application_evidence.py` converts the existing scorer TSVs
and four encoder-cache sidecars into this exact NPZ/manifest contract. The
single Slurm entrypoint
`tasks/run_msv000095331_integrated_multiencoder_identity.sbatch` then invokes
this release wrapper and verifies its scores against the original application
scorer; it does not reimplement the model.

## Build a portable archive

```bash
python -u tasks/build_unified_additive_release_bundle.py \
  --output deliverables/unified_additive_full23_v1_20261027 \
  --archive
```

After transfer or extraction, validate every bundled byte and replay the
golden inference fixture:

```bash
python -u tasks/validate_unified_additive_release_bundle.py \
  deliverables/unified_additive_full23_v1_20261027
```

Large upstream encoders are intentionally referenced by exact path and hash in
the manifest rather than duplicated into every decision-layer archive. The
builder never silently substitutes or discovers a different checkpoint.

This archive is explicitly the **decision-layer profile**. It needs only
NumPy and PyTorch once the seven candidate-score channels have been generated.
The raw-spectrum runtime is larger and remains external: in particular, the
frozen WSE path uses the official `ms_entropy==1.5.2` wheel and the frozen V2
policy was produced under Python 3.11, NumPy 1.26.4 and scikit-learn 1.7.2.
The manifest records these known locked constraints and every large model
identity, but it explicitly does **not** claim a complete raw-runtime
dependency lock. Missing assets must fail rather than be replaced.
