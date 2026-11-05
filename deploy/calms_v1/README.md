# CALMS v1.0

**CALMS** is **Context-Aware Learning for Metabolite Annotation from Mass
Spectra**. This directory is the frozen, executable release of the selected
algorithm implementation (`Full23 v1.3.2` in the internal development ledger).

CALMS combines three evidence layers on one fixed metabolite-candidate axis:

1. condition-robust spectral representations from four frozen encoders;
2. fragmentation matching and local spectral evidence;
3. candidate-specific fragmentation chemistry when that evidence is available.

The released score is exactly

```text
candidate-evidence parent + 0.9 × frozen embedding correction
```

No checkpoint, feature formula, encoder, or candidate universe is fitted by
this package. Missing embedding evidence keeps the valid parent score; missing
chemical evidence is explicitly unavailable and is never replaced by a
fabricated zero observation.

## Scope: scorer only, bring your own aligned inputs

This is the frozen **candidate annotation engine**, not a raw-LC-MS/MS
processing pipeline. It does not parse an MGF, generate structures, construct
a mass-compatible candidate set, run the four encoders, or calculate WSE,
P2b, MSG, or ChemAware V2 evidence. The caller must supply a fixed candidate
axis, all seven aligned candidate-evidence score families, the aligned
representation index, and the four frozen-encoder embedding banks described
below. CALMS does not expand or prune that candidate universe.

The present release is a closed-library scorer. It does not provide calibrated
no-match/open-set rejection. BioAware is not part of this scorer because that
conditional extension requires real sample-level event context.

## One-command inference

```bash
python deploy/calms_v1/calms.py \
  --evidence candidate_evidence.npz \
  --evidence-manifest candidate_evidence.json \
  --representation candidate_references.npz \
  --representation-manifest candidate_references.json \
  --embedding-bank noise_v1=noise_v1_embeddings.npz \
  --embedding-bank chemaware_stage1=stage1_embeddings.npz \
  --embedding-bank chemaware_phasea=phasea_embeddings.npz \
  --embedding-bank noise_chemaware_fullmsg=fullmsg_embeddings.npz \
  --output calms_predictions \
  --device cpu
```

`calms_predictions` must not already exist. The CLI fails rather than silently
overwriting an earlier run.

The output contains:

- `candidate_scores.npz`: fixed candidate axis and `calms_score`;
- `query_registry.npz`: per-query status and top candidate;
- `report.json`: model, input, coverage, and provenance record;
- `SHA256SUMS.txt`: hashes of the three output artifacts.

## Input contract

### Candidate evidence

`candidate_evidence.npz` follows schema
`unified_additive_candidate_evidence_v1`. For `Q` queries and `C` candidate
rows it contains the complete axis:

- `query_ptr`: integer shape `[Q + 1]`, zero-based, nondecreasing, ending at
  `C`;
- `query_ids`: unique, non-empty strings of shape `[Q]`;
- `candidate_query_ids`, `candidate_ids`, and `candidate_pair_ids`: string
  arrays of shape `[C]`; rows are contiguous by query, pair IDs are unique,
  and a query cannot repeat a candidate ID;
- seven score arrays of shape `[C]`, named exactly `noise_v1`,
  `weighted_spectral_entropy`, `p2b_noise_v1_frozen`,
  `noise_msg_fusion_stack_v1`, `chemaware_stage1`, `chemaware_phasea`, and
  `noise_chemaware_fullmsg`.

Every non-empty query is scored by the parent only when all seven score arrays
are finite for all its candidate rows; otherwise that query is retained as an
abstention. Zero-candidate and singleton queries are legal.

ChemAware V2 has exactly two modes. In `real` mode the NPZ also contains all
five shape-`[C]` arrays `chem_valid`, `candidate_utility_correct`,
`candidate_utility_zero`, `candidate_utility_reversed`, and
`candidate_utility_rotated`; `chem_valid` is boolean or integer 0/1 and the
four utilities are finite. In `unavailable` mode none of those arrays may be
present. Partial presence is rejected, and unavailable chemistry is not
encoded as zero evidence.

The evidence JSON records the NPZ SHA256, the exact ordered seven-score list,
and an exact `upstream_artifacts` registry containing `candidate_axis`, every
score family, and `chemaware_v2_continuous_null`. Frozen upstream identities
and the real/unavailable V2 contract are checked at load time.

### Aligned representations

`candidate_references.npz` contains:

- `candidate_pair_ids`, shape `[C]`, in exactly the evidence-bundle order;
- `query_positions`, integer shape `[C, 4]`;
- `reference_positions`, integer shape `[C, 4, 3]`;
- `reference_mask`, boolean shape `[C, 4, 3]`.

The four encoder columns use the frozen encoder order. Query positions must be
constant across every candidate row of a query. Active references must point
to valid bank rows, the first reference must be active, and reference
multiplicity must agree across the four encoders. Invalid or unavailable
representation rows cause a parent-score fallback for the whole query; they
are not replaced by zero vectors.

Each embedding bank is an NPZ with a finite `embeddings` array of shape
`[n, 1024]`; an optional unique integer `rows` registry of shape `[n]` is
accepted. Optional scalar `schema` and `encoder_name` entries, when present,
must equal `calms_embedding_bank_v1` and the registered bank name.

The representation manifest uses schema `calms_aligned_representation_v1` and
binds the evidence and representation SHA256 values, all four bank SHA256
values, and the four frozen encoder checkpoint identities. The CLI recomputes
every file hash. Templates and a retained synthetic CLI example are provided
under `examples/`. This scorer-only release does not include a generic raw-MGF
candidate generator or upstream seven-score/four-embedding producer.

## Installation, self-test, and retained smoke run

Only NumPy and PyTorch are required at runtime. The retained release smoke was
validated with Python 3.11.9, NumPy 2.4.6, and PyTorch 2.14.0+cpu; the declared
runtime floor remains NumPy 1.24 and PyTorch 2.0.

### Bash

```bash
python -m pip install -r deploy/calms_v1/requirements-runtime.txt
python deploy/calms_v1/test_calms_frozen_release.py
python deploy/calms_v1/examples/run_smoke.py --output-root calms_smoke
```

Single line:

```bash
python -m pip install -r deploy/calms_v1/requirements-runtime.txt && python deploy/calms_v1/test_calms_frozen_release.py && python deploy/calms_v1/examples/run_smoke.py --output-root calms_smoke
```

### PowerShell

```powershell
py -m pip install -r deploy/calms_v1/requirements-runtime.txt
py deploy/calms_v1/test_calms_frozen_release.py
py deploy/calms_v1/examples/run_smoke.py --output-root calms_smoke
```

Single line:

```powershell
py -m pip install -r deploy/calms_v1/requirements-runtime.txt; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; py deploy/calms_v1/test_calms_frozen_release.py; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; py deploy/calms_v1/examples/run_smoke.py --output-root calms_smoke
```

The release test is CPU-only, checks package hashes, replays a golden example,
tests batch invariance and fail-closed axes, and verifies that missing
representation evidence falls back exactly to the parent score.

The smoke helper reuses that deterministic synthetic fixture but retains both
`calms_smoke/inputs/` and the real CLI outputs in
`calms_smoke/predictions/`. It is an executable I/O example only, not a
scientific benchmark. Its `--output-root` must not already exist; choose a new
path or deliberately remove an old smoke directory before rerunning it.

The package is distributed under the repository's MIT license; see `LICENSE`.

## Evidence boundary

The frozen implementation was fitted on consumed Enveda-180 meta-development
data and selected/calibrated on consumed GNPS development data. GNPS includes
10,995 identity-disjoint and 5,261 formula-disjoint queries; the two panels
contain 11,019 unique queries jointly. These are development results, not an
unopened external confirmation. A genuinely unused source is required for the
next confirmation claim.

The historical `corrected - 2 × introduced` quantity is descriptive only and
is not a selection veto for this release.
