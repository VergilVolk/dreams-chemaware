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

## Scope

This is the frozen **candidate annotation engine**. It consumes a pre-aligned
candidate-evidence bundle and embedding caches. It does not pretend to turn an
arbitrary raw MGF into every upstream WSE, local-fragmentation, encoder, and
chemical feature in one undocumented step. Candidate generation and upstream
evidence production remain explicit preprocessing operations.

The present release is a closed-library scorer. It does not provide a
calibrated no-match probability. BioAware is a separate conditional extension
and may be applied only when real sample-level event context exists.

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

The output contains:

- `candidate_scores.npz`: fixed candidate axis and `calms_score`;
- `query_registry.npz`: per-query status and top candidate;
- `report.json`: model, input, coverage, and provenance record;
- `SHA256SUMS.txt`: hashes of the three output artifacts.

## Input contract

`candidate_evidence.npz` follows the existing
`unified_additive_candidate_evidence_v1` contract. Its JSON manifest binds the
candidate axis and all seven required score families to their frozen upstream
artifacts. ChemAware V2 arrays are all present or all absent.

`candidate_references.npz` contains:

- `candidate_pair_ids` with the exact same order as the evidence bundle;
- `query_positions`, shape `[candidates, 4]`;
- `reference_positions`, shape `[candidates, 4, 3]`;
- `reference_mask`, shape `[candidates, 4, 3]`.

Its manifest uses schema `calms_aligned_representation_v1`, records the SHA256
of the representation file and candidate-evidence file, and registers the four
embedding banks. Each bank is an NPZ with an `embeddings` array of shape
`[n, 1024]`; an optional unique `rows` registry is accepted. The CLI computes
every bank's file SHA256 itself and verifies the frozen encoder checkpoint
identity declared in the representation manifest.

Templates are provided under `examples/`. The application evidence adapter in
`tasks/build_full23_application_evidence.py` can produce the scalar evidence
contract without changing the frozen model.

## Installation and release test

Only NumPy and PyTorch are required at runtime:

```bash
python -m pip install -r deploy/calms_v1/requirements-runtime.txt
python deploy/calms_v1/test_calms_frozen_release.py
```

The release test is CPU-only, checks package hashes, replays a golden example,
tests batch invariance and fail-closed axes, and verifies that missing
representation evidence falls back exactly to the parent score.

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
