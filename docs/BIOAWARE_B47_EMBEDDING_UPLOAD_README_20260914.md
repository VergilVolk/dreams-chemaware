# BioAware B47 truth-blind embedding execution

## Purpose

This job encodes the frozen B47 query spectra and their mass-window reference
spectra with one frozen official DreaMS encoder. It does not open annotation
truth, select seeds, fit BioAware, compute retrieval performance, or use P2b.

## Server-side dependencies already expected in the DreaMS repository

- `data/models/MassSpecGym_MurckoHist_split.hdf5`
- `data/e1/official_embedding_slim.pt`
- `dreams/models/pretrained/ssl_model_server.pt`
- the `dreams` conda environment with `h5py`, `numpy`, and CUDA-enabled `torch`

The job verifies the exact frozen SHA256 of all three model/data artefacts before
encoding. It requests exactly one GPU and no explicit memory allocation.

## Uploaded bundle contents

- `data/validation/bioaware_b47_truthblind_candidate_graph_20260914_v1/`
- `tasks/bioaware_b47_truthblind_io.py`
- `tasks/encode_bioaware_b47_truthblind_embeddings.py`
- `tasks/test_bioaware_b47_truthblind_embedding.py`
- `tasks/validate_bioaware_b47_truthblind_embeddings.py`
- `tasks/run_bioaware_b47_truthblind_embedding.sbatch`

Extract the bundle at the repository root, then submit exactly:

```bash
sbatch tasks/run_bioaware_b47_truthblind_embedding.sbatch
```

Logs appear in the repository root as `bioaware_b47_embed_<jobid>.out` and
`bioaware_b47_embed_<jobid>.err`. A successful run atomically seals:

`data/validation/bioaware_b47_truthblind_embeddings_20260914_v1/report.json`

Do not delete or overwrite an existing sealed output. A failed job leaves only
an explicitly named `.partial.<jobid>` directory; it never masquerades as a
completed result.
