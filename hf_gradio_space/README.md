---
title: Chem-aware DreaMS
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Chem-aware DreaMS — annotation backend

Reference-free LC-MS/MS spectral annotation built on the frozen DreaMS
self-supervised encoder (Bushuiev et al., *Nat Biotechnol* 2025). This Space is
the **compute backend** for the Chem-aware DreaMS platform; the static showcase
UI lives in a separate Space (`jmwang24/chemaware-dreams`).

```
precursor = backbone(spec)[:, 0]                     # position-0 precursor token
embedding = L2_normalize( linear(precursor, weight, bias) )   # 1024-d
match     = cosine(query, library)  AND  |Δm/z| <= 20 ppm
```

## Two backends, one form

| mode | behaviour |
|---|---|
| **Ours (default)** | Local CPU inference on this Space (frozen backbone + linear head, 100 peaks). Demo-scale only. |
| **Reference** (Advanced → "Use reference backend") | Forwards the same request to the official DreaMS Space (`anton-bushuiev/DreaMS`) GPU backend via `gradio_client`. |

The form mirrors the official `/predict` contract exactly:
`lib_pth`, `in_pth`, `similarity_threshold`, `calculate_modified_cosine`,
`only_high_quality_input`.

## Checkpoints (git-lfs, not committed)

The Space needs two checkpoints placed at the Space **root** (relative to
`annotation/embed.py`):

| file | size | path in this repo |
|---|---|---|
| `ssl_model_server.pt` | 464 MB | `dreams/models/pretrained/ssl_model_server.pt` |
| `official_embedding_slim.pt` | 468 MB | `data/e1/official_embedding_slim.pt` |

Upload both with `git lfs track` (already in `.gitattributes`), or fetch them
from the project's checkpoint store before `demo.launch()`.

## Honest limits

- **CPU.** This Space runs on the free CPU tier; embedding a large library
  (100k-spectrum MONA) takes hours. Use the **Reference** backend for bulk.
- **Formats.** Input `.mgf` and `.hdf5` (query) are implemented; `.mzML` /
  `.mzXML` are reserved in our backend (the Reference backend accepts mzML).
- **Modified cosine** is accepted for interface parity; plain cosine is shown.
- **Confidence (FDR) and self-consistency (COSMIC)** are reported by the
  platform, not computed inside this demo Space.

## Citations

- Bushuiev et al., *Nat Biotechnol* 2025, DreaMS (DOI 10.1038/s41587-025-02663-3)
- Schymanski et al., *Environ Sci Technol* 2014 (DOI 10.1021/es5002105)
- Elias & Gygi, *Nat Methods* 2007 (DOI 10.1038/nmeth1013)
- Scheubert et al., *Nat Commun* 2017 (DOI 10.1038/s41467-017-01318-5)
- Hoffmann et al., *Nat Biotechnol* 2022 (DOI 10.1038/s41587-021-01045-9)
