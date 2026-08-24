---
title: Chem-aware DreaMS
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# Chem-aware DreaMS — static platform UI

Reference-free LC-MS/MS annotation platform for untargeted metabolomics. This
Space is the **static showcase** (free tier, no Python compute): it renders the
real MTBLS13729 smoke results across every panel — cosine matching, target-decoy
FDR confidence, self-consistency (COSMIC), chemical rule matching, chemical
explainability, differential analysis, and novel "dark-matter" findings.

All data is precomputed into [data/showcase.json](data/showcase.json) by
`tasks/make_hf_showcase.py` in the main repo. No inference runs in the browser.

## Where the compute lives

This static Space cannot run inference. Two backends are wired behind the
top-right selector:

| backend | where | notes |
|---|---|---|
| **Official DreaMS API** | `anton-bushuiev/DreaMS` | default; batch calls via `tasks/call_dreams_api_pos_rp.py` |
| **Self-hosted** | `jmwang24/disulf-posit-predict` | our own Gradio Space (frozen DreaMS + m/z constraint + reference toggle) |

## Panels

1. **Overview** — query/library counts, confident matches, self-retrieval FDR.
2. **Run annotation** — the `/predict` form (interface parity; static → no compute).
3. **Confident matches** — cos ≥ 0.7 ∧ |Δm/z| ≤ 20 ppm, sortable table.
4. **FDR confidence** — target-decoy q-value + honest collapse note; reliable scale
   = self-retrieval ground-truth FDR (4.19% / 10.10%).
5. **Self-consistency** — COSMIC coherence (null result, P = 0.44) + frozen probe (AUPRC 0.63).
6. **Chemical rules** — 266/335 Schymanski rules decoded.
7. **Explainability** — per-match evidence chain.
8. **HMDB / biochemical** — reserved, all NaN.
9. **Differential** — tumor vs normal (Fisher + BH).
10. **Novel findings** — structures the authors did not report.
11. **Author comparison** — per-panel coverage.

## Citations

- Bushuiev et al., *Nat Biotechnol* 2025 (DOI 10.1038/s41587-025-02663-3)
- Schymanski et al., *Environ Sci Technol* 2014 (DOI 10.1021/es5002105)
- Elias & Gygi, *Nat Methods* 2007 (DOI 10.1038/nmeth1013)
- Scheubert et al., *Nat Commun* 2017 (DOI 10.1038/s41467-017-01318-5)
- Hoffmann et al., *Nat Biotechnol* 2022 (DOI 10.1038/s41587-021-01045-9)
