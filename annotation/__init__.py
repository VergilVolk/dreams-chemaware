"""Chem-aware DreaMS annotation platform.

A reference-free LC-MS/MS spectrum annotation pipeline for non-targeted
metabolomics in medical/biological research, built on the DreaMS
self-supervised encoder.

Modules (each independently usable and ablatable):
    params      -- all thresholds/methods with literature citations (M0)
    embed       -- spectrum peak preprocessing (top-N, max normalization) + DreaMS official embedding (M1)
    retrieve    -- cosine top-k + precursor m/z hard constraint (M1)
    confidence  -- Schymanski confidence levels (M2)
    fdr         -- target-decoy FDR / q-value (M3)
    calibrate   -- Platt / isotonic posterior calibration (M4)
    diff        -- differential abundance between two groups (M5)
    pathway     -- pathway enrichment (M6)
    darkmatter  -- dark-matter candidate mining (M7)
    ablation    -- module on/off comparison harness (M8)
    cli         -- end-to-end command line entrypoint (M9)
    bioaware    -- conservative reaction-network evidence after embedding
    bioaware_negative_expert -- frozen risk-controlled [M-H]- network reranker
"""

__version__ = "0.1.0"
