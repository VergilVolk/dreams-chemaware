#!/usr/bin/env python
"""Adversarial score-definition audit for the targeted GSE axes."""

from __future__ import annotations

import sys

import audit_gse236696_axis_robustness as audit
from analyze_tcga_coadread_proline_sialic_axes import AXES


if __name__ == "__main__":
    audit.AXES = AXES
    audit.EXPECTED_DIRECTION = {"proline_synthesis": 1}
    sys.argv = [
        sys.argv[0],
        "--pseudobulk", "data/external/GSE236696/proline_sialic_by_lineage_v1/lineage_gene_pseudobulk.csv",
        "--output-dir", "data/external/GSE236696/proline_sialic_robustness_v1",
    ]
    audit.main()
