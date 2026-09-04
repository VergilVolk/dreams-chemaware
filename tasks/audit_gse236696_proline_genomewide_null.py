#!/usr/bin/env python
"""Expression-matched genome-wide null for the epithelial proline axis."""

from __future__ import annotations

import sys

import analyze_gse236696_mucinous_axes_by_lineage as lineage
from analyze_tcga_coadread_proline_sialic_axes import AXES


lineage.AXES = AXES
lineage.TARGET_GENES = sorted(AXES["proline_synthesis"])

import audit_gse236696_genomewide_matched_null as audit  # noqa: E402


if __name__ == "__main__":
    audit.AXES = AXES
    audit.TARGET_GENES = lineage.TARGET_GENES
    audit.EXPECTED_DIRECTION = {"proline_synthesis": 1}
    sys.argv = [
        sys.argv[0],
        "--output-dir", "data/external/GSE236696/proline_genomewide_matched_null_v1",
    ]
    audit.main()
