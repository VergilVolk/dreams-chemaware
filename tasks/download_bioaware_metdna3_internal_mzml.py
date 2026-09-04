#!/usr/bin/env python
"""Download the frozen, unopened-outcome NIST urine internal RPLC panel."""
from __future__ import annotations

import sys
from pathlib import Path

from download_bioaware_metdna3_development_mzml import main


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("this locked wrapper accepts no command-line overrides")
    sys.argv.extend([
        "--manifest",
        str(Path("data/validation/bioaware_metdna3_internal_rplc_manifest_v1/download_manifest.json")),
        "--output-dir",
        str(Path("data/external/metdna3_2025/mzml/internal_rplc")),
        "--scope",
        "internal_rplc",
        "--expected-files",
        "16",
    ])
    main()
