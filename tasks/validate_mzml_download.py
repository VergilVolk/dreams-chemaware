"""Validate an mzML download and recover a complete first XML document only.

Some HTTP range transfers from the public EBI FTP mirror append a repeated
partial tail after a complete ``</indexedmzML>`` document.  This script never
modifies the source file: it writes a separate validated copy only when the
first complete document parses successfully.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyteomics import mzml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = args.source.read_bytes()
    terminal = b"</indexedmzML>"
    end = payload.find(terminal)
    if end < 0:
        raise ValueError(f"No complete indexedmzML document in {args.source}")
    end += len(terminal)
    complete = payload[:end]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(complete)
    try:
        count = sum(1 for _ in mzml.read(str(args.output)))
    except Exception:
        args.output.unlink(missing_ok=True)
        raise
    print(f"Validated {args.output}: {count} spectra; discarded {len(payload) - end} duplicated trailing bytes")


if __name__ == "__main__":
    main()
