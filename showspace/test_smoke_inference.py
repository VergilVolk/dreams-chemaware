"""End-to-end local inference test against the real-spectrum smoke library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "showspace"))

from app import DreaMSInterface


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    args = parser.parse_args()
    query = json.loads(args.query.read_text(encoding="utf-8"))
    interface = DreaMSInterface()
    result = interface.process_spectrum(
        json.dumps(query["peaks"]), query["precursor_mz"], 1, query["adduct"],
        False, "official_dreams", 10.0, 5, True, None,
    )
    candidates = result[-1]
    if not candidates:
        raise RuntimeError(f"smoke query returned no candidates: {result[6]}")
    observed = str(candidates[0]["Ref. ID"])[:14]
    if observed != query["truth_ik14"]:
        raise RuntimeError(f"self-retrieval failed: expected={query['truth_ik14']} observed={observed}")
    print(json.dumps({
        "status": "showspace_local_smoke_inference_passed",
        "top1": observed,
        "candidate_count": len(candidates),
        "candidate_status": result[6],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
