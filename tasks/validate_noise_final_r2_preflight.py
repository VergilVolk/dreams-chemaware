from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "noise_final_r2_preflight_passed" or not report.get("pass"):
        raise RuntimeError("R2 preflight did not pass")
    if not all(report.get("gates", {}).values()):
        raise RuntimeError("R2 preflight contains a failed gate")
    if report.get("contracts", {}).get("P2b") != "forbidden":
        raise RuntimeError("P2b entered R2 preflight")
    print(
        "[validate_noise_final_r2_preflight] PASS "
        f"actions={report['corrective_actions']:,} formulas={report['corrective_formulas']:,}"
    )


if __name__ == "__main__":
    main()
