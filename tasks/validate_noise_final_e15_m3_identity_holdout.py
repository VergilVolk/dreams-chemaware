"""Fail-closed validation of E15-M3 identity-held training output."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import torch

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); required = [args.output_dir / name for name in ("report.json", "held_per_query.csv.gz", "shared_encoder.pt")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(missing)
    report = json.loads(required[0].read_text(encoding="utf-8")); held = pd.read_csv(required[1], low_memory=False)
    package = torch.load(required[2], map_location="cpu", weights_only=False)
    if report.get("status") != "noise_final_e15_m3_identity_holdout_complete" or not report.get("formal"):
        raise RuntimeError("invalid E15-M3 report")
    if len(held) != 256 or held["query_ik14"].nunique() != 256 or held["query_index"].nunique() != 256:
        raise RuntimeError("E15-M3 held ledger drifted")
    if package.get("status") != "noise_final_e15_m3_shared_dreams_encoder" or package.get("P2b_used") or not package.get("inference_clean_only"):
        raise RuntimeError("E15-M3 checkpoint violates inference contract")
    if report["contracts"].get("held_used_for_selection") or report["contracts"].get("P3_consumed"):
        raise RuntimeError("E15-M3 evaluation leakage")
    print(f"[validate_noise_final_e15_m3_identity_holdout] PASS pass_to_formula_fold={report['pass_to_formula_fold']}", flush=True)

if __name__ == "__main__": main()
