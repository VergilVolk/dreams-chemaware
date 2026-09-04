from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(args.output_dir / "training_actions.csv.gz")
    audit = pd.read_csv(args.output_dir / "outcome_audit_only.csv.gz")
    if report.get("status") != "noise_final_r0_faithful_s3a_manifest_complete" or not report.get("formal"):
        raise RuntimeError("R0 is not formal")
    if report["contracts"].get("P2b") != "forbidden":
        raise RuntimeError("P2b entered R0")
    forbidden = {"corrected", "introduced", "target_rank", "target_margin", "random_margin"}
    if forbidden & set(manifest):
        raise RuntimeError("outcome columns entered training manifest")
    if not forbidden.issubset(audit.columns):
        raise RuntimeError("outcome audit is incomplete")
    observed = set(zip(manifest["selector"].astype(str), manifest["attenuation"].astype(float)))
    if observed != {("candidate_gradient", 0.5), ("role_confounder", 1.0)}:
        raise RuntimeError(f"unexpected R0 policies: {observed}")
    candidate_steps = set(manifest.loc[manifest["selector"].eq("candidate_gradient"), "step"].astype(int))
    role_steps = set(manifest.loc[manifest["selector"].eq("role_confounder"), "step"].astype(int))
    if candidate_steps != {3, 4, 5, 6} or role_steps != {1, 2, 3, 4, 5}:
        raise RuntimeError("R0 step contract changed")
    if manifest["matched_control_paths"].astype(str).str.count(";").ne(1).any():
        raise RuntimeError("R0 does not retain two matched control paths")
    print(
        "[validate_noise_final_r0_faithful_s3a] PASS "
        f"rows={len(manifest):,} identities={manifest['query_ik14'].nunique():,}"
    )


if __name__ == "__main__":
    main()
