from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/g8r_noise_final_e4_target_cache"))
    args = parser.parse_args()
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    actions = pd.read_csv(args.output_dir / "actions.csv.gz")
    target = np.load(args.output_dir / "target_embedding_f16.npy", mmap_mode="r")
    if report.get("status") != "noise_final_e4_target_cache_complete":
        raise RuntimeError("wrong E4 target-cache status")
    if len(actions) != len(target) or len(actions) != report["actions"]:
        raise RuntimeError("E4 target-cache row mismatch")
    if target.shape[1] != 1024 or not np.isfinite(np.asarray(target[:min(1000, len(target))])).all():
        raise RuntimeError("invalid E4 target embeddings")
    expected = {"candidate_gradient", "acquisition_positive_gradient", "role_confounder"}
    if set(actions["e4_family"].astype(str)) != expected:
        raise RuntimeError("E4 target cache does not contain exactly three families")
    if report["contracts"].get("P2b") != "forbidden" or report["contracts"].get("P3_identity_overlap") != 0:
        raise RuntimeError("E4 target-cache contract violation")
    group_sum = actions.groupby(["e4_family", "query_ik14"])["identity_weight_within_family"].sum()
    if not np.allclose(group_sum.to_numpy(), 1.0, atol=1e-6):
        raise RuntimeError("identity weights are not normalized within family")
    print(f"[validate_noise_final_e4_target_cache] PASS actions={len(actions):,}")


if __name__ == "__main__":
    main()
