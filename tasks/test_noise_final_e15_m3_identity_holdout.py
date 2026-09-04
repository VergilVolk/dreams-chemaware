"""Fast behavioral tests for E15-M3 training helpers and job contract."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import train_noise_final_e15_m3_identity_holdout as m3


class _Graph:
    query_row = np.array([10])
    def query_block(self, query):
        # Positive molecule row 11; two negative molecules rows 12 and 13.
        return None, np.array([11, 12, 13]), np.array([0, 1, 2, 3]), None

def main() -> None:
    frame = pd.DataFrame({"query_index": [1, 1, 1, 2], "query_ik14": ["a", "a", "a", "b"],
                          "source": ["R0_N", "A4_exact", "A4_exact", "C1_support_disjoint"],
                          "action_family": ["x", "y", "y", "z"], "action_id": ["1", "2", "3", "4"],
                          "source_kind_percentile": [0.2, 0.8, 0.7, 0.5]})
    limited = m3.limit_actions(frame, 2)
    if limited.groupby("query_index").size().max() > 2 or limited.duplicated(["query_index", "source", "action_id"]).any():
        raise RuntimeError("global M3 action cap failed")
    dev1 = m3.identity_dev_split(frame, 0.25, 7); dev2 = m3.identity_dev_split(frame, 0.25, 7)
    if dev1 != dev2 or not dev1: raise RuntimeError("internal identity split is not deterministic")
    summary = m3.paired_summary(np.array([2, 1, 2]), np.array([1, 1, 2]), np.array([True, True, False]))
    if summary["corrected"] != 1 or summary["introduced"] != 0 or summary["risk_net_lambda2"] != 1:
        raise RuntimeError("paired M3 summary is wrong")
    source = Path(m3.__file__).read_text(encoding="utf-8")
    selection_position = source.index("selected = max(candidates")
    held_read_position = source.index('held = pd.read_csv(required["held"]')
    if held_read_position < selection_position:
        raise RuntimeError("frozen held ledger is read before M3 model selection")
    if "evaluate_queries_filtered" not in source:
        raise RuntimeError("internal development evaluation does not filter held/sentinel references")
    embeddings = np.array([[1.0, 0.0], [0.9, 0.0], [0.95, 0.0], [0.1, 0.0]], dtype=np.float32)
    rank, _ = m3.evaluate_queries_filtered(
        _Graph(), np.array([0]), embeddings, {10: 0, 11: 1, 12: 2, 13: 3},
        {11: "positive", 12: "held", 13: "safe"}, {"held"},
    )
    if int(rank[0]) != 1:
        raise RuntimeError("held negative reference was not filtered from internal development")
    kept, dropped = m3.retain_evaluable_queries(
        _Graph(), np.array([0]), {11: "positive", 12: "held", 13: "sentinel"},
        {"held", "sentinel"},
    )
    if len(kept) != 0 or dropped != [0]:
        raise RuntimeError("query without a legal negative was not removed cleanly")
    sbatch = Path(__file__).with_name("run_noise_final_e15_m3_identity_holdout.sbatch").read_text(encoding="utf-8")
    for token in ("#SBATCH --gpus=1", "set -euo pipefail", "run_${SLURM_JOB_ID}", "validate_noise_final_e15_m3_identity_holdout.py"):
        if token not in sbatch: raise RuntimeError(f"M3 sbatch missing {token}")
    if "#SBATCH --mem" in sbatch: raise RuntimeError("M3 sbatch must use the default per-GPU memory")
    print("[test_noise_final_e15_m3_identity_holdout] PASS", flush=True)

if __name__ == "__main__": main()
