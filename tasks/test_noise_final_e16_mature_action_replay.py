"""Fast tests for E16 exact mature-geometry scoring."""
from pathlib import Path
import numpy as np
import build_noise_final_e16_mature_action_replay as e16

def main():
    embeddings = np.array([[1., 0.], [.8, .6], [.9, 0.], [.1, 0.]], dtype=np.float32)
    index = {10: 0, 11: 1, 12: 2, 13: 3}; groups = [(11,), (12,), (13,)]
    rank, margin = e16.rank_margin(embeddings[0], groups, embeddings, index)
    if rank != 2 or not np.isclose(margin, -0.1): raise RuntimeError("E16 exact rank/margin failed")
    sbatch = Path(__file__).with_name("run_noise_final_e16_mature_action_replay.sbatch").read_text(encoding="utf-8")
    if "#SBATCH --gpus=1" not in sbatch or "set -euo pipefail" not in sbatch or "#SBATCH --mem" in sbatch:
        raise RuntimeError("E16 sbatch violates the single-GPU fail-closed contract")
    print("[test_noise_final_e16_mature_action_replay] PASS", flush=True)

if __name__ == "__main__": main()
