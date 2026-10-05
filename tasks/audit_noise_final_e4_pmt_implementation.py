"""Static fail-closed audit for the minimal E4-PMT implementation."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    files = {
        "core": ROOT / "tasks/noise_final_e4_pmt_core.py",
        "builder": ROOT / "tasks/build_noise_final_e4_pmt_manifest.py",
        "trainer": ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py",
        "summary": ROOT / "tasks/summarize_noise_final_e4_pmt.py",
        "sbatch": ROOT / "tasks/run_noise_final_e4_pmt_phase_a.sbatch",
    }
    text = {}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        text[name] = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            ast.parse(text[name])
    required = {
        "core": [
            "harmful = introduced |", "corrective = (~harmful)",
            'output["corrective_weight"]', "recycled before complete coverage",
        ],
        "builder": [
            'f"N|{int(row.query_index)}|{row.selector}|"',
            "selected not in controls", "M2_predictions_used", "P_actions_used",
            "current_geometry_full_candidate_replay", "contract_mismatches",
        ],
        "trainer": [
            "def paired_pmt_loss", "target_margin - control_margin",
            "clean_floor_each", "control_floor_each", "coverage_first_schedules",
            'args.pmt_arm == "paired_target"',
        ],
        "summary": [
            'f"{label}_vs_matched_random"', 'f"{label}_vs_clean_duplicate"',
            'formula_cluster_top1_ci"]["ci_low"] > 0',
        ],
        "sbatch": [
            "#SBATCH --gpus=1", "ALPHAS=(0 0 0.25 0.50)",
            "--backbone-lr 2e-6 --head-lr 1e-5", "--no-amp",
            "fold_0_run_2331284", "test_noise_final_e4_pmt.py",
        ],
    }
    missing = {
        name: [token for token in tokens if token not in text[name]]
        for name, tokens in required.items()
    }
    missing = {name: tokens for name, tokens in missing.items() if tokens}
    if missing:
        raise RuntimeError(f"E4-PMT implementation audit failed: {missing}")
    if "#SBATCH --mem" in text["sbatch"] or "P_intensity" in text["sbatch"]:
        raise RuntimeError("E4-PMT sbatch requests forbidden memory/P-intensity branch")
    print("[audit_noise_final_e4_pmt_implementation] PASS")


if __name__ == "__main__":
    main()
