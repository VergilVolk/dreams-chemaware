"""Static and data-contract tests for direct noise augmentation fine-tuning."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from train_noise_final_e4a_direct_augmentation import DirectExample, identity_balanced_epoch  # noqa: E402

SCRIPT = ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py"


def main() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)
    required = [
        "clean_and_augmented_raw_spectra_train_same_encoder",
        "model_weights_changed",
        "inference_clean_spectrum_only",
        '"teacher": "forbidden"',
        '"P2b": "forbidden"',
        "candidate_gradient",
        "role_confounder",
        "final_shared_encoder.pt",
    ]
    missing = [token for token in required if token not in source]
    if missing:
        raise RuntimeError(f"direct augmentation contract missing: {missing}")
    forbidden = ["corrective_teacher_actions", "privileged_teacher", "p2b_frozen"]
    found = [token for token in forbidden if token in source.lower()]
    if found:
        raise RuntimeError(f"teacher/reranker dependency entered direct training: {found}")

    r0 = ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a"
    if r0.is_dir() and (r0 / "training_actions.csv.gz").is_file():
        frame = pd.read_csv(r0 / "training_actions.csv.gz", nrows=10)
        outcome = {"corrected", "introduced", "target_rank", "target_margin", "random_margin"}
        if outcome.intersection(frame.columns):
            raise RuntimeError("R0 action manifest contains post-outcome leakage")
    dummy = []
    for identity, policies in (("A", ("candidate_gradient",)),
                               ("B", ("candidate_gradient", "role_confounder"))):
        for index, policy in enumerate(policies):
            dummy.append(DirectExample(
                query_index=index, query_row=index, identity=identity, formula="F",
                positive_rows=(1,), negative_rows=(2,), official_margin=0.0,
                official_rank=2, sample_weight=1.0, policy=policy,
            ))
    sampled = identity_balanced_epoch(dummy, np.random.default_rng(1), 2)
    if {identity: sum(item.identity == identity for item in sampled) for identity in ("A", "B")} != {"A": 2, "B": 2}:
        raise RuntimeError("identity-balanced sampler is not exact")
    b_policy = {item.policy for item in sampled if item.identity == "B"}
    if b_policy != {"candidate_gradient", "role_confounder"}:
        raise RuntimeError("combined-policy round robin dropped a fixed policy")
    print("[test_noise_final_e4a_direct_augmentation] PASS")


if __name__ == "__main__":
    main()
