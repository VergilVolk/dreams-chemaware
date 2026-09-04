"""Fail-closed semantic tests for E14 conditional action distillation."""
from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

import build_noise_final_e14_crossfit_p_teacher as builder  # noqa: E402
import train_noise_final_e4a_direct_augmentation as trainer  # noqa: E402
from noise_final_core import stable_fold  # noqa: E402


class FakeGraph:
    def query_block(self, query: int):
        if query != 0:
            raise IndexError(query)
        # Two positive spectra, then one spectrum for each of two negatives.
        rows = np.asarray([10, 11, 20, 30], dtype=np.int64)
        ptr = np.asarray([0, 2, 3, 4], dtype=np.int64)
        return slice(0, 4), rows, ptr, None


def main() -> None:
    for path in (
        ROOT / "tasks/build_noise_final_e14_crossfit_p_teacher.py",
        ROOT / "tasks/audit_noise_final_e14_capacity_amendment.py",
        ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py",
    ):
        ast.parse(path.read_text(encoding="utf-8"))

    definitions = builder.action_definitions()
    if len(definitions) != 60 or len({item.action_id for item in definitions}) != 60:
        raise RuntimeError("E14 must freeze exactly 60 unique complementary P actions")
    prior_ids = [builder.prior_cell_id(item) for item in definitions]
    if len(prior_ids) != len(definitions) or any(not value for value in prior_ids):
        raise RuntimeError("E14 prior fixed-cell mapping is incomplete")
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        source_status = {
            "E10B": "noise_final_e10b_positive_action_expansion_complete",
            "E11": "noise_final_e11_reference_diversity_complete",
            "E12B": "noise_final_e12b_relaxed_recurrence_complete",
        }
        paths = {}
        expected_actions = []
        for source, status in source_status.items():
            item = next(definition for definition in definitions if definition.source == source)
            expected_actions.append(item.action_id)
            path = temporary / f"{source}.json"
            path.write_text(json.dumps({
                "status": status,
                "formal": True,
                "passing_fixed_cells": [builder.prior_cell_id(item)],
                "contracts": {"P2b": "forbidden", "P3_consumed": False},
            }), encoding="utf-8")
            paths[source] = path
        filtered, audit = builder.load_prior_safe_definitions(SimpleNamespace(
            e10b_report=paths["E10B"],
            e11_report=paths["E11"],
            e12b_report=paths["E12B"],
        ))
        if {item.action_id for item in filtered} != set(expected_actions):
            raise RuntimeError("E14 prior fixed-cell filter did not reproduce fake reports")
        if audit["prior_safe_actions"] != 3:
            raise RuntimeError("E14 prior fixed-cell audit count drifted")

    # Regression guard for the original E14 manifest bug: selected rows must
    # store their own formula fold, never the outer fold that was excluded.
    query_formula = "C7H8O2"
    query_fold = stable_fold(query_formula, 5, 20260826)
    outer_fold = next(fold for fold in range(5) if fold != query_fold)
    source = (ROOT / "tasks/build_noise_final_e14_crossfit_p_teacher.py").read_text(
        encoding="utf-8"
    )
    if '"formula_fold": int(fold[int(query)])' not in source:
        raise RuntimeError("E14 selected rows do not materialize their actual formula fold")
    if query_fold == outer_fold:
        raise RuntimeError("E14 formula-fold regression test is malformed")
    required_fold_tokens = (
        'decision_path = args.student_checkpoint.parent / "decision.json"',
        '"checkpoint+decision+graph_reconstruction"',
        'int(configuration.get("formula_fold_seed", -1)) != args.formula_fold_seed',
        'held_ledger_present = held_ledger_path.is_file()',
        'if held_ledger_present:\n            expected_rank = held["final_rank"]',
        'reproduced != expected',
        'held_summary = decision.get("held_clean", {})',
        'action_scan_partial.npz',
        'stale E14 action-scan checkpoint',
    )
    missing_fold_tokens = [token for token in required_fold_tokens if token not in source]
    if missing_fold_tokens:
        raise RuntimeError(
            f"E14 missing fail-closed derived-ledger fallback: {missing_fold_tokens}"
        )
    if source.count('held["final_rank"]') != 1:
        raise RuntimeError("E14 has an unguarded or duplicated held-ledger rank dependency")

    embeddings = np.asarray([
        [1.0, 0.0],   # row 10: best positive
        [0.8, 0.6],   # row 11: another positive
        [0.9, 0.1],   # row 20: hardest negative
        [0.0, 1.0],   # row 30: easier negative
    ], dtype=np.float32)
    index = {10: 0, 11: 1, 20: 2, 30: 3}
    rank, margin, positive, negative = builder.rank_margin_rows(
        FakeGraph(), 0, np.asarray([1.0, 0.0], dtype=np.float32), embeddings, index,
    )
    if rank != 1 or positive != 10 or negative != 20 or not np.isclose(margin, 0.1):
        raise RuntimeError(
            f"E14 exact molecular-margin rows drifted: {(rank, margin, positive, negative)}"
        )

    # Cached evidence must be a purely computational optimization: applying it
    # must be bitwise-equivalent (within float precision) to the original path.
    clean = torch.tensor([
        [300.0, 1.0], [50.0, 0.7], [75.0, 0.4], [0.0, 0.0],
    ], dtype=torch.float32)
    references = [torch.tensor([
        [300.0, 1.0], [50.0, 0.8], [90.0, 0.3], [0.0, 0.0],
    ], dtype=torch.float32)]
    for definition in (definitions[0], next(
        item for item in definitions if item.source == "E12B"
    )):
        prevalence, target = builder.reference_profile(clean, references, 0.02)
        missing = builder.recurrent_missing_peaks(
            clean, references, 0.02,
            definition.prevalence, definition.maximum_peaks,
        )
        direct = builder.build_variant(clean, references, definition, 0.02)
        cached = builder.build_variant_from_evidence(
            clean, prevalence, target, missing, definition,
        )
        if not torch.equal(direct, cached):
            raise RuntimeError("E14 cached evidence changes an action variant")

    frame = pd.DataFrame([{
        "query_index": 0,
        "query_row": 100,
        "query_ik14": "IK",
        "query_formula": "C1",
        "positive_reference_rows": "10;11",
        "teacher_positive_row": 10,
        "teacher_hard_negative_row": 20,
        "teacher_pair_clean_margin": -0.03,
        "crossfit_clean_rank": 2,
        "guided_policy": definitions[0].action_id,
        "guided_family": definitions[0].family,
        "guided_dose": definitions[0].dose,
        "guided_auxiliary_dose": definitions[0].auxiliary_dose,
        "guided_recurrence_prevalence": definitions[0].prevalence,
        "guided_recurrence_max_peaks": definitions[0].maximum_peaks,
        "guided_support_weighted": definitions[0].support_weighted,
        "teacher_margin": 0.07,
    }])
    examples = trainer.make_guided_noise_examples(
        FakeGraph(), frame, np.asarray([2]), np.asarray([-0.5]), 4, 8,
    )
    example = examples[0]
    if (
        example.positive_rows != (10,)
        or example.negative_rows != (20,)
        or not np.isclose(example.official_margin, -0.03)
        or not np.isclose(example.teacher_margin, 0.07)
    ):
        raise RuntimeError("E14 trainer did not preserve the exact teacher pair/margins")

    source = (ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py").read_text(
        encoding="utf-8"
    )
    required = (
        "mature initialization replay changed",
        "audit_guided_teacher_replay",
        "teacher_pair_clean_margin",
        "initialization_formula_cluster_delta_recall1",
        "crossfit_teacher_outer_fold_excluded",
        "posthoc_clustered_amendment",
        "guided_crossfit_capacity_amendment_sha256",
        "outer_fold_isolated_privileged_action_margin",
        "checkpoint+decision+graph_aggregate",
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise RuntimeError(f"E14 transfer contracts are incomplete: {missing}")
    legacy_required_ledger = (
        "required.extend([\n"
        "            args.initial_student_checkpoint,\n"
        '            args.initial_student_checkpoint.parent / "held_per_query.csv.gz",'
    )
    if legacy_required_ledger in source:
        raise RuntimeError("E14 warm start still requires the derived held ledger")
    print("[test_noise_final_e14_crossfit_teacher] PASS", flush=True)


if __name__ == "__main__":
    main()
