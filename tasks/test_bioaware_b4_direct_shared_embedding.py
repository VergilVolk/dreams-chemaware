#!/usr/bin/env python
"""Cheap source-level contract checks for BioAware B4."""
from __future__ import annotations

import ast
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_mona_row_semantics() -> None:
    sys.path.insert(0, str(ROOT / "tasks"))
    from build_bioaware_b4_direct_manifest import load_mona_reference_tensors

    # PEPMASS=0 was excluded by the original MoNA cache builder.  The B4
    # streaming parser must not let that invalid record shift every later row.
    text = """BEGIN IONS
PEPMASS=0
INCHIKEY=INVALID-UHFFFAOYSA-N
50 1
END IONS
BEGIN IONS
PEPMASS=100
INCHIKEY=AAAAAAAAAAAAAA-UHFFFAOYSA-N
20 2
30 4
END IONS
BEGIN IONS
PEPMASS=200
INCHIKEY=BBBBBBBBBBBBBB-UHFFFAOYSA-N
40 5
END IONS
"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mgf = root / "tiny.mgf"
        manifest = root / "manifest.csv"
        mgf.write_text(text, encoding="utf-8")
        pd.DataFrame({
            "inchikey": [
                "AAAAAAAAAAAAAA-UHFFFAOYSA-N", "BBBBBBBBBBBBBB-UHFFFAOYSA-N",
            ],
            "precursor_mz": [100.0, 200.0],
        }).to_csv(manifest, index=False)
        tensor = load_mona_reference_tensors(
            mgf, manifest, np.asarray([1], dtype=np.int64), 3,
        )
    if tensor.shape != (1, 4, 2) or float(tensor[0, 0, 0]) != 200.0:
        raise AssertionError("MoNA selected-row tensor mapping failed")
    if not np.isclose(float(tensor[0, 1, 1]), 1.0):
        raise AssertionError("MoNA intensity normalization failed")


def main() -> None:
    builder = ROOT / "tasks/build_bioaware_b4_direct_manifest.py"
    trainer = ROOT / "tasks/train_bioaware_b4_direct_shared_embedding.py"
    summary = ROOT / "tasks/summarize_bioaware_b4_direct_shared_embedding.py"
    validator = ROOT / "tasks/validate_bioaware_b4_direct_manifest.py"
    sbatch = ROOT / "tasks/run_bioaware_b4_direct_shared_embedding.sbatch"
    contract = ROOT / "docs/BIOAWARE_MILESTONE_AND_B4_DIRECT_SHARED_EMBEDDING_20260905.md"
    for path in (builder, trainer, summary, validator):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if not contract.is_file():
        raise AssertionError("B4 frozen contract is missing")
    build_text = builder.read_text(encoding="utf-8")
    train_text = trainer.read_text(encoding="utf-8")
    summarize_text = summary.read_text(encoding="utf-8")
    for token in (
        "full_bioaware__transitions.csv.gz",
        "full_no_edge_gate__transitions.csv.gz",
        "reaction_neighbours_are_positives\": False",
        "baseline_rank_array == 1",
        "query_ids - set(tensor_by_query)",
        "load_mona_reference_tensors(",
        "cross_database_row_aliasing_forbidden\": True",
        "outer_formula_fold_excluded_from_router_fit\": True",
        "evaluate_fold(",
    ):
        if token not in build_text:
            raise AssertionError(f"builder lacks contract token: {token}")
    for token in (
        "load_base_model", "unfreeze_last_blocks", "model.eval()",
        "ManifestReferenceStore", "official MoNA reference replay mismatch",
        "official external-query replay mismatch",
        "bioaware_b4_official_replay_passed",
        "held_formula_candidate_references_excluded_from_training",
        "query_reference_encoder_shared\": True",
        "candidate_inputs_at_inference\": False",
        "P2b_used\": False", "held_fold_used_for_selection\": False",
        "training_router_inner_formula_crossfit\": True",
        "reference_rows = store.rows", "reference_position = store.position",
        "manifest tensor/reference-row universe mismatch",
        "non-finite embedding", "non-finite training objective",
        "non-finite gradients", "non-finite gradient norm",
        "numeric_integrity",
    ):
        if token not in train_text:
            raise AssertionError(f"trainer lacks contract token: {token}")
    if "formula_cluster_top1_bootstrap" not in summarize_text:
        raise AssertionError("summary lacks formula-cluster inference")
    sbatch_text = sbatch.read_text(encoding="utf-8")
    if "#SBATCH --gpus=1" not in sbatch_text:
        raise AssertionError("B4 sbatch does not request exactly one GPU")
    if "#SBATCH --mem" in sbatch_text:
        raise AssertionError("B4 sbatch must use the partition default memory")
    for dependency in (
        "data/models/mona_neg_full.mgf",
        "data/models/mona_neg_dreams_emb/manifest.csv",
        "data/models/mona_neg_dreams_emb/embeddings.npy",
    ):
        if dependency not in sbatch_text:
            raise AssertionError(f"B4 sbatch lacks MoNA dependency: {dependency}")
    if "data/models/MassSpecGym_MurckoHist_split.hdf5" in sbatch_text:
        raise AssertionError("B4 sbatch still aliases MoNA rows into MassSpecGym")
    if "--smoke" not in sbatch_text or "full_bioaware_safe full_no_edge_high_recall" not in sbatch_text:
        raise AssertionError("B4 sbatch lacks smoke-first two-arm execution")
    if sbatch_text.count("--no-amp") != 2 or re.search(r"(?<!no-)--amp\b", sbatch_text):
        raise AssertionError("B4 smoke and formal training must both run with AMP disabled")
    if "np.r_[train, held]" in train_text:
        raise AssertionError("B4 replay reference index is still derived from a filtered fold subset")
    for token in ("numeric_integrity", "np.isfinite", "AMP-enabled fold is forbidden"):
        if token not in summarize_text:
            raise AssertionError(f"summary lacks numeric fail-closed token: {token}")
    test_mona_row_semantics()
    print("[BioAware B4 source contract checks] PASS")


if __name__ == "__main__":
    main()
