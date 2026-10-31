#!/usr/bin/env python3
"""Two-fold formula-OOF local pilot for delayed-compression embedding readout.

This intentionally uses the three complete ChemAware-family GNPS embedding
banks available locally.  Noise remains present through the frozen Full23
parent but its missing local 1024-D bank is not fabricated.  GNPS is consumed
development; this script cannot produce an external claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from enveda_unified_candidate_integrator_core import weighted_query_residual_penalty
from evaluate_gnps_gold_silver_10ppm_embeddings import (
    graph_from_panel,
    load_embeddings,
)
from full23_multireference_readout_core import stable_topk_references
from full23_tensor_set_readout_core import (
    Full23TensorSetReadout,
    TensorSetSettings,
    weighted_anchor_safe_ranking_loss,
)
from train_enveda_unified_candidate_integrator import _comparison, _metric_summary


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DREAMS_DATA_ROOT", str(ROOT)))
PANEL = DATA_ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1/panel_formula_disjoint.npz"
MANIFEST = DATA_ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1/manifest.csv.gz"
PARENT = DATA_ROOT / "data/validation/full23_multireference_readout_run_2420264/gnps_development_adjudication/candidate_predictions_formula_disjoint.npz"
EMBEDDINGS = {
    "chemaware_stage1": DATA_ROOT / "data/validation/enveda_unified_candidate_integrator_run_2411772/gnps_inputs/chemaware_stage1_embeddings.npz",
    "chemaware_phasea": DATA_ROOT / "data/validation/enveda_unified_candidate_integrator_run_2411772/gnps_inputs/chemaware_phasea_embeddings.npz",
    "noise_chemaware_fullmsg": DATA_ROOT / "data/validation/enveda_unified_candidate_integrator_run_2411772/gnps_inputs/noise_chemaware_fullmsg_embeddings.npz",
}
OUTPUT = DATA_ROOT / "data/validation/full23_tensor_set_formula_oof_local_20261031"
SEED = 20261031
EPOCHS = 3
BATCH_QUERIES = 48
RESIDUAL_BOUND = 0.6255066394805908


def pack(indices, query_ptr, anchor, labels, topk, names):
    ranges = [np.arange(query_ptr[q], query_ptr[q + 1]) for q in indices]
    candidate = np.concatenate(ranges)
    lengths = np.asarray([len(x) for x in ranges], dtype=np.int64)
    ptr = np.concatenate(([0], np.cumsum(lengths)))
    query = np.stack([topk[name].query_positions[candidate] for name in names], axis=1)
    refs = np.stack([topk[name].reference_positions[candidate] for name in names], axis=1)
    mask = np.stack([topk[name].reference_mask[candidate] for name in names], axis=1)
    return anchor[candidate], labels[candidate], query, refs, mask, ptr, candidate


@torch.inference_mode()
def predict(model, indices, query_ptr, anchor, labels, topk, names, banks):
    output = np.full(len(anchor), np.nan, dtype=np.float32)
    for start in range(0, len(indices), BATCH_QUERIES):
        batch = indices[start:start + BATCH_QUERIES]
        a, _, q, r, m, _, candidate = pack(
            batch, query_ptr, anchor, labels, topk, names,
        )
        value = model(
            torch.as_tensor(a), torch.as_tensor(q), torch.as_tensor(r),
            torch.as_tensor(m), banks,
        ).cpu().numpy()
        output[candidate] = value
    return output


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, min(16, torch.get_num_threads())))
    manifest_rows = len(pd.read_csv(MANIFEST, usecols=["row"]))
    graph = graph_from_panel(PANEL)
    with np.load(PANEL, allow_pickle=False) as z:
        formulas = np.asarray(z["query_formula"]).astype("U")
        near = np.asarray(z["near_query"], dtype=bool)
    with np.load(PARENT, allow_pickle=False) as z:
        query_ptr = np.asarray(z["query_ptr"], dtype=np.int64)
        labels = np.asarray(z["labels"], dtype=np.int8)
        anchor = np.asarray(z["full23_v1"], dtype=np.float32)
    if not np.array_equal(query_ptr, graph.query_ptr) or not np.array_equal(labels, graph.molecule_label):
        raise RuntimeError("parent/panel candidate axis drifted")

    cpu_banks = {}
    topk = {}
    for name, path in EMBEDDINGS.items():
        rows, values = load_embeddings(path, manifest_rows)
        cpu_banks[name] = values
        topk[name] = stable_topk_references(graph, rows, values)
    names = tuple(EMBEDDINGS)
    banks = {name: torch.as_tensor(value) for name, value in cpu_banks.items()}

    unique = np.unique(formulas)
    rng = np.random.default_rng(SEED)
    shuffled = unique[rng.permutation(len(unique))]
    assignment = {formula: index % 2 for index, formula in enumerate(shuffled)}
    folds = np.asarray([assignment[value] for value in formulas], dtype=np.int8)
    oof = np.full(len(anchor), np.nan, dtype=np.float32)
    histories = []
    settings = TensorSetSettings(
        encoder_names=names,
        embedding_dimension=1024,
        pair_dimension=16,
        encoder_dimension=16,
        attention_heads=2,
        fusion_hidden=32,
    )
    for fold in range(2):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        torch.manual_seed(SEED + fold)
        model = Full23TensorSetReadout(settings, RESIDUAL_BOUND)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        fold_history = []
        for epoch in range(EPOCHS):
            order = train[rng.permutation(len(train))]
            total = 0.0
            mass = 0
            model.train()
            for start in range(0, len(order), BATCH_QUERIES):
                selected = order[start:start + BATCH_QUERIES]
                a, y, q, r, m, ptr, _ = pack(
                    selected, query_ptr, anchor, labels, topk, names,
                )
                a_t = torch.as_tensor(a)
                logits = model(
                    a_t, torch.as_tensor(q), torch.as_tensor(r),
                    torch.as_tensor(m), banks,
                )
                weights = np.ones(len(selected), dtype=np.float32)
                ranking, _ = weighted_anchor_safe_ranking_loss(
                    logits, a_t, ptr, y, weights,
                )
                loss = ranking + 1e-2 * weighted_query_residual_penalty(
                    logits - a_t, ptr, weights,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach()) * len(selected)
                mass += len(selected)
            fold_history.append(total / mass)
            print(json.dumps({
                "fold": fold, "epoch": epoch + 1,
                "loss": fold_history[-1], "train_queries": len(train),
            }), flush=True)
        histories.append(fold_history)
        prediction = predict(
            model.eval(), valid, query_ptr, anchor, labels, topk, names, banks,
        )
        candidate = np.concatenate([
            np.arange(query_ptr[q], query_ptr[q + 1]) for q in valid
        ])
        oof[candidate] = prediction[candidate]
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("OOF coverage incomplete")

    panel = SimpleNamespace(
        name="formula_disjoint", labels=labels, query_ptr=query_ptr,
        near=near, formulas=formulas,
    )
    parent_metrics, _ = _metric_summary(anchor, panel)
    oof_metrics, _ = _metric_summary(oof, panel)
    comparison = _comparison(oof, anchor, panel, 5000, SEED)
    OUTPUT.mkdir(parents=True)
    np.savez_compressed(
        OUTPUT / "candidate_predictions_formula_disjoint.npz",
        query_ptr=query_ptr, labels=labels, formulas=formulas,
        full23_v1=anchor, tensor_set_three_encoder_oof=oof, fold=folds,
    )
    report = {
        "status": "FULL23_THREE_ENCODER_TENSOR_SET_FORMULA_OOF_LOCAL_COMPLETE",
        "dataset": {
            "name": "GNPS Gold/Silver formula-disjoint",
            "role": "consumed local development",
            "queries": int(len(formulas)),
            "candidates": int(len(labels)),
            "reference_edges": int(len(graph.pair_candidate_row)),
            "formula_clusters": int(len(unique)),
        },
        "missing_local_asset": "Noise V1 1024D GNPS embedding bank",
        "noise_role": "present only through frozen Full23 parent",
        "encoders": list(names),
        "settings": settings.__dict__,
        "folds": "two-fold formula-group OOF",
        "training_history": histories,
        "metrics": {"full23_v1": parent_metrics, "tensor_set_oof": oof_metrics},
        "comparison": comparison,
        "claim_boundary": (
            "Consumed GNPS development only. This diagnoses delayed compression; "
            "it is not the complete four-encoder Enveda-trained production model."
        ),
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "output": str(OUTPUT),
        "recall1_parent": parent_metrics["recall@1"],
        "recall1_tensor": oof_metrics["recall@1"],
        "delta_pp": comparison["recall1_formula_cluster_ci"]["delta_pp"],
    }), flush=True)


if __name__ == "__main__":
    main()
