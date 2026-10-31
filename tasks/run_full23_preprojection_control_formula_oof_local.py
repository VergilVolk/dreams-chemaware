#!/usr/bin/env python3
"""Capacity-matched pre-interaction compression control for the local tensor pilot."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from enveda_unified_candidate_integrator_core import weighted_query_residual_penalty
from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel, load_embeddings
from full23_multireference_readout_core import stable_topk_references
from full23_tensor_set_readout_core import (
    Full23TensorSetReadout,
    TensorSetSettings,
    weighted_anchor_safe_ranking_loss,
)
from run_full23_tensor_set_formula_oof_local import (
    BATCH_QUERIES, DATA_ROOT, EMBEDDINGS, EPOCHS, MANIFEST, PANEL, PARENT,
    RESIDUAL_BOUND, SEED, pack, predict,
)
from train_enveda_unified_candidate_integrator import _comparison, _metric_summary


OUTPUT = DATA_ROOT / "data/validation/full23_preprojection_control_formula_oof_local_20261031"
TENSOR_RESULT = DATA_ROOT / "data/validation/full23_tensor_set_formula_oof_local_20261031/candidate_predictions_formula_disjoint.npz"


class PreprojectionControl(Full23TensorSetReadout):
    """Same set/cross-encoder readout, but q/r are compressed before interaction."""

    def __init__(self, settings, residual_bound, projection_dimension=32):
        super().__init__(settings, residual_bound)
        self.preprojections = nn.ModuleDict({
            name: nn.Linear(settings.embedding_dimension, projection_dimension, bias=False)
            for name in settings.encoder_names
        })
        self.pair_encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(2 * projection_dimension, settings.pair_dimension),
                nn.GELU(), nn.LayerNorm(settings.pair_dimension),
            )
            for name in settings.encoder_names
        })
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def _encoder_token(self, name, query, references, mask):
        dtype = self.preprojections[name].weight.dtype
        query = F.normalize(self.preprojections[name](query.to(dtype=dtype)), dim=-1)
        references = F.normalize(
            self.preprojections[name](references.to(dtype=dtype)), dim=-1,
        )
        expanded = query[:, None, :].expand_as(references)
        interaction = torch.cat((torch.abs(expanded - references), expanded * references), -1)
        pair = self.pair_encoders[name](interaction)
        attention_logits = self.reference_attention[name](pair).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, -torch.inf)
        attention = torch.softmax(attention_logits, dim=1)
        mean = torch.sum(attention[:, :, None] * pair, dim=1)
        centered = pair - mean[:, None, :]
        deviation = torch.sqrt(
            torch.sum(attention[:, :, None] * centered.square(), dim=1).clamp_min(1e-8)
        )
        count = mask.sum(1, keepdim=True).to(dtype=dtype) / float(self.settings.top_k)
        return self.encoder_summaries[name](torch.cat((mean, deviation, count), -1))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest_rows = len(pd.read_csv(MANIFEST, usecols=["row"]))
    graph = graph_from_panel(PANEL)
    with np.load(PANEL, allow_pickle=False) as z:
        formulas = np.asarray(z["query_formula"]).astype("U")
        near = np.asarray(z["near_query"], dtype=bool)
    with np.load(PARENT, allow_pickle=False) as z:
        query_ptr = np.asarray(z["query_ptr"], dtype=np.int64)
        labels = np.asarray(z["labels"], dtype=np.int8)
        anchor = np.asarray(z["full23_v1"], dtype=np.float32)
    with np.load(TENSOR_RESULT, allow_pickle=False) as z:
        tensor_oof = np.asarray(z["tensor_set_three_encoder_oof"], dtype=np.float32)
        folds = np.asarray(z["fold"], dtype=np.int8)
    names = tuple(EMBEDDINGS)
    banks, topk = {}, {}
    for name, path in EMBEDDINGS.items():
        rows, values = load_embeddings(path, manifest_rows)
        banks[name] = torch.as_tensor(values)
        topk[name] = stable_topk_references(graph, rows, values)
    settings = TensorSetSettings(
        encoder_names=names, embedding_dimension=1024, pair_dimension=16,
        encoder_dimension=16, attention_heads=2, fusion_hidden=32,
    )
    oof = np.full(len(anchor), np.nan, dtype=np.float32)
    histories = []
    parameter_counts = None
    rng = np.random.default_rng(SEED)
    for fold in range(2):
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        torch.manual_seed(SEED + fold)
        model = PreprojectionControl(settings, RESIDUAL_BOUND)
        if parameter_counts is None:
            tensor_model = Full23TensorSetReadout(settings, RESIDUAL_BOUND)
            parameter_counts = {
                "postinteraction_tensor": tensor_model.trainable_parameters,
                "preprojection_control": model.trainable_parameters,
            }
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        fold_history = []
        for epoch in range(EPOCHS):
            order = train[rng.permutation(len(train))]
            total = 0.0
            for start in range(0, len(order), BATCH_QUERIES):
                selected = order[start:start + BATCH_QUERIES]
                a, y, q, r, m, ptr, _ = pack(
                    selected, query_ptr, anchor, labels, topk, names,
                )
                a_t = torch.as_tensor(a)
                logits = model(a_t, torch.as_tensor(q), torch.as_tensor(r), torch.as_tensor(m), banks)
                weights = np.ones(len(selected), dtype=np.float32)
                ranking, _ = weighted_anchor_safe_ranking_loss(logits, a_t, ptr, y, weights)
                loss = ranking + 1e-2 * weighted_query_residual_penalty(
                    logits - a_t, ptr, weights,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach()) * len(selected)
            fold_history.append(total / len(train))
            print(json.dumps({
                "fold": fold, "epoch": epoch + 1, "loss": fold_history[-1],
                "train_queries": len(train),
            }), flush=True)
        histories.append(fold_history)
        values = predict(model.eval(), valid, query_ptr, anchor, labels, topk, names, banks)
        candidate = np.concatenate([np.arange(query_ptr[q], query_ptr[q + 1]) for q in valid])
        oof[candidate] = values[candidate]
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("control OOF coverage incomplete")
    panel = SimpleNamespace(
        name="formula_disjoint", labels=labels, query_ptr=query_ptr,
        near=near, formulas=formulas,
    )
    methods = {"full23_v1": anchor, "preprojection_control": oof, "postinteraction_tensor": tensor_oof}
    metrics = {name: _metric_summary(value, panel)[0] for name, value in methods.items()}
    comparisons = {
        "preprojection_vs_full23": _comparison(oof, anchor, panel, 5000, SEED + 10),
        "postinteraction_vs_preprojection": _comparison(tensor_oof, oof, panel, 5000, SEED + 11),
        "postinteraction_vs_full23": _comparison(tensor_oof, anchor, panel, 5000, SEED + 12),
    }
    OUTPUT.mkdir(parents=True)
    np.savez_compressed(
        OUTPUT / "candidate_predictions_formula_disjoint.npz",
        query_ptr=query_ptr, labels=labels, full23_v1=anchor,
        preprojection_control=oof, postinteraction_tensor=tensor_oof, fold=folds,
    )
    report = {
        "status": "FULL23_PREINTERACTION_COMPRESSION_CONTROL_FORMULA_OOF_COMPLETE",
        "dataset": {"role": "consumed GNPS local development", "queries": len(formulas)},
        "parameter_counts": parameter_counts,
        "training_history": histories,
        "metrics": metrics,
        "comparisons": comparisons,
        "claim_boundary": "Matched folds/loss/data. Parameter counts are reported, not assumed equal.",
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "output": str(OUTPUT),
        "parameter_counts": parameter_counts,
        "delta_post_vs_pre_pp": comparisons["postinteraction_vs_preprojection"]["recall1_formula_cluster_ci"]["delta_pp"],
    }), flush=True)


if __name__ == "__main__":
    main()
