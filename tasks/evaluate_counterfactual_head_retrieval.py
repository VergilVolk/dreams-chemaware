"""Evaluate head-only pilots on the full formula-isolated validation retrieval pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import pilot_multilevel_factor_activations as multi


def encode_hidden(model, data: Path, rows: np.ndarray, n_peaks: int, batch: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(multi.SpectrumRows(data, rows, n_peaks), batch_size=batch, shuffle=False, num_workers=0)
    values = []
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for position, spectra in enumerate(loader, start=1):
            values.append(model(spectra.to(device=device, dtype=dtype), None)[:, 0].float().cpu().numpy())
            if position % 100 == 0:
                print(f"  encoded {position}/{len(loader)} batches", flush=True)
    return np.concatenate(values)


def query_metrics(manifest: pd.DataFrame, embeddings: np.ndarray, validation_formulas: set[str], ppm: float) -> pd.DataFrame:
    rows = []
    for query in manifest.index[manifest["formula"].isin(validation_formulas)]:
        formula = manifest.at[query, "formula"]
        candidates = manifest.index[(manifest["formula"] == formula) & (manifest.index != query)].to_numpy(np.int64)
        if not len(candidates):
            continue
        query_mass = float(manifest.at[query, "precursor_mz"])
        mass = manifest.loc[candidates, "precursor_mz"].to_numpy(float)
        delta = np.abs(mass - query_mass) / ((mass + query_mass) / 2) * 1e6
        distinct = manifest.loc[candidates, "spectrum_hash"].to_numpy() != manifest.at[query, "spectrum_hash"]
        candidates = candidates[(delta <= ppm) & distinct]
        if not len(candidates):
            continue
        candidate_ik = manifest.loc[candidates, "ik14"].to_numpy()
        truth = manifest.at[query, "ik14"]
        positives = candidates[candidate_ik == truth]
        negatives = candidates[candidate_ik != truth]
        if not len(positives) or not len(negatives):
            continue
        similarity = embeddings[candidates] @ embeddings[query]
        positive_score = float(similarity[candidate_ik == truth].max())
        negative_frame = pd.DataFrame({"ik14": candidate_ik[candidate_ik != truth], "score": similarity[candidate_ik != truth]})
        negative_scores = negative_frame.groupby("ik14", sort=False)["score"].max().to_numpy(float)
        rank = 1 + int(np.sum(negative_scores >= positive_score))
        rows.append({
            "query_index": int(query), "ik14": truth, "formula": formula,
            "top1": rank == 1, "mrr": 1 / rank,
            "pairwise_accuracy": float(np.mean(positive_score > negative_scores)),
            "positive_score": positive_score,
            "best_negative_score": float(negative_scores.max()),
            "margin": positive_score - float(negative_scores.max()),
        })
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    labels = np.r_[np.ones(len(frame)), np.zeros(len(frame))]
    scores = np.r_[frame["positive_score"], frame["best_negative_score"]]
    return {
        "queries": len(frame), "molecules": int(frame["ik14"].nunique()),
        "formulas": int(frame["formula"].nunique()),
        "top1": float(frame["top1"].mean()), "mrr": float(frame["mrr"].mean()),
        "pairwise_accuracy": float(frame["pairwise_accuracy"].mean()),
        "hard_negative_roc_auc": float(roc_auc_score(labels, scores)),
        "mean_margin": float(frame["margin"].mean()),
    }


def bootstrap(baseline: pd.DataFrame, model: pd.DataFrame, column: str, iterations: int, seed: int) -> list[float]:
    merged = baseline[["query_index", "formula", column]].merge(
        model[["query_index", "formula", column]], on=["query_index", "formula"],
        suffixes=("_baseline", "_model"), validate="one_to_one",
    )
    groups = {
        formula: group[f"{column}_model"].to_numpy(float) - group[f"{column}_baseline"].to_numpy(float)
        for formula, group in merged.groupby("formula")
    }
    formulas = np.asarray(list(groups), object)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iterations):
        sampled = rng.choice(formulas, len(formulas), replace=True)
        draws.append(float(np.concatenate([groups[value] for value in sampled]).mean()))
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=Path("data/e1/counterfactual_peak_finetune/counterfactual_peak_finetune_split.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/validation/large_observability_embeddings_discovery/manifest.csv"))
    parser.add_argument("--official-embeddings", type=Path, default=Path("data/validation/large_observability_embeddings_discovery/official_embeddings.npy"))
    parser.add_argument("--cache", type=Path, default=Path("data/e1/counterfactual_head_cache/all_discovery_hidden.npy"))
    parser.add_argument("--heads", nargs="+", default=[
        "full=data/e1/counterfactual_head_ablation_full/counterfactual_head.pt",
        "triplet=data/e1/counterfactual_head_ablation_triplet/counterfactual_head.pt",
        "counterfactual_only=data/e1/counterfactual_head_ablation_cfonly/counterfactual_head.pt",
    ])
    parser.add_argument("--raw-checkpoint", type=Path, default=multi.DEFAULT_RAW)
    parser.add_argument("--official-checkpoint", type=Path, default=multi.DEFAULT_OFFICIAL)
    parser.add_argument("--data", type=Path, default=multi.DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("data/e1/counterfactual_head_retrieval"))
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    manifest = pd.read_csv(args.manifest)
    split = pd.read_csv(args.split)
    validation_formulas = set(split.loc[split["pilot_split"] == "validation", "formula"])
    if args.cache.exists():
        hidden = np.load(args.cache)
    else:
        raw = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
        official = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
        model = multi.reconstruct_backbone(raw, multi.official_backbone_state(official), device)
        model.eval()
        hidden = encode_hidden(model, args.data, manifest["hdf5_row"].to_numpy(np.int64), args.n_highest_peaks, args.batch_size, device)
        np.save(args.cache, hidden)
    official_embeddings = np.load(args.official_embeddings).astype(np.float32)
    baseline = query_metrics(manifest, official_embeddings, validation_formulas, args.ppm)
    baseline.to_csv(args.output_dir / "official_queries.csv", index=False)
    report = {"status": "counterfactual_head_full_retrieval", "baseline": summarize(baseline), "models": {}}
    for position, item in enumerate(args.heads):
        name, path = item.split("=", 1)
        package = torch.load(path, map_location="cpu", weights_only=False)
        state = package["head_state_dict"]
        with torch.inference_mode():
            embedding = F.normalize(
                F.linear(torch.from_numpy(hidden).float(), state["weight"].float(), state["bias"].float()), dim=-1
            ).numpy()
        query = query_metrics(manifest, embedding, validation_formulas, args.ppm)
        query.to_csv(args.output_dir / f"{name}_queries.csv", index=False)
        result = summarize(query)
        result.update({
            "top1_minus_official": result["top1"] - report["baseline"]["top1"],
            "top1_formula_bootstrap_ci95": bootstrap(baseline, query, "top1", args.bootstrap, 20260813 + position),
            "mrr_minus_official": result["mrr"] - report["baseline"]["mrr"],
            "mrr_formula_bootstrap_ci95": bootstrap(baseline, query, "mrr", args.bootstrap, 20260913 + position),
            "mean_embedding_cosine_to_official": float(np.mean(np.sum(embedding * official_embeddings, axis=1))),
        })
        report["models"][name] = result
    report["claim_limit"] = "internal formula-isolated validation; confirmation and test not evaluated"
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
