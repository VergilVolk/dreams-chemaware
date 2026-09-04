#!/usr/bin/env python
"""One-shot MoNA transfer evaluation for frozen BioAware embedding adapters.

Three preregistered final-refit seeds are evaluated individually.  The primary
candidate is the normalized mean of their embeddings; no seed is selected from
external outcomes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import ZeroInitPeakAdapter, sha256_file, strict_rank  # noqa: E402


class ExternalTokenStore:
    def __init__(self, directory: Path):
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if report.get("status") != "mona_identity_disjoint_transfer_token_cache_complete" or not report.get("formal"):
            raise RuntimeError("invalid MoNA transfer token cache")
        self.report = report
        self.rows = np.load(directory / "rows.npy")
        self.tokens = np.load(directory / "tokens_f16.npy", mmap_mode="r")
        self.mz = np.load(directory / "mz_f32.npy", mmap_mode="r")
        self.intensity = np.load(directory / "intensity_f32.npy", mmap_mode="r")
        self.valid = np.load(directory / "valid.npy", mmap_mode="r")
        with np.load(directory / "official_embeddings.npz") as body:
            embedding_rows = np.asarray(body["rows"], dtype=np.int64)
            self.official_embeddings = np.asarray(body["embeddings"], dtype=np.float32)
        if not np.array_equal(self.rows, embedding_rows):
            raise RuntimeError("token and embedding cache rows differ")
        self.official_embeddings /= np.clip(np.linalg.norm(self.official_embeddings, axis=1, keepdims=True), 1e-12, None)
        self.position = {int(row): index for index, row in enumerate(self.rows)}
        self.dimension = int(self.official_embeddings.shape[1])

    def adapt_all(self, adapter: ZeroInitPeakAdapter, device: torch.device, batch_size: int) -> np.ndarray:
        output = np.empty_like(self.official_embeddings)
        adapter.eval()
        with torch.inference_mode():
            for left in range(0, len(self.rows), batch_size):
                right = min(left + batch_size, len(self.rows))
                official = torch.from_numpy(self.official_embeddings[left:right]).to(device)
                tokens = torch.from_numpy(np.asarray(self.tokens[left:right])).to(device=device, dtype=torch.float32)
                mz = torch.from_numpy(np.asarray(self.mz[left:right])).to(device=device, dtype=torch.float32)
                intensity = torch.from_numpy(np.asarray(self.intensity[left:right])).to(device=device, dtype=torch.float32)
                valid = torch.from_numpy(np.asarray(self.valid[left:right])).to(device)
                output[left:right] = adapter(official, tokens, mz, intensity, valid)[0].cpu().numpy()
        return output


def ranks_for(panel: dict[str, np.ndarray], embedding_by_row: dict[int, np.ndarray]) -> np.ndarray:
    ranks = np.empty(len(panel["query_row"]), dtype=np.int16)
    for query, row in enumerate(panel["query_row"]):
        query_embedding = embedding_by_row[int(row)]
        scores = []
        for molecule in range(int(panel["query_ptr"][query]), int(panel["query_ptr"][query + 1])):
            left, right = map(int, panel["molecule_ptr"][molecule:molecule + 2])
            references = [embedding_by_row[int(value)] for value in panel["candidate_row"][left:right]]
            scores.append(float(np.max(np.asarray(references) @ query_embedding)))
        ranks[query] = strict_rank(np.asarray(scores))
    return ranks


def metrics(old: np.ndarray, new: np.ndarray) -> dict:
    old_correct, new_correct = old == 1, new == 1
    corrected = int(np.sum(~old_correct & new_correct))
    introduced = int(np.sum(old_correct & ~new_correct))
    discordant = corrected + introduced
    return {
        "recall1": float(np.mean(new_correct)),
        "delta_recall1": float(np.mean(new_correct) - np.mean(old_correct)),
        "mrr": float(np.mean(1.0 / new)),
        "delta_mrr": float(np.mean(1.0 / new) - np.mean(1.0 / old)),
        "corrected": corrected, "introduced": introduced,
        "mcnemar_exact_p": float(binomtest(min(corrected, introduced), discordant, 0.5).pvalue) if discordant else 1.0,
    }


def formula_bootstrap(formula: np.ndarray, old: np.ndarray, new: np.ndarray, repeats: int, seed: int) -> dict:
    table = pd.DataFrame({"formula": formula, "delta": (new == 1).astype(float) - (old == 1).astype(float)})
    groups = {str(key): group.delta.to_numpy() for key, group in table.groupby("formula", sort=True)}
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats)
    for index in range(repeats):
        selected = rng.choice(keys, len(keys), replace=True)
        values[index] = np.mean(np.concatenate([groups[str(value)] for value in selected]))
    return {"mean": float(np.mean((new == 1).astype(float) - (old == 1).astype(float))),
            "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)),
            "clusters": len(keys), "resamples": repeats}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=ROOT / "data/validation/mona_identity_disjoint_transfer_panel")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/mona_identity_disjoint_transfer_tokens")
    parser.add_argument("--model-root", type=Path, default=ROOT / "data/validation/bioaware_embedding_adapter")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260830, 20260831, 20260832])
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/mona_bioaware_embedding_transfer.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--minimum-preservation", type=float, default=0.995)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"fail-closed: external result already exists: {args.output}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    panel_report = json.loads((args.panel_dir / "report.json").read_text(encoding="utf-8"))
    if not panel_report.get("formal") or panel_report.get("development_identity_overlap") != 0:
        raise RuntimeError("MoNA transfer panel is not formally identity-disjoint")
    with np.load(args.panel_dir / "panel.npz") as body:
        panel = {key: body[key] for key in body.files}
    store = ExternalTokenStore(args.token_dir)
    if store.report["provenance"]["panel_sha256"] != sha256_file(args.panel_dir / "panel.npz"):
        raise RuntimeError("token cache was not encoded from this sealed panel")
    required_rows = set(map(int, np.concatenate((panel["query_row"], panel["candidate_row"]))))
    if required_rows != set(map(int, store.rows)):
        raise RuntimeError("transfer token cache row scope differs from sealed panel")
    official_by_row = {int(row): store.official_embeddings[index] for index, row in enumerate(store.rows)}
    old_rank = ranks_for(panel, official_by_row)
    baseline = {"recall1": float(np.mean(old_rank == 1)), "mrr": float(np.mean(1.0 / old_rank)),
                "errors": int(np.sum(old_rank != 1))}

    device = torch.device(args.device)
    seed_embeddings = []
    per_seed = {}
    model_hashes = {}
    hyperparameters = None
    for seed in args.seeds:
        directory = args.model_root / "final" / f"seed_{seed}"
        checkpoint_path, report_path = directory / "final.pt", directory / "report.json"
        if not checkpoint_path.exists() or not report_path.exists():
            raise FileNotFoundError([checkpoint_path, report_path])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "bioaware_embedding_adapter_final_refit_complete" or not report.get("formal"):
            raise RuntimeError(f"seed {seed} is not a formal final refit")
        current_hyperparameters = report.get("training", {}).get("frozen_hyperparameters")
        if hyperparameters is None:
            hyperparameters = current_hyperparameters
        elif current_hyperparameters != hyperparameters:
            raise RuntimeError("final seeds do not share the frozen OOF recipe")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        configuration = checkpoint["configuration"]
        adapter = ZeroInitPeakAdapter(
            store.dimension, int(configuration["hidden_dim"]), float(configuration["delta_bound"])
        ).to(device)
        adapter.load_state_dict(checkpoint["adapter"], strict=True)
        adapted = store.adapt_all(adapter, device, args.batch_size)
        adapted /= np.clip(np.linalg.norm(adapted, axis=1, keepdims=True), 1e-12, None)
        seed_embeddings.append(adapted)
        adapted_by_row = {int(row): adapted[index] for index, row in enumerate(store.rows)}
        ranks = ranks_for(panel, adapted_by_row)
        seed_result = metrics(old_rank, ranks)
        seed_result["preservation_mean"] = float(np.mean(np.sum(adapted * store.official_embeddings, axis=1)))
        seed_result["formula_cluster_bootstrap"] = formula_bootstrap(panel["query_formula"], old_rank, ranks, args.bootstrap, args.seed + seed)
        per_seed[str(seed)] = seed_result
        model_hashes[str(seed)] = {"checkpoint": sha256_file(checkpoint_path), "report": sha256_file(report_path)}
        del adapter

    ensemble = np.mean(np.stack(seed_embeddings).astype(np.float64), axis=0)
    ensemble /= np.clip(np.linalg.norm(ensemble, axis=1, keepdims=True), 1e-12, None)
    ensemble_by_row = {int(row): ensemble[index].astype(np.float32) for index, row in enumerate(store.rows)}
    ensemble_rank = ranks_for(panel, ensemble_by_row)
    primary = metrics(old_rank, ensemble_rank)
    primary["preservation_mean"] = float(np.mean(np.sum(ensemble * store.official_embeddings, axis=1)))
    primary["formula_cluster_bootstrap"] = formula_bootstrap(
        panel["query_formula"], old_rank, ensemble_rank, args.bootstrap, args.seed,
    )
    result = {
        "status": "mona_bioaware_embedding_transfer_complete", "formal": True,
        "baseline": baseline, "primary_three_seed_embedding_ensemble": primary,
        "individual_seeds": per_seed,
        "gates": {
            "ensemble_formula_ci_positive": primary["formula_cluster_bootstrap"]["ci_low"] > 0,
            "ensemble_mrr_positive": primary["delta_mrr"] > 0,
            "ensemble_corrected_gt_introduced": primary["corrected"] > primary["introduced"],
            "all_seeds_nonnegative": all(value["delta_recall1"] >= 0 for value in per_seed.values()),
            "all_preservation_ok": min(value["preservation_mean"] for value in per_seed.values()) >= args.minimum_preservation,
        },
        "contracts": {
            "external_outcome_used_for_seed_selection": False,
            "primary_is_normalized_mean_embedding": True,
            "query_reference_encoder_shared": True,
            "P2b": "forbidden", "phenotype_labels": "forbidden",
        },
        "frozen_hyperparameters": hyperparameters,
        "provenance": {
            "panel_sha256": sha256_file(args.panel_dir / "panel.npz"),
            "panel_report_sha256": sha256_file(args.panel_dir / "report.json"),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "models": model_hashes, "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": panel_report["claim_limit"],
    }
    result["gates"]["pass"] = all(result["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
