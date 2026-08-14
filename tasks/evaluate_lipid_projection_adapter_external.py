"""External, model-selection-free evaluation of lipid projection adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from train_lipid_projection_adapter import LowRankDelta  # noqa: E402
from e1_checkpoint_io import official_head_state, torch_load_compat  # noqa: E402


def is_phospholipid(unit: dict) -> bool:
    return unit["ring_class"] == "acyclic" and "P" in unit["formula"]


def score(units, embeddings, protocol):
    rows = []
    for unit in units:
        if not unit["is_query_anchor"] or not unit[protocol]: continue
        pair_id = int(unit["pair_id"]); negatives = [int(v) for v in unit[protocol]]
        for view in (0, 1):
            query = embeddings[pair_id, view]
            positive = float(query @ embeddings[pair_id, 1 - view])
            neg_scores = np.einsum("nvd,d->nv", embeddings[negatives], query).max(axis=1)
            best_position = int(np.argmax(neg_scores))
            best_pair_id = negatives[best_position]
            best = float(neg_scores[best_position])
            rows.append({
                "ik14": unit["ik14"], "view": view, "ring_class": unit["ring_class"],
                "phospholipid_like": is_phospholipid(unit), "positive": positive,
                "best_negative": best, "margin": positive - best,
                "top1": positive > best, "pairwise": float(np.mean(positive > neg_scores)),
                "n_negatives": len(negatives),
                "best_negative_pair_id": best_pair_id,
                "best_negative_ik14": units[best_pair_id]["ik14"],
            })
    return pd.DataFrame(rows)


def summary(frame):
    output = {}
    masks = {
        "overall": np.ones(len(frame), dtype=bool),
        "phospholipid_like": frame["phospholipid_like"].to_numpy(bool),
        "non_phospholipid": ~frame["phospholipid_like"].to_numpy(bool),
    }
    for name, mask in masks.items():
        part = frame.loc[mask]
        if part.empty: continue
        labels = np.concatenate((np.ones(len(part)), np.zeros(len(part))))
        scores = np.concatenate((part["positive"], part["best_negative"]))
        output[name] = {
            "molecules": int(part["ik14"].nunique()), "views": len(part),
            "top1": float(part["top1"].mean()), "pairwise": float(part["pairwise"].mean()),
            "hard_negative_auc": float(roc_auc_score(labels, scores)),
            "mean_margin": float(part["margin"].mean()),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, default=Path("data/validation/external_ring_balanced_pilot"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/validation/external_ring_balanced_embeddings"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/e1/lipid_projection_pilot/runs"))
    parser.add_argument("--official-checkpoint", type=Path, default=Path("data/e1/official_embedding_slim.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/lipid_projection_adapter_external"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    official_package = torch_load_compat(args.official_checkpoint, map_location="cpu")
    head = official_head_state(official_package)
    weight, bias = head["weight"].float(), head["bias"].float()
    checkpoints = sorted(args.run_dir.glob("seed_*/best.pt"))
    reports = {}; all_rows = []
    for split in ("discovery", "confirmation"):
        units = json.loads((args.pilot_dir / f"{split}_manifest.json").read_text(encoding="utf-8"))["units"]
        backbone_np = np.load(args.embedding_dir / f"{split}_backbone.npy").astype(np.float32)
        backbone = torch.from_numpy(backbone_np.reshape(-1, backbone_np.shape[-1]))
        raw = F.linear(backbone, weight, bias)
        official = F.normalize(raw, dim=-1).numpy().reshape(backbone_np.shape)
        official_frame = score(units, official, "same_formula_negative_pair_ids")
        reports.setdefault("official", {})[split] = summary(official_frame)
        official_rows = official_frame.copy(); official_rows["split"] = split; official_rows["seed"] = -1
        all_rows.append(official_rows)
        for ckpt_path in checkpoints:
            package = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = LowRankDelta(package["dimension"], package["rank"], package["scale"])
            model.load_state_dict(package["state_dict"]); model.eval()
            with torch.no_grad(): adapted = model(backbone, raw).numpy().reshape(backbone_np.shape)
            frame = score(units, adapted, "same_formula_negative_pair_ids")
            seed = str(package["result"]["seed"])
            reports.setdefault(f"seed_{seed}", {})[split] = summary(frame)
            merged = frame.copy(); merged["split"] = split; merged["seed"] = int(seed)
            all_rows.append(merged)
    pd.concat(all_rows, ignore_index=True).to_csv(args.output_dir / "query_results.csv", index=False)

    deltas = []
    for model_name, values in reports.items():
        if model_name == "official": continue
        for split in ("discovery", "confirmation"):
            for domain in ("overall", "phospholipid_like", "non_phospholipid"):
                for metric in ("top1", "pairwise", "hard_negative_auc", "mean_margin"):
                    deltas.append({
                        "model": model_name, "split": split, "domain": domain, "metric": metric,
                        "official": reports["official"][split][domain][metric],
                        "adapted": values[split][domain][metric],
                        "delta": values[split][domain][metric] - reports["official"][split][domain][metric],
                    })
    delta_frame = pd.DataFrame(deltas); delta_frame.to_csv(args.output_dir / "deltas.csv", index=False)
    key = delta_frame[(delta_frame["domain"] == "phospholipid_like") & (delta_frame["metric"].isin(["top1", "mean_margin"]))]
    report = {
        "status": "lipid_projection_adapter_external_evaluation", "models": reports,
        "key_deltas": key.to_dict(orient="records"),
        "decision_rule": "Proceed only if phospholipid direction improves across all seeds in both discovery and confirmation, while non-phospholipid Top-1 degradation is <=0.01.",
        "external_sets_used_for_selection": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(key.to_string(index=False))
    print(delta_frame[(delta_frame["domain"] == "non_phospholipid") & (delta_frame["metric"] == "top1")].to_string(index=False))


if __name__ == "__main__":
    main()
