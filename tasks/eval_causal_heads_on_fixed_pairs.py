"""Compare every saved causal head on the exact frozen large-v2 pair panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from e1_checkpoint_io import checkpoint_kind, official_head_state, torch_load_compat


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAIRS = ROOT / "data/validation/dreams_structure_residual_atlas_large_v2"


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan"), float("nan")
    return float(pearsonr(x[valid], y[valid]).statistic), float(
        spearmanr(x[valid], y[valid]).statistic
    )


def cosine_metrics(pairs: pd.DataFrame, cosine: np.ndarray) -> dict:
    target = pairs["tanimoto"].to_numpy(float)
    same_formula = pairs["same_formula"].astype(str).str.lower().eq("true")
    masks = {
        "all_pairs": np.ones(len(pairs), dtype=bool),
        "different_identity_pairs": pairs["pair_type"].eq("different_identity").to_numpy(),
        "same_formula_different_identity_pairs": (
            pairs["pair_type"].eq("different_identity") & same_formula
        ).to_numpy(),
        "identity_pairs": pairs["pair_type"].eq("same_identity").to_numpy(),
    }
    result = {}
    for name, mask in masks.items():
        pearson, spearman = corr(target[mask], cosine[mask]) if name != "identity_pairs" else (None, None)
        result[name] = {
            "n": int(mask.sum()), "pearson_r": pearson, "spearman_rho": spearman,
            "cosine_mean": float(np.mean(cosine[mask])) if mask.any() else None,
        }
    result["cosine"] = cosine
    return result


def pair_metrics(pairs: pd.DataFrame, embeddings: np.ndarray) -> dict:
    row_a = pairs["row_a"].to_numpy(np.int64)
    row_b = pairs["row_b"].to_numpy(np.int64)
    cosine = np.einsum("ij,ij->i", embeddings[row_a], embeddings[row_b])
    return cosine_metrics(pairs, cosine)


def encode(tokens: np.ndarray, checkpoint: Path, device: torch.device, batch_size: int) -> np.ndarray:
    package = torch_load_compat(checkpoint, map_location="cpu")
    if checkpoint_kind(package) != "causal_chemmask_head":
        raise ValueError(f"Not a causal ChemMask head: {checkpoint}")
    head = official_head_state(package)
    weight = head["weight"].to(device)
    bias = head["bias"].to(device)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            values = torch.from_numpy(np.asarray(tokens[start:start + batch_size])).to(device)
            chunks.append(F.normalize(F.linear(values, weight, bias), dim=-1).cpu().numpy())
    return np.concatenate(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path)
    parser.add_argument("--pairs-dir", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    cohorts = {}
    cohort_dirs = [("discovery", args.discovery_dir)]
    if args.confirmation_dir is not None:
        cohort_dirs.append(("confirmation", args.confirmation_dir))
    for name, directory in cohort_dirs:
        tokens_path = directory / "precursor_tokens.npy"
        if not tokens_path.is_file():
            raise FileNotFoundError(f"Missing cached precursor tokens: {tokens_path}")
        tokens = np.load(tokens_path, mmap_mode="r")
        pairs = pd.read_csv(args.pairs_dir / f"{name}_structure_pairs.csv")
        baseline_score = pairs["cosine_official_finetuned"].to_numpy(float)
        cohorts[name] = (tokens, pairs, baseline_score)

    checkpoints = sorted(args.run_dir.glob("epoch_*_causal_head.pt"))
    if not checkpoints:
        checkpoints = [args.run_dir / "best_causal_head.pt"]
    rows = []
    detailed = {"protocol": "exact frozen large-v2 pairs", "baseline_official": {}, "heads": {}}
    for cohort, (_, pairs, baseline_score) in cohorts.items():
        baseline_metrics = cosine_metrics(pairs, baseline_score)
        baseline_metrics.pop("cosine")
        detailed["baseline_official"][cohort] = baseline_metrics
    for checkpoint in checkpoints:
        package = torch_load_compat(checkpoint, map_location="cpu")
        epoch = int(package.get("epoch", -1))
        entry = {
            "checkpoint": str(checkpoint.resolve()), "epoch": epoch,
            "val_metrics": package.get("val_metrics", {}), "cohorts": {},
        }
        for cohort, (tokens, pairs, baseline_score) in cohorts.items():
            embeddings = encode(tokens, checkpoint, device, args.batch_size)
            metrics = pair_metrics(pairs, embeddings)
            metrics.pop("cosine")
            entry["cohorts"][cohort] = metrics
        detailed["heads"][checkpoint.name] = entry
        val = entry["val_metrics"]
        selection_cohort = "confirmation" if "confirmation" in entry["cohorts"] else "discovery"
        conf = entry["cohorts"][selection_cohort]
        rows.append({
            "checkpoint": checkpoint.name, "epoch": epoch,
            "evaluation_cohort": selection_cohort,
            "val_loss": val.get("loss"), "val_triplet_accuracy": val.get("triplet_accuracy"),
            "pearson_different_identity": conf["different_identity_pairs"]["pearson_r"],
            "spearman_different_identity": conf["different_identity_pairs"]["spearman_rho"],
            "pearson_same_formula": conf["same_formula_different_identity_pairs"]["pearson_r"],
            "spearman_same_formula": conf["same_formula_different_identity_pairs"]["spearman_rho"],
            "identity_cosine_mean": conf["identity_pairs"]["cosine_mean"],
        })
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("epoch").to_csv(args.output / "epoch_head_metrics.csv", index=False)
    (args.output / "epoch_head_metrics.json").write_text(
        json.dumps(detailed, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).sort_values("epoch").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
