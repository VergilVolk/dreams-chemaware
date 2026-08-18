"""CPU-feasible paired evaluation of a causal head against official DreaMS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from e1_checkpoint_io import checkpoint_kind, official_head_state, torch_load_compat
from train_causal_chemmask_head import (
    CausalDynamicTripletDataset, DEFAULT_DATA, DEFAULT_OFFICIAL, DEFAULT_RAW, DEFAULT_VAL,
)
from train_e1_identity import CandidatePool, load_base_model, seed_everything


def summarize(positive: np.ndarray, negative: np.ndarray, margin: float) -> dict:
    separation = positive - negative
    return {
        "n": int(len(separation)),
        "positive_cosine": float(positive.mean()),
        "negative_cosine": float(negative.mean()),
        "separation": float(separation.mean()),
        "triplet_accuracy": float((separation > 0).mean()),
        "margin_accuracy": float((separation >= margin).mean()),
        "triplet_loss": float(np.maximum(0.0, margin - separation).mean()),
    }


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--panel", choices=("random", "hard", "masked", "hard-masked"),
        default="random",
    )
    parser.add_argument("--triplets", type=int, default=0)
    parser.add_argument("--negative-probe-size", type=int, default=32)
    parser.add_argument(
        "--panel-seed", type=int, default=20260815,
        help="Checkpoint-independent seed for sampling anchors and candidate spectra.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))
        try:
            torch.set_num_interop_threads(min(2, max(1, args.cpu_threads)))
        except RuntimeError:
            pass

    package = torch_load_compat(args.checkpoint, map_location="cpu")
    kind = checkpoint_kind(package)
    if kind not in {"causal_chemmask_head", "counterfactual_dreams"}:
        raise ValueError("Checkpoint must be a causal ChemMask or counterfactual DreaMS head")
    config = package.get("config", {})
    checkpoint_seed = int(config.get("seed", 20260815))
    seed_everything(args.panel_seed)
    val_pool = CandidatePool(Path(config.get("val_pool", DEFAULT_VAL)))
    requested = args.triplets if args.triplets > 0 else int(config.get("val_triplets", 100))
    val_n = min(requested, len(val_pool))
    use_hard = args.panel in {"hard", "hard-masked"}
    use_mask = args.panel in {"masked", "hard-masked"}
    base_dataset = CausalDynamicTripletDataset(
        Path(config.get("data", DEFAULT_DATA)), val_pool, int(config.get("n_highest_peaks", 100)),
        args.panel_seed + 97, 1.0 if use_hard else 0.0,
        args.negative_probe_size if use_hard else 1,
        1.0 if use_mask else 0.0,
        float(config.get("identity_mask_max_fraction", 0.3)),
        int(config.get("identity_mask_max_peaks", 12)),
        float(config.get("fragment_tolerance", 0.02)), length=len(val_pool), fixed=True,
    )
    panel_rng = np.random.RandomState(args.panel_seed)
    panel_rows = panel_rng.choice(len(val_pool), size=val_n, replace=False).astype(np.int64)
    dataset = Subset(base_dataset, panel_rows.tolist())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Loading official DreaMS for paired CPU evaluation of {val_n} triplets...", flush=True)
    model, initialization_kind = load_base_model(
        Path(config.get("base_ckpt", DEFAULT_OFFICIAL)),
        Path(config.get("architecture_ckpt", DEFAULT_RAW)), device,
        int(config.get("n_highest_peaks", 100)),
    )
    official_weight = model.head.weight.detach().clone()
    official_bias = model.head.bias.detach().clone()
    model.head.load_state_dict(official_head_state(package), strict=True)
    model.eval()

    official_pos, official_neg, candidate_pos, candidate_neg, preservation = [], [], [], [], []
    hard_selected, hard_scores, masked, masked_counts = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            negative = batch["negative"].to(device)
            size = len(anchor)
            spectra = torch.cat((anchor, positive, negative), dim=0)
            dtype = next(model.backbone.parameters()).dtype
            precursor = model.backbone(spectra.to(dtype=dtype), None)[:, 0, :]
            official = F.normalize(F.linear(precursor, official_weight, official_bias), dim=-1)
            candidate = F.normalize(model.head(precursor), dim=-1)
            official_pos.append((official[:size] * official[size:2 * size]).sum(1).cpu().numpy())
            official_neg.append((official[:size] * official[2 * size:]).sum(1).cpu().numpy())
            candidate_pos.append((candidate[:size] * candidate[size:2 * size]).sum(1).cpu().numpy())
            candidate_neg.append((candidate[:size] * candidate[2 * size:]).sum(1).cpu().numpy())
            preservation.append((official * candidate).sum(1).cpu().numpy())
            hard_selected.extend(batch["hard_negative_selected"].numpy().tolist())
            hard_scores.extend(batch["hard_negative_score"].numpy().tolist())
            masked.extend(batch["identity_masked"].numpy().tolist())
            masked_counts.extend(batch["masked_peak_count"].numpy().tolist())

    op, on, cp, cn = map(np.concatenate, (official_pos, official_neg, candidate_pos, candidate_neg))
    preserve = np.concatenate(preservation)
    margin = float(config.get("margin", 0.1))
    official_sep, candidate_sep = op - on, cp - cn
    official_correct, candidate_correct = official_sep > 0, candidate_sep > 0
    rng = np.random.RandomState(args.panel_seed + 811)
    accuracy_delta_boot, separation_delta_boot = [], []
    for _ in range(args.n_bootstrap):
        index = rng.randint(0, len(op), size=len(op))
        accuracy_delta_boot.append(
            float(candidate_correct[index].mean() - official_correct[index].mean())
        )
        separation_delta_boot.append(float((candidate_sep[index] - official_sep[index]).mean()))

    report = {
        "status": "paired_cpu_validation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_kind": kind,
        "checkpoint_epoch": int(package.get("epoch", -1)),
        "checkpoint_seed": checkpoint_seed,
        "initialization": initialization_kind,
        "protocol": (
            "same fixed strict-10ppm validation triplets; one shared backbone forward; "
            f"panel={args.panel}"
        ),
        "panel_audit": {
            "requested_panel": args.panel,
            "hard_negative_selected_fraction": float(np.mean(hard_selected)),
            "hard_negative_score_mean_when_selected": (
                float(np.mean([x for x in hard_scores if x >= 0]))
                if any(x >= 0 for x in hard_scores) else None
            ),
            "identity_masked_fraction": float(np.mean(masked)),
            "masked_peak_count_mean_when_applied": (
                float(np.mean([x for x in masked_counts if x > 0]))
                if any(x > 0 for x in masked_counts) else None
            ),
            "negative_probe_size": args.negative_probe_size if use_hard else 1,
            "panel_seed": args.panel_seed,
            "anchor_sampling": "uniform_without_replacement_from_full_validation_anchor_pool",
            "anchor_pool_size": int(len(val_pool)),
            "anchor_pool_rows_sha256": hashlib.sha256(panel_rows.tobytes()).hexdigest()[:16],
        },
        "official": summarize(op, on, margin),
        "candidate": summarize(cp, cn, margin),
        "paired_delta": {
            "triplet_accuracy": float(candidate_correct.mean() - official_correct.mean()),
            "triplet_accuracy_95ci": percentile_ci(np.asarray(accuracy_delta_boot)),
            "separation": float((candidate_sep - official_sep).mean()),
            "separation_95ci": percentile_ci(np.asarray(separation_delta_boot)),
            "official_wrong_candidate_right": int((~official_correct & candidate_correct).sum()),
            "official_right_candidate_wrong": int((official_correct & ~candidate_correct).sum()),
        },
        "embedding_preservation_cosine": {
            "mean": float(preserve.mean()), "min": float(preserve.min()),
            "p05": float(np.quantile(preserve, 0.05)),
        },
        "interpretation_limit": (
            f"{val_n} triplets form an internal directional CPU audit; "
            "formal evidence still requires molecule-clustered uncertainty, multiple seeds, "
            "and locked confirmation/test evaluation"
        ),
    }
    output = args.output or args.checkpoint.parent / (
        f"cpu_paired_eval_{kind}_{args.panel}_n{val_n}_seed{args.panel_seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
