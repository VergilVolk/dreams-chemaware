"""Query-only peak masking stress test for the external balanced pilot.

``native_mask`` follows the DreaMS pretraining intervention: fragment peaks
are sampled proportional to intensity, precursor/base-peak tokens are kept,
and selected m/z values are replaced with the checkpoint mask value (-1).
``peak_dropout`` removes exactly the same selected tokens and is a paired
control.  Positive and negative library spectra always remain clean.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

import pilot_multilevel_factor_activations as multi  # noqa: E402
from e1_checkpoint_io import official_head_state  # noqa: E402
from pilot_paired_layer_cka import preprocess_spectrum  # noqa: E402


def stable_seed(*parts) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def select_tokens(clean: torch.Tensor, rate: float, seed: int) -> np.ndarray:
    values = clean.numpy()
    eligible = np.flatnonzero(
        (np.arange(len(values)) > 0)
        & (values[:, 0] > 0) & (values[:, 1] > 0) & (values[:, 1] < 1)
    )
    if len(eligible) <= 1:
        return np.asarray([], dtype=int)
    n_mask = min(max(2, round(len(eligible) * rate)), len(eligible) - 1)
    probability = values[eligible, 1].astype(float)
    probability /= probability.sum()
    return np.random.default_rng(seed).choice(eligible, n_mask, replace=False, p=probability)


def perturb(clean: torch.Tensor, selected: np.ndarray, mode: str) -> torch.Tensor:
    output = clean.clone()
    if mode == "native_mask":
        output[selected, 0] = -1.0
    elif mode == "peak_dropout":
        output[selected] = 0.0
    else:
        raise ValueError(mode)
    return output


def encode(model, head_weight, head_bias, tensors: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(tensors), batch_size=batch_size, shuffle=False)
    dtype = next(model.parameters()).dtype
    output = []
    with torch.inference_mode():
        for (batch,) in loader:
            precursor_token = model(batch.to(device=device, dtype=dtype), None)[:, 0]
            projected = F.linear(precursor_token, head_weight, head_bias)
            output.append(projected.float().cpu().numpy())
    values = np.concatenate(output)
    return values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12, None)


def score(query: np.ndarray, pair_id: int, view: int, units: list[dict], library: np.ndarray, protocol: str) -> dict:
    negatives = [int(value) for value in units[pair_id][protocol]]
    positive = float(query @ library[pair_id, 1 - view])
    matrix = np.einsum("nvd,d->nv", library[negatives], query)
    molecule_scores = matrix.max(axis=1)
    best = float(molecule_scores.max())
    return {
        "positive_similarity": positive,
        "best_negative_similarity": best,
        "margin": positive - best,
        "top1_correct": bool(positive > best),
        "pairwise_accuracy": float(np.mean(
            (positive > molecule_scores).astype(float)
            + 0.5 * (positive == molecule_scores).astype(float)
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--pilot-dir", type=Path, default=Path("data/validation/external_ring_balanced_pilot"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/validation/external_ring_balanced_embeddings"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--modes", nargs="+", choices=("native_mask", "peak_dropout"), default=["native_mask"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=20260812)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--official-checkpoint", type=Path, default=multi.DEFAULT_OFFICIAL)
    parser.add_argument("--raw-checkpoint", type=Path, default=multi.DEFAULT_RAW)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    units = json.loads((args.pilot_dir / f"{args.split}_manifest.json").read_text(encoding="utf-8"))["units"]
    source = np.load(args.pilot_dir / f"{args.split}_spectra.npz")
    spectra, precursor = source["spectra"], source["precursor_mz"]
    clean_library = np.load(args.embedding_dir / f"{args.split}_official.npy").astype(float)
    clean_library /= np.clip(np.linalg.norm(clean_library, axis=-1, keepdims=True), 1e-12, None)

    tensors, metadata = [], []
    for unit in units:
        if not unit["is_query_anchor"]:
            continue
        pair_id = int(unit["pair_id"])
        for view in (0, 1):
            clean = preprocess_spectrum(spectra[pair_id, view], float(precursor[pair_id, view]), args.n_highest_peaks)
            for rate in args.rates:
                for repeat in range(args.seeds):
                    selected = select_tokens(clean, rate, stable_seed(args.base_seed, args.split, unit["ik14"], view, rate, repeat))
                    for mode in args.modes:
                        tensors.append(perturb(clean, selected, mode))
                        metadata.append({
                            "pair_id": pair_id, "ik14": unit["ik14"], "query_view": view,
                            "ring_class": unit["ring_class"], "mode": mode,
                            "mask_rate": rate, "repeat": repeat,
                            "eligible_peak_count": int(np.sum(
                                (clean.numpy()[:, 0] > 0) & (clean.numpy()[:, 1] > 0)
                                & (clean.numpy()[:, 1] < 1)
                            )),
                            "removed_count": len(selected),
                            "removed_mz": "|".join(f"{float(clean[i, 0]):.5f}" for i in selected),
                        })
    raw_package = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official_package = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    model = multi.reconstruct_backbone(raw_package, multi.official_backbone_state(official_package), device)
    head = official_head_state(official_package)
    head_weight = head["weight"].to(device=device, dtype=next(model.parameters()).dtype)
    head_bias = head["bias"].to(device=device, dtype=next(model.parameters()).dtype)
    model.eval()
    encoded = encode(model, head_weight, head_bias, torch.stack(tensors), args.batch_size, device)
    del model, raw_package, official_package
    gc.collect()

    rows = []
    for vector, item in zip(encoded, metadata):
        pair_id, view = item["pair_id"], item["query_view"]
        row = dict(item)
        row["embedding_cosine_to_clean"] = float(vector @ clean_library[pair_id, view])
        for protocol in ("negative_pair_ids", "same_formula_negative_pair_ids"):
            if not units[pair_id][protocol]:
                continue
            result = score(vector, pair_id, view, units, clean_library, protocol)
            clean_result = score(clean_library[pair_id, view], pair_id, view, units, clean_library, protocol)
            rows.append(row | result | {
                "candidate_protocol": protocol,
                "clean_top1_correct": clean_result["top1_correct"],
                "clean_margin": clean_result["margin"],
                "margin_drop": clean_result["margin"] - result["margin"],
                "correct_to_wrong": bool(clean_result["top1_correct"] and not result["top1_correct"]),
                "wrong_to_correct": bool(not clean_result["top1_correct"] and result["top1_correct"]),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "perturbation_results.csv", index=False)
    grouped = frame.groupby(["candidate_protocol", "mode", "mask_rate", "ring_class"]).agg(
        rows=("ik14", "size"), molecules=("ik14", "nunique"),
        noisy_top1=("top1_correct", "mean"), clean_top1=("clean_top1_correct", "mean"),
        correct_to_wrong=("correct_to_wrong", "mean"), wrong_to_correct=("wrong_to_correct", "mean"),
        margin_drop=("margin_drop", "mean"), embedding_cosine=("embedding_cosine_to_clean", "mean"),
    ).reset_index()
    grouped.to_csv(args.output_dir / "summary.csv", index=False)
    report = {
        "status": "external_peak_mask_stress",
        "split": args.split, "rates": args.rates, "modes": args.modes,
        "seeds_per_query_view": args.seeds,
        "variants": len(metadata), "scored_rows": len(frame),
        "intervention": "Query-only intensity-weighted masking; clean positive and candidate library.",
        "embedding_definition": "L2-normalized official linear-head(backbone precursor token)",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
