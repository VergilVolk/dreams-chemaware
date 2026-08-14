"""Align Crosscoder features across seeds and audit decoder/activation stability."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from train_crosscoder_smoke import Crosscoder, load_activations  # noqa: E402
from pilot_paired_layer_cka import linear_cka  # noqa: E402


DEFAULT_RUNS = [
    ROOT / "data/validation/crosscoder_smoke_precursor_l7_seed42",
    ROOT / "data/validation/crosscoder_smoke_precursor_l7_seed43",
    ROOT / "data/validation/crosscoder_smoke_precursor_l7_seed44",
]
DEFAULT_CONFIRMATION = ROOT / "data/validation/multilevel_factor_confirm1000_qc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/crosscoder_smoke_precursor_l7_stability.json",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--minimum-support", type=int, default=10)
    return parser.parse_args()


def load_model(run_dir: Path) -> tuple[Crosscoder, dict, dict]:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    config = report["config"]
    try:
        package = torch.load(
            run_dir / "crosscoder.pt", map_location="cpu", weights_only=True
        )
    except TypeError:
        package = torch.load(run_dir / "crosscoder.pt", map_location="cpu")
    model = Crosscoder(
        input_dim=1024,
        hidden_dim=int(config["hidden_dim"]),
        top_k=int(config["top_k"]),
    )
    model.load_state_dict(package["state_dict"], strict=True)
    model.eval()
    normalization = package["normalization"]
    normalization = {
        key: value.numpy() if isinstance(value, torch.Tensor) else value
        for key, value in normalization.items()
    }
    return model, normalization, report


def decoder_directions(model: Crosscoder) -> np.ndarray:
    raw = model.decoder_raw.weight.detach().numpy().T
    official = model.decoder_official.weight.detach().numpy().T
    directions = np.concatenate([raw, official], axis=1)
    return directions / np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-12)


def external_activations(
    model: Crosscoder,
    normalization: dict,
    raw: np.ndarray,
    official: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    raw = (
        (raw - normalization["raw_mean"])
        / max(float(normalization["raw_rms"]), 1e-8)
    ).astype(np.float32)
    official = (
        (official - normalization["official_mean"])
        / max(float(normalization["official_rms"]), 1e-8)
    ).astype(np.float32)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(raw), batch_size):
            end = min(start + batch_size, len(raw))
            latent = model.encode(
                torch.from_numpy(raw[start:end]),
                torch.from_numpy(official[start:end]),
            )
            outputs.append(latent.numpy())
    return np.concatenate(outputs)


def feature_correlations(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    reference -= reference.mean(axis=0, keepdims=True)
    candidate -= candidate.mean(axis=0, keepdims=True)
    numerator = np.sum(reference * candidate, axis=0)
    denominator = np.linalg.norm(reference, axis=0) * np.linalg.norm(candidate, axis=0)
    return np.divide(
        numerator,
        denominator,
        out=np.full(reference.shape[1], np.nan),
        where=denominator > 0,
    )


def summarize(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"n": 0}
    return {
        "n": int(len(finite)),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p10": float(np.quantile(finite, 0.1)),
        "p90": float(np.quantile(finite, 0.9)),
        "fraction_ge_0_5": float(np.mean(finite >= 0.5)),
        "fraction_ge_0_7": float(np.mean(finite >= 0.7)),
        "fraction_ge_0_8": float(np.mean(finite >= 0.8)),
    }


def decoder_subspace_overlap(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Principal-angle overlap of decoder row spaces.

    For random 512-dimensional subspaces in 2048 dimensions, the expected
    mean squared cosine is approximately 0.25. Feature-wise alignment can be
    poor even when this subspace statistic is elevated.
    """
    q_reference, _ = np.linalg.qr(reference.T, mode="reduced")
    q_candidate, _ = np.linalg.qr(candidate.T, mode="reduced")
    singular_values = np.linalg.svd(
        q_reference.T @ q_candidate, compute_uv=False
    )
    return {
        "mean_squared_cosine": float(np.mean(np.square(singular_values))),
        "median_cosine": float(np.median(singular_values)),
        "p10_cosine": float(np.quantile(singular_values, 0.1)),
        "p90_cosine": float(np.quantile(singular_values, 0.9)),
        "random_subspace_reference": float(
            reference.shape[0] / reference.shape[1]
        ),
    }


def main() -> None:
    args = parse_args()
    if len(args.runs) < 2:
        raise ValueError("At least two Crosscoder runs are required")
    models, normalizations, reports = [], [], []
    for run in args.runs:
        model, normalization, report = load_model(run)
        models.append(model)
        normalizations.append(normalization)
        reports.append(report)
    configs = [report["config"] for report in reports]
    for config in configs:
        config.setdefault("token_type", "precursor")
    for key in ("layer", "hidden_dim", "top_k", "token_type"):
        if len({str(config[key]) for config in configs}) != 1:
            raise RuntimeError(f"Run configs differ for {key}")

    layer = int(configs[0]["layer"])
    token_type = configs[0]["token_type"]
    raw_external, official_external, _, _ = load_activations(
        args.confirmation, layer, token_type
    )
    directions = [decoder_directions(model) for model in models]
    activations = [
        external_activations(
            model, normalization, raw_external, official_external, args.batch_size
        )
        for model, normalization in zip(models, normalizations)
    ]

    reference_direction = directions[0]
    reference_activation = activations[0]
    reference_support = (reference_activation > 0).sum(axis=0)
    comparisons = []
    matched_by_run = []
    for run_index in range(1, len(args.runs)):
        similarity = reference_direction @ directions[run_index].T
        reference_indices, candidate_indices = linear_sum_assignment(-similarity)
        order = np.argsort(reference_indices)
        candidate_indices = candidate_indices[order]
        matched_similarity = similarity[np.arange(len(reference_indices)), candidate_indices]
        candidate_activation = activations[run_index][:, candidate_indices]
        activation_correlation = feature_correlations(
            reference_activation, candidate_activation
        )
        candidate_support = (candidate_activation > 0).sum(axis=0)
        supported = (
            (reference_support >= args.minimum_support)
            & (candidate_support >= args.minimum_support)
        )
        stable_screen = (
            supported
            & (matched_similarity >= 0.7)
            & (activation_correlation >= 0.5)
        )
        comparisons.append({
            "reference": str(args.runs[0].resolve()),
            "candidate": str(args.runs[run_index].resolve()),
            "decoder_cosine": summarize(matched_similarity),
            "decoder_subspace_overlap": decoder_subspace_overlap(
                reference_direction, directions[run_index]
            ),
            "latent_activation_cka": float(
                linear_cka(reference_activation, activations[run_index])
            ),
            "activation_correlation_supported": summarize(
                activation_correlation[supported]
            ),
            "supported_features": int(supported.sum()),
            "screen_stable_features": int(stable_screen.sum()),
            "screen_stable_fraction": float(stable_screen.mean()),
            "screen_definition": (
                f"support >= {args.minimum_support} in both runs, decoder cosine "
                ">= 0.7, activation correlation >= 0.5"
            ),
        })
        matched_by_run.append({
            "candidate_indices": candidate_indices,
            "decoder_cosine": matched_similarity,
            "activation_correlation": activation_correlation,
            "supported": supported,
            "stable_screen": stable_screen,
        })

    consensus = np.ones(len(reference_support), dtype=bool)
    for matched in matched_by_run:
        consensus &= matched["stable_screen"]
    consensus_indices = np.flatnonzero(consensus)
    output = {
        "status": "crosscoder_seed_stability_audit",
        "warning": (
            "Thresholds are conservative screening rules for a smoke test, not "
            "validated definitions of chemical factors."
        ),
        "runs": [str(path.resolve()) for path in args.runs],
        "confirmation": str(args.confirmation.resolve()),
        "layer": layer,
        "token_type": token_type,
        "hidden_dim": int(configs[0]["hidden_dim"]),
        "top_k": int(configs[0]["top_k"]),
        "external_reconstruction": [
            report["external_confirmation"] for report in reports
        ],
        "comparisons": comparisons,
        "consensus_stable_features": int(consensus.sum()),
        "consensus_stable_fraction": float(consensus.mean()),
        "consensus_reference_indices": consensus_indices.tolist(),
        "decision": (
            "Do not interpret individual factors yet. Use consensus features "
            "only as candidates for the next annotation pilot."
        ),
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Crosscoder seed stability")
    for comparison in comparisons:
        print(" ", Path(comparison["candidate"]).name)
        print("    decoder:", comparison["decoder_cosine"])
        print("    subspace:", comparison["decoder_subspace_overlap"])
        print("    latent CKA:", comparison["latent_activation_cka"])
        print("    activation:", comparison["activation_correlation_supported"])
        print(
            "    stable screen:", comparison["screen_stable_features"],
            f"/{configs[0]['hidden_dim']}"
        )
    print(
        "  consensus:", int(consensus.sum()), f"/{configs[0]['hidden_dim']}"
    )
    print("  saved:", args.output)


if __name__ == "__main__":
    main()
