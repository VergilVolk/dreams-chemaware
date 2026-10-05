"""Evaluate optional legacy DreaMS NIST20 callback assets for checkpoints.

This script implements the public ``SpecRetrievalValidation`` callback:
load an already-built pair table, score embedding cosine, and compute pooled
AUROC.  The public repository does not include the pair-table generator for
the callback's ``*_50k_pairs_retrieval.pkl`` asset, so this script must not be
described as an exact reproduction of the paper's approximately 750k-pair
Figure 4b protocol.  Missing NIST20 assets are non-blocking for the project's
MassSpecGym retrieval evaluation.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from e1_checkpoint_io import torch_load_compat  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_r2_shared_encoder import forward_embeddings  # noqa: E402


NOISE_SHARED_STATUS = "noise_final_e4a_direct_shared_dreams_encoder"


def load_noise_inference_model(
    official_checkpoint: Path,
    architecture_checkpoint: Path,
    device: torch.device,
    n_highest_peaks: int,
    shared_checkpoint: Path | None = None,
):
    """Load only the shared noise encoder; do not import optional ChemAware code."""
    model, initialization = load_base_model(
        official_checkpoint, architecture_checkpoint, device, n_highest_peaks,
    )
    metadata = {
        "kind": "official_dreams",
        "checkpoint": str(official_checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(official_checkpoint),
        "initialization": initialization,
    }
    if shared_checkpoint is not None:
        package = torch_load_compat(shared_checkpoint, map_location="cpu")
        if package.get("status") != NOISE_SHARED_STATUS:
            raise RuntimeError(
                "NIST20 evaluator received a non-noise shared checkpoint: "
                f"{package.get('status')!r}"
            )
        if package.get("inference_clean_only") is not True:
            raise RuntimeError("noise checkpoint is not clean-spectrum-only at inference")
        if package.get("P2b_used") is not False:
            raise RuntimeError("noise checkpoint violates the P2b-free contract")
        state = package.get("model_state")
        if not isinstance(state, dict) or not state:
            raise RuntimeError("noise checkpoint has no model_state")
        model.load_state_dict(state, strict=True)
        metadata = {
            "kind": "noise_shared_clean_spectrum_encoder",
            "checkpoint": str(shared_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(shared_checkpoint),
            "status": package["status"],
            "training_seed": int(package.get("seed", -1)),
            "training_outer_fold": int(package.get("outer_fold", -1)),
            "inference_clean_only": True,
            "P2b_used": False,
        }
        del package
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, metadata


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectra", type=Path,
        default=ROOT / "data/NIST20/nist20_clean_spec_entropy_[M+H]+_retrieval.pkl",
    )
    parser.add_argument(
        "--pairs", type=Path,
        default=ROOT / "data/NIST20/nist20_clean_spec_entropy_[M+H]+_50k_pairs_retrieval.pkl",
    )
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--clean-control-checkpoint", type=Path, required=True)
    parser.add_argument("--random-control-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--architecture-checkpoint", type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def encode(model, spectra: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    output: list[np.ndarray] = []
    for left in range(0, len(spectra), batch_size):
        right = min(left + batch_size, len(spectra))
        values = forward_embeddings(
            model, spectra[left:right].to(device), False,
        ).float().cpu().numpy()
        if not np.all(np.isfinite(values)):
            raise RuntimeError("NIST20 encoding produced non-finite values")
        output.append(values)
        if right == len(spectra) or right % (batch_size * 20) == 0:
            print(f"[NIST20 encode] {right:,}/{len(spectra):,}", flush=True)
    return np.concatenate(output)


def pair_scores(validation, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lookup = validation.df["i"].to_dict()
    try:
        left = np.asarray(
            [lookup[int(value)] for value in validation.df_pairs["i"]], dtype=np.int64,
        )
        right = np.asarray(
            [lookup[int(value)] for value in validation.df_pairs["j"]], dtype=np.int64,
        )
    except KeyError as error:
        raise RuntimeError(f"NIST20 pair ledger references a missing spectrum: {error}") from error
    labels = validation.df_pairs["label"].to_numpy(np.int8)
    if set(map(int, np.unique(labels))) != {0, 1}:
        raise RuntimeError("NIST20 pair labels are not binary")
    scores = np.einsum("ij,ij->i", embeddings[left], embeddings[right])
    return labels, scores


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    return {
        "pairs": int(len(labels)),
        "positive_pairs": int(np.sum(labels == 1)),
        "negative_pairs": int(np.sum(labels == 0)),
        "pooled_pairwise_auroc": float(roc_auc_score(labels, scores)),
        "pooled_pairwise_auprc": float(average_precision_score(labels, scores)),
        "positive_similarity_mean": float(np.mean(positive)),
        "negative_similarity_mean": float(np.mean(negative)),
        "similarity_separation": float(np.mean(positive) - np.mean(negative)),
    }


def paired_bootstrap(
    labels: np.ndarray, official: np.ndarray, student: np.ndarray,
    resamples: int, seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        index = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[index]
        if len(np.unique(sampled_labels)) < 2:
            continue
        values.append(float(
            roc_auc_score(sampled_labels, student[index])
            - roc_auc_score(sampled_labels, official[index])
        ))
    if not values:
        raise RuntimeError("NIST20 pair bootstrap produced no valid resamples")
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "resampling_unit": "pair_row_descriptive_only",
    }


def publish(output: Path, report: dict) -> None:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite NIST20 evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".nist20_pair_", dir=output.parent))
    try:
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    args = arguments()
    checkpoints = {
        "candidate_boundary": args.candidate_checkpoint,
        "clean_duplicate": args.clean_control_checkpoint,
        "matched_random": args.random_control_checkpoint,
    }
    required_model = [
        *checkpoints.values(), args.official_checkpoint, args.architecture_checkpoint,
    ]
    if missing := [str(path) for path in required_model if not path.is_file()]:
        raise FileNotFoundError(missing)
    missing_assets = [str(path) for path in (args.spectra, args.pairs) if not path.is_file()]
    if missing_assets:
        report = {
            "status": "noise_shared_nist20_legacy_callback_not_run",
            "formal": False,
            "available": False,
            "optional_missing_assets": missing_assets,
            "exact_paper_replication": False,
            "checkpoint_sha256": {
                label: sha256_file(path) for label, path in checkpoints.items()
            },
            "claim_limit": (
                "Optional legacy callback assets are absent. This does not block "
                "the frozen MassSpecGym query/candidate evaluation."
            ),
        }
        publish(args.output_dir, report)
        print(json.dumps(report, indent=2))
        return

    device = torch.device(args.device)
    from dreams.utils.data import SpecRetrievalValidation

    official_model, official_meta = load_noise_inference_model(
        args.official_checkpoint, args.architecture_checkpoint, device,
        args.n_highest_peaks,
    )
    validation = SpecRetrievalValidation(
        args.spectra, args.pairs, official_model.backbone.spec_preproc.dformat,
        official_model.backbone.spec_preproc,
    )
    spectra = validation.get_data()["spec"]
    official_embeddings = encode(official_model, spectra, device, args.batch_size)
    labels, official_scores = pair_scores(validation, official_embeddings)
    del official_model, official_embeddings
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    official_result = metrics(labels, official_scores)
    model_results: dict[str, dict] = {"official": official_result}
    model_scores: dict[str, np.ndarray] = {"official": official_scores}
    model_meta: dict[str, dict] = {"official": official_meta}
    for label, checkpoint in checkpoints.items():
        candidate_model, candidate_meta = load_noise_inference_model(
            args.official_checkpoint, args.architecture_checkpoint, device,
            args.n_highest_peaks, checkpoint,
        )
        candidate_embeddings = encode(
            candidate_model, spectra, device, args.batch_size,
        )
        labels_again, scores = pair_scores(validation, candidate_embeddings)
        if not np.array_equal(labels, labels_again):
            raise RuntimeError(f"NIST20 label order changed for {label}")
        model_results[label] = metrics(labels, scores)
        model_scores[label] = scores
        model_meta[label] = candidate_meta
        del candidate_model, candidate_embeddings
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_keys = (
        "pooled_pairwise_auroc", "pooled_pairwise_auprc",
        "positive_similarity_mean", "negative_similarity_mean", "similarity_separation",
    )
    comparisons: dict[str, dict] = {}
    for reference in ("official", "clean_duplicate", "matched_random"):
        key = f"candidate_boundary_vs_{reference}"
        comparisons[key] = {
            "delta": {
                metric: float(
                    model_results["candidate_boundary"][metric]
                    - model_results[reference][metric]
                )
                for metric in metric_keys
            },
            "paired_pair_bootstrap_auroc_delta": paired_bootstrap(
                labels, model_scores[reference], model_scores["candidate_boundary"],
                args.bootstrap_resamples, args.seed + len(comparisons),
            ),
        }
    nist_gates = {}
    for reference in ("official", "clean_duplicate", "matched_random"):
        delta = comparisons[f"candidate_boundary_vs_{reference}"]["delta"]
        nist_gates[f"candidate_auroc_nonnegative_vs_{reference}"] = bool(
            delta["pooled_pairwise_auroc"] >= 0
        )
        nist_gates[f"candidate_auprc_nonnegative_vs_{reference}"] = bool(
            delta["pooled_pairwise_auprc"] >= 0
        )
    report = {
        "status": "noise_shared_nist20_legacy_callback_complete",
        "formal": True,
        "available": True,
        "protocol": "public SpecRetrievalValidation callback on supplied private pair ledger",
        "exact_paper_replication": False,
        "metric": "pooled_pairwise_auroc",
        "models": model_results,
        "comparisons": comparisons,
        "gates": nist_gates,
        "provenance": {
            "spectra_sha256": sha256_file(args.spectra),
            "pairs_sha256": sha256_file(args.pairs),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "checkpoint_sha256": {
                label: sha256_file(path) for label, path in checkpoints.items()
            },
            "models": model_meta,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Same supplied NIST20 callback pair ledger for official and student. "
            "This is not an exact Figure 4b reproduction because the public code does "
            "not construct the callback's 50k pair asset. Pair-row bootstrap is "
            "descriptive because spectrum reuse may induce dependence."
        ),
    }
    publish(args.output_dir, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
