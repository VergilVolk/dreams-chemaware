#!/usr/bin/env python
"""Paired official-vs-ChemAware evaluation on the sealed MoNA polarity panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_mona_transfer_core import iter_mgf_records, validate_panel  # noqa: E402
from chemaware_v2_triplet_eval_core import binary_auc, paired_summary  # noqa: E402
from evaluate_chemaware_v2_direct_triplet import load_model  # noqa: E402
from train_e1_identity import preprocess_spectrum  # noqa: E402


TOP_K = (1, 3, 5, 10, 20, 50)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--positive-mgf", type=Path, default=ROOT / "data/models/mona_pos_full.mgf")
    parser.add_argument("--negative-mgf", type=Path, default=ROOT / "data/models/mona_neg_full.mgf")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="Named checkpoint as NAME=PATH; official must be first.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not re.fullmatch(r"[A-Za-z0-9_.-]+", name.strip()):
        raise ValueError(f"invalid checkpoint name: {name!r}")
    path = Path(raw_path)
    return name.strip(), path if path.is_absolute() else ROOT / path


def load_panel(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        panel = {key: np.asarray(loaded[key]) for key in loaded.files}
    validate_panel(panel)
    return panel


def required_rows(panel: dict[str, np.ndarray]) -> np.ndarray:
    return np.unique(np.concatenate((panel["query_row"], panel["candidate_row"])).astype(np.int64))


def read_selected_spectra(path: Path, rows: np.ndarray) -> dict[int, tuple[np.ndarray, float]]:
    required = set(map(int, rows))
    selected: dict[int, tuple[np.ndarray, float]] = {}
    for record in iter_mgf_records(path, include_peaks=True, compute_signature=False):
        row = int(record["row"])
        if row in required:
            selected[row] = (
                np.asarray(record["peaks"], dtype=np.float32),
                float(record["precursor_mz"]),
            )
    missing = sorted(required - set(selected))
    if missing:
        raise RuntimeError(f"MGF rows absent from source {path}: {missing[:20]}")
    return selected


@torch.no_grad()
def encode_rows(
    model, rows: np.ndarray, spectra: dict[int, tuple[np.ndarray, float]],
    device: torch.device, batch_size: int, n_highest_peaks: int,
) -> np.ndarray:
    output = []
    dtype = next(model.parameters()).dtype
    for left in range(0, len(rows), batch_size):
        batch_rows = rows[left:left + batch_size]
        batch = torch.stack([
            preprocess_spectrum(spectra[int(row)][0], spectra[int(row)][1], n_highest_peaks)
            for row in batch_rows
        ]).to(device=device, dtype=dtype)
        output.append(model(batch).float().cpu().numpy())
    encoded = np.concatenate(output).astype(np.float32, copy=False)
    norms = np.linalg.norm(encoded, axis=1)
    if not np.all(np.isfinite(encoded)) or np.max(np.abs(norms - 1.0)) > 2e-4:
        raise RuntimeError("MoNA embeddings are non-finite or not normalized")
    return encoded


def evaluate_panel(
    panel: dict[str, np.ndarray], rows: np.ndarray, encoded: np.ndarray,
) -> dict[str, np.ndarray]:
    position = {int(row): index for index, row in enumerate(rows)}
    n_queries = len(panel["query_row"])
    ranks = np.empty(n_queries, dtype=np.int32)
    positive = np.empty(n_queries, dtype=np.float32)
    hardest_negative = np.empty(n_queries, dtype=np.float32)
    macro_auc = np.empty(n_queries, dtype=np.float64)
    flat_scores: list[np.ndarray] = []
    flat_labels: list[np.ndarray] = []
    for query in range(n_queries):
        query_embedding = encoded[position[int(panel["query_row"][query])]]
        molecule_left, molecule_right = map(int, panel["query_ptr"][query:query + 2])
        spectrum_left = int(panel["molecule_ptr"][molecule_left])
        spectrum_right = int(panel["molecule_ptr"][molecule_right])
        candidate_rows = panel["candidate_row"][spectrum_left:spectrum_right]
        candidate_position = np.asarray([position[int(row)] for row in candidate_rows])
        spectrum_scores = encoded[candidate_position] @ query_embedding
        local_ptr = panel["molecule_ptr"][molecule_left:molecule_right + 1] - spectrum_left
        molecule_scores = np.maximum.reduceat(spectrum_scores, local_ptr[:-1])
        positive[query] = molecule_scores[0]
        negatives = np.asarray(molecule_scores[1:], dtype=np.float32)
        hardest_negative[query] = np.max(negatives)
        ranks[query] = 1 + int(np.sum(negatives >= molecule_scores[0]))
        macro_auc[query] = (
            np.sum(molecule_scores[0] > negatives)
            + 0.5 * np.sum(molecule_scores[0] == negatives)
        ) / len(negatives)
        flat_scores.append(np.asarray(molecule_scores, dtype=np.float32))
        flat_labels.append(np.r_[1, np.zeros(len(negatives), dtype=np.int8)])
    return {
        "rank": ranks,
        "positive": positive,
        "hardest_negative": hardest_negative,
        "macro_auc_by_query": macro_auc,
        "flat_scores": np.concatenate(flat_scores),
        "flat_labels": np.concatenate(flat_labels),
    }


def summarize(parts: list[dict[str, np.ndarray]]) -> dict[str, float | int]:
    ranks = np.concatenate([part["rank"] for part in parts])
    positive = np.concatenate([part["positive"] for part in parts])
    negative = np.concatenate([part["hardest_negative"] for part in parts])
    flat_scores = np.concatenate([part["flat_scores"] for part in parts])
    flat_labels = np.concatenate([part["flat_labels"] for part in parts]).astype(bool)
    macro = np.concatenate([part["macro_auc_by_query"] for part in parts])
    result: dict[str, float | int] = {"queries": int(len(ranks))}
    result.update({f"recall{k}": float(np.mean(ranks <= k)) for k in TOP_K})
    result.update({
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_positive_margin": float(np.mean(positive - negative)),
        "micro_auc": binary_auc(flat_labels, flat_scores),
        "macro_auc": float(np.mean(macro)),
    })
    return result


def paired_metrics(
    old_parts: list[dict[str, np.ndarray]], new_parts: list[dict[str, np.ndarray]],
    clusters: np.ndarray, draws: int, seed: int,
) -> dict[str, object]:
    old = np.concatenate([part["rank"] for part in old_parts])
    new = np.concatenate([part["rank"] for part in new_parts])
    result: dict[str, object] = paired_summary(old, new, clusters, draws, seed)
    for k in TOP_K:
        old_hit, new_hit = old <= k, new <= k
        result[f"delta_recall{k}"] = float(np.mean(new_hit) - np.mean(old_hit))
        result[f"corrected_at_{k}"] = int(np.sum(~old_hit & new_hit))
        result[f"introduced_at_{k}"] = int(np.sum(old_hit & ~new_hit))
    return result


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite evaluation: {args.output}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    root_report = json.loads((args.panel_dir / "report.json").read_text(encoding="utf-8"))
    if root_report.get("status") != "CHEMAWARE_MONA_POLARITY_TRANSFER_PANEL_SEALED":
        raise RuntimeError("MoNA polarity panel is not sealed")
    if not root_report.get("formal") or not all(root_report.get("gates", {}).values()):
        raise RuntimeError("MoNA polarity panel did not pass its construction gates")

    mgf_by_polarity = {"positive": args.positive_mgf, "negative": args.negative_mgf}
    panels: dict[str, dict[str, np.ndarray]] = {}
    rows_by_polarity: dict[str, np.ndarray] = {}
    spectra_by_polarity: dict[str, dict[int, tuple[np.ndarray, float]]] = {}
    for polarity, mgf in mgf_by_polarity.items():
        panel_path = args.panel_dir / polarity / "panel.npz"
        expected_hash = root_report["provenance"]["sources"][polarity]["panel_sha256"]
        if sha256_file(panel_path) != expected_hash:
            raise RuntimeError(f"{polarity} panel hash changed after sealing")
        if sha256_file(mgf) != root_report["provenance"]["sources"][polarity]["mgf_sha256"]:
            raise RuntimeError(f"{polarity} MGF changed after panel construction")
        panels[polarity] = load_panel(panel_path)
        rows_by_polarity[polarity] = required_rows(panels[polarity])
        print(
            f"Loading {polarity} selected spectra: {len(rows_by_polarity[polarity])} unique rows",
            flush=True,
        )
        spectra_by_polarity[polarity] = read_selected_spectra(
            mgf, rows_by_polarity[polarity],
        )

    clusters = np.concatenate([
        np.asarray([f"{polarity}:{formula}" for formula in panels[polarity]["query_formula"]])
        for polarity in ("positive", "negative")
    ])
    device = torch.device(args.device)
    results: list[dict[str, object]] = []
    evaluations: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    official_embeddings: dict[str, np.ndarray] = {}
    official_metrics: dict[str, float | int] | None = None
    embedding_cosines: dict[str, dict[str, float]] = {}
    checkpoint_hashes: dict[str, str] = {}

    for checkpoint_index, raw_checkpoint in enumerate(args.checkpoint):
        name, path = parse_checkpoint(raw_checkpoint)
        if checkpoint_index == 0 and name != "official":
            raise ValueError("first --checkpoint must be named official")
        if name in evaluations:
            raise ValueError(f"duplicate checkpoint name: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"Evaluating {name}: {path}", flush=True)
        model, kind = load_model(
            path, args.official_checkpoint, args.architecture_checkpoint,
            device, args.n_highest_peaks,
        )
        current: dict[str, dict[str, np.ndarray]] = {}
        cosines: dict[str, float] = {}
        for polarity in ("positive", "negative"):
            encoded = encode_rows(
                model, rows_by_polarity[polarity], spectra_by_polarity[polarity],
                device, args.batch_size, args.n_highest_peaks,
            )
            current[polarity] = evaluate_panel(
                panels[polarity], rows_by_polarity[polarity], encoded,
            )
            if name == "official":
                official_embeddings[polarity] = encoded
            else:
                cosines[polarity] = float(np.mean(np.sum(
                    encoded * official_embeddings[polarity], axis=1,
                )))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        evaluations[name] = current
        parts = [current[polarity] for polarity in ("positive", "negative")]
        metrics = summarize(parts)
        by_polarity = {
            polarity: summarize([current[polarity]])
            for polarity in ("positive", "negative")
        }
        row: dict[str, object] = {
            "name": name,
            "checkpoint": str(path.resolve()),
            "checkpoint_kind": kind,
            "metrics": metrics,
            "by_polarity": by_polarity,
        }
        if name == "official":
            official_metrics = metrics
        else:
            if official_metrics is None:
                raise AssertionError("official result missing")
            old_parts = [evaluations["official"][polarity] for polarity in ("positive", "negative")]
            row["paired_vs_official"] = paired_metrics(
                old_parts, parts, clusters, args.bootstrap_draws,
                args.seed + checkpoint_index,
            )
            row["paired_vs_official"].update({
                "delta_micro_auc": float(metrics["micro_auc"] - official_metrics["micro_auc"]),
                "delta_macro_auc": float(metrics["macro_auc"] - official_metrics["macro_auc"]),
                "delta_mean_positive_margin": float(
                    metrics["mean_positive_margin"] - official_metrics["mean_positive_margin"]
                ),
            })
            row["paired_by_polarity"] = {}
            for polarity_index, polarity in enumerate(("positive", "negative")):
                local_clusters = np.asarray([
                    f"{polarity}:{formula}" for formula in panels[polarity]["query_formula"]
                ])
                row["paired_by_polarity"][polarity] = paired_metrics(
                    [evaluations["official"][polarity]], [current[polarity]],
                    local_clusters, args.bootstrap_draws,
                    args.seed + 100 + checkpoint_index * 10 + polarity_index,
                )
            weighted_cosine = float(np.average(
                [cosines[polarity] for polarity in ("positive", "negative")],
                weights=[len(rows_by_polarity[polarity]) for polarity in ("positive", "negative")],
            ))
            row["mean_cosine_to_official_embedding"] = weighted_cosine
            row["mean_cosine_to_official_by_polarity"] = cosines
            embedding_cosines[name] = cosines
        results.append(row)
        checkpoint_hashes[name] = sha256_file(path)
        print(json.dumps(row, indent=2), flush=True)

    outcome_arrays: dict[str, np.ndarray] = {
        "polarity": np.concatenate([
            np.full(len(panels[polarity]["query_row"]), polarity)
            for polarity in ("positive", "negative")
        ]),
        "formula": clusters,
        "query_row": np.concatenate([
            panels[polarity]["query_row"] for polarity in ("positive", "negative")
        ]),
    }
    for name, current in evaluations.items():
        safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
        outcome_arrays[f"{safe}_rank"] = np.concatenate([
            current[polarity]["rank"] for polarity in ("positive", "negative")
        ])
        outcome_arrays[f"{safe}_positive"] = np.concatenate([
            current[polarity]["positive"] for polarity in ("positive", "negative")
        ])
        outcome_arrays[f"{safe}_hardest_negative"] = np.concatenate([
            current[polarity]["hardest_negative"] for polarity in ("positive", "negative")
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outcomes_path = args.output.with_name(args.output.stem + "_paired_outcomes.npz")
    if outcomes_path.exists():
        raise RuntimeError(f"refusing to overwrite paired outcomes: {outcomes_path}")
    np.savez_compressed(outcomes_path, **outcome_arrays)
    report = {
        "status": "CHEMAWARE_MONA_POLARITY_TRANSFER_EVALUATION_COMPLETE",
        "formal": True,
        "shared_query_reference_embedding": True,
        "candidate_features_used_at_inference": False,
        "chemical_rules_used_at_inference": False,
        "model_selection_on_mona": False,
        "queries": int(root_report["combined_queries"]),
        "results": results,
        "claim_limit": root_report["claim_limit"],
        "provenance": {
            "panel_report": str((args.panel_dir / "report.json").resolve()),
            "panel_report_sha256": sha256_file(args.panel_dir / "report.json"),
            "checkpoint_sha256": checkpoint_hashes,
            "paired_outcomes": str(outcomes_path.resolve()),
            "paired_outcomes_sha256": sha256_file(outcomes_path),
            "evaluator_sha256": sha256_file(Path(__file__)),
        },
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
