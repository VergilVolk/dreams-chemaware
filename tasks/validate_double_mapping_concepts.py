"""Validate embedding -> spectral concept -> peak evidence with causal deletion.

The probe is trained on frozen global embeddings.  For held-out molecules, the
script locates peaks that instantiate a decoded rule, deletes them, and asks
whether the corresponding probe logit falls more than after matched control
peak deletion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from e1_checkpoint_io import official_head_state, torch_load_compat
from train_causal_chemmask_head import DEFAULT_OFFICIAL, DEFAULT_RAW
from train_e1_identity import load_base_model, preprocess_spectrum


ROOT = Path(__file__).resolve().parent.parent


def test_molecules(ik14: np.ndarray, seed: int) -> set[str]:
    molecules = np.unique(ik14)
    rng = np.random.default_rng(seed)
    molecules = molecules[rng.permutation(len(molecules))]
    return set(molecules[int(round(0.85 * len(molecules))):].tolist())


def supporting_indices(mz: np.ndarray, rule: dict, tolerance: float) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(mz) & (mz > 0))
    values = mz[valid]
    kind = rule.get("match_type")
    value = rule.get("value")
    if kind == "peak_mz":
        return valid[np.abs(values - float(value)) < tolerance]
    if kind in {"mass_diff", "mass_range"}:
        differences = np.abs(values[:, None] - values[None, :])
        if kind == "mass_diff":
            matched = np.abs(differences - float(value)) < tolerance
        else:
            low, high = map(float, value)
            matched = (differences >= low) & (differences <= high)
        np.fill_diagonal(matched, False)
        involved = np.flatnonzero(matched.any(axis=0) | matched.any(axis=1))
        return valid[involved]
    return np.empty(0, dtype=np.int64)


def choose_control(
    mz: np.ndarray, intensity: np.ndarray, support: np.ndarray, count: int,
) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(mz) & (mz > 0) & (intensity > 0))
    candidates = np.setdiff1d(valid, support, assume_unique=False)
    if len(candidates) < count:
        return np.empty(0, dtype=np.int64)
    target_log_i = np.sort(np.log1p(intensity[support]))
    chosen = []
    available = candidates.tolist()
    mz_scale = max(float(np.ptp(mz[valid])), 1.0)
    target_mz = np.sort(mz[support])
    for position in range(count):
        reference_i = target_log_i[min(position, len(target_log_i) - 1)]
        reference_mz = target_mz[min(position, len(target_mz) - 1)]
        scores = [
            abs(float(np.log1p(intensity[index])) - reference_i)
            + 0.25 * abs(float(mz[index]) - reference_mz) / mz_scale
            for index in available
        ]
        best = int(np.argmin(scores))
        chosen.append(available.pop(best))
    return np.asarray(chosen, dtype=np.int64)


def bootstrap_ci(frame: pd.DataFrame, column: str, seed: int, draws: int = 2000) -> list[float]:
    per_molecule = frame.groupby("ik14")[column].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    values = [
        float(rng.choice(per_molecule, size=len(per_molecule), replace=True).mean())
        for _ in range(draws)
    ]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--labels", type=Path,
        default=ROOT / "data/validation/double_mapping/spectrum_rule_labels.npz",
    )
    parser.add_argument(
        "--probe-dir", type=Path,
        default=ROOT / "data/validation/double_mapping/frozen_concept_probe",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/double_mapping/causal_concept_mapping",
    )
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--raw-checkpoint", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--max-concepts", type=int, default=6)
    parser.add_argument("--spectra-per-concept", type=int, default=20)
    parser.add_argument("--max-delete-peaks", type=int, default=8)
    parser.add_argument("--min-test-auprc", type=float, default=0.60)
    parser.add_argument("--min-lift", type=float, default=2.0)
    parser.add_argument("--min-test-positives", type=int, default=30)
    parser.add_argument("--max-test-prevalence", type=float, default=0.30)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)

    label_cache = np.load(args.labels, allow_pickle=False)
    labels = label_cache["labels"]
    ik14 = label_cache["ik14"].astype(str)
    hdf_rows = label_cache["hdf5_row"].astype(np.int64)
    test_set = test_molecules(ik14, args.seed)
    test_mask = np.asarray([molecule in test_set for molecule in ik14])
    probe = torch_load_compat(args.probe_dir / "concept_probe.pt", map_location="cpu")
    rule_indices = probe["rule_indices"].numpy().astype(np.int64)
    metrics = pd.read_csv(args.probe_dir / "per_rule_metrics.csv")
    metrics["test_positives"] = metrics["test_positive_spectra"].astype(int)
    rules = json.loads(
        (ROOT / "dreams/models/chem_aware/chem_rules_data.json").read_text(encoding="utf-8")
    )["rules"]
    metrics["mass_pattern_key"] = metrics["rule_index"].map(
        lambda index: (
            rules[int(index)]["match_type"],
            json.dumps(rules[int(index)]["value"], sort_keys=True),
        )
    ).astype(str)
    selected = metrics.loc[
        metrics["test_auprc"].ge(args.min_test_auprc)
        & metrics["auprc_lift"].ge(args.min_lift)
        & metrics["test_positives"].ge(args.min_test_positives)
        & metrics["test_prevalence"].le(args.max_test_prevalence)
    ].sort_values(["test_auprc", "auprc_lift"], ascending=False).drop_duplicates(
        subset=["mass_pattern_key"], keep="first"
    ).head(args.max_concepts)
    if selected.empty:
        raise RuntimeError("No concepts met the causal panel criteria")

    if args.dry_run:
        payload = {
            "status": "double_mapping_causal_panel_ready",
            "concepts": selected[[
                "rule_name", "category", "rule_index", "test_auprc",
                "test_prevalence", "auprc_lift", "test_positives",
            ]].to_dict(orient="records"),
            "spectra_per_concept": args.spectra_per_concept,
            "planned_forward_spectra": int(
                3 * args.spectra_per_concept * len(selected)
            ),
            "intervention": "clean vs supporting-peak deletion vs matched-control deletion",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    model, initialization = load_base_model(
        args.official_checkpoint, args.raw_checkpoint, device, args.n_highest_peaks,
    )
    if args.head_checkpoint:
        package = torch_load_compat(args.head_checkpoint, map_location="cpu")
        model.head.load_state_dict(official_head_state(package), strict=True)
        initialization = f"{initialization}+{args.head_checkpoint.name}"
    model.eval()
    probe_layer = torch.nn.Linear(
        int(probe["embedding_mean"].numel()), len(rule_indices), bias=True,
    )
    probe_layer.load_state_dict(probe["state_dict"], strict=True)
    probe_layer = probe_layer.to(device).eval()
    embedding_mean = probe["embedding_mean"].to(device)
    embedding_std = probe["embedding_std"].to(device)

    rows = []
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.data, "r") as handle:
        for concept in selected.itertuples(index=False):
            rule_index = int(concept.rule_index)
            output_index = int(concept.probe_output_index)
            rule = rules[rule_index]
            positive_indices = np.flatnonzero(test_mask & (labels[:, rule_index] > 0))
            positive_indices = positive_indices[rng.permutation(len(positive_indices))]
            examples = []
            for cache_index in positive_indices:
                hdf_row = int(hdf_rows[cache_index])
                raw = np.asarray(handle["spectrum"][hdf_row], dtype=np.float32)
                precursor = float(handle["precursor_mz"][hdf_row])
                support = supporting_indices(raw[0], rule, args.tolerance)
                if len(support) == 0:
                    continue
                support = support[np.argsort(raw[1, support])[-args.max_delete_peaks:]]
                control = choose_control(raw[0], raw[1], support, len(support))
                if len(control) != len(support):
                    continue
                targeted = raw.copy()
                targeted[:, support] = 0
                control_raw = raw.copy()
                control_raw[:, control] = 0
                examples.append((cache_index, hdf_row, raw, targeted, control_raw, precursor, support, control))
                if len(examples) >= args.spectra_per_concept:
                    break
            for start in range(0, len(examples), args.batch_size):
                batch_examples = examples[start:start + args.batch_size]
                tensors = []
                for example in batch_examples:
                    for raw_variant in example[2:5]:
                        tensors.append(preprocess_spectrum(raw_variant, example[5], args.n_highest_peaks))
                batch = torch.stack(tensors).to(device)
                with torch.inference_mode():
                    embedding = model(batch)
                    logits = probe_layer((embedding - embedding_mean) / embedding_std)
                for offset, example in enumerate(batch_examples):
                    clean, targeted_embedding, control_embedding = embedding[3 * offset:3 * offset + 3]
                    clean_logit, target_logit, control_logit = logits[
                        3 * offset:3 * offset + 3, output_index
                    ].cpu().numpy().tolist()
                    cache_index, hdf_row, raw, _, _, precursor, support, control = example
                    rows.append({
                        "rule_index": rule_index,
                        "probe_output_index": output_index,
                        "rule_name": rule["name"],
                        "category": rule["category"],
                        "ik14": ik14[cache_index],
                        "hdf5_row": hdf_row,
                        "support_peak_count": len(support),
                        "support_peak_mz": ";".join(f"{raw[0, i]:.5f}" for i in support),
                        "control_peak_mz": ";".join(f"{raw[0, i]:.5f}" for i in control),
                        "clean_logit": clean_logit,
                        "target_logit": target_logit,
                        "control_logit": control_logit,
                        "target_logit_drop": clean_logit - target_logit,
                        "control_logit_drop": clean_logit - control_logit,
                        "selective_concept_drop": control_logit - target_logit,
                        "target_embedding_shift": 1.0 - float(F.cosine_similarity(clean, targeted_embedding, dim=0)),
                        "control_embedding_shift": 1.0 - float(F.cosine_similarity(clean, control_embedding, dim=0)),
                    })
    detail = pd.DataFrame(rows)
    detail.to_csv(args.output_dir / "paired_concept_peak_interventions.csv", index=False)
    summaries = []
    for rule_name, group in detail.groupby("rule_name", sort=True):
        ci = bootstrap_ci(group, "selective_concept_drop", args.seed + len(summaries))
        summaries.append({
            "rule_name": rule_name,
            "category": group["category"].iloc[0],
            "spectra": int(len(group)),
            "molecules": int(group["ik14"].nunique()),
            "selective_concept_drop_mean": float(group["selective_concept_drop"].mean()),
            "selective_concept_drop_molecule_bootstrap_ci95": ci,
            "selective_concept_drop_positive_fraction": float((group["selective_concept_drop"] > 0).mean()),
            "target_embedding_shift_mean": float(group["target_embedding_shift"].mean()),
            "control_embedding_shift_mean": float(group["control_embedding_shift"].mean()),
            "closed_loop_candidate": bool(ci[0] > 0),
        })
    report = {
        "status": "double_mapping_causal_pilot_complete",
        "initialization": initialization,
        "mapping_1": "frozen global embedding -> held-out spectrum-level concept logit",
        "mapping_2": "concept mass pattern -> supporting peak indices",
        "faithfulness_test": "supporting-peak deletion vs matched control deletion",
        "concepts_screened": int(len(selected)),
        "concepts_with_interventions": int(len(summaries)),
        "closed_loop_candidates": int(sum(row["closed_loop_candidate"] for row in summaries)),
        "concepts": summaries,
        "claim_limit": (
            "A positive intervention validates score faithfulness for an observed mass pattern; "
            "it does not uniquely identify a fragment structure or mechanism."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
