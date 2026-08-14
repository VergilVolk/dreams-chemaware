"""Localize peak evidence in queries changed by the gated reranker.

The analysis compares the actual identity molecule and the competing molecule
for every DreaMS/reranker disagreement inside the frozen low-confidence gate.
It is descriptive localization, not causal attribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from attribute_large_failure_peaks import load_rules
from audit_e0_observability_residual import greedy_matches, peaks
from build_frozen_panel_pair_features import peak_masks, read_spectra
from train_frozen_panel_reranker import RAW_FEATURES
from train_pairwise_delta_reranker import augment_features, fit_ranker, score


def candidate_choice(group: pd.DataFrame, score_column: str) -> tuple[pd.Series | None, float]:
    best_per_molecule = (
        group.sort_values(score_column, ascending=False)
        .drop_duplicates("candidate_ik14")
        .sort_values(score_column, ascending=False)
    )
    if len(best_per_molecule) < 2:
        return None, float("nan")
    best = best_per_molecule.iloc[0]
    margin = float(best_per_molecule.iloc[0][score_column] - best_per_molecule.iloc[1][score_column])
    return best, margin


def matched_peak_summary(
    query_index: int, identity_index: int, confounder_index: int,
    spectra: np.ndarray, precursor: np.ndarray, tokens: np.ndarray,
    token_mz: np.ndarray, valid: np.ndarray, panel_ids: list[str], nl_values: np.ndarray, tolerance: float,
) -> tuple[dict, list[dict]]:
    q_mz, q_intensity = peaks(spectra[query_index])
    i_mz, _ = peaks(spectra[identity_index])
    c_mz, _ = peaks(spectra[confounder_index])
    qi = greedy_matches(q_mz, i_mz, tolerance)
    qc = greedy_matches(q_mz, c_mz, tolerance)
    identity_match = {a: b for a, b in qi}
    confounder_match = {a: b for a, b in qc}
    total_intensity = max(float(q_intensity.sum()), 1e-12)
    masks = peak_masks(q_mz, q_intensity, precursor[query_index], panel_ids, nl_values, tolerance)
    categories = {
        "identity_only": sorted(set(identity_match) - set(confounder_match)),
        "confounder_only": sorted(set(confounder_match) - set(identity_match)),
        "shared": sorted(set(identity_match) & set(confounder_match)),
        "neither": sorted(set(range(len(q_mz))) - set(identity_match) - set(confounder_match)),
    }
    summary: dict[str, float] = {}
    details: list[dict] = []
    def token_at(spectrum_index: int, target_mz: float) -> np.ndarray | None:
        positions = np.flatnonzero(valid[spectrum_index])
        values = token_mz[spectrum_index, positions]
        nearest = int(np.argmin(np.abs(values - target_mz)))
        if abs(float(values[nearest]) - target_mz) > tolerance:
            return None
        return tokens[spectrum_index, positions[nearest]].astype(np.float32)
    for category, indices in categories.items():
        summary[f"{category}_count"] = len(indices)
        summary[f"{category}_intensity_fraction"] = float(q_intensity[indices].sum() / total_intensity) if indices else 0.0
    for panel_id in panel_ids:
        safe = panel_id.replace("::", "__").replace("%", "pct").replace("-", "_").replace(".", "p")
        mask = masks[panel_id]
        for category in ("identity_only", "confounder_only", "shared"):
            selected = [index for index in categories[category] if mask[index]]
            summary[f"{safe}_{category}_count"] = len(selected)
            summary[f"{safe}_{category}_intensity_fraction"] = (
                float(q_intensity[selected].sum() / total_intensity) if selected else 0.0
            )
    for q_peak in range(len(q_mz)):
        in_identity = q_peak in identity_match
        in_confounder = q_peak in confounder_match
        category = (
            "shared" if in_identity and in_confounder else
            "identity_only" if in_identity else
            "confounder_only" if in_confounder else "neither"
        )
        i_cos = np.nan
        c_cos = np.nan
        q_token = token_at(query_index, float(q_mz[q_peak]))
        if in_identity:
            i_peak = identity_match[q_peak]
            i_token = token_at(identity_index, float(i_mz[i_peak]))
            if q_token is not None and i_token is not None:
                i_cos = float(q_token @ i_token)
        if in_confounder:
            c_peak = confounder_match[q_peak]
            c_token = token_at(confounder_index, float(c_mz[c_peak]))
            if q_token is not None and c_token is not None:
                c_cos = float(q_token @ c_token)
        details.append({
            "query_peak_index": q_peak, "mz": float(q_mz[q_peak]),
            "neutral_loss": float(precursor[query_index] - q_mz[q_peak]),
            "intensity_fraction": float(q_intensity[q_peak] / total_intensity),
            "evidence_class": category,
            "identity_token_cosine": i_cos, "confounder_token_cosine": c_cos,
            "panel_features": "|".join(panel for panel, mask in masks.items() if mask[q_peak]),
        })
    return summary, details


def summarize_cases(frame: pd.DataFrame) -> dict:
    output = {"cases": int(len(frame))}
    for column in [
        "identity_only_count", "confounder_only_count", "shared_count",
        "identity_only_intensity_fraction", "confounder_only_intensity_fraction",
        "shared_intensity_fraction",
    ]:
        output[f"median_{column}"] = float(frame[column].median()) if len(frame) else None
    output["median_identity_minus_confounder_intensity"] = (
        float((frame["identity_only_intensity_fraction"] - frame["confounder_only_intensity_fraction"]).median())
        if len(frame) else None
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directional-dir", type=Path, default=Path("data/validation/directional_panel_candidate_features"))
    parser.add_argument("--pair-dir", type=Path, default=Path("data/validation/large_observability_residual_audit"))
    parser.add_argument("--token-feature-dir", type=Path, default=Path("data/validation/peak_token_pair_features"))
    parser.add_argument("--token-root", type=Path, default=Path("data/validation/official_peak_tokens/confirmation"))
    parser.add_argument("--manifest", type=Path, default=Path("data/validation/large_observability_embeddings_confirmation/manifest.csv"))
    parser.add_argument("--data", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--panel", type=Path, default=Path("data/validation/large_failure_peak_evidence_strata/frozen_test_panel.csv"))
    parser.add_argument("--rules", type=Path, default=Path("dreams/models/chem_aware/chem_rules_data.json"))
    parser.add_argument("--gate-report", type=Path, default=Path("data/validation/confidence_gated_reranker/report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/gated_reranker_peak_localization"))
    parser.add_argument("--hard-k", type=int, default=5)
    parser.add_argument("--c-value", type=float, default=0.01)
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    discovery_raw = pd.read_csv(args.directional_dir / "discovery_directional_features.csv")
    confirmation_raw = pd.read_csv(args.directional_dir / "confirmation_directional_features.csv")
    discovery, burden, _, _ = augment_features(discovery_raw, args.pair_dir, args.token_feature_dir, "discovery")
    confirmation, _, _, _ = augment_features(confirmation_raw, args.pair_dir, args.token_feature_dir, "confirmation")
    features = ["dreams_similarity"] + RAW_FEATURES + burden
    scaler, model, _ = fit_ranker(discovery, features, args.hard_k, args.c_value)
    confirmation = score(confirmation, features, scaler, model)
    gate_report = json.loads(args.gate_report.read_text(encoding="utf-8"))
    threshold = float(gate_report["models"]["raw_panel"]["selected_threshold"])

    manifest = pd.read_csv(args.manifest)
    with h5py.File(args.data, "r") as handle:
        spectra = read_spectra(handle, manifest["hdf5_row"].to_numpy(np.int64))
    precursor = manifest["precursor_mz"].to_numpy(float)
    tokens = np.load(args.token_root / "peak_tokens_f16.npy", mmap_mode="r")
    token_mz = np.load(args.token_root / "peak_mz.npy", mmap_mode="r")
    valid = np.load(args.token_root / "peak_valid.npy", mmap_mode="r")
    panel_ids = pd.read_csv(args.panel)["feature_id"].tolist()
    rules = load_rules(args.rules)
    nl_values = np.asarray(sorted({float(rule["value"]) for rule in rules if rule["category"] == "NL"}), float)

    case_rows, peak_rows = [], []
    for query, group in confirmation.groupby("query", sort=False):
        dreams_choice, confidence = candidate_choice(group, "dreams_similarity")
        if dreams_choice is None:
            continue
        if confidence > threshold:
            continue
        reranker_choice, _ = candidate_choice(group, "score")
        if dreams_choice.candidate_ik14 == reranker_choice.candidate_ik14:
            continue
        truth = str(group.iloc[0].query_ik14)
        dreams_correct = str(dreams_choice.candidate_ik14) == truth
        reranker_correct = str(reranker_choice.candidate_ik14) == truth
        transition = (
            "fixed" if (not dreams_correct and reranker_correct) else
            "broken" if (dreams_correct and not reranker_correct) else "wrong_to_wrong"
        )
        if reranker_correct:
            identity, confounder = int(reranker_choice.candidate), int(dreams_choice.candidate)
        elif dreams_correct:
            identity, confounder = int(dreams_choice.candidate), int(reranker_choice.candidate)
        else:
            positives = group.loc[group["label"] == 1]
            if positives.empty:
                continue
            identity = int(positives.loc[positives["score"].idxmax()].candidate)
            confounder = int(reranker_choice.candidate)
        summary, peaks_out = matched_peak_summary(
            int(query), identity, confounder, spectra, precursor, tokens, token_mz, valid,
            panel_ids, nl_values, args.tolerance,
        )
        base = {
            "query_index": int(query), "ik14": truth, "formula": str(group.iloc[0].formula),
            "transition": transition, "dreams_choice_ik14": str(dreams_choice.candidate_ik14),
            "reranker_choice_ik14": str(reranker_choice.candidate_ik14),
            "identity_candidate_index": identity, "confounder_candidate_index": confounder,
            "dreams_confidence_margin": confidence,
        }
        case_rows.append(base | summary)
        peak_rows.extend([base | row for row in peaks_out])
    cases = pd.DataFrame(case_rows)
    peak_table = pd.DataFrame(peak_rows)
    cases.to_csv(args.output_dir / "changed_query_peak_summary.csv", index=False)
    peak_table.to_csv(args.output_dir / "changed_query_peak_table.csv", index=False)
    report = {
        "status": "gated_reranker_peak_localization",
        "gate_threshold": threshold,
        "changed_queries": int(len(cases)),
        "transitions": {name: summarize_cases(group) for name, group in cases.groupby("transition")},
        "claim_limit": "descriptive peak localization; causal use in finetuning requires targeted masking validation",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
