"""Build a leakage-controlled peak-evidence finetuning manifest.

Reranker scores for discovery queries are formula-isolated out-of-fold scores.
The output contains no confirmation/test examples.  It records identity-only
and confounder-only query peaks for counterfactual peak-dropout objectives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from audit_e0_observability_residual import greedy_matches, peaks
from build_frozen_panel_pair_features import read_spectra
from train_confidence_gated_reranker import oof_score
from train_frozen_panel_reranker import RAW_FEATURES
from train_pairwise_delta_reranker import augment_features


def choice(group: pd.DataFrame, column: str, labels: set[int] | None = None) -> pd.Series | None:
    selected = group if labels is None else group.loc[group["label"].isin(labels)]
    if selected.empty:
        return None
    return selected.loc[selected[column].idxmax()]


def values(items: np.ndarray) -> str:
    return ";".join(f"{float(value):.5f}" for value in items)


def localize(query: int, identity: int, confounder: int, spectra: np.ndarray) -> dict[str, object]:
    q_mz, q_intensity = peaks(spectra[query])
    i_mz, _ = peaks(spectra[identity])
    c_mz, _ = peaks(spectra[confounder])
    identity_matches = {left for left, _ in greedy_matches(q_mz, i_mz, 0.02)}
    confounder_matches = {left for left, _ in greedy_matches(q_mz, c_mz, 0.02)}
    identity_only = np.asarray(sorted(identity_matches - confounder_matches), dtype=int)
    confounder_only = np.asarray(sorted(confounder_matches - identity_matches), dtype=int)
    shared = np.asarray(sorted(identity_matches & confounder_matches), dtype=int)
    scale = max(float(q_intensity.sum()), 1e-12)
    return {
        "identity_peak_count": len(identity_only),
        "identity_peak_mz": values(q_mz[identity_only]),
        "identity_peak_intensity_fraction": float(q_intensity[identity_only].sum() / scale),
        "confounder_peak_count": len(confounder_only),
        "confounder_peak_mz": values(q_mz[confounder_only]),
        "confounder_peak_intensity_fraction": float(q_intensity[confounder_only].sum() / scale),
        "shared_peak_count": len(shared),
        "shared_peak_intensity_fraction": float(q_intensity[shared].sum() / scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directional-dir", type=Path, default=Path("data/validation/directional_panel_candidate_features"))
    parser.add_argument("--pair-dir", type=Path, default=Path("data/validation/large_observability_residual_audit"))
    parser.add_argument("--token-dir", type=Path, default=Path("data/validation/peak_token_pair_features"))
    parser.add_argument("--manifest", type=Path, default=Path("data/validation/large_observability_embeddings_discovery/manifest.csv"))
    parser.add_argument("--data", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--gate-report", type=Path, default=Path("data/validation/confidence_gated_reranker/report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/e1/counterfactual_peak_finetune"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hard-k", type=int, default=5)
    parser.add_argument("--c-value", type=float, default=0.01)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.directional_dir / "discovery_directional_features.csv")
    frame, burden, _, _ = augment_features(raw, args.pair_dir, args.token_dir, "discovery")
    features = ["dreams_similarity"] + RAW_FEATURES + burden
    scored = oof_score(frame, features, args.folds, args.hard_k, args.c_value)
    threshold = float(json.loads(args.gate_report.read_text(encoding="utf-8"))["models"]["raw_panel"]["selected_threshold"])
    manifest = pd.read_csv(args.manifest)
    with h5py.File(args.data, "r") as handle:
        spectra = read_spectra(handle, manifest["hdf5_row"].to_numpy(np.int64))

    rows = []
    for query_index, group in scored.groupby("query", sort=False):
        molecules = group.sort_values("dreams_similarity", ascending=False).drop_duplicates("candidate_ik14")
        if len(molecules) < 2:
            continue
        confidence = float(molecules.iloc[0].dreams_similarity - molecules.iloc[1].dreams_similarity)
        if confidence > threshold:
            continue
        identity = choice(group, "dreams_similarity", {1})
        negative = choice(group, "dreams_similarity", {0})
        reranked = choice(group, "score")
        if identity is None or negative is None or reranked is None:
            continue
        truth = str(group.iloc[0].query_ik14)
        dreams_correct = str(molecules.iloc[0].candidate_ik14) == truth
        reranker_correct = str(reranked.candidate_ik14) == truth
        transition = (
            "fixed_oof" if (not dreams_correct and reranker_correct) else
            "broken_oof" if (dreams_correct and not reranker_correct) else
            "protected_correct" if dreams_correct else "residual_wrong"
        )
        localized = localize(int(query_index), int(identity.candidate), int(negative.candidate), spectra)
        query_row = manifest.loc[int(query_index)]
        identity_row = manifest.loc[int(identity.candidate)]
        confounder_row = manifest.loc[int(negative.candidate)]
        rows.append({
            "query_index": int(query_index), "query_hdf5_row": int(query_row.hdf5_row),
            "identity_index": int(identity.candidate), "identity_hdf5_row": int(identity_row.hdf5_row),
            "confounder_index": int(negative.candidate), "confounder_hdf5_row": int(confounder_row.hdf5_row),
            "ik14": truth, "confounder_ik14": str(negative.candidate_ik14),
            "formula": str(group.iloc[0].formula), "transition": transition,
            "dreams_confidence_margin": confidence,
            "dreams_pair_margin": float(identity.dreams_similarity - negative.dreams_similarity),
            "oof_reranker_pair_margin": float(identity.score - negative.score),
        } | localized)
    output = pd.DataFrame(rows)
    output.to_csv(args.output_dir / "discovery_oof_peak_evidence_manifest.csv", index=False)
    report = {
        "status": "counterfactual_peak_finetune_manifest",
        "source": "discovery only; reranker predictions formula-isolated OOF",
        "gate_threshold": threshold,
        "examples": len(output), "molecules": int(output["ik14"].nunique()),
        "formulas": int(output["formula"].nunique()),
        "transitions": output["transition"].value_counts().to_dict(),
        "examples_with_identity_peaks": int((output["identity_peak_count"] > 0).sum()),
        "examples_with_confounder_peaks": int((output["confounder_peak_count"] > 0).sum()),
        "recommended_objectives": {
            "identity_counterfactual": "removing identity-only peaks should lower positive-vs-negative margin",
            "confounder_counterfactual": "removing confounder-only peaks should raise positive-vs-negative margin",
            "protection": "protected_correct examples constrain clean retrieval and prevent over-correction",
        },
        "confirmation_and_test_usage": "none",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
