"""Molecule-disjoint validation of the preliminary conflict-mining rule panel."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

import pilot_rule_noise_stress as pilot


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--audit-csv", type=Path, default=ROOT / "data/validation/e0_failure_audit/e0_query_audit.csv")
    parser.add_argument("--screen-dir", type=Path, default=ROOT / "data/validation/compact_rule_panel")
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / "data/validation/rule_noise_pilot")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/conflict_rule_validation")
    parser.add_argument("--n-per-class", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def one_per_molecule(rows: list[dict[str, str]], rng: np.random.Generator, limit: int) -> list[dict[str, str]]:
    shuffled = [rows[int(index)] for index in rng.permutation(len(rows))]
    selected = []
    seen = set()
    for row in shuffled:
        ik = row["query_ik14"]
        if ik in seen:
            continue
        seen.add(ik)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def direction_score(query: np.ndarray, true: np.ndarray, wrong: np.ndarray, indices: np.ndarray) -> tuple[float, int, int]:
    q, t, w = query[indices], true[indices], wrong[indices]
    true_only = int((q & t & ~w).sum())
    wrong_only = int((q & w & ~t).sum())
    score = (wrong_only - true_only) / max(1, wrong_only + true_only)
    return float(score), true_only, wrong_only


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    differences = positives[:, None] - negatives[None, :]
    return float((np.count_nonzero(differences > 0) + 0.5 * np.count_nonzero(differences == 0)) / differences.size)


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray, seed: int, n_bootstrap: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    values = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sampled = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        values[index] = auc(labels[sampled], scores[sampled])
    point = auc(labels, scores)
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = read_csv(args.audit_csv)
    pilot_selected = read_csv(args.pilot_dir / "selected_queries.csv")
    rule_audit = read_csv(args.screen_dir / "rule_level_audit.csv")
    manifest = json.loads((ROOT / "data/validation/e0_baseline/e0_manifest.json").read_text(encoding="utf-8"))
    manifest_by_id = {row["spectrum_id"]: row for row in manifest}
    screen_iks = {manifest_by_id[row["spectrum_id"]]["inchikey_14"] for row in pilot_selected}

    all_error_iks = {row["query_ik14"] for row in audit if as_bool(row["is_top1_error"])}
    error_pool = [
        row for row in audit
        if as_bool(row["is_top1_error"]) and row["query_ik14"] not in screen_iks
    ]
    # Correct controls come from molecules with no Top-1 error anywhere in E0,
    # preventing replicate-level label ambiguity.
    control_pool = [
        row for row in audit
        if not as_bool(row["is_top1_error"])
        and row["query_ik14"] not in all_error_iks
        and row["query_ik14"] not in screen_iks
    ]
    rng = np.random.default_rng(args.seed)
    errors = one_per_molecule(error_pool, rng, args.n_per_class)
    controls = one_per_molecule(control_pool, rng, args.n_per_class)
    n = min(len(errors), len(controls))
    errors, controls = errors[:n], controls[:n]
    selected = [(1, row) for row in errors] + [(0, row) for row in controls]
    print(f"Molecule-disjoint validation: {len(errors)} errors + {len(controls)} controls", flush=True)

    wanted = set()
    for _, row in selected:
        wanted.update([
            row["query_spectrum_id"], row["correct_best_spectrum_id"], row["best_negative_spectrum_id"]
        ])
    h5_rows = pilot.hdf5_index(args.data, wanted)
    RuleEngine = pilot.load_rule_engine_class()
    engine = RuleEngine(tolerance=args.rule_tolerance, use_massbank=True)
    matcher = pilot.FastRuleMatcher(engine, args.rule_tolerance)

    spectra: dict[str, np.ndarray] = {}
    precursors: dict[str, float] = {}
    with h5py.File(args.data, "r") as handle:
        for sid, index in h5_rows.items():
            spectra[sid] = pilot.preprocess_spectrum(handle, index, None)
            precursors[sid] = float(handle["precursor_mz"][index])
    vectors = {sid: matcher(spectra[sid], precursors[sid]) for sid in wanted}

    conflict_indices = np.asarray([
        int(row["rule_index"]) for row in rule_audit if row["panel_class"] == "conflict_mining_only"
    ], dtype=np.int64)
    panels = {
        "conflict_panel": conflict_indices,
        "core_335": np.arange(min(335, len(engine.rules))),
        "all_3486": np.arange(len(engine.rules)),
    }
    detail: list[dict[str, Any]] = []
    for label, row in selected:
        query = vectors[row["query_spectrum_id"]]
        true = vectors[row["correct_best_spectrum_id"]]
        wrong = vectors[row["best_negative_spectrum_id"]]
        output: dict[str, Any] = {
            "label_error": label,
            "query_spectrum_id": row["query_spectrum_id"],
            "query_ik14": row["query_ik14"],
            "same_formula": row["same_formula"],
            "scaffold_relation": row["scaffold_relation"],
            "candidate_molecules": row["candidate_molecules"],
            "dreams_margin": row["score_margin_correct_minus_negative"],
        }
        for name, indices in panels.items():
            score, true_count, wrong_count = direction_score(query, true, wrong, indices)
            output[f"{name}_score"] = score
            output[f"{name}_true_only"] = true_count
            output[f"{name}_wrong_only"] = wrong_count
        detail.append(output)
    write_csv(args.output_dir / "molecule_disjoint_query_scores.csv", detail)

    labels = np.asarray([row["label_error"] for row in detail], dtype=np.int64)
    summary: dict[str, Any] = {
        "status": "molecule_disjoint_clean_spectrum_validation",
        "n_error_molecules": len(errors),
        "n_control_molecules": len(controls),
        "n_conflict_rules": int(len(conflict_indices)),
        "screen_molecules_excluded": len(screen_iks),
        "panels": {},
    }
    for name in panels:
        scores = np.asarray([row[f"{name}_score"] for row in detail], dtype=np.float64)
        point, lo, hi = bootstrap_auc(labels, scores, pilot.stable_seed(args.seed, name), args.n_bootstrap)
        summary["panels"][name] = {
            "roc_auc_error_detection": point,
            "roc_auc_95ci": [lo, hi],
            "mean_score_errors": float(scores[labels == 1].mean()),
            "mean_score_controls": float(scores[labels == 0].mean()),
            "fraction_positive_errors": float((scores[labels == 1] > 0).mean()),
            "fraction_positive_controls": float((scores[labels == 0] > 0).mean()),
        }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    names = list(panels)
    positions = np.arange(len(names))
    error_means = [summary["panels"][name]["mean_score_errors"] for name in names]
    control_means = [summary["panels"][name]["mean_score_controls"] for name in names]
    axes[0].bar(positions - 0.18, error_means, 0.36, color="#e15759", label="DreaMS errors")
    axes[0].bar(positions + 0.18, control_means, 0.36, color="#4e79a7", label="Correct controls")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(positions, names)
    axes[0].set_ylabel("Wrong minus true normalized rule evidence")
    axes[0].set_title("Molecule-disjoint evidence direction")
    axes[0].legend(frameon=False)
    aucs = [summary["panels"][name]["roc_auc_error_detection"] for name in names]
    lows = [summary["panels"][name]["roc_auc_95ci"][0] for name in names]
    highs = [summary["panels"][name]["roc_auc_95ci"][1] for name in names]
    axes[1].bar(positions, aucs, color=["#59a14f", "#9ecae1", "#f28e8b"])
    axes[1].errorbar(positions, aucs, yerr=[np.asarray(aucs)-lows, np.asarray(highs)-aucs], fmt="none", color="black", capsize=4)
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylim(0.45, 1.0)
    axes[1].set_xticks(positions, names)
    axes[1].set_ylabel("ROC-AUC for detecting a DreaMS error")
    axes[1].set_title("Does the panel generalize?")
    fig.tight_layout()
    fig.savefig(args.output_dir / "conflict_panel_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    panel_lines = []
    for name, values in summary["panels"].items():
        panel_lines.append(
            f"| {name} | {values['roc_auc_error_detection']:.3f} "
            f"({values['roc_auc_95ci'][0]:.3f}-{values['roc_auc_95ci'][1]:.3f}) | "
            f"{values['mean_score_errors']:.3f} | {values['mean_score_controls']:.3f} |"
        )
    report = f"""# Molecule-disjoint conflict-rule validation

The 60 screening queries and every molecule represented by them were excluded.
The validation uses one query spectrum per molecule: {len(errors)} DreaMS-error
molecules and {len(controls)} molecules with no Top-1 error anywhere in strict E0.

The score is normalized wrong-only minus true-only rule evidence. A positive
score means that the rule panel supports the candidate selected incorrectly by
DreaMS more strongly than the correct identity.

| Panel | Error-detection ROC-AUC (95% CI) | Mean error score | Mean control score |
|---|---:|---:|---:|
{chr(10).join(panel_lines)}

This validates only conflict detection on clean spectra. It does not validate
rule injection or peak masking inside DreaMS.
"""
    (args.output_dir / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
