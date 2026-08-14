"""Within-molecule paired validation of rule-based DreaMS error signals."""

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
import validate_conflict_rule_panel as base


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--audit-csv", type=Path, default=ROOT / "data/validation/e0_failure_audit/e0_query_audit.csv")
    parser.add_argument("--screen-dir", type=Path, default=ROOT / "data/validation/compact_rule_panel")
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / "data/validation/rule_noise_pilot")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/conflict_rule_validation")
    parser.add_argument("--n-pairs", type=int, default=300)
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = read_csv(args.audit_csv)
    pilot_selected = read_csv(args.pilot_dir / "selected_queries.csv")
    rule_audit = read_csv(args.screen_dir / "rule_level_audit.csv")
    manifest = json.loads((ROOT / "data/validation/e0_baseline/e0_manifest.json").read_text(encoding="utf-8"))
    manifest_by_id = {row["spectrum_id"]: row for row in manifest}
    screen_iks = {manifest_by_id[row["spectrum_id"]]["inchikey_14"] for row in pilot_selected}

    by_molecule: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in audit:
        ik = row["query_ik14"]
        if ik in screen_iks:
            continue
        by_molecule.setdefault(ik, {"error": [], "correct": []})[
            "error" if base.as_bool(row["is_top1_error"]) else "correct"
        ].append(row)
    eligible = [ik for ik, values in by_molecule.items() if values["error"] and values["correct"]]
    rng = np.random.default_rng(args.seed)
    eligible = [eligible[int(i)] for i in rng.permutation(len(eligible))[: args.n_pairs]]
    selected: list[tuple[int, str, dict[str, str]]] = []
    for ik in eligible:
        values = by_molecule[ik]
        error = values["error"][int(rng.integers(len(values["error"])))]
        correct = values["correct"][int(rng.integers(len(values["correct"])))]
        selected.extend([(1, ik, error), (0, ik, correct)])
    print(f"Within-molecule validation: {len(eligible)} paired molecules", flush=True)

    wanted = set()
    for _, _, row in selected:
        wanted.update([row["query_spectrum_id"], row["correct_best_spectrum_id"], row["best_negative_spectrum_id"]])
    h5_rows = pilot.hdf5_index(args.data, wanted)
    RuleEngine = pilot.load_rule_engine_class()
    engine = RuleEngine(tolerance=args.rule_tolerance, use_massbank=True)
    matcher = pilot.FastRuleMatcher(engine, args.rule_tolerance)
    spectra, precursors = {}, {}
    with h5py.File(args.data, "r") as handle:
        for sid, index in h5_rows.items():
            spectra[sid] = pilot.preprocess_spectrum(handle, index, None)
            precursors[sid] = float(handle["precursor_mz"][index])
    vectors = {sid: matcher(spectra[sid], precursors[sid]) for sid in wanted}

    conflict = np.asarray([
        int(row["rule_index"]) for row in rule_audit if row["panel_class"] == "conflict_mining_only"
    ], dtype=np.int64)
    panels = {
        "conflict_panel": conflict,
        "core_335": np.arange(min(335, len(engine.rules))),
        "all_3486": np.arange(len(engine.rules)),
    }
    detail = []
    for label, pair_ik, row in selected:
        query = vectors[row["query_spectrum_id"]]
        true = vectors[row["correct_best_spectrum_id"]]
        wrong = vectors[row["best_negative_spectrum_id"]]
        output: dict[str, Any] = {
            "pair_ik14": pair_ik,
            "label_error": label,
            "query_spectrum_id": row["query_spectrum_id"],
            "query_instrument": row["query_instrument"],
            "same_instrument_as_best_positive": row["same_instrument_as_best_positive"],
            "dreams_margin": row["score_margin_correct_minus_negative"],
        }
        for name, indices in panels.items():
            score, true_count, wrong_count = base.direction_score(query, true, wrong, indices)
            output[f"{name}_score"] = score
            output[f"{name}_true_only"] = true_count
            output[f"{name}_wrong_only"] = wrong_count
        detail.append(output)
    write_csv(args.output_dir / "within_molecule_paired_scores.csv", detail)

    labels = np.asarray([row["label_error"] for row in detail], dtype=np.int64)
    summary: dict[str, Any] = {
        "status": "within_molecule_paired_clean_spectrum_validation",
        "n_paired_molecules": len(eligible),
        "n_conflict_rules": int(len(conflict)),
        "panels": {},
    }
    for name in panels:
        scores = np.asarray([row[f"{name}_score"] for row in detail], dtype=np.float64)
        point, lo, hi = base.bootstrap_auc(labels, scores, pilot.stable_seed(args.seed, "paired", name), args.n_bootstrap)
        summary["panels"][name] = {
            "roc_auc_error_detection": point,
            "roc_auc_95ci": [lo, hi],
            "mean_score_errors": float(scores[labels == 1].mean()),
            "mean_score_correct": float(scores[labels == 0].mean()),
        }
    (args.output_dir / "within_molecule_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    names = list(panels)
    aucs = [summary["panels"][name]["roc_auc_error_detection"] for name in names]
    lows = [summary["panels"][name]["roc_auc_95ci"][0] for name in names]
    highs = [summary["panels"][name]["roc_auc_95ci"][1] for name in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    x = np.arange(len(names))
    ax.bar(x, aucs, color=["#59a14f", "#9ecae1", "#f28e8b"])
    ax.errorbar(x, aucs, yerr=[np.asarray(aucs)-lows, np.asarray(highs)-aucs], fmt="none", color="black", capsize=4)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 0.75)
    ax.set_xticks(x, names)
    ax.set_ylabel("ROC-AUC for error query within the same molecule")
    ax.set_title("Within-molecule paired validation")
    fig.tight_layout()
    fig.savefig(args.output_dir / "within_molecule_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    lines = []
    for name, values in summary["panels"].items():
        lines.append(
            f"| {name} | {values['roc_auc_error_detection']:.3f} "
            f"({values['roc_auc_95ci'][0]:.3f}-{values['roc_auc_95ci'][1]:.3f}) | "
            f"{values['mean_score_errors']:.3f} | {values['mean_score_correct']:.3f} |"
        )
    report = f"""# Within-molecule paired validation

For each of {len(eligible)} molecules, one Top-1 error spectrum and one correctly
retrieved spectrum were sampled. The molecular identity is therefore held
constant; only the experimental spectrum and its best competing candidate vary.

| Panel | ROC-AUC (95% CI) | Mean error score | Mean correct score |
|---|---:|---:|---:|
{chr(10).join(lines)}

If the AUC approaches 0.5 here, the earlier molecule-disjoint signal mainly
reflects which molecular families are difficult. If it remains above 0.5, rule
evidence also tracks spectrum-level failure within the same molecule.
"""
    (args.output_dir / "WITHIN_MOLECULE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
