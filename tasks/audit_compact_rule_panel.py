"""Derive preliminary corrective and conflict-mining rule panels.

This consumes the fixed query/candidate identities from the rule-noise pilot and
recomputes every rule under DreaMS-style intensity-proportional peak masking.
It does not train a model.  The output is a screening panel, not a validated
chemical rule set.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

import pilot_rule_noise_stress as pilot


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen compact rules from the completed noise pilot")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / "data/validation/rule_noise_pilot")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/compact_rule_panel")
    parser.add_argument("--mask-rates", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--min-molecule-support", type=int, default=3)
    parser.add_argument("--margin-threshold", type=float, default=0.08)
    parser.add_argument("--stability-threshold", type=float, default=0.60)
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


def molecule_support(records: list[dict[str, Any]], field: str, n_rules: int) -> np.ndarray:
    by_molecule: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in records:
        by_molecule[record["ik14"]].append(record[field])
    support = np.zeros(n_rules, dtype=np.int64)
    for values in by_molecule.values():
        support += np.logical_or.reduce(values).astype(np.int64)
    return support


def group_statistics(records: list[dict[str, Any]], n_rules: int) -> dict[str, Any]:
    if not records:
        zero = np.zeros(n_rules, dtype=np.float64)
        return {key: zero.copy() for key in [
            "coverage", "true_frequency", "wrong_frequency", "margin", "stability",
            "true_molecule_support", "wrong_molecule_support",
        ]} | {"n_queries": 0, "n_molecules": 0}
    clean_hit = np.stack([record["clean_query"] for record in records])
    true_clean = np.stack([record["clean_true_only"] for record in records])
    wrong_clean = np.stack([record["clean_wrong_only"] for record in records])
    noisy_hit = np.stack([record["noisy_query_frequency"] for record in records])
    noisy_true = np.stack([record["noisy_true_frequency"] for record in records])
    noisy_wrong = np.stack([record["noisy_wrong_frequency"] for record in records])
    denominator = clean_hit.sum(axis=0)
    stability = np.divide(
        (noisy_hit * clean_hit).sum(axis=0), denominator,
        out=np.zeros(n_rules, dtype=np.float64), where=denominator > 0,
    )
    return {
        "n_queries": len(records),
        "n_molecules": len({record["ik14"] for record in records}),
        "coverage": clean_hit.mean(axis=0),
        "true_frequency": noisy_true.mean(axis=0),
        "wrong_frequency": noisy_wrong.mean(axis=0),
        "margin": noisy_true.mean(axis=0) - noisy_wrong.mean(axis=0),
        "stability": stability,
        "true_molecule_support": molecule_support(records, "clean_true_only", n_rules),
        "wrong_molecule_support": molecule_support(records, "clean_wrong_only", n_rules),
        "true_query_support": true_clean.sum(axis=0),
        "wrong_query_support": wrong_clean.sum(axis=0),
    }


def classify_rule(
    error: dict[str, Any], control: dict[str, Any], total: dict[str, Any], index: int,
    min_support: int, margin_threshold: float, stability_threshold: float,
) -> str:
    error_margin = float(error["margin"][index])
    error_stability = float(error["stability"][index])
    true_support = int(error["true_molecule_support"][index])
    wrong_support = int(error["wrong_molecule_support"][index])
    if (
        true_support >= min_support
        and error_margin >= margin_threshold
        and error_stability >= stability_threshold
        and true_support >= wrong_support
    ):
        return "corrective_candidate"
    if (
        wrong_support >= min_support
        and error_margin <= -margin_threshold
        and error_stability >= stability_threshold
    ):
        return "conflict_mining_only"
    if float(total["coverage"][index]) >= 0.80:
        return "generic_high_coverage"
    if (
        float(total["coverage"][index]) > 0
        and float(total["stability"][index]) < 0.50
    ):
        return "noise_fragile"
    return "insufficient_or_nonspecific"


def safe_value(value: Any) -> str:
    if isinstance(value, tuple):
        return ":".join(f"{float(item):.6g}" for item in value)
    return f"{float(value):.6g}"


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    classes = Counter(row["panel_class"] for row in rows)
    order = [
        "corrective_candidate", "conflict_mining_only", "generic_high_coverage",
        "noise_fragile", "insufficient_or_nonspecific",
    ]
    colors = ["#59a14f", "#e15759", "#f28e2b", "#b279a2", "#bab0ac"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].barh(order[::-1], [classes[name] for name in order[::-1]], color=colors[::-1])
    axes[0].set_xlabel("Number of rules")
    axes[0].set_title("Preliminary rule-panel assignment")
    scatter_colors = {name: color for name, color in zip(order, colors)}
    for name in order:
        subset = [row for row in rows if row["panel_class"] == name]
        axes[1].scatter(
            [row["error_masked_margin"] for row in subset],
            [row["error_stability"] for row in subset],
            s=12, alpha=0.55, color=scatter_colors[name], label=name,
        )
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].axhline(0.60, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Error-set true minus wrong rule frequency")
    axes[1].set_ylabel("Mask stability")
    axes[1].set_title("Corrective direction and robustness")
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(rows: list[dict[str, Any]], limit: int = 12) -> str:
    header = "| Rule | Category | Source | Error margin | Error stability | True/Wrong molecule support |\n|---|---|---|---:|---:|---:|"
    lines = [header]
    for row in rows[:limit]:
        lines.append(
            f"| {row['rule_name']} | {row['category']} | {row['source']} | "
            f"{row['error_masked_margin']:.3f} | {row['error_stability']:.3f} | "
            f"{row['error_true_molecule_support']}/{row['error_wrong_molecule_support']} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = read_csv(args.pilot_dir / "selected_queries.csv")
    manifest = json.loads((ROOT / "data/validation/e0_baseline/e0_manifest.json").read_text(encoding="utf-8"))
    manifest_by_id = {row["spectrum_id"]: row for row in manifest}
    wanted = {row["spectrum_id"] for row in selected}
    wanted.update(row["true_spectrum_id"] for row in selected)
    wanted.update(row["wrong_spectrum_id"] for row in selected)
    h5_rows = pilot.hdf5_index(args.data, wanted)

    RuleEngine = pilot.load_rule_engine_class()
    engine = RuleEngine(tolerance=args.rule_tolerance, use_massbank=True)
    matcher = pilot.FastRuleMatcher(engine, args.rule_tolerance)
    n_rules = len(engine.rules)

    spectra: dict[str, np.ndarray] = {}
    precursors: dict[str, float] = {}
    with h5py.File(args.data, "r") as handle:
        for sid, index in h5_rows.items():
            spectra[sid] = pilot.preprocess_spectrum(handle, index, None)
            precursors[sid] = float(handle["precursor_mz"][index])
    clean = {sid: matcher(spectra[sid], precursors[sid]) for sid in wanted}

    records: list[dict[str, Any]] = []
    for row in selected:
        sid = row["spectrum_id"]
        query = clean[sid]
        true = clean[row["true_spectrum_id"]]
        wrong = clean[row["wrong_spectrum_id"]]
        noisy_vectors = []
        for rate in args.mask_rates:
            for repeat in range(args.n_seeds):
                noisy, _ = pilot.perturb(
                    spectra[sid], rate, "native_mask",
                    pilot.stable_seed(args.seed, sid, rate, repeat), -1.0,
                )
                noisy_vectors.append(matcher(noisy, precursors[sid]))
        noisy = np.stack(noisy_vectors).astype(np.float64)
        groups = row["groups"].split("|")
        records.append({
            "spectrum_id": sid,
            "ik14": manifest_by_id[sid]["inchikey_14"],
            "source": row["source"],
            "groups": groups,
            "clean_query": query,
            "clean_true_only": query & true & ~wrong,
            "clean_wrong_only": query & wrong & ~true,
            "noisy_query_frequency": noisy.mean(axis=0),
            "noisy_true_frequency": (noisy.astype(bool) & true & ~wrong).mean(axis=0),
            "noisy_wrong_frequency": (noisy.astype(bool) & wrong & ~true).mean(axis=0),
        })

    groups: dict[str, list[dict[str, Any]]] = {
        "all": records,
        "error": [record for record in records if record["source"] == "p0_error"],
        "control": [record for record in records if record["source"] == "control"],
        "local_mces_0_2": [record for record in records if "local_mces_0_2" in record["groups"]],
        "high_rule_conflict": [record for record in records if "high_rule_conflict" in record["groups"]],
        "cross_instrument": [record for record in records if "cross_instrument" in record["groups"]],
    }
    stats = {name: group_statistics(values, n_rules) for name, values in groups.items()}

    output_rows: list[dict[str, Any]] = []
    for index, rule in enumerate(engine.rules):
        panel_class = classify_rule(
            stats["error"], stats["control"], stats["all"], index,
            args.min_molecule_support, args.margin_threshold, args.stability_threshold,
        )
        row = {
            "rule_index": index,
            "rule_name": rule.name,
            "category": rule.category,
            "match_type": rule.match_type,
            "value": safe_value(rule.value),
            "source": rule.source,
            "panel_class": panel_class,
            "error_masked_margin": float(stats["error"]["margin"][index]),
            "control_masked_margin": float(stats["control"]["margin"][index]),
            "net_screening_score": float(
                stats["error"]["margin"][index] + 0.25 * stats["control"]["margin"][index]
            ),
            "error_stability": float(stats["error"]["stability"][index]),
            "control_stability": float(stats["control"]["stability"][index]),
            "all_clean_coverage": float(stats["all"]["coverage"][index]),
            "error_true_molecule_support": int(stats["error"]["true_molecule_support"][index]),
            "error_wrong_molecule_support": int(stats["error"]["wrong_molecule_support"][index]),
            "control_true_molecule_support": int(stats["control"]["true_molecule_support"][index]),
            "control_wrong_molecule_support": int(stats["control"]["wrong_molecule_support"][index]),
            "local_error_margin": float(stats["local_mces_0_2"]["margin"][index]),
            "high_rule_error_margin": float(stats["high_rule_conflict"]["margin"][index]),
            "cross_instrument_error_margin": float(stats["cross_instrument"]["margin"][index]),
        }
        output_rows.append(row)

    rank_order = {
        "corrective_candidate": 0, "conflict_mining_only": 1,
        "generic_high_coverage": 2, "noise_fragile": 3,
        "insufficient_or_nonspecific": 4,
    }
    output_rows.sort(key=lambda row: (
        rank_order[row["panel_class"]],
        -abs(row["net_screening_score"]),
        -row["error_stability"],
        row["rule_index"],
    ))
    write_csv(args.output_dir / "rule_level_audit.csv", output_rows)
    actionable = [row for row in output_rows if row["panel_class"] != "insufficient_or_nonspecific"]
    write_csv(args.output_dir / "screened_rule_panels.csv", actionable)
    plot(output_rows, args.output_dir / "compact_rule_panel.png")

    corrective = [row for row in output_rows if row["panel_class"] == "corrective_candidate"]
    conflict = [row for row in output_rows if row["panel_class"] == "conflict_mining_only"]
    generic = [row for row in output_rows if row["panel_class"] == "generic_high_coverage"]
    fragile = [row for row in output_rows if row["panel_class"] == "noise_fragile"]
    summary = {
        "status": "exploratory_screen_not_validated_rule_set",
        "n_rules": n_rules,
        "n_queries": len(records),
        "n_error_queries": len(groups["error"]),
        "n_control_queries": len(groups["control"]),
        "thresholds": {
            "min_error_molecule_support": args.min_molecule_support,
            "absolute_masked_margin": args.margin_threshold,
            "mask_stability": args.stability_threshold,
        },
        "panel_counts": dict(Counter(row["panel_class"] for row in output_rows)),
        "next_gate": "Validate candidates on a molecule-disjoint sample before any loss injection.",
    }
    (args.output_dir / "rule_panel_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# Preliminary compact rule panel

## Status

This is an exploratory screen from 36 selected DreaMS errors and 24 matched
correct controls. It is not a validated rule set and must not yet enter a loss.

## Assignment

| Panel | Rules | Intended role |
|---|---:|---|
| Corrective candidates | {len(corrective)} | Candidate concept-decoding supervision after independent validation |
| Conflict-mining only | {len(conflict)} | Find hard negatives; never define positive identity |
| Generic high coverage | {len(generic)} | Context/quality features, not discriminative labels |
| Noise fragile | {len(fragile)} | Exclude from the first robustness experiment |
| Insufficient/nonspecific | {n_rules-len(corrective)-len(conflict)-len(generic)-len(fragile)} | Do not use yet |

## Top corrective candidates

{markdown_table(corrective)}

## Strongest conflict-mining rules

{markdown_table(conflict)}

## Decision

The 3,486-rule library must not be injected as one block. Only independently
replicated corrective candidates may enter a concept head. Stable wrong-directed
rules are retained explicitly as conflict miners because they describe where
fragmentation evidence makes different structures look alike.
"""
    (args.output_dir / "RULE_PANEL_DECISION.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
