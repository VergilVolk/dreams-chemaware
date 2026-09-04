"""Audit whether ChemAware's five raw spectral views are independent evidence.

This is a post-hoc mechanism audit of already consumed observability splits. It
does not fit a gate, choose a threshold, or define a new decision rule.  The
report measures winner agreement and correlation of each view's truth-vs-DreaMS
wrong-candidate advantage.  High dependence means that "five-view consensus"
must not be presented as five independent experimental confirmations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
VIEWS = (
    "entropy_similarity",
    "sqrt_cosine",
    "linear_cosine",
    "top10_match_fraction",
    "intensity_coverage_min",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def winner(values: np.ndarray, identities: list[str]) -> str | None:
    maximum = float(np.max(values))
    top = np.flatnonzero(values == maximum)
    return identities[int(top[0])] if len(top) == 1 else None


def query_view_table(pair_path: Path, manifest_path: Path, split: str) -> pd.DataFrame:
    pairs = pd.read_csv(pair_path)
    manifest = pd.read_csv(manifest_path)
    if set(pairs["split"].astype(str)) != {split}:
        raise RuntimeError(f"pair split mismatch: {pair_path}")
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge, (left, right) in enumerate(
        pairs[["left", "right"]].itertuples(index=False, name=None)
    ):
        adjacency[int(left)].append((edge, int(right)))
        adjacency[int(right)].append((edge, int(left)))

    rows: list[dict] = []
    for query in range(len(manifest)):
        edges = adjacency.get(query, [])
        grouped: dict[str, list[int]] = defaultdict(list)
        for edge, candidate in edges:
            grouped[str(manifest.at[candidate, "ik14"])].append(edge)
        truth = str(manifest.at[query, "ik14"])
        if truth not in grouped or len(grouped) < 2:
            continue
        identities = sorted(grouped)
        score = np.empty((len(identities), len(VIEWS)), dtype=np.float64)
        dreams = np.empty(len(identities), dtype=np.float64)
        for index, identity in enumerate(identities):
            block = pairs.iloc[grouped[identity]]
            score[index] = block[list(VIEWS)].max(axis=0).to_numpy(float)
            dreams[index] = float(block["dreams_similarity"].max())
        view_winners = [winner(score[:, column], identities) for column in range(len(VIEWS))]
        dreams_winner = winner(dreams, identities)
        row: dict[str, object] = {
            "query_index": query,
            "truth": truth,
            "dreams_winner_rebuilt": dreams_winner,
        }
        for name, identity in zip(VIEWS, view_winners):
            row[f"winner_{name}"] = identity
            row[f"correct_{name}"] = identity == truth
        if dreams_winner is not None and dreams_winner != truth:
            truth_index = identities.index(truth)
            wrong_index = identities.index(dreams_winner)
            for column, name in enumerate(VIEWS):
                row[f"advantage_{name}"] = float(score[truth_index, column] - score[wrong_index, column])
        else:
            for name in VIEWS:
                row[f"advantage_{name}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_agreement(frame: pd.DataFrame) -> dict:
    output = {}
    for left in VIEWS:
        output[left] = {}
        a = frame[f"winner_{left}"]
        for right in VIEWS:
            b = frame[f"winner_{right}"]
            cast = a.notna() & b.notna()
            output[left][right] = {
                "both_cast": int(cast.sum()),
                "conditional_agreement": float((a[cast] == b[cast]).mean()) if cast.any() else None,
                "all_query_agreement": float(((a == b) & cast).mean()),
            }
    return output


def effective_rank(correlation: np.ndarray) -> float:
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    denominator = float(np.sum(eigenvalues ** 2))
    return float(np.sum(eigenvalues) ** 2 / denominator) if denominator else 0.0


def summarize(
    pair_path: Path, manifest_path: Path, ledger_path: Path, split: str,
) -> tuple[dict, pd.DataFrame]:
    table = query_view_table(pair_path, manifest_path, split)
    ledger = pd.read_csv(ledger_path)
    merged = ledger.merge(table, on="query_index", how="left", validate="one_to_one")
    if merged[[f"winner_{name}" for name in VIEWS]].isna().all(axis=None):
        raise RuntimeError(f"no reconstructed raw-view decisions for {split}")
    rebuilt = merged["dreams_winner_rebuilt"].fillna("").astype(str)
    recorded = merged["dreams_prediction"].fillna("").astype(str)
    # Recorded prediction can be lexicographic under a tie while the audit
    # explicitly abstains. Compare only unique rebuilt winners.
    unique = merged["dreams_winner_rebuilt"].notna()
    if not (rebuilt[unique] == recorded[unique]).all():
        raise RuntimeError(f"DreaMS winner reconstruction mismatch for {split}")

    winner_columns = [f"winner_{name}" for name in VIEWS]
    distinct = merged[winner_columns].nunique(axis=1, dropna=True)
    wrong = merged.loc[~merged["dreams_correct"].astype(bool)]
    advantage_columns = [f"advantage_{name}" for name in VIEWS]
    advantage = wrong[advantage_columns].dropna()
    correlation = advantage.corr(method="spearman").to_numpy(float)
    if np.any(~np.isfinite(correlation)):
        raise RuntimeError(f"non-finite advantage correlation for {split}")

    active = merged.loc[merged["route_activated"].astype(bool)]
    active_vote = {}
    for name in VIEWS:
        values = active[f"winner_{name}"]
        active_vote[name] = {
            "truth": int((values == active["ik14"]).sum()),
            "dreams_prediction": int((values == active["dreams_prediction"]).sum()),
            "other": int(
                (values.notna() & (values != active["ik14"])
                 & (values != active["dreams_prediction"])).sum()
            ),
            "abstain": int(values.isna().sum()),
        }
    result = {
        "eligible_queries": int(len(merged)),
        "dreams_wrong_queries": int((~merged["dreams_correct"].astype(bool)).sum()),
        "winner_dependence": {
            "all_five_same_nonabstaining_fraction": float(
                ((distinct == 1) & merged[winner_columns].notna().all(axis=1)).mean()
            ),
            "at_most_two_distinct_winners_fraction": float((distinct <= 2).mean()),
            "distinct_winners_mean": float(distinct.mean()),
            "pairwise": pairwise_agreement(merged),
        },
        "truth_vs_dreams_wrong_advantage": {
            "queries": int(len(advantage)),
            "spearman": {
                left: {right: float(correlation[i, j]) for j, right in enumerate(VIEWS)}
                for i, left in enumerate(VIEWS)
            },
            "correlation_effective_rank_of_five": effective_rank(correlation),
        },
        "frozen_gate_activated": {
            "queries": int(len(active)),
            "corrected": int(((~active["dreams_correct"]) & active["consensus_correct"]).sum()),
            "introduced": int((active["dreams_correct"] & ~active["consensus_correct"]).sum()),
            "per_view_votes": active_vote,
        },
    }
    return result, merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/chemaware_raw_view_dependence_20260902",
    )
    args = parser.parse_args()
    frozen = ROOT / "data/validation/chemaware_spectral_consensus_applicability_v4_frozen"
    residual = ROOT / "data/validation/large_observability_residual_audit"
    test_input = ROOT / "data/validation/chemaware_frozen_gate_test_inputs_20260902"
    test_output = ROOT / "data/validation/chemaware_frozen_spectral_gate_test_20260902"
    specifications = {
        "discovery": (
            residual / "discovery_pair_features.csv",
            ROOT / "data/validation/large_observability_embeddings_discovery/manifest.csv",
            frozen / "discovery_gate_ledger.csv.gz",
        ),
        "confirmation": (
            residual / "confirmation_pair_features.csv",
            ROOT / "data/validation/large_observability_embeddings_confirmation/manifest.csv",
            frozen / "confirmation_gate_ledger.csv.gz",
        ),
        "test": (
            test_input / "test_pair_features.csv.gz",
            test_input / "test_manifest.csv",
            test_output / "test_gate_ledger.csv.gz",
        ),
    }
    missing = [str(path) for values in specifications.values() for path in values if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")

    results = {}
    merged_tables = []
    for split, paths in specifications.items():
        results[split], table = summarize(*paths, split)
        table.insert(0, "audit_split", split)
        merged_tables.append(table)
    args.output_dir.mkdir(parents=True)
    ledger_path = args.output_dir / "raw_view_decision_ledger.csv.gz"
    pd.concat(merged_tables, ignore_index=True).to_csv(ledger_path, index=False)
    report = {
        "status": "chemaware_raw_view_dependence_audited",
        "post_hoc_only": True,
        "no_gate_fit_or_threshold_selection": True,
        "splits_already_consumed": ["discovery", "confirmation", "test"],
        "views": list(VIEWS),
        "results": results,
        "interpretation_contract": (
            "Decision agreement and effective rank quantify dependence, not causal independence. "
            "The five raw scores must not be described as five independent confirmations."
        ),
        "provenance": {
            "inputs": {
                split: {
                    "pair_sha256": sha256(paths[0]),
                    "manifest_sha256": sha256(paths[1]),
                    "gate_ledger_sha256": sha256(paths[2]),
                }
                for split, paths in specifications.items()
            },
            "decision_ledger_sha256": sha256(ledger_path),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
