#!/usr/bin/env python3
"""Validate and freeze an untouched KGMN/MetDNA2 author baseline run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> int:
    return int(len(pd.read_csv(path)))


def _tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {token.strip() for token in str(value).split(";") if token.strip() and token.strip().lower() != "nan"}


def _base_feature(value: object, truth_features: set[str]) -> str | None:
    feature = str(value)
    if feature in truth_features:
        return feature
    stripped = re.sub(r"_[a-z]+$", "", feature)
    return stripped if stripped in truth_features else None


def evaluate_200std_truth(final_path: Path, truth_path: Path) -> dict[str, object]:
    """Evaluate the frozen author table against its bundled 200STD identities.

    The public truth file can contain multiple acceptable ZhuLab IDs for one
    chromatographic feature.  The final pair table can suffix duplicate feature
    names with ``_a``, ``_b``, ...; that suffix is removed only when the result
    maps back to an exact truth feature.  Score ties are counted against Top-1.
    """

    truth = pd.read_csv(truth_path)
    required_truth = {"name", "id"}
    if not required_truth.issubset(truth.columns):
        raise RuntimeError(f"200STD truth misses columns: {sorted(required_truth - set(truth.columns))}")
    truth = truth.dropna(subset=["name", "id"]).copy()
    truth["name"] = truth["name"].astype(str)
    truth["id"] = truth["id"].astype(str)
    truth_by_feature = truth.groupby("name")["id"].agg(lambda values: set(values)).to_dict()
    truth_features = set(truth_by_feature)

    result = pd.read_csv(final_path)
    required_result = {"peak_name", "id_zhulab", "total_score"}
    if not required_result.issubset(result.columns):
        raise RuntimeError(
            f"MetDNA2 final table misses truth-evaluation columns: "
            f"{sorted(required_result - set(result.columns))}"
        )
    result["truth_feature"] = result["peak_name"].map(lambda value: _base_feature(value, truth_features))
    result = result[result["truth_feature"].notna()].copy()
    result["score_numeric"] = pd.to_numeric(result["total_score"], errors="coerce")
    result["candidate_ids"] = result["id_zhulab"].map(_tokens)

    top1_correct = 0
    top5_correct = 0
    annotated = 0
    ambiguous_top_ties = 0
    per_feature: list[dict[str, object]] = []
    for feature in sorted(truth_features):
        frame = result[result["truth_feature"] == feature].copy()
        accepted = truth_by_feature[feature]
        candidate_rows = frame[frame["candidate_ids"].map(bool)]
        is_annotated = not candidate_rows.empty
        annotated += int(is_annotated)
        is_top1 = False
        is_top5 = False
        tied = False
        if is_annotated:
            scores = candidate_rows["score_numeric"]
            if scores.notna().any():
                maximum = scores.max()
                top = candidate_rows[scores == maximum]
                top_ids = set().union(*top["candidate_ids"].tolist())
                tied = len(top_ids) != 1
                is_top1 = (not tied) and bool(top_ids & accepted)
                ordered = candidate_rows.sort_values(
                    ["score_numeric", "peak_name", "id_zhulab"],
                    ascending=[False, True, True],
                    kind="mergesort",
                )
            else:
                # Fail closed on an author table without usable scores.  File
                # order is not a documented ranking protocol.
                ordered = candidate_rows.iloc[0:0]
            first_five_ids: set[str] = set()
            for ids in ordered.head(5)["candidate_ids"]:
                first_five_ids.update(ids)
            is_top5 = bool(first_five_ids & accepted)
        top1_correct += int(is_top1)
        top5_correct += int(is_top5)
        ambiguous_top_ties += int(tied)
        per_feature.append(
            {
                "feature": feature,
                "truth_ids": sorted(accepted),
                "annotated": is_annotated,
                "top1_correct": is_top1,
                "top5_correct": is_top5,
                "ambiguous_top_tie": tied,
            }
        )

    denominator = len(truth_features)
    if denominator == 0:
        raise RuntimeError("200STD truth contains no evaluable features")
    return {
        "protocol": (
            "bundled annotation_initial.csv feature-to-ZhuLab-ID truth; all truth features are denominators; "
            "maximum total_score ranks candidates; ties count against Top-1"
        ),
        "truth_features": denominator,
        "truth_identity_rows": int(len(truth)),
        "annotated_features": annotated,
        "coverage": annotated / denominator,
        "top1_correct": top1_correct,
        "recall1": top1_correct / denominator,
        "top5_correct": top5_correct,
        "recall5": top5_correct / denominator,
        "ambiguous_top_ties": ambiguous_top_ties,
        "per_feature": per_feature,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = args.output or (run_dir / "frozen_author_baseline.json")
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    required = {
        "input_ms1": run_dir / "data.csv",
        "input_ms2": run_dir / "spectra.msp",
        "sample_info": run_dir / "sample.info.csv",
        "truth": run_dir / "annotation_initial.csv",
        "annotated_features": run_dir / "peak_table_annotated_200STD_neg_200805.csv",
        "parameters": run_dir / "frozen_parameters.R",
        "source_commit": run_dir / "source_commit.txt",
        "run_log": run_dir / "run.log.txt",
        "genform": run_dir / "_runtime_genform" / "GenForm",
        "initial_seed": run_dir / "01_result_initial_seed_annotation" / "ms2_match_annotation_result.csv",
        "credential": run_dir / "03_annotation_credential" / "annontation_credential_long.csv",
        "final_identification": run_dir / "00_annotation_table" / "table1_identification.csv",
        "final_pairs": run_dir / "00_annotation_table" / "table3_identification_pair.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"incomplete author baseline; missing outputs: {missing}")

    log_text = required["run_log"].read_text(encoding="utf-8", errors="replace")
    source_commit = required["source_commit"].read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise RuntimeError("frozen MetDNA2 source commit is not a 40-character git object")
    expected_stages = (
        "Initial seed annotation",
        "Metabolic reaction network based metabolite annotation",
        "Annotaion Credential",
        "Merge and export result tables",
    )
    absent_stages = [stage for stage in expected_stages if stage not in log_text]
    if absent_stages:
        raise RuntimeError(f"author run log misses stages: {absent_stages}")

    csv_files = sorted(path for path in run_dir.rglob("*.csv") if path.is_file())
    truth_evaluation = evaluate_200std_truth(required["final_pairs"], required["truth"])
    report = {
        "status": "kgmn_metdna2_200std_author_baseline_frozen",
        "formal": True,
        "run_dir": str(run_dir),
        "protocol": "MetDNA2 1.2.10 public 200STD negative-mode evaluation; author code and data",
        "source_commit": source_commit,
        "counts": {
            "input_features": read_csv_rows(required["input_ms1"]),
            "initial_seed_rows": read_csv_rows(required["initial_seed"]),
            "credential_rows": read_csv_rows(required["credential"]),
            "final_identification_rows": read_csv_rows(required["final_identification"]),
            "final_identification_pairs": read_csv_rows(required["final_pairs"]),
        },
        "author_200std_truth_evaluation": truth_evaluation,
        "artifacts": {
            str(path.relative_to(run_dir)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(set(required.values()) | set(csv_files))
        },
        "contracts": {
            "author_algorithm_modified": False,
            "full_annotation_credential": True,
            "rt_calibration": False,
            "dreams_used": False,
            "bioaware_used": False,
            "baseline_must_precede_integration": True,
        },
        "claim_limit": (
            "This evaluates the public negative-mode 200STD demo against its bundled feature/ZhuLab-ID truth. "
            "It is a reproducible author-protocol baseline, not an independent-study performance estimate."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
