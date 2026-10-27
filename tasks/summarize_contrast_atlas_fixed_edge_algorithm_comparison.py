#!/usr/bin/env python3
"""Compare fixed-edge Contrast-Atlas response runs after all scores are frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from encode_crossstudy_mgfs_checkpoint import sha256


RELATION_KEY = [
    "relation_direction",
    "left_study_id",
    "left_contrast_id",
    "right_study_id",
    "right_contrast_id",
    "evidence_world",
    "evidence_track",
]
TOP_KEY = [
    "query_study_id",
    "query_contrast_id",
    "evidence_world",
    "evidence_track",
    "response_face",
]


def parse_runs(values: list[str]) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"invalid --run value: {item}")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if not name or any(name == existing for existing, _ in output):
            raise ValueError(f"duplicate or empty run name: {name!r}")
        for filename in (
            "report.json",
            "registered_relation_ranks.tsv",
            "open_local_edge_top_hits.tsv",
        ):
            if not (path / filename).is_file():
                raise FileNotFoundError(path / filename)
        output.append((name, path))
    if len(output) < 2:
        raise ValueError("at least two runs are required")
    return output


def registered_summary(frame: pd.DataFrame, method: str) -> list[dict[str, object]]:
    primary = frame.loc[
        frame["eligible_primary_relevance"].astype(str).str.lower().eq("true")
    ].copy()
    rows: list[dict[str, object]] = []
    group_columns = ["relation_class", "evidence_world", "evidence_track"]
    for key, block in primary.groupby(group_columns, sort=True):
        finite_rank = pd.to_numeric(block["best_face_rank"], errors="coerce").dropna()
        rows.append(
            {
                "method": method,
                **dict(zip(group_columns, key, strict=True)),
                "directional_registered_relations": len(block),
                "best_face_top1": int(block["best_face_rank"].eq(1).sum()),
                "best_face_top5": int(block["best_face_rank"].le(5).sum()),
                "median_best_face_rank": (
                    float(finite_rank.median()) if len(finite_rank) else np.nan
                ),
                "best_face_maxT_005": int(
                    block["best_face_query_maxT_p"].le(0.05).sum()
                ),
            }
        )
    return rows


def canonical_relation_unit(row: pd.Series) -> str:
    left = (str(row["left_study_id"]), str(row["left_contrast_id"]))
    right = (str(row["right_study_id"]), str(row["right_contrast_id"]))
    low, high = sorted((left, right))
    return "||".join((*low, *high))


def paired_cluster_rank_summary(
    relations: dict[str, pd.DataFrame], runs: list[tuple[str, Path]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    methods = [name for name, _path in runs]
    index_columns = [
        "relation_direction",
        "left_study_id",
        "left_contrast_id",
        "right_study_id",
        "right_contrast_id",
        "evidence_world",
        "evidence_track",
    ]
    indexed = {name: frame.set_index(index_columns) for name, frame in relations.items()}
    for reference_position, reference_name in enumerate(methods[:-1]):
        for method_name in methods[reference_position + 1 :]:
            current = indexed[method_name]
            selected = current.loc[
                current["eligible_primary_relevance"].astype(str).str.lower().eq("true")
            ].copy()
            selected["reference_rank"] = indexed[reference_name].loc[
                selected.index, "best_face_rank"
            ].to_numpy(float)
            selected = selected.reset_index()
            selected["relation_unit"] = selected.apply(canonical_relation_unit, axis=1)
            selected["rank_delta"] = (
                pd.to_numeric(selected["best_face_rank"], errors="coerce")
                - pd.to_numeric(selected["reference_rank"], errors="coerce")
            )
            selected = selected.loc[np.isfinite(selected["rank_delta"])].copy()
            grouped = selected.groupby(
                ["relation_class", "evidence_world", "evidence_track", "relation_unit"],
                sort=True,
            )["rank_delta"].mean().reset_index()
            for key, block in grouped.groupby(
                ["relation_class", "evidence_world", "evidence_track"], sort=True
            ):
                delta = block["rank_delta"].dropna().to_numpy(float)
                nonzero = delta[delta != 0]
                sign_p = (
                    float(
                        binomtest(
                            int((nonzero < 0).sum()),
                            len(nonzero),
                            0.5,
                            alternative="greater",
                        ).pvalue
                    )
                    if len(nonzero)
                    else np.nan
                )
                signed_rank_p = (
                    float(wilcoxon(delta, alternative="less").pvalue)
                    if len(delta) and np.any(delta != 0)
                    else np.nan
                )
                rows.append(
                    {
                        "reference_method": reference_name,
                        "method": method_name,
                        "relation_class": key[0],
                        "evidence_world": key[1],
                        "evidence_track": key[2],
                        "undirected_relation_units": len(delta),
                        "units_rank_improved": int((delta < 0).sum()),
                        "units_rank_worsened": int((delta > 0).sum()),
                        "units_rank_unchanged": int((delta == 0).sum()),
                        "median_rank_delta": float(np.median(delta)),
                        "one_sided_sign_p": sign_p,
                        "one_sided_wilcoxon_p": signed_rank_p,
                    }
                )
    return pd.DataFrame(rows)


def paired_world_rank_summary(
    relations: dict[str, pd.DataFrame], runs: list[tuple[str, Path]]
) -> pd.DataFrame:
    """Compare all-entity ranks with known-only and unresolved-only ranks by cluster."""
    rows: list[dict[str, object]] = []
    pair_key = [
        "relation_direction",
        "left_study_id",
        "left_contrast_id",
        "right_study_id",
        "right_contrast_id",
        "evidence_track",
    ]
    for method, _path in runs:
        primary = relations[method].loc[
            relations[method]["eligible_primary_relevance"]
            .astype(str)
            .str.lower()
            .eq("true")
        ].copy()
        primary["relation_unit"] = primary.apply(canonical_relation_unit, axis=1)
        all_world = primary.loc[primary["evidence_world"].eq("all")].set_index(pair_key)
        for reference_world in ("known", "unresolved"):
            reference = primary.loc[
                primary["evidence_world"].eq(reference_world)
            ].set_index(pair_key)
            common = all_world.index.intersection(reference.index)
            selected = all_world.loc[common].reset_index().copy()
            selected["reference_rank"] = reference.loc[
                common, "best_face_rank"
            ].to_numpy(float)
            selected["rank_delta"] = (
                pd.to_numeric(selected["best_face_rank"], errors="coerce")
                - pd.to_numeric(selected["reference_rank"], errors="coerce")
            )
            selected = selected.loc[np.isfinite(selected["rank_delta"])].copy()
            grouped = selected.groupby(
                ["relation_class", "evidence_track", "relation_unit"], sort=True
            )["rank_delta"].mean().reset_index()
            for key, block in grouped.groupby(
                ["relation_class", "evidence_track"], sort=True
            ):
                delta = block["rank_delta"].to_numpy(float)
                nonzero = delta[delta != 0]
                sign_p = (
                    float(
                        binomtest(
                            int((nonzero < 0).sum()),
                            len(nonzero),
                            0.5,
                            alternative="greater",
                        ).pvalue
                    )
                    if len(nonzero)
                    else np.nan
                )
                signed_rank_p = (
                    float(wilcoxon(delta, alternative="less").pvalue)
                    if len(delta) and np.any(delta != 0)
                    else np.nan
                )
                rows.append(
                    {
                        "method": method,
                        "comparison": f"all_vs_{reference_world}",
                        "relation_class": key[0],
                        "evidence_track": key[1],
                        "undirected_relation_units": len(delta),
                        "units_all_rank_improved": int((delta < 0).sum()),
                        "units_all_rank_worsened": int((delta > 0).sum()),
                        "units_all_rank_unchanged": int((delta == 0).sum()),
                        "median_all_minus_reference_rank": float(np.median(delta)),
                        "one_sided_sign_p": sign_p,
                        "one_sided_wilcoxon_p": signed_rank_p,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    runs = parse_runs(args.run)
    baseline_name = runs[0][0]

    reports: dict[str, dict[str, object]] = {}
    relations: dict[str, pd.DataFrame] = {}
    top_hits: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    for name, path in runs:
        reports[name] = json.loads((path / "report.json").read_text(encoding="utf-8"))
        relations[name] = pd.read_csv(path / "registered_relation_ranks.tsv", sep="\t")
        top_hits[name] = pd.read_csv(path / "open_local_edge_top_hits.tsv", sep="\t")
        summary_rows.extend(registered_summary(relations[name], name))

    reference_relations = set(map(tuple, relations[baseline_name][RELATION_KEY].astype(str).to_numpy()))
    reference_top = set(map(tuple, top_hits[baseline_name][TOP_KEY].astype(str).to_numpy()))
    for name, _path in runs[1:]:
        if set(map(tuple, relations[name][RELATION_KEY].astype(str).to_numpy())) != reference_relations:
            raise RuntimeError(f"registered relation axis drifted for {name}")
        if set(map(tuple, top_hits[name][TOP_KEY].astype(str).to_numpy())) != reference_top:
            raise RuntimeError(f"open-query top-hit axis drifted for {name}")

    baseline_rel = relations[baseline_name].set_index(RELATION_KEY)
    delta_frames: list[pd.DataFrame] = []
    for name, _path in runs[1:]:
        current = relations[name].set_index(RELATION_KEY)
        delta = current.reset_index().copy()
        delta.insert(0, "baseline_method", baseline_name)
        delta.insert(1, "method", name)
        base_rank = baseline_rel.loc[current.index, "best_face_rank"].to_numpy(float)
        rank = current["best_face_rank"].to_numpy(float)
        delta["baseline_best_face_rank"] = base_rank
        delta["rank_delta_vs_baseline"] = rank - base_rank
        delta["rank_change_class"] = np.select(
            [rank < base_rank, rank > base_rank],
            ["improved", "worsened"],
            default="unchanged",
        )
        base_pass = (
            baseline_rel.loc[current.index, "best_face_query_maxT_p"].to_numpy(float)
            <= 0.05
        )
        now_pass = current["best_face_query_maxT_p"].to_numpy(float) <= 0.05
        delta["maxT_change_class"] = np.select(
            [now_pass & ~base_pass, ~now_pass & base_pass],
            ["gained", "lost"],
            default="unchanged",
        )
        delta_frames.append(delta)
    relation_deltas = pd.concat(delta_frames, ignore_index=True)

    baseline_top = top_hits[baseline_name].set_index(TOP_KEY)
    open_frames: list[pd.DataFrame] = []
    for name, _path in runs[1:]:
        current = top_hits[name].set_index(TOP_KEY)
        block = current.reset_index()[TOP_KEY].copy()
        block.insert(0, "baseline_method", baseline_name)
        block.insert(1, "method", name)
        block["baseline_candidate_study_id"] = baseline_top.loc[
            current.index, "candidate_study_id"
        ].astype(str).to_numpy()
        block["candidate_study_id"] = current["candidate_study_id"].astype(str).to_numpy()
        block["candidate_changed"] = block["baseline_candidate_study_id"].ne(
            block["candidate_study_id"]
        )
        block["baseline_result_class"] = baseline_top.loc[
            current.index, "result_class"
        ].astype(str).to_numpy()
        block["result_class"] = current["result_class"].astype(str).to_numpy()
        open_frames.append(block)
    open_changes = pd.concat(open_frames, ignore_index=True)
    paired_cluster = paired_cluster_rank_summary(relations, runs)
    paired_world = paired_world_rank_summary(relations, runs)

    # Within each method, quantify the unresolved-information contrast without
    # treating a registered relation as an independent external truth source.
    world_rows: list[dict[str, object]] = []
    pair_key = [
        "relation_direction",
        "left_study_id",
        "left_contrast_id",
        "right_study_id",
        "right_contrast_id",
        "evidence_track",
    ]
    for name, _path in runs:
        primary = relations[name].loc[
            relations[name]["eligible_primary_relevance"].astype(str).str.lower().eq("true")
        ]
        known = primary.loc[primary["evidence_world"].eq("known")].set_index(pair_key)
        all_world = primary.loc[primary["evidence_world"].eq("all")].set_index(pair_key)
        common = known.index.intersection(all_world.index)
        for track in sorted(primary["evidence_track"].unique()):
            selected = [index for index in common if index[-1] == track]
            if not selected:
                continue
            known_rank = known.loc[selected, "best_face_rank"].to_numpy(float)
            all_rank = all_world.loc[selected, "best_face_rank"].to_numpy(float)
            delta = all_rank - known_rank
            finite_delta = delta[np.isfinite(delta)]
            world_rows.append(
                {
                    "method": name,
                    "evidence_track": track,
                    "registered_relations": len(selected),
                    "all_rank_better_than_known": int((all_rank < known_rank).sum()),
                    "all_rank_worse_than_known": int((all_rank > known_rank).sum()),
                    "all_rank_same_as_known": int((all_rank == known_rank).sum()),
                    "median_all_minus_known_rank": (
                        float(np.median(finite_delta)) if len(finite_delta) else np.nan
                    ),
                }
            )

    args.output_dir.mkdir(parents=True)
    outputs = {
        "summary": args.output_dir / "registered_performance_by_method.tsv",
        "relation_deltas": args.output_dir / "registered_relation_deltas.tsv",
        "open_changes": args.output_dir / "open_top_hit_changes.tsv.gz",
        "world_increment": args.output_dir / "all_vs_known_registered_increment.tsv",
        "paired_cluster": args.output_dir / "paired_cluster_rank_comparison.tsv",
        "paired_world": args.output_dir / "paired_world_rank_comparison.tsv",
    }
    pd.DataFrame(summary_rows).to_csv(outputs["summary"], sep="\t", index=False)
    relation_deltas.to_csv(outputs["relation_deltas"], sep="\t", index=False)
    open_changes.to_csv(
        outputs["open_changes"],
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(world_rows).to_csv(outputs["world_increment"], sep="\t", index=False)
    paired_cluster.to_csv(outputs["paired_cluster"], sep="\t", index=False)
    paired_world.to_csv(outputs["paired_world"], sep="\t", index=False)

    report = {
        "status": "CONTRAST_ATLAS_FIXED_EDGE_ALGORITHM_COMPARISON_COMPLETE",
        "data_role": "development_consumed",
        "baseline_method": baseline_name,
        "methods": [name for name, _ in runs],
        "denominators": {
            "registered_rows_per_method": len(relations[baseline_name]),
            "open_top_hit_rows_per_method": len(top_hits[baseline_name]),
            "registered_relation_delta_rows": len(relation_deltas),
            "open_top_hit_delta_rows": len(open_changes),
        },
        "interpretation_boundary": [
            "Registered relations are consumed development controls, not sealed truth.",
            "Candidate edges and response effects are identical across methods; only fixed-edge strength changes.",
            "Open top-hit changes are unadjudicated hypotheses and are not counted as corrections.",
            "All-versus-known is a within-panel information comparison, not proof that unresolved entities are correctly identified.",
        ],
        "inputs": {
            name: {
                "run_dir": str(path),
                "report_sha256": sha256(path / "report.json"),
                "registered_sha256": sha256(path / "registered_relation_ranks.tsv"),
                "top_hits_sha256": sha256(path / "open_local_edge_top_hits.tsv"),
            }
            for name, path in runs
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in [*outputs.values(), report_path]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
