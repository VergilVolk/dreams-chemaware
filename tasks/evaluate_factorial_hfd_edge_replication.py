#!/usr/bin/env python3
"""Apply the frozen ChemProp edge-replication rule to factorial-HFD studies.

An edge is response support only when both endpoints pass within-panel BH FDR
0.05 for the genotype-by-HFD interaction and have the same effect direction.
Whole entity profiles are reassigned within analytical panel for the null.  The
two Hi-MYC experiments are the replicated-model control; the two comparisons
to NQO1 are explicitly exploratory discoveries, not registered truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTRAST = "genotype_by_hfd"
PAIR_ROLES = {
    ("MTBLS135", "MTBLS3316"): "registered_replicated_himyc_control",
    ("MTBLS135", "MTBLS2085"): "exploratory_himyc_to_nqo1",
    ("MTBLS3316", "MTBLS2085"): "exploratory_himyc_to_nqo1",
}
TRACKS = (
    "mutual_top1_all",
    "mutual_top1_identity_compatible",
    "mutual_top1_chemical_relation",
)


def unique_profiles(
    rows: pd.DataFrame, study_id: str,
) -> tuple[dict[tuple[str, str], tuple[float, int]], dict[str, list[str]]]:
    profiles = rows.loc[
        rows.study_id.eq(study_id)
        & rows.contrast_id.eq(CONTRAST)
        & rows.identity_status.eq("unresolved_entity")
    ].copy()
    if profiles.duplicated(["panel_id", "anchor_id"]).any():
        raise RuntimeError(f"conflicting entity response profiles for {study_id}")
    lookup = {
        (str(row.panel_id), str(row.anchor_id)): (
            float(row.effect_q), int(np.sign(float(row.beta)))
        )
        for row in profiles.itertuples(index=False)
    }
    by_panel: dict[str, list[str]] = {}
    for panel, anchor in lookup:
        by_panel.setdefault(panel, []).append(anchor)
    for panel in by_panel:
        by_panel[panel] = sorted(set(by_panel[panel]))
    return lookup, by_panel


def shuffled_donors(
    by_panel: dict[str, list[str]], rng: np.random.Generator,
) -> dict[tuple[str, str], tuple[str, str]]:
    donors: dict[tuple[str, str], tuple[str, str]] = {}
    for panel, anchors in sorted(by_panel.items()):
        shuffled = rng.permutation(anchors)
        donors.update({
            (panel, anchor): (panel, str(donor))
            for anchor, donor in zip(anchors, shuffled)
        })
    return donors


def support_masks(edges: pd.DataFrame) -> dict[str, np.ndarray]:
    identity = edges.evidence_track.eq("identity_compatible").to_numpy()
    return {
        "mutual_top1_all": np.ones(len(edges), dtype=bool),
        "mutual_top1_identity_compatible": identity,
        "mutual_top1_chemical_relation": ~identity,
    }


def count_support(
    edges: pd.DataFrame,
    left_profile: dict[tuple[str, str], tuple[float, int]],
    right_profile: dict[tuple[str, str], tuple[float, int]],
    left_donor: dict[tuple[str, str], tuple[str, str]] | None = None,
    right_donor: dict[tuple[str, str], tuple[str, str]] | None = None,
    fdr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_keys = list(zip(edges.query_panel_id.astype(str), edges.query_anchor_id.astype(str)))
    right_keys = list(zip(edges.target_panel_id.astype(str), edges.target_anchor_id.astype(str)))
    if left_donor is not None:
        left_keys = [left_donor[key] for key in left_keys]
    if right_donor is not None:
        right_keys = [right_donor[key] for key in right_keys]
    left_q = np.asarray([left_profile[key][0] for key in left_keys], dtype=float)
    right_q = np.asarray([right_profile[key][0] for key in right_keys], dtype=float)
    left_sign = np.asarray([left_profile[key][1] for key in left_keys], dtype=np.int8)
    right_sign = np.asarray([right_profile[key][1] for key in right_keys], dtype=np.int8)
    dual = (
        (left_q <= fdr) & (right_q <= fdr)
        & (left_sign != 0) & (left_sign == right_sign)
    )
    return dual, left_q, right_q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-contributions", required=True, type=Path)
    parser.add_argument("--entity-profiles", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--effect-fdr", type=float, default=0.05)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contributions = pd.read_csv(args.edge_contributions, sep="\t")
    profiles = pd.read_csv(args.entity_profiles, sep="\t", dtype={"anchor_id": str})
    required = {
        "query_study_id", "target_study_id", "query_contrast_id",
        "target_contrast_id", "evidence_world", "query_panel_id",
        "target_panel_id", "query_anchor_id", "target_anchor_id",
        "query_effect_q", "target_effect_q", "query_beta", "target_beta",
        "evidence_track", "cosine", "precursor_ppm",
    }
    if missing := required - set(contributions):
        raise ValueError(f"edge table lacks {sorted(missing)}")
    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, np.iinfo(np.int64).max, args.permutations, dtype=np.int64)
    summary_rows: list[dict[str, object]] = []
    supported_rows: list[pd.DataFrame] = []
    for (left_study, right_study), role in PAIR_ROLES.items():
        all_pair_rows = contributions.loc[
            contributions.query_study_id.eq(left_study)
            & contributions.target_study_id.eq(right_study)
            & contributions.evidence_world.eq("unresolved")
        ].copy()
        edges = all_pair_rows.loc[
            all_pair_rows.query_contrast_id.eq(CONTRAST)
            & all_pair_rows.target_contrast_id.eq(CONTRAST)
        ].drop_duplicates([
            "query_panel_id", "target_panel_id", "query_anchor_id", "target_anchor_id",
        ]).reset_index(drop=True)
        if not len(edges):
            raise RuntimeError(f"no unresolved interaction edges for {left_study}/{right_study}")
        left_profile, left_panels = unique_profiles(profiles, left_study)
        right_profile, right_panels = unique_profiles(profiles, right_study)
        observed, left_q, right_q = count_support(
            edges, left_profile, right_profile, fdr=args.effect_fdr,
        )
        masks = support_masks(edges)
        observed_counts = np.asarray(
            [int((observed & masks[track]).sum()) for track in TRACKS], dtype=int,
        )
        null = np.zeros((args.permutations, len(TRACKS)), dtype=np.int32)
        for permutation, seed in enumerate(seeds):
            prng = np.random.default_rng(int(seed))
            left_donor = shuffled_donors(left_panels, prng)
            right_donor = shuffled_donors(right_panels, prng)
            shuffled, _, _ = count_support(
                edges, left_profile, right_profile, left_donor, right_donor,
                args.effect_fdr,
            )
            null[permutation] = [
                int((shuffled & masks[track]).sum()) for track in TRACKS
            ]
        for track_index, track in enumerate(TRACKS):
            value = int(observed_counts[track_index])
            distribution = null[:, track_index]
            summary_rows.append({
                "left_study_id": left_study,
                "right_study_id": right_study,
                "pair_role": role,
                "contrast_id": CONTRAST,
                "track": track,
                "candidate_edges": int(masks[track].sum()),
                "dual_fdr_same_direction_edges": value,
                "null_mean": float(distribution.mean()),
                "null_q95": float(np.quantile(distribution, 0.95)),
                "empirical_p": float(
                    (1 + np.count_nonzero(distribution >= value))
                    / (args.permutations + 1)
                ),
            })
        if observed.any():
            supported = edges.loc[observed].copy()
            supported.insert(0, "pair_role", role)
            supported["query_effect_q_recomputed"] = left_q[observed]
            supported["target_effect_q_recomputed"] = right_q[observed]
            supported_rows.append(supported)

    summary = pd.DataFrame(summary_rows)
    supported = (
        pd.concat(supported_rows, ignore_index=True) if supported_rows
        else pd.DataFrame(columns=["pair_role"])
    )
    args.output_dir.mkdir(parents=True)
    summary.to_csv(args.output_dir / "pair_track_summary.tsv", sep="\t", index=False)
    supported.to_csv(
        args.output_dir / "dual_fdr_same_direction_edges.tsv.gz", sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    np.savez_compressed(
        args.output_dir / "profile_reassignment_null.npz",
        permutation_seed=seeds,
    )
    control = summary.loc[
        summary.pair_role.eq("registered_replicated_himyc_control")
    ]
    exploratory = summary.loc[summary.pair_role.eq("exploratory_himyc_to_nqo1")]
    report = {
        "schema": "factorial_hfd_edge_response_replication_v1",
        "status": "DEVELOPMENT_FACTORIAL_HFD_EDGE_REPLICATION_COMPLETE",
        "data_role": "development_consumed",
        "denominators": {
            "studies": 3, "study_pairs": len(PAIR_ROLES),
            "registered_control_pairs": 1, "exploratory_pairs": 2,
            "permutations": args.permutations,
        },
        "method": {
            "effect_gate": f"within-panel BH FDR <= {args.effect_fdr}",
            "direction_gate": "both endpoints nonzero and same beta sign",
            "chemical_edges": "unresolved same-polarity mutual top-1 DreaMS neighbours",
            "null": "whole entity-profile reassignment within analytical panel",
        },
        "registered_control_minimum_p_tracks": int(
            (control.empirical_p <= 1.0 / (args.permutations + 1)).sum()
        ),
        "exploratory_minimum_p_tracks": int(
            (exploratory.empirical_p <= 1.0 / (args.permutations + 1)).sum()
        ),
        "claim_boundary": [
            "The Hi-MYC pair is replicated-model development evidence, not independent-laboratory confirmation.",
            "NQO1 comparisons are discovery candidates and are not positive or negative truth.",
            "Identity-compatible edges remain identity hypotheses; isomers and ion redundancy are possible.",
            "Chemical-relation edges never pool effects as one molecule.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
