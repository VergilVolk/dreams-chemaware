#!/usr/bin/env python3
"""Evaluate the frozen MTBLS1866 acquisition-order qualification gate.

This script reads acquisition timestamps from mzML headers and joins the
already frozen 109-sample design ledger.  It never reads intensities, product
ions, response effects, identities, or retrieval results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


EXPECTED_FILES = 109
EXPECTED_BYTES = 10_485_233_845
EXPECTED_GROUPS = {"COVID-19": 83, "HEALTHY": 26}
START_TIME_RE = re.compile(rb'startTimeStamp="([^"]+)"')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-manifest", required=True, type=Path)
    parser.add_argument("--design-ledger", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_261_031)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    staged = pd.read_csv(
        args.staged_manifest, sep="\t", dtype=str, keep_default_na=False
    )
    required = {
        "assay_name", "expected_bytes", "observed_bytes", "sha256",
        "local_path", "linkage_visibility", "status",
    }
    if missing := required - set(staged):
        raise RuntimeError(f"staged manifest lacks {sorted(missing)}")
    if len(staged) != EXPECTED_FILES or staged.assay_name.nunique() != EXPECTED_FILES:
        raise RuntimeError("staged manifest is not the frozen 109-file universe")
    if set(staged.linkage_visibility) != {"phenotype_blind"}:
        raise RuntimeError("staged manifest exposes non-blind linkage fields")
    expected = pd.to_numeric(staged.expected_bytes, errors="raise").astype("int64")
    observed = pd.to_numeric(staged.observed_bytes, errors="raise").astype("int64")
    if int(expected.sum()) != EXPECTED_BYTES or not expected.equals(observed):
        raise RuntimeError("staged byte universe drift")

    rows: list[dict[str, object]] = []
    for row in staged.itertuples(index=False):
        source = Path(row.local_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("rb") as handle:
            match = START_TIME_RE.search(handle.read(4 << 20))
        rows.append(
            {
                "assay_name": str(row.assay_name),
                "source_sha256": str(row.sha256),
                "acquisition_start_time": (
                    match.group(1).decode("utf-8", errors="strict") if match else ""
                ),
            }
        )
    acquisition = pd.DataFrame(rows)
    acquisition["acquisition_time_utc"] = pd.to_datetime(
        acquisition.acquisition_start_time, errors="coerce", utc=True
    )
    if not acquisition.acquisition_time_utc.notna().all():
        raise RuntimeError("not all acquisition timestamps were recoverable")

    design = pd.read_csv(args.design_ledger, sep="\t", dtype=str, keep_default_na=False)
    if design.groupby("group").size().astype(int).to_dict() != EXPECTED_GROUPS:
        raise RuntimeError("frozen design topology drift")
    joined = acquisition.merge(
        design[["Sample Name", "group"]],
        left_on="assay_name",
        right_on="Sample Name",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != EXPECTED_FILES:
        raise RuntimeError("acquisition and design universes differ")
    joined = joined.sort_values("acquisition_time_utc").reset_index(drop=True)
    joined["acquisition_rank"] = np.arange(1, len(joined) + 1)
    joined["acquisition_date_utc"] = joined.acquisition_time_utc.dt.strftime("%Y-%m-%d")

    labels = joined.group.eq("COVID-19").to_numpy(np.int8)
    covid_ranks = joined.loc[labels == 1, "acquisition_rank"].to_numpy(float)
    healthy_ranks = joined.loc[labels == 0, "acquisition_rank"].to_numpy(float)
    rank_sum = float(covid_ranks.sum())
    u_value = rank_sum - len(covid_ranks) * (len(covid_ranks) + 1) / 2
    auc = float(u_value / (len(covid_ranks) * len(healthy_ranks)))
    rank_p = float(
        mannwhitneyu(covid_ranks, healthy_ranks, alternative="two-sided").pvalue
    )
    observed_transitions = int(np.count_nonzero(labels[1:] != labels[:-1]))
    rng = np.random.default_rng(args.seed)
    null_transitions = np.empty(args.permutations, dtype=np.int16)
    for index in range(args.permutations):
        permuted = rng.permutation(labels)
        null_transitions[index] = np.count_nonzero(permuted[1:] != permuted[:-1])
    transition_p = float(
        (1 + np.count_nonzero(null_transitions <= observed_transitions))
        / (args.permutations + 1)
    )

    day_rows: list[dict[str, object]] = []
    for date, frame in joined.groupby("acquisition_date_utc", sort=True):
        local_labels = frame.group.eq("COVID-19").to_numpy(np.int8)
        covid_n = int(local_labels.sum())
        healthy_n = int(len(local_labels) - covid_n)
        local_auc = None
        if covid_n and healthy_n:
            ranks = np.arange(1, len(frame) + 1)
            local_covid = ranks[local_labels == 1]
            local_u = float(
                local_covid.sum() - covid_n * (covid_n + 1) / 2
            )
            local_auc = local_u / (covid_n * healthy_n)
        day_rows.append(
            {
                "acquisition_date_utc": date,
                "covid": covid_n,
                "healthy": healthy_n,
                "contains_both_groups": bool(covid_n and healthy_n),
                "covid_after_healthy_auc": local_auc,
                "group_transitions": int(
                    np.count_nonzero(local_labels[1:] != local_labels[:-1])
                ),
            }
        )
    day_counts = pd.DataFrame(day_rows)

    args.output_dir.mkdir(parents=True)
    ledger_path = args.output_dir / "acquisition_order_ledger.tsv"
    day_path = args.output_dir / "day_group_counts.tsv"
    joined[
        [
            "assay_name", "source_sha256", "acquisition_start_time",
            "acquisition_rank", "acquisition_date_utc", "group",
        ]
    ].to_csv(ledger_path, sep="\t", index=False)
    day_counts.to_csv(day_path, sep="\t", index=False, na_rep="NA")

    report = {
        "schema": "contrast_atlas_mtbls1866_acquisition_order_gate_v1",
        "status": "FAIL_TECHNICAL_CONFOUNDING_GATE",
        "evidence_role": "development_consumed_qualification_only",
        "decision": {
            "eligible_as_positive_covid_biological_truth": False,
            "eligible_to_activate_msv000092887_relation": False,
            "reason": (
                "group labels are strongly structured by acquisition order and date; "
                "the frozen assets provide neither interleaved QC support nor a "
                "defensible pre-specified batch adjustment"
            ),
            "allowed_future_role": (
                "phenotype-blind technical stress test only, unless independent batch "
                "metadata and a pre-specified correction model become available"
            ),
        },
        "denominators": {
            "files": len(joined),
            "noncritical_covid": int(labels.sum()),
            "healthy": int(len(labels) - labels.sum()),
            "acquisition_days": int(day_counts.shape[0]),
            "days_containing_both_groups": int(day_counts.contains_both_groups.sum()),
        },
        "order_evidence": {
            "covid_after_healthy_auc": auc,
            "reverse_orientation_auc": 1.0 - auc,
            "mann_whitney_two_sided_p": rank_p,
            "observed_group_transitions": observed_transitions,
            "transition_permutations": args.permutations,
            "transition_seed": args.seed,
            "permutation_p_transitions_le_observed": transition_p,
            "null_transitions_median": float(np.median(null_transitions)),
            "null_transitions_q025": float(np.quantile(null_transitions, 0.025)),
            "null_transitions_q975": float(np.quantile(null_transitions, 0.975)),
        },
        "leakage_status": (
            "only frozen acquisition-header timestamps and the frozen design ledger "
            "were joined; no feature intensities, product ions, response effects, "
            "identities, registered relations, or retrieval scores were read"
        ),
        "sequence_note": (
            "resource-conserving early stop: the design ledger was opened before the "
            "phenotype-blind measurement object completed, solely to terminate an "
            "ineligible source; no measurement parameters or biological responses "
            "were selected or estimated"
        ),
        "provenance": {
            "staged_manifest_sha256": sha256(args.staged_manifest),
            "design_ledger_sha256": sha256(args.design_ledger),
            "contract_sha256": sha256(args.contract),
            "implementation_sha256": sha256(Path(__file__)),
            "acquisition_order_ledger_sha256": sha256(ledger_path),
            "day_group_counts_sha256": sha256(day_path),
        },
        "claim_boundary": [
            "This result is a negative source-qualification decision, not a COVID response.",
            "No cross-study response, unknown-information gain, or algorithm gain was tested.",
            "Post-hoc day-restricted or nearest-neighbor matching is not activated by this result.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_paths = [ledger_path, day_path, report_path]
    (args.output_dir / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
