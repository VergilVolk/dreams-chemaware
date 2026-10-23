#!/usr/bin/env python3
"""Build drug-exposure-specific microbiome response objects for MSV000095331.

The published experiment incubated four drug mixtures with four 72-hour
synthetic gut communities and an abiotic BHI control, each in triplicate.
For each feature and mixture the primary estimand is the endpoint interaction

    mean_g[(community_g,drug,T72 - community_g,no-drug,T72)]
      - (BHI,drug,T72 - BHI,no-drug,T72)

with equal weight for the four communities.  It removes the general 72-hour
community and medium response using the deposited no-drug endpoint controls.
Replicate indices are kept paired, yielding three response replicates.  The
older time difference-in-differences estimand is emitted only as a named
sensitivity analysis; it is not called drug-specific because it does not
remove no-drug community growth.

Only the four mixtures documented in the paper are used.  The deposited
``drug_mix5`` samples are deliberately excluded because they are not part of
the published mixture definition.  Drug names are carried only in an output
registry; they do not enter feature filtering or effect estimation.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_chemprop2_response_reproducibility_m1 import median_center, robust_log2
from run_contrast_atlas_sparse_fragment_measure_baseline import (
    iter_mgf,
    normalise_entity_id,
    sparse_spectrum,
    spectrum_id,
)


STATUS = "MSV000095331_DRUG_ENDPOINT_INTERACTION_RESPONSES_COMPLETE"
GNPS_TASK = "401db749414546aeb37222c752294861"
PRIMARY_GROUPS = (
    "Group_3_medium_slow",
    "Group_4_slow",
    "Full_syncom",
    "Syncom_red",
)
MIXTURE_DRUGS = {
    1: ("sertraline", "quinidine", "venlafaxine"),
    2: ("cetirizine", "losartan", "quetiapine", "ritonavir"),
    3: ("darunavir", "quinine", "atorvastatin", "omeprazole"),
    4: ("loratadine", "terbinafine", "trazodone", "abacavir"),
}
CHEMPROP_OVERLAP = {
    1: ("sertraline",),
    2: (),
    3: ("omeprazole",),
    4: ("loratadine",),
}
SAMPLE_PATTERN = re.compile(
    r"^(Group_3_medium_slow|Group_4_slow|Full_syncom|Syncom_red|BHI)"
    r"_T(0|72)_drug_mix_?([1-4])_([1-3])\.mzML$",
    re.IGNORECASE,
)
NO_DRUG_GROUP_PATTERN = re.compile(
    r"^(Group_3_medium_slow|Group_4_slow|Full_syncom|Syncom_red)"
    r"_T72_([1-3])\.mzML$",
    re.IGNORECASE,
)
NO_DRUG_BHI_PATTERN = re.compile(r"^BHI_([1-3])\.mzML$", re.IGNORECASE)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "DreaMS-research/1.0"})
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(request, timeout=180) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
    temporary.replace(path)


def acquire(root: Path) -> dict[str, Path]:
    paths = {
        "metadata": root / "Metadata_13072024.csv",
        "quantification": root / "quantification.csv",
        "mgf": root / "consensus.mgf",
        "library": root / "library_matches.tsv",
    }
    base = "https://gnps2.org"
    urls = {
        "metadata": (
            "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?"
            "file=f.MSV000095331%2Fmetadata%2FMetadata_13072024.csv&forceDownload=true"
        ),
        "quantification": (
            f"{base}/result?task={GNPS_TASK}&viewname=quantificationdownload&"
            "resultdisplay_type=task"
        ),
        "mgf": (
            f"{base}/result?task={GNPS_TASK}&viewname=specms&resultdisplay_type=task"
        ),
        "library": (
            f"{base}/resultfile?task={GNPS_TASK}&"
            "file=nf_output/library/merged_results_with_gnps.tsv"
        ),
    }
    for name, path in paths.items():
        download(urls[name], path)
    return paths


def sample_table(metadata: pd.DataFrame, quant_columns: list[str]) -> pd.DataFrame:
    available = {column[:-10]: column for column in quant_columns if column.endswith(" Peak area")}
    rows = []
    for filename in metadata["filename"].astype(str):
        match = SAMPLE_PATTERN.match(filename)
        if match is not None:
            if filename not in available:
                raise ValueError(f"primary sample absent from quantification: {filename}")
            group, time, mixture, replicate = match.groups()
            canonical_group = next(
                value for value in (*PRIMARY_GROUPS, "BHI")
                if value.lower() == group.lower()
            )
            rows.append({
                "filename": filename,
                "column": available[filename],
                "group": canonical_group,
                "time": int(time),
                "mixture": int(mixture),
                "exposure": f"drug_mix_{int(mixture)}",
                "replicate": int(replicate),
            })
            continue
        group_control = NO_DRUG_GROUP_PATTERN.match(filename)
        bhi_control = NO_DRUG_BHI_PATTERN.match(filename)
        if group_control is None and bhi_control is None:
            continue
        if filename not in available:
            raise ValueError(f"no-drug endpoint absent from quantification: {filename}")
        if group_control is not None:
            group, replicate = group_control.groups()
            canonical_group = next(
                value for value in PRIMARY_GROUPS if value.lower() == group.lower()
            )
        else:
            canonical_group, replicate = "BHI", bhi_control.group(1)
        rows.append({
            "filename": filename,
            "column": available[filename],
            "group": canonical_group,
            "time": 72,
            "mixture": 0,
            "exposure": "no_drug",
            "replicate": int(replicate),
        })
    output = pd.DataFrame(rows)
    drug_timecourse = len((*PRIMARY_GROUPS, "BHI")) * 2 * len(MIXTURE_DRUGS) * 3
    no_drug_endpoint = len((*PRIMARY_GROUPS, "BHI")) * 3
    expected = drug_timecourse + no_drug_endpoint
    if len(output) != expected:
        raise ValueError(f"expected {expected} analysis samples, found {len(output)}")
    key = ["group", "time", "mixture", "replicate"]
    if output.duplicated(key).any():
        raise ValueError("duplicate primary experimental cells")
    return output.sort_values(key).reset_index(drop=True)


def nonempty_spectra(path: Path) -> tuple[set[str], int]:
    identifiers: set[str] = set()
    total = 0
    for metadata, peaks in iter_mgf(path):
        total += 1
        identifier = normalise_entity_id(spectrum_id(metadata))
        if sparse_spectrum(peaks, 0.1)[0]:
            if identifier in identifiers:
                raise ValueError(f"duplicate nonempty spectrum ID {identifier}")
            identifiers.add(identifier)
    return identifiers, total


def endpoint_interaction_replicates(
    values: np.ndarray,
    lookup: dict[tuple[str, int, int, int], int],
    mixture: int,
    group: str | None,
) -> np.ndarray:
    """Drug-versus-no-drug endpoint difference, net of the abiotic difference."""
    rows = []
    groups = PRIMARY_GROUPS if group is None else (group,)
    for replicate in (1, 2, 3):
        microbial = []
        for current in groups:
            drug = values[lookup[(current, 72, mixture, replicate)]]
            no_drug = values[lookup[(current, 72, 0, replicate)]]
            microbial.append(drug - no_drug)
        abiotic = (
            values[lookup[("BHI", 72, mixture, replicate)]]
            - values[lookup[("BHI", 72, 0, replicate)]]
        )
        rows.append(np.mean(microbial, axis=0) - abiotic)
    return np.stack(rows)


def time_did_replicates(
    values: np.ndarray,
    lookup: dict[tuple[str, int, int, int], int],
    mixture: int,
    group: str | None,
) -> np.ndarray:
    """Historical time difference-in-differences sensitivity estimand."""
    rows = []
    groups = PRIMARY_GROUPS if group is None else (group,)
    for replicate in (1, 2, 3):
        microbial = []
        for current in groups:
            endpoint = values[lookup[(current, 72, mixture, replicate)]]
            baseline = values[lookup[(current, 0, mixture, replicate)]]
            microbial.append(endpoint - baseline)
        abiotic = (
            values[lookup[("BHI", 72, mixture, replicate)]]
            - values[lookup[("BHI", 0, mixture, replicate)]]
        )
        rows.append(np.mean(microbial, axis=0) - abiotic)
    return np.stack(rows)


def summarize_response(
    replicates: np.ndarray,
    identifiers: np.ndarray,
    selected_indices: np.ndarray,
    prevalence: np.ndarray,
    contrast_id: str,
    model: str,
) -> pd.DataFrame:
    selected = replicates[:, selected_indices]
    se = selected.std(axis=0, ddof=1) / np.sqrt(selected.shape[0])
    positive = se[np.isfinite(se) & (se > 0)]
    floor = max(float(np.quantile(positive, 0.01)) if len(positive) else 0.0, 1e-8)
    return pd.DataFrame({
        "contrast_id": contrast_id,
        "entity_id": identifiers[selected_indices],
        "beta": selected.mean(axis=0),
        "se": np.maximum(se, floor),
        "prevalence": prevalence[selected_indices],
        "effect_model": model,
        "response_replicates": selected.shape[0],
        "degrees_of_freedom": selected.shape[0] - 1,
    })


def endpoint_response_eligibility(
    raw_values: np.ndarray,
    samples: pd.DataFrame,
    mixture: int,
    groups: tuple[str, ...],
    ms2_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Require real detection in the cells used by one response.

    A feature is eligible only when at least two of three replicates are
    detected in one relevant group-by-time cell.  This prevents median
    centering of all-zero measurements from manufacturing a response with
    near-zero variance.
    """
    relevant_groups = (*groups, "BHI")
    relevant = (
        samples["time"].eq(72)
        & samples["mixture"].isin([0, mixture])
        & samples["group"].isin(relevant_groups)
    )
    positions = np.flatnonzero(relevant.to_numpy())
    prevalence = (raw_values[positions] > 0).mean(axis=0)
    maximum_cell = np.zeros(raw_values.shape[1], dtype=float)
    for group in relevant_groups:
        for current_mixture in (0, mixture):
            cell = (
                samples["mixture"].eq(current_mixture)
                & samples["group"].eq(group)
                & samples["time"].eq(72)
            )
            cell_positions = np.flatnonzero(cell.to_numpy())
            if len(cell_positions) != 3:
                raise ValueError(
                    f"expected three endpoint replicates for {group}/{current_mixture}"
                )
            maximum_cell = np.maximum(
                maximum_cell, (raw_values[cell_positions] > 0).mean(axis=0)
            )
    observed = maximum_cell >= (2.0 / 3.0)
    selected = ms2_indices[observed[ms2_indices]]
    return selected, prevalence


def time_did_response_eligibility(
    raw_values: np.ndarray,
    samples: pd.DataFrame,
    mixture: int,
    groups: tuple[str, ...],
    ms2_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    relevant_groups = (*groups, "BHI")
    relevant = samples["mixture"].eq(mixture) & samples["group"].isin(relevant_groups)
    positions = np.flatnonzero(relevant.to_numpy())
    prevalence = (raw_values[positions] > 0).mean(axis=0)
    maximum_cell = np.zeros(raw_values.shape[1], dtype=float)
    for current_group in relevant_groups:
        for time in (0, 72):
            cell = (
                samples["mixture"].eq(mixture)
                & samples["group"].eq(current_group)
                & samples["time"].eq(time)
            )
            cell_positions = np.flatnonzero(cell.to_numpy())
            if len(cell_positions) != 3:
                raise ValueError(
                    f"expected three time-course replicates for {current_group}/{mixture}/{time}"
                )
            maximum_cell = np.maximum(
                maximum_cell, (raw_values[cell_positions] > 0).mean(axis=0)
            )
    selected = ms2_indices[maximum_cell[ms2_indices] >= (2.0 / 3.0)]
    return selected, prevalence


def identity_partition(library_path: Path, eligible_ids: set[str]) -> pd.DataFrame:
    table = pd.read_csv(library_path, sep="\t", low_memory=False)
    table["scan"] = pd.to_numeric(table["#Scan#"], errors="coerce")
    table["score"] = pd.to_numeric(table["MQScore"], errors="coerce")
    table["peaks"] = pd.to_numeric(table["SharedPeaks"], errors="coerce")
    key = table["InChIKey-Planar"].astype(str).str.strip()
    supported = table.loc[
        table["scan"].notna()
        & table["score"].ge(0.9)
        & table["peaks"].ge(5)
        & ~key.isin({"", "N/A", "nan", "None"})
    ].sort_values(["scan", "score"], ascending=[True, False]).drop_duplicates("scan")
    known = {normalise_entity_id(value) for value in supported["scan"].astype(int)}
    rows = pd.DataFrame({"entity_id": sorted(eligible_ids)})
    rows["identity_status"] = np.where(
        rows["entity_id"].isin(known), "library_supported", "unresolved_entity"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("data/reference/msv000095331_drug_microbiome_20261023"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    paths = {
        "metadata": args.source_root / "Metadata_13072024.csv",
        "quantification": args.source_root / "quantification.csv",
        "mgf": args.source_root / "consensus.mgf",
        "library": args.source_root / "library_matches.tsv",
    }
    if not args.skip_download:
        paths = acquire(args.source_root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing source files: " + ", ".join(missing))

    metadata = pd.read_csv(paths["metadata"])
    quant = pd.read_csv(paths["quantification"])
    samples = sample_table(metadata, list(quant.columns))
    columns = samples["column"].tolist()
    raw_values = quant[columns].to_numpy(float).T
    values = median_center(robust_log2(raw_values))
    lookup = {
        (row.group, row.time, row.mixture, row.replicate): index
        for index, row in enumerate(samples.itertuples(index=False))
    }

    identifiers = np.asarray([normalise_entity_id(value) for value in quant["row ID"]])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("normalized quantification IDs are not unique")
    nonempty, mgf_total = nonempty_spectra(paths["mgf"])
    eligible = np.asarray([index for index, value in enumerate(identifiers) if value in nonempty])
    effects = []
    sensitivity_effects = []
    for mixture in MIXTURE_DRUGS:
        pooled = endpoint_interaction_replicates(values, lookup, mixture, None)
        pooled_indices, pooled_prevalence = endpoint_response_eligibility(
            raw_values, samples, mixture, PRIMARY_GROUPS, eligible,
        )
        effects.append(summarize_response(
            pooled, identifiers, pooled_indices, pooled_prevalence,
            f"drug_mix_{mixture}_pooled_microbiota",
            "equal-community mean of paired endpoint (microbial drug-no-drug)-(BHI drug-no-drug); n=3",
        ))
        pooled_sensitivity = time_did_replicates(values, lookup, mixture, None)
        pooled_sensitivity_indices, pooled_sensitivity_prevalence = (
            time_did_response_eligibility(
                raw_values, samples, mixture, PRIMARY_GROUPS, eligible,
            )
        )
        sensitivity_effects.append(summarize_response(
            pooled_sensitivity, identifiers, pooled_sensitivity_indices,
            pooled_sensitivity_prevalence,
            f"drug_mix_{mixture}_pooled_microbiota",
            "SENSITIVITY ONLY: equal-community mean of paired (microbial T72-T0)-(BHI T72-T0); n=3",
        ))
        for group in PRIMARY_GROUPS:
            group_response = endpoint_interaction_replicates(values, lookup, mixture, group)
            group_indices, group_prevalence = endpoint_response_eligibility(
                raw_values, samples, mixture, (group,), eligible,
            )
            effects.append(summarize_response(
                group_response, identifiers, group_indices, group_prevalence,
                f"drug_mix_{mixture}_{group.lower()}",
                "paired endpoint (microbial drug-no-drug)-(BHI drug-no-drug); n=3",
            ))
            group_sensitivity = time_did_replicates(values, lookup, mixture, group)
            group_sensitivity_indices, group_sensitivity_prevalence = (
                time_did_response_eligibility(
                    raw_values, samples, mixture, (group,), eligible,
                )
            )
            sensitivity_effects.append(summarize_response(
                group_sensitivity, identifiers, group_sensitivity_indices,
                group_sensitivity_prevalence,
                f"drug_mix_{mixture}_{group.lower()}",
                "SENSITIVITY ONLY: paired (microbial T72-T0)-(BHI T72-T0); n=3",
            ))
    effect_table = pd.concat(effects, ignore_index=True)
    sensitivity_table = pd.concat(sensitivity_effects, ignore_index=True)
    if not np.isfinite(effect_table[["beta", "se", "prevalence"]].to_numpy(float)).all():
        raise ValueError("non-finite response statistics")
    if not np.isfinite(
        sensitivity_table[["beta", "se", "prevalence"]].to_numpy(float)
    ).all():
        raise ValueError("non-finite sensitivity response statistics")

    response_entity_ids = set(effect_table["entity_id"].astype(str))
    identity = identity_partition(paths["library"], response_entity_ids)
    mixtures = []
    for mixture, drugs in MIXTURE_DRUGS.items():
        for drug in drugs:
            mixtures.append({
                "mixture": mixture,
                "drug": drug,
                "chemprop_overlap": drug in CHEMPROP_OVERLAP[mixture],
                "source": "published Methods; DOI 10.1038/s41467-025-65993-5",
            })
    mixture_table = pd.DataFrame(mixtures)

    args.output_dir.mkdir(parents=True)
    effect_path = args.output_dir / "effects.tsv"
    sensitivity_path = args.output_dir / "time_did_sensitivity_effects.tsv"
    identity_path = args.output_dir / "source_native_identity.tsv"
    sample_path = args.output_dir / "primary_sample_design.tsv"
    mixture_path = args.output_dir / "mixture_registry.tsv"
    effect_table.to_csv(effect_path, sep="\t", index=False)
    sensitivity_table.to_csv(sensitivity_path, sep="\t", index=False)
    identity.to_csv(identity_path, sep="\t", index=False)
    samples.to_csv(sample_path, sep="\t", index=False)
    mixture_table.to_csv(mixture_path, sep="\t", index=False)

    report = {
        "schema": "msv000095331_drug_endpoint_interaction_responses_v2",
        "status": STATUS,
        "study_id": "MSV000095331",
        "data_role": "new_source_development_not_sealed_confirmation",
        "gnps_task": GNPS_TASK,
        "scientific_question": (
            "Which LC-MS/MS entities show a drug-mixture-associated 72-hour response "
            "in microbial communities beyond both the no-drug community endpoint and "
            "the matched abiotic medium response?"
        ),
        "primary_estimand": (
            "equal-weight mean across four communities of "
            "(microbial drug T72 - microbial no-drug T72), minus "
            "(BHI drug T72 - BHI no-drug T72)"
        ),
        "sensitivity_estimand": (
            "historical (microbial T72-T0)-(BHI T72-T0), written separately and not "
            "interpreted as drug-specific because no-drug community growth is absent"
        ),
        "denominators": {
            "deposited_metadata_rows": int(len(metadata)),
            "analysis_samples": int(len(samples)),
            "drug_timecourse_samples": int(samples["mixture"].gt(0).sum()),
            "no_drug_endpoint_samples": int(samples["mixture"].eq(0).sum()),
            "primary_communities": len(PRIMARY_GROUPS),
            "published_drug_mixtures": len(MIXTURE_DRUGS),
            "response_contrasts": int(effect_table["contrast_id"].nunique()),
            "quantification_features": int(len(quant)),
            "mgf_spectra_total": int(mgf_total),
            "nonempty_ms2_linked_features": int(len(eligible)),
            "response_eligible_entity_union": int(len(response_entity_ids)),
            "response_rows": int(len(effect_table)),
            "sensitivity_response_rows": int(len(sensitivity_table)),
            "library_supported_entities": int(identity.identity_status.eq("library_supported").sum()),
            "unresolved_entities": int(identity.identity_status.eq("unresolved_entity").sum()),
            "chemprop_overlap_drugs": int(mixture_table.chemprop_overlap.sum()),
        },
        "prespecified_biological_truth": {
            "increased_microbiota_associated_products": [
                "ritonavir", "atorvastatin", "abacavir", "quetiapine"
            ],
            "abiotic_decreasing_product": "omeprazole m/z 330.127",
            "basis": "published before this cross-project response analysis",
        },
        "exclusions": {
            "drug_mix5": "not one of the four mixtures defined in the published Methods",
            "48_hour_groups": (
                "fast and medium-fast communities excluded from the primary common-time "
                "estimand because the matched abiotic endpoint is 72 hours"
            ),
        },
        "claim_boundary": [
            "MSV000095331 is a new experiment/project but shares the Dorrestein research ecosystem with the ChemProp sources; it is not independent-laboratory confirmation.",
            "The primary endpoint interaction removes the deposited no-drug endpoint and abiotic response; it does not establish a single-drug causal effect inside a multi-drug mixture.",
            "A mixture response cannot establish which co-administered drug caused a feature response.",
            "Three paired response replicates provide only two degrees of freedom; normal-approximation p-values are prohibited and feature-wise significance is exploratory even with a t reference.",
            "Library-supported means a stringent public-library match, not level-1 identification.",
            "The paper's named drug products are validation targets and do not enter feature filtering or beta/SE estimation.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
