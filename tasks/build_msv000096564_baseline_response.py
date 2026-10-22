#!/usr/bin/env python3
"""Build the frozen pre-intervention MSV000096564 IBD stool response."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from build_public_fbmn_response_panel import mgf_feature_ids, quantify_response


BASE = "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?"
T1 = re.compile(r"^15663_800805_(\d+)_T1_fecal_(\d+)\.mzML$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_asset(descriptor: str, expected_bytes: int, output: Path) -> None:
    if output.exists():
        if output.stat().st_size != expected_bytes:
            raise RuntimeError(f"size drift for {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(
        BASE + urllib.parse.urlencode({"file": descriptor}),
        headers={"User-Agent": "DreaMS-Contrast-Atlas/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
            while block := response.read(1 << 20):
                handle.write(block)
        if partial.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"downloaded size drift for {descriptor}: {partial.stat().st_size} != {expected_bytes}"
            )
        partial.replace(output)
    finally:
        if partial.exists():
            partial.unlink()


def participant_states(metadata: pd.DataFrame) -> dict[int, str]:
    frame = metadata.loc[metadata.disease_state.notna(), ["host_subject_id", "disease_state"]].copy()
    frame["subject_number"] = pd.to_numeric(
        frame.host_subject_id.astype(str).str.extract(r"(\d+)$")[0], errors="raise"
    ).astype(int)
    state_sets = frame.groupby("subject_number").disease_state.apply(
        lambda values: set(values.astype(str))
    )
    if state_sets.map(len).gt(1).any():
        raise RuntimeError("a participant has conflicting nonempty disease states")
    mapping: dict[int, str] = {}
    for subject, states in state_sets.items():
        state = next(iter(states))
        if state == "control":
            mapping[int(subject)] = "control"
        elif state in {"ibd_active", "ibd_remission"}:
            mapping[int(subject)] = "ibd"
    return mapping


def baseline_design(columns: list[str], metadata: pd.DataFrame, contract: dict) -> pd.DataFrame:
    states = participant_states(metadata)
    rows: list[dict[str, str | int]] = []
    for column in columns:
        filename = str(column)
        for suffix in (" Peak area", " peak area", " Area", " area"):
            if filename.endswith(suffix):
                filename = filename[:-len(suffix)]
                break
        match = T1.fullmatch(filename)
        if not match:
            continue
        subject = int(match.group(1))
        if subject not in states:
            raise RuntimeError(f"T1 subject {subject} lacks a frozen disease state")
        rows.append({
            "filename": filename,
            "biological_unit_id": f"800805-{subject:03d}",
            "subject_number": subject,
            "technical_replicate": int(match.group(2)),
            "level": states[subject],
        })
    design = pd.DataFrame(rows).sort_values(
        ["biological_unit_id", "technical_replicate"]
    ).reset_index(drop=True)
    if design.filename.duplicated().any():
        raise RuntimeError("duplicate T1 filenames")
    replicate_counts = design.groupby("biological_unit_id").size()
    expected_replicates = int(contract["contrast"]["expected_technical_files_per_unit"])
    if not replicate_counts.eq(expected_replicates).all():
        raise RuntimeError(f"T1 technical replicate drift: {replicate_counts.value_counts().to_dict()}")
    observed = design.drop_duplicates("biological_unit_id").groupby("level").size().to_dict()
    expected = contract["contrast"]["expected_biological_units"]
    if observed != expected:
        raise RuntimeError(f"biological-unit drift: {observed} != {expected}")
    return design


def identity_partition(quant: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    feature_id = quant["row ID"].astype(str)
    provisional = annotations.rename(columns={
        "id": "entity_id", "compound_name": "provisional_name", "score": "provisional_score"
    })[["entity_id", "provisional_name", "provisional_score", "method"]].copy()
    provisional["entity_id"] = provisional.entity_id.astype(str)
    result = pd.DataFrame({
        "entity_id": feature_id,
        "precursor_mz": quant["row m/z"],
        "retention_time": quant["row retention time"],
        "identity_status": "unresolved_entity",
        "identity_partition_rule": "no_complete_strict_direct_library_search_deposited",
    })
    result = result.merge(provisional, on="entity_id", how="left", validate="one_to_one")
    result["source_annotation_status"] = np.where(
        result.provisional_name.notna(), "provisional_mzmine_matchedlipid", "none"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    assets: dict[str, Path] = {}
    for name, spec in contract["assets"].items():
        suffix = Path(spec["remote_descriptor"]).suffix
        path = args.source_root / f"{name}{suffix}"
        stage_asset(spec["remote_descriptor"], int(spec["bytes"]), path)
        assets[name] = path

    quant = pd.read_csv(assets["quantification"])
    metadata = pd.read_csv(assets["metadata"], encoding="utf-8-sig")
    annotations = pd.read_csv(assets["mzmine_annotations"])
    design = baseline_design(list(quant.columns), metadata, contract)
    mgf_ids, mgf_blocks, mgf_without_id = mgf_feature_ids(assets["mgf"])
    identity = identity_partition(quant, annotations)
    contrast = {
        **contract["contrast"],
        "level_coefficients": contract["contrast"]["level_coefficients"],
    }
    effects, normalisation, summary = quantify_response(
        quant, design[["filename", "biological_unit_id", "level"]], contrast,
        mgf_ids, identity,
        float(contract["contrast"]["minimum_prevalence"]),
    )

    object_dir = args.output_dir / contract["contrast"]["object_id"]
    object_dir.mkdir(parents=True)
    effects.to_csv(
        object_dir / "response_object.tsv.gz", sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    normalisation.to_csv(object_dir / "sample_normalization.tsv", sep="\t", index=False)
    design.to_csv(args.output_dir / "design.tsv", sep="\t", index=False)
    identity.to_csv(
        args.output_dir / "identity_partition.tsv.gz", sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    report = {
        "schema": "msv000096564_baseline_ibd_response_result_v1",
        "status": "DEVELOPMENT_CONSUMED_RESPONSE_OBJECTS_BUILT_NO_CROSS_STUDY_RESULT",
        "study_id": contract["contrast"]["study_id"],
        "source_family": contract["source_family"],
        "contrast_id": contract["contrast"]["object_id"],
        "estimand": contract["contrast"]["estimand"],
        "denominators": {
            **summary,
            "design_files": int(len(design)),
            "mgf_blocks": int(mgf_blocks),
            "mgf_blocks_without_id": int(mgf_without_id),
            "identity_partition_library_supported": 0,
            "identity_partition_unresolved": int(len(identity)),
            "provisional_mzmine_matchedlipid": int(identity.provisional_name.notna().sum()),
        },
        "measurement_contract": {
            "timepoint": "T1 before the 14-day Mediterranean-diet intervention",
            "normalization": "positive-only log2 then per-technical-file median centering",
            "technical_replicates": "three T1 files median-collapsed within each participant",
            "effect": "IBD active-or-remission minus control, independent-participant Welch SE",
            "missingness": "zero/nonpositive remains missing; no imputation",
        },
        "provenance": {
            "contract_sha256": sha256(args.contract),
            **{f"{name}_sha256": sha256(path) for name, path in assets.items()},
        },
        "claim_boundary": [
            *contract["design_boundary"],
            *contract["identity_boundary"],
            "This response object is development/consumed and is not cross-study biological validation.",
        ],
    }
    (object_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
