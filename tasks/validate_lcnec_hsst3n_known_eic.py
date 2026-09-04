"""Validate raw-mzML EIC effects against author-reported HSST3n metabolites."""

from __future__ import annotations

import csv
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quantify_lcnec_hsst3n_dark_eic import classify, quantify_file


def main() -> None:
    zip_path = Path("data/validation/lcnec_zenodo19005638_preflight/MTB22_P073_HSST3n_mzML_public.zip")
    overview_path = Path("data/validation/lcnec_zenodo19005638_preflight/06_MTB22_P073_HSST3n_mzML_overview_v1.txt")
    targets_path = Path("data/validation/lcnec_hsst3n_author_overlap_gate/qualified_family_author_overlap.csv")
    output_dir = Path("data/validation/lcnec_hsst3n_known_eic_validation")

    with targets_path.open("r", encoding="utf-8", newline="") as handle:
        targets = [row for row in csv.DictReader(handle) if row["author_matched"].lower() == "true"]
    targets.sort(key=lambda row: float(row["rt_median_sec"]))
    if len(targets) != 42:
        raise RuntimeError(f"expected 42 author-matched positive controls, found {len(targets)}")
    target_rts = [float(row["rt_median_sec"]) for row in targets]
    overview = pd.read_csv(overview_path, sep="\t")
    study = overview[overview["NOTE"].eq("Study sample")].copy()
    if len(study) != 68:
        raise RuntimeError("expected 68 study injections")

    matrix = np.zeros((len(study), len(targets)), dtype=float)
    with zipfile.ZipFile(zip_path) as archive:
        members = {Path(info.filename).name: info for info in archive.infolist() if info.filename.lower().endswith(".mzml")}
        for index, (_, row) in enumerate(study.iterrows()):
            with archive.open(members[row["mzML_FILE_NAME"]]) as handle:
                area, _maximum, _scans = quantify_file(handle, targets, target_rts, 5.0, 15.0)
            matrix[index] = area / float(row["AMOUNT"])
            print(f"[known EIC] {index + 1}/68 {row['SAMPLE_ID']}", flush=True)

    pair_map: dict[str, dict[str, int]] = {}
    for local_index, (_, row) in enumerate(study.iterrows()):
        pair_map.setdefault(str(row["SAMPLE_CODE"]), {})[str(row["GROUP_CODE"])] = local_index
    output = []
    author_beta = []
    observed_effect = []
    for target_index, target in enumerate(targets):
        tu = np.asarray([matrix[value["TU"], target_index] for value in pair_map.values()])
        ng = np.asarray([matrix[value["NG"], target_index] for value in pair_map.values()])
        positive = np.concatenate((tu[tu > 0], ng[ng > 0]))
        pseudo = float(np.min(positive) / 2) if len(positive) else 1.0
        effect = float(np.mean(np.log2(tu + pseudo) - np.log2(ng + pseudo)))
        beta = float(target.get("author_beta", "nan")) if target.get("author_beta") else math.nan
        # Recover beta by metabolite from Table S2 because the overlap ledger stores identity but not statistics.
        output.append({"family_id": int(target["family_id"]), "author_metabolite": target["author_metabolite"], "observed_log2fc": effect})

    supplement = Path("data/validation/lcnec_zenodo19005638_preflight/article_mmc7.xlsx")
    table = pd.read_excel(supplement, sheet_name="Table S2", header=3).iloc[:, 1:]
    table = table[table["Platform"].astype(str).eq("HSST3n")]
    beta_by_name = table.groupby("Metabolite")["beta"].first().to_dict()
    valid_rows = []
    for row in output:
        beta = beta_by_name.get(row["author_metabolite"], math.nan)
        row["author_beta"] = beta
        row["sign_agrees"] = bool(math.isfinite(beta) and beta * row["observed_log2fc"] > 0)
        if math.isfinite(beta):
            author_beta.append(float(beta))
            observed_effect.append(float(row["observed_log2fc"]))
            valid_rows.append(row)

    rho = float(spearmanr(author_beta, observed_effect).statistic)
    sign_concordance = float(np.mean([row["sign_agrees"] for row in valid_rows]))
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(output_dir / "known_feature_effect_validation.csv", index=False)
    report = {
        "status": "lcnec_hsst3n_known_eic_validation_complete",
        "formal": True,
        "matched_features": len(targets),
        "features_with_author_beta": len(valid_rows),
        "effect_spearman_rho": rho,
        "effect_sign_concordance": sign_concordance,
        "gates": {"spearman_rho_ge_0_5": rho >= 0.5, "sign_concordance_ge_0_8": sign_concordance >= 0.8},
        "pass": rho >= 0.5 and sign_concordance >= 0.8,
        "claim_limit": "Positive-control validation of re-quantification only; it does not validate dark-feature identities.",
    }
    (output_dir / "known_eic_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
