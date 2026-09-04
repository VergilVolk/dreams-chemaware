"""Test patient-level coordination after adding proline/glutamate and Neu5Ac.

All modules are phenotype-selected, so this is exploratory and postselection.
The patient, not a feature or spectrum, is the statistical unit.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/mtbls13729/module_coordination_v1/patient_module_matrix.csv"
AMINO = ROOT / "data/mtbls13729/integrated_biology_ledger_v2/expanded_amino_acid_patient_matrix.csv"
ANCHORS = ROOT / "data/mtbls13729/integrated_biology_ledger_v2/new_anchor_patient_deltas.csv"
OUT = ROOT / "data/mtbls13729/module_coordination_v2"
SEED = 20260830
PERMUTATIONS = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def permutation_p(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    x_rank = rankdata(x).astype(float)
    y_rank = rankdata(y).astype(float)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = float(np.linalg.norm(x_rank) * np.linalg.norm(y_rank))
    observed = abs(float(np.dot(x_rank, y_rank) / denominator))
    exceed = 0
    done = 0
    while done < PERMUTATIONS:
        batch = min(10_000, PERMUTATIONS - done)
        order = np.argsort(rng.random((batch, len(y_rank))), axis=1)
        values = np.abs((y_rank[order] @ x_rank) / denominator)
        exceed += int(np.sum(values >= observed - 1e-12))
        done += batch
    return (exceed + 1) / (PERMUTATIONS + 1)


def main() -> None:
    for path in [OLD, AMINO, ANCHORS]:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)

    old = pd.read_csv(OLD).set_index("patient")
    amino = pd.read_csv(AMINO).set_index("patient")
    anchors = pd.read_csv(ANCHORS)
    neu5ac = (
        anchors.loc[anchors.feature_id.eq(703), ["patient", "log2_tumor_normal"]]
        .set_index("patient")
        .rename(columns={"log2_tumor_normal": "neu5ac_pool"})
    )
    matrix = pd.concat(
        [
            old[[
                "acetylated_polyamine_mta_turnover",
                "long_chain_acylcarnitine_accumulation",
                "purine_modified_nucleoside_pool",
            ]],
            amino[["module_mean_log2fc"]].rename(columns={"module_mean_log2fc": "expanded_amino_acid_pool"}),
            neu5ac,
        ],
        axis=1,
        join="outer",
    ).sort_index()
    if len(matrix) != 10:
        raise RuntimeError(f"expected 10 Rmu patients, found {len(matrix)}")
    matrix.to_csv(OUT / "patient_module_matrix_v2.csv")

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for left, right in combinations(matrix.columns, 2):
        pair = matrix[[left, right]].dropna()
        rho = float(spearmanr(pair[left], pair[right]).statistic)
        rows.append(
            {
                "module_left": left,
                "module_right": right,
                "patients": int(len(pair)),
                "spearman_rho": rho,
                "two_sided_permutation_p": permutation_p(
                    pair[left].to_numpy(float), pair[right].to_numpy(float), rng
                ),
            }
        )
    results = pd.DataFrame(rows)
    results["bh_q_across_all_module_pairs"] = bh(results.two_sided_permutation_p.to_numpy(float))
    results.to_csv(OUT / "module_pairwise_coordination_v2.csv", index=False)

    report = {
        "status": "mtbls13729_module_coordination_v2_complete",
        "formal": False,
        "patients": int(len(matrix)),
        "modules": matrix.columns.tolist(),
        "pairwise": results.to_dict(orient="records"),
        "decision": (
            "No shared causal regulator is inferred. Neu5Ac and the expanded amino-acid module are tested "
            "against the previously frozen modules only as patient-level exploratory coordination."
        ),
        "claim_limit": (
            "All modules are phenotype-selected and n=10. P-values and BH q-values describe exploratory "
            "coordination, not mediation, flux, subtype specificity, or a validated regulatory network."
        ),
        "parameters": {"permutations": PERMUTATIONS, "seed": SEED},
        "provenance": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in [OLD, AMINO, ANCHORS]},
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
