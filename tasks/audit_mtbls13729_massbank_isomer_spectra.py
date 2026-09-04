"""Compare feature 1597 recurrent fragments with authentic MassBank isomer spectra."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "data/reference/massbank_modified_guanosine_20260830"
LOCAL = ROOT / "data/mtbls13729/frozen_candidate_ms2_consensus_v1/candidate_recurrent_fragments.csv"
OUT = ROOT / "data/mtbls13729/massbank_isomer_spectral_audit_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spectrum_from_record(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    record = json.loads(path.read_text(encoding="utf-8"))
    peaks = record["peak"]["peak"]["values"]
    mz = np.asarray([x["mz"] for x in peaks], dtype=float)
    intensity = np.asarray([x["rel"] for x in peaks], dtype=float)
    return mz, intensity, record


def greedy_cosine(
    mz_a: np.ndarray,
    int_a: np.ndarray,
    mz_b: np.ndarray,
    int_b: np.ndarray,
    tolerance: float,
    transform: str,
) -> tuple[float, int]:
    a = np.sqrt(int_a) if transform == "sqrt" else int_a.copy()
    b = np.sqrt(int_b) if transform == "sqrt" else int_b.copy()
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    candidates = []
    for i, ma in enumerate(mz_a):
        for j, mb in enumerate(mz_b):
            if abs(ma - mb) <= tolerance:
                candidates.append((a[i] * b[j], i, j))
    used_a: set[int] = set()
    used_b: set[int] = set()
    dot = 0.0
    matched = 0
    for product, i, j in sorted(candidates, reverse=True):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        dot += float(product)
        matched += 1
    return dot, matched


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    local = pd.read_csv(LOCAL)
    local = local[local.feature_id == 1597].sort_values("fragment_mz")
    local_mz = local.fragment_mz.to_numpy(float)
    local_int = local.median_relative_intensity.to_numpy(float)
    rows = []
    for path in sorted(LIB.glob("*.json")):
        lib_mz, lib_int, record = spectrum_from_record(path)
        # The CE10 records contain residual precursor; the local recurrent-fragment
        # table is fragment-only, so remove library peaks above m/z 290 for parity.
        keep = lib_mz < 290.0
        lib_mz = lib_mz[keep]
        lib_int = lib_int[keep]
        title = record["title"]
        compound = "7-methylguanosine" if title.startswith("7-Methyl") else "N2-methylguanosine"
        ce = int(title.split("CE: ")[1].split(";")[0])
        linear, n_linear = greedy_cosine(local_mz, local_int, lib_mz, lib_int, 0.02, "linear")
        sqrt, n_sqrt = greedy_cosine(local_mz, local_int, lib_mz, lib_int, 0.02, "sqrt")
        rows.append(
            {
                "accession": record["accession"],
                "compound": compound,
                "collision_energy": ce,
                "library_peaks_below_290": len(lib_mz),
                "local_recurrent_peaks": len(local_mz),
                "matched_peaks": n_sqrt,
                "linear_cosine": linear,
                "sqrt_cosine": sqrt,
                "source_sha256": sha256(path),
            }
        )
    scores = pd.DataFrame(rows).sort_values(["compound", "collision_energy"])
    scores.to_csv(OUT / "massbank_isomer_scores.csv", index=False)
    best = scores.loc[scores.groupby("compound").sqrt_cosine.idxmax()].set_index("compound")
    score_gap = float(abs(best.loc["7-methylguanosine", "sqrt_cosine"] - best.loc["N2-methylguanosine", "sqrt_cosine"]))
    report = {
        "status": "mtbls13729_massbank_isomer_spectral_audit_complete",
        "formal": False,
        "query": "feature 1597 recurrent-fragment consensus",
        "mass_tolerance_da": 0.02,
        "scores": scores.to_dict(orient="records"),
        "best_sqrt_cosine": best.sqrt_cosine.to_dict(),
        "best_isomer_score_gap": score_gap,
        "verdict": "The recurrent spectrum is compatible with authentic methylguanosine spectra, but 7-methylguanosine and N2-methylguanosine remain spectrally non-identifiable under this sparse cross-instrument comparison." if score_gap < 0.05 else "One library isomer scores higher, but cross-instrument sparse spectra cannot establish positional identity without same-method RT and full-spectrum standard comparison.",
        "claim_limit": "This is a public-library consistency screen, not MSI Level 1/2 identity. Collision energy, instrument, peak filtering and consensus construction differ.",
        "provenance": {"local_consensus_sha256": sha256(LOCAL), "massbank_api": "https://massbank.eu/MassBank-api/", "script_sha256": sha256(Path(__file__))},
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
