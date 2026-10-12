#!/usr/bin/env python
"""Build the LCNEC public-evidence candidate edge table (M0 input, GNPS-negative source).

Queries: the 263 phenotype-blind QC-qualified ion-family representative spectra
(negative mode). References: the unified GNPS-negative public library
(29,564 spectra with InChIKey/adduct/source provenance, no instrument field).
Candidate edges are precursor-gated at 5 ppm and scored with two frozen
evidence families: pinned weighted spectral entropy and the public
MS2DeepScore 2.x dual-ionisation model (embedding API, cosine).

This is a LABEL-BLIND build: no LCNEC phenotype, structure truth or abundance
is read. Output columns follow build_public_spectral_evidence_certificates.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "tasks")
from noise_gnps_article_spectral_scores import weighted_entropy_similarity  # noqa: E402

PPM_GATE = 5.0
WSE_KEEP = 0.10
MS2DS_KEEP = 0.50
EDGE_GRID_MIN = 0.30  # keep edges with WSE >= 0.30 OR MS2DS >= 0.70 in the table


def parse_mgf(path: Path, with_metadata: bool) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    peaks: list[tuple[float, float]] = []
    meta: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                peaks, meta = [], {}
            elif line == "END IONS":
                mz = np.asarray([p[0] for p in peaks], dtype=np.float64)
                inten = np.asarray([p[1] for p in peaks], dtype=np.float64)
                order = np.argsort(mz)
                entries.append({
                    "name": meta.get("NAME", meta.get("TITLE", f"entry_{len(entries)}")),
                    "precursor": float(meta.get("PEPMASS", "nan").split()[0]),
                    "inchikey": meta.get("INCHIKEY", ""),
                    "adduct": meta.get("ADDUCT", ""),
                    "source": meta.get("SOURCE", ""),
                    "mz": mz[order],
                    "intensity": inten[order],
                })
            elif "=" in line:
                key, value = line.split("=", 1)
                meta[key.strip().upper()] = value.strip()
            elif line and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path,
                        default=Path("data/validation/lcnec_hsst3n_entity_ms2_20261010/priority_dark_modules.mgf"))
    parser.add_argument("--references", type=Path,
                        default=Path("data/reference/unified_v3/unified_neg.mgf"))
    parser.add_argument("--ms2ds-model", type=Path,
                        default=Path("data/reference/gnps_public_models_v1/ms2deepscore_model.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/validation/lcnec_public_edges_gnpsneg_v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.time()

    queries = parse_mgf(args.queries, with_metadata=True)
    references = parse_mgf(args.references, with_metadata=True)
    references = [r for r in references if np.isfinite(r["precursor"]) and len(r["mz"]) >= 4]
    print(f"[edges] queries={len(queries)} references={len(references)}", flush=True)

    # precursor gating
    ref_mz = np.asarray([r["precursor"] for r in references])
    ref_order = np.argsort(ref_mz)
    ref_mz_sorted = ref_mz[ref_order]
    gated: list[tuple[int, list[int]]] = []
    for qi, q in enumerate(queries):
        mz = float(q["precursor"])
        lo = int(np.searchsorted(ref_mz_sorted, mz * (1 - PPM_GATE * 1e-6)))
        hi = int(np.searchsorted(ref_mz_sorted, mz * (1 + PPM_GATE * 1e-6)))
        gated.append((qi, [int(ref_order[k]) for k in range(lo, hi)]))
    n_pairs = sum(len(ids) for _, ids in gated)
    print(f"[edges] gated pairs={n_pairs} queries_with_candidate="
          f"{sum(1 for _, ids in gated if ids)}", flush=True)

    # family 1: pinned WSE
    wse_rows = []
    for qi, ref_ids in gated:
        q = queries[qi]
        spec_q = np.vstack([q["mz"], q["intensity"]])
        for ri in ref_ids:
            r = references[ri]
            spec_r = np.vstack([r["mz"], r["intensity"]])
            try:
                score = weighted_entropy_similarity(spec_q, spec_r, 0.02)
            except Exception:
                score = float("nan")
            if np.isfinite(score) and score >= WSE_KEEP:
                wse_rows.append((qi, ri, float(score)))
    print(f"[edges] WSE kept={len(wse_rows)} ({time.time()-started:.0f}s)", flush=True)

    # family 2: MS2DeepScore public dual model
    import torch
    from matchms import Spectrum
    from ms2deepscore import MS2DeepScore
    from ms2deepscore.models import load_model

    def to_spectrum(entry: dict[str, object], ionmode: str) -> Spectrum:
        inten = entry["intensity"]
        top = float(inten.max()) if len(inten) else 1.0
        return Spectrum(mz=np.asarray(entry["mz"]).copy(),
                        intensities=(inten / top if top > 0 else inten).copy(),
                        metadata={"precursor_mz": float(entry["precursor"]), "ionmode": ionmode})

    model = load_model(str(args.ms2ds_model), allow_legacy=True)
    scorer = MS2DeepScore(model, progress_bar=False)
    embed_api = next((name for name in ("get_embedding_array", "get_embeddings",
                                        "calculate_embeddings", "_calculate_embeddings")
                      if hasattr(scorer, name)), None)
    if embed_api is None:
        raise RuntimeError(f"MS2DeepScore exposes no embedding API: {dir(scorer)}")
    embed_fn = getattr(scorer, embed_api)

    used_refs = sorted({ri for _, ids in gated for ri in ids})
    q_specs = [to_spectrum(q, "negative") for q in queries]
    r_specs = [to_spectrum(references[ri], "negative") for ri in used_refs]
    q_emb = np.asarray(embed_fn(q_specs), dtype=np.float64)
    r_emb = np.asarray(embed_fn(r_specs), dtype=np.float64)
    r_emb /= np.clip(np.linalg.norm(r_emb, axis=1, keepdims=True), 1e-12, None)
    q_emb /= np.clip(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12, None)
    r_pos = {ri: k for k, ri in enumerate(used_refs)}
    print(f"[edges] ms2deepscore embedded {len(q_specs)} queries + {len(r_specs)} refs "
          f"({time.time()-started:.0f}s)", flush=True)

    ms2ds_rows = []
    for qi, ref_ids in gated:
        for ri in ref_ids:
            score = float(q_emb[qi] @ r_emb[r_pos[ri]])
            if score >= MS2DS_KEEP:
                ms2ds_rows.append((qi, ri, score))
    print(f"[edges] MS2DS kept={len(ms2ds_rows)}", flush=True)

    # assemble edge table
    rows = []
    for family, family_rows in (("weighted_spectral_entropy", wse_rows),
                                ("ms2deepscore_2x_public", ms2ds_rows)):
        for qi, ri, score in family_rows:
            r = references[ri]
            rows.append({
                "query_id": str(queries[qi]["name"]),
                "public_spectrum_id": f"unifiedneg_{ri}_{r['name']}",
                "public_unit_id": str(r["inchikey"]),
                "public_source_id": f"gnps_neg:{r['source']}",
                "public_instrument": "not_recorded",
                "evidence_family": family,
                "score": round(float(score), 6),
            })
    edges = pd.DataFrame(rows)
    # drop rows far below every family floor to keep the table compact
    wide = edges.pivot_table(index=["query_id", "public_spectrum_id", "public_unit_id",
                                    "public_source_id", "public_instrument"],
                             columns="evidence_family", values="score")
    keep_mask = ((wide.get("weighted_spectral_entropy", pd.Series(0.0, index=wide.index)) >= EDGE_GRID_MIN)
                 | (wide.get("ms2deepscore_2x_public", pd.Series(0.0, index=wide.index)) >= 0.70))
    wide = wide.loc[keep_mask]
    edges_kept = wide.reset_index().melt(id_vars=["query_id", "public_spectrum_id", "public_unit_id",
                                                  "public_source_id", "public_instrument"],
                                         var_name="evidence_family", value_name="score").dropna()
    edges_kept.to_csv(args.output / "edges.csv", index=False)
    pd.DataFrame({"query_id": sorted(str(q["name"]) for q in queries)}).to_csv(
        args.output / "query_universe.csv", index=False)
    pd.DataFrame({"public_source_id": []}).to_csv(
        args.output / "excluded_public_sources.csv", index=False)

    report = {
        "status": "LCNEC_PUBLIC_EDGES_GNPSNEG_BUILT",
        "data_role": "development/consumed",
        "contract": "docs/LCNEC_PUBLIC_EVIDENCE_SPACE_M0_CONTRACT_20261011.md",
        "n_queries": len(queries),
        "n_references": len(references),
        "ppm_gate": PPM_GATE,
        "gated_pairs": n_pairs,
        "queries_with_any_candidate": sum(1 for _, ids in gated if ids),
        "wse_rows_kept": len(wse_rows),
        "ms2ds_rows_kept": len(ms2ds_rows),
        "edges_in_table": int(len(edges_kept)),
        "sources": sorted(set(edges_kept.get("public_source_id", pd.Series(dtype=str)))),
        "label_blind": True,
        "instrument_provenance": "not_recorded_in_source_mgf",
        "frozen_thresholds": "data/validation/lcnec_public_evidence_thresholds_v1/thresholds.json",
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
