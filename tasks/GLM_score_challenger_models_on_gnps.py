"""Score challenger methods on the sealed GNPS panels (task-1 extension).

Methods (each scored independently, results staged as .npy aligned to the
frozen pairs row order per panel, merged later into one bundle):

  entropy_raw_public       raw pipeline of the Denoising-Search authors:
                           truncate at pmz-1.6, entropy similarity,
                           ms2 tolerance 0.02, noise_threshold 0
  denoising_search_public  their full pipeline: electronic denoising of the
                           query, per-master-formula spectral denoising, then
                           entropy similarity (the paper's ranking key)
  spec2vec_2026_retrained  the MS2LDA 2.0 retrained positive-mode Spec2Vec
                           (150225_CleanedLibraries model), same calc_vector
                           pipeline as the 2020 model

Zero-error discipline mirrors GLM_score_public_models_on_gnps: manifest-row
spectra, package-native semantics, audits against the packages' own paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

_shim = types.ModuleType("matchms.Spikes")


class _Spikes:  # typing shim for spec2vec 0.9.1
    def __init__(self, mz=None, intensities=None):
        self.mz, self.intensities = mz, intensities


_shim.Spikes = _Spikes
sys.modules["matchms.Spikes"] = _shim

PANELS = ("identity_disjoint", "formula_disjoint")
ADDUCT = "[M+H]+"


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mgf_used(path: Path, used_rows: set[int]) -> dict[int, tuple]:
    out: dict[int, tuple] = {}
    row = -1
    peaks: list[tuple[float, float]] = []
    precursor = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                peaks, precursor = [], None
            elif line == "END IONS":
                row += 1
                if row in used_rows and peaks and precursor is not None:
                    out[row] = (np.asarray([p[0] for p in peaks], dtype=np.float64),
                                np.asarray([p[1] for p in peaks], dtype=np.float64),
                                precursor)
            elif line.startswith("PEPMASS"):
                try:
                    precursor = float(line.split("=", 1)[1].split()[0])
                except (ValueError, IndexError):
                    precursor = None
            elif "=" in line or not line:
                continue
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
    missing = used_rows - set(out)
    if missing:
        raise RuntimeError(f"mgf misses {len(missing)} rows, e.g. {sorted(missing)[:5]}")
    return out


def pair_arrays(queries: dict, pairs: dict):
    query_rows = {}
    query_index_per_pair = {}
    for panel in PANELS:
        table = queries[panel]
        query_rows[panel] = table["query_row"].to_numpy(np.int64)
        query_index_per_pair[panel] = pairs[panel]["query_index"].astype(np.int64)
    return query_rows, query_index_per_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True,
                        choices=("entropy_raw_public", "denoising_search_public",
                                 "spec2vec_2026_retrained"))
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--audit-sample", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    args.staging.mkdir(parents=True, exist_ok=True)

    queries, pairs = {}, {}
    for panel in PANELS:
        queries[panel] = pd.read_csv(
            args.frozen_run / f"evaluation/queries_{panel}_official_dreams.csv.gz",
            low_memory=False).sort_values("query_index", kind="stable")
        with np.load(args.benchmark / f"pairs_{panel}.npz") as body:
            pairs[panel] = {key: np.asarray(body[key]) for key in body.files}

    used = set()
    for panel in PANELS:
        used.update(int(v) for v in queries[panel]["query_row"])
        used.update(int(v) for v in pairs[panel]["reference_row"])
    spectra = parse_mgf_used(args.benchmark / "spectra.mgf", used)
    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz",
                           usecols=["row", "smiles", "formula"])
    smiles_by_row = manifest["smiles"].astype(str).to_numpy()
    print(f"[challengers] spectra loaded: {len(spectra):,}")

    query_rows, pair_qi = pair_arrays(queries, pairs)

    rng = np.random.default_rng(args.seed)

    def score_entropy_family(denoised: bool) -> dict[str, np.ndarray]:
        import spectral_denoising as sd
        # the package uses np.NAN (removed in modern NumPy); alias it
        np.NAN = np.nan
        from spectral_denoising.denoising_search import (
            get_all_master_formulas,
            spectral_denoising_with_master_formulas,
        )
        out = {}
        for panel in PANELS:
            q_rows = query_rows[panel]
            qi = pair_qi[panel]
            refs = pairs[panel]["reference_row"].astype(np.int64)
            scores = np.zeros(len(refs), dtype=np.float32)
            n_q = len(q_rows)
            # per query: denoise once per master formula, score all pairs
            order = np.argsort(qi, kind="stable")
            bounds = np.searchsorted(qi[order], np.arange(n_q + 1))
            for query in range(n_q):
                local = order[bounds[query]:bounds[query + 1]]
                if len(local) == 0:
                    continue
                q_row = int(q_rows[query])
                mz, inten, prec = spectra[q_row]
                msms_raw = np.stack([mz, inten], axis=1)
                ref_rows_panel = refs[local]
                if denoised:
                    frame = pd.DataFrame({
                        "smiles": smiles_by_row[ref_rows_panel],
                        "adduct": ADDUCT,
                    })
                    candidates, unique_formulas, benzene_tag = get_all_master_formulas(
                        frame, "smiles", "adduct")
                    msms_e = sd.electronic_denoising(msms_raw)
                    if isinstance(msms_e, float):
                        msms_e = msms_raw
                    denoised_by_formula = []
                    for formula, tag in zip(unique_formulas, benzene_tag):
                        result = spectral_denoising_with_master_formulas(
                            msms_e, formula, tag, prec)
                        denoised_by_formula.append(
                            msms_e if isinstance(result, float) else result)
                    formula_pos = {f: i for i, f in enumerate(unique_formulas)}
                    master = candidates["master_formula"].tolist()
                    for k, pair in enumerate(local):
                        denoised_spec = denoised_by_formula[
                            formula_pos[master[k]]] if master[k] in formula_pos \
                            else msms_e
                        rmz, rinten, _ = spectra[int(refs[pair])]
                        value = sd.entropy_similairty(
                            denoised_spec, np.stack([rmz, rinten], axis=1), pmz=prec)
                        scores[pair] = 0.0 if value != value else float(value)
                else:
                    for pair in local:
                        rmz, rinten, _ = spectra[int(refs[pair])]
                        value = sd.entropy_similairty(
                            msms_raw, np.stack([rmz, rinten], axis=1), pmz=prec)
                        scores[pair] = 0.0 if value != value else float(value)
                if query % 1000 == 0:
                    print(f"  [{panel}] query {query}/{n_q}", flush=True)
            out[panel] = scores
        return out

    if "entropy_raw_public" in args.methods:
        blocks = score_entropy_family(denoised=False)
        for panel in PANELS:
            np.save(args.staging / f"scores_{panel}_entropy_raw_public.npy",
                    blocks[panel])
        print("[challengers] entropy_raw_public staged")

    if "denoising_search_public" in args.methods:
        blocks = score_entropy_family(denoised=True)
        # save IMMEDIATELY so computation is never lost to audit issues
        for panel in PANELS:
            np.save(args.staging / f"scores_{panel}_denoising_search_public.npy",
                    blocks[panel])
        print("[challengers] denoising_search_public staged (pre-audit)")
        # non-fatal audit: compare our best score against the package's own
        # top-1 for a random query; the column is 'denoised_similarity'
        try:
            import spectral_denoising as sd
            np.NAN = np.nan
            panel = "identity_disjoint"
            qi = pair_qi[panel]
            query = int(rng.integers(len(query_rows[panel])))
            local = np.flatnonzero(qi == query)
            q_row = int(query_rows[panel][query])
            mz, inten, prec = spectra[q_row]
            lib = pd.DataFrame({
                "precursor_mz": [prec] * len(local),
                "smiles": smiles_by_row[pairs[panel]["reference_row"][local]],
                "adduct": [ADDUCT] * len(local),
                "peaks": [np.stack(spectra[int(r)][:2], axis=1)
                          for r in pairs[panel]["reference_row"][local]],
            })
            reference = sd.denoising_search(
                np.stack([mz, inten], axis=1), prec, lib, first_n="all")
            if len(reference) and "denoised_similarity" in reference.columns:
                their_best = float(reference["denoised_similarity"].max())
                my_best = float(blocks[panel][local].max())
                diff = abs(my_best - their_best)
                print(f"[challengers] denoising audit q={query}: "
                      f"my={my_best:.6f} pkg={their_best:.6f} |diff|={diff:.2e}")
                if diff > 1e-4:
                    print(f"[challengers] WARNING: denoising audit diff {diff:.6f}")
            else:
                print(f"[challengers] audit skipped: "
                      f"cols={list(reference.columns) if len(reference) else 'empty'}")
        except Exception as audit_error:
            print(f"[challengers] audit non-fatal: {audit_error}")
        print("[challengers] denoising_search_public staged")

    if "spec2vec_2026_retrained" in args.methods:
        import gensim
        from spec2vec.Spec2Vec import Spec2Vec
        from spec2vec.SpectrumDocument import SpectrumDocument
        from spec2vec.vector_operations import calc_vector
        model_path = args.models_dir / "150225_Spec2Vec_pos_CleanedLibraries.model"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        model = gensim.models.Word2Vec.load(str(model_path))
        s2v = Spec2Vec(model=model, intensity_weighting_power=0.5,
                       allowed_missing_percentage=10)
        rows_sorted = sorted(used)
        position = {row: index for index, row in enumerate(rows_sorted)}
        vectors = np.zeros((len(rows_sorted), s2v.vector_size), dtype=np.float64)
        for index, row in enumerate(rows_sorted):
            mz, inten, prec = spectra[row]
            top = float(inten.max()) if len(inten) else 1.0
            normalized = inten / top if top > 0 else inten
            from matchms import Spectrum as _Spectrum
            spectrum = _Spectrum(
                mz=mz.copy(), intensities=normalized.copy(),
                metadata={"precursor_mz": prec, "ionmode": "positive"})
            document = SpectrumDocument(spectrum, n_decimals=s2v.n_decimals)
            vectors[index] = calc_vector(model, document,
                                         intensity_weighting_power=0.5,
                                         allowed_missing_percentage=10)
        unit = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True),
                                 1e-12, None)
        for panel in PANELS:
            q_rows = query_rows[panel]
            qi = pair_qi[panel]
            refs = pairs[panel]["reference_row"].astype(np.int64)
            q_pos = np.asarray([position[int(q_rows[i])] for i in qi],
                               dtype=np.int64)
            r_pos = np.asarray([position[int(r)] for r in refs], dtype=np.int64)
            np.save(args.staging / f"scores_{panel}_spec2vec_2026_retrained.npy",
                    (np.sum(unit[q_pos] * unit[r_pos], axis=1)).astype(np.float32))
        meta = {"model": model_path.name, "md5": md5_of(model_path),
                "package": "spec2vec==0.9.1", "power": 0.5,
                "allowed_missing": 10.0}
        (args.staging / "spec2vec_2026_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        print("[challengers] spec2vec_2026_retrained staged")

    print(json.dumps({"staging": str(args.staging), "methods": args.methods},
                     indent=2))


if __name__ == "__main__":
    main()
