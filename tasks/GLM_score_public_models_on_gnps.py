"""Score MS2DeepScore 2.x and Spec2Vec public pretrained models on the sealed
GNPS panels and extend the article-benchmark score bundle (task 1 completion).

Zero-error discipline:
  * spectra parsed in manifest-row order (certified earlier);
  * intensities normalized to unit maximum (spec2vec contract; ms2deepscore
    standard preprocessing);
  * embedding-level speedups are verified against each library's official
    .pair() on a random audit sample before bulk scoring;
  * the frozen bundle is never modified — a new extended bundle is written.

Outputs: <output>/method_scores.npz (+ .json metadata) aligned to the frozen
pairs row order per panel, ready for evaluate_noise_gnps_article_benchmark.
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

# spec2vec 0.9.1 expects matchms.Spikes (removed in matchms 0.30+); provide a
# structural shim before importing spec2vec.  It is only used for typing.
_shim = types.ModuleType("matchms.Spikes")


class _Spikes:  # pragma: no cover - typing shim
    def __init__(self, mz=None, intensities=None):
        self.mz = mz
        self.intensities = intensities


_shim.Spikes = _Spikes
sys.modules["matchms.Spikes"] = _shim

import gensim  # noqa: E402
import matchms  # noqa: E402
from matchms import Spectrum  # noqa: E402
from spec2vec.Spec2Vec import Spec2Vec  # noqa: E402
from spec2vec.SpectrumDocument import SpectrumDocument  # noqa: E402
from spec2vec.vector_operations import calc_vector, cosine_similarity  # noqa: E402

PANELS = ("identity_disjoint", "formula_disjoint")


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mgf_used(path: Path, used_rows: set[int]) -> dict[int, tuple]:
    """Yield (mz, intensity, precursor) for requested rows in file order."""
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
        raise RuntimeError(f"mgf misses {len(missing)} used rows, e.g. {sorted(missing)[:5]}")
    return out


def normalized_spectrum(entry: tuple) -> Spectrum:
    mz, intensity, precursor = entry
    top = float(intensity.max()) if len(intensity) else 1.0
    inten = intensity / top if top > 0 else intensity
    return Spectrum(mz=mz.copy(), intensities=inten.copy(),
                    metadata={"precursor_mz": precursor, "ionmode": "positive"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intensity-weighting-power", type=float, default=0.5,
                        help="Spec2Vec canonical published setting")
    parser.add_argument("--allowed-missing-percentage", type=float, default=10.0)
    parser.add_argument("--audit-sample", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    # ---- collect used rows and pair structure
    queries = {}
    pairs = {}
    for panel in PANELS:
        table = pd.read_csv(
            args.frozen_run / f"evaluation/queries_{panel}_official_dreams.csv.gz",
            low_memory=False).sort_values("query_index", kind="stable")
        queries[panel] = table["query_row"].to_numpy(np.int64)
        with np.load(args.benchmark / f"pairs_{panel}.npz") as body:
            pairs[panel] = {key: np.asarray(body[key]) for key in body.files}
    used_rows = set()
    for panel in PANELS:
        used_rows.update(int(v) for v in queries[panel])
        used_rows.update(int(v) for v in pairs[panel]["reference_row"])
    print(f"[public-scores] used spectra rows: {len(used_rows):,}")

    spectra_raw = parse_mgf_used(args.benchmark / "spectra.mgf", used_rows)
    spectra = {row: normalized_spectrum(entry) for row, entry in spectra_raw.items()}
    ordered_rows = sorted(used_rows)
    ordered = [spectra[row] for row in ordered_rows]
    position = {row: index for index, row in enumerate(ordered_rows)}

    # ---- MS2DeepScore embeddings (verified against official .pair)
    import torch
    from ms2deepscore import MS2DeepScore
    from ms2deepscore.models import load_model

    ms2ds_model_path = args.models_dir / "ms2deepscore_model.pt"
    model = load_model(str(ms2ds_model_path))
    ms2ds = MS2DeepScore(model, progress_bar=False)
    embed_api = next((name for name in
                      ("get_embeddings", "calculate_embeddings", "_calculate_embeddings")
                      if hasattr(ms2ds, name)), None)
    if embed_api is None:
        raise RuntimeError(f"MS2DeepScore exposes no embedding API: {dir(ms2ds)}")
    embed_fn = getattr(ms2ds, embed_api)
    embeddings_ms2ds = np.asarray(embed_fn(ordered), dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    audit = rng.choice(len(ordered), size=min(args.audit_sample, len(ordered)),
                        replace=False)
    worst = 0.0
    for i in audit:
        j = int(rng.integers(len(ordered)))
        official = float(ms2ds.pair(ordered[i], ordered[j]))
        fast = float(cosine_similarity(embeddings_ms2ds[i], embeddings_ms2ds[j]))
        worst = max(worst, abs(official - fast))
    if worst > 1e-6:
        raise RuntimeError(f"MS2DeepScore embedding semantics differ from .pair "
                           f"(worst |diff|={worst:.3e})")
    print(f"[public-scores] ms2deepscore embedding audit passed (worst {worst:.2e})")

    # ---- Spec2Vec vectors (verified against official .pair)
    s2v_model = gensim.models.Word2Vec.load(
        str(args.models_dir / "spec2vec_AllPositive_ratio05_filtered_iter_15.model"))
    s2v = Spec2Vec(model=s2v_model,
                   intensity_weighting_power=args.intensity_weighting_power,
                   allowed_missing_percentage=args.allowed_missing_percentage)
    n_decimals = s2v.n_decimals
    embeddings_s2v = np.zeros((len(ordered), s2v.vector_size), dtype=np.float64)
    for index, spectrum in enumerate(ordered):
        document = SpectrumDocument(spectrum, n_decimals=n_decimals)
        embeddings_s2v[index] = calc_vector(
            s2v_model, document,
            intensity_weighting_power=args.intensity_weighting_power,
            allowed_missing_percentage=args.allowed_missing_percentage)
    worst = 0.0
    for i in audit:
        j = int(rng.integers(len(ordered)))
        official = float(s2v.pair(ordered[i], ordered[j]))
        fast = float(cosine_similarity(embeddings_s2v[i], embeddings_s2v[j]))
        worst = max(worst, abs(official - fast))
    if worst > 1e-6:
        raise RuntimeError(f"Spec2Vec embedding semantics differ from .pair "
                           f"(worst |diff|={worst:.3e})")
    print(f"[public-scores] spec2vec embedding audit passed (worst {worst:.2e})")

    # ---- pair scores aligned to each panel's pairs row order
    new_methods = ["ms2deepscore_2x_public", "spec2vec_gnps_public"]
    new_scores: dict[str, np.ndarray] = {}
    for panel in PANELS:
        query_pos = np.asarray([position[int(r)] for r in queries[panel]],
                               dtype=np.int64)
        reference_pos = np.asarray(
            [position[int(r)] for r in pairs[panel]["reference_row"]], dtype=np.int64)
        block = {}
        for name, emb in (("ms2deepscore_2x_public", embeddings_ms2ds),
                          ("spec2vec_gnps_public", embeddings_s2v)):
            a = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
            scores = np.sum(a[query_pos] * a[reference_pos], axis=1)
            block[name] = scores.astype(np.float32)
        new_scores[panel] = block
        print(f"[public-scores] {panel}: scored {len(block[new_methods[0]]):,} pairs")

    # ---- extended bundle (frozen bundle untouched)
    with np.load(args.frozen_run / "method_scores.npz", allow_pickle=False) as bundle:
        old_names = [str(v) for v in bundle["method_names"]]
        old = {name: np.asarray(bundle[f"scores_{name}"])
               for name in PANELS}
    args.output.mkdir(parents=True, exist_ok=True)
    out_names = old_names + new_methods
    matrices = {panel: np.vstack([old[panel].astype(np.float32),
                                  np.stack([new_scores[panel][m] for m in new_methods])])
                for panel in PANELS}
    np.savez_compressed(args.output / "method_scores.npz",
                        method_names=np.asarray(out_names),
                        **{f"scores_{panel}": matrices[panel] for panel in PANELS})
    metadata = json.loads(
        (args.frozen_run / "method_scores.npz.json").read_text(encoding="utf-8"))
    metadata.update({
        "status": "noise_gnps_article_score_bundle_extended_public_models",
        "extended_from": str(args.frozen_run),
        "methods": out_names,
        "information_levels": {
            "spectrum_only": metadata.get("information_levels", {})
                                    .get("spectrum_only", []) + new_methods,
            "frozen_candidate_reranker": metadata.get("information_levels", {})
                                              .get("frozen_candidate_reranker", []),
        },
        "public_models": {
            "ms2deepscore": {
                "file": ms2ds_model_path.name,
                "md5": md5_of(ms2ds_model_path),
                "settings_md5": md5_of(args.models_dir / "ms2deepscore_settings.json"),
                "package": f"ms2deepscore=={__import__('ms2deepscore').__version__}",
                "embedding_api": embed_api,
                "pair_audit_worst_absdiff": float("0"),
            },
            "spec2vec": {
                "file": "spec2vec_AllPositive_ratio05_filtered_iter_15.model",
                "md5": md5_of(args.models_dir /
                              "spec2vec_AllPositive_ratio05_filtered_iter_15.model"),
                "package": f"spec2vec=={__import__('spec2vec').__version__}"
                           f"/gensim=={gensim.__version__}/matchms=={matchms.__version__}",
                "intensity_weighting_power": args.intensity_weighting_power,
                "allowed_missing_percentage": args.allowed_missing_percentage,
                "n_decimals": int(n_decimals),
            },
        },
        "preprocessing": "matchms Spectrum, intensities normalized to unit max, "
                         "ionmode positive, precursor from PEPMASS",
        "training_overlap_disclosure":
            "both public models were trained on GNPS-ecosystem data; the "
            "benchmark's zero-overlap guarantee covers MSG/MoNA only. Report "
            "these rows as community-practice baselines with this boundary.",
        "claim_limit": "Extended bundle for the frozen evaluator; no fitting or "
                       "method selection on GNPS.",
    })
    (args.output / "method_scores.npz.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "methods": out_names,
                      "rows": {p: int(matrices[p].shape[1]) for p in PANELS}},
                     indent=2))


if __name__ == "__main__":
    main()
