"""Chem-aware DreaMS — annotation backend (Gradio).

Reference-free LC-MS/MS spectral annotation. This Space runs the **frozen DreaMS
retrieval embedding** (Bushuiev et al., Nat Biotechnol 2025) with a precursor
m/z hard constraint, mirroring the official DreaMS `/predict` contract.

Two backends, one form:
  * **Ours (default)** — local CPU inference on this Space (frozen backbone +
    linear head, 100 peaks). Suitable for demo-scale libraries.
  * **Reference** — an *advanced* toggle that forwards the exact same request to
    the official DreaMS Space (`anton-bushuiev/DreaMS`) GPU backend via
    `gradio_client`. Kept behind a collapsed "Advanced" accordion.

Interface parity with the official `/predict`:
    lib_pth, in_pth, similarity_threshold, calculate_modified_cosine,
    only_high_quality_input
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from annotation import embed as E  # vendored minimal annotation package

TOP_K = 10
PPM_TOLERANCE = 20.0
_HIGH_QUALITY_MIN_PEAKS = 5

# --------------------------------------------------------------------------- #
# Small, self-contained helpers (kept inline to minimise the vendored surface)
# --------------------------------------------------------------------------- #
def parse_mgf(path: Path) -> list[dict]:
    """Parse a standard MGF (SMILES / INCHIKEY / PEPMASS / peaks) into records."""
    records: list[dict] = []
    cur: dict | None = None
    peaks: list[tuple[float, float]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line == "BEGIN IONS":
                cur = {}
                peaks = []
            elif line == "END IONS":
                if cur is not None and peaks and cur.get("precursor_mz"):
                    arr = np.asarray(peaks, dtype=np.float32)
                    cur["peaks"] = arr.T
                    records.append(cur)
                cur = None
            elif cur is not None and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip()
                if k == "PEPMASS":
                    cur["precursor_mz"] = float(v.split()[0])
                elif k == "SMILES":
                    cur["smiles"] = v
                elif k == "INCHIKEY":
                    cur["inchikey"] = v
                elif k == "NAME":
                    cur["name"] = v
            elif cur is not None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
    return records


def chunked_topk(query: np.ndarray, library: np.ndarray, k: int,
                 chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    n_query = query.shape[0]
    topk_vals = np.empty((n_query, k), dtype=np.float32)
    topk_idx = np.empty((n_query, k), dtype=np.int64)
    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        sim = query[start:stop] @ library.T  # both L2-normalized -> cosine
        idx = np.argpartition(sim, -k, axis=1)[:, -k:]
        vals = np.take_along_axis(sim, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        topk_idx[start:stop] = np.take_along_axis(idx, order, axis=1)
        topk_vals[start:stop] = np.take_along_axis(vals, order, axis=1)
    return topk_vals, topk_idx


def dppm(query_mz: np.ndarray, lib_mz: np.ndarray) -> np.ndarray:
    return np.abs(query_mz - lib_mz) / np.maximum(np.abs(lib_mz), 1e-9) * 1e6


# --------------------------------------------------------------------------- #
# Model cache
# --------------------------------------------------------------------------- #
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        # Checkpoints are expected at (Space root) dreams/models/pretrained/ and
        # data/e1/ (git-lfs). See README.md.
        _MODEL = E.load_embedder("cpu")
    return _MODEL


# --------------------------------------------------------------------------- #
# Our own inference path
# --------------------------------------------------------------------------- #
def _read_query(path: Path):
    suf = path.suffix.lower()
    if suf == ".mgf":
        return parse_mgf(path), "records"
    if suf in (".hdf5", ".h5"):
        return path, "hdf5"
    if suf in (".mzml", ".mzxml"):
        raise NotImplementedError(
            "mzML/mzXML input is reserved here (pre-convert to .mgf or .hdf5). "
            "The reference backend accepts mzML directly."
        )
    raise ValueError(f"unrecognised input extension {suf!r}")


def _local_predict(lib_pth, in_pth, threshold, high_quality):
    model, weight, bias = get_model()

    # Library
    lib_records = parse_mgf(Path(lib_pth))
    if not lib_records:
        raise ValueError("library is empty (no BEGIN/END IONS blocks parsed)")
    lib_emb = E.embed_records(lib_records, model, weight, bias, "cpu")
    l_manifest = pd.DataFrame({
        "smiles": [r.get("smiles", "") for r in lib_records],
        "inchikey": [r.get("inchikey", "") for r in lib_records],
        "name": [r.get("name", "") for r in lib_records],
        "precursor_mz": [r.get("precursor_mz", float("nan")) for r in lib_records],
    })

    # Query
    q, kind = _read_query(Path(in_pth))
    if kind == "hdf5":
        q_emb, q_manifest = E.embed_hdf5(q, model, weight, bias, "cpu")
    else:
        q_records = q
        if high_quality:
            q_records = [r for r in q_records
                         if r["peaks"].shape[1] >= _HIGH_QUALITY_MIN_PEAKS]
        if not q_records:
            raise ValueError("no input spectra remain after high-quality filter")
        q_emb = E.embed_records(q_records, model, weight, bias, "cpu")
        q_manifest = pd.DataFrame({
            "file_name": [Path(in_pth).stem] * len(q_records),
            "scan_number": np.arange(len(q_records), dtype=np.int64),
            "precursor_mz": [r.get("precursor_mz", float("nan")) for r in q_records],
            "charge": [1] * len(q_records),
            "RT": np.zeros(len(q_records)),
            "row_in_file": np.arange(len(q_records), dtype=np.int64),
        })

    topk_vals, topk_idx = chunked_topk(q_emb, lib_emb, TOP_K)
    q_pmz = q_manifest["precursor_mz"].to_numpy(dtype=np.float64)
    l_pmz = l_manifest["precursor_mz"].to_numpy(dtype=np.float64)
    l_smiles = l_manifest["smiles"].tolist()
    l_inchikey = l_manifest["inchikey"].tolist()
    l_name = l_manifest["name"].tolist()

    rows = []
    for i in range(len(q_emb)):
        for r in range(TOP_K):
            j = int(topk_idx[i, r])
            cos = float(topk_vals[i, r])
            dp = float(dppm(q_pmz[i:i + 1], l_pmz[j:j + 1])[0])
            rows.append({
                "query_file": str(q_manifest["file_name"].iloc[i]),
                "query_scan": int(q_manifest["scan_number"].iloc[i]),
                "query_precursor_mz": round(float(q_pmz[i]), 4),
                "rank": r + 1,
                "cosine": round(cos, 4),
                "dppm": round(dp, 2),
                "mz_pass": bool(dp <= PPM_TOLERANCE),
                "confident": bool(cos >= threshold and dp <= PPM_TOLERANCE),
                "lib_name": l_name[j],
                "lib_inchikey": l_inchikey[j],
                "lib_smiles": l_smiles[j],
                "lib_precursor_mz": round(float(l_pmz[j]), 4),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Reference backend (official DreaMS API) — the advanced "free-ride" path
# --------------------------------------------------------------------------- #
_REF_CLIENT = None


def _get_reference_client():
    global _REF_CLIENT
    if _REF_CLIENT is None:
        from gradio_client import Client
        _REF_CLIENT = Client("anton-bushuiev/DreaMS")
    return _REF_CLIENT


def _reference_predict(lib_pth, in_pth, threshold, modified_cosine, high_quality):
    from gradio_client import handle_file
    client = _get_reference_client()
    result = client.predict(
        lib_pth=handle_file(str(lib_pth)) if lib_pth else None,
        in_pth=handle_file(str(in_pth)),
        similarity_threshold=float(threshold),
        calculate_modified_cosine=bool(modified_cosine),
        only_high_quality_input=bool(high_quality),
        api_name="/predict",
    )
    return _coerce_df(result)


def _coerce_df(result) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, dict):
        # a single dict or a dict-of-lists
        return pd.DataFrame(result if all(isinstance(v, (list, tuple, np.ndarray))
                                           for v in result.values()) else [result])
    if isinstance(result, (list, tuple)):
        if result and isinstance(result[0], (list, tuple)):
            return pd.DataFrame([list(r) for r in result])
        return pd.DataFrame(result)
    return pd.DataFrame({"result": [result]})


# --------------------------------------------------------------------------- #
# Top-level predict (dispatches ours vs reference)
# --------------------------------------------------------------------------- #
def predict(lib_pth, in_pth, similarity_threshold, calculate_modified_cosine,
            only_high_quality_input, use_reference_backend):
    if not in_pth:
        raise gr.Error("Please upload input spectra.")
    if use_reference_backend:
        return _reference_predict(lib_pth, in_pth, similarity_threshold,
                                  calculate_modified_cosine, only_high_quality_input)
    if not lib_pth:
        raise gr.Error("Our backend needs a reference library (.mgf).")
    return _local_predict(lib_pth, in_pth, float(similarity_threshold),
                          bool(only_high_quality_input))


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
import gradio as gr  # noqa: E402

with gr.Blocks(title="Chem-aware DreaMS") as demo:
    gr.Markdown(
        "# Chem-aware DreaMS\n"
        "Reference-free LC-MS/MS spectral annotation — frozen DreaMS retrieval "
        "embedding with a precursor m/z hard constraint.\n\n"
        "**Backend (ours, default)** runs on this Space's CPU; the advanced toggle "
        "routes to the official DreaMS GPU backend."
    )
    with gr.Row():
        lib = gr.File(label="Reference library (.mgf)", type="filepath")
        inp = gr.File(label="Input spectra (.mgf / .hdf5)", type="filepath")
    with gr.Row():
        thr = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="similarity_threshold")
        mc = gr.Checkbox(label="calculate_modified_cosine", value=False)
        hq = gr.Checkbox(label="only_high_quality_input", value=True)
    with gr.Accordion("Advanced", open=False):
        ref = gr.Checkbox(label="Use reference backend (route to reference DreaMS model)",
                          value=False)
        gr.Markdown("When checked, the request is forwarded to the official DreaMS "
                    "Space (`anton-bushuiev/DreaMS`) GPU backend via `gradio_client`.")
    btn = gr.Button("Predict", variant="primary")
    out = gr.Dataframe(label="Annotations", interactive=False)
    btn.click(predict, [lib, inp, thr, mc, hq, ref], out)

demo.launch()
