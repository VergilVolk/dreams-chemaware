"""Our own Gradio backend for the Chem-aware DreaMS platform.

This is the **self-hosted backend** behind the static `hf_space/` UI (the
"our own compute" toggle). It mirrors the official DreaMS Space
(`anton-bushuiev/DreaMS`) `/predict` contract exactly, so the static front-end
can call *either* backend with the same request shape.

Unlike the old `hf_space/app.py` (which vendored `annotation/` + `dreams/` into
the Space repo), this file imports the **main-repo** `annotation` package
directly. Run it on any machine with the two checkpoints present
(see `annotation/embed.py` DEFAULT_RAW / DEFAULT_OFFICIAL).

Interface (parity with the official `/predict`):
    lib_pth                    reference library (.mgf)
    in_pth                     input spectra (.mgf; .hdf5 query accepted;
                               .mzML / .mzXML are reserved and raise clearly)
    similarity_threshold       cosine cutoff (default 0.7)
    calculate_modified_cosine  accepted for parity; plain cosine shown (reserved)
    only_high_quality_input    accepted for parity (no-op here)

Run (conda dreams_env):
    python tasks/hf_gradio_backend.py            # serve http://127.0.0.1:7860
    python tasks/hf_gradio_backend.py --share     # public URL
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Import the main-repo annotation package. Requires CWD=ROOT or ROOT on sys.path.
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation import embed as E  # noqa: E402
from annotation.cli import parse_mgf  # noqa: E402
from annotation.params import Params  # noqa: E402
from annotation.retrieve import chunked_topk, dppm  # noqa: E402

TOP_K = 10
PPM_TOLERANCE = 20.0
_HIGH_QUALITY_MIN_PEAKS = 5


# --------------------------------------------------------------------------- #
# Input readers
# --------------------------------------------------------------------------- #
def read_query(path: Path):
    """Return (records | (emb, manifest), kind)."""
    suf = path.suffix.lower()
    if suf == ".mgf":
        return parse_mgf(path), "records"
    if suf in (".hdf5", ".h5"):
        return path, "hdf5"
    if suf in (".mzml", ".mzxml"):
        raise NotImplementedError(
            "mzML/mzXML input is reserved in this backend (the DreaMS Space accepts "
            "mzML; here it must be pre-converted to .hdf5 or .mgf). See the smoke "
            "pipeline for the conversion step."
        )
    raise ValueError(f"unrecognised input extension {suf!r}")


def read_library(path: Path) -> list[dict]:
    if path.suffix.lower() != ".mgf":
        raise NotImplementedError("reference library must be .mgf in this backend")
    return parse_mgf(path)


# --------------------------------------------------------------------------- #
# Embed + retrieve (in-memory, stateless)
# --------------------------------------------------------------------------- #
def _build_query_table(query_emb, q_manifest, lib_emb, l_manifest, threshold):
    topk_vals, topk_idx = chunked_topk(query_emb, lib_emb, TOP_K)
    q_pmz = q_manifest["precursor_mz"].to_numpy(dtype=np.float64)
    l_pmz = l_manifest["precursor_mz"].to_numpy(dtype=np.float64)
    l_smiles = l_manifest["smiles"].tolist()
    l_inchikey = l_manifest["inchikey"].tolist()
    l_name = l_manifest["name"].tolist()

    rows = []
    for i in range(len(query_emb)):
        for r in range(TOP_K):
            j = int(topk_idx[i, r])
            cos = float(topk_vals[i, r])
            dp = float(dppm(q_pmz[i : i + 1], l_pmz[j : j + 1])[0])
            rows.append({
                "query_idx": i,
                "query_file": str(q_manifest["file_name"].iloc[i]),
                "query_scan": int(q_manifest["scan_number"].iloc[i]),
                "query_precursor_mz": float(q_pmz[i]),
                "rank": r + 1,
                "cosine": cos,
                "lib_smiles": l_smiles[j],
                "lib_inchikey": l_inchikey[j],
                "lib_name": l_name[j],
                "lib_precursor_mz": float(l_pmz[j]),
                "dppm": dp,
                "mz_pass": bool(dp <= PPM_TOLERANCE),
                "pass": bool(cos >= threshold and dp <= PPM_TOLERANCE),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Model cache
# --------------------------------------------------------------------------- #
_MODEL = None


def _get_model(device: str = "cpu"):
    global _MODEL
    if _MODEL is None:
        _MODEL = E.load_embedder(device)
    return _MODEL


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #
def predict(
    lib_pth,
    in_pth,
    similarity_threshold: float = 0.7,
    calculate_modified_cosine: bool = False,
    only_high_quality_input: bool = True,
) -> pd.DataFrame:
    """Mirror of the official DreaMS /predict. Returns the annotation table."""
    model, weight, bias = _get_model("cpu")

    # Library
    lib_records = read_library(Path(lib_pth))
    lib_emb = E.embed_records(lib_records, model, weight, bias, "cpu")
    l_manifest = pd.DataFrame({
        "smiles": [r.get("smiles", "") for r in lib_records],
        "inchikey": [r.get("inchikey", "") for r in lib_records],
        "name": [r.get("name", "") for r in lib_records],
        "precursor_mz": [r.get("precursor_mz", float("nan")) for r in lib_records],
    })

    # Query
    q, kind = read_query(Path(in_pth))
    if kind == "hdf5":
        q_emb, q_manifest = E.embed_hdf5(q, model, weight, bias, "cpu")
    else:
        q_records = q
        if only_high_quality_input:
            q_records = [r for r in q_records
                         if r["peaks"].shape[1] >= _HIGH_QUALITY_MIN_PEAKS]
        q_emb = E.embed_records(q_records, model, weight, bias, "cpu")
        q_manifest = pd.DataFrame({
            "file_name": [Path(in_pth).stem] * len(q_records),
            "scan_number": np.arange(len(q_records), dtype=np.int64),
            "precursor_mz": [r.get("precursor_mz", float("nan")) for r in q_records],
            "charge": [1] * len(q_records),
            "RT": np.zeros(len(q_records)),
            "row_in_file": np.arange(len(q_records), dtype=np.int64),
        })

    out = _build_query_table(q_emb, q_manifest, lib_emb, l_manifest,
                             float(similarity_threshold))
    # drop the boolean 'pass' flag from the displayed table (interface parity),
    # keep mz_pass for transparency.
    return out.drop(columns=["pass"])


# --------------------------------------------------------------------------- #
# Serve
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--share", action="store_true", help="create a public gradio link")
    p.add_argument("--server-name", default="0.0.0.0")
    p.add_argument("--server-port", type=int, default=7860)
    args = p.parse_args()

    import gradio as gr

    with gr.Blocks(title="Chem-aware DreaMS (self-hosted)") as demo:
        gr.Markdown("# Chem-aware DreaMS — self-hosted backend\n\n"
                    "Mirrors the official DreaMS `/predict`. Reserved: modified "
                    "cosine, mzML input, HMDB/biochemical confidence.")
        with gr.Row():
            lib = gr.File(label="Reference library (.mgf)")
            inp = gr.File(label="Input spectra (.mgf / .hdf5)")
        with gr.Row():
            thr = gr.Slider(0.0, 1.0, value=0.7, label="similarity_threshold")
            mc = gr.Checkbox(label="calculate_modified_cosine (reserved)", value=False)
            hq = gr.Checkbox(label="only_high_quality_input", value=True)
        btn = gr.Button("Predict")
        out = gr.Dataframe(headers=["query_file", "query_scan", "query_precursor_mz",
                                    "rank", "cosine", "lib_smiles", "lib_inchikey",
                                    "lib_name", "lib_precursor_mz", "dppm", "mz_pass"])
        btn.click(predict, [lib, inp, thr, mc, hq], out)

    demo.launch(server_name=args.server_name, server_port=args.server_port,
                share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
