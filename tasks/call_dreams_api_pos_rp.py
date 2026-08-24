"""Call the official DreaMS Space API (anton-bushuiev/DreaMS) for MTBLS13729.

Part B of the plan: use the official DreaMS Space's GPU backend (not our own
compute) to run the MTBLS13729 annotation. This script is *exploratory* -- the
first thing it must do is DISCOVER the API contract (output shape, whether a
reference library must be uploaded or is built-in, rate limits), then batch.

Requires:  pip install gradio_client

Two modes:
    --probe   (default) one minimal call, dump the raw result structure.
              Do this FIRST and read the output before --batch.
    --batch   iterate a directory of input files, one /predict call each,
              save the raw result best-effort to --out-dir.

The batch mode is intentionally format-agnostic: it serialises whatever the
Space returns (JSON if possible, else repr) so we can parse it later once the
probe has pinned the schema.

Usage (conda dreams_env):
    python tasks/call_dreams_api_pos_rp.py --probe
    python tasks/call_dreams_api_pos_rp.py --batch --input-dir data/mtbls13729/mzml/pos_rp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPACE = "anton-bushuiev/DreaMS"
SMOKE_INPUT = ROOT / "data/mtbls13729/smoke/mzml/P01-LN.mzML"
SMOKE_LIB = ROOT / "data/mtbls13729/smoke/lib/mona_pos_smoke.mgf"

# Official DreaMS reference library (MassSpecGym = GNPS + MoNA + Pluskal lab). Passed
# as a URL so the Space downloads it server-side, instead of us re-uploading 1.54 GB per call.
MASS_SPEC_GYM_URL = (
    "https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/"
    "main/data/auxiliary/MassSpecGym_DreaMS.hdf5"
)


def _to_serializable(obj, depth: int = 0):
    """Best-effort convert a gradio/FileData/numpy/pandas result to JSON-ish."""
    if depth > 4:
        return repr(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v, depth + 1) for v in obj]
    if hasattr(obj, "to_dict"):  # pandas DataFrame / Series
        try:
            return obj.to_dict(orient="records")
        except Exception:
            pass
    if hasattr(obj, "tolist"):  # numpy array
        try:
            return _to_serializable(obj.tolist(), depth + 1)
        except Exception:
            pass
    if hasattr(obj, "path"):  # gradio FileData
        return {"path": obj.path, "orig_name": getattr(obj, "orig_name", None)}
    return repr(obj)


def _dump(obj, label: str) -> None:
    print(f"\n===== {label} =====", flush=True)
    print(f"type: {type(obj).__module__}.{type(obj).__name__}", flush=True)
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"  key={k!r}  type={type(v).__name__}  value={v!r}"[:400], flush=True)
    elif isinstance(obj, (list, tuple)):
        print(f"  len={len(obj)}", flush=True)
        if obj:
            print(f"  first elem: {obj[0]!r}"[:400], flush=True)
    else:
        print(f"  {obj!r}"[:800], flush=True)


def make_client(hf_token: str | None):
    from gradio_client import Client
    kwargs = {"hf_token": hf_token} if hf_token else {}
    return Client(SPACE, **kwargs)


def call_predict(client, lib_pth, in_pth, similarity_threshold, calculate_modified_cosine, only_high_quality_input):
    from gradio_client import handle_file
    return client.predict(
        lib_pth=handle_file(str(lib_pth)) if lib_pth else None,
        in_pth=handle_file(str(in_pth)),
        similarity_threshold=float(similarity_threshold),
        calculate_modified_cosine=bool(calculate_modified_cosine),
        only_high_quality_input=bool(only_high_quality_input),
        api_name="/predict",
    )


def probe(args) -> int:
    from gradio_client import Client
    client = make_client(args.hf_token)
    print(f"[probe] connected to {SPACE}", flush=True)
    try:
        info = client.view_api(return_format="dict", print_info=False)
        print("[probe] endpoints:", flush=True)
        print(json.dumps(_to_serializable(info), ensure_ascii=False, indent=2)[:3000], flush=True)
    except Exception as exc:  # view_api is optional / may not exist on all versions
        print(f"[probe] view_api unavailable: {exc}", flush=True)

    in_pth = Path(args.input) if args.input else SMOKE_INPUT
    lib_pth = Path(args.lib) if args.lib else SMOKE_LIB
    if not in_pth.exists():
        print(f"[probe] FATAL: input missing: {in_pth}", file=sys.stderr)
        return 1
    if lib_pth and not lib_pth.exists():
        print(f"[probe] FATAL: library missing: {lib_pth}", file=sys.stderr)
        return 1

    print(f"[probe] calling /predict lib={lib_pth} in={in_pth} (this uploads files; may take a minute)", flush=True)
    t0 = time.time()
    try:
        result = call_predict(client, lib_pth, in_pth, args.similarity_threshold, False, True)
    except Exception as exc:
        print(f"[probe] /predict raised: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"[probe] /predict returned in {time.time()-t0:.0f}s", flush=True)
    _dump(result, "/predict result")
    return 0


def _df_rows(df) -> int:
    """Best-effort row count of a gradio Dataframe return (dict or pandas)."""
    if isinstance(df, dict):
        data = df.get("data")
        if data is None and isinstance(df.get("value"), dict):
            data = df["value"].get("data")
        if data is not None:
            return len(data)
    if hasattr(df, "shape"):
        return int(df.shape[0])
    return -1


def batch(args) -> int:
    import shutil
    from gradio_client import Client
    client = make_client(args.hf_token)
    in_dir = Path(args.input_dir)
    files = sorted(in_dir.glob("*.mzML")) + sorted(in_dir.glob("*.mzml"))
    if args.limit:
        files = files[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lib_pth = args.lib if args.lib else MASS_SPEC_GYM_URL
    print(f"[batch] {len(files)} input files -> {out_dir}", flush=True)
    print(f"[batch] reference library: {lib_pth}", flush=True)
    manifest = []
    for idx, f in enumerate(files, 1):
        tsv_path = out_dir / f"{f.stem}.tsv"
        if tsv_path.exists():
            print(f"[batch] {idx}/{len(files)} {f.name} -> skip (already present)", flush=True)
            manifest.append({"file": f.name, "status": "skip", "seconds": 0})
            continue
        t0 = time.time()
        rec = {"file": f.name}
        try:
            result = call_predict(client, lib_pth, f, args.similarity_threshold, False, True)
            # /predict returns (Dataframe, File-update). The file-update's `value` is the
            # local path gradio_client already downloaded the server-generated .tsv to.
            df, file_update = result
            n_rows = _df_rows(df)
            src = file_update.get("value") if isinstance(file_update, dict) else file_update
            rec["rows"] = n_rows
            if src and Path(src).exists():
                shutil.copy2(src, tsv_path)
                rec["status"] = "ok"
            else:
                (out_dir / f"{f.stem}.json").write_text(
                    json.dumps(_to_serializable(df), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                rec["status"] = "empty-or-no-tsv"
        except Exception as exc:
            rec["status"] = f"error:{type(exc).__name__}"
            rec["error"] = str(exc)[:200]
        rec["seconds"] = round(time.time() - t0, 1)
        manifest.append(rec)
        print(f"[batch] {idx}/{len(files)} {f.name} -> {rec['status']} "
              f"rows={rec.get('rows', '?')} ({rec['seconds']}s)", flush=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--probe", action="store_true", help="one minimal call, dump contract (default)")
    g.add_argument("--batch", action="store_true", help="iterate --input-dir, save raw results")
    p.add_argument("--lib", default=None,
                   help="reference library: local path OR http(s) URL. Default in --batch = official MassSpecGym URL.")
    p.add_argument("--input", default=None, help="single input file for --probe")
    p.add_argument("--input-dir", default=str(ROOT / "data/mtbls13729/mzml/pos_rp"))
    p.add_argument("--out-dir", default=str(ROOT / "data/mtbls13729/dreams_api/pos_rp"))
    p.add_argument("--similarity-threshold", type=float, default=0.5,
                   help="DreaMS similarity cutoff (Space slider min=0.5; its UI default is 0.8, which yields ~0 hits here)")
    p.add_argument("--limit", type=int, default=0, help="max files in --batch (0 = all)")
    p.add_argument("--hf-token", default=None, help="HF token if the Space is private/rate-limited")
    args = p.parse_args()
    if args.batch:
        return batch(args)
    return probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
