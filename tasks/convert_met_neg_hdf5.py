"""Met/neg mzML -> hdf5 批量转换（MSData.load，断点续传）。

Usage (conda dreams_env, CPU):
    python tasks/convert_met_neg_hdf5.py \
        --mzml-dir data/msv100574/Metabolomics/neg

对每个 <stem>.mzML 调用 MSData.load() 生成同目录 <stem>.hdf5，
已存在的 hdf5 跳过。日志打印每个文件的耗时与累计吞吐。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dreams.utils.data import MSData  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mzml-dir", type=Path, required=True)
    args = parser.parse_args()

    mzml_dir: Path = args.mzml_dir
    mzml_files = sorted(mzml_dir.glob("*.mzML")) + sorted(mzml_dir.glob("*.mzml"))
    if not mzml_files:
        print(f"[hdf5] 无 .mzML 文件于 {mzml_dir}", flush=True)
        return

    t_start = time.time()
    done = skip = fail = 0
    for i, mzml in enumerate(mzml_files, 1):
        hdf5 = mzml.with_suffix(".hdf5")
        if hdf5.exists():
            skip += 1
            continue
        t0 = time.time()
        try:
            MSData.load(mzml, in_mem=False)
            done += 1
            print(f"[{i}/{len(mzml_files)}] {mzml.stem} -> hdf5 {time.time()-t0:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"[{i}/{len(mzml_files)}] FAIL {mzml.stem}: {e}", flush=True)

    dt = time.time() - t_start
    print(f"[hdf5] 完成 done={done} skip={skip} fail={fail} 总耗时 {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
