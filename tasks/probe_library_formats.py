"""Probe 4 个参考库原始文件的真实 schema（字段名/格式），好据实写转换器。

下载完跑一次，把每个文件的第一条记录 + 字段名 dump 出来。GB 级文件只读头部，秒出。

用法 (本地 conda dreams_env):
    python tasks/probe_library_formats.py \
        data/reference/massspecgym/data/MassSpecGym.tsv \
        data/reference/gnps/ALL_GNPS.mgf \
        data/reference/massbank/MassBank_NIST.msp \
        data/reference/lipidblast/LipidBlast-pos.msp \
        data/reference/lipidblast/LipidBlast-neg.msp
（路径按实际下载位置改；MassSpecGym 的 .tsv 具体路径用 dir 找一下）
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _trunc(s, n=160):
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def probe_tsv(path: Path) -> None:
    import pandas as pd

    print(f"\n===== TSV: {path.name} ({path.stat().st_size/1e9:.2f} GB) =====")
    df = pd.read_csv(path, sep="\t", nrows=3)
    print(f"列 ({len(df.columns)}): {list(df.columns)}")
    print("首行:")
    for c in df.columns:
        print(f"  {c} = {_trunc(df[c].iloc[0])}")


def _first_block(path: Path, begin: str, end: str) -> list[str]:
    """抓 MGF 的一个 BEGIN IONS...END IONS 块（只读开头一小段）。"""
    lines: list[str] = []
    started = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not started:
                if line.strip() == begin:
                    started = True
            if started:
                lines.append(line.rstrip("\n"))
                if line.strip() == end:
                    break
            if len(lines) > 2000:
                break
    return lines


def probe_mgf(path: Path) -> None:
    print(f"\n===== MGF: {path.name} ({path.stat().st_size/1e9:.2f} GB) =====")
    blk = _first_block(path, "BEGIN IONS", "END IONS")
    fields, npeaks = [], 0
    for ln in blk:
        if "=" in ln:
            fields.append(ln.split("=", 1)[0].strip())
        elif ln.strip() and not ln.startswith(("BEGIN", "END")) and ln[0].isdigit():
            npeaks += 1
    print(f"第一条字段: {fields}")
    print(f"第一谱峰数(块内): {npeaks}")
    print("块前 12 行:")
    for ln in blk[:12]:
        print(f"  {_trunc(ln)}")


def probe_msp(path: Path) -> None:
    print(f"\n===== MSP: {path.name} ({path.stat().st_size/1e9:.2f} GB) =====")
    fields: list[str] = []
    started_peaks = False
    npeaks = 0
    header_lines: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip():
                if started_peaks or header_lines:
                    break
                continue
            if s.startswith("Num Peaks") or s.startswith("Num peaks"):
                started_peaks = True
                fields.append("Num Peaks")
                continue
            if started_peaks:
                npeaks += 1
                if npeaks > 5:
                    break
                continue
            if ":" in s:
                fields.append(s.split(":", 1)[0].strip())
            if len(header_lines) < 12:
                header_lines.append(s)
    print(f"字段: {fields}")
    print("记录头 12 行:")
    for ln in header_lines:
        print(f"  {_trunc(ln)}")
    print(f"(前 {npeaks} 个峰已读)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path)
    args = p.parse_args()
    for f in args.files:
        if not f.exists():
            print(f"[probe] 不存在，跳过: {f}")
            continue
        if f.suffix.lower() in (".tsv", ".csv", ".txt") and "MassSpecGym" in f.name:
            probe_tsv(f)
        elif f.suffix.lower() == ".mgf":
            probe_mgf(f)
        elif f.suffix.lower() == ".msp":
            probe_msp(f)
        else:
            print(f"\n===== 未识别: {f.name} =====")
            with open(f, encoding="utf-8", errors="replace") as fh:
                head = [fh.readline().rstrip("\n") for _ in range(5)]
            for ln in head:
                print(f"  {_trunc(ln)}")
            print("  提示：先手动看头部，告诉我字段名。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
