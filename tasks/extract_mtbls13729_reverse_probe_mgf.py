"""Extract the frozen reverse-probe spectra into small, order-locked MGF files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def extract_blocks(source: Path, selection: dict[int, str], destination: Path) -> list[dict[str, object]]:
    record = -1
    block: list[str] | None = None
    written: list[dict[str, object]] = []
    with source.open(encoding="utf-8", errors="replace") as reader, destination.open(
        "w", encoding="utf-8", newline=""
    ) as writer:
        for raw in reader:
            line = raw.strip()
            if line == "BEGIN IONS":
                record += 1
                block = [raw] if record in selection else None
            elif block is not None:
                if line == "END IONS":
                    # The custom field is for provenance only; the encoder ignores it.
                    block.insert(1, f"REVERSE_PROBE_ID={selection[record]}\n")
                    writer.writelines(block)
                    writer.write(raw)
                    written.append({
                        "subset_row": len(written),
                        "source_mgf_record": record,
                        "reference_spectrum_id": selection[record],
                    })
                    block = None
                else:
                    block.append(raw)
    missing = set(selection) - {int(row["source_mgf_record"]) for row in written}
    if missing:
        raise RuntimeError(f"failed to extract MGF records: {sorted(missing)[:10]}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=ROOT / "data/mtbls13729/reverse_probe_manifest_v4",
    )
    parser.add_argument("--reference-neg", type=Path, default=ROOT / "data/reference/unified_v2/unified_neg.mgf")
    parser.add_argument("--reference-pos", type=Path, default=ROOT / "data/reference/unified_v2/unified_pos.mgf")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/mtbls13729/reverse_probe_mgf_v1",
    )
    args = parser.parse_args()
    spectra_path = args.manifest_dir / "reference_spectra.csv.gz"
    if not spectra_path.is_file():
        raise FileNotFoundError(spectra_path)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    spectra = pd.read_csv(spectra_path)
    reports = {}
    all_rows = []
    for panel, source in (("neg_rp", args.reference_neg), ("pos_rp", args.reference_pos)):
        selected = spectra[spectra.panel.eq(panel)]
        selection = dict(zip(selected.mgf_record.astype(int), selected.reference_spectrum_id.astype(str)))
        destination = out / f"{panel}__reverse_probes.mgf"
        rows = extract_blocks(source, selection, destination)
        for row in rows:
            row["panel"] = panel
        all_rows.extend(rows)
        reports[panel] = {
            "spectra": len(rows),
            "mgf_sha256": sha256(destination),
        }
    alignment = pd.DataFrame(all_rows)[
        ["panel", "subset_row", "source_mgf_record", "reference_spectrum_id"]
    ]
    alignment_path = out / "alignment.csv"
    alignment.to_csv(alignment_path, index=False)
    report = {
        "status": "mtbls13729_reverse_probe_mgf_v1_complete",
        "panels": reports,
        "alignment_sha256": sha256(alignment_path),
        "contract": "exact frozen MGF blocks; order locked by alignment.csv; no outcome data",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
