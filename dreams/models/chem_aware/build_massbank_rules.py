"""build_massbank_rules.py — MassBank 规则抽取与增量构建脚本。

目标：
- 为 MassBank / GNPS / ChemFrag / CFM-ID 这类高置信来源提供统一的规则抽取入口
- 在没有真实源文件时，先生成模板文件与空审查结果，便于后续填充
- 输出与现有 chem_rules_data.json 兼容的 MassBank 增量规则文件

说明：
- 这是 MassBank 的脚手架，不修改现有规则库，不做删减
- 真实的外部来源文件可以后续按模板放入 massbank_sources/ 目录

用法：
  python build_massbank_rules.py
  python build_massbank_rules.py --init-templates
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import median
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
MassBank_DIR = HERE / "massbank_sources"
OUT_JSON = HERE / "chem_rules_massbank.json"
OUT_CSV = HERE / "chem_rules_massbank_rules.csv"
OUT_MD = HERE / "chem_rules_massbank_report.md"
SUMMARY_JSON = HERE / "chem_rules_massbank_summary.json"

EXPECTED_SOURCE_FILES = {
    "MassBank": ["massbank_rules.csv", "massbank_rules.json", "massbank_rules.tsv"],
    "GNPS": ["gnps_rules.csv", "gnps_rules.json", "gnps_rules.tsv"],
    "ChemFrag": ["chemfrag_rules.csv", "chemfrag_rules.json", "chemfrag_rules.tsv"],
    "CFM-ID": ["cfmid_rules.csv", "cfmid_rules.json", "cfmid_rules.tsv"],
}

TEMPLATE_COLUMNS = [
    "name", "category", "match_type", "value", "mode", "scope",
    "evidence_level", "support", "confidence", "source", "notes",
]


@dataclass
class MassBankRule:
    name: str
    category: str
    match_type: str
    value: float | list[float]
    source: str
    mode: str = "unknown"
    scope: str = "unknown"
    evidence_level: str = "medium"
    support: int = 0
    confidence: str = "unknown"
    notes: str = ""
    alias_group: str = ""
    tier: str = "extended"
    enabled_by_default: bool = False
    recommended_action: str = "keep"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MassBank incremental rule files")
    p.add_argument("--input-dir", type=str, default=str(MassBank_DIR), help="Directory containing MassBank source files")
    p.add_argument("--massbank-dir", type=str, default="", help="MassBank-data-dev root containing record txt files")
    p.add_argument("--output-json", type=str, default=str(OUT_JSON), help="Output incremental JSON")
    p.add_argument("--init-templates", action="store_true", help="Create source templates and exit")
    return p.parse_args()


def ensure_massbank_dir() -> Path:
    MassBank_DIR.mkdir(parents=True, exist_ok=True)
    return MassBank_DIR


def init_templates() -> None:
    massbank_dir = ensure_massbank_dir()
    manifest = {
        "purpose": "MassBank source manifest",
        "expected_fields": TEMPLATE_COLUMNS,
        "expected_sources": EXPECTED_SOURCE_FILES,
        "notes": [
            "Put source files here and rerun build_massbank_rules.py.",
            "CSV/TSV files should contain a header row matching the template columns.",
            "JSON files may be either a list of rule dicts or an object with a 'rules' array.",
        ],
    }
    (massbank_dir / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for source_name, files in EXPECTED_SOURCE_FILES.items():
        for fname in files:
            path = massbank_dir / fname
            if path.suffix.lower() in {".csv", ".tsv"} and not path.exists():
                with path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=TEMPLATE_COLUMNS)
                    writer.writeheader()
            elif path.suffix.lower() == ".json" and not path.exists():
                path.write_text("[]\n", encoding="utf-8")
    print(f"[MassBank] 模板已生成: {massbank_dir}")


def canonical_mass(value: float, tol: float = 0.005) -> float:
    return round(value / tol) * tol


def infer_alias_group(rule: dict[str, Any], tol: float = 0.005) -> str:
    mt = rule.get("match_type", "")
    if mt in {"mass_diff", "peak_mz"}:
        return f"{mt}:{canonical_mass(float(rule['value']), tol):.4f}"
    if mt == "mass_range":
        lo, hi = rule["value"]
        return f"mass_range:{round(float(lo), 4):.4f}-{round(float(hi), 4):.4f}"
    return f"{mt}:{rule.get('name', '')}"


def normalize_value(raw: Any, match_type: str) -> float | list[float]:
    if match_type == "mass_range":
        if isinstance(raw, str):
            parts = [float(x) for x in re.split(r"[,;\s]+", raw.strip()) if x]
            if len(parts) != 2:
                raise ValueError(f"mass_range 需要两个值, got={raw!r}")
            return [parts[0], parts[1]]
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return [float(raw[0]), float(raw[1])]
        raise ValueError(f"无法解析 mass_range 值: {raw!r}")
    if isinstance(raw, (list, tuple)):
        if len(raw) == 1:
            return float(raw[0])
        raise ValueError(f"非 mass_range 不应为多值: {raw!r}")
    return float(raw)


def default_tier(rule: dict[str, Any]) -> str:
    evidence = rule.get("evidence_level", "medium")
    scope = (rule.get("scope") or "unknown").lower()
    cat = rule.get("category", "")
    name = rule.get("name", "")

    if evidence == "high" and scope in {"general", "unknown"}:
        return "core"
    if cat == "HR":
        return "core"
    if cat == "ISO" and name in {"ISO:Cl35_Cl37", "ISO:Br79_Br81", "ISO:S32_S34"}:
        return "core"
    return "extended"


def default_action(rule: dict[str, Any]) -> tuple[str, bool]:
    tier = rule.get("tier", "extended")
    evidence = rule.get("evidence_level", "medium")
    specificity = rule.get("specificity", "medium")
    cat = rule.get("category", "")

    if cat in {"NR", "EE"} and tier != "core":
        return "downgrade", False
    if tier == "core" and evidence == "high" and specificity != "low":
        return "keep", True
    if tier == "extended" and evidence in {"high", "medium"} and specificity != "low":
        return "keep", True
    if evidence == "low" or specificity == "low":
        return "disable", False
    return "downgrade", False


def infer_specificity(rule: dict[str, Any], alias_count: int) -> str:
    cat = rule.get("category", "")
    name = rule.get("name", "")
    base = name.split(":", 1)[-1]
    generic = {
        "H2O", "NH3", "CO", "CO2", "CH2O", "CH3OH", "HCOOH", "CH3COOH",
        "C2H4", "C3H6", "C4H8", "C6H12", "CH3CN", "C5H8", "CH3", "C2H5",
        "acetyl", "propionyl", "butyryl", "phenyl", "tropylium", "benzyl",
        "phosphate_frag", "hexose_oxonium", "pentose_oxonium", "deoxyhexose_oxonium",
        "disaccharide_oxonium",
    }
    high_specific = {
        "phosphocholine_head", "N_acetylhexosamine", "caffeine_frag", "guanine_frag",
        "cytosine_frag", "indole_frag", "cholesterol_skel",
        "glucuronide", "glucuronide+H2O", "glutathione", "cysteinylglycine",
        "taurine_conj", "Gly_res", "Ala_res", "Phe_res", "Tyr_res", "Trp_res",
    }

    if cat == "HR":
        return "high"
    if base in high_specific:
        return "high"
    if base in generic:
        return "low"
    if alias_count > 1:
        return "medium"
    if cat in {"NR", "EE"}:
        return "low"
    return "medium"


def infer_frequency(rule: dict[str, Any], alias_count: int) -> str:
    cat = rule.get("category", "")
    name = rule.get("name", "")
    base = name.split(":", 1)[-1]
    if base in {"H2O", "NH3", "CO", "CO2", "CH2O", "CH3OH", "HCOOH", "CH3COOH", "C2H4", "C3H6"}:
        return "high"
    if cat == "HR":
        return "medium"
    if cat == "ISO":
        return "low"
    if alias_count > 1:
        return "medium"
    if cat in {"NR", "EE"}:
        return "medium"
    return "medium"


def load_csv_rules(path: Path, source_name: str) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;") if sample.strip() else csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            if not row:
                continue
            name = (row.get("name") or row.get("Name") or "").strip()
            if not name:
                continue
            match_type = (row.get("match_type") or row.get("match") or "mass_diff").strip()
            raw_value = row.get("value") or row.get("Value") or row.get("mass") or row.get("mz") or ""
            if not raw_value:
                continue
            rule = {
                "name": name,
                "category": (row.get("category") or row.get("Category") or "").strip() or name.split(":", 1)[0],
                "match_type": match_type,
                "value": normalize_value(raw_value, match_type),
                "source": (row.get("source") or source_name).strip(),
                "mode": (row.get("mode") or "unknown").strip() or "unknown",
                "scope": (row.get("scope") or row.get("class") or "unknown").strip() or "unknown",
                "evidence_level": (row.get("evidence_level") or "medium").strip() or "medium",
                "support": int(float(row.get("support") or row.get("count") or 0)),
                "confidence": (row.get("confidence") or "unknown").strip() or "unknown",
                "notes": (row.get("notes") or row.get("ref") or "").strip(),
            }
            rules.append(rule)
    return rules


def load_json_rules(path: Path, source_name: str) -> list[dict[str, Any]]:
    if path.stat().st_size == 0:
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw or raw in {"[]", "{}"}:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict) and "rules" in data:
        data = data["rules"]
    if not isinstance(data, list):
        raise ValueError(f"不支持的 JSON 结构: {path}")

    rules: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        if not name:
            continue
        match_type = (item.get("match_type") or "mass_diff").strip()
        value = item.get("value")
        if value is None:
            continue
        rule = {
            "name": name,
            "category": (item.get("category") or name.split(":", 1)[0]).strip(),
            "match_type": match_type,
            "value": normalize_value(value, match_type),
            "source": (item.get("source") or source_name).strip(),
            "mode": (item.get("mode") or "unknown").strip() or "unknown",
            "scope": (item.get("scope") or item.get("class") or "unknown").strip() or "unknown",
            "evidence_level": (item.get("evidence_level") or "medium").strip() or "medium",
            "support": int(float(item.get("support") or item.get("count") or 0)),
            "confidence": (item.get("confidence") or "unknown").strip() or "unknown",
            "notes": (item.get("notes") or item.get("ref") or "").strip(),
        }
        rules.append(rule)
    return rules


def discover_source_files(input_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not input_dir.exists():
        return files
    for pattern in ("*.csv", "*.tsv", "*.json"):
        files.extend(sorted(input_dir.glob(pattern)))
    return [p for p in files if p.is_file() and p.name != "source_manifest.json"]


MASSBANK_KEYS = {
    "accession": re.compile(r"^ACCESSION:\s*(.+)$", re.M),
    "title": re.compile(r"^RECORD_TITLE:\s*(.+)$", re.M),
    "formula": re.compile(r"^CH\$FORMULA:\s*(.+)$", re.M),
    "exact_mass": re.compile(r"^CH\$EXACT_MASS:\s*(.+)$", re.M),
    "compound_class": re.compile(r"^CH\$COMPOUND_CLASS:\s*(.+)$", re.M),
    "ion_mode": re.compile(r"^AC\$MASS_SPECTROMETRY:\s*ION_MODE\s+(.+)$", re.M),
    "precursor_mz": re.compile(r"^MS\$FOCUSED_ION:\s*PRECURSOR_M/Z\s+(.+)$", re.M),
    "precursor_type": re.compile(r"^MS\$FOCUSED_ION:\s*PRECURSOR_TYPE\s+(.+)$", re.M),
    "peak": re.compile(r"^\s*(\d+\.\d+)\s+\d+(?:\.\d+)?\s+\d+\s*$", re.M),
    "pk_peak": re.compile(r"^PK\$PEAK:\s*m/z\s+int\.\s+rel\.int\.$|^\s*(\d+\.\d+)\s+\d+(?:\.\d+)?\s+\d+\s*$", re.M),
}


def parse_massbank_record(path: Path) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "PK$PEAK" not in text:
        return None

    def _first(key: str, default: str = "") -> str:
        m = MASSBANK_KEYS[key].search(text)
        return m.group(1).strip() if m else default

    peaks_block = text.split("PK$PEAK:", 1)
    peaks = []
    if len(peaks_block) == 2:
        for line in peaks_block[1].splitlines()[1:]:
            line = line.strip()
            if not line or line == "//":
                break
            m = re.match(r"^(\d+\.\d+)\s+\d+(?:\.\d+)?\s+\d+", line)
            if m:
                peaks.append(float(m.group(1)))
    if not peaks:
        peaks = [float(m.group(1)) for m in MASSBANK_KEYS["peak"].finditer(text)]
    if not peaks:
        return None

    accession = _first("accession") or path.stem
    title = _first("title") or accession
    formula = _first("formula")
    exact_mass = _first("exact_mass")
    compound_class = _first("compound_class")
    ion_mode = _first("ion_mode", "unknown").lower()
    precursor_mz = _first("precursor_mz")
    precursor_type = _first("precursor_type")

    return {
        "accession": accession,
        "title": title,
        "formula": formula,
        "exact_mass": float(exact_mass) if exact_mass and exact_mass != "NA" else None,
        "compound_class": compound_class,
        "ion_mode": ion_mode,
        "precursor_mz": float(precursor_mz) if precursor_mz and precursor_mz != "NA" else None,
        "precursor_type": precursor_type,
        "peaks": peaks,
    }


def infer_massbank_rules(record: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    accession = record["accession"]
    title = record["title"]
    precursor_mz = record.get("precursor_mz")
    exact_mass = record.get("exact_mass")
    ion_mode = record.get("ion_mode", "unknown")
    peaks = sorted(set(round(p, 4) for p in record.get("peaks", [])))

    if precursor_mz and exact_mass:
        neutral_loss = round(abs(float(precursor_mz) - float(exact_mass)), 4)
        if 0.5 <= neutral_loss <= 600:
            rules.append({
                "name": f"NL:massbank_{accession}",
                "category": "NL",
                "match_type": "mass_diff",
                "value": neutral_loss,
                "source": "MassBank record-derived",
                "mode": ion_mode,
                "scope": record.get("compound_class", "unknown") or "unknown",
                "evidence_level": "medium",
                "support": 1,
                "confidence": "observed",
                "notes": title,
            })

    for mz in peaks[:3]:
        if 30.0 <= mz <= 700.0:
            rules.append({
                "name": f"CF:massbank_{accession}_{str(mz).replace('.', '_')}",
                "category": "CF",
                "match_type": "peak_mz",
                "value": mz,
                "source": "MassBank record-derived",
                "mode": ion_mode,
                "scope": record.get("compound_class", "unknown") or "unknown",
                "evidence_level": "medium",
                "support": 1,
                "confidence": "observed",
                "notes": title,
            })
    return rules


def parse_rule_files(files: list[Path]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for p in files:
        stem = p.stem.lower()
        if "massbank" in stem:
            source_name = "MassBank"
        elif "gnps" in stem:
            source_name = "GNPS"
        elif "chemfrag" in stem:
            source_name = "ChemFrag"
        elif "cfmid" in stem or "cfm" in stem:
            source_name = "CFM-ID"
        else:
            source_name = p.stem

        if p.suffix.lower() in {".csv", ".tsv"}:
            rules.extend(load_csv_rules(p, source_name))
        elif p.suffix.lower() == ".json":
            rules.extend(load_json_rules(p, source_name))
    return rules


def build_alias_map(rules: list[dict[str, Any]]) -> dict[str, int]:
    groups = defaultdict(list)
    for r in rules:
        groups[infer_alias_group(r)].append(r)
    return {k: len(v) for k, v in groups.items()}


def attach_metadata(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alias_map = build_alias_map(rules)
    out: list[dict[str, Any]] = []
    for r in rules:
        ag = infer_alias_group(r)
        alias_count = alias_map.get(ag, 1)
        rec = dict(r)
        rec["alias_group"] = ag
        rec["tier"] = default_tier(rec)
        rec["enabled_by_default"] = rec["tier"] == "core" and rec.get("evidence_level", "medium") != "low"
        rec["specificity"] = infer_specificity(rec, alias_count)
        rec["expected_frequency"] = infer_frequency(rec, alias_count)
        rec["recommended_action"], enabled = default_action(rec)
        rec["enabled_by_default"] = bool(rec["enabled_by_default"] or enabled)
        out.append(rec)
    return out


def summarize_rules(rules: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_rules": len(rules),
        "by_category": dict(Counter(r.get("category", "") for r in rules)),
        "by_source": dict(Counter(r.get("source", "") for r in rules)),
        "by_tier": dict(Counter(r.get("tier", "") for r in rules)),
        "by_evidence_level": dict(Counter(r.get("evidence_level", "") for r in rules)),
        "by_specificity": dict(Counter(r.get("specificity", "") for r in rules)),
        "by_expected_frequency": dict(Counter(r.get("expected_frequency", "") for r in rules)),
        "by_recommended_action": dict(Counter(r.get("recommended_action", "") for r in rules)),
        "by_enabled_by_default": dict(Counter(str(bool(r.get("enabled_by_default", False))) for r in rules)),
    }


def build_massbank_rules(massbank_dir: Path, limit: int = 2000) -> list[dict[str, Any]]:
    files = sorted(massbank_dir.rglob("*.txt")) if massbank_dir.exists() else []
    records = []
    for path in files[:limit]:
        rec = parse_massbank_record(path)
        if rec:
            records.append(rec)

    if not records:
        return []

    out: list[dict[str, Any]] = []
    for rec in records:
        out.extend(infer_massbank_rules(rec))

    deduped: list[dict[str, Any]] = []
    seen = set()
    for r in out:
        key = (r["category"], r["match_type"], round(float(r["value"]), 4))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def write_outputs(rules: list[dict[str, Any]], input_dir: Path) -> None:
    summary = summarize_rules(rules)
    payload = {
        "version": "massbank-framework",
        "input_dir": str(input_dir),
        "rules": rules,
        "summary": summary,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name", "category", "match_type", "value", "source", "mode", "scope",
                "evidence_level", "support", "confidence", "notes", "alias_group",
                "tier", "enabled_by_default", "specificity", "expected_frequency",
                "recommended_action",
            ],
        )
        writer.writeheader()
        for r in rules:
            row = dict(r)
            if isinstance(row["value"], list):
                row["value"] = json.dumps(row["value"], ensure_ascii=False)
            writer.writerow(row)

    lines: list[str] = []
    lines.append("# MassBank 规则提取报告")
    lines.append("")
    lines.append(f"- 输入目录: `{input_dir}`")
    lines.append(f"- 规则总数: `{summary['n_rules']}`")
    lines.append("")
    lines.append("## 分类统计")
    for k, v in sorted(summary["by_category"].items(), key=lambda x: x[0]):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 来源统计")
    for k, v in sorted(summary["by_source"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 分层统计")
    for k, v in sorted(summary["by_tier"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 证据等级统计")
    for k, v in sorted(summary["by_evidence_level"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 特异性统计")
    for k, v in sorted(summary["by_specificity"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 预期频率统计")
    for k, v in sorted(summary["by_expected_frequency"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 推荐动作统计")
    for k, v in sorted(summary["by_recommended_action"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 默认启用统计")
    for k, v in sorted(summary["by_enabled_by_default"].items(), key=lambda x: x[0]):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 下一步")
    lines.append("- 将高置信外部规则文件放入 `massbank_sources/` 后重新运行脚本。")
    lines.append("- 根据 CSV / JSON 输出，筛选 core 与 extended 候选。")
    lines.append("- 若没有源文件，本脚本会保持为模板与空审查结果的框架。")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[MassBank] 规则数: {summary['n_rules']}")
    print(f"[MassBank] 输出: {OUT_JSON.name}, {OUT_CSV.name}, {OUT_MD.name}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    massbank_dir = Path(args.massbank_dir) if args.massbank_dir else None
    if args.init_templates:
        init_templates()
        return

    init_templates()
    files = discover_source_files(input_dir)
    rules = parse_rule_files(files) if files else []
    if massbank_dir is not None:
        rules.extend(build_massbank_rules(massbank_dir))
    rules = attach_metadata(rules)
    write_outputs(rules, input_dir)


if __name__ == "__main__":
    main()
