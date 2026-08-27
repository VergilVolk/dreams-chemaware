"""DreaMS-based ChemAware public Gradio showcase."""

from __future__ import annotations

import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from config import APP_CONFIG, RULE_CATEGORIES_INFO
from inference_utils import (
    analyze_chemical_rules,
    batch_generate_embeddings,
    calculate_cosine_similarity,
    export_embedding_to_csv,
    generate_embedding,
    get_runtime_device,
    load_model,
    validate_spectrum,
)
try:
    from molecule_matcher import MoleculeDatabase
except ImportError:
    MoleculeDatabase = None

try:
    from smiles_visualizer import get_molecule_info, smiles_to_base64
except ImportError:
    get_molecule_info = None
    smiles_to_base64 = None

try:
    from rdkit import Chem
except ImportError:
    Chem = None

from spectrum_parser import SpectrumParser

SUPPORTED_FORMATS = [".mgf", ".mzml", ".mzxml", ".hdf5", ".h5", ".hd5", ".json"]
DEFAULT_SPECTRUM = "[[100.5, 0.8], [200.3, 1.0], [301.2, 0.5], [402.1, 0.6], [500.0, 0.9]]"


def parse_spectrum_input(text: str) -> np.ndarray:
    """Parse JSON or two-column text into an Nx2 peak array."""
    text = (text or "").strip()
    if not text:
        raise ValueError("请输入谱图数据")

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("peaks", data.get("spectrum", data))
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            parts = line.replace(",", " ").replace("\t", " ").split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError("逐行输入时，每行必须包含 m/z 和 intensity 两个数值")
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except ValueError as exc:
                raise ValueError("峰值必须是数字") from exc
        data = rows

    peaks = np.asarray(data, dtype=np.float32)
    if peaks.ndim != 2 or peaks.shape[1] != 2:
        raise ValueError("谱图必须是形如 [[m/z, intensity], ...] 的二维数组")
    return peaks


def validate_record(record: Dict[str, Any], official: bool = False) -> Dict[str, Any]:
    """Validate one parsed spectrum record and normalize metadata."""
    peaks = validate_spectrum(
        record["peaks"], float(record.get("precursor_mz", 0)), int(record.get("charge", 1))
    )
    precursor_mz = float(record.get("precursor_mz"))
    charge = int(record.get("charge", 1))
    if official:
        if charge != 1:
            raise ValueError("官方 DreaMS embedding 目前只支持 charge=1")
        if len(peaks) < 3:
            raise ValueError("官方 DreaMS embedding 至少需要 3 个峰")
        if precursor_mz > 1000 or float(np.max(peaks[:, 0])) > 1000:
            raise ValueError("官方 DreaMS DataFormatA 要求 m/z 不超过 1000")
    return {
        "spectrum_id": str(record.get("spectrum_id", "spectrum")),
        "precursor_mz": precursor_mz,
        "charge": charge,
        "peaks": peaks,
    }


def make_spectrum_plot(peaks: np.ndarray, precursor_mz: float):
    """Create a labeled stick plot."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.vlines(peaks[:, 0], 0, peaks[:, 1], color="#5b6cff", linewidth=1.2)
    ax.scatter(peaks[:, 0], peaks[:, 1], color="#4338ca", s=12, zorder=2)
    ax.axvline(precursor_mz, color="#b45309", linestyle="--", linewidth=1, label="precursor m/z")
    ax.set_xlabel("m/z")
    ax.set_ylabel("intensity")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def safe_spectrum_plot(text: str, precursor_mz: float):
    """Return a preview plot or None for incomplete input."""
    try:
        peaks = parse_spectrum_input(text)
        if not np.isfinite(precursor_mz) or precursor_mz <= 0:
            return None
        return make_spectrum_plot(peaks, precursor_mz)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def format_rules_analysis(rules_data: dict) -> str:
    """Format heuristic chemical-rule evidence for the UI."""
    if not rules_data:
        return "### 化学规则提示\n\n暂无结果。"
    lines = ["### 化学规则提示", "", "> 这些是谱图模式的启发式提示，不是结构鉴定或鉴定置信度。", ""]
    matched = False
    for category in ["NL", "CF", "ISO", "NR", "EE", "HR"]:
        items = rules_data.get(category, [])
        if not items:
            continue
        matched = True
        info = RULE_CATEGORIES_INFO.get(category, {})
        lines.append(f"**{info.get('name', category)}**")
        lines.extend(f"- {item['name']}: {item['description']}" for item in items[:5])
        lines.append("")
    if not matched:
        lines.append("未发现明显规则模式。")
    return "\n".join(lines)


def choice_index(choice: Any) -> int:
    """Extract the spectrum/candidate index from a Gradio choice."""
    if isinstance(choice, (tuple, list)):
        choice = choice[-1] if choice else "0"
    return int(str(choice).split(":", 1)[0])


def candidate_columns() -> List[str]:
    return [
        "Rank", "Precursor m/z", "Delta m/z", "Ref. name", "Ref. ID",
        "Formula", "SMILES", "DreaMS similarity",
    ]


def deduplicate_candidates(candidates: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
    """Keep the highest-similarity candidate for each molecular structure."""
    best_by_structure: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        smiles = str(candidate.get("SMILES", "")).strip()
        if not smiles or smiles in {"N/A", "nan"}:
            key = f"candidate:{candidate.get('Ref. ID', id(candidate))}"
        elif Chem is not None:
            molecule = Chem.MolFromSmiles(smiles)
            key = Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else f"smiles:{smiles}"
        else:
            key = f"smiles:{smiles}"

        similarity = float(candidate.get("DreaMS similarity", 0.0))
        previous = best_by_structure.get(key)
        if previous is None or similarity > float(previous.get("DreaMS similarity", 0.0)):
            best_by_structure[key] = candidate

    unique = sorted(
        best_by_structure.values(),
        key=lambda candidate: float(candidate.get("DreaMS similarity", 0.0)),
        reverse=True,
    )[:max_results]
    for rank, candidate in enumerate(unique, 1):
        candidate["Rank"] = rank
    return unique


def candidate_table_rows(candidates: List[Dict[str, Any]]) -> List[List[Any]]:
    """Convert candidate records to the fixed Dataframe schema."""
    return [[candidate.get(column, "") for column in candidate_columns()] for candidate in candidates]


def format_molecule_info(smiles: str) -> str:
    """Format RDKit descriptors."""
    if not smiles or smiles in {"N/A", "nan"}:
        return "### 分子信息\n\n该候选没有可用 SMILES。"
    if get_molecule_info is None:
        return "### 分子信息\n\n当前环境未安装 RDKit。"
    info = get_molecule_info(smiles)
    if not info:
        return "### 分子信息\n\nSMILES 无效，无法生成分子性质。"
    return "### 分子信息\n\n" + "\n".join([
        f"- 分子量: {info['molecular_weight']}",
        f"- LogP: {info['logp']}",
        f"- H 供体: {info['num_h_donors']}",
        f"- H 受体: {info['num_h_acceptors']}",
        f"- 可旋转键: {info['num_rotatable_bonds']}",
        f"- 原子数 / 重原子数: {info['num_atoms']} / {info['num_heavy_atoms']}",
    ])


class DreaMSInterface:
    """Stateful adapter used by the Gradio callbacks."""

    def __init__(self):
        self.model = None
        self.chem_rule_engine = None
        self.loaded_model_type: Optional[str] = None
        self.device = get_runtime_device()
        self.database: Optional[MoleculeDatabase] = None
        self.database_error: Optional[str] = None

    def ensure_model(self, model_type: str = "official_dreams") -> None:
        if self.loaded_model_type != model_type:
            self.model, self.chem_rule_engine = load_model(model_type=model_type, device=self.device)
            self.loaded_model_type = model_type

    def load_model_wrapper(self, model_type: str = "official_dreams") -> str:
        try:
            self.ensure_model(model_type)
            if model_type == "official_dreams":
                return (
                    "✅ 官方 DreaMS embedding 后端已就绪\n\n"
                    f"- 设备: `{self.device}`\n"
                    "- 模式: `official_dreams`\n"
                    "- 权重: `embedding_model.ckpt` + `ssl_model.ckpt`\n"
                    "- 预处理: 官方 SpectrumPreprocessor / DataFormatA"
                )
            return (
                "✅ 演示后端已就绪\n\n"
                f"- 设备: `{self.device}`\n"
                "- 模式: `demo`\n"
                "- 注意: 演示向量不代表真实 DreaMS embedding"
            )
        except Exception as exc:
            return f"❌ 后端初始化失败: {type(exc).__name__}"

    def parse_uploaded(self, file_path: Optional[str]):
        """Parse an uploaded multi-spectrum file."""
        if not file_path:
            return gr.update(choices=[], value=None), [], "请先上传质谱文件。"
        try:
            records = SpectrumParser.parse_file(file_path)
            normalized = [validate_record(record) for record in records]
            state_records = [
                {**record, "peaks": record["peaks"].tolist()} for record in normalized
            ]
            choices = [
                f"{i}: {record['spectrum_id']} · m/z {record['precursor_mz']:.4f} · {len(record['peaks'])} peaks"
                for i, record in enumerate(normalized)
            ]
            return gr.update(choices=choices, value=choices[0] if choices else None), state_records, (
                f"✅ 已解析 {len(normalized)} 个谱图。请选择一个进行单谱分析，或直接运行批量分析。"
            )
        except ImportError as exc:
            return gr.update(choices=[], value=None), [], f"❌ 当前格式需要额外依赖: {str(exc)}"
        except (ValueError, OSError) as exc:
            return gr.update(choices=[], value=None), [], f"❌ 文件解析失败: {str(exc)}"
        except Exception:
            return gr.update(choices=[], value=None), [], "❌ 文件解析失败，请检查格式和文件内容。"

    @staticmethod
    def select_uploaded(choice: Optional[str], records: List[Dict[str, Any]]):
        """Load one selected record into the common text-input flow."""
        if not records or not choice:
            return "", None, 1, None, []
        index = choice_index(choice)
        record = records[index]
        peaks = np.asarray(record["peaks"], dtype=np.float32)
        return (
            json.dumps(peaks.tolist()),
            record["precursor_mz"],
            record["charge"],
            make_spectrum_plot(peaks, record["precursor_mz"]),
            peaks.tolist(),
        )

    def _get_database(self) -> Tuple[Optional[MoleculeDatabase], Optional[str]]:
        if self.database is not None or self.database_error is not None:
            return self.database, self.database_error
        try:
            if MoleculeDatabase is None:
                self.database_error = "候选功能需要安装 h5py 和 pandas"
                return self.database, self.database_error
            database = MoleculeDatabase()
            if database.embeddings is None:
                self.database_error = (
                    "未找到 MassSpecGym_DreaMS.hdf5；请设置 DREAMS_MOLECULE_DB 指向数据库文件"
                )
            else:
                self.database = database
        except Exception as exc:
            self.database_error = f"候选数据库不可用（{type(exc).__name__}）"
        return self.database, self.database_error

    def _match_candidates(self, embedding: np.ndarray, precursor_mz: float, tolerance_da: float, max_results: int):
        database, error = self._get_database()
        if error or database is None:
            return None, error or "候选数据库不可用"
        try:
            return database.get_candidate_molecules(embedding, precursor_mz, tolerance_da, max_results), None
        except Exception:
            return None, "候选检索失败，请检查数据库结构"

    def process_spectrum(
        self, spectrum_text: str, precursor_mz: float, charge: int, analyze_rules: bool,
        model_type: str, tolerance_da: float = 0.05, max_results: int = 10,
    ):
        """Run one spectrum through embedding, rules, matching and exports."""
        empty = ("", "", "", None, [], [], "", None, None, [])
        try:
            peaks = parse_spectrum_input(spectrum_text)
            record = validate_record({
                "spectrum_id": "manual_input",
                "peaks": peaks,
                "precursor_mz": precursor_mz,
                "charge": charge,
            }, official=model_type == "official_dreams")["peaks"]
            self.ensure_model(model_type)
            embedding, metadata = generate_embedding(
                self.model, record, float(precursor_mz), int(charge), self.device,
                chem_aware=False,
            )
            rules = analyze_chemical_rules(self.chem_rule_engine, record, float(precursor_mz)) if analyze_rules else {}
            candidates, match_error = self._match_candidates(embedding, float(precursor_mz), float(tolerance_da), int(max_results) * 2)
            candidates_data = candidates.to_dict("records") if candidates is not None and not candidates.empty else []
            candidates_data = deduplicate_candidates(candidates_data, int(max_results))
            stats = f"""### ✅ 单谱分析完成

**输入质量**
- 峰数: {len(record)}
- 前体 m/z: {float(precursor_mz):.4f}
- 电荷: {int(charge)}
- 处理时间: {metadata.get('processing_time', 0):.3f}s

**Embedding**
- 维度: {len(embedding)}
- L2 范数: {np.linalg.norm(embedding):.6f}
- 后端: `{metadata.get('backend')}`
- 设备: `{metadata.get('device')}`
"""
            embedding_md = "### Embedding 前 10 个分量\n\n```text\n" + "\n".join(
                f"{i}: {value:.6f}" for i, value in enumerate(embedding[:10])
            ) + f"\n... 共 {len(embedding)} 维\n```"
            candidate_status = (
                f"⚠️ {match_error}。候选功能需要缓存数据库后才能使用。"
                if match_error else "✅ 候选已按前体 m/z 和 embedding 相似度排序；不是鉴定置信度。"
            )
            rules_md = format_rules_analysis(rules) + "\n\n" + candidate_status
            result = {
                "spectrum_id": "manual_input",
                "peaks": record.tolist(),
                "precursor_mz": float(precursor_mz), "charge": int(charge),
                "model_type": model_type, "metadata": metadata,
                "embedding": embedding.tolist(), "chemical_rules": rules,
                "candidates": candidates_data,
            }
            json_path = self._write_json(result)
            csv_path = self._write_csv([(embedding, {"spectrum_id": "manual_input", **metadata})])
            return (
                stats, embedding_md, rules_md, make_spectrum_plot(record, float(precursor_mz)),
                record.tolist(), candidate_table_rows(candidates_data), candidate_status,
                json_path, csv_path, candidates_data
            )
        except (ValueError, TypeError) as exc:
            return (*empty[:-1], f"❌ 输入错误: {str(exc)}")
        except Exception as exc:
            return (*empty[:-1], f"❌ 分析失败: {type(exc).__name__}")

    def compare_models(self, spectrum_text: str, precursor_mz: float, charge: int = 1) -> str:
        """Compare the two reproducible demo representation modes."""
        try:
            peaks = parse_spectrum_input(spectrum_text)
            peaks = validate_spectrum(peaks, float(precursor_mz), int(charge))
            self.ensure_model("official_dreams")
            chem_embedding, _ = generate_embedding(
                self.model, peaks, float(precursor_mz), int(charge), self.device, chem_aware=True
            )
            base_embedding, _ = generate_embedding(
                self.model, peaks, float(precursor_mz), int(charge), self.device, chem_aware=False
            )
            cosine = calculate_cosine_similarity(chem_embedding, base_embedding)
            distance = float(np.linalg.norm(chem_embedding - base_embedding))
            return (
                "### 演示表示对比\n\n"
                "> 这是同一可复现演示后端的两种模式对比，不是两个真实 checkpoint 的性能评测。\n\n"
                f"- Cosine 相似度: {cosine:.6f}\n"
                f"- Euclidean 距离: {distance:.6f}\n"
                f"- 设备: `{self.device}`"
            )
        except (ValueError, TypeError) as exc:
            return f"❌ 输入错误: {str(exc)}"
        except Exception as exc:
            return f"❌ 对比失败: {type(exc).__name__}"

    @staticmethod
    def _write_json(payload: dict) -> str:
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="dreams_result_", delete=False, encoding="utf-8")
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return handle.name

    @staticmethod
    def _write_csv(items) -> str:
        handle = tempfile.NamedTemporaryFile(suffix=".csv", prefix="dreams_embedding_", delete=False)
        handle.close()
        export_embedding_to_csv(items, handle.name)
        return handle.name

    def render_molecules(self, top_n: int, candidates: List[Dict[str, Any]]):
        """Render each of the top N candidates as one self-contained card."""
        if not candidates:
            return "<div class='candidate-empty'>暂无可展示的候选分子。</div>"

        top_n = max(1, min(int(top_n), len(candidates)))
        cards = []
        for index, candidate in enumerate(candidates[:top_n], 1):
            smiles = str(candidate.get("SMILES", ""))
            image_html = "<div class='candidate-no-image'>暂无结构图</div>"
            if smiles_to_base64 and smiles not in {"", "N/A", "nan"}:
                image_data = smiles_to_base64(smiles, size=260)
                if image_data:
                    image_html = f"<img src='data:image/png;base64,{image_data}' alt='候选 {index} 的分子结构'>"

            info = get_molecule_info(smiles) if get_molecule_info and smiles not in {"", "N/A", "nan"} else None
            info_html = "<p class='muted'>无法计算分子性质（需要有效 SMILES 和 RDKit）。</p>"
            if info:
                info_html = "".join([
                    f"<div><b>分子量</b><br>{html.escape(str(info['molecular_weight']))}</div>",
                    f"<div><b>LogP</b><br>{html.escape(str(info['logp']))}</div>",
                    f"<div><b>H 供体</b><br>{info['num_h_donors']}</div>",
                    f"<div><b>H 受体</b><br>{info['num_h_acceptors']}</div>",
                    f"<div><b>可旋转键</b><br>{info['num_rotatable_bonds']}</div>",
                    f"<div><b>重原子数</b><br>{info['num_heavy_atoms']}</div>",
                ])

            name = html.escape(str(candidate.get("Ref. name", candidate.get("Ref. ID", "candidate"))))
            formula = html.escape(str(candidate.get("Formula", "N/A")))
            escaped_smiles = html.escape(smiles)
            cards.append(f"""
            <article class='candidate-card'>
              <div class='candidate-card-header'><span class='candidate-rank'>#{index}</span><h3>{name}</h3></div>
              <div class='candidate-card-body'>
                <div class='candidate-structure'>{image_html}</div>
                <div class='candidate-details'>
                  <div class='candidate-metrics'>
                    <div><b>相似度</b><br>{html.escape(str(candidate.get('DreaMS similarity', 'N/A')))}</div>
                    <div><b>Δm/z</b><br>{html.escape(str(candidate.get('Delta m/z', 'N/A')))}</div>
                    <div><b>Formula</b><br>{formula}</div>
                    <div><b>Ref. ID</b><br>{html.escape(str(candidate.get('Ref. ID', 'N/A')))}</div>
                  </div>
                  <p><b>SMILES</b><br><code>{escaped_smiles}</code></p>
                  <div class='candidate-properties'>{info_html}</div>
                </div>
              </div>
            </article>
            """)
        return "<div class='candidate-list'>" + "".join(cards) + "</div>"

    def batch_analyze(self, file_path: Optional[str], model_type: str, tolerance_da: float, max_results: int):
        """Analyze every valid spectrum in an uploaded file."""
        if not file_path:
            return [], None, None, "请先上传文件。"
        try:
            official = model_type == "official_dreams"
            records = [validate_record(record, official=official) for record in SpectrumParser.parse_file(file_path)]
            self.ensure_model(model_type)
            embeddings = batch_generate_embeddings(
                self.model, [r["peaks"] for r in records], [r["precursor_mz"] for r in records],
                [r["charge"] for r in records], self.device,
                spectrum_ids=[r["spectrum_id"] for r in records],
                chem_aware=False,
            )
            rows, errors, complete = [], [], []
            for record, (embedding, metadata) in zip(records, embeddings):
                if embedding is None:
                    errors.append(record["spectrum_id"] + ": embedding 失败")
                    continue
                candidates, error = self._match_candidates(embedding, record["precursor_mz"], tolerance_da, int(max_results) * 2)
                candidate_records = candidates.to_dict("records") if candidates is not None and not candidates.empty else []
                candidate_records = deduplicate_candidates(candidate_records, int(max_results))
                rows.append({
                    "spectrum_id": record["spectrum_id"], "precursor_mz": record["precursor_mz"],
                    "charge": record["charge"], "embedding_dim": len(embedding),
                    "top_similarity": candidate_records[0].get("DreaMS similarity") if candidate_records else None,
                    "candidate_count": len(candidate_records),
                })
                complete.append({
                    "input": {
                        "spectrum_id": record["spectrum_id"],
                        "precursor_mz": record["precursor_mz"],
                        "charge": record["charge"],
                        "peaks": record["peaks"].tolist(),
                    },
                    "metadata": metadata,
                    "embedding": embedding.tolist(),
                    "candidates": candidate_records,
                    "candidate_error": error,
                })
            csv_path = self._write_csv(embeddings)
            backend = "official_dreams_embedding" if model_type == "official_dreams" else "reproducible_demo"
            json_path = self._write_json({"schema_version": 1, "model_type": model_type, "backend": backend, "records": complete})
            status = f"✅ 批量完成: {len(rows)} 成功，{len(errors)} 失败。"
            if errors:
                status += "\n\n" + "\n".join(f"- {error}" for error in errors)
            return rows, csv_path, json_path, status
        except ImportError as exc:
            return [], None, None, f"❌ 文件格式依赖缺失: {str(exc)}"
        except Exception as exc:
            return [], None, None, f"❌ 批量分析失败: {type(exc).__name__}"


def build_app():
    """Build the public Gradio application."""
    interface = DreaMSInterface()
    css = """
    .hero{padding:24px;border-radius:16px;background:linear-gradient(135deg,#4f46e5,#7c3aed);color:white}
    .note{padding:12px 16px;border-radius:12px;background:#f6f7ff}
    .candidate-list{display:flex;flex-direction:column;gap:16px;margin-top:12px}
    .candidate-card{border:1px solid #d9deea;border-radius:14px;padding:18px;background:#ffffff;color:#1f2937;box-shadow:0 2px 8px rgba(20,30,60,.08)}
    .candidate-card-header{display:flex;align-items:center;gap:10px;margin-bottom:14px;border-bottom:1px solid #e5e7eb;padding-bottom:10px;color:#1f2937}
    .candidate-card-header h3{margin:0;font-size:1.1rem;color:#1f2937}
    .candidate-rank{background:#4f46e5;color:white;border-radius:999px;padding:4px 10px;font-weight:700}
    .candidate-card-body{display:flex;gap:22px;align-items:flex-start}
    .candidate-structure{min-width:270px;min-height:270px;display:flex;align-items:center;justify-content:center;background:white;border-radius:10px;padding:6px}
    .candidate-structure img{max-width:260px;height:auto}
    .candidate-no-image{color:#4b5563;padding:40px 20px;text-align:center}
    .candidate-empty{color:#1f2937;padding:16px}
    .candidate-details{flex:1;min-width:0}
    .candidate-metrics{display:grid;grid-template-columns:repeat(4,minmax(90px,1fr));gap:10px;margin-bottom:14px}
    .candidate-metrics>div,.candidate-properties>div{padding:9px 10px;border-radius:8px;background:#eef2ff;color:#1f2937;font-size:.9rem;word-break:break-word}
    .candidate-metrics b,.candidate-properties b,.candidate-details p b{color:#111827}
    .candidate-properties{display:grid;grid-template-columns:repeat(3,minmax(90px,1fr));gap:8px}
    .candidate-details code{display:block;white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f4f6;color:#111827;padding:8px;border-radius:8px}
    .candidate-details p{color:#1f2937}
    .muted{color:#4b5563}
    @media(max-width:700px){.candidate-card-body{flex-direction:column}.candidate-structure{width:100%}.candidate-metrics,.candidate-properties{grid-template-columns:repeat(2,minmax(90px,1fr))}}
    """

    with gr.Blocks(title=APP_CONFIG.get("title", "DreaMS ChemAware"), theme=gr.themes.Soft(), css=css) as app:
        gr.Markdown("<div class='hero'><h1>🧪 DreaMS-based ChemAware 展示面板</h1><p>在 DreaMS 基础上扩展的质谱表示与化学分析体验。</p></div>")
        gr.Markdown("<div class='note'><b>模型状态：</b>默认使用官方 DreaMS embedding checkpoint（embedding_model.ckpt + ssl_model.ckpt）。仅在用户主动选择时使用 demo 模式。候选分子需要 MassSpecGym 数据库，结果是相似度排序而非鉴定结论。</div>")

        with gr.Tab("单谱体验"):
            with gr.Row():
                upload = gr.File(label="上传质谱文件（支持 MGF / mzML / mzXML / HDF5 / HD5 / JSON）", type="filepath", file_types=SUPPORTED_FORMATS)
                spectrum_choice = gr.Dropdown(label="文件中的谱图", choices=[])
            uploaded_status = gr.Markdown("支持单谱或多谱图文件。")
            spectra_state = gr.State([])
            upload.change(interface.parse_uploaded, inputs=[upload], outputs=[spectrum_choice, spectra_state, uploaded_status])

            with gr.Row():
                spectrum_input = gr.Textbox(label="MS/MS 峰表（JSON 或逐行 m/z intensity）", lines=7, value=DEFAULT_SPECTRUM)
                with gr.Column():
                    precursor_mz = gr.Number(label="前体 m/z", value=500.0)
                    charge = gr.Number(label="电荷", value=1, precision=0)
                    model_type = gr.Dropdown(label="表示模式", choices=[("官方 DreaMS（真实权重）", "official_dreams"), ("演示模式（非模型）", "demo")], value="official_dreams")
                    analyze_rules = gr.Checkbox(label="启用化学规则提示", value=True)
                    tolerance_da = gr.Number(label="候选 m/z 容差（Da）", value=0.05, minimum=0.0001)
                    max_results = gr.Slider(label="候选数量", minimum=1, maximum=50, step=1, value=10)
            select_btn = gr.Button("载入所选谱图")
            with gr.Row():
                run_btn = gr.Button("运行单谱分析", variant="primary")
                load_btn = gr.Button("准备演示后端")
                model_status = gr.Markdown("尚未准备后端")
            load_btn.click(interface.load_model_wrapper, inputs=[model_type], outputs=[model_status])
            spectrum_plot = gr.Plot(label="谱图预览")
            spectrum_input.change(safe_spectrum_plot, inputs=[spectrum_input, precursor_mz], outputs=[spectrum_plot])
            precursor_mz.change(safe_spectrum_plot, inputs=[spectrum_input, precursor_mz], outputs=[spectrum_plot])
            with gr.Row():
                stats_out = gr.Markdown()
                embedding_out = gr.Markdown()
            rules_out = gr.Markdown()
            with gr.Row():
                peak_table = gr.Dataframe(headers=["m/z", "intensity"], label="峰表", interactive=False)
                candidates_out = gr.Dataframe(headers=candidate_columns(), label="候选分子（相似度排序，不是鉴定结论）", interactive=False)
            select_btn.click(
                interface.select_uploaded,
                inputs=[spectrum_choice, spectra_state],
                outputs=[spectrum_input, precursor_mz, charge, spectrum_plot, peak_table],
            )
            candidate_state = gr.State([])
            with gr.Row():
                structure_top_n = gr.Slider(
                    label="同时展示前 N 个候选结构",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=3,
                )
                show_structures_btn = gr.Button("展示候选结构", variant="secondary")
            molecule_cards = gr.HTML(
                label="候选结构与完整信息（从上到下按相似度排序）",
                value="<div class='candidate-empty'>运行单谱分析后，选择展示数量并点击按钮。</div>",
            )
            candidate_status = gr.Markdown()
            with gr.Row():
                json_download = gr.File(label="下载完整 JSON", interactive=False)
                csv_download = gr.File(label="下载 embedding CSV", interactive=False)
            run_btn.click(interface.process_spectrum, inputs=[spectrum_input, precursor_mz, charge, analyze_rules, model_type, tolerance_da, max_results], outputs=[stats_out, embedding_out, rules_out, spectrum_plot, peak_table, candidates_out, candidate_status, json_download, csv_download, candidate_state])
            show_structures_btn.click(
                interface.render_molecules,
                inputs=[structure_top_n, candidate_state],
                outputs=[molecule_cards],
            )

        with gr.Tab("批量处理"):
            gr.Markdown("上传包含多个谱图的文件，逐谱生成 embedding 并导出汇总结果。")
            batch_file = gr.File(label="批量质谱文件", type="filepath", file_types=SUPPORTED_FORMATS)
            with gr.Row():
                batch_model = gr.Dropdown(label="表示模式", choices=[("官方 DreaMS（真实权重）", "official_dreams"), ("演示模式（非模型）", "demo")], value="official_dreams")
                batch_tolerance = gr.Number(label="候选 m/z 容差（Da）", value=0.05, minimum=0.0001)
                batch_k = gr.Slider(label="候选数量", minimum=1, maximum=50, step=1, value=10)
            batch_btn = gr.Button("运行批量分析", variant="primary")
            batch_table = gr.Dataframe(label="批量汇总", interactive=False)
            batch_csv = gr.File(label="下载批量 embedding CSV", interactive=False)
            batch_json = gr.File(label="下载批量 JSON", interactive=False)
            batch_status = gr.Markdown()
            batch_btn.click(interface.batch_analyze, inputs=[batch_file, batch_model, batch_tolerance, batch_k], outputs=[batch_table, batch_csv, batch_json, batch_status])

        with gr.Tab("能力与限制"):
            gr.Markdown("""## 当前面板能力

- 文件上传：`.mgf`、`.mzML`、`.mzXML`、`.hdf5`、`.h5`、`.hd5`、`.json`
- 多谱图选择与批量 embedding 导出
- 谱图 stick plot、峰表和完整 JSON/CSV 下载
- 候选分子检索、SMILES 结构图和 RDKit 分子性质（需要本地数据库/RDKit）
- 化学规则提示和演示表示对比

## 明确限制

默认入口使用官方 DreaMS embedding checkpoint；只有用户主动选择 demo 时才使用可复现替代向量。官方模式的候选结果才是基于官方 embedding 的相似度排序，仍不是结构鉴定或置信度。正式 annotation 的 FDR、校准、Schymanski、dark matter、Atlas、氟预测、attention 和 pathway 功能需要对应权重、数据库或外部映射，当前不会以伪结果呈现。
""")

    return app


if __name__ == "__main__":
    demo = build_app()
    port = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860")))
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    share = os.getenv("GRADIO_SHARE", "false").strip().lower() in {"1", "true", "yes", "on"}
    frontend_check = os.getenv("GRADIO_FRONTEND_CHECK", "false").strip().lower() in {"1", "true", "yes", "on"}
    demo.launch(
        server_name=server_name,
        server_port=port,
        share=share,
        _frontend=frontend_check,
    )
