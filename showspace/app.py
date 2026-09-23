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
    available_model_types,
    batch_generate_embeddings,
    calculate_cosine_similarity,
    export_embedding_to_csv,
    generate_embedding,
    get_runtime_device,
    load_model,
    validate_spectrum,
)
from bioaware_service import BioAwareService
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
        "Embedding rank", "P2b preview rank", "Precursor m/z", "Delta ppm",
        "Ref. name", "Ref. ID", "Formula", "SMILES", "Embedding similarity",
        "P2b preview score", "BioAware support", "BioAware paths",
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

        similarity = float(candidate.get("Embedding similarity", candidate.get("DreaMS similarity", 0.0)))
        previous = best_by_structure.get(key)
        if previous is None or similarity > float(previous.get("Embedding similarity", previous.get("DreaMS similarity", 0.0))):
            best_by_structure[key] = candidate

    unique = sorted(
        best_by_structure.values(),
        key=lambda candidate: float(candidate.get("Embedding similarity", candidate.get("DreaMS similarity", 0.0))),
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
        self.databases: Dict[str, MoleculeDatabase] = {}
        self.database_errors: Dict[str, str] = {}
        self.bioaware = BioAwareService()

    def ensure_model(self, model_type: str = "official_dreams") -> None:
        if self.loaded_model_type != model_type:
            self.model, self.chem_rule_engine = load_model(model_type=model_type, device=self.device)
            self.loaded_model_type = model_type

    def load_model_wrapper(self, model_type: str = "official_dreams") -> str:
        try:
            self.ensure_model(model_type)
            if model_type in {"official_dreams", "e4a_shared", "e8_shared"}:
                fingerprint = getattr(self.model, "fingerprint", "legacy")
                return (
                    f"✅ {model_type} shared embedding 后端已就绪\n\n"
                    f"- 设备: `{self.device}`\n"
                    f"- checkpoint SHA256: `{fingerprint}`\n"
                    "- 查询与参考谱必须使用同一 fingerprint\n"
                    "- P2b / BioAware 均为 embedding 后的独立证据层"
                )
            return (
                "✅ 演示后端已就绪\n\n"
                f"- 设备: `{self.device}`\n"
                "- 模式: `demo`\n"
                "- 注意: 演示向量不代表真实 DreaMS embedding"
            )
        except Exception as exc:
            return f"❌ 后端初始化失败: {type(exc).__name__}: {str(exc)}"

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

    def _get_database(self, model_type: str) -> Tuple[Optional[MoleculeDatabase], Optional[str]]:
        fingerprint = getattr(self.model, "fingerprint", None)
        key = f"{model_type}:{fingerprint}"
        if key in self.databases or key in self.database_errors:
            return self.databases.get(key), self.database_errors.get(key)
        try:
            if MoleculeDatabase is None:
                self.database_errors[key] = "候选功能需要安装 h5py 和 pandas"
                return None, self.database_errors[key]
            database = MoleculeDatabase(model_type=model_type, model_fingerprint=fingerprint)
            if not database.ready:
                self.database_errors[key] = database.error or "候选数据库不可用"
            else:
                self.databases[key] = database
        except Exception as exc:
            self.database_errors[key] = f"候选数据库不可用（{type(exc).__name__}: {exc}）"
        return self.databases.get(key), self.database_errors.get(key)

    def _match_candidates(
        self, model_type: str, embedding: np.ndarray, peaks: np.ndarray,
        precursor_mz: float, adduct: str, ppm: float, max_results: int,
        p2b_preview: bool,
    ):
        database, error = self._get_database(model_type)
        if error or database is None:
            return None, {}, error or "候选数据库不可用"
        try:
            frame, report = database.retrieve(
                embedding, peaks, precursor_mz, adduct=adduct, ppm=ppm,
                max_results=max_results, p2b_preview=p2b_preview,
            )
            return frame, report, None
        except Exception as exc:
            return None, {}, f"候选检索失败: {type(exc).__name__}: {exc}"

    def process_spectrum(
        self, spectrum_text: str, precursor_mz: float, charge: int, adduct: str,
        analyze_rules: bool, model_type: str, ppm: float = 10.0,
        max_results: int = 10, p2b_preview: bool = True,
        bioaware_seed_file: Optional[str] = None,
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
            }, official=model_type in {"official_dreams", "e4a_shared", "e8_shared"})["peaks"]
            self.ensure_model(model_type)
            embedding, metadata = generate_embedding(
                self.model, record, float(precursor_mz), int(charge), self.device,
                chem_aware=False,
            )
            rules = analyze_chemical_rules(self.chem_rule_engine, record, float(precursor_mz)) if analyze_rules else {}
            candidates, retrieval_report, match_error = self._match_candidates(
                model_type, embedding, record, float(precursor_mz), str(adduct or ""),
                float(ppm), int(max_results) * 2, bool(p2b_preview),
            )
            candidates_data = candidates.to_dict("records") if candidates is not None and not candidates.empty else []
            candidates_data = deduplicate_candidates(candidates_data, int(max_results))
            bioaware_report = {"state": "abstained_no_context", "applied": False}
            if candidates_data:
                candidates_data, bioaware_report = self.bioaware.preview(candidates_data, bioaware_seed_file)
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
            if match_error:
                candidate_status = f"⚠️ {match_error}"
            else:
                reference_scope = str(retrieval_report.get("reference_scope", "unknown_reference"))
                if reference_scope.startswith("engineering_smoke"):
                    scope_note = (
                        "⚠️ **LOCAL SMOKE 参考库**：当前仅用于本机端到端工程联调，"
                        "不得用于宣称正式注释率、检索性能或 SOTA。\n\n"
                    )
                else:
                    scope_note = f"参考库范围：`{reference_scope}`。\n\n"
                candidate_status = (
                    scope_note
                    + "✅ Primary rank = model-aligned shared embedding. "
                    f"P2b = `{retrieval_report.get('p2b_state')}`（仅预览，near-core 封存结果为负）；"
                    f"BioAware = `{bioaware_report.get('state')}`（上下文证据层）。"
                )
            rules_md = format_rules_analysis(rules) + "\n\n" + candidate_status
            result = {
                "spectrum_id": "manual_input",
                "peaks": record.tolist(),
                "precursor_mz": float(precursor_mz), "charge": int(charge),
                "model_type": model_type, "metadata": metadata, "adduct": adduct,
                "embedding": embedding.tolist(), "chemical_rules": rules,
                "candidates": candidates_data, "retrieval": retrieval_report,
                "bioaware": bioaware_report,
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
                    <div><b>Embedding</b><br>{html.escape(str(candidate.get('Embedding similarity', 'N/A')))}</div>
                    <div><b>P2b preview rank</b><br>{html.escape(str(candidate.get('P2b preview rank', 'N/A')))}</div>
                    <div><b>BioAware</b><br>{html.escape(str(candidate.get('BioAware support', 'N/A')))}</div>
                    <div><b>Δppm</b><br>{html.escape(str(candidate.get('Delta ppm', 'N/A')))}</div>
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
            official = model_type in {"official_dreams", "e4a_shared", "e8_shared"}
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
                candidates, retrieval_report, error = self._match_candidates(
                    model_type, embedding, record["peaks"], record["precursor_mz"],
                    "", tolerance_da / max(record["precursor_mz"], 1e-12) * 1e6,
                    int(max_results) * 2, False,
                )
                candidate_records = candidates.to_dict("records") if candidates is not None and not candidates.empty else []
                candidate_records = deduplicate_candidates(candidate_records, int(max_results))
                rows.append({
                    "spectrum_id": record["spectrum_id"], "precursor_mz": record["precursor_mz"],
                    "charge": record["charge"], "embedding_dim": len(embedding),
                    "top_similarity": candidate_records[0].get("Embedding similarity") if candidate_records else None,
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
                    "candidate_error": error, "retrieval": retrieval_report,
                })
            csv_path = self._write_csv(embeddings)
            backend = model_type if model_type != "demo" else "reproducible_demo"
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
    registry = available_model_types()
    model_choices = [
        (f"{body['label']} {'● ready' if body['ready'] else '○ not configured'}", key)
        for key, body in registry.items()
    ]
    if os.getenv("ALLOW_DEMO", "false").lower() in {"1", "true", "yes"}:
        model_choices.append(("Demo vector (not a model)", "demo"))
    default_model = next((key for key in ("e8_shared", "e4a_shared", "official_dreams") if registry[key]["ready"]), "official_dreams")
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

    reference_label = html.escape(os.getenv("SHOWSPACE_REFERENCE_LABEL", "UNSPECIFIED_REFERENCE"))
    with gr.Blocks(title=APP_CONFIG.get("title", "DreaMS ChemAware"), theme=gr.themes.Soft(), css=css) as app:
        gr.Markdown("<div class='hero'><h1>DreaMS ChemAware Workbench</h1><p>共享谱图表示、局部谱学证据与生化网络证据的可审计组合平台。</p></div>")
        gr.Markdown(f"<div class='note'><b>当前参考库：</b><code>{reference_label}</code></div>")
        gr.Markdown("<div class='note'><b>严格模块边界：</b>E4/E8 改变 shared embedding；P2b 是候选组内的谱学重排预览；BioAware 是需要独立上下文种子的后验网络证据。三者不会被合并成一个虚假的“总模型分数”。</div>")

        with gr.Tab("单谱体验"):
            with gr.Row():
                upload = gr.File(label="上传质谱文件（支持 MGF / mzML / mzXML / HDF5 / HD5 / JSON）", type="filepath", file_types=SUPPORTED_FORMATS)
                spectrum_choice = gr.Dropdown(label="文件中的谱图", choices=[])
            uploaded_status = gr.Markdown("支持单谱或多谱图文件。")
            spectra_state = gr.State([])
            upload.change(interface.parse_uploaded, inputs=[upload], outputs=[spectrum_choice, spectra_state, uploaded_status], queue=False)

            with gr.Row():
                spectrum_input = gr.Textbox(label="MS/MS 峰表（JSON 或逐行 m/z intensity）", lines=7, value=DEFAULT_SPECTRUM)
                with gr.Column():
                    precursor_mz = gr.Number(label="前体 m/z", value=500.0)
                    charge = gr.Number(label="电荷", value=1, precision=0)
                    model_type = gr.Dropdown(label="共享 embedding 模型", choices=model_choices, value=default_model)
                    adduct = gr.Dropdown(
                        label="前体 adduct（P2b exact protocol 必填）",
                        choices=["", "[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+K]+"],
                        value="",
                        allow_custom_value=True,
                    )
                    analyze_rules = gr.Checkbox(label="启用化学规则提示", value=True)
                    tolerance_ppm = gr.Number(label="候选容差（ppm）", value=10.0, minimum=1.0, maximum=50.0)
                    p2b_preview = gr.Checkbox(label="计算 P2b 重排预览（不覆盖 primary rank）", value=True)
                    bioaware_seed_file = gr.File(
                        label="BioAware context seeds CSV（可选）",
                        type="filepath", file_types=[".csv", ".gz"],
                    )
                    max_results = gr.Slider(label="候选数量", minimum=1, maximum=50, step=1, value=10)
            select_btn = gr.Button("载入所选谱图")
            with gr.Row():
                run_btn = gr.Button("运行单谱分析", variant="primary")
                load_btn = gr.Button("加载所选共享编码器")
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
                queue=False,
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
            run_btn.click(
                interface.process_spectrum,
                inputs=[spectrum_input, precursor_mz, charge, adduct, analyze_rules, model_type,
                        tolerance_ppm, max_results, p2b_preview, bioaware_seed_file],
                outputs=[stats_out, embedding_out, rules_out, spectrum_plot, peak_table,
                         candidates_out, candidate_status, json_download, csv_download, candidate_state],
                api_name="analyze_spectrum",
            )
            show_structures_btn.click(
                interface.render_molecules,
                inputs=[structure_top_n, candidate_state],
                outputs=[molecule_cards],
            )

        with gr.Tab("批量处理"):
            gr.Markdown("上传包含多个谱图的文件，逐谱生成 embedding 并导出汇总结果。")
            batch_file = gr.File(label="批量质谱文件", type="filepath", file_types=SUPPORTED_FORMATS)
            with gr.Row():
                batch_model = gr.Dropdown(label="共享 embedding 模型", choices=model_choices, value=default_model)
                batch_tolerance = gr.Number(label="候选 m/z 容差（Da）", value=0.05, minimum=0.0001)
                batch_k = gr.Slider(label="候选数量", minimum=1, maximum=50, step=1, value=10)
            batch_btn = gr.Button("运行批量分析", variant="primary")
            batch_table = gr.Dataframe(label="批量汇总", interactive=False)
            batch_csv = gr.File(label="下载批量 embedding CSV", interactive=False)
            batch_json = gr.File(label="下载批量 JSON", interactive=False)
            batch_status = gr.Markdown()
            batch_btn.click(interface.batch_analyze, inputs=[batch_file, batch_model, batch_tolerance, batch_k], outputs=[batch_table, batch_csv, batch_json, batch_status])

        with gr.Tab("证据与边界"):
            gr.Markdown("""## 当前组合系统

- 文件上传：`.mgf`、`.mzML`、`.mzXML`、`.hdf5`、`.h5`、`.hd5`、`.json`
- Official / E4-A / E8 clean shared embedding（只有已配置 checkpoint 才能启用）
- 与 checkpoint SHA256 完全一致的 reference embedding index
- strict mass-window candidate generation 与 molecule-level aggregation
- 冻结 P2b 谱学融合的 side-by-side preview
- phenotype-blind BioAware one-hop Rhea evidence；无独立 seeds 时自动弃权

## 已验证结果不能混写

- **E4-A**：5 formula folds × 3 seeds 的开发 OOF 结果支持 shared embedding 有小幅稳定改善；它不是 3–4 pp 的 P2b 结果。
- **P2b**：开发 nested OOF 约 +3.91 pp；sealed P3 main +1.07 pp，但 near-core −4.23 pp。因此网页默认不让 P2b 覆盖 primary rank。
- **BioAware**：目前是可审计的上下文证据/弃权模块，尚无可靠外部准确率提升，不可宣传为结构鉴定器。

## 明确限制与上线合同

候选排名不是 MSI Level 1/2 身份结论。E4/E8 查询端不能与官方 reference index 混用；索引 fingerprint 不匹配时服务会 fail-closed。BioAware 不接收 phenotype、q-value、fold-change 等字段，避免用疾病标签反证身份。P2b 只作预览，直到推理时可用的 near-safety gate 在新封存集通过。
""")

    return app


if __name__ == "__main__":
    demo = build_app()
    port = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860")))
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    share = os.getenv("GRADIO_SHARE", "false").strip().lower() in {"1", "true", "yes", "on"}
    frontend_check = os.getenv("GRADIO_FRONTEND_CHECK", "false").strip().lower() in {"1", "true", "yes", "on"}
    username = os.getenv("SHOWSPACE_USERNAME", "").strip()
    password = os.getenv("SHOWSPACE_PASSWORD", "")
    auth = (username, password) if username and password else None
    if bool(username) != bool(password):
        raise RuntimeError("SHOWSPACE_USERNAME and SHOWSPACE_PASSWORD must be configured together")
    if os.getenv("SHOWSPACE_PUBLIC_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}:
        allow_anonymous = os.getenv("SHOWSPACE_ALLOW_ANONYMOUS", "false").strip().lower() in {"1", "true", "yes", "on"}
        if auth is None and not allow_anonymous:
            raise RuntimeError("public mode requires Showspace credentials or an explicit SHOWSPACE_ALLOW_ANONYMOUS=true")
    blocked_paths = []
    for variable in (
        "DREAMS_OFFICIAL_SLIM_CKPT", "DREAMS_ARCHITECTURE_CKPT", "DREAMS_E4_CKPT", "DREAMS_E8_CKPT",
        "DREAMS_MOLECULE_DB", "DREAMS_OFFICIAL_EMBEDDING_INDEX", "DREAMS_E4_EMBEDDING_INDEX", "DREAMS_E8_EMBEDDING_INDEX",
    ):
        value = os.getenv(variable, "").strip()
        if value and Path(value).exists():
            blocked_paths.append(str(Path(value).resolve()))
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=server_name,
        server_port=port,
        share=share,
        auth=auth,
        show_api=False,
        max_threads=2,
        state_session_capacity=100,
        max_file_size=os.getenv("SHOWSPACE_MAX_FILE_SIZE", "200mb"),
        blocked_paths=blocked_paths,
        _frontend=frontend_check,
    )
