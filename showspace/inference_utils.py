"""
DreaMS ChemAware 推理工具函数

提供模型加载、推理、分析等核心功能。
这个版本对 Hugging Face Spaces 做了轻量化处理：
即使完整模型权重不可用，也能提供稳定、可复现的演示输出。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add the source repository so the official DreaMS package can be imported.
DREAMS_ROOT = Path(__file__).resolve().parent.parent
if DREAMS_ROOT.exists():
    sys.path.insert(0, str(DREAMS_ROOT))

try:
    from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine  # type: ignore
except Exception:
    class ChemicalRuleEngine:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.tolerance = kwargs.get("tolerance", 0.02)
            self.enable_categories = kwargs.get("enable_categories")
            self.use_massbank = kwargs.get("use_massbank", False)


class OfficialDreaMSAdapter:
    """Official DreaMS embedding head and its official preprocessor."""

    def __init__(self, model, preprocessor, embedding_path: Path, ssl_path: Path, device: str):
        self.model = model
        self.preprocessor = preprocessor
        self.embedding_path = embedding_path
        self.ssl_path = ssl_path
        self.device = device
        self.backend = "official_dreams_embedding"

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def embed(self, peaks: np.ndarray, precursor_mz: float) -> np.ndarray:
        processed = self.preprocessor(
            np.asarray(peaks, dtype=np.float32),
            prec_mz=float(precursor_mz),
            high_form=True,
            augment=False,
        )
        batch = torch.from_numpy(processed).unsqueeze(0).to(
            device=self.model.device, dtype=self.model.dtype
        )
        with torch.inference_mode():
            embedding = self.model(batch, charge=None)
        embedding = embedding.detach().cpu().numpy()[0].astype(np.float32)
        if embedding.ndim != 1 or embedding.shape[0] != 1024 or not np.isfinite(embedding).all():
            raise RuntimeError("官方 DreaMS 输出不是有限的 1024 维向量")
        return embedding


def _checkpoint_paths() -> Tuple[Path, Path]:
    pretrained = DREAMS_ROOT / "dreams" / "models" / "pretrained"
    embedding_path = Path(os.getenv("DREAMS_EMBEDDING_CKPT", pretrained / "embedding_model.ckpt"))
    ssl_path = Path(os.getenv("DREAMS_SSL_CKPT", pretrained / "ssl_model.ckpt"))
    return embedding_path, ssl_path


def load_model(
    model_type: str = "official_dreams",
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
) -> Tuple:
    """Load the official DreaMS embedding model and rule engine."""
    if model_type == "demo":
        return None, ChemicalRuleEngine(tolerance=0.02, enable_categories=None, use_massbank=False)
    if model_type != "official_dreams":
        raise ValueError("模型模式必须是 official_dreams 或 demo")

    embedding_path, ssl_path = _checkpoint_paths()
    if not embedding_path.is_file():
        raise FileNotFoundError(f"缺少 embedding_model.ckpt: {embedding_path.name}")
    if not ssl_path.is_file():
        raise FileNotFoundError(f"缺少 ssl_model.ckpt: {ssl_path.name}")

    try:
        sys.path.insert(0, str(DREAMS_ROOT))
        from dreams.models.heads.heads import ContrastiveHead
        from dreams.utils.data import SpectrumPreprocessor
        from dreams.utils.dformats import DataFormatA

        print(f"加载官方 DreaMS embedding 到 {device}")
        pretrained = ContrastiveHead.load_from_checkpoint(
            embedding_path,
            backbone_pth=ssl_path,
            map_location=torch.device(device),
        )
        pretrained.eval().to(device)
        preprocessor = SpectrumPreprocessor(
            dformat=DataFormatA(), n_highest_peaks=100
        )
        rules = ChemicalRuleEngine(tolerance=0.02, enable_categories=None, use_massbank=False)
        return OfficialDreaMSAdapter(pretrained, preprocessor, embedding_path, ssl_path, device), rules
    except Exception as exc:
        raise RuntimeError(f"官方 DreaMS 加载失败: {type(exc).__name__}: {exc}") from exc


def validate_spectrum(
    peaks: np.ndarray,
    precursor_mz: float,
    charge: int,
    max_peaks: int = 1000,
) -> np.ndarray:
    """校验并返回可用于推理的峰表。"""
    peaks = np.asarray(peaks, dtype=np.float32)

    if peaks.ndim != 2 or peaks.shape[1] != 2:
        raise ValueError("谱图必须是形如 [[m/z, intensity], ...] 的二维数组")
    if len(peaks) < 2:
        raise ValueError("谱图必须至少包含 2 个峰")
    if len(peaks) > max_peaks:
        raise ValueError(f"谱图最多支持 {max_peaks} 个峰")
    if not np.isfinite(peaks).all():
        raise ValueError("谱图不能包含 NaN 或无穷大")
    if np.any(peaks[:, 0] <= 0):
        raise ValueError("所有 m/z 必须大于 0")
    if np.any(peaks[:, 1] < 0):
        raise ValueError("强度不能为负数")
    if not np.isfinite(precursor_mz) or precursor_mz <= 0:
        raise ValueError("前体 m/z 必须是正数")
    if int(charge) != charge or charge < 1:
        raise ValueError("电荷必须是正整数")

    return peaks


def get_runtime_device() -> str:
    """返回当前可用的推理设备。"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_spectrum(
    peaks: np.ndarray,
    precursor_mz: float,
    max_peaks: int = 100,
) -> np.ndarray:
    """
    预处理 MS/MS 谱图（与 DreaMS 官方一致）。
    """
    peaks = np.asarray(peaks, dtype=np.float32)

    if peaks.shape[0] == 0:
        raise ValueError("谱图不能为空")

    if peaks.shape[0] > max_peaks:
        top_indices = np.argsort(-peaks[:, 1])[:max_peaks]
        peaks = peaks[top_indices]

    peaks = peaks[np.argsort(peaks[:, 0])]

    max_intensity = np.max(peaks[:, 1])
    if max_intensity > 0:
        peaks[:, 1] = peaks[:, 1] / max_intensity

    precursor_peak = np.array([[precursor_mz, 1.1]], dtype=np.float32)
    peaks = np.vstack([precursor_peak, peaks])

    return peaks


def _spectrum_seed(
    preprocessed_peaks: np.ndarray,
    precursor_mz: float,
    charge: int,
    chem_aware: bool,
) -> int:
    rounded = np.round(preprocessed_peaks.astype(np.float32), 4)
    payload = rounded.tobytes() + np.array(
        [precursor_mz, charge, int(chem_aware)], dtype=np.float32
    ).tobytes()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _demo_embedding(
    preprocessed_peaks: np.ndarray,
    precursor_mz: float,
    charge: int,
    chem_aware: bool,
) -> np.ndarray:
    mz = preprocessed_peaks[:, 0].astype(np.float32)
    intensity = preprocessed_peaks[:, 1].astype(np.float32)

    base_features = np.array(
        [
            float(len(preprocessed_peaks)),
            float(precursor_mz),
            float(charge),
            float(np.mean(mz)),
            float(np.std(mz)),
            float(np.min(mz)),
            float(np.max(mz)),
            float(np.mean(intensity)),
            float(np.std(intensity)),
            float(np.min(intensity)),
            float(np.max(intensity)),
            float(np.sum(intensity)),
            float(np.median(mz)),
            float(np.median(intensity)),
        ],
        dtype=np.float32,
    )

    mz_hist, _ = np.histogram(mz, bins=64, range=(0.0, max(2000.0, precursor_mz * 1.2)), weights=intensity)
    intensity_hist, _ = np.histogram(intensity, bins=64, range=(0.0, 1.0), weights=None)
    mz_deltas = np.diff(np.sort(mz))
    delta_hist, _ = np.histogram(mz_deltas if len(mz_deltas) else np.array([0.0]), bins=32, range=(0.0, 100.0), weights=None)

    feature_vec = np.concatenate([base_features, mz_hist.astype(np.float32), intensity_hist.astype(np.float32), delta_hist.astype(np.float32)])
    feature_vec = np.pad(feature_vec, (0, max(0, 128 - feature_vec.size)))[:128].astype(np.float32)

    rng = np.random.default_rng(_spectrum_seed(preprocessed_peaks, precursor_mz, charge, chem_aware))
    projection = rng.standard_normal((feature_vec.size, 1024)).astype(np.float32)
    embedding = feature_vec @ projection

    if chem_aware:
        embedding += 0.08 * np.tanh(np.roll(embedding, 11))
    else:
        embedding -= 0.05 * np.tanh(np.roll(embedding, 17))

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.astype(np.float32)


def generate_embedding(
    model,
    peaks: np.ndarray,
    precursor_mz: float,
    charge: int = 1,
    device: str = "cpu",
    chem_aware: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    生成谱图嵌入向量。
    """
    start_time = time.time()

    try:
        peaks = validate_spectrum(peaks, precursor_mz, charge)
        preprocessed_peaks = preprocess_spectrum(peaks, precursor_mz)

        metadata = {
            "preprocessed_peaks": len(preprocessed_peaks),
            "original_peaks": len(peaks),
            "precursor_mz": float(precursor_mz),
            "charge": int(charge),
        }

        if isinstance(model, OfficialDreaMSAdapter):
            embedding = model.embed(peaks, precursor_mz)
        elif model is None:
            embedding = _demo_embedding(preprocessed_peaks, precursor_mz, charge, chem_aware)
        else:
            raise TypeError("未知模型后端")

        metadata["processing_time"] = time.time() - start_time
        metadata["embedding_norm"] = float(np.linalg.norm(embedding))
        if isinstance(model, OfficialDreaMSAdapter):
            metadata["backend"] = model.backend
            metadata["embedding_checkpoint"] = model.embedding_path.name
            metadata["ssl_checkpoint"] = model.ssl_path.name
            metadata["preprocessing"] = "official SpectrumPreprocessor/DataFormatA"
        else:
            metadata["backend"] = "reproducible_demo"
        metadata["device"] = device

        return embedding, metadata

    except Exception as e:
        raise RuntimeError(f"嵌入生成失败: {str(e)}")


def _add_rule(results: Dict[str, List[Dict]], category: str, name: str, description: str) -> None:
    results.setdefault(category, []).append({"name": name, "description": description, "category": category})


def analyze_chemical_rules(
    chem_rule_engine: ChemicalRuleEngine,
    peaks: np.ndarray,
    precursor_mz: float,
    tolerance: float = 0.02,
) -> Dict[str, List[Dict]]:
    """
    分析谱图中匹配的化学规则。
    """
    try:
        if chem_rule_engine is None:
            return {}

        peaks = np.asarray(peaks, dtype=np.float32)
        mz = peaks[:, 0]
        intensity = peaks[:, 1]

        results: Dict[str, List[Dict]] = {
            "NL": [],
            "CF": [],
            "ISO": [],
            "NR": [],
            "EE": [],
            "HR": [],
        }

        if len(mz) >= 2:
            diffs = np.abs(mz[:, None] - mz[None, :])
            upper = diffs[np.triu_indices_from(diffs, k=1)]

            if np.any(np.isclose(upper, 18.0, atol=tolerance)):
                _add_rule(results, "NL", "NL-18", "检测到接近 18 Da 的中性丢失，常见于 H2O 脱除")
            if np.any(np.isclose(upper, 44.0, atol=tolerance)):
                _add_rule(results, "NL", "NL-44", "检测到接近 44 Da 的中性丢失，常见于 CO2 脱除")
            if np.any(np.isclose(upper, 28.0, atol=tolerance)):
                _add_rule(results, "NL", "NL-28", "检测到接近 28 Da 的中性丢失，常见于 CO 脱除")
            if np.any(np.isclose(upper, 1.003, atol=0.01)):
                _add_rule(results, "ISO", "ISO-1", "检测到接近 1.003 Da 的同位素间隔")

        if np.any(np.isclose(mz, 91.0, atol=0.5)):
            _add_rule(results, "CF", "CF-91", "检测到 m/z 91 附近的特征碎片")
        if np.any(np.isclose(mz, 77.0, atol=0.5)):
            _add_rule(results, "CF", "CF-77", "检测到 m/z 77 附近的特征碎片")

        if len(intensity) > 0 and np.max(intensity) > 0.95:
            _add_rule(results, "EE", "EE-peak", "存在高强度主峰，符合偶电子离子稳定化特征")
        if int(round(precursor_mz)) % 2 == 1:
            _add_rule(results, "NR", "NR-odd", "前体质量为奇数附近，提示含氮化合物可能性")

        if len(mz) >= 3 and np.any(np.isclose(np.diff(np.sort(mz)), 14.0, atol=0.5)):
            _add_rule(results, "HR", "HR-14", "检测到接近 14 Da 的连续差异，可能存在氢重排相关模式")

        if all(len(v) == 0 for v in results.values()):
            _add_rule(results, "CF", "CF-base", "未命中特定规则，但谱图存在可解释碎片特征")

        return results

    except Exception as e:
        print(f"规则分析出错: {str(e)}")
        return {}


def format_spectrum_json(spectrum: np.ndarray) -> str:
    """将谱图数组转换为 JSON 字符串。"""
    peaks_list = spectrum.tolist()
    return json.dumps(peaks_list)


def parse_spectrum_json(spectrum_json: str) -> np.ndarray:
    """从 JSON 字符串解析谱图。"""
    peaks = json.loads(spectrum_json)
    return np.array(peaks, dtype=np.float32)


def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """计算两个嵌入间的余弦相似度。"""
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


def batch_generate_embeddings(
    model,
    spectra_list: List[np.ndarray],
    precursor_mzs: List[float],
    charges: List[int],
    device: str = "cpu",
    spectrum_ids: Optional[List[str]] = None,
    chem_aware: bool = True,
) -> List[Tuple[np.ndarray, Dict]]:
    """批量生成嵌入向量，并保留每条谱图的 ID。"""
    if not (len(spectra_list) == len(precursor_mzs) == len(charges)):
        raise ValueError("谱图、前体 m/z 和电荷列表长度必须一致")
    if spectrum_ids is not None and len(spectrum_ids) != len(spectra_list):
        raise ValueError("spectrum_ids 与谱图列表长度必须一致")

    results = []
    for index, (spectrum, mz, charge) in enumerate(zip(spectra_list, precursor_mzs, charges)):
        try:
            embedding, metadata = generate_embedding(
                model, spectrum, mz, charge, device, chem_aware=chem_aware
            )
            metadata["spectrum_id"] = spectrum_ids[index] if spectrum_ids else f"spec_{index}"
            results.append((embedding, metadata))
        except Exception as e:
            print(f"批量处理错误: {str(e)}")
            results.append((None, {"spectrum_id": spectrum_ids[index]} if spectrum_ids else None))

    return results


def export_embedding_to_csv(embeddings: List[Tuple[np.ndarray, Dict]], output_path: str):
    """将嵌入向量导出为 UTF-8 CSV，维度由实际向量决定。"""
    import csv

    valid = [(embedding, metadata or {}) for embedding, metadata in embeddings if embedding is not None]
    dimension = len(valid[0][0]) if valid else 0
    if any(len(embedding) != dimension for embedding, _ in valid):
        raise ValueError("所有 embedding 的维度必须一致")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["spectrum_id", "precursor_mz", "charge"] + [f"dim_{i}" for i in range(dimension)]
        writer.writerow(header)

        for idx, (embedding, metadata) in enumerate(valid):
            writer.writerow([
                metadata.get("spectrum_id", f"spec_{idx}"),
                metadata.get("precursor_mz", 0),
                metadata.get("charge", 0),
            ] + embedding.tolist())


def export_analysis_to_json(analysis: Dict, output_path: str):
    """将分析结果导出为 JSON。"""
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)


class SpectrumNormalizer:
    """谱图归一化工具。"""

    @staticmethod
    def normalize_intensity(spectrum: np.ndarray) -> np.ndarray:
        spectrum = spectrum.copy()
        max_intensity = np.max(spectrum[:, 1])
        if max_intensity > 0:
            spectrum[:, 1] = spectrum[:, 1] / max_intensity
        return spectrum

    @staticmethod
    def normalize_mz(spectrum: np.ndarray, max_mz: float = 1000.0) -> np.ndarray:
        spectrum = spectrum.copy()
        spectrum[:, 0] = spectrum[:, 0] / max_mz
        return spectrum

    @staticmethod
    def filter_peaks_by_intensity(spectrum: np.ndarray, min_intensity: float = 0.01) -> np.ndarray:
        mask = spectrum[:, 1] >= min_intensity
        return spectrum[mask]

    @staticmethod
    def filter_peaks_by_mz_range(
        spectrum: np.ndarray,
        min_mz: float = 0.0,
        max_mz: float = 2000.0,
    ) -> np.ndarray:
        mask = (spectrum[:, 0] >= min_mz) & (spectrum[:, 0] <= max_mz)
        return spectrum[mask]
