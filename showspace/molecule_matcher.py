"""
DreaMS ChemAware 数据库匹配模块

用于返回候选分子结构和相似度
使用真实的 MassSpecGym_DreaMS.hdf5 数据库
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class MoleculeDatabase:
    """分子数据库管理 - 使用真实的 MassSpecGym 数据库"""

    def __init__(self, hdf5_path: Optional[str] = None):
        """初始化分子数据库，可通过环境变量或参数指定 HDF5 路径。"""
        configured_path = hdf5_path or os.getenv("DREAMS_MOLECULE_DB")
        self.hdf5_path = Path(configured_path) if configured_path else self._find_massspecgym_database()
        self.h5file = None
        self.parent_mass = None
        self.embeddings = None
        self.error: Optional[str] = None

        if self.hdf5_path:
            self._load_hdf5_data()
        else:
            self.error = "未找到 MassSpecGym_DreaMS.hdf5"

    @staticmethod
    def _find_massspecgym_database() -> Path:
        """查找 MassSpecGym_DreaMS.hdf5 数据库文件"""
        base_path = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--roman-bushuiev--GeMS" / "snapshots"

        if base_path.exists():
            snapshots = list(base_path.glob("*/data/auxiliary/MassSpecGym_DreaMS.hdf5"))
            if snapshots:
                return snapshots[0]

        return None

    def _load_hdf5_data(self):
        """加载 HDF5 数据库中的关键数据"""
        try:
            import h5py
            self.h5file = h5py.File(str(self.hdf5_path), 'r')
            self.parent_mass = self.h5file['PARENT_MASS'][:]
            self.embeddings = self.h5file['DreaMS_embedding'][:]

            # 加载所有可用的元数据
            self.formula = self.h5file.get('FORMULA', None)
            self.inchikey = self.h5file.get('INCHIKEY', None)
            self.identifier = self.h5file.get('IDENTIFIER', None)
            self.smiles = self.h5file.get('smiles', None)
            self.name = self.h5file.get('name', None)
            self.instrument = self.h5file.get('INSTRUMENT_TYPE', None)

            print(f"Loaded MassSpecGym database: {len(self.parent_mass)} molecules")
            print(f"Available metadata: FORMULA, INCHIKEY, IDENTIFIER, SMILES, NAME")
        except Exception as e:
            print(f"Warning: Could not load HDF5 database: {e}")
            self.h5file = None
            self.parent_mass = None
            self.embeddings = None
            self.error = f"数据库加载失败: {type(e).__name__}"

    def search_by_precursor_mz(
        self,
        precursor_mz: float,
        tolerance_da: float = 0.05,
        max_results: int = 10
    ) -> List[Dict]:
        """按前体 m/z 搜索分子"""
        if self.parent_mass is None:
            return []

        results = []
        delta_mz = np.abs(self.parent_mass - precursor_mz)
        matching_indices = np.where(delta_mz <= tolerance_da)[0]

        selected_indices = matching_indices if max_results is None else matching_indices[:max_results]
        for idx in selected_indices:
            results.append({
                'index': int(idx),
                'Precursor_mz': float(self.parent_mass[idx]),
                'Delta_mz': float(delta_mz[idx]),
                'embedding': self.embeddings[idx],
            })

        return results

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个嵌入的相似度"""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(max(0.0, min(1.0, similarity)))

    def get_candidate_molecules(
        self,
        query_embedding: np.ndarray,
        precursor_mz: float,
        tolerance_da: float = 0.05,
        max_results: int = 10
    ) -> pd.DataFrame:
        """获取候选分子及其相似度排名"""
        if self.embeddings is None:
            return pd.DataFrame()

        candidates = self.search_by_precursor_mz(precursor_mz, tolerance_da, None)

        if not candidates:
            return pd.DataFrame()

        results = []

        for idx, mol in enumerate(candidates, 1):
            mol_embedding = mol.get('embedding')
            similarity = self.calculate_similarity(query_embedding, mol_embedding)
            mol_idx = mol['index']

            # 从 HDF5 中提取元数据
            formula = 'N/A'
            identifier = 'N/A'
            smiles = 'N/A'

            if self.formula is not None and mol_idx < len(self.formula):
                try:
                    formula = self.formula[mol_idx]
                    if isinstance(formula, bytes):
                        formula = formula.decode('utf-8')
                except:
                    pass

            if self.identifier is not None and mol_idx < len(self.identifier):
                try:
                    identifier = self.identifier[mol_idx]
                    if isinstance(identifier, bytes):
                        identifier = identifier.decode('utf-8')
                except:
                    pass

            if self.smiles is not None and mol_idx < len(self.smiles):
                try:
                    smiles = self.smiles[mol_idx]
                    if isinstance(smiles, bytes):
                        smiles = smiles.decode('utf-8')
                except:
                    pass

            results.append({
                'Rank': idx,
                'Precursor m/z': f"{mol['Precursor_mz']:.4f}",
                'Ref. precursor m/z': f"{mol['Precursor_mz']:.4f}",
                'Ref. molecule': 'MassSpecGym',
                'Ref. name': identifier if identifier != 'N/A' else f"Compound_{mol_idx}",
                'Ref. ID': f"MSGM_{mol_idx}",
                'SMILES': smiles,
                'Formula': formula,
                'MW': 'N/A',
                'DreaMS similarity': f"{similarity:.4f}",
                'Delta m/z': f"{mol.get('Delta_mz', 0):.6f}",
            })

        results.sort(key=lambda x: float(x['DreaMS similarity']), reverse=True)

        for idx, result in enumerate(results, 1):
            result['Rank'] = idx

        return pd.DataFrame(results[:max_results])
