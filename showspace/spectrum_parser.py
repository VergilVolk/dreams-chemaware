"""
质谱数据文件格式解析模块

支持格式：
- .mgf (Mascot Generic Format)
- .mzML (mzML)
- .mzXML (mzXML)
- .hdf5 (HDF5)
"""

import json
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path


class SpectrumParser:
    """质谱数据文件解析器"""

    @staticmethod
    def parse_file(file_path: str) -> List[Dict]:
        """
        解析质谱数据文件

        参数:
            file_path: 文件路径

        返回:
            谱图列表，每个谱图包含:
            {
                'spectrum_id': str,
                'precursor_mz': float,
                'charge': int,
                'peaks': np.ndarray (n, 2)
            }
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == '.mgf':
            return SpectrumParser.parse_mgf(file_path)
        elif suffix == '.mzml':
            return SpectrumParser.parse_mzml(file_path)
        elif suffix == '.mzxml':
            return SpectrumParser.parse_mzxml(file_path)
        elif suffix in {'.hdf5', '.h5', '.hd5'}:
            return SpectrumParser.parse_hdf5(file_path)
        elif suffix == '.json':
            return SpectrumParser.parse_json(file_path)
        else:
            supported = ', '.join(SpectrumParser.get_supported_formats())
            raise ValueError(f"不支持的文件格式: {suffix or '(无扩展名)'}；支持: {supported}")

    @staticmethod
    def parse_mgf(file_path: Path) -> List[Dict]:
        """
        解析 MGF (Mascot Generic Format) 文件

        格式示例:
        BEGIN IONS
        TITLE=Spectrum_1
        PEPMASS=500.1234
        CHARGE=1+
        100.5 0.8
        200.3 1.0
        END IONS
        """
        spectra = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 分割每个 IONS 块
            ions_blocks = content.split('BEGIN IONS')[1:]

            for block_idx, block in enumerate(ions_blocks):
                lines = block.strip().split('\n')

                spectrum = {
                    'spectrum_id': f'spectrum_{block_idx}',
                    'precursor_mz': 0.0,
                    'charge': 1,
                    'peaks': np.array([])
                }

                peaks = []

                for line in lines:
                    line = line.strip()

                    if line.startswith('TITLE='):
                        spectrum['spectrum_id'] = line.split('=', 1)[1]

                    # 支持 PEPMASS 和 PRECURSORMZ 两种格式
                    elif line.startswith('PEPMASS='):
                        try:
                            # PEPMASS 可能包含强度，格式: PEPMASS=500.1234 100.5
                            mz_str = line.split('=')[1].split()[0]
                            spectrum['precursor_mz'] = float(mz_str)
                        except:
                            pass

                    elif line.startswith('PRECURSORMZ='):
                        try:
                            spectrum['precursor_mz'] = float(line.split('=')[1])
                        except:
                            pass

                    elif line.startswith('CHARGE='):
                        try:
                            charge_str = line.split('=')[1]
                            spectrum['charge'] = int(charge_str.rstrip('+-'))
                        except:
                            pass

                    elif line and not line.startswith(('BEGIN', 'END', 'NAME=', 'DESCRIPTION=', 'EXACTMASS=', 'FORMULA=', 'INCHI=', 'SMILES=', 'FEATURE_ID=', 'MSLEVEL=', 'RETENTION_TIME=', 'ADDUCT=', 'SPECTYPE=', 'Collision', 'FRAGMENTATION=', 'ISOLATION=', 'Acquisition=', 'INSTRUMENT=', 'SOURCE=', 'IMS=', 'ION=', 'PI=', 'DATACOLLECTOR=', 'DATASET=', 'USI=', 'SCANS=', 'PRECURSOR=', 'QUALITY=')):
                        # 解析峰值
                        try:
                            parts = line.split()
                            if len(parts) >= 2:
                                mz = float(parts[0])
                                intensity = float(parts[1])
                                peaks.append([mz, intensity])
                        except:
                            pass

                if peaks:
                    spectrum['peaks'] = np.array(peaks, dtype=np.float32)
                    spectra.append(spectrum)

        except Exception as e:
            raise ValueError(f"解析 MGF 文件失败: {str(e)}")

        return spectra

    @staticmethod
    def parse_mzml(file_path: Path) -> List[Dict]:
        """
        解析 mzML 文件

        需要 pyteomics 库
        """
        try:
            from pyteomics import mzml
        except ImportError:
            raise ImportError("需要安装 pyteomics 库: pip install pyteomics")

        spectra = []

        try:
            with mzml.read(str(file_path)) as reader:
                for idx, spectrum in enumerate(reader):
                    if spectrum['ms level'] == 2:  # MS2 spectra
                        spec_dict = {
                            'spectrum_id': spectrum.get('ID', f'spectrum_{idx}'),
                            'precursor_mz': spectrum.get('precursor', {}).get('mz', 0.0),
                            'charge': spectrum.get('precursor', {}).get('charge', [1])[0],
                            'peaks': np.array([])
                        }

                        # 提取峰值
                        mz_array = spectrum.get('m/z array', [])
                        intensity_array = spectrum.get('intensity array', [])

                        if len(mz_array) > 0 and len(intensity_array) > 0:
                            peaks = np.column_stack([mz_array, intensity_array])
                            spec_dict['peaks'] = peaks.astype(np.float32)
                            spectra.append(spec_dict)

        except Exception as e:
            raise ValueError(f"解析 mzML 文件失败: {str(e)}")

        return spectra

    @staticmethod
    def parse_mzxml(file_path: Path) -> List[Dict]:
        """
        解析 mzXML 文件

        需要 pyteomics 库
        """
        try:
            from pyteomics import mzxml
        except ImportError:
            raise ImportError("需要安装 pyteomics 库: pip install pyteomics")

        spectra = []

        try:
            with mzxml.read(str(file_path)) as reader:
                for idx, spectrum in enumerate(reader):
                    if spectrum.get('msLevel') == 2:  # MS2 spectra
                        spec_dict = {
                            'spectrum_id': spectrum.get('ID', f'spectrum_{idx}'),
                            'precursor_mz': 0.0,
                            'charge': 1,
                            'peaks': np.array([])
                        }

                        # 提取前体信息
                        if 'precursorMz' in spectrum:
                            precursor_list = spectrum['precursorMz']
                            if isinstance(precursor_list, list) and len(precursor_list) > 0:
                                spec_dict['precursor_mz'] = precursor_list[0]['mz']
                                spec_dict['charge'] = int(precursor_list[0].get('precursorCharge', 1))

                        # 提取峰值
                        peaks_data = spectrum.get('peaks', [])
                        if peaks_data and len(peaks_data) > 0:
                            peaks = np.array(peaks_data, dtype=np.float32)
                            spec_dict['peaks'] = peaks
                            spectra.append(spec_dict)

        except Exception as e:
            raise ValueError(f"解析 mzXML 文件失败: {str(e)}")

        return spectra

    @staticmethod
    def parse_hdf5(file_path: Path) -> List[Dict]:
        """
        解析 HDF5 文件

        需要 h5py 库

        期望的 HDF5 结构:
        /spectra/
          - spectrum_0/
            - mz (dataset)
            - intensity (dataset)
            - precursor_mz (attribute)
            - charge (attribute)
          - spectrum_1/
            ...
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("需要安装 h5py 库: pip install h5py")

        spectra = []

        try:
            with h5py.File(file_path, 'r') as f:
                # 查找 spectra 组
                if 'spectra' in f:
                    spectra_group = f['spectra']
                elif 'spectrum' in f:
                    spectra_group = f['spectrum']
                else:
                    spectra_group = f

                # Support the columnar MSData layout used by the bundled examples.
                # It stores one row per spectrum and a padded `spectrum` array with
                # shape (n_spectra, 2, n_peaks).
                if 'spectrum' in f and hasattr(f['spectrum'], 'shape') and len(f['spectrum'].shape) == 3:
                    spectrum_array = np.asarray(f['spectrum'][:], dtype=np.float32)
                    n_spectra = spectrum_array.shape[0]

                    def read_value(name, index, default):
                        if name not in f:
                            return default
                        value = f[name][index]
                        if isinstance(value, bytes):
                            return value.decode('utf-8', errors='replace')
                        if hasattr(value, 'item'):
                            return value.item()
                        return value

                    for index in range(n_spectra):
                        mz_array = spectrum_array[index, 0]
                        intensity_array = spectrum_array[index, 1]
                        valid = np.isfinite(mz_array) & np.isfinite(intensity_array) & (mz_array > 0)
                        peaks = np.column_stack([mz_array[valid], intensity_array[valid]])
                        spectra.append({
                            'spectrum_id': str(read_value('name', index, f'spectrum_{index}')),
                            'precursor_mz': float(read_value('precursor_mz', index, 0.0)),
                            'charge': int(read_value('charge', index, 1)),
                            'peaks': peaks.astype(np.float32),
                        })
                else:
                    # Also support the nested /spectra/<id>/{mz,intensity} layout.
                    for spectrum_id in sorted(spectra_group.keys()):
                        spectrum_data = spectra_group[spectrum_id]
                        if not hasattr(spectrum_data, 'keys'):
                            continue

                        spec_dict = {
                            'spectrum_id': spectrum_id,
                            'precursor_mz': 0.0,
                            'charge': 1,
                            'peaks': np.array([])
                        }

                        if 'precursor_mz' in spectrum_data.attrs:
                            spec_dict['precursor_mz'] = float(spectrum_data.attrs['precursor_mz'])
                        if 'charge' in spectrum_data.attrs:
                            spec_dict['charge'] = int(spectrum_data.attrs['charge'])

                        if 'mz' in spectrum_data and 'intensity' in spectrum_data:
                            mz_array = np.array(spectrum_data['mz'], dtype=np.float32)
                            intensity_array = np.array(spectrum_data['intensity'], dtype=np.float32)
                            valid = np.isfinite(mz_array) & np.isfinite(intensity_array) & (mz_array > 0)
                            spec_dict['peaks'] = np.column_stack([mz_array[valid], intensity_array[valid]])
                            spectra.append(spec_dict)

        except Exception as e:
            raise ValueError(f"解析 HDF5 文件失败: {str(e)}")

        return spectra

    @staticmethod
    def parse_json(file_path: Path) -> List[Dict]:
        """
        解析 JSON 文件

        期望格式:
        [
          {
            "spectrum_id": "spectrum_1",
            "precursor_mz": 500.1234,
            "charge": 1,
            "peaks": [[100.5, 0.8], [200.3, 1.0], ...]
          },
          ...
        ]
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            spectra = []

            if isinstance(data, dict) and 'peaks' in data:
                data = [data]

            if isinstance(data, list):
                for idx, spectrum in enumerate(data):
                    if not isinstance(spectrum, dict):
                        raise ValueError(f"第 {idx + 1} 个谱图必须是对象")
                    spectra.append({
                        'spectrum_id': spectrum.get('spectrum_id', f'spectrum_{idx}'),
                        'precursor_mz': spectrum.get('precursor_mz', 0.0),
                        'charge': spectrum.get('charge', 1),
                        'peaks': np.array(spectrum.get('peaks', []), dtype=np.float32)
                    })
            elif isinstance(data, dict):
                for spectrum_id, spectrum in data.items():
                    if not isinstance(spectrum, dict):
                        raise ValueError(f"谱图 {spectrum_id} 必须是对象")
                    spectra.append({
                        'spectrum_id': spectrum_id,
                        'precursor_mz': spectrum.get('precursor_mz', 0.0),
                        'charge': spectrum.get('charge', 1),
                        'peaks': np.array(spectrum.get('peaks', []), dtype=np.float32)
                    })
            else:
                raise ValueError("JSON 顶层必须是谱图对象、对象列表或以谱图 ID 为键的对象")

            if not spectra:
                raise ValueError("文件中没有可用谱图")

        except Exception as e:
            if isinstance(e, ValueError) and str(e).startswith(('JSON 顶层', '文件中', '谱图')):
                raise
            raise ValueError(f"解析 JSON 文件失败: {str(e)}") from e

        return spectra

    @staticmethod
    def get_supported_formats() -> List[str]:
        """获取支持的文件格式列表"""
        return ['.mgf', '.mzml', '.mzxml', '.hdf5', '.h5', '.hd5', '.json']
