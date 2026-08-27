"""
DreaMS ChemAware HF-Spaces 简单测试脚本
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 添加当前面板目录到 path
sys.path.insert(0, str(Path(__file__).parent))

from inference_utils import (
    calculate_cosine_similarity,
    generate_embedding,
    parse_spectrum_json,
    preprocess_spectrum,
    SpectrumNormalizer,
    validate_spectrum,
)


def test_spectrum_parsing():
    """测试谱图 JSON 解析"""
    print("🧪 测试 1: 谱图解析")

    spectrum_json = '[[100.5, 0.8], [200.3, 1.0], [301.2, 0.5]]'
    spectrum = parse_spectrum_json(spectrum_json)

    assert spectrum.shape == (3, 2), f"Expected shape (3, 2), got {spectrum.shape}"
    assert spectrum[0, 0] == 100.5, "m/z parsing error"
    assert spectrum[1, 1] == 1.0, "intensity parsing error"

    print("  ✅ 谱图解析成功")


def test_spectrum_validation():
    """测试谱图和参数边界。"""
    import numpy as np

    valid = np.array([[100.0, 0.4], [200.0, 1.0]], dtype=np.float32)
    validated = validate_spectrum(valid, precursor_mz=500.0, charge=1)
    assert validated.shape == (2, 2)

    for invalid in [
        np.array([[0.0, 1.0], [200.0, 0.5]]),
        np.array([[100.0, -1.0], [200.0, 0.5]]),
        np.array([[100.0, np.nan], [200.0, 0.5]]),
    ]:
        try:
            validate_spectrum(invalid, precursor_mz=500.0, charge=1)
            raise AssertionError("invalid spectrum was accepted")
        except ValueError:
            pass

    try:
        validate_spectrum(valid, precursor_mz=500.0, charge=0)
        raise AssertionError("invalid charge was accepted")
    except ValueError:
        pass


def test_demo_embedding_is_deterministic():
    """测试演示嵌入对相同输入保持稳定。"""
    import numpy as np

    peaks = np.array([[100.0, 0.4], [200.0, 1.0]], dtype=np.float32)
    first, first_meta = generate_embedding(None, peaks, 500.0, 1)
    second, second_meta = generate_embedding(None, peaks, 500.0, 1)
    assert np.array_equal(first, second)
    assert first_meta["backend"] == "reproducible_demo"
    assert second_meta["device"] == "cpu"


def test_spectrum_preprocessing():
    """测试谱图预处理"""
    print("🧪 测试 2: 谱图预处理")

    peaks = parse_spectrum_json('[[100.5, 0.8], [200.3, 1.0], [301.2, 0.5]]')
    preprocessed = preprocess_spectrum(peaks, precursor_mz=500.0)

    # 应该有 4 个峰（3 个原始 + 1 个前体）
    assert preprocessed.shape[0] == 4, f"Expected 4 peaks, got {preprocessed.shape[0]}"

    # 第一个应该是前体峰
    assert preprocessed[0, 0] == 500.0, "Precursor peak not prepended correctly"

    # 强度应该归一化到 [0, 1]
    assert preprocessed[:, 1].max() <= 1.10001, "Intensity normalization error"

    print("  ✅ 谱图预处理成功")


def test_spectrum_normalization():
    """测试谱图归一化"""
    print("🧪 测试 3: 谱图归一化")

    peaks = parse_spectrum_json('[[100.0, 0.4], [200.0, 1.0], [300.0, 0.5]]')

    # 测试强度归一化
    normalized = SpectrumNormalizer.normalize_intensity(peaks)
    assert normalized[:, 1].max() <= 1.00001, "Intensity normalization failed"
    assert abs(normalized[1, 1] - 1.0) < 1e-6, "Max intensity should be 1.0"

    # 测试 m/z 范围过滤
    filtered = SpectrumNormalizer.filter_peaks_by_mz_range(
        peaks, min_mz=150.0, max_mz=250.0
    )
    assert filtered.shape[0] == 1, f"Expected 1 peak in range, got {filtered.shape[0]}"
    assert filtered[0, 0] == 200.0, "m/z filtering error"

    print("  ✅ 谱图归一化成功")


def test_cosine_similarity():
    """测试余弦相似度计算"""
    print("🧪 测试 4: 余弦相似度")

    import numpy as np

    # 创建两个测试向量
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])

    # 相同向量的相似度应该是 1.0
    sim_same = calculate_cosine_similarity(v1, v2)
    assert abs(sim_same - 1.0) < 0.001, f"Expected 1.0, got {sim_same}"

    # 正交向量的相似度应该是 0.0
    sim_orthogonal = calculate_cosine_similarity(v1, v3)
    assert abs(sim_orthogonal) < 0.001, f"Expected 0.0, got {sim_orthogonal}"

    print("  ✅ 余弦相似度计算成功")


def test_embedding_export():
    """测试嵌入向量导出"""
    print("🧪 测试 5: 嵌入向量导出")

    import numpy as np
    from inference_utils import export_embedding_to_csv
    import tempfile
    import os

    # 创建测试数据
    embeddings = [
        (np.random.randn(1024), {
            'precursor_mz': 500.0,
            'charge': 1
        }),
        (np.random.randn(1024), {
            'precursor_mz': 600.0,
            'charge': 2
        }),
    ]

    # 导出到临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_file = f.name

    try:
        export_embedding_to_csv(embeddings, temp_file)

        # 验证文件存在且包含数据
        assert os.path.exists(temp_file), "CSV file not created"

        with open(temp_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 3, f"Expected 3 lines (header + 2 data), got {len(lines)}"

            # 检查头部
            header = lines[0].strip().split(',')
            assert len(header) == 1027, f"Expected 1027 columns, got {len(header)}"  # 3 metadata + 1024 dims

        print("  ✅ 嵌入向量导出成功")

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_json_export():
    """测试 JSON 导出"""
    print("🧪 测试 6: JSON 导出")

    from inference_utils import export_analysis_to_json
    import tempfile
    import os
    import json

    # 创建测试数据
    analysis = {
        "spectrum_id": "test_001",
        "rules_matched": {
            "NL": ["NL-18", "NL-44"],
            "CF": ["CF-base"],
        },
        "precursor_mz": 500.0,
        "charge": 1,
    }

    # 导出到临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name

    try:
        export_analysis_to_json(analysis, temp_file)

        # 验证文件
        assert os.path.exists(temp_file), "JSON file not created"

        with open(temp_file, 'r') as f:
            loaded = json.load(f)
            assert loaded["spectrum_id"] == "test_001", "JSON data mismatch"
            assert "NL-18" in loaded["rules_matched"]["NL"], "Rule data not saved"

        print("  ✅ JSON 导出成功")

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🧬 DreaMS ChemAware HF-Spaces 测试套件")
    print("="*60 + "\n")

    tests = [
        test_spectrum_parsing,
        test_spectrum_validation,
        test_demo_embedding_is_deterministic,
        test_spectrum_preprocessing,
        test_spectrum_normalization,
        test_cosine_similarity,
        test_embedding_export,
        test_json_export,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  ❌ 测试失败: {str(e)}")
            failed += 1

    print("\n" + "="*60)
    print(f"📊 测试结果: {passed} 通过, {failed} 失败")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
