"""
DreaMS ChemAware 应用配置
"""

# ============================================================================
# 应用配置
# ============================================================================

APP_CONFIG = {
    "title": "DreaMS ChemAware - MS/MS Mass Spectra Encoder",
    "description": "Interactive platform for chemical-aware mass spectra embeddings",
    "version": "1.0.0",
    "author": "DreaMS Contributors",
    "theme": "soft",
    "share": True,
    "server_name": "0.0.0.0",
    "server_port": 7860,
}

# ============================================================================
# 模型配置
# ============================================================================

MODEL_CONFIG = {
    "default_device": "cuda",  # 优先 GPU
    "fallback_device": "cpu",  # GPU 不可用时使用 CPU
    "model_type": "chem_aware",  # "chem_aware" 或 "baseline"
    "embedding_dim": 1024,
    "checkpoint_dir": "dreams/models/pretrained/",
}

# ============================================================================
# 谱图预处理配置
# ============================================================================

SPECTRUM_CONFIG = {
    "max_peaks": 100,  # 最大峰值数
    "min_intensity": 0.0,  # 最小相对强度
    "normalize_intensity": True,  # 按最大峰强度归一化
    "precursor_peak_intensity": 1.1,  # 前体峰强度
    "mz_range": (0.0, 2000.0),  # m/z 范围
}

# ============================================================================
# 化学规则配置
# ============================================================================

CHEMICAL_RULES_CONFIG = {
    "tolerance_da": 0.02,  # 质量容差 Da
    "enabled_categories": ["NL", "CF", "ISO", "NR", "EE", "HR"],  # 启用的规则类别
    "use_massbank": False,  # 是否使用 MassBank 规则
    "rule_database_path": "dreams/models/chem_aware/chem_rules_data.json",
}

# ============================================================================
# 推理配置
# ============================================================================

INFERENCE_CONFIG = {
    "batch_size": 32,
    "enable_chem_awareness": True,
    "enable_attention_visualization": False,  # 待实现
    "precision": "float32",
}

# ============================================================================
# Gradio UI 配置
# ============================================================================

UI_CONFIG = {
    "spectrum_examples": [
        {
            "peaks": "[[100.5, 0.8], [200.3, 1.0], [301.2, 0.5], [402.1, 0.6], [500.0, 0.9]]",
            "precursor_mz": 500.0,
            "charge": 1,
            "description": "Simple spectrum example"
        },
        {
            "peaks": "[[50.0, 0.5], [100.0, 0.7], [150.5, 1.0], [250.2, 0.8], [400.0, 0.6]]",
            "precursor_mz": 400.0,
            "charge": 1,
            "description": "Another spectrum example"
        },
    ],
    "theme": "soft",
    "layout": "vertical",
    "show_api": True,
}

# ============================================================================
# 输出配置
# ============================================================================

OUTPUT_CONFIG = {
    "save_embeddings": True,
    "save_analysis": True,
    "output_format": "json",  # "json" 或 "csv"
    "output_dir": "./outputs/",
    "max_examples_to_display": 10,
}

# ============================================================================
# 高级选项
# ============================================================================

ADVANCED_CONFIG = {
    "enable_caching": True,
    "cache_size_mb": 1024,
    "enable_logging": True,
    "log_level": "INFO",
    "enable_profiling": False,
    "timeout_seconds": 300,
}

# ============================================================================
# 化学规则类别说明
# ============================================================================

RULE_CATEGORIES_INFO = {
    "NL": {
        "name": "Neutral Loss (中性丢失)",
        "description": "Loss of neutral molecules (e.g., H2O, CO2) from precursor",
        "examples": ["NL-18 (H2O)", "NL-44 (CO2)", "NL-28 (CO)"]
    },
    "CF": {
        "name": "Characteristic Fragment (特征碎片)",
        "description": "Frequent characteristic fragment ions",
        "examples": ["m/z 91 (Tropylium)", "m/z 77 (Phenyl cation)"]
    },
    "ISO": {
        "name": "Isotope (同位素)",
        "description": "Isotope patterns (e.g., 13C, 2H)",
        "examples": ["M+1 (13C isotope)", "M+2 (doubly substituted)"]
    },
    "NR": {
        "name": "Nitrogen Rule (氮规则)",
        "description": "Mass/charge relationship for nitrogen-containing compounds",
        "examples": ["Odd m/z for odd N", "Even m/z for even N"]
    },
    "EE": {
        "name": "Even Electron (偶电子)",
        "description": "Even-electron ion rules",
        "examples": ["EE ions more stable", "OE ions rearrange"]
    },
    "HR": {
        "name": "Hydrogen Rearrangement (氢重排)",
        "description": "Hydrogen rearrangement during fragmentation",
        "examples": ["McLafferty rearrangement", "Retro-Diels-Alder"]
    },
}
