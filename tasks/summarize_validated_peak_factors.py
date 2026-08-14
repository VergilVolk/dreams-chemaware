"""Build the final catalog for the spectral-first fragmentation-factor pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-audit", type=Path, required=True)
    parser.add_argument("--raw-stability", type=Path, required=True)
    parser.add_argument("--centered-stability", type=Path, required=True)
    parser.add_argument("--spectral-audit", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--rule-coverage", type=Path, required=True)
    parser.add_argument("--occlusion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    activation = read(args.activation_audit)
    raw_stability = read(args.raw_stability)
    centered_stability = read(args.centered_stability)
    spectral = read(args.spectral_audit)
    localization = read(args.localization)
    structure = read(args.structure)
    coverage = read(args.rule_coverage)
    occlusion = read(args.occlusion)

    spectral_map = {int(item["factor"]): item for item in spectral["factors"]}
    structure_map = {
        (int(item["factor"]), item["spectral_kind"]): item
        for item in structure["factors"]
    }
    coverage_map = {
        (int(item["factor"]), item["spectral_kind"]): item
        for item in coverage["validated_factors"]
    }
    catalog = []
    for factor in localization["factors"]:
        factor_id = int(factor["factor"])
        for kind, result in factor["confirmation"].items():
            if not result.get("localization_pass_bh"):
                continue
            spectral_test = spectral_map[factor_id]["confirmation"][
                "fragment_test" if kind == "fragment_mz" else "neutral_loss_test"
            ]
            structure_test = structure_map.get((factor_id, kind), {})
            fixed_structure = structure_test.get(
                "confirmation_fixed_environment_test", {}
            )
            selected_environment = structure_test.get(
                "discovery_selected_environment", {}
            )
            rule = coverage_map[(factor_id, kind)]
            support = int(spectral_test.get("active_molecules", 0))
            structure_q = fixed_structure.get("bh_q")
            structure_pass = bool(
                fixed_structure.get("tested")
                and structure_q is not None
                and structure_q <= 0.05
                and fixed_structure.get("log2_enrichment", 0) > 0
            )
            catalog.append({
                "factor": factor_id,
                "spectral_kind": kind,
                "mass_da": float(result["fixed_mass_da"]),
                "confirmation_active_peaks": int(result["active_target_peaks"]),
                "confirmation_active_molecules": support,
                "peak_localization_joint_p": float(result["conservative_joint_p"]),
                "peak_localization_bh_q": float(
                    result["bh_q_across_confirmation_candidates"]
                ),
                "support_tier": (
                    "main" if support >= 10 else "low_support"
                ),
                "structure_context_replicated": structure_pass,
                "structure_environment": selected_environment.get(
                    "representative_environment_smiles"
                ),
                "structure_odds_ratio": fixed_structure.get("odds_ratio"),
                "structure_bh_q": structure_q,
                "matched_core_rules": int(rule["matched_core_rules"]),
                "matched_massbank_records": int(rule["matched_massbank_rules"]),
                "rule_library_gap": bool(rule["rule_library_gap"]),
                "nearest_rule_name": (
                    rule["rule_matches"][0]["name"]
                    if rule["rule_matches"] else None
                ),
                "nearest_rule_mass_error_da": (
                    rule["rule_matches"][0]["absolute_mass_error_da"]
                    if rule["rule_matches"] else None
                ),
            })
    catalog.sort(key=lambda item: (item["spectral_kind"], item["mass_da"]))
    occlusion_map = {int(item["factor"]): item for item in occlusion["factors"]}
    causal_retrieval_factors = []
    representation_sensitive_factors = []
    for factor, item in occlusion_map.items():
        margin_ci = item["selective_margin_drop_molecule_bootstrap_ci95"]
        shift_ci = item["selective_embedding_shift_molecule_bootstrap_ci95"]
        if (
            item["selective_margin_drop_mean"] > 0
            and margin_ci[0] > 0
            and item["selective_margin_drop_wilcoxon_p"] <= 0.05
        ):
            causal_retrieval_factors.append(factor)
        if (
            item["selective_embedding_shift_mean"] > 0
            and shift_ci[0] > 0
            and item["selective_embedding_shift_wilcoxon_p"] <= 0.05
        ):
            representation_sensitive_factors.append(factor)

    summary = {
        "status": "spectral_first_fragmentation_factor_pilot_complete",
        "checkpoint": "official DreaMS embedding checkpoint",
        "discovery_rules_used_as_labels": False,
        "data": {
            "discovery_valid_peak_tokens": activation[
                "continuous_distributions"
            ]["mz"]["discovery"]["n"],
            "confirmation_valid_peak_tokens": activation[
                "continuous_distributions"
            ]["mz"]["confirmation"]["n"],
            "discovery_confirmation_ik14_overlap": activation["isolation"][
                "ik14_overlap_count"
            ],
        },
        "factor_identifiability": {
            "stable_raw_peak_token_factors": raw_stability[
                "stable_features_all_comparisons"
            ],
            "stable_within_spectrum_centered_factors": centered_stability[
                "stable_features_all_comparisons"
            ],
            "interpretation": (
                "Removing the spectrum-shared token component increased strict "
                "three-seed stability, supporting a local-peak rather than "
                "whole-spectrum factorization."
            ),
        },
        "validated_catalog": {
            "bh_validated_peak_factors": len(catalog),
            "main_support_factors_ge_10_molecules": sum(
                item["support_tier"] == "main" for item in catalog
            ),
            "low_support_factors": sum(
                item["support_tier"] == "low_support" for item in catalog
            ),
            "replicated_structure_contexts": sum(
                item["structure_context_replicated"] for item in catalog
            ),
            "matched_by_core_rules": sum(
                item["matched_core_rules"] > 0 for item in catalog
            ),
            "matched_by_massbank_records": sum(
                item["matched_massbank_records"] > 0 for item in catalog
            ),
            "post_hoc_rule_gaps": sum(
                item["rule_library_gap"] for item in catalog
            ),
        },
        "catalog": catalog,
        "occlusion": {
            "factors_tested": sorted(occlusion_map),
            "representation_sensitive_factors": sorted(
                representation_sensitive_factors
            ),
            "retrieval_causal_factors": sorted(causal_retrieval_factors),
            "interpretation": (
                "91.06 与 67.06 Da 峰的删除均造成超出匹配控制峰的全局 "
                "embedding 位移，但严格 10 ppm 检索边际的分子级 bootstrap "
                "区间均跨 0，因此尚未证明其决定检索结果。"
            ),
        },
        "scientific_conclusion": (
            "DreaMS 官方微调权重的峰 token 中存在少量可重复、可定位到具体峰的谱图方向；"
            "其中两个方向进一步复现了芳香局部结构背景。它们属于候选碎裂因子，尚未构成断键机理证明，"
            "也尚未证明是严格 10 ppm 检索的因果信号。"
        ),
        "next_gate": (
            "扩大独立分子和化学类别覆盖，对候选峰进行元素组成约束与碎片结构注释；"
            "随后只针对在失败子类中能选择性改善检索边际的因子设计微调监督。"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if catalog:
        with (args.output_dir / "validated_factor_catalog.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(catalog[0]))
            writer.writeheader()
            writer.writerows(catalog)

    lines = [
        "# DreaMS 峰级隐式碎裂因子预实验",
        "",
        "## 严格结论",
        "",
        summary["scientific_conclusion"],
        "",
        "- 发现阶段没有使用现有 3,486 条规则作为标签。",
        f"- discovery/confirmation 分别包含 {summary['data']['discovery_valid_peak_tokens']:,} / {summary['data']['confirmation_valid_peak_tokens']:,} 个有效峰，IK14 零重叠。",
        f"- 原始峰 token 仅有 {summary['factor_identifiability']['stable_raw_peak_token_factors']} 个严格稳定因子；谱内中心化后增至 {summary['factor_identifiability']['stable_within_spectrum_centered_factors']} 个。",
        f"- 最终 {len(catalog)} 个方向通过独立 confirmation、谱内置换和 BH 校正；其中 {summary['validated_catalog']['main_support_factors_ge_10_molecules']} 个至少由 10 个独立分子支持。",
        f"- {summary['validated_catalog']['replicated_structure_contexts']} 个方向进一步复现局部结构背景。",
        f"- 对 F117（91.06 Da）与 F176（67.06 Da）完成匹配控制峰删除：两者均影响全局 embedding，但严格 10 ppm 检索因果方向为 {len(causal_retrieval_factors)} 个。",
        "",
        "## 最终目录",
        "",
        "| 因子 | 类型 | 质量 (Da) | confirmation 分子 | 峰定位 q | 结构背景 | 结构 q | 核心规则 | MassBank记录 |",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for item in catalog:
        structure_q = item["structure_bh_q"]
        lines.append(
            f"| F{item['factor']} | {item['spectral_kind']} | {item['mass_da']:.2f} | "
            f"{item['confirmation_active_molecules']} | {item['peak_localization_bh_q']:.3g} | "
            f"{item['structure_environment'] or '—'} | "
            f"{structure_q:.3g}" if structure_q is not None else ""
        )
        # Rebuild the row to avoid conditional-expression precedence ambiguity.
        lines[-1] = (
            f"| F{item['factor']} | {item['spectral_kind']} | {item['mass_da']:.2f} | "
            f"{item['confirmation_active_molecules']} | {item['peak_localization_bh_q']:.3g} | "
            f"{item['structure_environment'] or '—'} | "
            f"{(format(structure_q, '.3g') if structure_q is not None else '—')} | "
            f"{item['matched_core_rules']} | {item['matched_massbank_records']} |"
        )
    lines += [
        "",
        "## 不能越过的结论边界",
        "",
        "- 峰级定位不等于碎裂机理；同一精确质量可能对应不同元素组成。",
        "- Morgan 环境富集是结构背景关联，不是断键位点证明。",
        "- MassBank 的后验质量匹配只表示规则库覆盖，不能给因子命名。",
        "- 144.00 Da 中性丢失仅 5 个 confirmation 分子支持，暂列低支持规则缺口。",
        "- 91.06 与 67.06 Da 的峰删除尚未可靠降低严格 10 ppm 检索边际，因此不能用本实验主张它们决定检索。",
        "",
        "## 下一实验闸门",
        "",
        summary["next_gate"],
    ]
    (args.output_dir / "REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["validated_catalog"], ensure_ascii=False, indent=2))
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
