"""Step 2 —— 建立大参考库：把已下载的开源谱库合并成统一 MGF。

目标：把「唯一化合物天花板」从 MONA-pos 的 3,150 抬高到开源实验谱库的上限（~3 万，
Nature Methods 2025 七库对比）。输入两个已下载的库：

  * MassSpecGym (NeurIPS 2024 Spotlight, arXiv:2410.23326)
      data/reference/massspecgym/data/auxiliary/MassSpecGym.mgf   (231,104 谱)
      SIMULATION_CHALLENGE 是该谱是否进入 spectrum-simulation benchmark 子任务的资格标记，
      不是实验谱/模拟谱来源标签；不得据此筛谱。本构建器纳入 True/False 两类，并在输出中
      原样保留该标记，便于后续按 benchmark 合同审计。
  * GNPS 公共库 (ALL_GNPS.mgf, 2,091,446 谱)
      用 INCHI= 而非 INCHIKEY=；LIBRARYQUALITY：1=Gold / 2=Silver / 3=Bronze / 4=无
      InChI 的低置信条目（实测全库分布：Gold 1.51M / Silver 6.4k / Bronze 424k / 4 149k；
      官方等级见 https://ccms-ucsd.github.io/GNPSDocumentation/spectrumcuration/）。
      默认只留 Gold+Silver（--min-gnps-quality silver）；Bronze=「推测性」注释，
      进参考库会引入错误 SMILES/InChI，但能换来更长的化合物尾巴；「4」无 InChI、始终丢弃。

输出统一 MGF 严格对齐 annotation.cli.parse_mgf 的契约：
    BEGIN IONS / NAME= / SMILES= / INCHIKEY= / PEPMASS= / 峰行(m/z 强度) / END IONS
InChIKey 一律用 rdkit 从 (INCHI 优先，其次 SMILES) 现算 27 位全键（含立体层）；
MassSpecGym 自带的 14 位 InChIKey 只有骨架层、不够区分立体异构体，故重算。

去重：键 = (完整 InChIKey, 电离模式, 推断加合物)。每个键默认只保留一张代表谱；
不同极性和不同加合物不能互相替代。代表谱选法：质量(金>银>铜)优先，再按峰数。
输出 unified_pos.mgf / unified_neg.mgf + build_report.json。

用途边界：这个输出是压缩后的代表谱检索库，不是 ChemAware 的重复谱训练库。v3 会为
MassSpecGym 代表谱保留原始 identifier、仪器类型、碰撞能、fold 和 simulation-challenge
资格标记，但同键跨条件重复谱仍会被去重。需要完整重复条件的训练任务必须从原始库另建
metadata-preserving replicate bank，不能复用本输出冒充多正样本。

已知限制（v1 不做，留待后续）：
  * 不加合离子归一化：GNPS 同一化合物可能以 [M+H]+ / [M+Na]+ / [M+NH4]+ 等多张谱出现，
    按峰数选代表可能选到 [M+Na]+（其 m/z 与 [M+H]+ 查询差 ~22，20ppm 硬约束会拒掉 →
    该化合物成「死重」）。--max-spectra-per-compound 调大（如 3）可概率缓解；根治需用
    rdkit 精确质量反推加合离子并只留 [M+H]+/[M-H]-。

用法 (本地 conda dreams_env，Windows):
    # 冒烟测试（各读前 3000 条，秒出）
    python tasks/build_reference_library.py --limit 3000 \
        --massspecgym data/reference/massspecgym/data/auxiliary/MassSpecGym.mgf \
        --gnps data/reference/gnps/ALL_GNPS.mgf \
        --out-dir data/reference/unified_v3_smoke
    # 全量（GNPS 2.09M 谱 + MassSpecGym 231,104 谱，rdkit 算 InChIKey 约 20-30 分钟）
    python tasks/build_reference_library.py \
        --massspecgym data/reference/massspecgym/data/auxiliary/MassSpecGym.mgf \
        --gnps data/reference/gnps/ALL_GNPS.mgf \
        --out-dir data/reference/unified_v3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except Exception:  # pragma: no cover - 环境缺 rdkit 时给出明确报错
    HAS_RDKIT = False

# GNPS LIBRARYQUALITY 整数 -> 质量分（越高越可信）；MassSpecGym 视为金级等价。
_QUALITY_SCORE = {1: 3, 2: 2, 3: 1}          # Gold=3, Silver=2, Bronze=1
_MASSSPECGYM_QUALITY = 3                      # curated benchmark，与 Gold 同级
_MIN_QUALITY_RANK = {"gold": 1, "silver": 2, "bronze": 3}  # 保留的 GNPS 质量整数上限

_PROTON = 1.007276466621
_POSITIVE_ADDUCTS = {
    "[M+H]+": (1.0, _PROTON),
    "[M+NH4]+": (1.0, 18.033823),
    "[M+Na]+": (1.0, 22.989218),
    "[M+K]+": (1.0, 38.963158),
    "[M+2H]2+": (0.5, _PROTON),
    "[M+H+Na]2+": (0.5, (_PROTON + 22.989218) / 2.0),
    "[M+3H]3+": (1.0 / 3.0, _PROTON),
}
_NEGATIVE_ADDUCTS = {
    "[M-H]-": (1.0, -_PROTON),
    "[M+Cl]-": (1.0, 34.969402),
    "[M+FA-H]-": (1.0, 44.998201),
    "[M+CH3COO]-": (1.0, 59.013851),
    "[M-2H]2-": (0.5, -_PROTON),
    "[M-3H]3-": (1.0 / 3.0, -_PROTON),
}


def resolve_identity(inchi: str | None, smiles: str | None) -> tuple[str | None, str | None]:
    """从 (INCHI 优先, 其次 SMILES) 算出 (27 位全键 InChIKey, 规范 SMILES)。失败返回 (None, None)。"""
    mol = None
    if inchi:
        mol = Chem.MolFromInchi(inchi)
    if mol is None and smiles:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    return Chem.MolToInchiKey(mol), Chem.MolToSmiles(mol, isomericSmiles=True)


def infer_common_adduct(smiles: str, precursor_mz: float, ion: str, ppm: float, abs_da: float) -> tuple[str | None, float]:
    """Infer a common adduct from structure exact mass and validate PEPMASS.

    Returns ``(adduct, ppm_error)``.  ``adduct`` is ``None`` when PEPMASS is
    inconsistent with every pre-specified common adduct.  The absolute floor
    avoids rejecting low-resolution library records solely because of rounding.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, float("inf")
    exact_mass = float(Descriptors.ExactMolWt(mol))
    hypotheses = _POSITIVE_ADDUCTS if ion == "pos" else _NEGATIVE_ADDUCTS
    best_name, best_ppm = None, float("inf")
    for name, (mass_scale, offset) in hypotheses.items():
        expected = exact_mass * mass_scale + offset
        error_da = abs(precursor_mz - expected)
        error_ppm = error_da / max(expected, 1e-12) * 1e6
        if error_da <= max(abs_da, expected * ppm * 1e-6) and error_ppm < best_ppm:
            best_name, best_ppm = name, error_ppm
    return best_name, best_ppm


def validate_precursor(rec: dict, ppm: float, abs_da: float) -> bool:
    adduct, error_ppm = infer_common_adduct(rec["smiles"], rec["precursor_mz"], rec["ion"], ppm, abs_da)
    rec["adduct"] = adduct
    rec["structure_mass_ppm"] = error_ppm
    return adduct is not None


def iter_mgf_blocks(path: Path):
    """流式逐块 yield (fields: dict, peaks: list[tuple[mz, intensity]])，内存有界。

    峰行兼容空格/制表符分隔（GNPS 用制表符，MassSpecGym 用空格）。
    """
    cur: dict | None = None
    peaks: list[tuple[float, float]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line == "BEGIN IONS":
                cur = {}
                peaks = []
            elif line == "END IONS":
                if cur is not None:
                    yield cur, peaks
                cur = None
            elif cur is not None and "=" in line:
                k, v = line.split("=", 1)
                cur[k.strip()] = v.strip()
            elif cur is not None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz, it = float(parts[0]), float(parts[1])
                        if mz > 0.0 and it > 0.0:
                            peaks.append((mz, it))
                    except ValueError:
                        pass


def normalize_massspecgym(fields: dict, peaks: list[tuple[float, float]]) -> dict | None:
    """MassSpecGym 记录 -> 统一 record；无法确定身份或前体质量时返回 None。

    ``SIMULATION_CHALLENGE`` 仅表示该记录是否满足官方 spectrum-simulation
    benchmark 的子集条件，不表示谱图是模拟生成的，因此这里只保留字段而不筛选。
    """
    try:
        pmz = float(fields["PRECURSOR_MZ"].split()[0])
    except (KeyError, ValueError, IndexError):
        return None
    ik, smi = resolve_identity(None, fields.get("SMILES", "").strip() or None)
    if ik is None:
        return None
    adduct = fields.get("ADDUCT", "[M+H]+")
    ion = "neg" if "-" in adduct else "pos"
    return {
        "inchikey": ik, "smiles": smi, "name": fields.get("IDENTIFIER", "").strip(),
        "precursor_mz": pmz, "peaks": peaks, "ion": ion,
        "quality_score": _MASSSPECGYM_QUALITY, "n_peaks": len(peaks), "source": "massspecgym",
        "instrument_type": fields.get("INSTRUMENT_TYPE", "").strip(),
        "collision_energy": fields.get("COLLISION_ENERGY", "").strip(),
        "fold": fields.get("FOLD", "").strip(),
        "simulation_challenge": fields.get("SIMULATION_CHALLENGE", "").strip(),
    }


def normalize_gnps(fields: dict, peaks: list[tuple[float, float]], max_q: int) -> dict | None:
    """GNPS 记录 -> 统一 record；质量/身份/极性不过关返回 None（None 也可能=被质量滤掉）。"""
    try:
        q = int(fields.get("LIBRARYQUALITY", "0").strip() or "0")
    except ValueError:
        q = 0
    if not (1 <= q <= max_q):
        return None
    try:
        pmz = float(fields["PEPMASS"].split()[0])
    except (KeyError, ValueError, IndexError):
        return None
    inchi = fields.get("INCHI", "").strip()
    inchi = None if inchi in ("", "N/A", "NA", "n/a") else inchi
    smiles = fields.get("SMILES", "").strip()
    smiles = None if smiles in ("", "N/A", "NA", "n/a") else smiles
    ik, smi = resolve_identity(inchi, smiles)
    if ik is None:
        return None
    ionmode = fields.get("IONMODE", "").strip().lower()
    if "neg" in ionmode:
        ion = "neg"
    elif "pos" in ionmode:
        ion = "pos"
    else:
        return None  # 无法确定极性，不猜
    return {
        "inchikey": ik, "smiles": smi, "name": fields.get("NAME", "").strip(),
        "precursor_mz": pmz, "peaks": peaks, "ion": ion,
        "quality_score": _QUALITY_SCORE[q], "n_peaks": len(peaks), "source": "gnps",
    }


def _score_key(r: dict) -> tuple[int, int]:
    return (r["quality_score"], r["n_peaks"])


def feed(best: dict, rec: dict, max_n: int, min_peaks: int) -> None:
    """并入代表谱表；每个 (inchikey, ion, adduct) 只留最高质量的 max_n 条。"""
    if rec["n_peaks"] < min_peaks:
        return
    # Keep independently queryable adducts.  Collapsing all adducts to one
    # spectrum creates dead entries under a precursor-mass hard constraint.
    key = (rec["inchikey"], rec["ion"], rec.get("adduct", "unknown"))
    lst = best.setdefault(key, [])
    score = _score_key(rec)
    if len(lst) < max_n:
        lst.append(rec)
        lst.sort(key=_score_key, reverse=True)
    elif score > _score_key(lst[-1]):
        lst[-1] = rec
        lst.sort(key=_score_key, reverse=True)


def write_mgf(records: list[dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write("BEGIN IONS\n")
            f.write(f"NAME={rec['name']}\n")
            f.write(f"SMILES={rec['smiles']}\n")
            f.write(f"INCHIKEY={rec['inchikey']}\n")
            f.write(f"PEPMASS={rec['precursor_mz']:.6f}\n")
            f.write(f"ADDUCT={rec.get('adduct', '')}\n")
            f.write(f"SOURCE={rec['source']}\n")
            for key, field_name in (
                ("instrument_type", "INSTRUMENT_TYPE"),
                ("collision_energy", "COLLISION_ENERGY"),
                ("fold", "FOLD"),
                ("simulation_challenge", "SIMULATION_CHALLENGE"),
            ):
                value = rec.get(key, "")
                if value != "":
                    f.write(f"{field_name}={value}\n")
            for mz, it in sorted(rec["peaks"]):
                f.write(f"{mz:.6f} {it:.6f}\n")
            f.write("END IONS\n")
    return len(records)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--massspecgym", type=Path,
                   default=ROOT / "data/reference/massspecgym/data/auxiliary/MassSpecGym.mgf")
    p.add_argument("--gnps", type=Path,
                   default=ROOT / "data/reference/gnps/ALL_GNPS.mgf")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/reference/unified_v3")
    p.add_argument("--min-gnps-quality", choices=["gold", "silver", "bronze"], default="silver")
    p.add_argument("--max-spectra-per-compound", type=int, default=1)
    p.add_argument("--min-peaks", type=int, default=3)
    p.add_argument("--precursor-validation-ppm", type=float, default=30.0)
    p.add_argument("--precursor-validation-abs-da", type=float, default=0.01)
    p.add_argument("--no-precursor-validation", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="每库只读前 N 条（0=全量，冒烟测试用）")
    args = p.parse_args()

    if not HAS_RDKIT:
        raise SystemExit("[build] 需要 rdkit：conda activate dreams_env && pip install rdkit")
    if not args.massspecgym.exists():
        raise SystemExit(f"[build] 找不到 MassSpecGym: {args.massspecgym}")
    if not args.gnps.exists():
        raise SystemExit(f"[build] 找不到 GNPS: {args.gnps}")

    max_q = _MIN_QUALITY_RANK[args.min_gnps_quality]
    best: dict = {}
    stats = {
        "massspecgym": {
            "n_in": 0,
            "n_simulation_challenge_member": 0,
            "n_simulation_challenge_nonmember": 0,
            "n_invalid": 0,
            "n_precursor_invalid": 0,
            "n_kept": 0,
        },
        "gnps": {"n_in": 0, "n_quality_dropped": 0, "n_invalid": 0,
                 "n_no_polarity": 0, "n_precursor_invalid": 0, "n_kept": 0},
    }

    print(f"[build] MassSpecGym <- {args.massspecgym}", flush=True)
    t0 = time.time()
    for i, (fields, peaks) in enumerate(iter_mgf_blocks(args.massspecgym)):
        if args.limit and i >= args.limit:
            break
        stats["massspecgym"]["n_in"] += 1
        if fields.get("SIMULATION_CHALLENGE", "False").strip() == "True":
            stats["massspecgym"]["n_simulation_challenge_member"] += 1
        else:
            stats["massspecgym"]["n_simulation_challenge_nonmember"] += 1
        rec = normalize_massspecgym(fields, peaks)
        if rec is None:
            stats["massspecgym"]["n_invalid"] += 1
            continue
        if not args.no_precursor_validation and not validate_precursor(
            rec, args.precursor_validation_ppm, args.precursor_validation_abs_da
        ):
            stats["massspecgym"]["n_precursor_invalid"] += 1
            continue
        if args.no_precursor_validation:
            rec["adduct"] = "unvalidated"
        stats["massspecgym"]["n_kept"] += 1
        feed(best, rec, args.max_spectra_per_compound, args.min_peaks)
        if i and i % 20000 == 0:
            print(f"    MassSpecGym {i} 谱, 累计唯一键 {len(best)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[build] GNPS <- {args.gnps} (min_quality={args.min_gnps_quality})", flush=True)
    t0 = time.time()
    for i, (fields, peaks) in enumerate(iter_mgf_blocks(args.gnps)):
        if args.limit and i >= args.limit:
            break
        stats["gnps"]["n_in"] += 1
        try:
            q = int(fields.get("LIBRARYQUALITY", "0").strip() or "0")
        except ValueError:
            q = 0
        if not (1 <= q <= max_q):
            stats["gnps"]["n_quality_dropped"] += 1
            continue
        rec = normalize_gnps(fields, peaks, max_q)
        if rec is None:
            # 极性未知 vs 其它非法，分别计数
            ionmode = fields.get("IONMODE", "").strip().lower()
            if "neg" not in ionmode and "pos" not in ionmode:
                stats["gnps"]["n_no_polarity"] += 1
            else:
                stats["gnps"]["n_invalid"] += 1
            continue
        if not args.no_precursor_validation and not validate_precursor(
            rec, args.precursor_validation_ppm, args.precursor_validation_abs_da
        ):
            stats["gnps"]["n_precursor_invalid"] += 1
            continue
        if args.no_precursor_validation:
            rec["adduct"] = "unvalidated"
        stats["gnps"]["n_kept"] += 1
        feed(best, rec, args.max_spectra_per_compound, args.min_peaks)
        if i and i % 200000 == 0:
            print(f"    GNPS {i} 谱, 累计唯一键 {len(best)} ({time.time()-t0:.0f}s)", flush=True)

    # 摊平去重表 -> pos/neg 两桶
    pos_recs: list[dict] = []
    neg_recs: list[dict] = []
    for (_ik, ion, _adduct), lst in best.items():
        for rec in lst:
            (pos_recs if ion == "pos" else neg_recs).append(rec)
    pos_recs.sort(key=lambda r: r["precursor_mz"])
    neg_recs.sort(key=lambda r: r["precursor_mz"])

    pos_out = args.out_dir / "unified_pos.mgf"
    neg_out = args.out_dir / "unified_neg.mgf"
    n_pos = write_mgf(pos_recs, pos_out)
    n_neg = write_mgf(neg_recs, neg_out)

    pos_ik = {r["inchikey"] for r in pos_recs}
    neg_ik = {r["inchikey"] for r in neg_recs}
    union = pos_ik | neg_ik
    report = {
        "schema_semantics": {
            "SIMULATION_CHALLENGE": (
                "MassSpecGym spectrum-simulation benchmark subset membership; "
                "not experimental-versus-synthetic provenance and never used as a source filter"
            ),
            "library_role": "deduplicated representative-spectrum retrieval library, not a replicate training bank",
        },
        "filters": {
            "min_gnps_quality": args.min_gnps_quality,
            "max_spectra_per_compound": args.max_spectra_per_compound,
            "min_peaks": args.min_peaks,
            "precursor_validation": not args.no_precursor_validation,
            "precursor_validation_ppm": args.precursor_validation_ppm,
            "precursor_validation_abs_da": args.precursor_validation_abs_da,
            "limit": args.limit,
        },
        "per_source": stats,
        "library": {
            "n_pos_spectra": n_pos,
            "n_neg_spectra": n_neg,
            "n_unique_pos_inchikey": len(pos_ik),
            "n_unique_neg_inchikey": len(neg_ik),
            "n_unique_total_inchikey": len(union),
            "n_unique_both_polarities": len(pos_ik & neg_ik),
            "n_unique_skeletons_block1": len({ik.split("-")[0] for ik in union}),
        },
        "outputs": {"pos": str(pos_out), "neg": str(neg_out)},
    }
    (args.out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 建库结果 =====", flush=True)
    print(json.dumps(report["per_source"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["library"], ensure_ascii=False, indent=2), flush=True)
    print(f"\n[build] pos -> {pos_out} ({n_pos} 谱)", flush=True)
    print(f"[build] neg -> {neg_out} ({n_neg} 谱)", flush=True)
    print(f"[build] report -> {args.out_dir / 'build_report.json'}", flush=True)
    print(f"[build] 唯一化合物（完整 InChIKey）= {len(union)}，骨架级(block1) = "
          f"{len({ik.split('-')[0] for ik in union})}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
