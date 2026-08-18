"""
scan_rt_coverage.py — RT 数据门扫描（只读，不训练）

目的：回答"引入 RT 先验前，我们到底有没有 RT"这一个数据门槛问题。
对 data/ 下的谱文件（.mgf / .msp / .mgf.gz / .msp.gz）逐条流式扫描，
只读元数据（不加载完整峰列表），统计：

  - 每个文件有多少张谱、多少张带 RT；
  - RT 用的是哪个字段名（RTINSECONDS / RETENTION_TIME / RETENTIONTIME / RT / ...）；
  - RT 单位（秒/分钟/无法确定，含推断规则）；
  - RT 数值范围（min/max/mean/抽样中位数），用于核对单位是否合理。

注意：
  - 保留指数 Retention_index / RetentionIndex 是 RI，不是 RT，明确排除；
  - 本脚本只做元数据普查，不做任何 RT 归一化或建模；
  - 单位"?"表示字段名本身不含单位（如裸 RT / RETENTION_TIME），此时按数值
    量级给一个"推断单位"（s?/m?），只作提示、不作定论。

用法：
  python tasks/scan_rt_coverage.py                        # 全量扫描 data/
  python tasks/scan_rt_coverage.py --max-per-file 2000    # 每文件只读前 2000 谱（快检）
  python tasks/scan_rt_coverage.py --lib gnps,massbank    # 只扫指定库
"""
import argparse
import csv
import gzip
import json
import os
import re
from collections import defaultdict

# 谱文件扩展名（文本格式；.gz 透明解压）
SPEC_EXTS = ('.mgf', '.msp', '.mgf.gz', '.msp.gz')

# 排除的派生文件（已知不含 RT 的构建产物）
EXCLUDE_BASENAMES = {'annotated01.mgf'}

# 字段名归一化：小写 + 去掉空格/下划线/连字符
_NORM = re.compile(r'[\s_\-]+')


def normalize_key(k: str) -> str:
    return _NORM.sub('', k.lower())


# 归一化字段名 -> 单位提示（s=秒 m=分钟 ?=字段名不含单位）
RT_UNIT = {
    'rtinseconds': 's', 'rtinsec': 's', 'rtsec': 's', 'rtseconds': 's',
    'retentiontimeseconds': 's', 'retentiontimesec': 's',
    'rtinminutes': 'm', 'rtinmin': 'm', 'rtmin': 'm', 'rtminutes': 'm',
    'retentiontimeminutes': 'm', 'retentiontimemin': 'm',
    'rt': '?', 'rtime': '?', 'retentiontime': '?',
}

# 归一化字段名 -> 明确排除（RI 保留指数不是 RT）
NOT_RT = {
    'retentionindex', 'retentiontimeindex', 'ri', 'kovats', 'kovatsindex',
    'retentionindices',
}

_VALUE = re.compile(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?')

# 元数据行：KEY 后接 = 或 :（首字符字母，键内允许空格/下划线/连字符/数字）
_META = re.compile(r'^([A-Za-z][A-Za-z0-9 _\-]*)\s*[:=]\s*(.*)$')


def parse_rt_value(raw: str):
    """从 RT 字段的原始值里抽出数值 + 值内单位提示（如有 "min"/"sec"）。"""
    s = raw.strip().strip('"\'')
    if not s:
        return None, None
    low = s.lower()
    unit_hint = None
    if 'min' in low:
        unit_hint = 'm'
    elif 'sec' in low:
        unit_hint = 's'
    m = _VALUE.search(s)
    if not m:
        return None, unit_hint
    return float(m.group()), unit_hint


class _FileAgg:
    """单个文件的流式统计（内存有界）。"""

    def __init__(self):
        self.n_total = 0
        self.n_with_rt = 0
        self.n_rt_non_numeric = 0
        self.fields = set()
        self.units = set()
        self.v_count = 0
        self.v_min = None
        self.v_max = None
        self.v_sum = 0.0
        self.v_reservoir = []   # 抽样前 N 个数值，用于中位数估计
        self.v_reservoir_cap = 2000

    def add_rt(self, field: str, val, unit: str):
        self.fields.add(field)
        self.units.add(unit)
        if val is None:
            self.n_rt_non_numeric += 1
            return
        self.v_count += 1
        self.v_min = val if self.v_min is None else min(self.v_min, val)
        self.v_max = val if self.v_max is None else max(self.v_max, val)
        self.v_sum += val
        if len(self.v_reservoir) < self.v_reservoir_cap:
            self.v_reservoir.append(val)

    def median(self):
        if not self.v_reservoir:
            return None
        s = sorted(self.v_reservoir)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _resolve_unit(field_unit: str, val_unit_hint, val):
    """单位判定优先级：字段名 > 值内单位 > 数值量级推断。"""
    if field_unit in ('s', 'm'):
        return field_unit
    if val_unit_hint in ('s', 'm'):
        return val_unit_hint
    if val is not None and val > 120:
        return 's?'   # 量级像秒（LC 单次运行通常 2–40 min，即 120–2400 s）
    if val is not None:
        return 'm?'   # 量级像分钟
    return '?'


def _fmt_of(path: str) -> str:
    """按扩展名判断谱格式：msp 用 'Name:' 分谱，mgf 用空行/BEGIN IONS 分谱。"""
    p = path[:-3] if path.lower().endswith('.gz') else path
    return 'msp' if p.lower().endswith('.msp') else 'mgf'


def _scan_file(path: str, max_spectra, fmt: str):
    """流式扫描一个谱文件，返回 (_FileAgg, 说明, 错误信息)。"""
    agg = _FileAgg()
    is_gz = path.lower().endswith('.gz')
    opener = gzip.open if is_gz else open
    kw = {'encoding': 'utf-8', 'errors': 'ignore'}

    cur = {'has_rt': False, 'any_content': False}

    def finalize():
        if cur['any_content']:
            agg.n_total += 1
            if cur['has_rt']:
                agg.n_with_rt += 1
        cur['has_rt'] = False
        cur['any_content'] = False

    try:
        f = opener(path, 'rt', **kw)
    except OSError as e:
        return agg, 'unreadable', str(e)

    try:
        with f:
            for line in f:
                if max_spectra and agg.n_total >= max_spectra:
                    break
                s = line.rstrip('\n').rstrip('\r').strip()
                if not s:
                    finalize()
                    continue
                # 谱边界：MGF 的 BEGIN IONS；MSP 的 Name:
                if fmt == 'mgf' and s.upper() == 'BEGIN IONS':
                    finalize()
                    cur['any_content'] = True
                    continue
                if fmt == 'msp' and re.match(r'^Name\s*:', s, re.IGNORECASE):
                    finalize()
                    cur['any_content'] = True
                    # 继续解析这一行（Name 行非 RT，安全）
                # 元数据行
                m = _META.match(s)
                if m:
                    key, raw = m.group(1), m.group(2)
                    nk = normalize_key(key)
                    if nk in RT_UNIT:
                        val, vh = parse_rt_value(raw)
                        unit = _resolve_unit(RT_UNIT[nk], vh, val)
                        agg.add_rt(key, val, unit)
                        cur['has_rt'] = True
                        cur['any_content'] = True
                    elif nk in NOT_RT:
                        cur['any_content'] = True
                    elif len(key) > 1:
                        cur['any_content'] = True
                    continue
                # 峰行（数字或负号开头）也算有内容
                if s and (s[0].isdigit() or s[0] == '-'):
                    cur['any_content'] = True
    except Exception as e:
        return agg, 'error', str(e)

    finalize()
    return agg, ('gzip' if is_gz else 'text'), None


def _library_of(root, path):
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if len(parts) == 1:
        return '(root)'
    return parts[0]


def main():
    ap = argparse.ArgumentParser(description='RT 数据门扫描')
    ap.add_argument('--root', default='data', help='谱数据根目录')
    ap.add_argument('--max-per-file', type=int, default=None,
                    help='每文件最多读多少张谱（默认全量）')
    ap.add_argument('--lib', default=None,
                    help='逗号分隔的库名过滤（如 gnps,massbank），默认全部')
    ap.add_argument('--out-dir', default='data/validation/rt_coverage_scan',
                    help='输出目录')
    args = ap.parse_args()

    lib_filter = None
    if args.lib:
        lib_filter = {x.strip().lower() for x in args.lib.split(',') if x.strip()}

    files = []
    for r, dirs, names in os.walk(args.root):
        for n in names:
            low = n.lower()
            if not low.endswith(SPEC_EXTS):
                continue
            if n in EXCLUDE_BASENAMES:
                continue
            p = os.path.join(r, n)
            lib = _library_of(args.root, p).lower()
            if lib_filter and lib not in lib_filter:
                continue
            files.append((lib, p))
    files.sort()

    print(f'[scan_rt_coverage] 找到 {len(files)} 个谱文件，开始扫描...\n')

    rows = []
    capped = False
    for i, (lib, path) in enumerate(files, 1):
        fmt = _fmt_of(path)
        agg, fmt_label, err = _scan_file(path, args.max_per_file, fmt)
        if err:
            print(f'  [{i}/{len(files)}] {os.path.relpath(path, args.root)}: {fmt_label} {err}')
        cov = (agg.n_with_rt / agg.n_total) if agg.n_total else None
        mean = (agg.v_sum / agg.v_count) if agg.v_count else None
        if args.max_per_file and agg.n_total >= args.max_per_file:
            capped = True
        rows.append({
            'library': lib,
            'file': os.path.relpath(path, args.root),
            'format': fmt_label,
            'n_total': agg.n_total,
            'n_with_rt': agg.n_with_rt,
            'n_rt_non_numeric': agg.n_rt_non_numeric,
            'rt_coverage': round(cov, 4) if cov is not None else None,
            'rt_fields': ','.join(sorted(agg.fields)),
            'rt_unit': ','.join(sorted(agg.units)),
            'rt_min': agg.v_min,
            'rt_max': agg.v_max,
            'rt_mean': round(mean, 3) if mean is not None else None,
            'rt_sample_median': round(agg.median(), 3) if agg.median() is not None else None,
        })

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'rt_coverage_summary.csv')
    cols = ['library', 'file', 'format', 'n_total', 'n_with_rt', 'n_rt_non_numeric',
            'rt_coverage', 'rt_fields', 'rt_unit', 'rt_min', 'rt_max',
            'rt_mean', 'rt_sample_median']
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json_path = os.path.join(args.out_dir, 'rt_coverage_detail.json')
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump({'args': vars(args), 'rows': rows}, fh, indent=2, ensure_ascii=False)

    print('\n===== 按库汇总 =====')
    bylib = defaultdict(lambda: {'files': 0, 'total': 0, 'with_rt': 0, 'units': set()})
    for r in rows:
        b = bylib[r['library']]
        b['files'] += 1
        b['total'] += r['n_total']
        b['with_rt'] += r['n_with_rt']
        if r['rt_unit']:
            for u in r['rt_unit'].split(','):
                b['units'].add(u)
    print(f'{"库":<18}{"文件":>5}{"总谱":>10}{"带RT":>10}{"覆盖率":>9}  单位')
    for lib in sorted(bylib):
        b = bylib[lib]
        cov = (b['with_rt'] / b['total']) if b['total'] else 0.0
        print(f'{lib:<18}{b["files"]:>5}{b["total"]:>10}{b["with_rt"]:>10}'
              f'{cov*100:>8.1f}%  {",".join(sorted(b["units"])) or "-"}')
    total = sum(b['total'] for b in bylib.values())
    total_rt = sum(b['with_rt'] for b in bylib.values())
    cov_all = (total_rt / total) if total else 0.0
    print(f'\n合计：{total} 张谱，{total_rt} 张带 RT，覆盖率 {cov_all*100:.1f}%')
    if capped:
        print(f'  (使用了 --max-per-file {args.max_per_file}，覆盖率是抽样值)')
    print(f'\n详细结果：{csv_path}\n原始 JSON：{json_path}')


if __name__ == '__main__':
    main()
