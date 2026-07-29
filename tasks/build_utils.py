"""
共用数据加载模块 — 从 annotated01.mgf 构建各类索引

一次解析，多次使用：
  ik_to_smi     — InChIKey → SMILES
  ik_to_fm      — InChIKey → FORMULA
  ik_to_pm      — InChIKey → precursor_mz (取第一张谱的)
  ik_to_murcko  — InChIKey → Murcko scaffold
  ik_to_specs   — InChIKey → list of spectra (peak lists)
  fm_to_iks     — FORMULA → list of IKs
  murcko_to_iks — Murcko scaffold → list of IKs

用法:
    from tasks.build_utils import load_indices
    idx = load_indices()
"""
import os, hashlib, json
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


MGF_PATH = 'data/annotated01.mgf'
CACHE_DIR = 'tasks/_cache'
CACHE_PATH = f'{CACHE_DIR}/indices.json'


def parse_annotated01(mgf_path=MGF_PATH):
    """
    Parse annotated01.mgf and return:
      ik_to_smi, ik_to_fm, ik_to_pm, ik_to_peaks, ik_counts
    where ik_to_peaks[ik] = [(mz1,int1), (mz2,int2), ...]  (first spectrum only)
    """
    ik_to_smi = {}
    ik_to_fm = {}
    ik_to_pm = {}
    ik_to_peaks = {}
    ik_counts = defaultdict(int)

    cur_ik = None; cur_smi = None; cur_fm = None
    cur_pm = None; cur_peaks = []

    with open(mgf_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc='Parse annotated01', total=138_000_000, unit=' lines', unit_scale=True):
            line = line.strip()
            if not line:
                if cur_ik and cur_smi and cur_peaks:
                    ik_counts[cur_ik] += 1
                    if cur_ik not in ik_to_smi:
                        ik_to_smi[cur_ik] = cur_smi
                        if cur_fm: ik_to_fm[cur_ik] = cur_fm
                        if cur_pm: ik_to_pm[cur_ik] = cur_pm
                        ik_to_peaks[cur_ik] = cur_peaks
                cur_ik = None; cur_smi = None; cur_fm = None
                cur_pm = None; cur_peaks = []
                continue

            if line.startswith('SMILES='):
                cur_smi = line[7:].strip()
            elif line.startswith('INCHIKEY='):
                key = line[9:].strip()
                cur_ik = key[:27] if len(key) >= 27 else key
            elif line.startswith('FORMULA='):
                cur_fm = line[8:].strip()
            elif line.startswith('PEPMASS='):
                try: cur_pm = float(line[8:].strip().split('/')[0])
                except: pass
            elif line[0].isdigit() or (line[0] == '-' and len(line) > 1 and line[1].isdigit()):
                p = line.split()
                if len(p) >= 2:
                    try:
                        mz, intensity = float(p[0]), float(p[1])
                        if mz > 0 and intensity > 0:
                            cur_peaks.append((mz, intensity))
                    except: pass

        # Last entry (no trailing blank line)
        if cur_ik and cur_smi and cur_peaks:
            ik_counts[cur_ik] += 1
            if cur_ik not in ik_to_smi:
                ik_to_smi[cur_ik] = cur_smi
                if cur_fm: ik_to_fm[cur_ik] = cur_fm
                if cur_pm: ik_to_pm[cur_ik] = cur_pm
                ik_to_peaks[cur_ik] = cur_peaks

    return ik_to_smi, ik_to_fm, ik_to_pm, ik_to_peaks, dict(ik_counts)


def compute_murcko(ik_to_smi):
    """Compute Murcko scaffold for each unique IK"""
    ik_to_murcko = {}
    for ik, smi in tqdm(ik_to_smi.items(), desc='Murcko scaffolds'):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                if scaff:
                    ik_to_murcko[ik] = scaff
            except: pass
    return ik_to_murcko


def build_indices(ik_to_smi, ik_to_fm, ik_to_pm, ik_to_peaks, ik_counts, ik_to_murcko):
    """Build compound grouping indices"""
    # T1: Formula → IKs
    fm_to_iks = defaultdict(list)
    for ik, fm in ik_to_fm.items():
        fm_to_iks[fm].append(ik)
    # Keep only formulas with ≥2 different IKs
    fm_to_iks_multi = {fm: iks for fm, iks in fm_to_iks.items() if len(iks) >= 2}

    # T2: Murcko → IKs
    murcko_to_iks = defaultdict(list)
    for ik, scaff in ik_to_murcko.items():
        murcko_to_iks[scaff].append(ik)
    murcko_to_iks_multi = {s: iks for s, iks in murcko_to_iks.items() if len(iks) >= 2}

    return {
        'ik_to_smi': ik_to_smi,
        'ik_to_fm': ik_to_fm,
        'ik_to_pm': ik_to_pm,
        'ik_to_peaks': ik_to_peaks,
        'ik_counts': ik_counts,
        'ik_to_murcko': ik_to_murcko,
        'fm_to_iks': fm_to_iks_multi,
        'murcko_to_iks': murcko_to_iks_multi,
    }


def load_indices(force_rebuild=False):
    """
    Load indices from cache or build from scratch.
    Returns dict with all index structures.
    """
    if not force_rebuild and os.path.exists(CACHE_PATH):
        print(f'[load_indices] Loading from cache: {CACHE_PATH}')
        with open(CACHE_PATH, 'r') as f:
            raw = json.load(f)
        # Convert JSON-serializable types back
        raw['ik_counts'] = {k: int(v) for k, v in raw['ik_counts'].items()}
        for key in ['fm_to_iks', 'murcko_to_iks']:
            raw[key] = {k: v for k, v in raw[key].items()}
        return raw

    print('[load_indices] Building indices from annotated01.mgf...')
    ik_to_smi, ik_to_fm, ik_to_pm, ik_to_peaks, ik_counts = parse_annotated01()
    ik_to_murcko = compute_murcko(ik_to_smi)
    indices = build_indices(ik_to_smi, ik_to_fm, ik_to_pm, ik_to_peaks, ik_counts, ik_to_murcko)

    # Cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Convert to JSON-safe types
    cache_data = {}
    for k, v in indices.items():
        if k == 'ik_to_peaks':
            # Keep only first 50 peaks for caching (used as reference)
            cache_data[k] = {ik: [(round(mz, 4), round(i, 4)) for mz, i in pk[:50]]
                           for ik, pk in v.items()}
        elif isinstance(v, dict):
            cache_data[k] = {str(kk): (vv if not isinstance(vv, dict) else {str(kkk): vvv for kkk, vvv in vv.items()})
                           for kk, vv in v.items()}
        else:
            cache_data[k] = v

    with open(CACHE_PATH, 'w') as f:
        json.dump(cache_data, f)
    print(f'[load_indices] Cached to {CACHE_PATH}')

    return indices


def compute_morgan_fp(smi, radius=2, nbits=2048):
    """Compute Morgan fingerprint for a SMILES string"""
    try:
        from rdkit.Chem import AllChem, DataStructs
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)
    except: return None


def compute_tanimoto(smi_a, smi_b, radius=2, nbits=2048):
    """Compute Tanimoto similarity between two SMILES"""
    try:
        from rdkit.Chem import AllChem, DataStructs
        ma = Chem.MolFromSmiles(smi_a); mb = Chem.MolFromSmiles(smi_b)
        if ma is None or mb is None: return -1
        fpa = AllChem.GetMorganFingerprintAsBitVect(ma, radius, nbits)
        fpb = AllChem.GetMorganFingerprintAsBitVect(mb, radius, nbits)
        return DataStructs.TanimotoSimilarity(fpa, fpb)
    except: return -1


def compute_mces(smi_a, smi_b):
    """Compute MCES between two SMILES using myopic-mces"""
    from myopic_mces import MCES
    try:
        result = MCES(smi_a, smi_b)
        mces_raw = result[1]
        return mces_raw
    except Exception as e:
        return None

# Stats helpers
def print_pair_stats(pairs, task_name):
    """Print summary statistics for a pair list"""
    print(f'\n=== {task_name} STATS ===')

    if not pairs:
        print('  No pairs!')
        return

    # Count by type
    type_counts = defaultdict(int)
    for p in pairs:
        t = p.get('type', 'unknown')
        type_counts[t] += 1
    for t, c in type_counts.items():
        print(f'  {t}: {c}')

    total = len(pairs)
    print(f'  Total: {total}')

    # Formula diversity
    if 'fm_a' in pairs[0]:
        formulas = set()
        for p in pairs:
            formulas.add(p.get('fm_a', ''))
            formulas.add(p.get('fm_b', ''))
        print(f'  Unique formulas: {len(formulas)}')

    # IK diversity
    iks = set()
    for p in pairs:
        for k in ('ik', 'ik_a', 'ik_b'):
            if k in p:
                iks.add(p[k])
    print(f'  Unique IKs: {len(iks)}')

    return type_counts
