import json, sys, urllib.request, re

def get_assay_files(acc):
    url = f"https://www.ebi.ac.uk/metabolights/ws/studies/{acc}/files"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
    except Exception as e:
        return []
    files = []
    for f in d.get('study', []):
        if f.get('type') == 'metadata_assay':
            files.append(f.get('file'))
    return files

def parse_assay(txt):
    # ISA-tab; find header row with 'Parameter Value[Instrument]' etc
    lines = txt.splitlines()
    header = None
    idx = {}
    for i, ln in enumerate(lines):
        if 'Parameter Value[Instrument]' in ln or 'Parameter Value[Scan polarity]' in ln:
            header = ln.split('\t')
            for j, h in enumerate(header):
                idx[h] = j
            data = []
            for ln2 in lines[i+1:]:
                if not ln2.strip():
                    continue
                cols = ln2.split('\t')
                if len(cols) <= max(idx.values(), default=0):
                    continue
                row = {k: (cols[v] if v < len(cols) else '') for k, v in idx.items()}
                data.append(row)
            return data
    return []

def main(accs):
    for acc in accs:
        afiles = get_assay_files(acc)
        instrs = set(); pols = set(); analyzers = set()
        for af in afiles:
            txt = fetch_file(acc, af)
            for row in parse_assay(txt):
                if row.get('Parameter Value[Instrument]'):
                    instrs.add(row['Parameter Value[Instrument]'])
                if row.get('Parameter Value[Scan polarity]'):
                    pols.add(row['Parameter Value[Scan polarity]'])
                if row.get('Parameter Value[Mass analyzer]'):
                    analyzers.add(row['Parameter Value[Mass analyzer]'])
        print(f"{acc}: INSTR={sorted(instrs)} POLARITY={sorted(pols)} ANALYZER={sorted(analyzers)}")

def fetch_file(acc, fname):
    url = f"https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{acc}/{fname}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read().decode('utf-8', 'ignore')
    except Exception as e:
        return ""

if __name__ == '__main__':
    main(sys.argv[1:])
