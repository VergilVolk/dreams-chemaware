import json, sys, re, urllib.request

def clean(html):
    t = re.sub(r'<[^>]+>', ' ', html or '')
    t = re.sub(r'&nbsp;', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def fetch(acc):
    url = f"https://www.ebi.ac.uk/metabolights/ws/studies/{acc}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
    except Exception as e:
        return f"{acc}: ERROR {e}"
    inv = d.get('isaInvestigation', {})
    studies = inv.get('studies', [])
    if not studies:
        return f"{acc}: no studies"
    s = studies[0]
    title = s.get('title','')
    desc = clean(s.get('description',''))[:600]
    pubs = s.get('publications', [])
    pub_info = []
    for p in pubs:
        pub_info.append({
            'title': p.get('title',''),
            'doi': p.get('doi',''),
            'pmid': p.get('pubMedID',''),
        })
    platforms = set()
    for a in s.get('assays', []):
        tp = a.get('technologyPlatform','')
        if tp:
            platforms.add(tp)
    # search description for MS2 hints
    ms_hints = []
    for kw in ['MS/MS','MS2','DDA','DIA','Orbitrap','Q-Exactive','Q Exactive','QTOF','Q-TOF','tandem','DDA','collision','MRM','multiple reaction']:
        if kw.lower() in (title+desc).lower():
            ms_hints.append(kw)
    out = {
        'acc': acc,
        'title': title[:300],
        'desc': desc,
        'pubs': pub_info,
        'platforms': sorted(platforms),
        'ms_hints': sorted(set(ms_hints)),
    }
    return out

for acc in sys.argv[1:]:
    o = fetch(acc)
    if isinstance(o, str):
        print(o)
    else:
        print(json.dumps(o, ensure_ascii=False, indent=1))
    print()
