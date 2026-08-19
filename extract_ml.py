import json, sys, urllib.request

def get(acc):
    url=f"https://www.ebi.ac.uk/metabolights/ws/studies/{acc}"
    req=urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    return json.load(urllib.request.urlopen(req, timeout=60))

ids=sys.argv[1:]
for acc in ids:
    try:
        d=get(acc)
    except Exception as e:
        print(f"=== {acc} ERROR {e}"); continue
    inv=d.get('isaInvestigation',{})
    st=inv.get('studies',[{}])[0]
    print("="*80)
    print(f"ACC: {acc}")
    print("TITLE:", inv.get('title'))
    desc=(inv.get('description') or '')
    print("DESC:", desc[:300])
    # platforms
    plats=[]
    for a in st.get('assays',[]):
        p=a.get('technologyPlatform')
        if p: plats.append(p)
    print("PLATFORMS:", sorted(set(plats)))
    # protocols instrument hints
    instrs=set()
    for p in st.get('protocols',[]):
        nm=p.get('name','')
        for c in p.get('components',[]):
            for k in ('parameterValue','termAccession'):
                pass
        txt=json.dumps(p)
        for kw in ['Q Exactive','Orbitrap','Q-TOF','QTOF','TripleTOF','6600','Synapt','Agilent','Waters','Thermo','Bruker','timsTOF','Exploris']:
            if kw.lower() in txt.lower(): instrs.add(kw)
    print("INSTR_HINTS:", sorted(instrs))
    # samples characteristics
    orgs=set(); parts=set(); diseases=set()
    for s in st.get('samples',[]):
        ch=s.get('characteristics',{}) if isinstance(s.get('characteristics'),dict) else {}
        if ch.get('Organism'): orgs.add(str(ch['Organism']))
        if ch.get('Organism part'): parts.add(str(ch['Organism part']))
        for k in ch:
            if 'disease' in k.lower() or 'variant' in k.lower():
                diseases.add(f"{k}={ch[k]}")
    print("ORGANISMS:", sorted(orgs))
    print("PARTS:", sorted(parts))
    print("DISEASE/VARIANT:", sorted(diseases))
    pubs=inv.get('publications') or st.get('publications') or []
    print("PUBS:", [(p.get('doi'), (p.get('title') or '')[:70]) for p in pubs][:3])
    print("PUBDATE:", inv.get('publicReleaseDate'))
