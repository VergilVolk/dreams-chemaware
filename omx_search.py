import json, sys, urllib.request, urllib.parse

def search(q, size=6):
    url = "https://www.omicsdi.org/ws/dataset/search?database=MetaboLights&size=%d&query=%s" % (size, urllib.parse.quote(q))
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
    except Exception as e:
        return [("ERROR", str(e), "")]
    out = []
    for ds in d.get('datasets', []):
        out.append((ds.get('id',''), ds.get('title',''), (ds.get("description") or "")[:250]))
    return out

queries = sys.argv[1:]
for q in queries:
    print("="*80)
    print("QUERY:", q)
    for id, title, desc in search(q):
        print("  %s | %s" % (id, title))
        if desc:
            print("      %s" % desc.replace('\n',' ')[:200])
