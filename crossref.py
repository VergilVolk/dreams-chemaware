import json, sys, urllib.request, urllib.parse
def cr(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
        m = d['message']
        year = None
        for k in ('published-print','published-online','published','created'):
            if m.get(k) and m[k].get('date-parts'):
                year = m[k]['date-parts'][0][0]; break
        journal = (m.get('container-title') or [''])[0]
        title = (m.get('title') or [''])[0]
        return f"{doi} | {title[:70]} | {journal} | {year}"
    except Exception as e:
        return f"{doi} | ERROR {e}"
for doi in sys.argv[1:]:
    print(cr(doi))
