import json,sys,urllib.request
BB="72.12,36.13,72.30,36.235"
def links(sn,want):
    d=json.load(urllib.request.urlopen(f"https://cmr.earthdata.nasa.gov/search/granules.json?short_name={sn}&bounding_box={BB}&page_size=200"))
    out=[]
    for g in d['feed']['entry']:
        if any(w in g['title'] for w in want):
            for l in g['links']:
                h=l['href']
                if h.startswith('https') and 'data#' in l.get('rel','') and not h.endswith('.xml'): out.append(h)
    return out
emit=links('EMITL2BMIN',['20230630T','20231005T','20240822T','20241005T','20230423T'])
aster=links('AST_07XT',['00409082002','00409222002','00409282001','00410132003055902'])
tir=links('AST_05',['00409082002','00409222002','00409282001','00410132003055902'])
open('links_emit.txt','w').write('\n'.join(emit)); open('links_aster.txt','w').write('\n'.join(aster+tir))
print(len(emit),len(aster),len(tir)); print('\n'.join(emit[:12])); print('\n'.join(aster[:12])); print('\n'.join(tir[:8]))
