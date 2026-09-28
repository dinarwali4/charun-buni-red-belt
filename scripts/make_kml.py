import numpy as np, json
from skimage import measure
from shapely.geometry import Polygon
from pyproj import Transformer
d=np.load('data/dem20.npz'); T=d['T']; g=np.load('data/integr.npz'); belt=g['belt']
tr=Transformer.from_crs(32643,4326,always_xy=True)
polys=[]
for cnt in measure.find_contours(np.pad(belt.astype(float),1),0.5):
    rr,cc=cnt[:,0]-1,cnt[:,1]-1; x=T[2]+(cc+.5)*20; y=T[5]-(rr+.5)*20
    p=Polygon(zip(x,y)).simplify(25)
    if p.is_valid and p.area>0.3e6: polys.append(p)
print('belt polygons',len(polys),'area km2',round(sum(p.area for p in polys)/1e6,1))
def pm(name,desc,lat,lon,style): return f'<Placemark><name>{name}</name><description><![CDATA[{desc}]]></description><styleUrl>#{style}</styleUrl><Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>'
out=['<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Charun–Buni (Upper Chitral) — terrain and prospectivity</name>',
 '<Style id="ss"><IconStyle><color>ffffe500</color><scale>1.0</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/triangle.png</href></Icon></IconStyle></Style>',
 '<Style id="emit"><IconStyle><color>ff0ad6ff</color><scale>1.2</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/star.png</href></Icon></IconStyle></Style>',
 '<Style id="ref"><IconStyle><color>ffffffff</color><scale>0.9</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/info-i.png</href></Icon></IconStyle></Style>',
 '<Style id="belt"><LineStyle><color>ff303bff</color><width>2</width></LineStyle><PolyStyle><color>40303bff</color></PolyStyle></Style>',
 '<Style id="aoi"><LineStyle><color>ffffffff</color><width>1.5</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>']
out.append('<Folder><name>Analysis area</name><Placemark><name>Analysis AOI (72.05–72.32E, 36.11–36.34N)</name><styleUrl>#aoi</styleUrl><Polygon><outerBoundaryIs><LinearRing><coordinates>72.05,36.11,0 72.32,36.11,0 72.32,36.34,0 72.05,36.34,0 72.05,36.11,0</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Folder>')
out.append('<Folder><name>Red-bed belt (Sentinel-2 redness; ≈ Reshun Formation)</name>')
for i,p in enumerate(polys):
    lo,la=tr.transform(*np.array(p.exterior.coords).T)
    out.append(f'<Placemark><name>Red-bed belt {i+1} ({p.area/1e6:.1f} km²)</name><description>Outline from Sentinel-2 redness on exposed rock. USGS OFR 75-556 maps the Reshun Fm (hematite-red siltstone + conglomerate) along this bank from Reshun to Buni; its upper and lower limits are NW-dipping thrusts.</description><styleUrl>#belt</styleUrl><Polygon><outerBoundaryIs><LinearRing><coordinates>'+' '.join(f'{a:.6f},{b:.6f},0' for a,b in zip(lo,la))+'</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')
out.append('</Folder><Folder><name>Stream-sediment sites (SS01–SS14)</name>')
for s in json.load(open('stream_sites_final.json')):
    out.append(pm(s['id'],f"Stream crossing the red-bed belt margin. Catchment {s['catchment_km2']} km² (DEM, 40 m D8), elev {s['elev']} m. Suggested: -80 mesh sediment + panned heavy-mineral concentrate; analyse Sb, As, Au, Ag, Pb, Zn, Cu, W, Hg (aqua-regia ICP-MS for sediment; fire assay Au; scheelite count under UV).",s['lat'],s['lon'],'ss'))
out.append('</Folder><Folder><name>Hyperspectral leads (EMIT)</name>')
out.append(pm('E1 jarosite/copiapite cluster','EMIT L2B: sulphate minerals in ~9 cells (60 m), identified on up to 4 of 5 dates (2023-04 → 2024-10). Elev 3,590–3,835 m, ~1 km from the red-bed margin. Sentinel-2 shows NO corroborating ferric/Al-OH anomaly. Jarosite also forms on pyritic black shale — not diagnostic of ore. Check on the ground.',36.2090,72.2700,'emit'))
out.append(pm('E2 "azurite" scatter','EMIT library matches to azurite in 23 cells on ≥3 dates, scattered over ~1 km² near Buni. Unconfirmed: azurite has carbonate-like SWIR features, and the whole area is carbonate-rich.',36.2620,72.2890,'emit'))
out.append('</Folder><Folder><name>Published reference localities</name>')
out.append(pm('Charun Gol — type locality, Charun Quartzite','USGS OFR 75-556 (Stauffer 1975): white quartzite, ~250 ft, type section at lat 36°13′N, long 72°12′E (coordinates as published, minute precision).',36.2167,72.2000,'ref'))
out.append(pm('Kuragh fossil ridge (Hayden 1914)','Most prolific Devonian fossil locality, ridge <1 mile S of Kuragh (36°13′N 72°10′E as published).',36.2167,72.1667,'ref'))
out.append(pm('Damun Gol laterite (approx.)','USGS OFR 75-556: bright-red nodular lateritic layer 5–20 ft thick in Shogram Fm, ~1 mile due E of Reshun; 35% Al₂O₃, 16% Fe₂O₃, 10% FeO, 22% SiO₂. Position approximate (described, not surveyed).',36.1540,72.1160,'ref'))
out.append('</Folder></Document></kml>')
open('Charun_Buni_Prospectivity.kml','w').write('\n'.join(out)); print('kml ok')
