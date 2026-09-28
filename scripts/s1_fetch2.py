import numpy as np, rasterio, pystac_client, planetary_computer as pc, collections, concurrent.futures as cf
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
d=np.load('data/dem20.npz'); T=rasterio.Affine(*d['T']); H,W=d['Z'].shape
c=pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',modifier=pc.sign_inplace)
items=list(c.search(collections=['sentinel-1-rtc'],bbox=[72.05,36.08,72.35,36.36],datetime='2025-07-01/2025-09-30').items())
groups=collections.defaultdict(list)
for it in items: groups[(it.properties['sat:relative_orbit'],it.properties['sat:orbit_state'])].append(it)
def rd(href):
    with rasterio.open(href) as s, WarpedVRT(s,crs='EPSG:32643',transform=T,width=W,height=H,resampling=Resampling.average,src_nodata=0,nodata=0) as v:
        return v.read(1).astype('float32')
out={}
for key,its in groups.items():
    its=sorted(its,key=lambda i:i.datetime)[::max(1,len(its)//5)][:5]
    for p in ('vv','vh'):
        with cf.ThreadPoolExecutor(5) as ex: arrs=list(ex.map(rd,[it.assets[p].href for it in its]))
        s=np.stack(arrs); s[s<=0]=np.nan; out[f'{p}_{key[0]}_{key[1]}']=np.nanmedian(s,0).astype('float32')
        print(key,p,len(arrs),'valid',round(float(np.isfinite(out[f"{p}_{key[0]}_{key[1]}"]).mean()),3),flush=True)
np.savez_compressed('data/s1_rtc_summer2025.npz',**out); print('saved')
