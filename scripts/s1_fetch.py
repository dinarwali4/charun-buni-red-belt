import numpy as np, rasterio, pystac_client, planetary_computer as pc, collections
from rasterio.warp import reproject, Resampling
d=np.load('data/dem20.npz'); T=rasterio.Affine(*d['T']); H,W=d['Z'].shape
c=pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',modifier=pc.sign_inplace)
items=list(c.search(collections=['sentinel-1-rtc'],bbox=[72.05,36.08,72.35,36.36],datetime='2025-07-01/2025-09-30').items())
groups=collections.defaultdict(list)
for it in items: groups[(it.properties['sat:relative_orbit'],it.properties['sat:orbit_state'])].append(it)
print({k:len(v) for k,v in groups.items()})
out={}
for key,its in groups.items():
    if len(its)<3: continue
    stk={'vv':[],'vh':[]}
    for it in its[:8]:
        for p in ('vv','vh'):
            a=np.zeros((H,W),'float32')
            with rasterio.open(it.assets[p].href) as s:
                reproject(rasterio.band(s,1),a,dst_transform=T,dst_crs='EPSG:32643',resampling=Resampling.average,src_nodata=0,dst_nodata=0)
            stk[p].append(a)
        print(key,it.id,flush=True)
    for p in stk:
        s=np.stack(stk[p]); s[s<=0]=np.nan
        out[f'{p}_{key[0]}_{key[1]}']=np.nanmedian(s,0).astype('float32')
np.savez_compressed('data/s1_rtc_summer2025.npz',**out); print('saved',list(out))
