import numpy as np, netCDF4, glob, json, collections
from pyproj import Transformer
AOI=(72.05,36.11,72.32,36.34); CORE=(72.12,36.13,72.30,36.235)
res={}
allpix=[]
for f in sorted(glob.glob('data/emit/EMIT_L2B_MIN_001_*.nc')):
    ds=netCDF4.Dataset(f); ds.set_auto_mask(False)
    gt=ds.getncattr('geotransform'); gx=ds['location/glt_x'][:]; gy=ds['location/glt_y'][:]
    names=[str(n) for n in ds['mineral_metadata/name'][:]]; idx=ds['mineral_metadata/index'][:]
    id2n=dict(zip(idx.tolist(),names))
    H,W=gx.shape
    lon=gt[0]+(np.arange(W)+.5)*gt[1]; lat=gt[3]+(np.arange(H)+.5)*gt[5]
    LON,LAT=np.meshgrid(lon,lat)
    v=(gx>0)&(gy>0)
    out={}
    for g in (1,2):
        raw=ds[f'group_{g}_mineral_id'][:]; bd=ds[f'group_{g}_band_depth'][:]
        o=np.full(gx.shape,-9999,'int32'); o[v]=raw[gy[v]-1,gx[v]-1]
        b=np.full(gx.shape,np.nan,'float32'); b[v]=bd[gy[v]-1,gx[v]-1]
        out[g]=(o,b)
    date=f.split('_')[4][:8]
    for nm,(a,b_,c,e) in {'aoi':AOI,'core':CORE}.items():
        m=(LON>=a)&(LON<=c)&(LAT>=b_)&(LAT<=e)
        n=m.sum(); 
        if n==0: res[(date,nm)]=dict(pixels=0); continue
        o1=out[1][0][m]; o2=out[2][0][m]
        retrieved=((o1!=-9999)|(o2!=-9999)).sum(); ident=((o1>0)|(o2>0)).sum()
        cnt=collections.Counter([id2n.get(int(x),str(x)) for x in np.r_[o1[o1>0],o2[o2>0]]])
        res[(date,nm)]=dict(pixels=int(n),retrieved=int(retrieved),identified=int(ident),top=cnt.most_common(15))
        if nm=='aoi':
            for g in (1,2):
                o,bd=out[g]; mm=m&(o>0)
                for la,lo,x,bb in zip(LAT[mm],LON[mm],o[mm],bd[mm]): allpix.append((date,g,float(la),float(lo),id2n.get(int(x)),float(bb)))
for k,v in res.items(): print(k,{kk:vv for kk,vv in v.items() if kk!='top'}); print('    ',v.get('top','')[:12] if v.get('top') else '')
json.dump(allpix,open('emit_pixels.json','w'))
