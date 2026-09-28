import numpy as np, rasterio, glob, re
from rasterio.warp import reproject, Resampling
d=np.load('data/dem20.npz'); T=rasterio.Affine(*d['T']); H,W=d['Z'].shape
grans=sorted({re.search(r'AST_07XT_(\d+)_',f).group(1) for f in glob.glob('data/aster/AST_07XT_*')})
acc={k:[] for k in ('AlOH','kaol_alun','MgOH_carb','ferric')}
for g in grans:
    B={}; bad=[]
    for b in ['01','02','03N','04','05','06','07','08','09']:
        fs=glob.glob(f'data/aster/AST_07XT_{g}_*B{b}.tif')
        if not fs: bad.append(b); continue
        a=np.full((H,W),np.nan,'float32')
        with rasterio.open(fs[0]) as s:
            src=s.read(1).astype('float32'); nod=src<=0
            src=src*0.001; src[nod]=np.nan
            v=src[np.isfinite(src)]
            if v.size and np.median(v)>=0.999*v.max(): bad.append(b+'(clipped)')
            reproject(src,a,src_transform=s.transform,src_crs=s.crs,dst_transform=T,dst_crs='EPSG:32643',resampling=Resampling.bilinear,src_nodata=np.nan,dst_nodata=np.nan)
        B[b]=a
    if bad: print(g,'REJECT',bad); continue
    ndvi=(B['03N']-B['02'])/(B['03N']+B['02']); ndsi=(B['01']-B['04'])/(B['01']+B['04'])
    ok=np.all([np.isfinite(B[b])&(B[b]>0.02) for b in B],0)&(ndvi<0.25)&(ndsi<0.4)&(B['01']<0.6)
    print(g,'usable frac',round(float(ok.mean()),3),'NDVI p50',np.nanmedian(ndvi).round(3))
    idx={'AlOH':(B['05']+B['07'])/B['06'],'kaol_alun':(B['04']+B['06'])/B['05'],'MgOH_carb':(B['06']+B['09'])/(B['07']+B['08']),'ferric':B['02']/B['01']}
    for k,v in idx.items(): acc[k].append(np.where(ok,v,np.nan))
out={k:np.nanmedian(np.stack(v),0).astype('float32') for k,v in acc.items()}
out['n']=np.isfinite(np.stack(acc['AlOH'])).sum(0).astype('uint8')
np.savez_compressed('data/aster_swir.npz',**out); print('pixels with >=1 usable',round(float((out['n']>=1).mean()),3))
