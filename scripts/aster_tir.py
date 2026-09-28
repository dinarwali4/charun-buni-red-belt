import numpy as np, rasterio, glob, re
from rasterio.warp import reproject, Resampling
d=np.load('data/dem20.npz'); T=rasterio.Affine(*d['T']); H,W=d['Z'].shape
grans=sorted({re.search(r'AST_05_(\d+)_',f).group(1) for f in glob.glob('data/aster/AST_05_*')})
QI=[];CI=[];MI=[]
for g in grans:
    E={}
    for b in (10,11,12,13,14):
        f=glob.glob(f'data/aster/AST_05_{g}_*B{b}.tif')[0]; a=np.full((H,W),np.nan,'float32')
        with rasterio.open(f) as s:
            src=s.read(1).astype('float32')/1000; src[src<=0]=np.nan
            reproject(src,a,src_transform=s.transform,src_crs=s.crs,dst_transform=T,dst_crs='EPSG:32643',resampling=Resampling.bilinear,src_nodata=np.nan,dst_nodata=np.nan)
        E[b]=a
    ok=np.all([(E[b]>0.80)&(E[b]<0.995) for b in E],axis=0)
    qi=E[11]**2/(E[10]*E[12]); ci=E[13]/E[14]; mi=E[12]/E[13]
    for arr,L in ((qi,QI),(ci,CI),(mi,MI)): L.append(np.where(ok,arr,np.nan))
    print(g,'valid frac',round(float(ok.mean()),3),'QI p50',np.nanmedian(qi[ok]).round(4),'CI p50',np.nanmedian(ci[ok]).round(4))
out={k:np.nanmedian(np.stack(v),0).astype('float32') for k,v in (('QI',QI),('CI',CI),('MI',MI))}
out['n']=np.sum(np.isfinite(np.stack(QI)),0).astype('uint8')
np.savez_compressed('data/aster_tir.npz',**out); print('saved; pixels with >=2 scenes',(out['n']>=2).mean())
