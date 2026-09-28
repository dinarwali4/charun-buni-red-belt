import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
f=np.load('data/s2_raw.npz'); S=f['stack'].astype('float32'); ids=f['ids']; bn=list(f['bandnames'])
N=S.shape[0]; print('scenes',N,ids)
scl=S[:,bn.index('SCL')]
good=np.isin(scl,[4,5,6,7])            # veg, bare, water, unclassified(low-prob cloud) ; excludes shadow(2,3), cloud(8,9,10), snow(11)
R=S[:,:10]/10000.0                      # earthsearch boa_offset_applied=True -> no -1000 subtraction
R[~np.repeat(good[:,None],10,1)]=np.nan
R[R<=0.0]=np.nan
comp=np.nanmedian(R,0); nobs=good.sum(0)
b={k:comp[i] for i,k in enumerate(bn[:10])}
np.savez_compressed('data/s2_comp.npz',comp=comp.astype('float32'),nobs=nobs.astype('uint8'))
dem=np.load('data/dem20.npz')['Z']
ndvi=(b['B08']-b['B04'])/(b['B08']+b['B04']); ndsi=(b['B03']-b['B11'])/(b['B03']+b['B11']); ndwi=(b['B03']-b['B08'])/(b['B03']+b['B08'])
floor=0.02
rock=np.isfinite(comp).all(0)&(ndvi<0.25)&(ndsi<0.30)&(ndwi<0.1)&(b['B02']>floor)&(b['B04']>floor)&(nobs>=3)&(dem>100)
print('rock-visible fraction of grid',round(float(rock.mean()),3))
idx={
 'ferric_B4B2':b['B04']/b['B02'],
 'redness':(b['B04']-b['B03'])/(b['B04']+b['B03']),
 'ferrous':b['B12']/b['B08']+b['B03']/b['B04'],
 'gossan_B11B4':b['B11']/b['B04'],
 'alOH_carb_B11B12':b['B11']/b['B12'],
 'fe3_900nm_B8AB7':b['B8A']/b['B07'],    # <1 where Fe3+ 0.87-0.9um absorption (hematite/goethite)
 'albedo':(b['B02']+b['B03']+b['B04']+b['B08']+b['B11']+b['B12'])/6,
}
np.savez_compressed('data/s2_idx.npz',rock=rock,ndvi=ndvi.astype('float32'),ndsi=ndsi.astype('float32'),**{k:v.astype('float32') for k,v in idx.items()})
# ---- colour-unit clustering on rock pixels (log band ratios + albedo: illumination-robust)
feat=np.stack([np.log(b['B04']/b['B02']),np.log(b['B04']/b['B03']),np.log(b['B11']/b['B12']),np.log(b['B8A']/b['B07']),
               np.log(b['B11']/b['B8A']),np.log(b['B12']/b['B04']),np.log(idx['albedo'])],-1)
X=feat[rock]; mu=X.mean(0); sd=X.std(0); Xn=(X-mu)/sd
rng=np.random.default_rng(0); sub=Xn[rng.choice(len(Xn),min(200000,len(Xn)),replace=False)]
K=7; km=KMeans(K,n_init=10,random_state=0).fit(sub); lab=km.predict(Xn)
L=np.full(rock.shape,-1,'int8'); L[rock]=lab
np.save('data/units.npy',L)
summ=[]
for k in range(K):
    m=L==k
    summ.append(dict(unit=k,frac=round(float(m.sum()/rock.sum()),3),
      rgb=[round(float(np.nanmedian(b[x][m])),4) for x in ('B04','B03','B02')],
      **{n:round(float(np.nanmedian(v[m])),4) for n,v in idx.items()}, elev_med=round(float(np.median(dem[m])))))
for s in summ: print(s)
json.dump(summ,open('units.json','w'),indent=1)
