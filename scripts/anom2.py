import numpy as np, json
from scipy import ndimage as ndi
from pyproj import Transformer
I=np.load('data/s2_idx.npz'); rock=I['rock']; L=np.load('data/units.npy'); d=np.load('data/dem20.npz'); Z=d['Z']; T=d['T']; H,W=Z.shape
g=np.load('data/integr.npz'); ep={k[2:]:g[k] for k in g.files if k.startswith('E_')}
bands=[1800,2500,3000,3500,4000,4500,5000,7000]
def rz(v,m):
    med=np.nanmedian(v[m]); mad=1.4826*np.nanmedian(np.abs(v[m]-med)); return (v-med)/mad
Zs={}
for nm,v in (('ferric',I['ferric_B4B2']),('gossan',I['gossan_B11B4']),('alOH',I['alOH_carb_B11B12']),('fe900',1/I['fe3_900nm_B8AB7'])):
    z=np.full((H,W),np.nan)
    for k in range(7):
        for a,b in zip(bands[:-1],bands[1:]):
            m=(L==k)&rock&(Z>=a)&(Z<b)
            if m.sum()>500: z[m]=rz(v,m)[m]
    Zs[nm]=z
r,c=np.mgrid[0:H,0:W]; lon,lat=Transformer.from_crs(32643,4326,always_xy=True).transform(T[2]+(c+.5)*20,T[5]-(r+.5)*20)
res={}
for thr in (2.5,3.0,3.5):
    score=sum((np.nan_to_num(Zs[k])>thr).astype(int) for k in Zs)
    cand=ndi.binary_opening((score>=2)&rock,iterations=1)
    lab,n=ndi.label(cand,structure=np.ones((3,3))); sz=ndi.sum(cand,lab,range(1,n+1))
    res[thr]=(n,int((sz>=6).sum()))
    rate=[round(float(cand[rock&(Z>=a)&(Z<b)].mean()*100),3) for a,b in zip(bands[:-1],bands[1:])]
    print('thr',thr,'clusters',n,'>=6px',(sz>=6).sum(),'rate by elev band %',rate)
    if thr==3.0: keep=(cand,lab,sz,score)
cand,lab,sz,score=keep
out=[]
for k in np.where(sz>=6)[0]+1:
    m=lab==k
    out.append(dict(lat=round(float(lat[m].mean()),5),lon=round(float(lon[m].mean()),5),area_ha=round(float(m.sum()*0.04),2),elev=int(Z[m].mean()),
        unit=int(np.bincount(L[m][L[m]>=0]).argmax()),**{'z_'+kk:round(float(np.nanmean(v[m])),1) for kk,v in Zs.items()},
        emit_sulphate=int(ep['sulphate'][m].max()),emit_cu=int(ep['cu_carbonate'][m].max()),emit_wm=int(ep['whitemica_clay'][m].max()),emit_nonmin=int(ep['nonmineral'][m].max()),
        in_belt=bool(g['belt'][m].mean()>.5)))
out=sorted(out,key=lambda t:-t['area_ha'])
for t in out[:20]: print(t)
json.dump(out,open('targets_v2.json','w'),indent=1)
np.savez_compressed('data/anom2.npz',cand=cand,**{'z_'+k:v.astype('float32') for k,v in Zs.items()})
