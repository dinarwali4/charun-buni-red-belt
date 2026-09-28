import numpy as np, heapq, json
from scipy import ndimage as ndi
from pyproj import Transformer
d=np.load('data/dem20.npz'); Z0=d['Z'].astype('float64'); T=d['T']
st=2; Z=Z0[::st,::st].copy(); H,W=Z.shape; cell=40.0          # 40 m for speed
bad=Z<100; Z[bad]=-1
# priority-flood fill
filled=Z.copy(); seen=np.zeros_like(bad); pq=[]
for r in range(H):
    for c in (0,W-1): heapq.heappush(pq,(filled[r,c],r,c)); seen[r,c]=True
for c in range(W):
    for r in (0,H-1):
        if not seen[r,c]: heapq.heappush(pq,(filled[r,c],r,c)); seen[r,c]=True
nb=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
while pq:
    z,r,c=heapq.heappop(pq)
    for dr,dc in nb:
        rr,cc=r+dr,c+dc
        if 0<=rr<H and 0<=cc<W and not seen[rr,cc]:
            seen[rr,cc]=True; 
            if filled[rr,cc]<=z: filled[rr,cc]=z+1e-3
            heapq.heappush(pq,(filled[rr,cc],rr,cc))
# D8 receivers
pad=np.pad(filled,1,constant_values=np.inf); best=np.zeros((H,W)); rec=np.full((H,W),-1)
for k,(dr,dc) in enumerate(nb):
    dist=cell*np.hypot(dr,dc); drop=(filled-pad[1+dr:1+dr+H,1+dc:1+dc+W])/dist
    better=drop>best; best[better]=drop[better]; rec[better]=k
order=np.argsort(-filled,axis=None); acc=np.ones(H*W)
dr_=np.array([n[0] for n in nb]); dc_=np.array([n[1] for n in nb])
rflat=rec.ravel()
for i in order:
    k=rflat[i]
    if k<0: continue
    r,c=divmod(i,W); j=(r+dr_[k])*W+(c+dc_[k]); acc[j]+=acc[i]
A=acc.reshape(H,W)*cell*cell/1e6     # km2
np.savez_compressed('data/drainage40.npz',A=A.astype('float32'),rec=rec.astype('int8'))
# stream crossings of red-belt margins
g=np.load('data/integr.npz'); edge=g['edge'][::st,::st]; belt=g['belt'][::st,::st]
streams=A>=1.0
cross=streams&ndi.binary_dilation(edge,iterations=1)
lab,n=ndi.label(cross,structure=np.ones((3,3)))
tr=Transformer.from_crs(32643,4326,always_xy=True)
pts=[]
for k in range(1,n+1):
    m=lab==k; i=np.argmax(np.where(m,A,0)); r,c=divmod(i,W)
    lo,la=tr.transform(T[2]+(c*st+1)*20,T[5]-(r*st+1)*20)
    pts.append(dict(lat=round(la,5),lon=round(lo,5),catchment_km2=round(float(A[r,c]),1),elev=int(Z0[r*st,c*st])))
pts=[p for p in pts if 72.05<=p['lon']<=72.33 and 36.11<=p['lat']<=36.34]
pts=sorted(pts,key=lambda p:-p['catchment_km2'])
print(len(pts)); [print(p) for p in pts[:25]]
json.dump(pts,open('stream_sites.json','w'),indent=1)
# sanity: Mastuj river accumulation at Buni
r,c=[int(v) for v in ((T[5]-Transformer.from_crs(4326,32643,always_xy=True).transform(72.2625,36.2701)[1])/40,(Transformer.from_crs(4326,32643,always_xy=True).transform(72.2625,36.2701)[0]-T[2])/40)]
print('max acc near Buni (km2):',A[r-15:r+15,c-15:c+15].max())
