import matplotlib.patheffects
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, json
from pyproj import Transformer
d=np.load('data/dem20.npz'); T=d['T']; Z=d['Z']
cmp=np.load('data/s2_comp.npz')['comp']; I=np.load('data/s2_idx.npz'); dd=np.load('data/dem_derivs.npz')
fw=Transformer.from_crs(4326,32643,always_xy=True)
def rc(lon,lat): x,y=fw.transform(lon,lat); return (T[5]-y)/20,(x-T[2])/20
A=(72.08,36.12,72.32,36.30); (r0,c0),(r1,c1)=rc(A[0],A[3]),rc(A[2],A[1])
r0,r1,c0,c1=int(r0),int(r1),int(c0),int(c1)
rgb=np.stack([cmp[2],cmp[1],cmp[0]],-1)[r0:r1,c0:c1]; rgb=np.nan_to_num(np.clip(rgb/0.32,0,1)**(1/1.5))
hs=dd['hs'][r0:r1,c0:c1]
places=json.load(open('places_sel.json'))
fig,ax=plt.subplots(1,2,figsize=(20,8.6),dpi=110)
ax[0].imshow(rgb*(0.55+0.45*hs[...,None])); ax[0].set_title('Sentinel-2 true colour (12-scene Aug–Sep median) × hillshade')
red=np.where(I['rock'],I['redness'],np.nan)[r0:r1,c0:c1]
ax[1].imshow(hs,cmap='gray'); im=ax[1].imshow(red,cmap='RdYlBu_r',vmin=0.0,vmax=0.22,alpha=.85); ax[1].set_title('Redness (B4−B3)/(B4+B3) on exposed rock — the hematite-red bands')
plt.colorbar(im,ax=ax[1],fraction=.03)
for a in ax:
    for n,(la,lo) in places.items():
        r,c=rc(lo,la); r-=r0; c-=c0
        if 0<r<r1-r0 and 0<c<c1-c0: a.plot(c,r,'o',ms=3,color='yellow'); a.text(c+6,r,n,color='white',fontsize=7,path_effects=[matplotlib.patheffects.withStroke(linewidth=2,foreground='k')])
    # graticule
    for lo in np.arange(72.10,72.33,0.05):
        r,c=rc(lo,36.21); a.axvline(c-c0,color='w',lw=.3,alpha=.5); a.text(c-c0,r1-r0-5,f'{lo:.2f}E',color='w',fontsize=7)
    for la in np.arange(36.15,36.31,0.05):
        r,c=rc(72.2,la); a.axhline(r-r0,color='w',lw=.3,alpha=.5); a.text(3,r-r0-3,f'{la:.2f}N',color='w',fontsize=7)
    a.set_xticks([]); a.set_yticks([])
import matplotlib.patheffects
plt.tight_layout(); plt.savefig('fig/mapview_red.png',bbox_inches='tight')
