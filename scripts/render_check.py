# Render Sentinel-2 true colour draped on the DEM from the fitted camera; compare with screenshot
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from pyproj import Transformer
exec(open('camfit2.py').read().split('best=None')[0])
p=np.load('cam2.npy')
d=np.load('data/dem20.npz'); Z=d['Z']; T=d['T']; H,W=Z.shape
cmp=np.load('data/s2_comp.npz')['comp']
rgb=np.stack([cmp[2],cmp[1],cmp[0]],-1); rgb=np.nan_to_num(np.clip(rgb/0.30,0,1)**(1/1.4)*255)
r,c=np.mgrid[0:H,0:W]; X=T[2]+(c+.5)*20; Y=T[5]-(r+.5)*20
lon,lat=Transformer.from_crs(32643,4326,always_xy=True).transform(X,Y)
x=(lon-lon0)*111320*np.cos(np.radians(lat0)); y=(lat-lat0)*110574
uv,zc=project(p,np.c_[x.ravel(),y.ravel(),Z.ravel()])
ok=(zc>0)&(Z.ravel()>100)&(uv[:,0]>=16)&(uv[:,0]<1984)&(uv[:,1]>=228)&(uv[:,1]<1285)
# painter's algorithm far->near with splat radius by distance
idx=np.where(ok)[0]; idx=idx[np.argsort(-zc[idx])]
img=np.zeros((1057,1968,3),np.uint8); cols=rgb.reshape(-1,3)
u=(uv[idx,0]-16).astype(int); v=(uv[idx,1]-228).astype(int); rad=np.clip((p[6]*28/zc[idx]).astype(int),0,14)
for R in range(0,15):
    s=rad==R
    for du in range(-R,R+1):
        for dv in range(-R,R+1):
            uu=np.clip(u[s]+du,0,1967); vv=np.clip(v[s]+dv,0,1056)
            img[vv,uu]=cols[idx[s]]
# re-paint in strict far->near order for correct occlusion at small radii
img2=np.zeros_like(img)
for k0 in range(0,len(idx),200000):
    k=idx[k0:k0+200000]; uu=(uv[k,0]-16).astype(int); vv=(uv[k,1]-228).astype(int); rr=np.clip((p[6]*28/zc[k]).astype(int),0,14)
    for R in (0,1,2,3,5,8,14):
        s=rr>=R
        for du in (-R,0,R):
            for dv in (-R,0,R):
                img2[np.clip(vv[s]+dv,0,1056),np.clip(uu[s]+du,0,1967)]=cols[k[s]]
shot=np.array(Image.open('shot.png'))[228:1285,16:1984]
g=lambda a: np.clip((a.astype(float)/np.percentile(a[a>0],99))*255,0,255).astype(np.uint8)
fig,ax=plt.subplots(2,1,figsize=(13,17),dpi=100)
ax[0].imshow(shot); ax[0].set_title('Your screenshot (Mapbox, Maxar)'); ax[1].imshow(g(img2)); ax[1].set_title('Independent render: Sentinel-2 12-scene Aug–Sep median on Copernicus DEM from fitted camera')
for a in ax: a.axis('off')
plt.tight_layout(); plt.savefig('fig/render_vs_shot.png',bbox_inches='tight')
Image.fromarray(g(img2)).save('fig/render_only.png')
# numeric co-registration check: correlate 'redness' of screenshot vs render on 16px blocks
def red(a): a=a.astype(float); return (a[...,0]-a[...,1])/(a[...,0]+a[...,1]+1)
m=img2.sum(2)>0
def blk(a): h,w=(a.shape[0]//16)*16,(a.shape[1]//16)*16; return a[:h,:w].reshape(h//16,16,w//16,16).mean((1,3))
rs,rr_,mm=blk(red(shot)),blk(red(img2)),blk(m.astype(float))>0.9
cc=np.corrcoef(rs[mm],rr_[mm])[0,1]
# null: shift render by 200 px horizontally
rr_sh=np.roll(rr_,12,axis=1); cc0=np.corrcoef(rs[mm],rr_sh[mm])[0,1]
print('redness correlation screenshot vs render: r=%.3f (control shifted 190px: r=%.3f); blocks=%d'%(cc,cc0,mm.sum()))
