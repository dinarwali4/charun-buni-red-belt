import numpy as np
from scipy.optimize import least_squares
from PIL import Image
from pyproj import Transformer
exec(open('camfit2.py').read().split('best=None')[0])
F=528.5/np.tan(np.radians(36.87/2))
d=np.load('data/dem20.npz'); Z=d['Z']; T=d['T']; H,W=Z.shape
cmp=np.load('data/s2_comp.npz')['comp']; rgb=np.stack([cmp[2],cmp[1],cmp[0]],-1); rgb=np.nan_to_num(np.clip(rgb/0.30,0,1))
st=4; Zs=Z[::st,::st]; rg=rgb[::st,::st]
r,c=np.mgrid[0:H:st,0:W:st]; lon,lat=Transformer.from_crs(32643,4326,always_xy=True).transform(T[2]+(c+.5)*20,T[5]-(r+.5)*20)
gx_=(lon-lon0)*111320*np.cos(np.radians(lat0)); gy_=(lat-lat0)*110574
shot=np.array(Image.open('shot.png')).astype(float)[228:1285,16:1984]
def red(a): return (a[...,0]-a[...,1])/(a[...,0]+a[...,1]+1e-3)
def blk(a,n=24): h,w=(a.shape[0]//n)*n,(a.shape[1]//n)*n; return a[:h,:w].reshape(h//n,n,w//n,n,*a.shape[2:]).mean((1,3))
shot_r=blk(red(shot)); shot_b=blk(shot.mean(2))
def render_score(p,ex):
    XX=np.c_[gx_.ravel(),gy_.ravel(),Zs.ravel()*ex]
    uv,zc=project(np.r_[p[:6],F],XX)
    ok=(zc>0)&(Zs.ravel()>100)&(uv[:,0]>=16)&(uv[:,0]<1984)&(uv[:,1]>=228)&(uv[:,1]<1285)
    i=np.where(ok)[0]; i=i[np.argsort(-zc[i])]
    img=np.zeros((1057,1968,3)); u=(uv[i,0]-16).astype(int); v=(uv[i,1]-228).astype(int)
    rad=np.clip((F*80/zc[i]).astype(int),0,40)
    for R in (0,2,5,10,20,40):
        s=rad>=R
        for du in (-R,0,R):
            for dv in (-R,0,R):
                img[np.clip(v[s]+dv,0,1056),np.clip(u[s]+du,0,1967)]=rg.reshape(-1,3)[i[s]]
    m=blk((img.sum(2)>0).astype(float))>0.95
    if m.sum()<200: return -1,m.mean()
    rr=blk(red(img)); bb=blk(img.mean(2))
    return np.corrcoef(shot_r[m],rr[m])[0,1]+np.corrcoef(shot_b[m],bb[m])[0,1], m.mean()
results=[]
for ex in (1.0,1.5,2.0):
    Xe=X.copy(); Xe[:,2]*=ex
    def res(p): return (project(np.r_[p,F],Xe)[0]-U).ravel()
    for yaw0 in (-20,-10,0,10):
      for dist in (15e3,25e3,40e3,60e3):
        for pit in (25,35,45,60):
          yr=np.radians(yaw0)
          p0=[-dist*np.sin(yr)*np.cos(np.radians(pit)),4000-dist*np.cos(yr)*np.cos(np.radians(pit)),2500*ex+dist*np.sin(np.radians(pit)),yaw0,pit,0]
          try: r_=least_squares(res,p0,x_scale=[1e3,1e3,1e3,5,5,2])
          except Exception: continue
          rms=np.sqrt(np.mean(r_.fun**2)*2)
          results.append((rms,ex,tuple(np.round(r_.x,2))))
results=sorted(set(results))
seen=[];cands=[]
for rms,ex,p in results:
    if any(np.allclose(p[:3],q[:3],atol=500) and e==ex for e,q in seen): continue
    seen.append((ex,p)); cands.append((rms,ex,p))
cands=[c_ for c_ in cands if c_[0]<60][:15]
out=[]
for rms,ex,p in cands:
    sc,cov=render_score(np.array(p),ex)
    lat_c=lat0+p[1]/110574; lon_c=lon0+p[0]/(111320*np.cos(np.radians(lat0)))
    out.append((sc,rms,ex,p)); print(f'score={sc:.3f} cover={cov:.2f} rms={rms:.1f}px ex={ex} cam=({lat_c:.3f}N,{lon_c:.3f}E, {p[2]:.0f} m) yaw={p[3]} pitch={p[4]} roll={p[5]}')
best=max(out); np.save('cam_best.npy',np.r_[best[3],F,best[2]]); print('BEST',best)
