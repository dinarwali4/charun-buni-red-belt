import numpy as np, rasterio
from scipy.optimize import least_squares
exec(open('camfit2.py').read().split('best=None')[0])
def resid8(q): P=q[:7]; uv,_=project(P,X); return (uv-U-np.array([0,q[7]])).ravel()
p0=np.r_[np.load('cam2.npy'),0]
r=least_squares(resid8,p0,x_scale=[1e3,1e3,1e3,5,5,2,100,5]); q=r.x
res=np.linalg.norm(resid8(q).reshape(-1,2),axis=1)
print('with label offset: rms',np.sqrt((res**2).mean()),'offset px',q[7], 'params',np.round(q,2))
p=q[:7]; np.save('cam3.npy',q)
# Project every DEM pixel in a generous window and z-buffer
dem=rasterio.open('data/dem_n36e072.tif')
win=rasterio.windows.from_bounds(71.95,36.0,72.45,36.40,dem.transform)
Z=dem.read(1,window=win).astype(float); T=dem.window_transform(win)
rows,cols=np.mgrid[0:Z.shape[0],0:Z.shape[1]]
lon=T.c+(cols+0.5)*T.a; lat=T.f+(rows+0.5)*T.e
x=(lon-lon0)*111320*np.cos(np.radians(lat0)); y=(lat-lat0)*110574
XX=np.c_[x.ravel(),y.ravel(),Z.ravel()]
uv,zc=project(p,XX)
W,H=(16,228,1984,1285),None
inb=(zc>0)&(uv[:,0]>=16)&(uv[:,0]<1984)&(uv[:,1]>=228)&(uv[:,1]<1285)
# z-buffer on 2px screen grid
iu=((uv[:,0]-16)//2).astype(int); iv=((uv[:,1]-228)//2).astype(int)
zbuf=np.full((529,985),np.inf); idx=np.where(inb)[0]
order=idx[np.argsort(zc[idx])]
vis=np.zeros(len(XX),bool)
for k in order:
    a,b=iv[k],iu[k]
    if zc[k]<zbuf[a,b]*1.0+1e-9 and np.isinf(zbuf[a,b]): zbuf[a,b]=zc[k]; vis[k]=True
    elif zc[k]<zbuf[a,b]+150: vis[k]=True
V=vis.reshape(Z.shape)
print('visible DEM pixels',V.sum(),'area km2',V.sum()*30*22/1e6)  # approx pixel size at 36N: 30.9 x 25 m
la,lo=lat[V],lon[V]
print('visible extent lat',la.min(),la.max(),'lon',lo.min(),lo.max())
# near-field (bottom 40% of screen)
fg=V.ravel()&(uv[:,1]>850)
print('foreground (screen y>850) lat',lat.ravel()[fg].min(),lat.ravel()[fg].max(),'lon',lon.ravel()[fg].min(),lon.ravel()[fg].max(),
      'elev',Z.ravel()[fg].min(),Z.ravel()[fg].max())
np.savez('vis.npz',V=V,u=uv[:,0].reshape(Z.shape),v=uv[:,1].reshape(Z.shape),Z=Z,T=np.array(T)[:6])
