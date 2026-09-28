import numpy as np, rasterio, json
exec(open('camfit2.py').read().split('best=None')[0])
p=np.load('cam2.npy')
dem=rasterio.open('data/dem_n36e072.tif')
win=rasterio.windows.from_bounds(71.95,36.0,72.45,36.40,dem.transform)
Z=dem.read(1,window=win).astype(float); T=dem.window_transform(win)
rows,cols=np.mgrid[0:Z.shape[0],0:Z.shape[1]]
lon=T.c+(cols+0.5)*T.a; lat=T.f+(rows+0.5)*T.e
x=(lon-lon0)*111320*np.cos(np.radians(lat0)); y=(lat-lat0)*110574
uv,zc=project(p,np.c_[x.ravel(),y.ravel(),Z.ravel()])
inb=(zc>0)&(uv[:,0]>=16)&(uv[:,0]<1984)&(uv[:,1]>=228)&(uv[:,1]<1285)
cell=3
iu=((uv[:,0]-16)//cell).astype(int); iv=((uv[:,1]-228)//cell).astype(int)
nh,nw=1057//cell+1,1968//cell+1
zb=np.full((nh,nw),np.inf); k=np.where(inb)[0]
np.minimum.at(zb,(iv[k],iu[k]),zc[k])
vis=np.zeros(zc.shape,bool); vis[k]=zc[k]<=zb[iv[k],iu[k]]*1.03
V=vis.reshape(Z.shape)
print('visible px',V.sum(),'lat',lat[V].min(),lat[V].max(),'lon',lon[V].min(),lon[V].max())
cover=np.isfinite(zb).mean(); print('screen cells covered',cover)
fg=vis&(uv[:,1]>900)
print('foreground lat',lat.ravel()[fg].min(),lat.ravel()[fg].max(),'lon',lon.ravel()[fg].min(),lon.ravel()[fg].max(),'elev',Z.ravel()[fg].min(),Z.ravel()[fg].max())
# screen corners -> ground: nearest visible pixel to each corner
for nm,(cu,cvv) in {'bottom-left':(20,1280),'bottom-centre':(1000,1280),'bottom-right':(1980,1280),'mid-left':(20,750),'mid-right':(1980,750),'centre':(1000,756)}.items():
    d=(uv[:,0]-cu)**2+(uv[:,1]-cvv)**2; d[~vis]=np.inf; i=np.argmin(d)
    print(f'{nm:14s} {lat.ravel()[i]:.4f}N {lon.ravel()[i]:.4f}E  z={Z.ravel()[i]:.0f} m')
np.savez('vis.npz',V=V,u=uv[:,0].reshape(Z.shape),v=uv[:,1].reshape(Z.shape),Z=Z,T=np.array(list(T)[:6]))
