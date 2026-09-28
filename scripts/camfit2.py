import numpy as np
from scipy.optimize import least_squares
exec(open('camfit.py').read().split('def dlt')[0])   # reuse pts, X (ENU+z), U
cx,cy=1000,756.5
def R(yaw,pitch,roll):
    # camera looks along heading yaw (deg from north, clockwise), tilted down by pitch from horizontal
    y,p,r=np.radians([yaw,pitch,roll])
    f=np.array([np.sin(y)*np.cos(p),np.cos(y)*np.cos(p),-np.sin(p)])   # forward
    rt=np.array([np.cos(y),-np.sin(y),0.])                               # right
    up=np.cross(rt,f)
    rt2=np.cos(r)*rt+np.sin(r)*up; up2=np.cos(r)*up-np.sin(r)*rt
    return rt2,up2,f
def project(p,X):
    x,y,z,yaw,pitch,roll,fl=p
    rt,up,f=R(yaw,pitch,roll); d=X-np.array([x,y,z])
    zc=d@f; return np.c_[cx+fl*(d@rt)/zc, cy-fl*(d@up)/zc], zc
def resid(p): return (project(p,X)[0]-U).ravel()
best=None
for yaw0 in [330,345,0,15]:
  for dist in [15e3,30e3]:
    for pit in [20,35,50]:
      yr=np.radians(yaw0); tx,ty=0,4000
      p0=[tx-dist*np.sin(yr)*np.cos(np.radians(pit)), ty-dist*np.cos(yr)*np.cos(np.radians(pit)), 2500+dist*np.sin(np.radians(pit)), yaw0,pit,0,1585]
      r=least_squares(resid,p0,x_scale=[1e3,1e3,1e3,5,5,2,100])
      if best is None or r.cost<best.cost: best=r
p=best.x; res=np.linalg.norm(resid(p).reshape(-1,2),axis=1)
print('params x,y,z,yaw,pitch,roll,f =',np.round(p,2))
print('rms px',np.sqrt((res**2).mean()),'max',res.max(), max(zip(res,names)))
print('camera lat',lat0+p[1]/110574,'lon',lon0+p[0]/(111320*np.cos(np.radians(lat0))),'alt',p[2])
print('vertical fov deg', 2*np.degrees(np.arctan(528.5/p[6])))
np.save('cam2.npy',p)
