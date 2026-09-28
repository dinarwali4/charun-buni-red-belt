import numpy as np, rasterio, json
from rasterio.transform import rowcol
pts = {  # name: (px, py, lat, lon)
 'Uthol Mulkhow':(1035,268,36.3031,72.1818),'Zani':(1150,264,36.3073,72.1991),'Ghat':(1244,245,36.3204,72.2127),
 'Sehat Payeen':(1367,256,36.3169,72.2321),'Kussum Payeen':(1647,252,36.325,72.276),'Kussum Bala':(1795,279,36.3131,72.2945),
 'Nogram':(1018,297,36.2884,72.1823),'Warijun':(1241,288,36.2972,72.2134),'Muzhgole':(997,340,36.2699,72.1833),
 'Mudare Mulkhow':(831,349,36.2637,72.1629),'Muzhdeh Kosht':(691,383,36.2497,72.1502),'Kosht':(666,416,36.2397,72.1517),
 'Bumbagh':(842,439,36.2363,72.1734),'Shinjuran':(414,452,36.2264,72.1295),'Sarwarabad':(597,468,36.2255,72.1507),
 'Kuragh':(666,498,36.2196,72.1609),'Charun':(934,481,36.2273,72.1861),'Charun Oweer':(1173,499,36.2265,72.2106),
 'Junali Kuch':(1237,390,36.2567,72.2151),'Gahli':(1543,333,36.2826,72.254),'Buni':(1629,370,36.2701,72.2625),
 'Pachili Awi Lusht':(1915,347,36.2839,72.3005),'Sholiri Gohkir':(73,519,36.2073,72.1067),'Khilani':(168,509,36.2105,72.114),
}
dem = rasterio.open('data/dem_n36e072.tif'); Z = dem.read(1)
lat0, lon0 = 36.22, 72.2
def enu(lat, lon):
    return (lon-lon0)*111320*np.cos(np.radians(lat0)), (lat-lat0)*110574
names=list(pts); X=[];U=[]
for n in names:
    px,py,la,lo=pts[n]; r,c=rowcol(dem.transform,lo,la); z=float(Z[r,c])
    x,y=enu(la,lo); X.append((x,y,z)); U.append((px,py))
X=np.array(X);U=np.array(U,float)
def dlt(X,U):
    A=[]
    for (x,y,z),(u,v) in zip(X,U):
        A.append([x,y,z,1,0,0,0,0,-u*x,-u*y,-u*z,-u]); A.append([0,0,0,0,x,y,z,1,-v*x,-v*y,-v*z,-v])
    _,_,Vt=np.linalg.svd(np.array(A)); return Vt[-1].reshape(3,4)
def proj(P,X):
    h=np.c_[X,np.ones(len(X))]@P.T; return h[:,:2]/h[:,2:]
# normalise for conditioning
m=X.mean(0); s=X.std(0); Xn=(X-m)/s
P=dlt(Xn,U); res=np.linalg.norm(proj(P,Xn)-U,axis=1)
for n,r in sorted(zip(names,res),key=lambda t:-t[1])[:6]: print(f'{n:20s} {r:6.1f}px')
print('median residual',np.median(res),'rms',np.sqrt((res**2).mean()))
# drop worst outliers (>3x median) and refit
keep=res<max(3*np.median(res),25); P=dlt(Xn[keep],U[keep]); res2=np.linalg.norm(proj(P,Xn)-U,axis=1)
print('kept',keep.sum(),'median',np.median(res2[keep]),'rms',np.sqrt((res2[keep]**2).mean()))
# camera centre
Pd = P.copy(); M=Pd[:,:3]; C=-np.linalg.solve(M,Pd[:,3]); Cw=C*s+m
print('camera ENU (m):',Cw, ' -> lat',lat0+Cw[1]/110574,'lon',lon0+Cw[0]/(111320*np.cos(np.radians(lat0))))
np.savez('cam.npz',P=P,m=m,s=s,lat0=lat0,lon0=lon0)
