import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from scipy import ndimage as ndi
tr=Transformer.from_crs(4326,32643,always_xy=True)
xs,ys=tr.transform([72.0,72.4,72.0,72.4],[36.05,36.05,36.38,36.38])
B=(min(xs)//20*20,min(ys)//20*20,max(xs)//20*20+20,max(ys)//20*20+20)
W=int((B[2]-B[0])/20);H=int((B[3]-B[1])/20)
T=rasterio.transform.from_origin(B[0],B[3],20,20)
src=rasterio.open('data/dem_n36e072.tif')
Z=np.zeros((H,W),'float32')
reproject(rasterio.band(src,1),Z,dst_transform=T,dst_crs='EPSG:32643',resampling=Resampling.bilinear)
np.savez_compressed('data/dem20.npz',Z=Z,T=np.array(list(T)[:6]),B=np.array(B))
gy,gx=np.gradient(Z,20); slope=np.degrees(np.arctan(np.hypot(gx,gy)))
aspect=(np.degrees(np.arctan2(-gx,gy))+360)%360  # direction of downslope? (gx east, gy north... rows go south)
print('grid',W,H,'elev',Z.min(),Z.max(),'mean slope',slope.mean())
np.save('data/slope20.npy',slope.astype('float32'))
