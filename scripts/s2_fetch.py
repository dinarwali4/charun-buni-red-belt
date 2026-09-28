import numpy as np, rasterio, pystac_client, concurrent.futures as cf
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import Transformer
ids="20230813 20230910 20230912 20240812 20240924 20250805 20250822 20250901 20250924 20260805 20260909 20250827".split()
c=pystac_client.Client.open('https://earth-search.aws.element84.com/v1')
items=[i for i in c.search(collections=['sentinel-2-l2a'],bbox=[72.0,36.05,72.4,36.38],datetime='2023-08-01/2026-09-20').items()
       if '43SBA' in i.id and i.id.split('_')[2] in ids]
seen=set(); items=[i for i in items if not (i.id.split('_')[2] in seen or seen.add(i.id.split('_')[2]))]
print(len(items),[i.id for i in items])
tr=Transformer.from_crs(4326,32643,always_xy=True)
xs,ys=tr.transform([72.0,72.4,72.0,72.4],[36.05,36.05,36.38,36.38])
B=(min(xs)//20*20,min(ys)//20*20,max(xs)//20*20+20,max(ys)//20*20+20)
W=int((B[2]-B[0])/20);H=int((B[3]-B[1])/20); print('grid',W,H,B)
bands={'blue':'B02','green':'B03','red':'B04','rededge1':'B05','rededge2':'B06','rededge3':'B07','nir':'B08','nir08':'B8A','swir16':'B11','swir22':'B12','scl':'SCL'}
def rd(href,res):
    with rasterio.open(href) as s:
        w=from_bounds(*B,s.transform)
        return s.read(1,window=w,out_shape=(H,W),resampling=res,boundless=True,fill_value=0)
out={}
def job(it):
    arr=[]
    for k,b in bands.items():
        arr.append(rd(it.assets[k].href, Resampling.nearest if k=='scl' else Resampling.average))
    return it.id, np.stack(arr)
with cf.ThreadPoolExecutor(6) as ex:
    for iid,a in ex.map(job,items): out[iid]=a; print('got',iid,flush=True)
np.savez_compressed('data/s2_raw.npz',ids=np.array(list(out)),stack=np.stack(list(out.values())),bounds=np.array(B),bandnames=np.array(list(bands.values())))
# also a 10 m true-colour from the clearest scene for visuals
best=[i for i in items if '20250901' in i.id][0]
with rasterio.open(best.assets['visual'].href) as s:
    w=from_bounds(*B,s.transform); tci=s.read(window=w,out_shape=(3,H*2,W*2))
np.save('data/tci_20250901.npy',tci); print('done')
