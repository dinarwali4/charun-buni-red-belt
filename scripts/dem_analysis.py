import numpy as np, json, heapq, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from pyproj import Transformer
d=np.load('data/dem20.npz'); Z=d['Z'].astype('float64'); T=d['T']; H,W=Z.shape
valid=Z>100
x0,px,_,_,y0=T[2],T[0],T[1],T[3],T[5]
tr=Transformer.from_crs(32643,4326,always_xy=True)
def ll(r,c): return tr.transform(T[2]+(c+.5)*20, T[5]-(r+.5)*20)
def rc(lon,lat):
    x,y=Transformer.from_crs(4326,32643,always_xy=True).transform(lon,lat); return int((T[5]-y)//20), int((x-T[2])//20)
Zf=np.where(valid,Z,np.nan)
gy,gx=np.gradient(np.where(valid,Z,np.nanmean(Zf)),20)
slope=np.degrees(np.arctan(np.hypot(gx,gy)))
# hillshade az 315 alt 45 (rows increase southward so north-gradient = -gy)
az,alt=np.radians(315),np.radians(45)
sl=np.arctan(np.hypot(gx,gy)); asp=np.arctan2(-gx,gy)   # aspect with rows southward
hs=np.sin(alt)*np.cos(sl)+np.cos(alt)*np.sin(sl)*np.cos(az-asp)
hs=np.clip(hs,0,1)
relief=ndi.maximum_filter(Z,size=51)-ndi.minimum_filter(Z,size=51)   # ~1 km window
tpi=Z-ndi.uniform_filter(Z,size=25)
# AOIs
aois={'screenshot_footprint':(72.05,36.11,72.32,36.34),'south_bank_core':(72.12,36.13,72.30,36.235)}
stats={}
for k,(a,b,c,e) in aois.items():
    r1,c1=rc(a,e); r2,c2=rc(c,b); r1,r2=sorted((r1,r2)); c1,c2=sorted((c1,c2))
    z=Z[r1:r2,c1:c2]; s=slope[r1:r2,c1:c2]; v=z>100; z=z[v]; s=s[v]
    stats[k]=dict(bbox=[a,b,c,e],area_km2=round(v.sum()*400/1e6,1),elev_min=round(float(z.min())),elev_max=round(float(z.max())),
        elev_mean=round(float(z.mean())),relief=round(float(z.max()-z.min())),slope_mean=round(float(s.mean()),1),
        slope_p90=round(float(np.percentile(s,90)),1),frac_gt35=round(float((s>35).mean()),3),frac_gt45=round(float((s>45).mean()),3),
        hypsometric_integral=round(float((z.mean()-z.min())/(z.max()-z.min())),3),
        elev_pct={p:round(float(np.percentile(z,p))) for p in (5,25,50,75,95)})
print(json.dumps(stats,indent=1))
# ---- Lineaments: Hessian eigen-analysis on smoothed DEM
Zs=ndi.gaussian_filter(np.where(valid,Z,np.nanmean(Zf)),3)
hxx=ndi.gaussian_filter(Zs,1,order=(0,2)); hyy=ndi.gaussian_filter(Zs,1,order=(2,0)); hxy=ndi.gaussian_filter(Zs,1,order=(1,1))
tmp=np.sqrt(((hxx-hyy)/2)**2+hxy**2); l1=(hxx+hyy)/2+tmp; l2=(hxx+hyy)/2-tmp
big=np.where(np.abs(l1)>np.abs(l2),l1,l2); small=np.where(np.abs(l1)>np.abs(l2),l2,l1)
# eigenvector of the SMALL-|curvature| eigenvalue = along-feature direction
# for matrix [[hxx,hxy],[hxy,hyy]] (x=col east, y=row south): v=(hxy, small-hxx)
vx=hxy; vy=small-hxx
theta=(np.degrees(np.arctan2(vx,-vy))+180)%180   # azimuth: east comp=vx, north comp=-vy
strength=np.abs(big)
m=valid&(strength>np.percentile(strength[valid],90))
ridge=m&(big<0); valley=m&(big>0)
def rose(mask,wts):
    h,_=np.histogram(theta[mask],bins=18,range=(0,180),weights=wts[mask]); return h/h.sum()
r1c1=rc(72.12,36.235); r2c2=rc(72.30,36.13)
core=np.zeros_like(valid); core[r1c1[0]:r2c2[0],r1c1[1]:r2c2[1]]=True
rose_all=rose(m,strength); rose_core=rose(m&core,strength); rose_ridge=rose(ridge&core,strength); rose_valley=rose(valley&core,strength)
bins=np.arange(5,180,10)
print('core lineament rose peak bins', [ (int(bins[i]),round(float(rose_core[i]),3)) for i in np.argsort(rose_core)[::-1][:4]])
print('core ridge peak',[ (int(bins[i]),round(float(rose_ridge[i]),3)) for i in np.argsort(rose_ridge)[::-1][:3]])
print('core valley peak',[ (int(bins[i]),round(float(rose_valley[i]),3)) for i in np.argsort(rose_valley)[::-1][:3]])
ne=(theta>=30)&(theta<90); nnw=(theta>=150)|(theta<10)
stats['lineaments']=dict(core_frac_NE_030_090=round(float(strength[m&core&ne].sum()/strength[m&core].sum()),3),
    core_frac_NNW_150_010=round(float(strength[m&core&nnw].sum()/strength[m&core].sum()),3),
    rose_core=[round(float(v),4) for v in rose_core], rose_ridge=[round(float(v),4) for v in rose_ridge], rose_valley=[round(float(v),4) for v in rose_valley])
# sanity control: orientation of Mastuj river valley near Charun->Buni should be ~NE (~55-65 deg)
a=rc(72.1609,36.2196); b=rc(72.2625,36.2701)
import math
dx=(b[1]-a[1])*20; dy=(a[0]-b[0])*20; stats['control_river_azimuth_Kuragh_to_Buni']=round(math.degrees(math.atan2(dx,dy))%180,1)
print('control: Kuragh->Buni azimuth',stats['control_river_azimuth_Kuragh_to_Buni'])
json.dump(stats,open('dem_stats.json','w'),indent=1)
np.savez_compressed('data/dem_derivs.npz',slope=slope.astype('float32'),hs=hs.astype('float32'),relief=relief.astype('float32'),theta=theta.astype('float32'),m=m,ridge=ridge,valley=valley)
# ---- figures
ext=None
fig,ax=plt.subplots(1,3,figsize=(18,6.4),dpi=110)
ax[0].imshow(hs,cmap='gray'); im=ax[0].imshow(np.where(valid,Z,np.nan),cmap='terrain',alpha=.55); ax[0].set_title('Elevation over hillshade (m)'); plt.colorbar(im,ax=ax[0],fraction=.046)
im=ax[1].imshow(slope,cmap='magma',vmin=0,vmax=60); ax[1].set_title('Slope (°)'); plt.colorbar(im,ax=ax[1],fraction=.046)
im=ax[2].imshow(relief,cmap='viridis'); ax[2].set_title('Local relief, 1 km window (m)'); plt.colorbar(im,ax=ax[2],fraction=.046)
for a_ in ax: a_.axis('off')
plt.tight_layout(); plt.savefig('fig/dem_panels.png',bbox_inches='tight'); plt.close()
fig=plt.figure(figsize=(12,4.2),dpi=110)
for i,(nm,rs) in enumerate([('All strong curvature (core)',rose_core),('Ridges',rose_ridge),('Valleys',rose_valley)]):
    a_=fig.add_subplot(1,3,i+1,projection='polar'); a_.set_theta_zero_location('N'); a_.set_theta_direction(-1)
    th=np.radians(np.r_[bins,bins+180]); a_.bar(th,np.r_[rs,rs],width=np.radians(10),color='#c0563a',edgecolor='white'); a_.set_title(nm); a_.set_yticklabels([])
plt.tight_layout(); plt.savefig('fig/lineament_rose.png',bbox_inches='tight'); plt.close()
print('figs ok')
