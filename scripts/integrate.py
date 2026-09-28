import numpy as np, json
from scipy import ndimage as ndi
from pyproj import Transformer
d=np.load('data/dem20.npz'); Z=d['Z']; T=d['T']; H,W=Z.shape
I=np.load('data/s2_idx.npz'); rock=I['rock']; L=np.load('data/units.npy')
tir=np.load('data/aster_tir.npz'); sw=np.load('data/aster_swir.npz'); ep=np.load('data/emit_persist.npz'); dd=np.load('data/dem_derivs.npz')
r,c=np.mgrid[0:H,0:W]; lon,lat=Transformer.from_crs(32643,4326,always_xy=True).transform(T[2]+(c+.5)*20,T[5]-(r+.5)*20)
x0,y0,res=ep['grid']; ei=((y0-lat)/res).astype(int); ej=((lon-x0)/res).astype(int)
inE=(ei>=0)&(ej>=0)&(ei<ep['carbonate'].shape[0])&(ej<ep['carbonate'].shape[1])
def emitlayer(f): o=np.zeros((H,W),'uint8'); o[inE]=ep[f][ei[inE],ej[inE]]; return o
E={f:emitlayer(f) for f in ['carbonate','ferric','propylitic','whitemica_clay','sulphate','cu_carbonate','nonmineral']}
aoi=inE
# --- unit interpretation
units=json.load(open('units.json'))
for u in units:
    m=(L==u['unit'])&aoi
    u.update(QI=round(float(np.nanmedian(tir['QI'][m])),4),CI=round(float(np.nanmedian(tir['CI'][m])),4),MI=round(float(np.nanmedian(tir['MI'][m])),4),
             AlOH=round(float(np.nanmedian(sw['AlOH'][m])),4),MgOH_carb=round(float(np.nanmedian(sw['MgOH_carb'][m])),4),
             **{'emit_'+f:round(float((E[f][m]>=3).mean()),3) for f in ['carbonate','ferric','propylitic','whitemica_clay']})
    print({k:u[k] for k in ('unit','frac','rgb','redness','QI','CI','MgOH_carb','AlOH','emit_carbonate','emit_ferric','emit_propylitic','emit_whitemica_clay','elev_med')})
json.dump(units,open('units.json','w'),indent=1)
# --- Reshun red-bed belt mask and its margins (USGS: upper & lower limits are thrusts)
red=np.where(rock,I['redness'],0); belt=ndi.binary_closing(ndi.uniform_filter(red,9)>0.13,iterations=4)
lab,n=ndi.label(belt); sz=ndi.sum(belt,lab,range(1,n+1)); keep=np.isin(lab,1+np.where(sz>2000)[0]); belt=keep
edge=belt^ndi.binary_erosion(belt,iterations=2)
dist_margin=ndi.distance_transform_edt(~edge)*20
print('red-belt area km2',round(belt.sum()*400/1e6,1))
# --- lithology-normalised anomalies
def rz(v,m):
    med=np.nanmedian(v[m]); mad=1.4826*np.nanmedian(np.abs(v[m]-med)); return (v-med)/mad
Zs={}
for nm,v in (('ferric',I['ferric_B4B2']),('gossan',I['gossan_B11B4']),('alOH',I['alOH_carb_B11B12']),('fe900',1/I['fe3_900nm_B8AB7'])):
    z=np.full((H,W),np.nan)
    for k in range(7):
        m=(L==k)&rock; z[m]=rz(v,m)[m]
    Zs[nm]=z
# positive-control / dynamic range: raw ferric z across ALL rock (red belt should be strongly positive)
zraw=rz(I['ferric_B4B2'],rock)
print('dynamic range check: raw ferric z  in red belt p50=%.1f  outside p50=%.1f'%(np.nanmedian(zraw[belt&rock]),np.nanmedian(zraw[~belt&rock])))
score=(np.nan_to_num(Zs['ferric'])>2.5).astype(int)+(np.nan_to_num(Zs['gossan'])>2.5)+(np.nan_to_num(Zs['alOH'])>2.5)+(np.nan_to_num(Zs['fe900'])>2.5)
cand=(score>=2)&rock&aoi
cand=ndi.binary_opening(cand,iterations=1)
lab,n=ndi.label(cand,structure=np.ones((3,3))); print('S2 multi-index anomaly clusters',n)
tg=[]
for k in range(1,n+1):
    m=lab==k; npx=m.sum()
    if npx<6: continue
    rr,cc=np.where(m); i=np.argmax(score[m]+0*rr)
    la=float(lat[m].mean()); lo=float(lon[m].mean())
    ev=dict(id=k,lat=round(la,5),lon=round(lo,5),area_ha=round(npx*0.04,2),elev=round(float(Z[m].mean())),slope=round(float(dd['slope'][m].mean()),1),
       unit=int(np.bincount(L[m][L[m]>=0]).argmax()),z_ferric=round(float(np.nanmean(Zs['ferric'][m])),1),z_gossan=round(float(np.nanmean(Zs['gossan'][m])),1),
       z_alOH=round(float(np.nanmean(Zs['alOH'][m])),1),z_fe900=round(float(np.nanmean(Zs['fe900'][m])),1),
       dist_redbelt_margin_m=round(float(dist_margin[m].min())),in_redbelt=bool(belt[m].mean()>0.5),
       emit_sulphate_max=int(E['sulphate'][m].max()),emit_cu_max=int(E['cu_carbonate'][m].max()),emit_whitemica_max=int(E['whitemica_clay'][m].max()),
       emit_nonmineral_max=int(E['nonmineral'][m].max()),
       aster_AlOH=round(float(np.nanmean(sw['AlOH'][m])),3),aster_QI=round(float(np.nanmean(tir['QI'][m])),4),
       structural_density=round(float(dd['m'][max(rr.min()-25,0):rr.max()+25,max(cc.min()-25,0):cc.max()+25].mean()),3))
    tg.append(ev)
# evidence score: spectral anomaly (base) + EMIT sulphate/Cu persistence + near thrust margin (<500 m) + EMIT white mica
for t in tg:
    t['evidence']=int(t['emit_sulphate_max']>=2)+int(t['emit_cu_max']>=2)+int(t['dist_redbelt_margin_m']<500)+int(t['emit_whitemica_max']>=3)+int(t['area_ha']>=1)
    t['red_flag_nonmineral']=t['emit_nonmineral_max']>=2
tg=sorted(tg,key=lambda t:(-t['evidence'],-t['area_ha']))
print('clusters >=6px:',len(tg)); 
for t in tg[:15]: print(t)
json.dump(tg,open('targets.json','w'),indent=1)
np.savez_compressed('data/integr.npz',belt=belt,edge=edge,score=score.astype('uint8'),cand=cand,**{'z_'+k:v.astype('float32') for k,v in Zs.items()},**{'E_'+k:v for k,v in E.items()})
# EMIT sulphate cluster check against S2 at 36.209N 72.270E
sel=(np.abs(lat-36.209)<0.003)&(np.abs(lon-72.270)<0.004)
print('EMIT sulphate cluster zone: rock frac %.2f, ndsi p50 %.2f, units %s, z_ferric p50 %.1f, z_alOH p50 %.1f, redbelt frac %.2f, dist margin min %d m'%(
  rock[sel].mean(),np.nanmedian(I['ndsi'][sel]),np.bincount(L[sel][L[sel]>=0],minlength=7),np.nanmedian(Zs['ferric'][sel]),np.nanmedian(Zs['alOH'][sel]),belt[sel].mean(),dist_margin[sel].min()))
