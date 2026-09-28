import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
s1=np.load('data/s1_rtc_summer2025.npz'); d=np.load('data/dem20.npz'); Z=d['Z'].astype(float)
L=np.load('data/units.npy'); I=np.load('data/s2_idx.npz'); g=np.load('data/integr.npz')
gy,gx=np.gradient(Z,20)          # gx: d/dEast, gy: d/dRow(south)
nE,nN,nU=-gx,gy,np.ones_like(Z); nn=np.sqrt(nE**2+nN**2+1); nE,nN,nU=nE/nn,nN/nn,nU/nn
geo={'ascending':(-12.0,39.0),'descending':(-168.0,39.0)}     # (heading deg, mid incidence deg); right-looking
out={}; keys=sorted({k.split('_',1)[1] for k in s1.files})
for key in keys:
    orb,state=key.split('_'); hd,inc=geo[state]
    look_az=np.radians(hd+90); th=np.radians(inc)
    # unit vector from ground to satellite
    sE,sN,sU=-np.sin(th)*np.sin(look_az),-np.sin(th)*np.cos(look_az),np.cos(th)
    cosl=nE*sE+nN*sN+nU*sU; lia=np.degrees(np.arccos(np.clip(cosl,-1,1)))
    # slope component toward the sensor in the range plane
    # layover: terrain facing sensor steeper than incidence angle -> range slope angle > inc
    shadow=lia>=88
    rs=np.degrees(np.arctan(-(gx*(-np.sin(look_az))+(-gy)*(-np.cos(look_az)))))  # slope angle facing the sensor (descending toward it)
    layover=rs>inc
    vv=s1[f'vv_{key}']; vh=s1[f'vh_{key}']
    valid=np.isfinite(vv)&np.isfinite(vh)&~shadow&~layover&(Z>100)
    out[key]=dict(shadow=round(float(shadow[Z>100].mean()),3),layover=round(float(layover[Z>100].mean()),3),valid=round(float(valid.mean()),3))
    vvdb=10*np.log10(vv); vhdb=10*np.log10(vh)
    ok=valid&(lia>25)&(lia<55)&I['rock']
    out[key]['units']={int(k):dict(vv_db=round(float(np.nanmedian(vvdb[ok&(L==k)])),2),vh_db=round(float(np.nanmedian(vhdb[ok&(L==k)])),2),n=int((ok&(L==k)).sum())) for k in range(7)}
    out[key]['redbelt_vs_rest_vh']=(round(float(np.nanmedian(vhdb[ok&g['belt']])),2),round(float(np.nanmedian(vhdb[ok&~g['belt']])),2))
    np.save(f'data/sar_valid_{key}.npy',valid)
print(json.dumps(out,indent=1))
json.dump(out,open('sar_stats.json','w'),indent=1)
va=[np.load(f'data/sar_valid_{k}.npy') for k in keys]; anyv=np.any(va,0)
print('valid in >=1 orbit:',round(float(anyv[Z>100].mean()),3))
# figure: VH per best orbits
fig,ax=plt.subplots(1,len(keys),figsize=(6*len(keys),6),dpi=100)
for a,k,v in zip(np.atleast_1d(ax),keys,va):
    x=10*np.log10(s1[f'vh_{k}']); x[~v]=np.nan; im=a.imshow(x[700:1500,700:1600],cmap='gray',vmin=-26,vmax=-10); a.set_title(f'S1 VH γ⁰ (dB), orbit {k}; layover/shadow masked'); a.axis('off')
plt.colorbar(im,ax=ax,fraction=.02); plt.savefig('fig/sar_vh.png',bbox_inches='tight')
