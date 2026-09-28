import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from pyproj import Transformer
d=np.load('data/dem20.npz'); T=d['T']; Z=d['Z']; dd=np.load('data/dem_derivs.npz'); hs=dd['hs']
I=np.load('data/s2_idx.npz'); L=np.load('data/units.npy'); g=np.load('data/integr.npz'); tir=np.load('data/aster_tir.npz'); sw=np.load('data/aster_swir.npz')
cmp=np.load('data/s2_comp.npz')['comp']
fw=Transformer.from_crs(4326,32643,always_xy=True)
def rc(lon,lat): x,y=fw.transform(lon,lat); return (T[5]-y)/20,(x-T[2])/20
A=(72.06,36.11,72.33,36.33); (r0,c0),(r1,c1)=rc(A[0],A[3]),rc(A[2],A[1]); r0,r1,c0,c1=int(r0),int(r1),int(c0),int(c1)
S=lambda a: a[r0:r1,c0:c1]
places=json.load(open('places_sel.json'))
def deco(a,labels=True):
    if labels:
        for n,(la,lo) in places.items():
            r,c=rc(lo,la); r-=r0; c-=c0
            if 0<r<r1-r0 and 0<c<c1-c0:
                a.plot(c,r,'o',ms=3,color='yellow',mec='k',mew=.5); a.text(c+8,r,n,color='white',fontsize=7,path_effects=[pe.withStroke(linewidth=2,foreground='k')])
    for lo in np.arange(72.10,72.33,0.05):
        r,c=rc(lo,36.22); a.axvline(c-c0,color='w',lw=.3,alpha=.4); a.text(c-c0+3,r1-r0-8,f'{lo:.2f}°E',color='w',fontsize=6.5,path_effects=[pe.withStroke(linewidth=1.5,foreground='k')])
    for la in np.arange(36.15,36.33,0.05):
        r,c=rc(72.2,la); a.axhline(r-r0,color='w',lw=.3,alpha=.4); a.text(4,r-r0-4,f'{la:.2f}°N',color='w',fontsize=6.5,path_effects=[pe.withStroke(linewidth=1.5,foreground='k')])
    a.set_xticks([]); a.set_yticks([])
H_=S(hs)
# 1 units
units=json.load(open('units.json'))
names={1:'U1 hematite-red siltstone (Reshun Fm core)',6:'U6 red-brown clastics, terraces & red-bed debris',4:'U4 pale ferruginous carbonate',2:'U2 bright pale rock (carbonate-dominant; incl. high scree)',
       0:'U0 brown slope debris / mixed',5:'U5 dark ferruginous high ground (Buni Zom flank)',3:'U3 grey shale/slate (Sarikol Shale N of river; Chitral Slate S)'}
cols={1:'#b3261e',6:'#e07a4a',4:'#e8c39e',2:'#f4f1e8',0:'#9c8263',5:'#5d4a3a',3:'#6b7b8c'}
Lc=S(L).astype(float); Lc[Lc<0]=np.nan
cm=ListedColormap([cols[k] for k in range(7)])
fig,ax=plt.subplots(figsize=(11,9),dpi=120); ax.imshow(H_,cmap='gray'); ax.imshow(Lc,cmap=cm,vmin=-.5,vmax=6.5,alpha=.78,interpolation='nearest'); deco(ax)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=cols[k],label=names[k]+f"  ({units[k]['frac']*100:.0f}%)") for k in (1,6,4,2,0,5,3)],loc='lower right',fontsize=7,framealpha=.9)
ax.set_title('Spectral colour units (k-means on 7 illumination-robust Sentinel-2 features, exposed rock only)'); plt.savefig('fig/units_map.png',bbox_inches='tight'); plt.close()
# 2 EMIT persistence
fig,axs=plt.subplots(2,2,figsize=(14,11),dpi=110)
for a,(f,t) in zip(axs.ravel(),[('ferric','Ferric iron (hematite, goethite)'),('carbonate','Carbonate (calcite, dolomite)'),('propylitic','Chlorite / epidote / prehnite'),('whitemica_clay','White mica / illite / clay')]):
    a.imshow(H_,cmap='gray'); v=S(g['E_'+f]).astype(float); v[v==0]=np.nan
    im=a.imshow(v,cmap='viridis',vmin=1,vmax=5,alpha=.8,interpolation='nearest'); a.set_title(f'EMIT: {t} — dates identified (of 5)',fontsize=10); deco(a,labels=False)
plt.colorbar(im,ax=axs,fraction=.02); plt.savefig('fig/emit_persist.png',bbox_inches='tight'); plt.close()
# 3 ASTER TIR
fig,axs=plt.subplots(1,2,figsize=(16,7),dpi=110)
for a,(k,t,cmap_,lim) in zip(axs,[('QI','ASTER TIR Quartz Index (B11²/B10·B12)','magma',(1.0,1.06)),('CI','ASTER TIR Carbonate Index (B13/B14)','viridis',(0.985,1.015))]):
    a.imshow(H_,cmap='gray'); v=S(tir[k]); v=np.where(S(I['rock']),v,np.nan); im=a.imshow(v,cmap=cmap_,vmin=lim[0],vmax=lim[1],alpha=.85); a.set_title(t); deco(a,labels=False); plt.colorbar(im,ax=a,fraction=.035)
plt.tight_layout(); plt.savefig('fig/aster_tir.png',bbox_inches='tight'); plt.close()
# 4 targets & sampling map
sites=json.load(open('stream_sites.json'))
ded=[]
for s_ in sorted(sites,key=lambda s:-s['catchment_km2']):
    if all(abs(s_['lat']-q['lat'])>0.006 or abs(s_['lon']-q['lon'])>0.006 for q in ded): ded.append(s_)
ded=[s_ for s_ in ded if s_['catchment_km2']>=1.5][:14]
for i,s_ in enumerate(ded): s_['id']=f'SS{i+1:02d}'
json.dump(ded,open('stream_sites_final.json','w'),indent=1)
rgb=np.stack([cmp[2],cmp[1],cmp[0]],-1); rgb=np.nan_to_num(np.clip(S(rgb)/0.32,0,1)**(1/1.5))
fig,ax=plt.subplots(figsize=(12,9.6),dpi=120); ax.imshow(rgb*(0.55+0.45*H_[...,None]))
ax.contour(S(g['belt']).astype(float),levels=[.5],colors=['#ff3b30'],linewidths=1.4)
ax.contour(S(np.load('data/drainage40.npz')['A']).repeat(2,0).repeat(2,1)[:r1-r0,:c1-c0]>=1.0,levels=[.5],colors=['#4fc3f7'],linewidths=.4)
for s_ in ded:
    r,c=rc(s_['lon'],s_['lat']); ax.plot(c-c0,r-r0,'v',ms=8,color='#00e5ff',mec='k'); ax.text(c-c0+8,r-r0+10,s_['id'],color='#00e5ff',fontsize=7.5,weight='bold',path_effects=[pe.withStroke(linewidth=2,foreground='k')])
r,c=rc(72.270,36.209); ax.plot(c-c0,r-r0,'*',ms=16,color='#ffd60a',mec='k'); ax.text(c-c0+12,r-r0-10,'E1 EMIT jarosite/copiapite\n(4 of 5 dates; S2 does not corroborate)',color='#ffd60a',fontsize=7.5,path_effects=[pe.withStroke(linewidth=2,foreground='k')])
r,c=rc(72.289,36.262); ax.plot(c-c0,r-r0,'P',ms=12,color='#34c759',mec='k'); ax.text(c-c0+12,r-r0,'E2 EMIT "azurite" scatter\n(unconfirmed library match)',color='#34c759',fontsize=7.5,path_effects=[pe.withStroke(linewidth=2,foreground='k')])
deco(ax); ax.set_title('Where to look: red-bed belt outlined from Sentinel-2 (≈ Reshun Fm; USGS 1975: its margins are thrusts), streams ≥1 km² (blue),\nstream-sediment sites SS01–SS14 at stream–margin crossings, EMIT leads E1–E2',fontsize=10)
plt.savefig('fig/targets_map.png',bbox_inches='tight'); plt.close()
# 5 elevation artefact chart
bands=['1.8–3.0','3.0–4.0','4.0–4.5','4.5–5.0','5.0–6.5']; before=[0.01,0.00,0.13,0.97,0.90]; after=[0.011,0.005,0.034,0.0,0.005]
fig,ax=plt.subplots(figsize=(8,3.8),dpi=120); x=np.arange(5); ax.bar(x-.2,before,.4,label='normalised by colour unit only',color='#e07a4a'); ax.bar(x+.2,after,.4,label='normalised by unit × elevation band (thr 3.0)',color='#2f6690')
ax.set_xticks(x); ax.set_xticklabels([b+' km' for b in bands]); ax.set_ylabel('% of rock pixels flagged'); ax.legend(frameon=False,fontsize=8); ax.spines[['top','right']].set_visible(False)
ax.set_title('The "anomalies" were an altitude effect',fontsize=10); plt.tight_layout(); plt.savefig('fig/anomaly_elevation.png',bbox_inches='tight'); plt.close()
print('ok',len(ded)); [print(s_) for s_ in ded]
