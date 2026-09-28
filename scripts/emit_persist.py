import json,re,numpy as np,collections
px=json.load(open('emit_pixels.json'))
fam={'carbonate':r'calcite|dolomite|aragonite|siderite|strontian|magnesite|ankerite','ferric':r'hematite|goethite|fe-hydrox|ferrihyd|lepidocro',
 'propylitic':r'chlorit|epidot|prehnite|clinochlore|actinol','whitemica_clay':r'muscov|illite|kaolin|montmor|beidell|sericit|pyrophyl|dickit',
 'sulphate':r'jarosit|copiap|alunit|gypsum','cu_carbonate':r'azurite|malachite|chrysocol','nonmineral':r'grass|plastic|tarp|plywood|paint|gas'}
A=(72.05,36.11,72.32,36.34); res=0.0006
nx=int((A[2]-A[0])/res)+1; ny=int((A[3]-A[1])/res)+1
dates=sorted({p[0] for p in px})
pres={f:np.zeros((len(dates),ny,nx),bool) for f in fam}
for d,g,la,lo,n,bd in px:
    i=int((A[3]-la)/res); j=int((lo-A[0])/res); k=dates.index(d)
    for f,rx in fam.items():
        if re.search(rx,n,re.I): pres[f][k,i,j]=True
out={}
for f in fam:
    cnt=pres[f].sum(0)
    hist=[int((cnt==k).sum()) for k in range(len(dates)+1)]
    # expected >=3-date cells under independence (per-date rate p_k)
    p=pres[f].reshape(len(dates),-1).mean(1)
    # Poisson-binomial prob of >=3 of 5
    from itertools import combinations
    probs=np.zeros(len(dates)+1); probs[0]=1
    for pk in p: probs=np.r_[probs[:1]*(1-pk), probs[1:]*(1-pk)+probs[:-1]*pk]
    exp3=probs[3:].sum()*cnt.size
    out[f]=dict(per_date_rate=[round(float(x),4) for x in p],hist=hist,obs_ge3=int((cnt>=3).sum()),exp_ge3_if_random=round(float(exp3),1))
    print(f,out[f])
np.savez_compressed('data/emit_persist.npz',**{f:pres[f].sum(0).astype('uint8') for f in fam},dates=np.array(dates),grid=np.array([A[0],A[3],res]))
json.dump(out,open('emit_persist.json','w'),indent=1)
