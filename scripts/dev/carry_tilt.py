import csv, sys, numpy as np
r=list(csv.DictReader(open(sys.argv[1])))
g=lambda k:np.array([float(x[k]) if x[k] not in ("",) else np.nan for x in r])
t,tilt,yaw,hold,z=g("t"),g("tilt"),g("yaw"),g("holding"),g("ee_z")
m=hold==1
print("while holding a box: tilt deg median %.1f p95 %.1f max %.1f | yaw range %.0f..%.0f deg"%(np.median(tilt[m]),np.percentile(tilt[m],95),tilt[m].max(),yaw[m].min(),yaw[m].max()))
# per-carry yaw change
runs=[];k=0
while k<len(t):
    if m[k]:
        j=k
        while j<len(t) and m[j]: j+=1
        yy=np.unwrap(np.radians(yaw[k:j])); runs.append((t[k],tilt[k:j].max(),np.degrees(yy.max()-yy.min())));k=j
    else: k+=1
for a,b,c in runs: print(" carry from t=%.0f: max tilt %.1f deg, yaw swing %.0f deg"%(a,b,c))
