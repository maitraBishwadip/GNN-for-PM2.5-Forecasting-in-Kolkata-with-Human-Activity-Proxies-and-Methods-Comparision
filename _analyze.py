import csv, math, statistics as st
from collections import defaultdict, OrderedDict

F = r"D:\d drive\GNN_PM2.5_Kolkata\Kolkata_sat_ground.csv"

rows = []
with open(F, newline='') as fh:
    r = csv.DictReader(fh)
    cols = r.fieldnames
    for d in r:
        rows.append(d)

print("COLUMNS:", cols)
print("TOTAL ROWS:", len(rows))

# station coords
coords = OrderedDict()
for d in rows:
    s = d['Station']
    if s not in coords:
        coords[s] = (float(d['Latitude']), float(d['Longitude']))
print("\nSTATIONS & COORDS:")
for s,(la,lo) in coords.items():
    print(f"  {s:20s} lat={la:.5f} lon={lo:.5f}")

# pairwise distances (km) haversine
def hav(a,b):
    R=6371.0
    la1,lo1=map(math.radians,a); la2,lo2=map(math.radians,b)
    dla=la2-la1; dlo=lo2-lo1
    h=math.sin(dla/2)**2+math.cos(la1)*math.cos(la2)*math.sin(dlo/2)**2
    return 2*R*math.asin(math.sqrt(h))
ss=list(coords)
print("\nPAIRWISE DISTANCE (km):")
print("        "+ " ".join(f"{x[:6]:>7s}" for x in ss))
maxd=0
for a in ss:
    line=f"{a[:7]:7s} "
    for b in ss:
        dd=hav(coords[a],coords[b]); maxd=max(maxd,dd)
        line+=f"{dd:7.2f} "
    print(line)
print(f"  MAX pairwise distance: {maxd:.2f} km")

# datetime range
dts=sorted(d['datetime_IST'] for d in rows)
print("\nDATE RANGE:", dts[0], "->", dts[-1])

numcols=['PM25','Temperature','RH','Wind_Speed','Wind_Direction','PBLH','Rainfall_mm','Cloud_Cover','AOD','Night_Light','Ground_PM2.5','PM25_difference']
def fnum(v):
    try:
        if v is None or v=='' : return None
        return float(v)
    except: return None

print("\nPER-COLUMN STATS (n, miss%, min, mean, max, std):")
data={c:[] for c in numcols}
for d in rows:
    for c in numcols:
        data[c].append(fnum(d[c]))
for c in numcols:
    vals=[x for x in data[c] if x is not None]
    miss=100*(len(data[c])-len(vals))/len(data[c])
    if vals:
        print(f"  {c:16s} n={len(vals):6d} miss={miss:5.2f}%  min={min(vals):9.3f} mean={st.mean(vals):9.3f} max={max(vals):10.3f} std={st.pstdev(vals):8.3f}")

# correlation of features with Ground_PM2.5
def pear(x,y):
    pts=[(a,b) for a,b in zip(x,y) if a is not None and b is not None]
    if len(pts)<3: return None,0
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    mx=st.mean(xs); my=st.mean(ys)
    num=sum((a-mx)*(b-my) for a,b in pts)
    den=math.sqrt(sum((a-mx)**2 for a in xs)*sum((b-my)**2 for b in ys))
    return (num/den if den else None), len(pts)
g=data['Ground_PM2.5']
print("\nPEARSON r WITH Ground_PM2.5:")
for c in numcols:
    if c=='Ground_PM2.5': continue
    rr,n=pear(data[c],g)
    if rr is not None:
        print(f"  {c:16s} r={rr:+.3f}  (n={n})")

# satellite vs ground
pairs=[(fnum(d['PM25']),fnum(d['Ground_PM2.5'])) for d in rows]
pairs=[(a,b) for a,b in pairs if a is not None and b is not None]
sat=[a for a,b in pairs]; gr=[b for a,b in pairs]
mae=st.mean(abs(a-b) for a,b in pairs)
bias=st.mean(a-b for a,b in pairs)
rmse=math.sqrt(st.mean((a-b)**2 for a,b in pairs))
rr,_=pear(sat,gr)
print(f"\nSATELLITE PM25 vs GROUND: n={len(pairs)} MAE={mae:.2f} bias(sat-grd)={bias:+.2f} RMSE={rmse:.2f} r={rr:+.3f}")

# monthly seasonality of Ground_PM2.5
mon=defaultdict(list)
for d in rows:
    v=fnum(d['Ground_PM2.5'])
    if v is not None:
        m=d['datetime_IST'][5:7]
        mon[m].append(v)
print("\nMONTHLY MEAN Ground_PM2.5:")
for m in sorted(mon):
    vs=mon[m]
    print(f"  month {m}: mean={st.mean(vs):7.2f}  n={len(vs)}")

# seasonal grouping
seas={'Winter(DJF)':['12','01','02'],'Pre-monsoon(MAM)':['03','04','05'],'Monsoon(JJAS)':['06','07','08','09'],'Post-monsoon(ON)':['10','11']}
print("\nSEASONAL MEAN Ground_PM2.5:")
for nm,ms in seas.items():
    vs=[v for m in ms for v in mon.get(m,[])]
    if vs: print(f"  {nm:18s} mean={st.mean(vs):7.2f} std={st.pstdev(vs):6.2f} n={len(vs)}")

# diurnal
hr=defaultdict(list)
for d in rows:
    v=fnum(d['Ground_PM2.5'])
    if v is not None:
        hr[d['datetime_IST'][11:13]].append(v)
print("\nDIURNAL MEAN Ground_PM2.5 (by hour):")
for h in sorted(hr):
    print(f"  {h}:00 mean={st.mean(hr[h]):7.2f}")

# Night_Light variation
nl=data['Night_Light']
nlv=[x for x in nl if x is not None]
print(f"\nNight_Light distinct count: {len(set(nlv))}  min={min(nlv):.3f} max={max(nlv):.3f}")
# per station night light
print("Night_Light per station (mean):")
pernl=defaultdict(list)
for d in rows:
    v=fnum(d['Night_Light'])
    if v is not None: pernl[d['Station']].append(v)
for s in coords:
    print(f"  {s:20s} {st.mean(pernl[s]):.3f}")

# per-station mean ground pm
print("\nPER-STATION MEAN Ground_PM2.5:")
perg=defaultdict(list)
for d in rows:
    v=fnum(d['Ground_PM2.5'])
    if v is not None: perg[d['Station']].append(v)
for s in coords:
    print(f"  {s:20s} mean={st.mean(perg[s]):7.2f} n={len(perg[s])}")

print("\nDONE")
