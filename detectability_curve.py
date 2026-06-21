import importlib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
GRAVITY=9.80665; DEG2RAD=np.pi/180; EARTH_R=6378137.0; IMU_HZ=100; CHUNK=100
ef=importlib.import_module("2_prepare_data").extract_features
FC=importlib.import_module("2_prepare_data").FEATURE_COLS
WIN,STRIDE=200,50

def load_imu(p):
    df=pd.read_csv(p);o=pd.DataFrame();o['timestamp']=(df['ts_ms']-df['ts_ms'].iloc[0])/1000.0
    o['ax']=df['accelX']*GRAVITY;o['ay']=df['accelY']*GRAVITY;o['az']=df['accelZ']*GRAVITY
    o['gx']=df['gyroX']*DEG2RAD;o['gy']=df['gyroY']*DEG2RAD;o['gz']=df['gyroZ']*DEG2RAD
    if o['az'].mean()<0: o['az']=-o['az'];o['gz']=-o['gz']
    return o.reset_index(drop=True)
def load_gps(p,t0=0.0):
    df=pd.read_csv(p);rlat=df['lat'].iloc[0];rlon=df['lon'].iloc[0];lr=rlat*DEG2RAD;o=pd.DataFrame()
    o['timestamp']=(df['ts_ms']-t0)/1000.0
    o['gps_pn']=(df['lat']-rlat)*DEG2RAD*EARTH_R;o['gps_pe']=(df['lon']-rlon)*DEG2RAD*EARTH_R*np.cos(lr)
    o['gps_vn']=df['velN']/1000.0;o['gps_ve']=df['velE']/1000.0
    if 'fixType' in df.columns:
        b=(df['fixType']<2).values;o.loc[b,['gps_pn','gps_pe','gps_vn','gps_ve']]=np.nan
    return o.reset_index(drop=True)
def merge(imu,gps):
    m=imu.copy()
    for c in ['gps_pn','gps_pe','gps_vn','gps_ve']:m[c]=np.nan
    its=m['timestamp'].values;v=gps.dropna(subset=['gps_pn'])
    if len(v)>0:
        idx=np.clip(np.searchsorted(its,v['timestamp'].values),0,len(its)-1)
        for c in ['gps_pn','gps_pe','gps_vn','gps_ve']:m.iloc[idx,m.columns.get_loc(c)]=v[c].values
    m['label']=-1;return m
def chunk(base,rng):
    tot=base['timestamp'].iloc[-1];ms=max(0.0,tot-CHUNK);ts=rng.uniform(0,ms)
    c=base[(base['timestamp']>=ts)&(base['timestamp']<ts+CHUNK)].copy()
    c['timestamp']-=c['timestamp'].iloc[0];c=c.reset_index(drop=True);n=len(c)
    c['ax']+=rng.normal(0,0.03,n);c['ay']+=rng.normal(0,0.03,n);c['az']+=rng.normal(0,0.02,n)
    c['gx']+=rng.normal(0,0.002,n);c['gy']+=rng.normal(0,0.002,n);c['gz']+=rng.normal(0,0.001,n)
    gm=c['gps_pn'].notna();ng=gm.sum()
    if ng>0:
        for col,s in [('gps_pn',1.5),('gps_pe',1.5),('gps_vn',0.05),('gps_ve',0.05)]:
            c.loc[gm,col]+=rng.normal(0,s,ng)
    return c
def drift(c,accel,seed):
    c=c.copy();c['label']=0
    if accel<=0:return c
    ang=seed*0.17;ca,sa=np.cos(ang),np.sin(ang);t=c['timestamp'].values;tc=t[-1]
    s0=tc*0.3;s1=tc*0.85;gm=c['gps_pn'].notna().values;insp=(t>=s0)&(t<=s1);sg=gm&insp;el=t[sg]-s0
    pos=0.5*accel*el**2;vel=accel*el
    c.loc[sg,'gps_pn']+=pos*ca;c.loc[sg,'gps_pe']+=pos*sa
    c.loc[sg,'gps_vn']+=vel*ca;c.loc[sg,'gps_ve']+=vel*sa
    c.loc[insp,'label']=1;return c

KIMU=["imu_kitti_0015.csv","imu_kitti_0019.csv","imu_kitti_0022.csv","imu_kitti_0023.csv","imu_kitti_0027.csv","imu_kitti_0028.csv","imu_kitti_0029.csv","imu_kitti_0032.csv","imu_kitti_0035.csv","imu_kitti_0036.csv","imu_kitti_0039.csv","imu_kitti_0051.csv","imu_kitti_0056.csv","imu_kitti_0059.csv","imu_kitti_0061.csv","imu_kitti_0064.csv","imu_kitti_0070.csv","imu_kitti_0084.csv","imu_kitti_0086.csv","imu_kitti_0087.csv","imu_kitti_0091.csv","imu_kitti_0101.csv","imu_kitti_09.csv","imu_kitti_103_0027.csv","imu_kitti_103_0042.csv","imu_kitti_103_0047.csv","imu_kitti_14.csv","imu_kitti_30_0016.csv","imu_kitti_30_0018.csv","imu_kitti_30_0020.csv","imu_kitti_30_0034.csv","imu_20260519_223444.csv"]
KGPS=[f.replace("imu_","gps_") for f in KIMU]
print("base kuruluyor...")
i0=pd.read_csv(KIMU[0]);idf=load_imu(KIMU[0]);gdf=load_gps(KGPS[0],float(i0['ts_ms'].iloc[0]))
for ip,gp in zip(KIMU[1:],KGPS[1:]):
    r=pd.read_csv(ip);ei=load_imu(ip);eg=load_gps(gp,float(r['ts_ms'].iloc[0]))
    off=idf['timestamp'].iloc[-1]+1.0/IMU_HZ;ei['timestamp']+=off;eg['timestamp']+=off
    idf=pd.concat([idf,ei],ignore_index=True);gdf=pd.concat([gdf,eg],ignore_index=True)
base=merge(idf,gdf)

LEVELS=[0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0]
def feats(c):
    g=ef(c.assign(trajectory_id=0));fv=g[FC].values;lb=g['label'].values;n=len(g)
    Xs=[];in_atk=[]
    for s in range(0,n-WIN,STRIDE):
        wl=lb[s:s+WIN];valid=wl[wl>=0]
        if len(valid)==0: continue
        fr=(wl>0).mean()
        if fr==0: y=0
        elif fr>=0.7: y=1
        else: continue
        W=fv[s:s+WIN]
        Xs.append(np.concatenate([W.mean(0),W.std(0),W.min(0),W.max(0),W[-1]-W[0]]));in_atk.append(y)
    return np.array(Xs,np.float32),np.array(in_atk)

# TRAIN: karisik seviyeli drift + normal
rng=np.random.RandomState(1);Xtr=[];Ytr=[]
for i in range(40):
    c=chunk(base,rng)
    if i%2==0: c=drift(c,rng.uniform(0.1,2.0),i+1)
    else: c['label']=0
    X,y=feats(c)
    if len(X): Xtr.append(X);Ytr.append(y)
Xtr=np.vstack(Xtr);Ytr=np.concatenate(Ytr)
sc=StandardScaler().fit(Xtr)
clf=HistGradientBoostingClassifier(max_iter=300,max_depth=6,learning_rate=0.08,
    l2_regularization=1.0,class_weight="balanced",random_state=0).fit(sc.transform(Xtr),Ytr)

# normal yanlis alarm
rng2=np.random.RandomState(99);fa=0;fatot=0
for i in range(15):
    c=chunk(base,rng2);c['label']=0;X,y=feats(c)
    if len(X):
        p=clf.predict(sc.transform(X));fa+=int(p.sum());fatot+=len(p)
print(f"\nNormal yanlis alarm: %{100*fa/max(1,fatot):.1f}\n")
print("DRIFT TESPIT EGRISI (her seviye 8 trajectory, tespit=saldiri icinde >=1 pencere alarm):")
print(f"{'accel(m/s2)':>12} {'~hiz ofseti/30s':>16} {'tespit %':>10}")
for L in LEVELS:
    det=0;N=8;rngL=np.random.RandomState(int(L*1000)+7)
    for j in range(N):
        c=chunk(base,rngL);c=drift(c,L,j+1);X,y=feats(c)
        if len(X) and (clf.predict(sc.transform(X))[y==1]>0).any() if (y==1).any() else False:
            det+=1
    voff=L*30
    print(f"{L:>12.2f} {voff:>14.1f} m/s {100*det/N:>9.0f}")
