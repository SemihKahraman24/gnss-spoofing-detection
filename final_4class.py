import importlib, pickle, os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
m2=importlib.import_module("2_prepare_data"); ef=m2.extract_features; FC=m2.FEATURE_COLS
WIN,STRIDE=200,25; NAMES=["Normal","Yavas Kayma","Ani Atlama","Tekrar Oynat"]
def agg(W): return np.concatenate([W.mean(0),W.std(0),W.min(0),W.max(0),W[-1]-W[0]])
def build_hybrid(df):
    third=WIN//3; X,Y,TID,C,ON,AE=[],[],[],[],[],[]
    for tid,g in df.groupby("trajectory_id"):
        g=g.reset_index(drop=True);fv=g[FC].values;lb=g["label"].values;n=len(g)
        ai=np.where(lb>0)[0];on=int(ai[0]) if len(ai) else -1;ae=int(ai[-1]) if len(ai) else -1
        for s in range(0,n-WIN,STRIDE):
            wl=lb[s:s+WIN];atk=(wl>0);fr=atk.mean()
            if fr==0: y=0
            else:
                dom=int(np.median(wl[wl>0]))
                onset=atk[:third].mean()<0.2 and atk[-third:].mean()>0.5
                if onset: y=dom                         # her tur: onset
                elif fr>=0.8 and dom in (1,3): y=dom     # drift/replay: ic pencere
                else: continue                           # jump ici / karisik -> at
            X.append(agg(fv[s:s+WIN]));Y.append(y);TID.append(tid);C.append(s+WIN//2);ON.append(on);AE.append(ae)
    return (np.array(X,np.float32),np.array(Y),np.array(TID),np.array(C),np.array(ON),np.array(AE))
def split(TID,seed=0):
    rng=np.random.RandomState(seed);u=np.unique(TID);rng.shuffle(u);k=max(1,int(len(u)*.25))
    return ~np.isin(TID,list(u[:k])),np.isin(TID,list(u[:k]))
def trajlevel(TID,C,ON,AE,pred,Ytrue_type):
    # her saldiri trajectory'sinde dogru tur yakalandi mi
    res={1:[0,0],2:[0,0],3:[0,0]}  # tip-> [dogru, toplam]
    fa=0;fatot=0
    for tid in np.unique(TID):
        m=TID==tid;on=ON[m][0];ae=AE[m][0];c=C[m];p=pred[m]
        if on<0:
            fa+=int((p>0).sum());fatot+=len(p);continue
        # gercek tur = bu trajectory'deki pozitif etiketlerin modu
        yt=Ytrue_type[m];tt=int(np.bincount(yt[yt>0]).argmax()) if (yt>0).any() else 0
        if tt==0: continue
        atkmask=(c>=on)&(c<=ae)
        fired=p[atkmask][p[atkmask]>0]
        res[tt][1]+=1
        if len(fired)>0 and int(np.bincount(fired).argmax())==tt: res[tt][0]+=1
        fa+=int((p[c<on]>0).sum());fatot+=int((c<on).sum())
    return res,100*fa/max(1,fatot)

print("[final] gercekci veri (tespit edilebilir drift) + ozellik...")
df=ef(pd.read_csv("data/dataset_realistic_train.csv"))
X,Y,TID,C,ON,AE=build_hybrid(df)
print("sinif dagilimi:",dict(zip(*np.unique(Y,return_counts=True))))
tr,te=split(TID)
sc=StandardScaler().fit(X[tr])
clf=HistGradientBoostingClassifier(max_iter=350,max_depth=6,learning_rate=0.08,
    l2_regularization=1.0,class_weight="balanced",random_state=0).fit(sc.transform(X[tr]),Y[tr])
p=clf.predict(sc.transform(X[te]))
print("\n[HELD-OUT 4-sinif — hibrit etiketleme]")
print(classification_report(Y[te],p,labels=[0,1,2,3],target_names=NAMES,digits=3,zero_division=0))
res,fa=trajlevel(TID[te],C[te],ON[te],AE[te],p,Y[te])
print(f"Trajectory-bazli dogru-tur yakalama:  Yavas {res[1][0]}/{res[1][1]}  Ani {res[2][0]}/{res[2][1]}  Tekrar {res[3][0]}/{res[3][1]}  | yanlis alarm %{fa:.1f}")

# tum train ile yeniden egit + kaydet
scf=StandardScaler().fit(X)
clff=HistGradientBoostingClassifier(max_iter=350,max_depth=6,learning_rate=0.08,
    l2_regularization=1.0,class_weight="balanced",random_state=0).fit(scf.transform(X),Y)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "onset_4class_final.pkl")
os.makedirs(os.path.dirname(out), exist_ok=True)
pickle.dump({"clf":clff,"scaler":scf,"features":FC,"win":WIN,"stride":STRIDE,"names":NAMES},open(out,"wb"))
print("model ->",out)

dte=ef(pd.read_csv("data/dataset_realistic_test.csv"))
Xt,Yt,Tt,Ct,Ot,At=build_hybrid(dte)
pt=clff.predict(scf.transform(Xt))
print("\n[DIS TEST 4-sinif]")
print(classification_report(Yt,pt,labels=[0,1,2,3],target_names=NAMES,digits=3,zero_division=0))
res,fa=trajlevel(Tt,Ct,Ot,At,pt,Yt)
print(f"Trajectory-bazli dogru-tur yakalama:  Yavas {res[1][0]}/{res[1][1]}  Ani {res[2][0]}/{res[2][1]}  Tekrar {res[3][0]}/{res[3][1]}  | yanlis alarm %{fa:.1f}")
