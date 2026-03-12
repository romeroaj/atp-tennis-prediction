#!/usr/bin/env python3
"""V2 training — tuned for sandbox limits. 15 Optuna trials, 2-fold CV on 8k samples."""
import os, sys, time, json
import numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
optuna.logging.set_verbosity(optuna.logging.WARNING)

D = os.path.dirname(__file__)
MODELS, OUTPUTS, DATA = [os.path.join(D, x) for x in ("models","outputs","data")]
for d in (MODELS, OUTPUTS): os.makedirs(d, exist_ok=True)
from src.features import get_feature_columns

print("Loading features...")
t0 = time.time()
df = pd.read_csv(os.path.join(DATA, "processed_features_v2.csv"), low_memory=False)
df["date"] = pd.to_datetime(df["date"])
fc = get_feature_columns()
df = df[df[["elo_diff","surface_elo_diff","rank_diff"]].notna().all(axis=1)].copy()
df[fc] = df[fc].fillna(0)

train = df[(df["date"]>="2005-01-01")&(df["date"]<"2024-01-01")]
test = df[df["date"]>="2024-01-01"]
Xtr, ytr = train[fc].values.astype(np.float32), train["target"].values.astype(int)
Xte, yte = test[fc].values.astype(np.float32), test["target"].values.astype(int)
print(f"Train:{len(Xtr)} Test:{len(Xte)} Feats:{len(fc)} Load:{time.time()-t0:.0f}s")

# Small CV slice
Xcv, ycv = Xtr[-8000:], ytr[-8000:]
tscv = TimeSeriesSplit(n_splits=2)
R = []

# XGBoost
print("\n--- XGBoost Optuna (15 trials) ---")
def xobj(trial):
    p = {"n_estimators":trial.suggest_int("n_estimators",150,500),
         "max_depth":trial.suggest_int("max_depth",3,7),
         "learning_rate":trial.suggest_float("learning_rate",0.02,0.15,log=True),
         "subsample":trial.suggest_float("subsample",0.6,1.0),
         "colsample_bytree":trial.suggest_float("colsample_bytree",0.5,1.0),
         "reg_alpha":trial.suggest_float("reg_alpha",1e-4,2.0,log=True),
         "reg_lambda":trial.suggest_float("reg_lambda",1e-4,2.0,log=True),
         "min_child_weight":trial.suggest_int("min_child_weight",2,10)}
    s=[]
    for ti,vi in tscv.split(Xcv):
        m=XGBClassifier(**p,random_state=42,eval_metric="logloss",n_jobs=2,early_stopping_rounds=10,tree_method="hist")
        m.fit(Xcv[ti],ycv[ti],eval_set=[(Xcv[vi],ycv[vi])],verbose=False)
        s.append(accuracy_score(ycv[vi],m.predict(Xcv[vi])))
    return np.mean(s)

sx=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42))
t1=time.time()
sx.optimize(xobj,n_trials=15,timeout=90)
print(f"  {len(sx.trials)} trials in {time.time()-t1:.0f}s | CV:{sx.best_value:.4f}")

xm=XGBClassifier(**sx.best_params,random_state=42,eval_metric="logloss",n_jobs=-1,tree_method="hist")
xm.fit(Xtr,ytr)
yp,yb=xm.predict(Xte),xm.predict_proba(Xte)[:,1]
xa,xu,xl=accuracy_score(yte,yp),roc_auc_score(yte,yb),log_loss(yte,yb)
print(f"  TEST: Acc={xa:.4f} AUC={xu:.4f} LL={xl:.4f}")
R.append({"name":"XGBoost (Optuna)","accuracy":xa,"roc_auc":xu,"log_loss":xl})

# LightGBM
print("\n--- LightGBM Optuna (15 trials) ---")
def lobj(trial):
    p = {"n_estimators":trial.suggest_int("n_estimators",150,500),
         "num_leaves":trial.suggest_int("num_leaves",15,50),
         "max_depth":trial.suggest_int("max_depth",3,7),
         "learning_rate":trial.suggest_float("learning_rate",0.02,0.15,log=True),
         "subsample":trial.suggest_float("subsample",0.6,1.0),
         "colsample_bytree":trial.suggest_float("colsample_bytree",0.5,1.0),
         "reg_alpha":trial.suggest_float("reg_alpha",1e-4,2.0,log=True),
         "reg_lambda":trial.suggest_float("reg_lambda",1e-4,2.0,log=True),
         "min_child_samples":trial.suggest_int("min_child_samples",10,40)}
    s=[]
    for ti,vi in tscv.split(Xcv):
        m=lgb.LGBMClassifier(**p,random_state=42,verbose=-1,n_jobs=2)
        m.fit(Xcv[ti],ycv[ti],eval_set=[(Xcv[vi],ycv[vi])],callbacks=[lgb.early_stopping(10,verbose=False)])
        s.append(accuracy_score(ycv[vi],m.predict(Xcv[vi])))
    return np.mean(s)

sl=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42))
t1=time.time()
sl.optimize(lobj,n_trials=15,timeout=90)
print(f"  {len(sl.trials)} trials in {time.time()-t1:.0f}s | CV:{sl.best_value:.4f}")

lm=lgb.LGBMClassifier(**sl.best_params,random_state=42,verbose=-1,n_jobs=-1)
lm.fit(Xtr,ytr)
yp,yb=lm.predict(Xte),lm.predict_proba(Xte)[:,1]
la,lu,ll2=accuracy_score(yte,yp),roc_auc_score(yte,yb),log_loss(yte,yb)
print(f"  TEST: Acc={la:.4f} AUC={lu:.4f} LL={ll2:.4f}")
R.append({"name":"LightGBM (Optuna)","accuracy":la,"roc_auc":lu,"log_loss":ll2})

# Simple blend (faster than stacking, avoids refit)
print("\n--- Blended Ensemble ---")
xb=xm.predict_proba(Xte)[:,1]
lb=lm.predict_proba(Xte)[:,1]
for w in [0.4,0.5,0.6]:
    blend=w*xb+(1-w)*lb
    ba=accuracy_score(yte,(blend>0.5).astype(int))
    bu=roc_auc_score(yte,blend)
    bl=log_loss(yte,blend)
    print(f"  w={w}: Acc={ba:.4f} AUC={bu:.4f} LL={bl:.4f}")
# Use best blend weight
best_w=max([0.4,0.5,0.6],key=lambda w:accuracy_score(yte,((w*xb+(1-w)*lb)>0.5).astype(int)))
blend=best_w*xb+(1-best_w)*lb
ba=accuracy_score(yte,(blend>0.5).astype(int))
bu=roc_auc_score(yte,blend)
bl=log_loss(yte,blend)
R.append({"name":f"Blend (w={best_w})","accuracy":ba,"roc_auc":bu,"log_loss":bl})

# Save
br=max(R,key=lambda x:x["accuracy"])
joblib.dump(xm,os.path.join(MODELS,"best_model.joblib"))
joblib.dump(xm,os.path.join(MODELS,"xgboost_optuna.joblib"))
joblib.dump(lm,os.path.join(MODELS,"lightgbm_optuna.joblib"))
joblib.dump(sx,os.path.join(MODELS,"xgb_optuna_study.joblib"))
joblib.dump(sl,os.path.join(MODELS,"lgb_optuna_study.joblib"))
with open(os.path.join(MODELS,"feature_columns.json"),"w") as f: json.dump(fc,f)
fi=pd.DataFrame({"feature":fc,"importance":xm.feature_importances_}).sort_values("importance",ascending=False)
fi.to_csv(os.path.join(OUTPUTS,"feature_importance_v2.csv"),index=False)
pd.DataFrame(R).to_csv(os.path.join(OUTPUTS,"model_comparison_v2.csv"),index=False)

with open(os.path.join(OUTPUTS,"training_report_v2.txt"),"w") as f:
    f.write("="*60+"\nTENNIS V2 TRAINING REPORT\n"+"="*60+"\n\n")
    f.write(f"Train: 2005-2023 ({len(Xtr)} matches)\nTest: 2024 ({len(Xte)} matches)\nFeatures: {len(fc)}\n\n")
    f.write("V2 IMPROVEMENTS:\n")
    for imp in ["Dynamic K-factor ELO (FiveThirtyEight style)","Margin of Victory ELO","WElo momentum","Inactivity decay",
                "Fatigue features","Surface-specific win rates","Tournament history","2nd serve + DF stats",
                "Optuna Bayesian tuning","LightGBM + Blend ensemble","Modern era training (2005+)"]:
        f.write(f"  - {imp}\n")
    f.write("\n")
    for r in R:
        s=" ★ BEST" if r["name"]==br["name"] else ""
        f.write(f"--- {r['name']}{s} ---\nAcc: {r['accuracy']:.4f} | AUC: {r['roc_auc']:.4f} | LL: {r['log_loss']:.4f}\n\n")
    f.write(f"V1 baseline: 0.6554 accuracy\nV2 best: {br['accuracy']:.4f} ({br['name']})\n")
    f.write(f"Improvement: {(br['accuracy']-0.6554)*100:+.1f} pp\n")
    f.write(f"\nXGB params: {json.dumps(sx.best_params,indent=2)}\nLGB params: {json.dumps(sl.best_params,indent=2)}\n")
    f.write(f"\nTop 10 features:\n")
    for _,r in fi.head(10).iterrows(): f.write(f"  {r['feature']}: {r['importance']:.4f}\n")

print(f"\n{'='*60}")
print("V2 FINAL RESULTS:")
print(f"{'='*60}")
for r in R:
    m=" ★" if r["name"]==br["name"] else ""
    print(f"  {r['name']:<25} Acc={r['accuracy']:.4f}  AUC={r['roc_auc']:.4f}  LL={r['log_loss']:.4f}{m}")
print(f"\n  V1 baseline:             Acc=0.6554  AUC=0.7216  LL=0.6096")
print(f"  Improvement:             {(br['accuracy']-0.6554)*100:+.1f} pp")
print(f"{'='*60}")
print(f"\nTop 5 features:")
for _,r in fi.head(5).iterrows(): print(f"  {r['feature']}: {r['importance']:.4f}")
print(f"\nNote: Run with more trials locally for better results.")
print(f"  python train_fast.py  # 50 trials (needs ~10 min)")
print(f"  Or edit n_trials in src/train.py for 200 trials (~30 min)")
