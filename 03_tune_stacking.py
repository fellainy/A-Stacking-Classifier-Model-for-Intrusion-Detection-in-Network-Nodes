from pathlib import Path
import json
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from src.io_utils import load_config, load_dataset, make_binary_target, save_json
from src.models import make_xgb
from src.metrics import classification_metrics

ROOT=Path(__file__).resolve().parent
cfg=load_config(ROOT / "config.yaml")
df=load_dataset(cfg, ROOT)
_, yser, _=make_binary_target(df,cfg)
train_idx=pd.read_csv(ROOT / "data/splits/train_indices.csv")["index"].to_numpy()
y=yser.iloc[train_idx].to_numpy(dtype=int)
oof=pd.read_csv(ROOT / "outputs/tables/oof_all.csv")
summary=pd.read_csv(ROOT / "outputs/tables/table6_summary.csv")
base=summary.sort_values("F1",ascending=False).head(cfg["stacking"]["n_base_learners"])["Classifier"].tolist()
Z=oof[base].to_numpy(dtype=float)
opt=cfg["optuna_tpe"]
sp=opt["search_space"]


def suggest(trial, name):
    s=sp[name]
    if s["type"]=="int": return trial.suggest_int(name,s["low"],s["high"])
    return trial.suggest_float(name,s["low"],s["high"],log=s.get("log",False))


def objective(trial):
    params={name:suggest(trial,name) for name in sp}
    cv=StratifiedKFold(n_splits=opt["cv_splits"],shuffle=True,random_state=opt["random_state"])
    aucs=[]
    for tr,va in cv.split(Z,y):
        model=make_xgb(params,opt["random_state"])
        model.fit(Z[tr],y[tr])
        aucs.append(roc_auc_score(y[va],model.predict_proba(Z[va])[:,1]))
    return float(np.mean(aucs))

sampler=optuna.samplers.TPESampler(seed=opt["random_state"])
study=optuna.create_study(direction=opt["direction"],sampler=sampler)
study.optimize(objective,n_trials=opt["n_trials"])
best=study.best_params
save_json({"base_learners":base,"best_params":best,"best_cv_auc":study.best_value,"n_trials":opt["n_trials"],"search_space":sp},ROOT/"outputs/logs/tpe_optimization.json")

# Recalculate Table 7/8 stacking CV using best params.
cv=StratifiedKFold(n_splits=cfg["cross_validation"]["n_splits"],shuffle=True,random_state=cfg["cross_validation"]["random_state"])
rows=[]
for fold,(tr,va) in enumerate(cv.split(Z,y),start=1):
    m=make_xgb(best,cfg["project"]["random_state"]+fold)
    m.fit(Z[tr],y[tr])
    score=m.predict_proba(Z[va])[:,1]
    pred=(score>=cfg["stacking"]["decision_threshold"]).astype(int)
    rows.append({"Fold":fold,**classification_metrics(y[va],pred,score)})
t7=pd.DataFrame(rows)
t7.to_csv(ROOT/"outputs/tables/table7_tuned_meta_cv.csv",index=False)
t7.to_csv(ROOT/"outputs/tables/table8_stacking_cv.csv",index=False)
print("Base learners:",base)
print("Best TPE params:",best)
print("Best CV AUC:",study.best_value)
print(pd.concat([t7.mean(numeric_only=True).rename("Mean"),t7.std(numeric_only=True,ddof=1).rename("SD")],axis=1))
