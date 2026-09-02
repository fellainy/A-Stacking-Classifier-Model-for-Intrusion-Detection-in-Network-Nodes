from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from src.io_utils import load_config, load_dataset, make_binary_target, ensure_dirs, save_json
from src.preprocessing import fit_fold_preprocessor, undersample_training, fit_scaler
from src.boruta_shadow import ShadowBorutaSelector
from src.models import build_baseline_models, score_values, make_xgb
from src.metrics import classification_metrics, confusion_counts

ROOT=Path(__file__).resolve().parent
cfg=load_config(ROOT/"config.yaml")
ensure_dirs(ROOT)
df=load_dataset(cfg,ROOT)
Xdf,yser,_=make_binary_target(df,cfg)
tr_idx=pd.read_csv(ROOT/"data/splits/train_indices.csv")["index"].to_numpy()
te_idx=pd.read_csv(ROOT/"data/splits/test_indices.csv")["index"].to_numpy()
Xtr=Xdf.iloc[tr_idx].to_numpy(dtype=float); ytr=yser.iloc[tr_idx].to_numpy(dtype=int)
Xte=Xdf.iloc[te_idx].to_numpy(dtype=float); yte=yser.iloc[te_idx].to_numpy(dtype=int)

with open(ROOT/"outputs/logs/tpe_optimization.json",encoding="utf-8") as f: tune=json.load(f)
base=tune["base_learners"]; best=tune["best_params"]
oof=pd.read_csv(ROOT/"outputs/tables/oof_all.csv")
Ztrain=oof[base].to_numpy(dtype=float)

t0=time.perf_counter()
imputer,Xtr_imp=fit_fold_preprocessor(Xtr); Xte_imp=imputer.transform(Xte)
us=cfg["undersampling"]
Xbal,ybal,_=undersample_training(Xtr_imp,ytr,us["target_benign_fraction"],us["random_state"])
bc=cfg["boruta_compatible"]
selector=ShadowBorutaSelector(bc["n_iterations"],bc["n_estimators"],bc["random_state"],bc["class_weight"],bc["n_jobs"]).fit(Xbal,ybal)
Xbal_sel=selector.transform(Xbal); Xte_sel=selector.transform(Xte_imp)
scaler,Xbal_sc=fit_scaler(Xbal_sel); Xte_sc=scaler.transform(Xte_sel)

all_models=build_baseline_models(cfg)
rows=[]; test_scores={}; test_preds={}
for name,proto in all_models.items():
    model=clone(proto); model.fit(Xbal_sc,ybal)
    score=score_values(model,Xte_sc); pred=model.predict(Xte_sc)
    test_scores[name]=score; test_preds[name]=pred
    rows.append({"Classifier":name,**classification_metrics(yte,pred,score),**confusion_counts(yte,pred)})

# Fit level-1 learner only on training-derived OOF meta-features.
meta=make_xgb(best,cfg["project"]["random_state"])
meta.fit(Ztrain,ytr)
Ztest=np.column_stack([test_scores[n] for n in base])
stack_score=meta.predict_proba(Ztest)[:,1]
stack_pred=(stack_score>=cfg["stacking"]["decision_threshold"]).astype(int)
rows.append({"Classifier":"Proposed Stacking Classifier",**classification_metrics(yte,stack_pred,stack_score),**confusion_counts(yte,stack_pred)})
end_to_end_seconds=time.perf_counter()-t0

res=pd.DataFrame(rows).sort_values("F1",ascending=False)
res.to_csv(ROOT/"outputs/tables/table10_test_performance.csv",index=False)
np.savez_compressed(ROOT/"outputs/models/test_predictions.npz",y_test=yte,stack_score=stack_score,stack_pred=stack_pred,**{f"score_{i}":test_scores[n] for i,n in enumerate(base)})
feature_table=pd.DataFrame({"Feature":Xdf.columns,"Final_selected":selector.support_.astype(int),"Final_shadow_hits":selector.hits_,"One_sided_p":selector.p_values_,"Mean_RF_importance":selector.importances_})
feature_table=feature_table[feature_table.Final_selected==1].sort_values("Mean_RF_importance",ascending=False)
feature_table.to_csv(ROOT/"outputs/tables/table9_final_features.csv",index=False)

save_json({"end_to_end_final_fit_and_test_seconds":end_to_end_seconds,"base_learners":base,"n_final_selected_features":int(selector.support_.sum()),"test_n":int(len(yte)),"test_benign":int((yte==0).sum()),"test_attack":int((yte==1).sum())},ROOT/"outputs/logs/final_runtime.json")
joblib.dump({"imputer":imputer,"selector":selector,"scaler":scaler,"meta":meta,"base_learners":base},ROOT/"outputs/models/final_pipeline_components.joblib")
print(res)
print(f"Final-fit + untouched-test stage runtime: {end_to_end_seconds:.3f} s")
