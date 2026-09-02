from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from src.io_utils import load_config, load_dataset, make_binary_target, ensure_dirs
from src.preprocessing import fit_fold_preprocessor, undersample_training, fit_scaler
from src.boruta_shadow import ShadowBorutaSelector
from src.models import build_baseline_models, score_values
from src.metrics import classification_metrics

ROOT = Path(__file__).resolve().parent
cfg = load_config(ROOT / "config.yaml")
ensure_dirs(ROOT)
df = load_dataset(cfg, ROOT)
Xdf, yser, _ = make_binary_target(df, cfg)
train_idx = pd.read_csv(ROOT / "data/splits/train_indices.csv")["index"].to_numpy()
X = Xdf.iloc[train_idx].to_numpy(dtype=float)
y = yser.iloc[train_idx].to_numpy(dtype=int)
feature_names = np.array(Xdf.columns)

cv_cfg = cfg["cross_validation"]
cv = StratifiedKFold(n_splits=cv_cfg["n_splits"], shuffle=cv_cfg["shuffle"], random_state=cv_cfg["random_state"])
models = build_baseline_models(cfg)
oof = {name: np.full(len(y), np.nan, dtype=float) for name in models}
rows, feature_freq = [], np.zeros(X.shape[1], dtype=int)

for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
    Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]
    imputer, Xtr_imp = fit_fold_preprocessor(Xtr)
    Xva_imp = imputer.transform(Xva)
    us = cfg["undersampling"]
    Xbal, ybal, _ = undersample_training(Xtr_imp, ytr, us["target_benign_fraction"], us["random_state"] + fold)

    bc = cfg["boruta_compatible"]
    selector = ShadowBorutaSelector(
        n_iterations=bc["n_iterations"], n_estimators=bc["n_estimators"],
        random_state=bc["random_state"] + fold, class_weight=bc["class_weight"], n_jobs=bc["n_jobs"]
    ).fit(Xbal, ybal)
    feature_freq += selector.support_.astype(int)
    Xbal_sel = selector.transform(Xbal)
    Xva_sel = selector.transform(Xva_imp)
    scaler, Xbal_sc = fit_scaler(Xbal_sel)
    Xva_sc = scaler.transform(Xva_sel)

    for name, prototype in models.items():
        model = clone(prototype)
        t0 = time.perf_counter()
        model.fit(Xbal_sc, ybal)
        fit_seconds = time.perf_counter() - t0
        score = score_values(model, Xva_sc)
        pred = model.predict(Xva_sc)
        oof[name][va] = score
        m = classification_metrics(yva, pred, score)
        rows.append({"Fold": fold, "Classifier": name, "TrainingTime": fit_seconds, **m})
    print(f"completed fold {fold}/{cv_cfg['n_splits']} - selected {selector.support_.sum()} features")

raw = pd.DataFrame(rows)
raw.to_csv(ROOT / "outputs/tables/table6_cv_raw.csv", index=False)
metrics = ["Accuracy","AUC","Recall","Precision","F1","Kappa","MCC","TrainingTime"]
summary_rows=[]
for name, g in raw.groupby("Classifier", sort=False):
    row={"Classifier":name}
    for metric in metrics:
        row[metric]=g[metric].mean()
        row[f"{metric}_SD"]=g[metric].std(ddof=1)
    summary_rows.append(row)
summary=pd.DataFrame(summary_rows).sort_values("F1", ascending=False)
summary.to_csv(ROOT / "outputs/tables/table6_summary.csv", index=False)
pd.DataFrame(oof).to_csv(ROOT / "outputs/tables/oof_all.csv", index=False)
pd.DataFrame({"Feature":feature_names, "CV_folds_selected":feature_freq}).sort_values("CV_folds_selected", ascending=False).to_csv(ROOT / "outputs/tables/feature_freq.csv", index=False)
print(summary[["Classifier","Accuracy","AUC","Recall","Precision","F1"]])
