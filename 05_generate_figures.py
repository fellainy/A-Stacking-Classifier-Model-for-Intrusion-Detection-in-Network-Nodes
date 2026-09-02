from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
from src.io_utils import load_config, load_dataset, make_binary_target

ROOT=Path(__file__).resolve().parent
cfg=load_config(ROOT/"config.yaml")
figdir=ROOT/"outputs/figures"; figdir.mkdir(parents=True,exist_ok=True)

# Feature-selection stability
ff=pd.read_csv(ROOT/"outputs/tables/feature_freq.csv").head(20).sort_values("CV_folds_selected")
plt.figure(figsize=(8,7)); plt.barh(ff["Feature"],ff["CV_folds_selected"]); plt.xlabel("CV folds selected (out of 10)"); plt.tight_layout(); plt.savefig(figdir/"figure2_boruta_stability.png",dpi=300); plt.close()

# Final model comparison
res=pd.read_csv(ROOT/"outputs/tables/table10_test_performance.csv").sort_values("F1",ascending=False)
plot=res.head(6).set_index("Classifier")[["Accuracy","Recall","Precision","F1"]]
ax=plot.plot(kind="bar",figsize=(10,6)); ax.set_ylim(0.75,1.005); ax.set_ylabel("Score"); plt.xticks(rotation=30,ha="right"); plt.tight_layout(); plt.savefig(figdir/"figure3_test_metric_comparison.png",dpi=300); plt.close()

# ROC and stacking confusion matrix use stored predictions.
npz=np.load(ROOT/"outputs/models/test_predictions.npz")
y=npz["y_test"]
with open(ROOT/"outputs/logs/tpe_optimization.json",encoding="utf-8") as f: base=json.load(f)["base_learners"]
plt.figure(figsize=(8,6))
for i,name in enumerate(base):
    s=npz[f"score_{i}"]; fpr,tpr,_=roc_curve(y,s); plt.plot(fpr,tpr,label=f"{name} (AUC={auc(fpr,tpr):.6f})")
s=npz["stack_score"]; fpr,tpr,_=roc_curve(y,s); plt.plot(fpr,tpr,label=f"Stacking (AUC={auc(fpr,tpr):.6f})",linewidth=2)
plt.plot([0,1],[0,1],linestyle="--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(figdir/"figure4_roc_curves.png",dpi=300); plt.close()

cm=confusion_matrix(y,npz["stack_pred"],labels=[0,1])
disp=ConfusionMatrixDisplay(cm,display_labels=["Benign","Attack"]); disp.plot(values_format="d"); plt.tight_layout(); plt.savefig(figdir/"figure10_stacking_confusion_matrix.png",dpi=300); plt.close()
print(f"Figures written to {figdir}")
