from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
ref=pd.read_csv(ROOT/"outputs/reference_results/reference_table10.csv")
cur_path=ROOT/"outputs/tables/table10_test_performance.csv"
if not cur_path.exists():
    raise SystemExit("Run python run_all.py first.")
cur=pd.read_csv(cur_path)
merged=ref.merge(cur,on="Classifier",suffixes=("_reference","_rerun"))
for metric in ["Accuracy","AUC","Recall","Precision","F1","Kappa","MCC"]:
    merged[f"abs_diff_{metric}"]=(merged[f"{metric}_reference"]-merged[f"{metric}_rerun"]).abs()
cols=["Classifier"]+[c for c in merged.columns if c.startswith("abs_diff_")]
print(merged[cols].to_string(index=False))
print("\nNote: small numerical differences can occur across scikit-learn/XGBoost versions or CPU parallelism. The split seed and leakage-control protocol are fixed.")
