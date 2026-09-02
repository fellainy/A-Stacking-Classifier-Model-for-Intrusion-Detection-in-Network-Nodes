from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.io_utils import load_config, load_dataset, make_binary_target, ensure_dirs, save_json

ROOT = Path(__file__).resolve().parent
cfg = load_config(ROOT / "config.yaml")
ensure_dirs(ROOT)
df = load_dataset(cfg, ROOT)
X, y, labels = make_binary_target(df, cfg)
idx = pd.RangeIndex(len(df)).to_numpy()
train_idx, test_idx = train_test_split(
    idx,
    test_size=cfg["data_split"]["test_size"],
    stratify=y,
    random_state=cfg["project"]["random_state"],
)
pd.DataFrame({"index": train_idx}).to_csv(ROOT / "data/splits/train_indices.csv", index=False)
pd.DataFrame({"index": test_idx}).to_csv(ROOT / "data/splits/test_indices.csv", index=False)
summary = {
    "rows": int(len(df)), "columns_including_target": int(df.shape[1]),
    "train_n": int(len(train_idx)), "test_n": int(len(test_idx)),
    "original_labels": labels.value_counts().to_dict(),
    "train_binary": y.iloc[train_idx].value_counts().sort_index().to_dict(),
    "test_binary": y.iloc[test_idx].value_counts().sort_index().to_dict(),
}
save_json(summary, ROOT / "outputs/logs/data_summary.json")
print(summary)
