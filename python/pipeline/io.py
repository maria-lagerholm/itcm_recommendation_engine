# load/save helpers

#------imports------
from pathlib import Path
import pandas as pd, yaml

#------load config------
def load_cfg(path="configs/base.yaml"):
    return yaml.safe_load(Path(path).read_text())

#------read csv as str------
def read_csv_str(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])

#------write parquet------
def write_parquet(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
