from __future__ import annotations
import pandas as pd

def normalize_quantity_to_str(
    df: pd.DataFrame,
    *,
    quantity_col: str = "quantity",
) -> pd.DataFrame:
    out = df.copy()
    q = pd.to_numeric(out[quantity_col], errors="coerce").round().astype("Int64")
    out[quantity_col] = q.astype(str)
    return out

def compute_and_filter_line_total_sek(
    df: pd.DataFrame,
    *,
    price_col: str = "price_sek",
    quantity_col: str = "quantity",
    out_col: str = "line_total_sek",
) -> pd.DataFrame:
    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out[quantity_col] = pd.to_numeric(out[quantity_col], errors="coerce")
    out[out_col] = (out[price_col] * out[quantity_col]).round()
    out[out_col] = pd.to_numeric(out[out_col], errors="coerce")
    out = out.dropna(subset=[out_col])
    out = out[out[out_col] != 0]
    return out.reset_index(drop=True)