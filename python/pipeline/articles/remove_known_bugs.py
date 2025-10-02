# python/pipeline/articles/remove_known_bugs.py
from __future__ import annotations
import pandas as pd

#------constants-----
DROP_COLS_DEFAULT: tuple[str, ...] = (
    "status", "incommingQuantity", "length", "width", "height", "weight",
    "fabricId", "fabric", "description", "colorId", "color", "sizeId", "size",
    "audience", "audienceId", "publishedDate", "quantity"
)

PRICE_COLS_DEFAULT: tuple[str, ...] = ("priceSEK", "priceEUR", "priceNOK", "priceDKK")

#------core-----
def drop_noise_columns(
    df: pd.DataFrame,
    *,
    drop_cols: tuple[str, ...] = DROP_COLS_DEFAULT,
) -> pd.DataFrame:
    out = df.copy()
    existing = [c for c in drop_cols if c in out.columns]
    if existing:
        out = out.drop(columns=existing, errors="ignore")
    return out

#------core-----
def remove_rows_all_prices_na(
    df: pd.DataFrame,
    *,
    price_cols: tuple[str, ...] = PRICE_COLS_DEFAULT,
) -> pd.DataFrame:
    out = df.copy()
    for c in price_cols:
        if c not in out.columns:
            out[c] = pd.NA
    keep = ~out[list(price_cols)].isna().all(axis=1)
    return out.loc[keep].reset_index(drop=True)
