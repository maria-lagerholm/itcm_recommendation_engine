# python/pipeline/articles/remove_known_bugs.py
from __future__ import annotations
import pandas as pd

# Columns to drop up-front (from your notebook)
DROP_COLS_DEFAULT: tuple[str, ...] = (
    "status", "incommingQuantity", "length", "width", "height", "weight",
    "fabricId", "fabric", "description", "colorId", "color", "sizeId", "size",
    "audience", "audienceId", "publishedDate", "quantity"
)

# Price columns used to test for "all missing"
PRICE_COLS_DEFAULT: tuple[str, ...] = ("priceSEK", "priceEUR", "priceNOK", "priceDKK")


def drop_noise_columns(
    df: pd.DataFrame,
    *,
    drop_cols: tuple[str, ...] = DROP_COLS_DEFAULT,
) -> pd.DataFrame:
    """Return a copy with the specified columns dropped (ignores if missing)."""
    out = df.copy()
    existing = [c for c in drop_cols if c in out.columns]
    if existing:
        out = out.drop(columns=existing, errors="ignore")
    return out


def remove_rows_all_prices_na(
    df: pd.DataFrame,
    *,
    price_cols: tuple[str, ...] = PRICE_COLS_DEFAULT,
) -> pd.DataFrame:
    """Return a copy with rows removed where *all* given price columns are NA."""
    out = df.copy()
    for c in price_cols:
        if c not in out.columns:
            out[c] = pd.NA
    keep = ~out[list(price_cols)].isna().all(axis=1)
    return out.loc[keep].reset_index(drop=True)
