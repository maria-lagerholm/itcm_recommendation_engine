from __future__ import annotations
import pandas as pd

# columns you said we can drop early
DROP_COLS_DEFAULT = (
    "status", "incommingQuantity", "length", "width", "height",
    "weight", "fabricId", "fabric",
)

PRICE_COLS_DEFAULT = ("priceSEK", "priceEUR", "priceNOK", "priceDKK")

def drop_noise_columns(df: pd.DataFrame, cols: tuple[str, ...] = DROP_COLS_DEFAULT) -> pd.DataFrame:
    """Drop unneeded columns if present."""
    return df.drop(columns=list(cols), errors="ignore")

def remove_rows_all_prices_na(
    df: pd.DataFrame,
    price_cols: tuple[str, ...] = PRICE_COLS_DEFAULT
) -> pd.DataFrame:
    """
    Remove rows where *all* price columns are NA.
    Works with pandas StringDtype (<NA>) as well as regular NaN.
    """
    missing = [c for c in price_cols if c not in df.columns]
    if missing:
        # if a price column is missing entirely, treat it as NA for the check by creating it
        df = df.copy()
        for c in missing:
            df[c] = pd.Series(pd.NA, index=df.index, dtype="string")
    mask_all_na = df[list(price_cols)].isna().all(axis=1)
    return df.loc[~mask_all_na].reset_index(drop=True)