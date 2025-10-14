import pandas as pd

DROP_COLS_DEFAULT: tuple[str, ...] = (
    "status","incommingQuantity","length","width","height","weight",
    "fabricId","fabric","description","colorId","color","sizeId","size",
    "audience","audienceId","publishedDate","quantity"
)
PRICE_COLS_DEFAULT: tuple[str, ...] = ("priceSEK","priceEUR","priceNOK","priceDKK")

def drop_noise_columns(
    df: pd.DataFrame,
    *,
    drop_cols: tuple[str, ...] = DROP_COLS_DEFAULT,
) -> pd.DataFrame:
    return df.drop(columns=list(drop_cols), errors="ignore")

def remove_rows_all_prices_na(
    df: pd.DataFrame,
    *,
    price_cols: tuple[str, ...] = PRICE_COLS_DEFAULT,
) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in price_cols if c not in out]
    if missing:
        out[missing] = pd.NA
    keep = ~out[list(price_cols)].isna().all(axis=1)
    return out.loc[keep].reset_index(drop=True)
