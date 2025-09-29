# python/pipeline/customers/city_rep.py
from __future__ import annotations
import pandas as pd

S = pd.StringDtype()

def mode_or_first(s: pd.Series):
    m = s.mode(dropna=True)
    if not m.empty:
        return m.iat[0]
    non_null = s.dropna()
    return non_null.iat[0] if non_null.size else pd.NA

def build_shopuser_city(customers: pd.DataFrame, *, id_col="shopUserId", city_col="invoiceCity") -> pd.Series:
    """Series: shopUserId -> representative city (mode; ties/none -> first non-null)."""
    return (
        customers.groupby(id_col, dropna=False)[city_col]
        .agg(mode_or_first)
        .astype(S)
    )

def assign_city_to_transactions(tx: pd.DataFrame, cust_city: pd.Series, *, id_col="shopUserId",
                                city_col="invoiceCity", unknown_token="Unknown") -> pd.DataFrame:
    """Copy of tx with city mapped; unmatched -> unknown_token."""
    out = tx.copy()
    out[city_col] = out[id_col].map(cust_city).fillna(unknown_token).astype(S)
    return out

def enrich_and_dedup_customers(
    customers: pd.DataFrame,
    *,
    id_col: str = "shopUserId",
    city_col: str = "invoiceCity",
) -> pd.DataFrame:
    cust_city = build_shopuser_city(customers, id_col=id_col, city_col=city_col)

    df = customers.copy()
    df[city_col] = df[city_col].astype(S)
    df[city_col] = df[city_col].fillna(df[id_col].map(cust_city))

    tmp = df.reset_index(names="_row")
    tmp["_city_isna"] = tmp[city_col].isna()      # ← temp key
    tmp = tmp.sort_values([id_col, "_city_isna", "_row"])
    out = (
        tmp.drop_duplicates(subset=[id_col], keep="first")
           .drop(columns=["_row", "_city_isna"])
           .reset_index(drop=True)
    )
    return out
