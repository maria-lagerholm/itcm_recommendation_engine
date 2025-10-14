import pandas as pd

def _mode_or_first(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return pd.NA
    # Most frequent; ties resolved by first appearance in value_counts order
    return s.value_counts().idxmax()

def build_shopuser_city(
    customers: pd.DataFrame, *, id_col="shopUserId", city_col="invoiceCity"
) -> pd.Series:
    return (
        customers.groupby(id_col, dropna=False)[city_col]
        .agg(_mode_or_first)
        .astype("string")
    )

def assign_city_to_transactions(
    tx: pd.DataFrame, cust_city: pd.Series, *, id_col="shopUserId",
    city_col="invoiceCity", unknown_token="Unknown"
) -> pd.DataFrame:
    out = tx.copy()
    out[city_col] = out[id_col].map(cust_city).fillna(unknown_token).astype("string")
    return out

def enrich_and_dedup_customers(
    customers: pd.DataFrame, *, id_col: str = "shopUserId", city_col: str = "invoiceCity"
) -> pd.DataFrame:
    cust_city = build_shopuser_city(customers, id_col=id_col, city_col=city_col)

    df = customers.copy()
    df[city_col] = df[city_col].astype("string").fillna(df[id_col].map(cust_city))

    # Prefer rows where city is not NA; if all NA within an id, keep the first row
    is_na = df[city_col].isna()
    idx = is_na.groupby(df[id_col], dropna=False).idxmin()
    return df.loc[idx].reset_index(drop=True)
