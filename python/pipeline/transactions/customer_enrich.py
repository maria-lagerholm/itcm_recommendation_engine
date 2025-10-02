from __future__ import annotations
import pandas as pd

#------enrich transactions with customers------
def enrich_tx_with_customers(
    tx: pd.DataFrame,
    customers: pd.DataFrame,
    *,
    id_col: str = "shopUserId",
    customer_cols: tuple[str, ...] = ("Age", "Gender"),
) -> pd.DataFrame:
    needed = (id_col, *customer_cols)
    missing = [c for c in customer_cols if c not in customers.columns]
    if missing:
        raise KeyError(f"Missing customer columns: {missing}")
    out = tx.merge(
        customers[list(needed)],
        on=id_col,
        how="left",
        validate="many_to_one",
    )
    return out

#------filter transactions by age------
def filter_tx_by_age(
    tx: pd.DataFrame,
    *,
    age_col: str = "Age",
    lo: int = 10,
    hi: int = 105,
) -> pd.DataFrame:
    mask = tx[age_col].isna() | ((tx[age_col] >= lo) & (tx[age_col] <= hi))
    return tx.loc[mask].reset_index(drop=True)
