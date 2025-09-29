from __future__ import annotations
import pandas as pd

def enrich_tx_with_customers(
    tx: pd.DataFrame,
    customers: pd.DataFrame,
    *,
    id_col: str = "shopUserId",
    customer_cols: tuple[str, ...] = ("Age", "Gender"),
) -> pd.DataFrame:
    """
    Left-join selected customer columns into transactions on id_col.
    Does not change dtypes (assumes parquet-preserved dtypes).
    """
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

def filter_tx_by_age(
    tx: pd.DataFrame,
    *,
    age_col: str = "Age",
    lo: int = 10,
    hi: int = 105,
) -> pd.DataFrame:
    """Keep rows where Age is NA or lo ≤ Age ≤ hi."""
    mask = tx[age_col].isna() | ((tx[age_col] >= lo) & (tx[age_col] <= hi))
    return tx.loc[mask].reset_index(drop=True)
