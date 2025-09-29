from __future__ import annotations
import pandas as pd

COUNTRY_MAP_ALPHA = {"SE": "Sweden", "DK": "Denmark", "FI": "Finland", "NO": "Norway"}

def label_country(
    df: pd.DataFrame,
    *,
    src_col: str = "currency_country",
    out_col: str = "country",
    drop_src: bool = True,
    mapping: dict[str, str] = COUNTRY_MAP_ALPHA,
) -> pd.DataFrame:
    """Map alpha country codes to names; optionally drop the source column."""
    out = df.copy()
    out[out_col] = out[src_col].map(mapping).fillna(out[src_col])
    if drop_src and src_col in out.columns:
        out = out.drop(columns=[src_col])
    return out
