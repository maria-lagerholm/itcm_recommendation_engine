# python/pipeline/transactions/remove_known_bugs.py
from __future__ import annotations
import pandas as pd

#------constants------
DROP_COLS_DEFAULT = ("invoiceEmail", "orderLineId")

#------prepare article lookup------
def prepare_article_lookup(
    articles: pd.DataFrame,
    *,
    group_col: str = "groupId",
    sku_col: str = "sku",
    keep_cols: tuple[str, ...] = ("category", "brand"),
) -> pd.DataFrame:
    cols = [group_col, sku_col, *keep_cols]
    a = articles[cols].drop_duplicates(subset=[group_col, sku_col])
    dup_mask = a.duplicated(subset=[group_col, sku_col], keep=False)
    if dup_mask.any():
        sample = a.loc[dup_mask, [group_col, sku_col]].head(5).to_dict("records")
        raise ValueError(f"Articles not unique on ({group_col}, {sku_col}). Examples: {sample}")
    return a

#------remove known bugs------
def remove_known_bugs(
    tx: pd.DataFrame,
    article_lookup: pd.DataFrame,
    *,
    group_col: str = "groupId",
    tx_sku_col: str = "sku",
    art_sku_col: str = "sku",
    created_col: str = "created",
    min_created: str = "2024-06-01",
    drop_cols: tuple[str, ...] = DROP_COLS_DEFAULT,
) -> pd.DataFrame:
    df = tx.copy()
    df = df.drop(columns=list(drop_cols), errors="ignore")
    if tx_sku_col != art_sku_col and art_sku_col in article_lookup.columns:
        art_lu = article_lookup.rename(columns={art_sku_col: tx_sku_col})
    else:
        art_lu = article_lookup
    df = df.merge(art_lu, on=[group_col, tx_sku_col], how="left", validate="many_to_one")
    df = df[~(df["category"].isna() & df["brand"].isna())]
    if created_col in df.columns:
        df[created_col] = pd.to_datetime(df[created_col], errors="coerce")
        df = df[df[created_col] >= pd.to_datetime(min_created)]
    return df.reset_index(drop=True)
