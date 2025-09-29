# python/pipeline/transactions/remove_known_bugs.py
from __future__ import annotations
import pandas as pd

DROP_COLS_DEFAULT = ("invoiceEmail", "orderLineId")

def prepare_article_lookup(
    articles: pd.DataFrame,
    *,
    group_col: str = "groupId",
    sku_col: str = "sku",
    keep_cols: tuple[str, ...] = ("category", "brand"),
) -> pd.DataFrame:
    """
    Build a unique (groupId, sku) lookup with the requested columns (e.g., category, brand).
    Raises if duplicates remain after de-dup.
    """
    cols = [group_col, sku_col, *keep_cols]
    a = articles[cols].drop_duplicates(subset=[group_col, sku_col])
    # safety: ensure uniqueness
    dup_mask = a.duplicated(subset=[group_col, sku_col], keep=False)
    if dup_mask.any():
        sample = a.loc[dup_mask, [group_col, sku_col]].head(5).to_dict("records")
        raise ValueError(f"Articles not unique on ({group_col}, {sku_col}). Examples: {sample}")
    return a

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
    """
    First-pass cleanup to remove known issues:
      - drop noisy columns (invoiceEmail, orderLineId)
      - join (category, brand) from articles on (groupId, sku)
      - drop rows where both category & brand are missing
      - keep tx with created >= min_created (migration guard)
    Returns a new DataFrame.
    """
    df = tx.copy()

    # 1) drop known-noise columns
    df = df.drop(columns=list(drop_cols), errors="ignore")

    # 2) align sku column name if needed
    if tx_sku_col != art_sku_col and art_sku_col in article_lookup.columns:
        art_lu = article_lookup.rename(columns={art_sku_col: tx_sku_col})
    else:
        art_lu = article_lookup

    # 3) merge category/brand (many tx → one article row)
    df = df.merge(art_lu, on=[group_col, tx_sku_col], how="left", validate="many_to_one")

    # 4) remove rows where both category & brand are missing
    df = df[~(df["category"].isna() & df["brand"].isna())]

    # 5) filter by created date (coerce first)
    if created_col in df.columns:
        df[created_col] = pd.to_datetime(df[created_col], errors="coerce")
        df = df[df[created_col] >= pd.to_datetime(min_created)]

    return df.reset_index(drop=True)
