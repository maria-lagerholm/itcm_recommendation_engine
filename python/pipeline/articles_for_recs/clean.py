# python/pipeline/articles_for_recs/clean.py

from pathlib import Path
import pandas as pd
from collections.abc import Sequence

COLS_TO_DROP = ['priceEUR', 'priceNOK', 'priceDKK', 'forSale', 'sizeId', 'brandId', 'categoryId']
COLS_TO_ADD = ['description', 'color']

def dedup_color(val):
    if pd.isna(val):
        return val
    seen = set()
    tokens = [x.strip() for x in str(val).split(',')]
    deduped = []
    for token in tokens:
        if token and token not in seen:
            deduped.append(token)
            seen.add(token)
    return ','.join(deduped) if deduped else pd.NA

def merge_list(series: pd.Series):
    """
    Merge values across rows into a deduped, sorted list.
    - Flattens list/tuple elements
    - Accepts scalars (including comma-separated strings)
    - Drops empty/unknown/nan/none (case-insensitive)
    """
    keep = []
    for x in series:
        if pd.isna(x):
            continue
        if isinstance(x, Sequence) and not isinstance(x, str):
            for y in x:
                if pd.isna(y):
                    continue
                s = str(y).strip()
                if s.lower() in {"", "unknown", "nan", "none"}:
                    continue
                keep.append(s)
        else:
            parts = str(x).split(',')
            for y in parts:
                s = y.strip()
                if s.lower() in {"", "unknown", "nan", "none"}:
                    continue
                keep.append(s)
    return sorted(set(keep)) if keep else []

def run(external_dir: Path, processed_dir: Path) -> None:
    full_articles = pd.read_csv(external_dir.joinpath("products.csv"), dtype="string")
    articles_clean = pd.read_parquet(processed_dir.joinpath("articles_clean.parquet")).query("forSale.notna()")
    articles = articles_clean.drop(columns=COLS_TO_DROP, errors="ignore").copy()
    articles = articles.merge(full_articles[['sku'] + COLS_TO_ADD], on="sku", how="left")

    articles['color'] = articles['color'].apply(dedup_color)

    articles = articles.sort_values("sku")

    agg_map = {
        col: (merge_list if col in ("color", "size") else "first")
        for col in articles.columns
        if col not in ("sku", "groupId")
    }

    articles = articles.groupby("groupId", as_index=False).agg(agg_map)
    articles = articles[articles["name"].notna()].reset_index(drop=True)
    articles.to_parquet(processed_dir.joinpath("articles_for_recs.parquet"), index=False)
