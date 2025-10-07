# python/pipeline/articles_for_recs/clean.py

from pathlib import Path
import pandas as pd

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

def merge_colors(series):
    colors = [str(c).strip() for c in series if pd.notna(c) and str(c).strip().lower() not in {"", "unknown", "nan", "none"}]
    return list(sorted(set(colors))) if colors else []

def run(external_dir: Path, processed_dir: Path) -> None:
    full_articles = pd.read_csv(external_dir.joinpath("products.csv"), dtype="string")
    articles_clean = pd.read_parquet(processed_dir.joinpath("articles_clean.parquet")).query("forSale.notna()")
    articles = articles_clean.drop(columns=COLS_TO_DROP, errors="ignore").copy()
    articles = articles.merge(full_articles[['sku'] + COLS_TO_ADD], on="sku", how="left")
    articles['color'] = articles['color'].apply(dedup_color)
    articles = articles.sort_values("sku")
    agg_map = {col: (merge_colors if col == "color" else "first") for col in articles.columns if col != "sku"}
    articles = articles.groupby("groupId", as_index=False).agg(agg_map)
    articles = articles[articles["name"].notna()].reset_index(drop=True)
    articles.to_parquet(processed_dir.joinpath("articles_for_recs.parquet"), index=False)
