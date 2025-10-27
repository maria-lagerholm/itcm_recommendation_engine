from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

from pipeline.io import load_cfg
from pipeline.articles.remove_known_bugs import (
    drop_noise_columns,
    remove_rows_all_prices_na,
)
from pipeline.articles.category import normalize_categories
from pipeline.articles.brand import normalize_brands
from pipeline.articles.price import fill_priceSEK_no_decimals
from pipeline.articles.audience import clean_audience

def run(cfg_path: str) -> None:
    cfg = load_cfg(cfg_path)
    external = Path(cfg["external"])
    processed = Path(cfg["processed"])

    articles = pd.read_csv(external / "products.csv", dtype="string")
    articles = drop_noise_columns(articles)
    articles = remove_rows_all_prices_na(articles)
    articles = normalize_categories(articles)
    articles = normalize_brands(articles)
    articles = clean_audience(articles)
    
    overrides = {
        "270607-5254": 1310,
        "270534-03xl": 419,
    }
    articles = fill_priceSEK_no_decimals(
        articles,
        overrides_priceSEK=overrides,
    )
    out_dir = processed
    articles.to_parquet(out_dir / "articles_clean.parquet", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/base.yaml")
    args = parser.parse_args()
    run(args.cfg)
