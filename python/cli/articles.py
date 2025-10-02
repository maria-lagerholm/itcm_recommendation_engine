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


def run(cfg_path: str) -> None:
    # Load configuration and set up paths
    cfg = load_cfg(cfg_path)
    external = Path(cfg["external"])
    processed = Path(cfg["processed"])
    staging = Path(cfg.get("processed_staging", processed / "processed_staging"))
    staging.mkdir(parents=True, exist_ok=True)

    # Load raw articles data
    articles = pd.read_csv(external / "products.csv", dtype="string")

    # Data cleaning: remove noise columns and rows with all price columns missing
    articles = drop_noise_columns(articles)
    articles = remove_rows_all_prices_na(articles)

    # Normalize category columns
    articles, _ = normalize_categories(articles)

    # Normalize brand columns
    articles, _ = normalize_brands(
        articles,
        brand_col="brand",
        brand_id_col="brandId",
        fill_unknown_text="unknown",
        add_missing_flag=True,        # Place brand_missing flag after brandId
        backfill_brand_from_id=True,  # Fill brand text from brandId if possible
    )

    # Fill priceSEK using EUR/NOK/DKK columns and apply SKU-specific overrides
    overrides = {
        "270607-5254": 1310,
        "270534-03xl": 419,
    }
    articles, _ = fill_priceSEK_no_decimals(
        articles,
        overrides_priceSEK=overrides,
    )

    # Write outputs to staging directory
    out_dir = staging
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write full cleaned dataset
    articles.to_parquet(out_dir / "articles_clean.parquet", index=False)

    # Write subset of articles available for sale
    articles_for_sale = articles[articles["forSale"].notna()].copy()
    articles_for_sale.to_parquet(out_dir / "articles_for_sale.parquet", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/base.yaml")
    args = parser.parse_args()
    run(args.cfg)
