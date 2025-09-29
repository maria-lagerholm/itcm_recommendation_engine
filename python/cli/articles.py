from pathlib import Path
import argparse
import pandas as pd

from pipeline.io import load_cfg, write_parquet
from pipeline.articles.remove_known_bugs import (
    drop_noise_columns,
    remove_rows_all_prices_na,
)

def run(cfg_path: str) -> None:
    cfg = load_cfg(cfg_path)
    external = Path(cfg["external"])
    processed = Path(cfg["processed"])
    out_dir = processed / "processed_staging"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw products as strings (match your notebook)
    articles = pd.read_csv(external / "products.csv", dtype="string")

    # Step 1: drop early noise columns
    articles = drop_noise_columns(articles)

    # Step 2: remove rows where all price cols are NA
    before = len(articles)
    articles = remove_rows_all_prices_na(articles)
    after = len(articles)
    print(f"Rows after removing all-prices-NA: {after} (was {before})")

    # Save staged output
    write_parquet(articles, out_dir / "articles_clean.parquet")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/base.yaml")
    args = p.parse_args()
    run(args.cfg)

# python/cli/articles.py (excerpt)
from pipeline.articles.name import fill_color_and_size_from_name, drop_unwanted_by_name

# ... after remove_rows_all_prices_na(...)
articles, stats = fill_color_and_size_from_name(articles)
print(
    f"Color replacements: {stats['color_replacements']} | "
    f"Size replacements: {stats['size_replacements']} | "
    f"Name clean replacements: {stats['name_clean_replacements']}"
)

articles, removed = drop_unwanted_by_name(articles)
articles = articles[articles["name"].notna()].reset_index(drop=True)
print(f"Rows removed due to unwanted phrases: {removed}")

# python/cli/articles.py (add import)
from pipeline.articles.color import normalize_colors, report_color_stats

# ...after remove_rows_all_prices_na(...)
articles, stats = normalize_colors(articles)

# minimal, readable log
print(f"Filled color from description: {stats['filled_color_from_description']}")
if stats['rare_merges']:
    print("Rare→base merges:", stats['rare_merges'])
if stats['reassigned_colorIds']:
    print("Collapsed colorIds per color:", stats['reassigned_colorIds'])

# optional deeper stats
# deep = report_color_stats(articles)
# print(deep)

# save
write_parquet(articles, out_dir / "articles_clean.parquet")


from pipeline.articles.fabric import enrich_fabrics

# ...
articles, fstats = enrich_fabrics(articles, desc_col="description")
print(f"Fabrics extracted for {fstats['with_any_fabric']}/{fstats['rows_scanned']} rows; "
      f"primary set for {fstats['with_primary']}.")

from pipeline.articles.category import normalize_categories

articles, cstats = normalize_categories(articles)
# Optionally log cstats if you want, but no prints are required.

from pipeline.articles.size import normalize_sizes

articles, size_stats = normalize_sizes(articles)
# optionally log size_stats
from pipeline.articles.brand import normalize_brands

articles, brand_stats = normalize_brands(articles)
# log brand_stats if you want a quick sanity report
