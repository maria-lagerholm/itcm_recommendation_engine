# python/cli/transactions.py
from pathlib import Path
import argparse
import pandas as pd

from pipeline.io import load_cfg, write_parquet
from pipeline.transactions.remove_known_bugs import (
    prepare_article_lookup,
    remove_known_bugs,
)
from pipeline.transactions.currency import (
    fix_six_digit_prices,
    unify_price_to_sek,
)
from pipeline.transactions.customer_enrich import (
    enrich_tx_with_customers,
    filter_tx_by_age,
)
from pipeline.transactions.line_totals import (
    normalize_quantity_to_str,
    compute_and_filter_line_total_sek,
)
from pipeline.transactions.country_label import label_country


def run(cfg_path: str, min_created: str = "2024-06-01") -> None:
    cfg = load_cfg(cfg_path)
    processed = Path(cfg["processed"])
    out_dir = processed / "processed_staging" 
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Inputs from processed
    tx = pd.read_parquet(processed / "transactions_canonical.parquet")
    articles = pd.read_parquet(
        processed / "articles_clean.parquet",
        columns=["sku", "groupId", "category", "brand"],
    )
    customers = pd.read_parquet(processed / "customers_clean.parquet")

    # 2) Known-bug cleanup + article join + date filter
    a_lu = prepare_article_lookup(articles)
    tx = remove_known_bugs(
        tx,
        a_lu,
        group_col="groupId",
        tx_sku_col="sku",
        created_col="created",
        min_created=min_created,
    )

    # 3) Fix 6-digit raw prices before currency conversion
    tx = fix_six_digit_prices(tx, price_col="price")

    # 4) Convert to SEK using live API rates (adds currency_country + sek_rate)
    tx = unify_price_to_sek(
        tx,
        price_col="price",
        currency_id_col="currencyId",
        out_col="price_sek",
        add_cols=True,
    )

    # 5) Country label from currency_country, then drop currency_country
    tx = label_country(tx, src_col="currency_country", out_col="country", drop_src=True)

    # 6) Enrich with customer demographics
    tx = enrich_tx_with_customers(
        tx, customers, id_col="shopUserId", customer_cols=("Age", "Gender")
    )

    # 7) Filter by plausible age range
    tx = filter_tx_by_age(tx, age_col="Age", lo=10, hi=105)

    # 8) Normalize quantity (round → Int64 → string) to mirror notebook
    tx = normalize_quantity_to_str(tx, quantity_col="quantity")

    # 9) Compute line total in SEK and drop NA/zero totals
    tx = compute_and_filter_line_total_sek(
        tx,
        price_col="price_sek",
        quantity_col="quantity",
        out_col="line_total_sek",
    )
    # ensure price is a consistent numeric dtype (not object mix)
    tx["price"] = pd.to_numeric(tx["price"], errors="coerce").astype("Float64")

    # 10) Output
    write_parquet(tx, out_dir / "transactions_clean.parquet")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/base.yaml")
    p.add_argument("--min-created", default="2024-06-01")
    args = p.parse_args()
    run(args.cfg, args.min_created)
