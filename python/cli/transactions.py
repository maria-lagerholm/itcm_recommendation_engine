# python/cli/transactions.py
# Reads customers_canonical (from processed) + raw transactions (from external),
# canonicalizes IDs, assigns representative city per user,
# and writes transactions_canonical.parquet to processed_staging (or processed).

from pathlib import Path
import argparse
import pandas as pd

from pipeline.io import load_cfg, read_csv_str, write_parquet
from pipeline.customers.shopuserid import build_id_remap, apply_id_remap
from pipeline.customers.city_rep import build_shopuser_city, assign_city_to_transactions

def run(cfg_path: str) -> None:
    cfg = load_cfg(cfg_path)
    ext = Path(cfg["external"])
    processed = Path(cfg["processed"])
    out_dir = Path(cfg.get("processed_staging", processed))  # prefer staging

    # 1) Inputs (customers already cleaned/canonical by customers.py)
    customers = pd.read_parquet(processed / "customers_canonical.parquet")
    tx = read_csv_str(ext / "transactions.csv")

    # 2) Canonicalize transaction IDs to match customers
    remap = build_id_remap(customers)        # Series: id -> canonical_id
    tx = apply_id_remap(tx, remap)

    # 3) Assign representative city per user to transactions
    cust_city = build_shopuser_city(customers, id_col="shopUserId", city_col="invoiceCity")
    tx = assign_city_to_transactions(tx, cust_city, id_col="shopUserId",
                                     city_col="invoiceCity", unknown_token="Unknown")

    # 4) Output
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(tx, out_dir / "transactions_canonical.parquet")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/base.yaml")
    args = p.parse_args()
    run(args.cfg)