# python/cli/customers.py
from pathlib import Path
import argparse
import pandas as pd  # for to_numeric/map

from pipeline.io import load_cfg, read_csv_str, write_parquet
from pipeline.customers.name_last_name import clean_customer_name_fields
from pipeline.customers.city_names import clean_city_series  # now no fill_unknown arg
from pipeline.customers.shopuserid import build_id_remap, apply_id_remap
from pipeline.customers.city_rep import enrich_and_dedup_customers
from pipeline.customers.ssn import derive_gender_age, filter_age_range

COUNTRY_MAP = {58: "Denmark", 205: "Sweden", 160: "Norway", 72: "Finland"}

def run(cfg_path: str, fill_unknown: str | None = None) -> None:
    cfg = load_cfg(cfg_path)
    ext = Path(cfg["external"])
    out_dir = Path(cfg.get("processed_staging", cfg["processed"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    customers = read_csv_str(ext / "customers.csv")

    customers = clean_customer_name_fields(customers)
    if "invoiceCity" in customers.columns:
        customers["invoiceCity"] = clean_city_series(customers["invoiceCity"])

    

    remap = build_id_remap(customers)
    customers = apply_id_remap(customers, remap)

    customers = enrich_and_dedup_customers(customers, id_col="shopUserId", city_col="invoiceCity")

    num_id = pd.to_numeric(customers["invoiceCountryId"], errors="coerce")
    customers["Country"] = num_id.map(COUNTRY_MAP).fillna(customers["invoiceCountryId"]).astype(str)

    customers = derive_gender_age(customers, ssn_col="invoiceSSN", country_col="invoiceCountryId",
                                  gender_col="Gender", age_col="Age")
    customers = filter_age_range(customers, age_col="Age", lo=10, hi=105)

    write_parquet(customers, out_dir / "customers_clean.parquet")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/base.yaml")
    p.add_argument("--fill-unknown", default=None, help='(unused now)')
    args = p.parse_args()
    run(args.cfg, args.fill_unknown)