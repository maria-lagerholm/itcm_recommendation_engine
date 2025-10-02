# cli/combine.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from pipeline.io import load_cfg
from pipeline.combine.build_json import export_country_json, split_nordics, NORDICS
from pipeline.combine.json_to_tables import build_tables_from_dir
from pipeline.combine.analytics import run_analytics

#------main-----
def run(cfg_path: str) -> None:
    cfg = load_cfg(cfg_path)
    processed = Path(cfg["processed"]).expanduser().resolve()
    out_dir = processed

    tx = pd.read_parquet(processed / "transactions_clean.parquet")
    art_path = processed / "articles_clean.parquet"
    articles = pd.read_parquet(art_path) if art_path.exists() else None

    #------build per-country JSON-----
    by_country = split_nordics(tx)
    for country in NORDICS:
        df = by_country.get(country)
        if df is not None and not df.empty:
            export_country_json(df, tx, country, out_dir=str(out_dir), articles=articles)

    #------flatten JSON to Parquet-----
    build_tables_from_dir(out_dir, out_dir)

    #------run analytics-----
    run_analytics(out_dir)

#------cli entrypoint-----
def main() -> None:
    ap = argparse.ArgumentParser(description="Export JSONs, flatten to Parquet, and compute analytics.")
    ap.add_argument("-c", "--config", "--cfg", dest="cfg_path", required=True)
    args = ap.parse_args()
    run(args.cfg_path)

if __name__ == "__main__":
    main()