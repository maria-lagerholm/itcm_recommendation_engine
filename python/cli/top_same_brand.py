# python/cli/recs/top_same_brand.py
import argparse
from pathlib import Path

try:
    from pipeline.io import load_cfg
except ImportError:
    import yaml
    def load_cfg(p):
        with open(p, "r") as f:
            return yaml.safe_load(f)

from pipeline.recs.top_same_brand import run as run_core

def main():
    ap = argparse.ArgumentParser(description="Build same-brand recommendations.")
    ap.add_argument("-c", "--config", "--cfg", dest="cfg_path", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg_path)
    processed    = Path(cfg["processed"]).expanduser().resolve()
    transactions = cfg.get("transactions", "transactions_clean.parquet")
    available    = cfg.get("available",    "articles_for_recs.parquet")
    output       = cfg.get("output",       "top_same_brand.parquet")
    min_recs     = int(cfg.get("min_recs", 4))
    max_recs     = int(cfg.get("max_recs", 10))

    run_core(
        processed_dir=processed,
        transactions=transactions,
        available=available,
        output=output,
        min_recs=min_recs,
        max_recs=max_recs,
    )

if __name__ == "__main__":
    main()
