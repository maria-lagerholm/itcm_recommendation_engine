# python/cli/lift.py
import argparse
from pathlib import Path

try:
    from pipeline.io import load_cfg
except ImportError:
    import yaml
    def load_cfg(p):
        with open(p, "r") as f:
            return yaml.safe_load(f)

from pipeline.recs.lift import run as run_core

def main():
    ap = argparse.ArgumentParser(description="Build Top-K complements from association rules.")
    ap.add_argument("-c", "--config", "--cfg", dest="cfg_path", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg_path)
    processed    = Path(cfg["processed"]).expanduser().resolve()
    transactions = cfg.get("transactions", "order_items.parquet")
    available    = cfg.get("available",    "articles_for_recs.parquet")
    output       = cfg.get("output",       "pair_complements.parquet")
    min_support  = float(cfg.get("min_support", 0.001))
    min_conf     = float(cfg.get("min_confidence", 0.10))
    lower_q      = float(cfg.get("lower_q", 0.50))
    upper_q      = float(cfg.get("upper_q", 0.97))
    top_k        = int(cfg.get("top_k", 10))

    _ = run_core(
        processed_dir=processed,
        transactions=transactions,
        available=available,
        output=output,
        min_support=min_support,
        min_confidence=min_conf,
        lower_q=lower_q,
        upper_q=upper_q,
        top_k=top_k,
    )

if __name__ == "__main__":
    main()
