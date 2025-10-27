# python/cli/iicf_ease.py
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.recs.iicf_ease import run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build EASE-based basket completion parquet.")
    p.add_argument("--processed-dir", type=Path, default=Path("/workspace/data/processed"))
    p.add_argument("--out-filename", type=str, default="basket_completion.parquet")
    p.add_argument("--min-distinct-users", type=int, default=25)
    p.add_argument("--require-min-items-per-user", type=int, default=2)
    p.add_argument("--item-freq-q-low", type=float, default=0.0)
    p.add_argument("--item-freq-q-high", type=float, default=0.96)
    p.add_argument("--rel-min", type=float, default=0.50)
    p.add_argument("--k-min", type=int, default=4)
    p.add_argument("--k-max", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ = run(
        processed_dir=args.processed_dir,
        out_filename=args.out_filename,
        min_distinct_users=args.min_distinct_users,
        require_min_items_per_user=args.require_min_items_per_user,
        item_freq_q_low=args.item_freq_q_low,
        item_freq_q_high=args.item_freq_q_high,
        rel_min=args.rel_min,
        k_min=args.k_min,
        k_max=args.k_max,
    )


if __name__ == "__main__":
    main()
