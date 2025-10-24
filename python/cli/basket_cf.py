# /workspace/python/cli/basket_cf.py
import argparse
import yaml
from pathlib import Path
from pipeline.articles_for_recs.basket_cf import run

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--out", default="basket_completion.parquet")
    p.add_argument("--q-low", type=float, default=0.5)
    p.add_argument("--q-high", type=float, default=0.96)
    p.add_argument("--min-support", type=int, default=12)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--test-size", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args, _ = p.parse_known_args()

    with Path(args.cfg).open("r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["processed"])
    if not processed_dir.is_absolute():
        processed_dir = Path.cwd() / processed_dir

    run(
        processed_dir=processed_dir,
        out_filename=args.out,
        item_freq_q_low=args.q_low,
        item_freq_q_high=args.q_high,
        min_support=args.min_support,
        topk=args.topk,
        test_size=args.test_size,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
