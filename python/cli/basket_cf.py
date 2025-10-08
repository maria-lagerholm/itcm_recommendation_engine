# /workspace/python/cli/basket_cf.py
import argparse
import yaml
from pathlib import Path
from pipeline.articles_for_recs.basket_cf import run

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--min-item-support", type=int, default=10)
    p.add_argument("--min-pair-support", type=int, default=5)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--thr", type=float, default=0.02)
    p.add_argument("--topk", type=int, default=10)
    args, _ = p.parse_known_args()

    with Path(args.cfg).open("r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["processed"])
    if not processed_dir.is_absolute():
        processed_dir = Path.cwd().joinpath(processed_dir)

    run(
        processed_dir=processed_dir,
        min_item_support=args["min_item_support"] if isinstance(args, dict) else args.min_item_support,
        min_pair_support=args["min_pair_support"] if isinstance(args, dict) else args.min_pair_support,
        k_neighbors=args["k"] if isinstance(args, dict) else args.k,
        score_threshold=args["thr"] if isinstance(args, dict) else args.thr,
        topk=args["topk"] if isinstance(args, dict) else args.topk,
    )

if __name__ == "__main__":
    main()