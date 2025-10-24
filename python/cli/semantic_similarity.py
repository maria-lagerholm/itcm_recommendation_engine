import argparse, yaml
from pathlib import Path
from pipeline.articles_for_recs.semantic_similarity import run

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--cos-min", type=float, default=0.60)
    p.add_argument("--min-price", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=1)
    args, _ = p.parse_known_args()

    with Path(args.cfg).open("r") as f:
        cfg = yaml.safe_load(f)
    processed_dir = Path(cfg["processed"])
    if not processed_dir.is_absolute():
        processed_dir = Path.cwd().joinpath(processed_dir)

    run(
        processed_dir=processed_dir,
        batch_size=args.batch_size,
        k=args.k,
        cos_min=args.cos_min,
        min_price=args.min_price,
        num_threads=args.threads,
    )

if __name__ == "__main__":
    main()
