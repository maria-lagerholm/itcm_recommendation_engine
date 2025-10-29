# python/cli/hybrid.py
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pipeline.recs.hybrid import build_hybrid, make_topk_hybrid_parquet


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build hybrid recommendations and export top-K parquet")
    p.add_argument("--cfg", default="configs/base.yaml")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--basket", default="basket_completion.parquet")
    p.add_argument("--pair", default="pair_complements.parquet")
    p.add_argument("--semantic", default="semantic_similarity_recs.parquet")
    p.add_argument("--out", default="hybrid_pairs.parquet")
    p.add_argument("--w-basket", type=float, default=10.0)
    p.add_argument("--w-pair", type=float, default=1.0)
    p.add_argument("--w-semantic", type=float, default=0.1)
    return p.parse_args(argv)


def load_paths(cfg_path: Path) -> Path:
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    return Path(cfg.get("processed", "data/processed"))


def main(argv=None) -> int:
    args = parse_args(argv)
    processed_dir = load_paths(Path(args.cfg))
    weights = {"score_basket": args.w_basket, "score_pair": args.w_pair, "score_semantic": args.w_semantic}

    hybrid = build_hybrid(
        processed_dir=processed_dir,
        weights=weights,
        inputs=(args.basket, args.pair, args.semantic),
    )

    make_topk_hybrid_parquet(
        df=hybrid,
        processed_dir=processed_dir,
        out_filename=args.out,
        k=args.k,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())