# python/cli/articles_for_recs.py

import argparse
import yaml
from pathlib import Path
from pipeline.articles_for_recs.clean import run

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    args, unknown = p.parse_known_args()
    cfg_path = Path(args.cfg)
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    external_dir = Path(cfg["external"])
    processed_dir = Path(cfg["processed"])
    if not external_dir.is_absolute():
        external_dir = Path.cwd().joinpath(external_dir)
    if not processed_dir.is_absolute():
        processed_dir = Path.cwd().joinpath(processed_dir)
    run(external_dir=external_dir, processed_dir=processed_dir)

if __name__ == "__main__":
    main()