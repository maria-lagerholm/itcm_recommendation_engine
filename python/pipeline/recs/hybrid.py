# python/pipeline/recs/hybrid.py
from __future__ import annotations

import pandas as pd
from functools import reduce
from pathlib import Path
from typing import Dict, List, Tuple

TOP_PREFIX, SCORE_PREFIX = "Top ", "Score "


def wide_to_long(path: Path | str, score_col_name: str) -> pd.DataFrame:
    df = pd.read_parquet(path, dtype_backend="numpy_nullable")

    ranks: List[int] = sorted(
        int(c.split()[1])
        for c in df.columns
        if c.startswith(TOP_PREFIX) and len(c.split()) == 2 and c.split()[1].isdigit()
    )

    long_parts: List[pd.DataFrame] = []
    for r in ranks:
        top_col = f"{TOP_PREFIX}{r}"
        score_col = f"{SCORE_PREFIX}{r}"
        if top_col not in df.columns or score_col not in df.columns:
            continue

        part = (
            df[["Product ID", top_col, score_col]]
            .rename(columns={"Product ID": "product_id", top_col: "rec_id", score_col: score_col_name})
            .copy()
        )
        part["product_id"] = part["product_id"].astype("string")
        part["rec_id"]     = part["rec_id"].astype("string")
        part = part.dropna(subset=["product_id", "rec_id"])

        if part.empty:           # ← guard: skip empties
            continue
        long_parts.append(part)

    if not long_parts:
        return pd.DataFrame(columns=["product_id", "rec_id", score_col_name]).astype(
            {"product_id": "string", "rec_id": "string"}
        )

    parts = [p for p in long_parts if not p.empty]   # ← exclude empties before concat
    if not parts:
        return pd.DataFrame(columns=["product_id", "rec_id", score_col_name]).astype(
            {"product_id": "string", "rec_id": "string"}
        )

    return pd.concat(parts, ignore_index=True)



def build_hybrid(
    processed_dir: Path,
    weights: Dict[str, float] | None = None,
    inputs: Tuple[str, str, str] = (
        "basket_completion.parquet",
        "pair_complements.parquet",
        "semantic_similarity_recs.parquet",
    ),
) -> pd.DataFrame:
    """Load the three sources from <processed_dir>, merge (outer) on keys,
    and compute the weighted hybrid score.
    """
    if weights is None:
        weights = {"score_basket": 10.0, "score_pair": 1.0, "score_semantic": 0.1}

    path_basket, path_pair, path_semantic = (processed_dir / f for f in inputs)

    basket_long = wide_to_long(path_basket, "score_basket")
    pair_long = wide_to_long(path_pair, "score_pair")
    semantic_long = wide_to_long(path_semantic, "score_semantic")

    dfs = [basket_long, pair_long, semantic_long]
    hybrid = reduce(
        lambda left, right: pd.merge(
            left, right, on=["product_id", "rec_id"], how="outer"
        ),
        dfs,
    )

    hybrid["product_id"] = hybrid["product_id"].astype("string")
    hybrid["rec_id"] = hybrid["rec_id"].astype("string")
    hybrid = hybrid.dropna(subset=["product_id", "rec_id"]).copy()

    # Ensure numeric and compute weighted sum with missing -> 0
    for c in weights:
        if c in hybrid.columns:
            hybrid[c] = pd.to_numeric(hybrid[c], errors="coerce")

    hybrid["hybrid_score"] = sum(
        hybrid.get(c, 0).fillna(0) * w for c, w in weights.items()
    )

    return hybrid


def make_topk_hybrid_parquet(
    df: pd.DataFrame,
    processed_dir: Path,
    out_filename: str = "hybrid_pairs.parquet",
    k: int = 10,
) -> Path:
    """
    For each product_id, find the top-k recommendations by hybrid_score
    """
    out_path = processed_dir / out_filename

    df_sorted = df.sort_values(["product_id", "hybrid_score"], ascending=[True, False])

    def topk(group: pd.DataFrame) -> pd.Series:
        top = group.head(k).reset_index(drop=True)
        return pd.Series(
            {
                **{"Product ID": group.name},
                **{
                    f"Top {i+1}": str(top.loc[i, "rec_id"]) if i < len(top) else None
                    for i in range(k)
                },
                **{
                    f"Score {i+1}": float(top.loc[i, "hybrid_score"]) if i < len(top) else None
                    for i in range(k)
                },
            }
        )

    topk_df = (
        df_sorted.groupby("product_id", group_keys=False)
        .apply(topk, include_groups=False)
        .reset_index(drop=True)
    )

    topk_df.to_parquet(out_path, index=False)
    return out_path