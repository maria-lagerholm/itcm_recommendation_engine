# python/pipeline/recs/iicf_ease.py
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from cornac.data import Dataset
from cornac.models.ease import EASE


BAD_IDS: set[str] = {"12025DK", "12025FI", "12025NO", "12025SE", "970300", "459978"}


def load_filtered_transactions(
    processed_dir: Path,
    bad_ids: Iterable[str] = BAD_IDS,
    cols: tuple[str, str, str] = ("shopUserId", "orderId", "groupId"),
) -> pd.DataFrame:
    """Load transactions, keep only items present in availability, and drop BAD_IDS."""
    trans_path = processed_dir / "transactions_clean.parquet"
    avail_path = processed_dir / "articles_for_recs.parquet"

    df = pd.read_parquet(trans_path, columns=list(cols))
    avail_df = pd.read_parquet(avail_path, columns=["groupId"])

    gid = df["groupId"].astype(str).str.strip()
    avail_ids = set(avail_df["groupId"].astype(str).str.strip().unique())

    return df.loc[gid.isin(avail_ids) & ~gid.isin(bad_ids)].reset_index(drop=True)


def make_user_item_pairs(
    df: pd.DataFrame,
    user_col: str = "shopUserId",
    order_col: str = "orderId",
    item_col: str = "groupId",
    pref_value: float = 1.0,
) -> pd.DataFrame:
    """Return unique (user, item) pairs with a binary preference column."""
    ui = (
        df[[user_col, order_col, item_col]]
        .drop_duplicates()                  # unique (user, order, item)
        .drop(columns=order_col)            # collapse orders
        .drop_duplicates()                  # unique (user, item)
        .copy()
    )
    ui["pref"] = float(pref_value)
    return ui


def product_pair_user_counts(
    pairs: pd.DataFrame,
    user_col: str = "shopUserId",
    item_col: str = "groupId",
) -> pd.DataFrame:
    """Count distinct users per unordered item pair."""
    ui = pairs[[user_col, item_col]].drop_duplicates()
    combos = (
        ui.groupby(user_col)[item_col]
        .apply(lambda s: list(combinations(sorted(s.unique()), 2)))
        .explode()
        .dropna()
        .reset_index(name="pair")
    )
    combos[[f"{item_col}_a", f"{item_col}_b"]] = pd.DataFrame(
        combos["pair"].tolist(), index=combos.index
    )
    combos = combos.drop(columns="pair")
    return (
        combos.drop_duplicates([user_col, f"{item_col}_a", f"{item_col}_b"])
        .groupby([f"{item_col}_a", f"{item_col}_b"])[user_col]
        .nunique()
        .reset_index(name="distinct_users")
        .sort_values("distinct_users", ascending=False)
        .reset_index(drop=True)
    )


def filter_pairs_by_popular_pairs(
    pairs: pd.DataFrame,
    product_pairs_df: pd.DataFrame,
    *,
    user_col: str = "shopUserId",
    item_col: str = "groupId",
    min_distinct_users: int = 25,
    require_min_items_per_user: int | None = 2,
) -> pd.DataFrame:
    """
    Keep items that appear in any pair seen by ≥ min_distinct_users; optionally
    enforce a per-user minimum number of distinct items.
    """
    qual = product_pairs_df.loc[
        product_pairs_df["distinct_users"] >= min_distinct_users,
        [f"{item_col}_a", f"{item_col}_b"],
    ]
    if qual.empty:
        return pairs.iloc[0:0].copy()

    allowed = pd.unique(
        pd.concat([qual[f"{item_col}_a"], qual[f"{item_col}_b"]], ignore_index=True)
    )
    filtered = pairs[pairs[item_col].isin(allowed)].copy()

    if require_min_items_per_user is not None:
        users_keep = (
            filtered[[user_col, item_col]]
            .drop_duplicates()
            .groupby(user_col)[item_col]
            .nunique()
            .loc[lambda s: s >= require_min_items_per_user]
            .index
        )
        filtered = filtered[filtered[user_col].isin(users_keep)].copy()

    return filtered


def filter_pairs_by_item_frequency(
    pairs: pd.DataFrame,
    item_col: str = "groupId",
    q_low: float = 0.0,
    q_high: float = 0.96,
    inclusive: str = "both",
) -> pd.DataFrame:
    """Quantile-trim items by frequency and return the filtered pairs."""
    gid = pairs[item_col].astype(str).str.strip()
    counts = gid.value_counts()
    low, high = counts.quantile([q_low, q_high])
    mask = gid.map(counts).between(low, high, inclusive=inclusive)
    return pairs.loc[mask].reset_index(drop=True)


def _to_uir(
    pairs: pd.DataFrame,
    user_col: str = "shopUserId",
    item_col: str = "groupId",
    pref_col: str = "pref",
):
    """Return (user, item, rating) triplets for Cornac."""
    return list(
        zip(
            pairs[user_col].astype(str),
            pairs[item_col].astype(str),
            pairs[pref_col].astype(float),
        )
    )


def build_ease_topk_wide(
    uir,
    rel_min: float = 0.50,
    k_min: int = 4,
    k_max: int = 10,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Train EASE and emit a wide Top-K dataframe with scores.
    - Keep positive neighbors ≥ rel_min * row_max
    - Require at least k_min neighbors; cap at k_max
    """
    train_set = Dataset.from_uir(uir)
    model = EASE(verbose=False)
    model.fit(train_set)

    item_ids = train_set.item_ids
    B = model.B.astype(np.float32, copy=True)
    np.fill_diagonal(B, np.nan)
    B_df = pd.DataFrame(B, index=item_ids, columns=item_ids)

    def _pack_row(s: pd.Series) -> pd.Series:
        v = s.dropna()
        if v.empty:
            return pd.Series({"Product ID": s.name})
        mx = v.max()
        v = v[(v > 0) & (v >= mx * rel_min)].nlargest(k_max)
        if len(v) < k_min:
            return pd.Series({"Product ID": s.name})
        row = {"Product ID": s.name}
        for i, (it, sc) in enumerate(v.items(), start=1):
            row[f"Top {i}"] = it
            row[f"Score {i}"] = float(sc)
        for j in range(len(v) + 1, k_max + 1):
            row[f"Top {j}"] = pd.NA
            row[f"Score {j}"] = np.nan
        return pd.Series(row)

    wide = B_df.apply(_pack_row, axis=1)

    must_have = [f"Top {i}" for i in range(1, k_min + 1)]
    wide = wide[wide[must_have].notna().all(axis=1)].reset_index(drop=True)

    cols = ["Product ID"] + [x for i in range(1, k_max + 1) for x in (f"Top {i}", f"Score {i}")]
    wide = wide.reindex(columns=cols)

    top_cols = [f"Top {i}" for i in range(1, k_max + 1)]
    score_cols = [f"Score {i}" for i in range(1, k_max + 1)]
    wide[top_cols] = wide[top_cols].astype("string")
    wide[score_cols] = wide[score_cols].astype("Float32")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(out_path, index=False)
    return wide


def run(
    processed_dir: Path,
    out_filename: str,
    min_distinct_users: int = 25,
    require_min_items_per_user: int | None = 2,
    item_freq_q_low: float = 0.0,
    item_freq_q_high: float = 0.96,
    rel_min: float = 0.50,
    k_min: int = 4,
    k_max: int = 10,
) -> Path:
    """Full pipeline: filter → pairs → co-occur filter → freq trim → train EASE → write parquet."""
    df = load_filtered_transactions(processed_dir)
    pairs = make_user_item_pairs(df)
    pair_counts = product_pair_user_counts(pairs)

    pairs = filter_pairs_by_popular_pairs(
        pairs,
        pair_counts,
        min_distinct_users=min_distinct_users,
        require_min_items_per_user=require_min_items_per_user,
    )

    pairs = filter_pairs_by_item_frequency(
        pairs, item_col="groupId", q_low=item_freq_q_low, q_high=item_freq_q_high
    )

    uir = _to_uir(pairs)
    out_path = processed_dir / out_filename

    _ = build_ease_topk_wide(
        uir,
        rel_min=rel_min,
        k_min=k_min,
        k_max=k_max,
        out_path=out_path,
    )
    return out_path
