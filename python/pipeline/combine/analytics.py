from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence
import pandas as pd


# ---------- Generic helpers ----------

def _norm_str(s: pd.Series, unknown: str = "Unknown") -> pd.Series:
    s = s.astype("string[python]")
    s = s.where(~s.isna(), unknown)
    return s.str.strip().fillna(unknown)


def _season_from_month(m: int) -> str:
    if m in (12, 1, 2):   return "Winter"
    if m in (3, 4, 5):    return "Spring"
    if m in (6, 7, 8):    return "Summer"
    return "Autumn"


def _ensure_cols(
    df: pd.DataFrame,
    created_col: str = "created",
    quantity_col: str = "quantity",
    quantity_default: int = 1,
) -> pd.DataFrame:
    """Return a copy with 'created' as datetime and 'quantity' as int64 (default=1)."""
    out = df.copy()
    out[created_col] = pd.to_datetime(out.get(created_col), errors="coerce")
    q = pd.to_numeric(out.get(quantity_col), errors="coerce")
    out[quantity_col] = q.fillna(quantity_default).astype("int64")
    return out


def _season_label_from_created(created: pd.Series) -> pd.Series:
    m = created.dt.month
    y = created.dt.year
    season = m.map(_season_from_month).fillna("Unknown")
    season_year = y - (m <= 2).astype(int)
    return (season + " " + season_year.astype(str)).astype("string[python]")


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    # align index
    return pd.Series(pd.NA, index=df.index)


def _explode_unique_str_list(
    s: pd.Series, sep: str
) -> pd.Series:
    """
    Split by sep, strip, drop blanks, and dedupe while preserving order.
    Returns Series of python lists (not exploded).
    """
    def _split(parts: list[str] | None) -> list[str]:
        if not parts:
            return []
        seen = set()
        out = []
        for p in parts:
            p = (p or "").strip()
            if not p:
                continue
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    return (
        s.astype("string[python]")
         .where(~s.isna(), None)
         .str.split(sep)
         .apply(_split)
    )


def _group_rank(df: pd.DataFrame, by: Sequence[str], value_col: str, rank_col: str = "rank") -> pd.DataFrame:
    df = df.sort_values(list(by) + [value_col], ascending=[True]*len(by) + [False])
    df[rank_col] = df.groupby(list(by))[value_col].rank(method="first", ascending=False).astype(int)
    return df


def _representative_name_brand(
    df: pd.DataFrame,
    keys: Sequence[str],
    name_col: str = "name",
    brand_col: str = "brand",
    qty_col: str = "quantity",
) -> pd.DataFrame:
    """
    For each (keys), pick (name, brand) with highest qty; tie-break by name, brand.
    Returns a frame with keys + rep_name, rep_brand.
    """
    tmp = (
        df.groupby(list(keys) + [name_col, brand_col], dropna=False)[qty_col]
          .sum().reset_index(name="qty")
          .sort_values(list(keys) + ["qty", name_col, brand_col],
                       ascending=[True]*len(keys) + [False, True, True])
          .drop_duplicates(list(keys))
          .rename(columns={name_col: "rep_name", brand_col: "rep_brand"})
    )
    return tmp[list(keys) + ["rep_name", "rep_brand"]]


# ---------- Domain helpers ----------

def _prepare_common_tx(tx_items: pd.DataFrame) -> pd.DataFrame:
    """created, quantity, country normalized; adds season_label."""
    tx = _ensure_cols(tx_items)
    tx["country"] = _norm_str(tx.get("country"))
    tx["season_label"] = _season_label_from_created(tx["created"])
    return tx


def _prepare_name_and_brand(tx: pd.DataFrame) -> pd.DataFrame:
    """Add normalized 'name' and 'brand' columns using common candidate names."""
    name_base = _first_existing(tx, ["name", "productName", "title", "product_name"])
    tx["name"] = _norm_str(name_base)
    tx["brand"] = _norm_str(tx.get("brand"))
    return tx


# ---------- Analytics ----------

def build_top_categories_by_season(
    tx_items: pd.DataFrame,
    category_sep: str = ",",
    keep_top_n: int = 10,
) -> pd.DataFrame:
    tx = _prepare_common_tx(tx_items)
    cat_lists = _explode_unique_str_list(tx.get("category"), category_sep)
    tx_exp = tx.loc[cat_lists.index, ["country", "season_label", "quantity"]].copy()
    tx_exp["category"] = _norm_str(cat_lists.explode().reset_index(drop=True))
    tx_exp = tx_exp.dropna(subset=["category"])  # after norm: removes explicit <NA>

    agg = (
        tx_exp.groupby(["country", "season_label", "category"], dropna=False)["quantity"]
              .sum().reset_index(name="count").astype({"count": "int64"})
    )
    agg = _group_rank(agg, ["country", "season_label"], "count")
    out = (agg[agg["rank"] <= keep_top_n]
           .sort_values(["country", "season_label", "rank"])
           .reset_index(drop=True))
    return out[["country", "season_label", "category", "count", "rank"]]


def build_top_groupids_by_season(
    tx_items: pd.DataFrame,
    keep_top_n: int = 100,
) -> pd.DataFrame:
    tx = _prepare_common_tx(tx_items)
    tx["groupId"] = _norm_str(tx.get("groupId"))
    tx = _prepare_name_and_brand(tx)

    gid_agg = (
        tx.groupby(["country", "season_label", "groupId"], dropna=False)["quantity"]
          .sum().reset_index(name="count").astype({"count": "int64"})
    )
    rep_meta = _representative_name_brand(
        tx, keys=["country", "season_label", "groupId"]
    )

    gid_agg = gid_agg.merge(rep_meta, on=["country", "season_label", "groupId"], how="left")
    gid_agg = _group_rank(gid_agg, ["country", "season_label"], "count")

    top = (gid_agg[gid_agg["rank"] <= keep_top_n]
           .sort_values(["country", "season_label", "rank"])
           .rename(columns={"groupId": "value", "rep_name": "name", "rep_brand": "brand"})
           .reset_index(drop=True))
    return top[["country", "season_label", "value", "name", "brand", "count", "rank"]]


def build_top_repurchase_groupids_by_country_unique_days(
    tx_items: pd.DataFrame,
    unique_days_threshold: int = 1,
    keep_top_n: int = 100,
) -> pd.DataFrame:
    tx = _ensure_cols(tx_items)  # created, quantity
    tx = tx.dropna(subset=["created"])
    tx["purchase_date"] = tx["created"].dt.normalize()

    def _safe(col: str) -> pd.Series:
        return tx[col] if col in tx.columns else pd.Series(pd.NA, index=tx.index)

    tx["country"]     = _norm_str(_safe("country"))
    tx["groupId"]     = _norm_str(_safe("groupId"))
    tx["customer_id"] = _norm_str(_safe("customer_id"))

    tx = _prepare_name_and_brand(tx)

    # Count unique purchase days per (country, groupId, customer_id)
    per_cust = (
        tx.groupby(["country", "groupId", "customer_id"], dropna=False)["purchase_date"]
          .nunique().reset_index(name="unique_days")
    )
    eligible = per_cust[per_cust["unique_days"] > unique_days_threshold]

    rep_counts = (
        eligible.groupby(["country", "groupId"], dropna=False)["customer_id"]
                .nunique().reset_index(name="repurchasers").astype({"repurchasers": "int64"})
    )

    rep_keys = eligible[["country", "groupId", "customer_id"]]
    tx_rep = tx.merge(rep_keys, on=["country", "groupId", "customer_id"], how="inner")
    rep_meta = _representative_name_brand(tx_rep, keys=["country", "groupId"])

    out = rep_counts.merge(rep_meta, on=["country", "groupId"], how="left")
    out = _group_rank(out, ["country"], "repurchasers")

    top = (out[out["rank"] <= keep_top_n]
           .sort_values(["country", "rank"])
           .rename(columns={"groupId": "value", "rep_name": "name", "rep_brand": "brand"})
           .reset_index(drop=True))
    return top[["country", "value", "name", "brand", "repurchasers", "rank"]]


def build_top_brands_by_country(tx_items: pd.DataFrame, keep_top_n: int = 10) -> pd.DataFrame:
    tx = _ensure_cols(tx_items)
    tx["country"] = _norm_str(tx.get("country"))
    tx["brand"]   = _norm_str(tx.get("brand"))

    bad = {"unknown", "na", ""}
    tx = tx[~tx["brand"].str.lower().isin(bad)]

    agg = (
        tx.groupby(["country", "brand"], dropna=False)["quantity"]
          .sum().reset_index(name="count").astype({"count": "int64"})
    )
    agg = _group_rank(agg, ["country"], "count")
    out = (agg[agg["rank"] <= keep_top_n]
           .sort_values(["country", "rank"])
           .reset_index(drop=True))
    return out[["country", "brand", "count", "rank"]]


def bucket_return_days(d: int) -> str | pd.NA:
    if pd.isna(d) or d <= 0: return pd.NA
    if d <= 7:   return "week 1"
    if d <= 14:  return "week 2"
    if d <= 21:  return "week 3"
    if d <= 30:  return "1 month"
    # Months 2..12 in 30-day buckets
    for m in range(2, 13):
        if (30*(m-1) + 1) <= d <= 30*m:
            return f"{m} months"
    return "> 1 year" if d > 365 else pd.NA


def count_return_buckets(tx_items: pd.DataFrame) -> pd.DataFrame:
    tx = tx_items.copy()
    tx["created"] = pd.to_datetime(tx.get("created"), errors="coerce")
    tx = tx.dropna(subset=["created", "customer_id"])
    tx["purchase_date"] = tx["created"].dt.date

    uniq = tx[["customer_id", "purchase_date"]].drop_duplicates()
    first_date = uniq.groupby("customer_id")["purchase_date"].min().rename("first_date")

    tmp = uniq.merge(first_date, on="customer_id")
    tmp = tmp[tmp["purchase_date"] > tmp["first_date"]]

    first_return = tmp.groupby("customer_id")["purchase_date"].min().rename("first_return_date")
    timeline = first_date.to_frame().merge(first_return, left_index=True, right_index=True, how="left")

    days = (pd.to_datetime(timeline["first_return_date"]) - pd.to_datetime(timeline["first_date"])).dt.days
    timeline["bucket"] = days.apply(bucket_return_days)

    ret = timeline.dropna(subset=["bucket"])
    counts = (
        ret.groupby("bucket").size().reset_index(name="customers")
           .sort_values("customers", ascending=False).reset_index(drop=True)
    )

    order = (["week 1", "week 2", "week 3", "1 month"]
             + [f"{m} months" for m in range(2, 13)] + ["> 1 year"])
    order_map = {b: i for i, b in enumerate(order)}
    counts["order"] = counts["bucket"].map(order_map)
    counts = counts.sort_values("order").drop(columns="order").reset_index(drop=True)
    return counts



def run_analytics(output_dir: Path) -> None:
    tx_items = pd.read_parquet(output_dir / "order_items.parquet")

    build_top_categories_by_season(tx_items).to_parquet(
        output_dir / "top_categories_by_season.parquet", index=False
    )
    build_top_groupids_by_season(tx_items).to_parquet(
        output_dir / "top_groupids_by_season.parquet", index=False
    )
    build_top_repurchase_groupids_by_country_unique_days(tx_items).to_parquet(
        output_dir / "top_repurchase_groupids_by_country.parquet", index=False
    )
    build_top_brands_by_country(tx_items).to_parquet(
        output_dir / "top_brands_by_country.parquet", index=False
    )
    count_return_buckets(tx_items).to_parquet(
        output_dir / "return_buckets_overall.parquet", index=False
    )
