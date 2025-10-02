# pipeline/combine/analytics.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import itertools, math
from collections import Counter

import pandas as pd


# -------------------- shared helpers --------------------

def _norm_str(s: pd.Series) -> pd.Series:
    s = s.astype("string[python]")
    s = s.where(~s.isna(), "Unknown")
    return s.str.strip().fillna("Unknown")

def _season_from_month(m: int) -> str:
    if m in (12, 1, 2):   return "Winter"
    if m in (3, 4, 5):    return "Spring"
    if m in (6, 7, 8):    return "Summer"
    return "Autumn"


# ------------------ Top Categories by Season ------------------

def build_top_categories_by_season(
    tx_items: pd.DataFrame,
    category_sep: str = ",",
    keep_top_n: int = 10,
) -> pd.DataFrame:
    tx = tx_items.copy()
    tx["created"]  = pd.to_datetime(tx["created"], errors="coerce")
    tx["quantity"] = pd.to_numeric(tx.get("quantity"), errors="coerce").fillna(1).astype("int64")
    tx["country"]  = _norm_str(tx.get("country"))
    tx["category"] = _norm_str(tx.get("category"))
    m = tx["created"].dt.month
    y = tx["created"].dt.year
    season = m.map(_season_from_month).fillna("Unknown")
    season_year = y - (m <= 2).astype(int)
    tx["season_label"] = (season + " " + season_year.astype(str)).astype("string[python]")
    cat_lists = (
        tx["category"]
          .str.split(category_sep)
          .apply(lambda parts: list(dict.fromkeys([p.strip() for p in (parts or []) if p and p.strip()])))
    )
    tx_exp = tx.loc[cat_lists.index, ["country","season_label","quantity"]].copy()
    tx_exp["category"] = cat_lists
    tx_exp = tx_exp.explode("category", ignore_index=True)
    tx_exp["category"] = _norm_str(tx_exp["category"])
    agg = (
        tx_exp.groupby(["country","season_label","category"], dropna=False)["quantity"]
              .sum().reset_index(name="count")
    )
    agg["count"] = agg["count"].astype("int64")
    agg = agg.sort_values(["country","season_label","count"], ascending=[True, True, False])
    agg["rank"] = agg.groupby(["country","season_label"])["count"].rank(method="first", ascending=False).astype(int)
    top = (
        agg[agg["rank"] <= keep_top_n]
        .sort_values(["country","season_label","rank"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    return top[["country","season_label","category","count","rank"]]


# ------------------ Top Products (groupId) by Season ------------------

def build_top_groupids_by_season(
    tx_items: pd.DataFrame,
    keep_top_n: int = 100,
) -> pd.DataFrame:
    tx = tx_items.copy()
    tx["created"]  = pd.to_datetime(tx["created"], errors="coerce")
    tx["quantity"] = pd.to_numeric(tx.get("quantity"), errors="coerce").fillna(1).astype("int64")
    tx["country"] = _norm_str(tx.get("country"))
    tx["groupId"] = _norm_str(tx.get("groupId"))
    tx["brand"]   = _norm_str(tx.get("brand"))
    if   "name" in tx.columns:         base_name = tx["name"]
    elif "productName" in tx.columns:  base_name = tx["productName"]
    elif "title" in tx.columns:        base_name = tx["title"]
    elif "product_name" in tx.columns: base_name = tx["product_name"]
    else:                              base_name = pd.Series(pd.NA, index=tx.index)
    tx["name"] = _norm_str(base_name)
    m = tx["created"].dt.month
    y = tx["created"].dt.year
    season = m.map(_season_from_month).fillna("Unknown")
    season_year = y - (m <= 2).astype(int)
    tx["season_label"] = (season + " " + season_year.astype(str)).astype("string[python]")
    gid_agg = (
        tx.groupby(["country","season_label","groupId"], dropna=False)["quantity"]
          .sum().reset_index(name="count")
    ).astype({"count":"int64"})
    rep_meta = (
        tx.groupby(["country","season_label","groupId","name","brand"], dropna=False)["quantity"]
          .sum().reset_index(name="qty")
          .sort_values(["country","season_label","groupId","qty","name","brand"],
                       ascending=[True, True, True, False, True, True])
          .drop_duplicates(["country","season_label","groupId"])
          .rename(columns={"name":"rep_name","brand":"rep_brand"})
          [["country","season_label","groupId","rep_name","rep_brand"]]
    )
    gid_agg = gid_agg.merge(rep_meta, on=["country","season_label","groupId"], how="left")
    gid_agg = gid_agg.sort_values(["country","season_label","count"], ascending=[True, True, False])
    gid_agg["rank"] = gid_agg.groupby(["country","season_label"])["count"].rank(method="first", ascending=False).astype(int)
    top = (
        gid_agg[gid_agg["rank"] <= keep_top_n]
        .sort_values(["country","season_label","rank"])
        .rename(columns={"groupId":"value", "rep_name":"name", "rep_brand":"brand"})
        .reset_index(drop=True)
    )
    return top[["country","season_label","value","name","brand","count","rank"]]


# ------------- Top Repurchased Products by Country -------------

def build_top_repurchase_groupids_by_country_unique_days(
    tx_items: pd.DataFrame,
    unique_days_threshold: int = 1,
    keep_top_n: int = 100,
) -> pd.DataFrame:
    tx = tx_items.copy()
    tx["created"] = pd.to_datetime(tx.get("created"), errors="coerce")
    tx = tx.dropna(subset=["created"])
    tx["purchase_date"] = tx["created"].dt.normalize()
    if "quantity" in tx.columns:
        q = pd.to_numeric(tx["quantity"], errors="coerce")
        tx["quantity"] = q.fillna(1).astype("int64")
    else:
        tx["quantity"] = 1
    def _safe_series(colname: str) -> pd.Series:
        return tx[colname] if colname in tx.columns else pd.Series(pd.NA, index=tx.index)
    tx["country"]     = _norm_str(_safe_series("country"))
    tx["groupId"]     = _norm_str(_safe_series("groupId"))
    tx["customer_id"] = _norm_str(_safe_series("customer_id"))
    tx["brand"]       = _norm_str(_safe_series("brand"))
    for cand in ["name","productName","title","product_name"]:
        if cand in tx.columns:
            base_name = tx[cand]
            break
    else:
        base_name = pd.Series(pd.NA, index=tx.index)
    tx["name"] = _norm_str(base_name)
    per_cust = (
        tx.groupby(["country","groupId","customer_id"], dropna=False)
          .agg(unique_days=("purchase_date", pd.Series.nunique))
          .reset_index()
    )
    eligible = per_cust[per_cust["unique_days"] > unique_days_threshold].copy()
    rep_counts = (
        eligible.groupby(["country","groupId"], dropna=False)["customer_id"]
                .nunique().reset_index(name="repurchasers")
        .astype({"repurchasers":"int64"})
    )
    rep_keys = eligible[["country","groupId","customer_id"]]
    tx_rep = tx.merge(rep_keys, on=["country","groupId","customer_id"], how="inner")
    name_weight = (
        tx_rep.groupby(["country","groupId","name","brand"], dropna=False)["quantity"]
              .sum().reset_index(name="qty")
    )
    rep_meta = (
        name_weight.sort_values(["country","groupId","qty","name","brand"],
                                ascending=[True, True, False, True, True])
                   .drop_duplicates(["country","groupId"])
                   .rename(columns={"name":"rep_name", "brand":"rep_brand"})
                   [["country","groupId","rep_name","rep_brand"]]
    )
    out = rep_counts.merge(rep_meta, on=["country","groupId"], how="left")
    out = out.sort_values(["country","repurchasers"], ascending=[True, False])
    out["rank"] = out.groupby("country")["repurchasers"].rank(method="first", ascending=False).astype(int)
    top = (out[out["rank"] <= keep_top_n]
           .sort_values(["country","rank"])
           .rename(columns={"groupId":"value", "rep_name":"name", "rep_brand":"brand"})
           .reset_index(drop=True))
    return top[["country","value","name","brand","repurchasers","rank"]]


# -------------------- Pair Co-occurrences --------------------

def build_pair_cooccurrences(
    tx_items: pd.DataFrame,
    support_min: int = 40,
    log_lift_min: float = 0.5,
    max_pairs_per_item: int = 6,
) -> pd.DataFrame:
    df = tx_items.copy()
    df = df[["order_id","groupId","name","country"]].dropna(subset=["order_id","groupId"])
    df = df.astype({"order_id": "string", "groupId": "string"})
    sw = (
        df[df["country"] == "Sweden"].dropna(subset=["name"])
          .groupby("groupId")["name"].agg(lambda s: s.value_counts().index[0]).to_dict()
    )
    glob = (
        df.dropna(subset=["name"]).groupby("groupId")["name"]
          .agg(lambda s: s.value_counts().index[0]).to_dict()
    )
    name_map = {gid: sw.get(gid, glob.get(gid)) for gid in set(df["groupId"])}
    order_items = (
        df.drop_duplicates(["order_id","groupId"])
          .groupby("order_id")["groupId"].apply(list)
    )
    N = len(order_items)
    item_freq = Counter(itertools.chain.from_iterable(map(set, order_items)))
    pair_counts = Counter()
    for items in order_items:
        sitems = sorted(set(items))
        for a, b in itertools.combinations(sitems, 2):
            pair_counts[(a, b)] += 1
    rows = []
    for (a, b), tf in pair_counts.items():
        if tf < support_min:
            continue
        dfa, dfb = item_freq[a], item_freq[b]
        expected = (dfa * dfb) / N
        if expected <= 0:
            continue
        lift = tf / expected
        log_lift = math.log(lift)
        if log_lift <= log_lift_min:
            continue
        score = math.log(1 + tf) / math.sqrt(dfa * dfb)
        rows.append((a, name_map.get(a), b, name_map.get(b), tf, dfa, dfb, lift, log_lift, score))
    res = pd.DataFrame(rows, columns=[
        "item_a","item_a_name","item_b","item_b_name",
        "pair_orders","itemA_orders","itemB_orders","lift","log_lift","score"
    ])
    if res.empty:
        return pd.DataFrame(columns=["Rank by Affinity Strength","Product A ID","Product A","Product B ID","Product B"])
    res_sorted = res.sort_values("score", ascending=False).reset_index(drop=True)
    kept_rows, appear_counts = [], Counter()
    for _, r in res_sorted.iterrows():
        a, b = r["item_a"], r["item_b"]
        if appear_counts[a] < max_pairs_per_item and appear_counts[b] < max_pairs_per_item:
            kept_rows.append(r)
            appear_counts[a] += 1
            appear_counts[b] += 1
    final = pd.DataFrame(kept_rows).reset_index(drop=True)
    final["rank"] = final["score"].rank(method="dense", ascending=False).astype(int)
    out = final[["rank","item_a","item_a_name","item_b","item_b_name"]].copy()
    out.columns = ["Rank by Affinity Strength", "Product A ID", "Product A", "Product B ID", "Product B"]
    return out


# -------------------- Top Brands by Country --------------------

def build_top_brands_by_country(tx: pd.DataFrame, keep_top_n: int = 10) -> pd.DataFrame:
    tx = tx.copy()
    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce").fillna(1).astype("int64")
    tx["country"]  = tx["country"].astype("string")
    tx["brand"]    = tx["brand"].astype("string").str.strip()
    mask = tx["brand"].notna() & ~tx["brand"].str.lower().isin({"unknown", "na", ""})
    tx = tx[mask]
    agg = (tx.groupby(["country","brand"], dropna=False)["quantity"]
             .sum().reset_index(name="count").astype({"count":"int64"}))
    agg = agg.sort_values(["country","count","brand"], ascending=[True, False, True])
    agg["rank"] = agg.groupby("country")["count"].rank(method="first", ascending=False).astype(int)
    return (agg[agg["rank"] <= keep_top_n]
              .sort_values(["country","rank"])
              .reset_index(drop=True)[["country","brand","count","rank"]])


# -------------------- Return Buckets --------------------

def bucket_return_days(d: int) -> str | pd.NA:
    if pd.isna(d) or d <= 0:          return pd.NA
    if 1 <= d <= 7:                   return "week 1"
    if 8 <= d <= 14:                  return "week 2"
    if 15 <= d <= 21:                 return "week 3"
    if 22 <= d <= 30:                 return "1 month"
    for m in range(2, 13):
        lo, hi = 30*(m-1)+1, 30*m
        if lo <= d <= hi:             return f"{m} months"
    if d > 365:                       return "> 1 year"
    return pd.NA

def count_return_buckets(tx_items: pd.DataFrame) -> pd.DataFrame:
    tx = tx_items.copy()
    tx["created"] = pd.to_datetime(tx["created"], errors="coerce")
    tx = tx.dropna(subset=["created", "customer_id"])
    tx["purchase_date"] = tx["created"].dt.date
    uniq = tx[["customer_id","purchase_date"]].drop_duplicates()
    first_date = uniq.groupby("customer_id")["purchase_date"].min().rename("first_date")
    tmp = uniq.merge(first_date, on="customer_id")
    tmp = tmp[tmp["purchase_date"] > tmp["first_date"]]
    first_return = tmp.groupby("customer_id")["purchase_date"].min().rename("first_return_date")
    timeline = first_date.to_frame().merge(first_return, left_index=True, right_index=True, how="left")
    timeline["days_to_return"] = (
        pd.to_datetime(timeline["first_return_date"]) - pd.to_datetime(timeline["first_date"])
    ).dt.days
    timeline["bucket"] = timeline["days_to_return"].apply(bucket_return_days)
    ret = timeline.dropna(subset=["bucket"])
    counts = (ret.groupby("bucket").size().reset_index(name="customers")
                .sort_values("customers", ascending=False).reset_index(drop=True))
    order = (["week 1","week 2","week 3","1 month"] +
             [f"{m} months" for m in range(2,13)] + ["> 1 year"])
    counts["order"] = counts["bucket"].map({b:i for i,b in enumerate(order)})
    counts = counts.sort_values("order").drop(columns="order").reset_index(drop=True)
    return counts


# -------------------- final --------------------

def run_analytics(output_dir: Path) -> None:
    tx_items = pd.read_parquet(output_dir / "order_items.parquet")
    build_top_categories_by_season(tx_items).to_parquet(output_dir / "top_categories_by_season.parquet", index=False)
    build_top_groupids_by_season(tx_items).to_parquet(output_dir / "top_groupids_by_season.parquet", index=False)
    build_top_repurchase_groupids_by_country_unique_days(tx_items).to_parquet(
        output_dir / "top_repurchase_groupids_by_country.parquet", index=False
    )
    build_pair_cooccurrences(tx_items).to_parquet(output_dir / "pair_cooccurrences.parquet", index=False)
    build_top_brands_by_country(tx_items).to_parquet(output_dir / "top_brands_by_country.parquet", index=False)
    count_return_buckets(tx_items).to_parquet(output_dir / "return_buckets_overall.parquet", index=False)