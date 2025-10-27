# pipeline/top_same_brand.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

GENDER_TOKENS = {"dam", "herr"}

def load_filtered_transactions(
    trans_path: str | Path,
    avail_path: str | Path,
    bad_ids: set[str] = {"12025DK", "12025FI", "12025NO", "12025SE", "970300", "459978"},
    cols: tuple[str, ...] = ("shopUserId", "orderId", "groupId", "category", "brand", "audience"),
) -> pd.DataFrame:
    df = pd.read_parquet(trans_path, columns=list(cols))
    avail_df = pd.read_parquet(avail_path)
    gid = df["groupId"].astype(str).str.strip()
    avail_ids = set(avail_df["groupId"].astype(str).str.strip().unique())
    return df.loc[gid.isin(avail_ids) & ~gid.isin(bad_ids)].reset_index(drop=True)

def aggregate_by_groupid(
    df: pd.DataFrame,
    item_col: str = "groupId",
    brand_col: str = "brand",
    category_col: str = "category",
    audience_col: str = "audience",
) -> pd.DataFrame:
    return (
        df.groupby([item_col, brand_col, category_col, audience_col])
          .size()
          .reset_index(name="transactions")
    )

def _cat_to_set(s) -> set[str]:
    if pd.isna(s): return set()
    toks = {t.strip().lower() for t in str(s).split(",") if t.strip()}
    return {t for t in toks if t != "unknown"}

def _categories_match(a: set[str], b: set[str]) -> bool:
    core_a, core_b = a - GENDER_TOKENS, b - GENDER_TOKENS
    if not (core_a & core_b): return False
    if "dam" in a and "dam" not in b: return False
    if "herr" in a and "herr" not in b: return False
    return True

def preprocess_pairs(pairs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = pairs.copy()
    df["brand_norm"] = df["brand"].astype(str).str.strip().str.lower()
    df["aud_norm"] = df["audience"].astype(str).str.strip().str.lower()
    skipped = int(df["brand_norm"].eq("unknown").sum())
    df = df.loc[~df["brand_norm"].eq("unknown")].copy()
    df["cat_set"] = df["category"].map(_cat_to_set)
    return df, skipped

def build_recs(filtered_pairs: pd.DataFrame, min_recs=1, max_recs=10) -> tuple[pd.DataFrame, int]:
    recs, insufficient = {}, 0
    gdf = filtered_pairs.copy()
    gdf["transactions"] = pd.to_numeric(gdf["transactions"], errors="coerce").fillna(0)

    for _, g in gdf.groupby("brand_norm", sort=False):
        g = g.sort_values("transactions", ascending=False)
        cat_map = g["cat_set"].to_dict()
        cutoff = g["transactions"].quantile(0.95)
        is_bestseller = g["transactions"] >= cutoff

        for idx, row in g.iterrows():
            gid = str(row["groupId"])
            my_cats = row["cat_set"]
            my_aud  = row["aud_norm"]
            if not my_cats:
                insufficient += 1
                continue

            mask = (
                (g["groupId"].astype(str) != gid)
                & (g.index != idx)
                & (~is_bestseller)                 # drop top 5%
                & (g["aud_norm"] == my_aud)        # exact audience match
                & g.index.map(lambda j: _categories_match(my_cats, cat_map[j]))
            )

            tops = g.loc[mask, "groupId"].astype(str).head(max_recs).tolist()
            if len(tops) >= min_recs:
                recs[gid] = {f"Top {i+1}": t for i, t in enumerate(tops)}
            else:
                insufficient += 1

    rows = [{"Product ID": k, **v} for k, v in recs.items()]
    out = pd.DataFrame(rows)
    if not out.empty:
        ordered = ["Product ID"] + [c for i in range(1, max_recs + 1) if (c := f"Top {i}") in out.columns]
        out = out[ordered]
    return out, insufficient

def save_parquet(df: pd.DataFrame, path: str | Path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)

def run(
    processed_dir: str | Path,
    transactions: str = "transactions_clean.parquet",
    available: str = "articles_for_recs.parquet",
    output: str = "top_same_brand.parquet",
    min_recs: int = 1,
    max_recs: int = 10,
) -> pd.DataFrame:
    processed_dir = Path(processed_dir)
    df = load_filtered_transactions(
        trans_path=processed_dir / transactions,
        avail_path=processed_dir / available,
    )
    pairs = aggregate_by_groupid(df)
    filtered, skipped = preprocess_pairs(pairs)
    export_df, insufficient = build_recs(filtered, min_recs=min_recs, max_recs=max_recs)
    out_path = processed_dir / output
    save_parquet(export_df, out_path)
    print(f"Skipped {skipped} rows due to Unknown brand")
    print(f"Removed {insufficient} groupIds due to having fewer than {min_recs} recommendations")
    print(f"Saved {len(export_df)} rows to {out_path}")
    return export_df
