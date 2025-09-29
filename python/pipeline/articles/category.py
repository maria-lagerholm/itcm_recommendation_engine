from __future__ import annotations
import pandas as pd

# ---------- helpers ----------
def _dedup_csv(s):
    if pd.isna(s):
        return pd.NA
    out, seen = [], set()
    for t in map(str.strip, str(s).split(",")):
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return ",".join(out) if out else pd.NA

def _toks(s):
    return [t.strip() for t in str(s).split(",") if t.strip()] if pd.notna(s) else []

def _move_after(df: pd.DataFrame, cols: list[str], after: str) -> pd.DataFrame:
    cols_all = list(df.columns)
    for c in cols:
        if c in cols_all:
            cols_all.remove(c)
    i = cols_all.index(after) + 1 if after in cols_all else len(cols_all)
    return df[cols_all[:i] + cols + cols_all[i:]]

# ---------- core ----------
def normalize_categories(articles: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Transform-only normalization:
      - dedup comma lists in category & categoryId
      - learn token→id map from paired data (most frequent id per token)
      - rebuild categoryId from category via that map
      - fill missing category with 'unknown'
      - add 'category_missing' int8 flag and place next to 'category'
    Returns (df, stats) with a small stats dict (no prints).
    """
    df = articles.copy()

    # Ensure presence; if missing, create empty string cols to keep pipeline consistent
    if "category" not in df.columns:
        df["category"] = pd.Series([pd.NA] * len(df), dtype="string")
    if "categoryId" not in df.columns:
        df["categoryId"] = pd.Series([pd.NA] * len(df), dtype="string")

    # Normalize to string dtype
    df["category"] = df["category"].astype("string")
    df["categoryId"] = df["categoryId"].astype("string")

    # 1) dedup comma lists
    df["category"] = df["category"].apply(_dedup_csv).astype("string")
    df["categoryId"] = df["categoryId"].apply(_dedup_csv).astype("string")

    # 2) learn token→id map from rows where both sides present
    pairs = []
    mismatched = 0
    for cat, cid in df[["category", "categoryId"]].dropna().itertuples(index=False):
        ct, it = _toks(cat), _toks(cid)
        n = min(len(ct), len(it))
        if n == 0:
            continue
        if len(ct) != len(it):
            mismatched += 1
        pairs.extend(zip(ct[:n], it[:n]))

    token2id: dict[str, str] = {}
    if pairs:
        dfp = pd.DataFrame(pairs, columns=["cat_tok", "id_tok"])
        # most frequent id per token
        token2id = (
            dfp.groupby(["cat_tok", "id_tok"])
               .size()
               .reset_index(name="n")
               .sort_values(["cat_tok", "n", "id_tok"], ascending=[True, False, True])
               .drop_duplicates("cat_tok")
               .set_index("cat_tok")["id_tok"]
               .to_dict()
        )

    # 3) rebuild categoryId from category using the learned map
    def _rebuild_ids(cat):
        ct = _toks(cat)
        mapped = [token2id.get(t, pd.NA) for t in ct if t in token2id]
        return ",".join(mapped) if mapped else pd.NA

    df["categoryId"] = df["category"].apply(_rebuild_ids).astype("string")

    # 4) fill missing category with 'unknown'
    df["category"] = df["category"].fillna("unknown").astype("string")

    # 5) add missing flag and position next to category
    df["category_missing"] = (df["category"] == "unknown").astype("int8")
    df = _move_after(df, ["category_missing"], "category")

    # minimal stats (for optional logging)
    stats = {
        "mismatched_token_id_rows": int(mismatched),
        "tokens_learned": len(token2id),
        "unique_category_tokens": int(df["category"].str.split(",").explode().nunique(dropna=True)),
        "unique_categoryIds": int(df["categoryId"].str.split(",").explode().nunique(dropna=True)),
        "num_unknown_category": int((df["category"] == "unknown").sum()),
    }
    return df.reset_index(drop=True), stats