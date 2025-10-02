from __future__ import annotations
import pandas as pd

#------helpers-----
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

#------core-----
def normalize_categories(articles: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = articles.copy()
    if "category" not in df.columns:
        df["category"] = pd.Series([pd.NA] * len(df), dtype="string")
    if "categoryId" not in df.columns:
        df["categoryId"] = pd.Series([pd.NA] * len(df), dtype="string")
    df["category"] = df["category"].astype("string")
    df["categoryId"] = df["categoryId"].astype("string")
    df["category"] = df["category"].apply(_dedup_csv).astype("string")
    df["categoryId"] = df["categoryId"].apply(_dedup_csv).astype("string")
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
        token2id = (
            dfp.groupby(["cat_tok", "id_tok"])
               .size()
               .reset_index(name="n")
               .sort_values(["cat_tok", "n", "id_tok"], ascending=[True, False, True])
               .drop_duplicates("cat_tok")
               .set_index("cat_tok")["id_tok"]
               .to_dict()
        )
    def _rebuild_ids(cat):
        ct = _toks(cat)
        mapped = [token2id.get(t, pd.NA) for t in ct if t in token2id]
        return ",".join(mapped) if mapped else pd.NA
    df["categoryId"] = df["category"].apply(_rebuild_ids).astype("string")
    df["category"] = df["category"].fillna("unknown").astype("string")
    stats = {
        "mismatched_token_id_rows": int(mismatched),
        "tokens_learned": len(token2id),
        "unique_category_tokens": int(df["category"].str.split(",").explode().nunique(dropna=True)),
        "unique_categoryIds": int(df["categoryId"].str.split(",").explode().nunique(dropna=True)),
        "num_unknown_category": int((df["category"] == "unknown").sum()),
    }
    return df.reset_index(drop=True), stats