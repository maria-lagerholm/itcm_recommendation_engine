import pandas as pd

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

def normalize_categories(articles: pd.DataFrame) -> pd.DataFrame:
    df = articles.copy()
    for c in ("category", "categoryId"):
        if c not in df:
            df[c] = pd.Series(pd.NA, index=df.index, dtype="string")
        df[c] = df[c].astype("string")
    df["category"]  = df["category"].apply(_dedup_csv).astype("string")
    df["categoryId"] = df["categoryId"].apply(_dedup_csv).astype("string")
    pairs = []
    for cat, cid in df[["category", "categoryId"]].dropna().itertuples(index=False):
        ct, it = _toks(cat), _toks(cid)
        n = min(len(ct), len(it))
        if n:
            pairs.extend(zip(ct[:n], it[:n]))
    if pairs:
        dfp = pd.DataFrame(pairs, columns=["cat_tok", "id_tok"])
        token2id = (
            dfp.groupby(["cat_tok", "id_tok"]).size()
               .reset_index(name="n")
               .sort_values(["cat_tok", "n", "id_tok"], ascending=[True, False, True])
               .drop_duplicates("cat_tok")
               .set_index("cat_tok")["id_tok"]
               .to_dict()
        )
    else:
        token2id = {}
    def _rebuild_ids(cat):
        ct = _toks(cat)
        mapped = [token2id.get(t) for t in ct if t in token2id]
        return ",".join(mapped) if mapped else pd.NA
    df["categoryId"] = df["category"].apply(_rebuild_ids).astype("string")
    df["category"] = df["category"].fillna("unknown").astype("string")
    return df.reset_index(drop=True)
