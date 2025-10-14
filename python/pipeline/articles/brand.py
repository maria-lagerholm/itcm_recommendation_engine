import pandas as pd

def normalize_brands(
    articles: pd.DataFrame,
    *,
    brand_col: str = "brand",
    brand_id_col: str = "brandId",
    fill_unknown_text: str = "unknown",
    backfill_brand_from_id: bool = True,
) -> pd.DataFrame:
    df = articles.copy()
    for c in (brand_col, brand_id_col):
        if c not in df:
            df[c] = pd.Series(pd.NA, index=df.index, dtype="string")
        df[c] = df[c].astype("string")
    df[brand_col] = (
        df[brand_col].str.replace(r"\s+", " ", regex=True).str.strip()
        .replace("", pd.NA)
    )
    df[brand_id_col] = df[brand_id_col].str.strip()
    known = df[[brand_col, brand_id_col]].dropna().drop_duplicates()
    if not known.empty:
        name_to_id = known.groupby(brand_col)[brand_id_col] \
                          .apply(lambda s: s.value_counts().idxmax())
        id_to_name = known.groupby(brand_id_col)[brand_col] \
                          .apply(lambda s: s.value_counts().idxmax())
    else:
        name_to_id = pd.Series(dtype="string")
        id_to_name = pd.Series(dtype="string")
    df[brand_id_col] = df[brand_id_col].fillna(df[brand_col].map(name_to_id))
    if backfill_brand_from_id:
        df[brand_col] = df[brand_col].fillna(df[brand_id_col].map(id_to_name))
    df[brand_col] = df[brand_col].fillna(fill_unknown_text)
    return df.reset_index(drop=True)
