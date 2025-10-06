from __future__ import annotations
import pandas as pd

#------brand normalization helpers-----
def _norm_brand(s) -> pd.StringDtype:
    if pd.isna(s):
        return pd.NA
    s = " ".join(str(s).strip().split())
    return s or pd.NA

def _move_after(df: pd.DataFrame, cols: list[str], after: str) -> pd.DataFrame:
    cols_all = list(df.columns)
    for c in cols:
        if c in cols_all:
            cols_all.remove(c)
    i = cols_all.index(after) + 1 if after in cols_all else len(cols_all)
    return df[cols_all[:i] + cols + cols_all[i:]]

def _mode_or_first(s: pd.Series) -> str:
    m = s.mode(dropna=True)
    return (m.iat[0] if not m.empty else s.dropna().iat[0]) if s.size else pd.NA

#------normalize brands main-----
def normalize_brands(
    articles: pd.DataFrame,
    *,
    brand_col: str = "brand",
    brand_id_col: str = "brandId",
    fill_unknown_text: str = "unknown",
    add_missing_flag: bool = True,  # kept for compatibility, but has no effect now
    backfill_brand_from_id: bool = True,
) -> tuple[pd.DataFrame, dict]:

    df = articles.copy()

    if brand_col not in df.columns:
        df[brand_col] = pd.Series([pd.NA] * len(df), dtype="string")
    if brand_id_col not in df.columns:
        df[brand_id_col] = pd.Series([pd.NA] * len(df), dtype="string")

    df[brand_col] = df[brand_col].astype("string").apply(_norm_brand)
    df[brand_id_col] = df[brand_id_col].astype("string").str.strip()

    known = (
        df.dropna(subset=[brand_col, brand_id_col])[[brand_col, brand_id_col]]
          .drop_duplicates()
    )
    if not known.empty:
        name_to_id = (
            known.groupby(brand_col)[brand_id_col]
                 .apply(_mode_or_first)
        )
        id_to_name = (
            known.groupby(brand_id_col)[brand_col]
                 .apply(_mode_or_first)
        )
    else:
        name_to_id = pd.Series(dtype="string")
        id_to_name = pd.Series(dtype="string")

    mask_id_missing = df[brand_id_col].isna() & df[brand_col].notna()
    filled_ids = df.loc[mask_id_missing, brand_col].map(name_to_id).astype("string")
    filled_ids = filled_ids.reindex(df.index)
    df.loc[mask_id_missing, brand_id_col] = filled_ids

    filled_names_ct = 0
    if backfill_brand_from_id:
        mask_name_missing = df[brand_col].isna() & df[brand_id_col].notna()
        filled_names = df.loc[mask_name_missing, brand_id_col].map(id_to_name).astype("string")
        filled_names = filled_names.reindex(df.index)
        df.loc[mask_name_missing, brand_col] = filled_names
        filled_names_ct = int(mask_name_missing.sum())

    df[brand_col] = df[brand_col].fillna(fill_unknown_text).astype("string")

    # No need to create missing_brand column

    stats = {
        "known_pairs": int(len(known)),
        "name_to_id_learned": int(name_to_id.size),
        "id_to_name_learned": int(id_to_name.size),
        "brandId_filled_from_name": int(mask_id_missing.sum()),
        "brand_filled_from_id": filled_names_ct,
        "brand_unknown_after_fill": int((df[brand_col] == fill_unknown_text).sum()),
        "unique_brands": int(df[brand_col].nunique(dropna=True)),
        "unique_brandIds": int(df[brand_id_col].nunique(dropna=True)),
    }

    return df.reset_index(drop=True), stats