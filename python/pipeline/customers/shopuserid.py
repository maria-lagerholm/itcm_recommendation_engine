# python/pipeline/customers/shopuserid.py
from __future__ import annotations
import pandas as pd

#------build id remap-----
def build_id_remap(
    customers: pd.DataFrame,
    *,
    id_col: str = "shopUserId",
    key_cols: tuple[str, ...] = ("invoiceFirstName", "invoiceLastName", "invoiceZip"),
) -> pd.Series:
    mask = customers[list(key_cols)].notna().all(axis=1)
    base = customers.loc[mask, [*key_cols, id_col]].copy()
    first = (
        base.groupby(list(key_cols), sort=False, as_index=False)[id_col]
            .first()
            .rename(columns={id_col: "canonical"})
    )
    all_ids = (
        base.groupby(list(key_cols), sort=False, as_index=False)[id_col]
            .agg(lambda s: pd.unique(s))
            .rename(columns={id_col: "all_ids"})
    )
    canon = first.merge(all_ids, on=list(key_cols), how="inner")
    remap = (
        canon.explode("all_ids", ignore_index=True)
             .rename(columns={"all_ids": id_col})
             [[id_col, "canonical"]]
             .dropna()
             .drop_duplicates()
    )
    self_map = remap["canonical"].drop_duplicates().to_frame(name=id_col)
    self_map["canonical"] = self_map[id_col]
    remap = pd.concat([remap, self_map], ignore_index=True).drop_duplicates()
    return pd.Series(remap["canonical"].values, index=remap[id_col].values, name="canonical")

#------apply id remap-----
def apply_id_remap(df: pd.DataFrame, remap: pd.Series, *, id_col: str = "shopUserId") -> pd.DataFrame:
    out = df.copy()
    out[id_col] = out[id_col].map(remap).fillna(out[id_col])
    return out