# python/pipeline/recs/lift.py
from __future__ import annotations
from typing import Iterable, Tuple
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules

def load_filtered_order_items(order_items_path: Path, articles_path: Path, bad_ids: Iterable[str] = None) -> pd.DataFrame:
    """Filter out bad/unknown groupIds to avoid skew."""
    BAD = {"12025DK", "12025FI", "12025NO", "12025SE", "970300", "459978"} if bad_ids is None else set(bad_ids)
    order_items = pd.read_parquet(order_items_path)
    articles = pd.read_parquet(articles_path)
    articles["groupId"] = articles["groupId"].astype(str).str.strip()
    order_items["groupId"] = order_items["groupId"].astype(str).str.strip()
    allow = set(articles["groupId"].dropna())
    return (order_items.loc[~order_items["groupId"].isin(BAD)]
                      .loc[order_items["groupId"].isin(allow)]
                      .reset_index(drop=True))

def filter_group_ids_by_quantile(df: pd.DataFrame, lower_q: float = 0.5, upper_q: float = 0.97) -> Tuple[pd.DataFrame, float, float]:
    """Trim extremely rare/common items."""
    counts = df["groupId"].value_counts()
    lo, hi = counts.quantile(lower_q), counts.quantile(upper_q)
    keep = counts[(counts >= lo) & (counts <= hi)].index
    return df[df["groupId"].isin(keep)].reset_index(drop=True), float(lo), float(hi)

def create_basket(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup per order; keep baskets size ≥2."""
    b = df.groupby("order_id")["groupId"].apply(lambda x: sorted(set(x))).reset_index()
    return b[b["groupId"].apply(lambda x: len(x) >= 2)].reset_index(drop=True)

def create_basket_df(basket: pd.DataFrame) -> pd.DataFrame:
    """Orders×items boolean matrix."""
    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(basket["groupId"])
    return pd.DataFrame.sparse.from_spmatrix(X, index=basket["order_id"], columns=mlb.classes_).astype(bool)

def get_frequent_itemsets(basket_df: pd.DataFrame, min_support: float = 0.001) -> pd.DataFrame:
    """Frequent itemsets via apriori."""
    return apriori(basket_df, min_support=min_support, use_colnames=True)

def build_rules(fis: pd.DataFrame, min_confidence: float = 0.1) -> pd.DataFrame:
    """Confidence-filtered rules; sort by lift."""
    return association_rules(fis, metric="confidence", min_threshold=min_confidence).sort_values("lift", ascending=False)

def rules_to_parquet_topk(rules: pd.DataFrame, output_path: Path, top_k: int = 10) -> pd.DataFrame:
    """Top-K consequents per antecedent (score=confidence; tie=lift)."""
    r = rules[(rules["antecedents"].apply(lambda s: len(s) == 1)) & (rules["consequents"].apply(lambda s: len(s) == 1))].copy()
    r["A"] = r["antecedents"].apply(lambda s: str(next(iter(s))))
    r["B"] = r["consequents"].apply(lambda s: str(next(iter(s))))
    r = r.sort_values(["A", "confidence", "lift"], ascending=[True, False, False]).groupby("A").head(top_k)

    def pack(g: pd.DataFrame) -> pd.Series:
        pairs = list(zip(g["B"], g["confidence"]))[:top_k]
        pairs += [(None, None)] * (top_k - len(pairs))
        d = {"Product ID": g.name}
        for i, (cid, sc) in enumerate(pairs, 1):
            d[f"Top {i}"] = cid
            d[f"Score {i}"] = float(sc) if sc is not None else None
        return pd.Series(d)

    out = (r.groupby("A").apply(pack, include_groups=False)
              .reset_index(drop=True)
              .sort_values("Product ID")
              .reset_index(drop=True))
    out.to_parquet(output_path, index=False)
    return out

def run(
    processed_dir: Path,
    transactions: str,
    available: str,
    output: str,
    min_support: float = 0.001,
    min_confidence: float = 0.10,
    lower_q: float = 0.50,
    upper_q: float = 0.97,
    top_k: int = 10,
) -> Path:
    """End-to-end transform with paths provided by the caller."""
    order_items_path = processed_dir / transactions
    articles_path = processed_dir / available
    out_path = processed_dir / output

    df = load_filtered_order_items(order_items_path, articles_path)[["order_id", "groupId"]]
    df, _, _ = filter_group_ids_by_quantile(df, lower_q=lower_q, upper_q=upper_q)
    basket = create_basket(df)
    X = create_basket_df(basket)
    fis = get_frequent_itemsets(X, min_support=min_support)
    rules = build_rules(fis, min_confidence=min_confidence)
    rules_to_parquet_topk(rules, output_path=out_path, top_k=top_k)
    return out_path
