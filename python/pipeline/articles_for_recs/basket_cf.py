# /workspace/python/pipeline/articles_for_recs/basket_cf.py
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations

BAD = {"12025DK","12025FI","12025NO","12025SE","970300","459978"}

def mk_baskets(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["created"] = pd.to_datetime(x["created"], errors="coerce")
    return x.groupby("orderId", as_index=False).agg(
        items=("groupId", lambda s: list(dict.fromkeys(map(str, s)))),
        t=("created", "max"),
    )

def supports(baskets: pd.DataFrame):
    a = Counter(); p2 = Counter()
    for s in baskets["items"]:
        u = list(dict.fromkeys(map(str, s)))
        a.update(u)
        p2.update(tuple(sorted(c)) for c in combinations(u, 2))
    return a, p2

def nbrs(a: Counter, p2: Counter, mi: int, mp: int, k: int):
    d = defaultdict(list)
    for (i, j), n in p2.items():
        if n < mp: 
            continue
        ni, nj = a[i], a[j]
        if ni < mi or nj < mi:
            continue
        s = n / (ni + nj - n) # jaccard similarity
        d[i].append((j, s)); d[j].append((i, s))
    for i in list(d):
        d[i] = sorted(d[i], key=lambda x: x[1], reverse=True)[:k]
    return d

def nb_df(nb: dict) -> pd.DataFrame:
    r = []
    for i, lst in nb.items():
        for j, s in lst:
            r.append((str(i), str(j), float(s)))
    return pd.DataFrame(r, columns=["item_id", "neighbor_id", "score"])

def run(
    processed_dir: Path,
    min_item_support: int = 10,
    min_pair_support: int = 5,
    k_neighbors: int = 100,
    score_threshold: float = 0.02, # comes from tuning
    topk: int = 10,
) -> None:
    tx = pd.read_parquet(processed_dir.joinpath("transactions_clean.parquet"), columns=["orderId","groupId","created"])
    tx = tx[~tx["groupId"].astype(str).str.strip().isin(BAD)].reset_index(drop=True)
    baskets = mk_baskets(tx)
    a_all, p2_all = supports(baskets)
    nb_all = nbrs(a_all, p2_all, mi=min_item_support, mp=min_pair_support, k=k_neighbors)

    art = pd.read_parquet(processed_dir.joinpath("articles_for_recs.parquet"))
    valid = set(art["groupId"].astype(str).unique())

    df = nb_df(nb_all)
    df = df[df["neighbor_id"].isin(valid)]
    df = df[df["item_id"].isin(valid)]
    df = df[df["score"] >= score_threshold]
    df["rank"] = df.groupby("item_id")["score"].rank(method="first", ascending=False)
    df = df[df["rank"] <= topk].sort_values(["item_id", "rank"]).reset_index(drop=True)
    df.to_parquet(processed_dir.joinpath("basket_completion.parquet"), index=False)
