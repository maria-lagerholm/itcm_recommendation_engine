# /workspace/python/pipeline/articles_for_recs/basket_cf.py
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse as sp
import cornac
from cornac.eval_methods import RatioSplit
from cornac.models.ease import EASE

BAD_IDS = {"12025DK", "12025FI", "12025NO", "12025SE", "970300", "459978"}

def _load_filtered_transactions(processed_dir: Path,
                                cols=("shopUserId", "orderId", "groupId")) -> pd.DataFrame:
    tx = pd.read_parquet(processed_dir / "transactions_clean.parquet", columns=list(cols))
    avail = pd.read_parquet(processed_dir / "articles_for_recs.parquet", columns=["groupId"])
    gid = tx["groupId"].astype(str).str.strip()
    avail_ids = set(avail["groupId"].astype(str).str.strip().unique())
    return tx.loc[gid.isin(avail_ids) & ~gid.isin(BAD_IDS)].reset_index(drop=True)

def _make_user_item_pairs(df: pd.DataFrame,
                          user_col="shopUserId",
                          order_col="orderId",
                          item_col="groupId",
                          pref_value=1.0) -> pd.DataFrame:
    x = (df.drop_duplicates(subset=[user_col, order_col, item_col])
           .drop_duplicates(subset=[user_col, item_col])[[user_col, item_col]].copy())
    x["pref"] = float(pref_value)
    return x

def _filter_pairs_by_item_frequency(pairs: pd.DataFrame,
                                    item_col="groupId",
                                    q_low=0.5,
                                    q_high=0.96,
                                    inclusive="both") -> pd.DataFrame:
    gid = pairs[item_col].astype(str).str.strip()
    counts = gid.value_counts()
    low, high = counts.quantile([q_low, q_high])
    mask = gid.map(counts).between(low, high, inclusive=inclusive)
    return pairs.loc[mask].reset_index(drop=True)

def _to_uir(pairs: pd.DataFrame,
            user_col="shopUserId",
            item_col="groupId",
            pref_col="pref"):
    return list(zip(pairs[user_col].astype(str),
                    pairs[item_col].astype(str),
                    pairs[pref_col].astype(float)))

def _fit_ease(uir,
              test_size=0.10,
              seed=42):
    rs = RatioSplit(
        data=uir,
        test_size=test_size,
        val_size=0.0,
        exclude_unknowns=True,
        seed=seed,
        verbose=False,
    )
    model = EASE(name="EASE-final", verbose=False)
    model.fit(rs.train_set)
    return model, rs

def _prepare_item_data(model: EASE):
    B = np.asarray(model.get_item_vectors(), dtype=float)
    ts = model.train_set
    raw_item_ids = np.asarray(ts.item_ids, dtype=str)
    X = ts.X if hasattr(ts, "X") else ts.matrix
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    return B, raw_item_ids, X

def _cooccurrence_support(X: sp.csr_matrix):
    Xb = X.copy(); Xb.data[:] = 1
    S = (Xb.T @ Xb).tocsr()
    S.setdiag(0); S.eliminate_zeros()
    return S

def _top_supported_for_item(i: int,
                            support: sp.csr_matrix,
                            B: np.ndarray,
                            raw_item_ids: np.ndarray,
                            min_support=12,
                            topk=10):
    row = support.getrow(i)
    if row.nnz == 0:
        return []
    mask = row.data >= min_support
    if not np.any(mask):
        return []
    cand = row.indices[mask]
    if cand.size < topk:
        return []
    w = B[i, cand]
    order = np.argsort(w)[::-1][:topk]
    return list(raw_item_ids[cand[order]])

def _build_basket_completion(model: EASE,
                             min_support=12,
                             topk=10) -> pd.DataFrame:
    """Use EASE item embeddings gated by co-occurrence support to form robust Top-K complements."""
    B, raw_item_ids, X = _prepare_item_data(model)
    support = _cooccurrence_support(X)
    rows = []
    for i, pid in enumerate(raw_item_ids):
        recs = _top_supported_for_item(i, support, B, raw_item_ids,
                                       min_support=min_support, topk=topk)
        if len(recs) < topk:
            continue
        rows.append({"Product ID": pid, **{f"Top {k}": recs[k-1] for k in range(1, topk + 1)}})
    cols = ["Product ID"] + [f"Top {k}" for k in range(1, topk + 1)]
    return pd.DataFrame(rows, columns=cols)

def run(processed_dir: Path,
        out_filename: str = "basket_completion.parquet",
        item_freq_q_low: float = 0.5,
        item_freq_q_high: float = 0.96,
        min_support: int = 12,
        topk: int = 10,
        test_size: float = 0.10,
        seed: int = 42) -> Path:
    df_tx = _load_filtered_transactions(processed_dir)
    pairs = _make_user_item_pairs(df_tx)
    pairs = _filter_pairs_by_item_frequency(pairs,
                                            q_low=item_freq_q_low,
                                            q_high=item_freq_q_high)
    uir = _to_uir(pairs)
    model, _ = _fit_ease(uir, test_size=test_size, seed=seed)
    df_bc = _build_basket_completion(model, min_support=min_support, topk=topk)
    out_path = processed_dir / out_filename
    df_bc.to_parquet(out_path, engine="pyarrow", index=False)
    return out_path
