from __future__ import annotations
import re
import pandas as pd

# ---------- patterns & helpers ----------
NOISE_TOKENS = {
    "***missing***", "rosa", "svart", "blå", "offwhite", "vinröd", "greige",
    "kuddfodral", "påslakan", "tomte", "ostbricka", "plommonlila",
}
SIZE_PATTERNS = [
    r"^\d{2}$", r"^\d{2}/\d{2}$", r"^[A-Z]{1,2}/?[A-Z]{0,2}\d{2}$",
    r"^\d{2}[A-Z]/[A-Z]$", r"^[A-Z]{1,3}$", r"^\d{2}x\d{2,3}\s*cm$",
    r"^\d{2,3}x\d{2,3}\s*cm$", r"^\d+(\.\d+)?\s*mm$", r"^\d{2}x\d{2}$",
    r"^\d+$", r"^\d+[- ]?PACK$", r"^[A-Z]/[A-Z]\d{2}$",
]
SIZE_RE = re.compile("|".join(f"(?:{p})" for p in SIZE_PATTERNS), re.I)

def _move_after(df: pd.DataFrame, cols: list[str], after: str) -> pd.DataFrame:
    cols_all = list(df.columns)
    for c in cols:
        if c in cols_all:
            cols_all.remove(c)
    i = cols_all.index(after) + 1 if after in cols_all else len(cols_all)
    return df[cols_all[:i] + cols + cols_all[i:]]

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

def _canon_size_token(t: str) -> str:
    t = t.strip().replace("×", "x").replace(" X ", "x").replace(" x ", "x")
    t = re.sub(r"\s*x\s*", "x", t)
    t = re.sub(r"\s+cm\b", " cm", t, flags=re.I)
    t = re.sub(r"\s+mm\b", " mm", t, flags=re.I)
    if re.search(r"\d,\d", t):
        t = t.replace(",", ".")
    upper_set = {"xs", "s", "m", "l", "xl", "xxl", "3xl", "4xl", "5xl", "6xl", "one size"}
    t = t.upper() if t.lower() in upper_set else t
    t = t.replace("–", "-").replace("—", "-")
    if re.fullmatch(r"\d{2}-\d{2}", t):
        t = t.replace("-", "/")
    return t

def _is_size_token(t: str) -> bool:
    t0 = t.strip().lower()
    return t0 not in NOISE_TOKENS and bool(SIZE_RE.match(t.strip()))

def _normalize_size_cell(s):
    if pd.isna(s):
        return pd.NA
    toks = [_canon_size_token(t) for t in str(s).split(",") if t.strip()]
    toks = [t for t in toks if _is_size_token(t)]
    out = []
    for t in toks:
        if t not in out:
            out.append(t)
    return ",".join(out) if out else pd.NA

# ---------- public API ----------
def normalize_sizes(
    articles: pd.DataFrame,
    *,
    size_col: str = "size",
    size_id_col: str = "sizeId",
    add_missing_flag: bool = True,
    fill_unknown_text: str = "unknown",
) -> tuple[pd.DataFrame, dict]:
    """
    Transform-only size normalization:
      - Clean/normalize tokens in `size` (comma lists → canonical tokens)
      - Nullify `sizeId` when size is NA/unknown
      - Learn token→id mapping from existing paired rows (most frequent id per token)
      - Rebuild `sizeId` from `size` using that mapping
      - Dedup CSVs; fill missing size text with 'unknown'
      - Add `size_missing` flag (optional) and place it after `size`
    Returns (df, stats). No prints.
    """
    df = articles.copy()

    # Ensure columns exist
    if size_col not in df.columns:
        df[size_col] = pd.Series([pd.NA] * len(df), dtype="string")
    if size_id_col not in df.columns:
        df[size_id_col] = pd.Series([pd.NA] * len(df), dtype="string")

    # String dtypes
    df[size_col] = df[size_col].astype("string")
    df[size_id_col] = df[size_id_col].astype("string")

    # 1) normalize size text
    df[size_col] = df[size_col].apply(_normalize_size_cell).astype("string")

    # 2) if size is NA/unknown -> sizeId = NA
    na_or_unknown = df[size_col].isna() | (df[size_col].str.lower() == "unknown")
    df.loc[na_or_unknown, size_id_col] = pd.NA

    # 3) learn size token → id mapping from rows where both are present
    pairs, mismatched = [], 0
    for sz, sid in df[[size_col, size_id_col]].dropna().itertuples(index=False):
        st, it = _toks(sz), _toks(sid)
        n = min(len(st), len(it))
        if n == 0:
            continue
        if len(st) != len(it):
            mismatched += 1
        pairs.extend(zip(st[:n], it[:n]))

    token2id: dict[str, str] = {}
    if pairs:
        dfp = pd.DataFrame(pairs, columns=["size_tok", "id_tok"])
        token2id = (
            dfp.groupby(["size_tok", "id_tok"])
               .size()
               .reset_index(name="n")
               .sort_values(["size_tok", "n", "id_tok"], ascending=[True, False, True])
               .drop_duplicates("size_tok")
               .set_index("size_tok")["id_tok"]
               .to_dict()
        )

    # 4) rebuild sizeId from size using the learned map
    def _rebuild_ids(sz):
        st = [t for t in _toks(sz) if _is_size_token(t)]
        mapped = [token2id.get(t, pd.NA) for t in st if t in token2id]
        return ",".join(mapped) if mapped else pd.NA

    df[size_id_col] = df[size_col].apply(_rebuild_ids).astype("string")

    # 5) final dedup of csv cells
    df[size_col] = df[size_col].apply(_dedup_csv).astype("string")
    df[size_id_col] = df[size_id_col].apply(_dedup_csv).astype("string")

    # 6) fill missing size with 'unknown' and add flag
    df[size_col] = df[size_col].fillna(fill_unknown_text).astype("string")
    stats = {
        "mismatched_token_id_rows": int(mismatched),
        "tokens_learned": len(token2id),
        "size_unknown_after_fill": int((df[size_col] == fill_unknown_text).sum()),
        "unique_size_tokens": int(df[size_col].str.split(",").explode().nunique(dropna=True)),
        "unique_size_ids": int(df[size_id_col].str.split(",").explode().nunique(dropna=True)),
    }

    if add_missing_flag:
        miss_col = "size_missing"
        df[miss_col] = (df[size_col] == fill_unknown_text).astype("int8")
        df = _move_after(df, [miss_col], size_col)

    return df.reset_index(drop=True), stats