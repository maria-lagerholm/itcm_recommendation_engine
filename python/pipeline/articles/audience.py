from __future__ import annotations
import re
import pandas as pd

# canonical audience → id
AUD2ID: dict[str, str] = {
    "dam": "6",
    "herr": "15",
    "baby & barn": "12",
    "barn & ungdom": "42",
    "generic": "99",
    "hemmet": "222",
}

# ---- helpers ----
REA_TOKEN = re.compile(r"(^|,)\s*rea\s*(?=,|$)")

def _move_after(df: pd.DataFrame, cols: list[str], after: str) -> pd.DataFrame:
    cols_all = list(df.columns)
    for c in cols:
        if c in cols_all:
            cols_all.remove(c)
    i = cols_all.index(after) + 1 if after in cols_all else len(cols_all)
    return df[cols_all[:i] + cols + cols_all[i:]]

def _norm_audience(a) -> pd.StringDtype:
    if pd.isna(a):
        return pd.NA
    toks = {t.strip().lower() for t in str(a).split(",") if t.strip()}
    if any("dam" in t for t in toks):  # “dam” anywhere wins
        return "dam"
    keep = [t for t in toks if t in AUD2ID]
    return ",".join(keep) if keep else pd.NA

def _to_ids(a) -> pd.StringDtype:
    if pd.isna(a):
        return pd.NA
    ids = sorted({AUD2ID[t] for t in a.split(",") if t in AUD2ID}, key=int)
    return ",".join(ids) if ids else pd.NA

def _strip_rea(s: str) -> str:
    s = s.lower()
    s = REA_TOKEN.sub(lambda m: "," if m.group(1) else "", s)
    return re.sub(r",+", ",", s).strip(", ").strip()

# keyword bags (same as notebook)
DAM = set("""
dam bh trosor underkläder body bodykorselett korsett korsetter
klänning klänningar tunika tunikor topp toppar kjol kjolar
byxa byxor blus blusar nattlinne bikinibh bikini t-shirt-bh
minimizer kofta koftor väst västar skor väskor sjalar
""".split())
HEM = set("""
frottéhanddukar badlakan bad badrumsmattor kökshanddukar vaxdukar dukar
pläd plädar kanallängder kanalkappa gardiner påslakanset bädd
lakan örngott hemtextil kuddfodral överkast gardinstänger kökshjälpmedel
dekorationer metervara prydnadssaker belysning servetter
""".split())
GEN = set("""
inkontinens stödartiklar vardagshjälpmedel rollator rollatorer stödstrumpor
skotillbehör fotvård hobbyhörnan pussel sytillbehör symaskiner lust
massage synhjälpmedel medicin halkskydd träning & motion
""".split())
HER = set("herr skjorta skjortor kostym kavaj boxer".split())

def _classify_from_category(cat) -> pd.StringDtype:
    if pd.isna(cat):
        return pd.NA
    s = _strip_rea(str(cat))
    if not s:
        return pd.NA
    # simple substring membership against token bags
    # split on punctuation/commas/spaces
    tokens = set(re.split(r"[,\s/&\-]+", s))
    if tokens & DAM:
        return "dam"
    if tokens & HER:
        return "herr"
    if tokens & HEM:
        return "hemmet"
    if tokens & GEN:
        return "generic"
    return pd.NA

# ---- public API ----
def normalize_audience(
    articles: pd.DataFrame,
    *,
    audience_col: str = "audience",
    category_col: str = "category",
    audience_id_col: str = "audienceId",
    audience_missing_col: str = "audience_missing",
    fill_unknown: str = "unknown",
) -> tuple[pd.DataFrame, dict]:
    """
    Deterministic transform:
      1) Normalize existing `audience` (comma list → canonical tokens, 'dam' wins).
      2) Fill missing `audience` via category keywords (ignoring 'REA' token).
      3) Build `audienceId` from `audience` using AUD2ID.
      4) Add `audience_missing` flag; place id/flag after `audience`.
      5) Fill remaining missing `audience` with `fill_unknown`.

    Returns (df, stats) without printing.
    """
    df = articles.copy()
    # ensure columns
    if audience_col not in df.columns:
        df[audience_col] = pd.Series([pd.NA] * len(df), dtype="string")
    if category_col not in df.columns:
        df[category_col] = pd.Series([pd.NA] * len(df), dtype="string")

    df[audience_col] = df[audience_col].astype("string")
    df[category_col] = df[category_col].astype("string")

    # 1) normalize what’s present
    before_na = int(df[audience_col].isna().sum())
    df[audience_col] = df[audience_col].apply(_norm_audience).astype("string")

    # 2) fill from category
    na_mask = df[audience_col].isna()
    classified = df.loc[na_mask, category_col].apply(_classify_from_category)
    to_fill_idx = classified.dropna().index
    df.loc[to_fill_idx, audience_col] = classified.loc[to_fill_idx].astype("string")
    filled = int(len(to_fill_idx))

    # 3) ids & flag
    df[audience_id_col] = df[audience_col].apply(_to_ids).astype("string")
    df[audience_missing_col] = df[audience_col].isna().astype("int8")
    df = _move_after(df, [audience_id_col, audience_missing_col], audience_col)

    # 4) final fill for audience text
    df[audience_col] = df[audience_col].fillna(fill_unknown).astype("string")

    stats = {
        "audience_na_before": before_na,
        "audience_filled_from_category": filled,
        "audience_na_after": int(df[audience_col].isna().sum()),
        "audience_unknown_after_fill": int((df[audience_col] == fill_unknown).sum()),
    }
    return df.reset_index(drop=True), stats