from __future__ import annotations
import re
import pandas as pd

# -----------------------
# Helpers & regex patterns
# -----------------------
def _is_missing(s):
    return pd.isna(s) or str(s).strip().lower() in {"", "unknown", "nan", "<na>"}

COLORS = [
    "svart","vit","offwhite","off white","grå","ljusgrå","mörkgrå","blå","ljusblå","mellanblå","mörkblå","marin","navy","turkos",
    "grön","mörkgrön","ljusgrön","oliv","khaki","röd","vinröd","rosa","cerise","lila","plommon","gul","orange",
    "beige","sand","natur","brun","kaffe","kamel","taupe","multi","multicolor","flerfärgad",
    "silver","silvergrå","guld","grårosa","vitblå","gråbeige","offvit","gråbrun","gråsvart"
]
_color_alt = [r"off\s*white" if c == "off white" else re.escape(c) for c in COLORS]
COLOR_RE = re.compile(r"\b(" + "|".join(_color_alt) + r")\b", re.IGNORECASE)

DIM_RE       = re.compile(r"\b\d{1,3}\s*[x×]\s*\d{1,3}\s*(?:cm|mm)?\b", re.IGNORECASE)
LONE_DIM_RE  = re.compile(r"\b\d{1,4}(?:[.,]\d+)?\s*(?:cm|mm)\b", re.IGNORECASE)
DIAM_RE      = re.compile(r"[Øø]\s*\d{1,3}\s*(?:cm|mm)\b", re.IGNORECASE)
WEIGHT_RE    = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:kg|g)\b", re.IGNORECASE)

PACK_ANY_RE      = re.compile(r"\b(\d+)\s*[- ]?\s*(?:pack|pk|st\.?|st|p|d|del(?:ar)?)\b", re.IGNORECASE)
LETTER_SIZE_RE   = re.compile(r"\b(XXXL|XXL|XL|XS|S|M|L)\b", re.IGNORECASE)
BRA_SIZE_RE      = re.compile(r"\b([A-H][0-9]{2})\b", re.IGNORECASE)
EU_SIZE_RE       = re.compile(r"\b([2-6][0-9])\b(?!\s*(?:cm|mm))", re.IGNORECASE)

_UNWANTED_PHRASES = [
    "Övrigt","Frakt & exp. avgift","Aviavgift","Administrationsavgift","Express","Pf avg","Svarsporto","Krav outl",
    "Hemleverans 1","Hemleverans 2","Hemleverans 3","Hemleverans 4","Tillägg frakt","Krav outlöst","Returporto",
    "Katalogporto","Färgkarta porto","Manual till 293076","Rabatt"
]
_UNWANTED_RE = re.compile("|".join(map(re.escape, _UNWANTED_PHRASES)), re.IGNORECASE)

def _canon_pack(num: str, unit: str) -> str:
    u = unit.lower().replace("st.", "st")
    if u in {"pack", "p"}: return f"{num}-pack"
    if u == "pk":         return f"{num} pk"
    if u == "st":         return f"{num} st"
    return f"{num} delar"

def extract_color(txt: str | None) -> str | None:
    if not isinstance(txt, str): return None
    m = COLOR_RE.search(txt)
    return m.group(0).lower() if m else None

def extract_sizes(txt: str | None) -> list[str]:
    if not isinstance(txt, str): return []
    out: list[str] = []
    out += DIM_RE.findall(txt)
    out += [m.group(0) for m in DIAM_RE.finditer(txt)]
    out += LONE_DIM_RE.findall(txt)
    out += WEIGHT_RE.findall(txt)
    out += [
        _canon_pack(
            m.group(1),
            (re.search(r"(pack|pk|st\.?|st|p|d|del(?:ar)?)", m.group(0), re.IGNORECASE) or ["pack"])[0]
        )
        for m in PACK_ANY_RE.finditer(txt)
    ]
    out += [m.group(0).upper() for m in LETTER_SIZE_RE.finditer(txt)]
    out += [m.group(0).upper() for m in BRA_SIZE_RE.finditer(txt)]
    out += [m.group(1) for m in EU_SIZE_RE.finditer(txt)]

    seen, uniq = set(), []
    for t in out:
        k = t.lower().strip()
        if k not in seen:
            seen.add(k)
            uniq.append(t.strip())
    return uniq

def clean_name(txt: str | None, found_color: str | None = None, *, count_only=False):
    if not isinstance(txt, str):
        return (txt, 0) if count_only else txt
    s, n = txt, 0
    s, n_bes = re.subn(r"\bBeskrivning\s+", "", s); n += n_bes
    if found_color:
        s, n1 = COLOR_RE.subn(" ", s); n += n1
    for pat in (DIM_RE, DIAM_RE, LONE_DIM_RE, WEIGHT_RE, PACK_ANY_RE, LETTER_SIZE_RE, BRA_SIZE_RE, EU_SIZE_RE):
        s, k = pat.subn(" ", s); n += k
    n += sum(s.count(c) for c in ['\\','“','”','"'])
    s = s.replace("\\"," ").replace("“"," ").replace("”"," ").replace('"'," ")
    for pat in [r"\(\s*\)", r"\s*[-–/]\s*", r"\s{2,}"]:
        s, k = re.subn(pat, " ", s); n += k
    s = s.strip(" -–,.;").strip()
    return (s, n) if count_only else s

# -----------------------
# DataFrame-level ops
# -----------------------
def fill_color_and_size_from_name(articles: pd.DataFrame) -> tuple[pd.DataFrame, dict[str,int]]:
    df = articles.copy()
    for c in ("name","color","size"):
        if c in df.columns:
            df[c] = df[c].astype("string")

    found_colors = df["name"].apply(extract_color) if "name" in df else pd.Series([None]*len(df))
    found_sizes  = df["name"].apply(extract_sizes) if "name" in df else pd.Series([[]]*len(df))

    # color
    color_replacements = 0
    if "color" in df.columns:
        mask_c = df["color"].apply(_is_missing)
        idx_c  = mask_c & found_colors.notna()
        color_replacements = int(idx_c.sum())
        df.loc[idx_c, "color"] = found_colors.loc[idx_c].str.lower()

    # size
    size_replacements = 0
    if "size" in df.columns:
        joined_sizes = found_sizes.apply(lambda xs: " / ".join(xs) if xs else pd.NA)
        mask_s = df["size"].apply(_is_missing)
        idx_s  = mask_s & joined_sizes.notna()
        size_replacements = int(idx_s.sum())
        df.loc[idx_s, "size"] = joined_sizes.loc[idx_s]

    # name clean
    cleaned_and_counts = [clean_name(n, c, count_only=True) for n, c in zip(df.get("name", pd.Series([None]*len(df))), found_colors)]
    if "name" in df.columns:
        df.loc[:, "name"] = [x[0] for x in cleaned_and_counts]
    name_replacement_count = int(sum(x[1] for x in cleaned_and_counts))

    stats = {
        "color_replacements": color_replacements,
        "size_replacements":  size_replacements,
        "name_clean_replacements": name_replacement_count,
    }
    return df, stats

def drop_unwanted_by_name(articles: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = articles.copy()
    mask = df["name"].astype(str).apply(lambda x: bool(_UNWANTED_RE.search(x)))
    removed = int(mask.sum())
    df = df.loc[~mask].reset_index(drop=True)
    return df, removed
