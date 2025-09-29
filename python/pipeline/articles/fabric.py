from __future__ import annotations
import re
import unicodedata
import pandas as pd

# --- 1) Fabric pairs (svenska → engelska). Add your full list as needed. ---
FABRIC_PAIRS: list[tuple[str, str]] = [
    # Naturfibrer
    ("bomull", "cotton"), ("linne", "linen"), ("ull", "wool"),
    ("kashmir", "cashmere"), ("merinoull", "merino wool"),
    ("alpaca", "alpaca"), ("mohair", "mohair"), ("silke", "silk"),
    ("hampa", "hemp"), ("jute", "jute"), ("raffia", "raffia"),
    ("bambu", "bamboo fabric"), ("bananfiber", "banana fiber"),
    ("kokosfiber", "coir"),
    # Regenererade / syntet
    ("polyester", "polyester"), ("nylon", "nylon"), ("akryl", "acrylic"),
    ("spandex", "spandex"), ("elastan", "elastane"), ("acetat", "acetate"),
    ("viskos", "viscose"), ("modal", "modal"), ("lyocell", "lyocell"),
    ("triacetat", "triacetate"), ("neopren", "neoprene"), ("mikrofiber", "microfiber"),
    # Tygtyper
    ("frotté", "terry cloth"), ("satin", "satin"), ("sammet", "velvet"),
    ("flanell", "flannel"), ("denim", "denim"), ("manchester", "corduroy"),
    ("fleece", "fleece"), ("jersey", "jersey knit"), ("interlock", "interlock knit"),
    ("trikå", "tricot knit"), ("gabardin", "gabardine"), ("tweed", "tweed"),
    ("canvas", "canvas"), ("poplin", "poplin"), ("oxford", "oxford cloth"),
    ("seersucker", "seersucker"), ("taft", "taffeta"), ("tyll", "tulle"),
    ("mesh", "mesh"), ("organza", "organza"), ("brokad", "brocade"),
    ("batist", "batiste"), ("crêpe", "crepe"), ("georgette", "georgette"),
    ("muslin", "muslin"),
    # Broderi/väv
    ("färgtryckt väv", "printed weave"), ("bakgrundstryckt", "background printed fabric"),
    ("stramalj", "mono canvas"), ("aida", "aida cloth"),
    ("ritade broderier", "pre-printed embroidery fabric"),
    ("ullgarn", "wool yarn"), ("alpackagarn", "alpaca yarn"), ("rya", "rya"),
    # Läder/päls
    ("skinn", "leather"), ("mocka", "suede"), ("päls", "fur"), ("fuskpäls", "faux fur"),
    # Tekniska
    ("softshell", "softshell"), ("gore-tex", "gore-tex"), ("kevlar", "kevlar"), ("tyvek", "tyvek"),
    # Traditionella/etniska
    ("batik", "batik"), ("ikat", "ikat"), ("khadi", "khadi"), ("madras", "madras"),
    ("shantung", "shantung silk"), ("dupion", "dupioni silk"),
    ("kente", "kente cloth"), ("ankara", "ankara / african wax print"),
    ("tartan", "tartan / plaid"), ("paisley-tyg", "paisley fabric"),
    ("bogolan", "mud cloth"), ("adire", "adire cloth"),
]

# --- 2) Synonyms/alias → canonical Swedish name ---
SYNONYMS: dict[str, str] = {
    "läder": "skinn",
    "leather": "skinn",
    "suede": "mocka",
    "rayon": "viskos",
    "lycra": "elastan",
    "tencel": "lyocell",
    "corduroy": "manchester",
    "oxford cloth": "oxford",
    "jersey": "jersey", "jersey knit": "jersey",
    "tricot": "trikå",
    "neoprene": "neopren",
    "microfiber": "mikrofiber",
    "faux fur": "fuskpäls",
    "fur": "päls",
    "wool": "ull",
    "alpaca": "alpaca",
    "wool yarn": "ullgarn",
    "alpaca yarn": "alpackagarn",
    "printed weave": "färgtryckt väv",
    "background printed fabric": "bakgrundstryckt",
    "aida cloth": "aida",
    "pre-printed embroidery fabric": "ritade broderier",
    "mono canvas": "stramalj",
    "terry cloth": "frotté",
    "coir": "kokosfiber",
    "gore tex": "gore-tex",
    "goretex": "gore-tex",
}

# ---------- Build canonical alias map ----------
_alias_to_canon: dict[str, str] = {}

def _add_alias(alias: str, canonical: str) -> None:
    alias = alias.strip().lower()
    if alias:
        _alias_to_canon[alias] = canonical

for sv, en in FABRIC_PAIRS:
    sv_can = sv.strip().lower()
    _add_alias(sv_can, sv_can)                 # Swedish canonical
    _add_alias(en.strip().lower(), sv_can)     # English alias → Swedish canonical

for alias, canon in SYNONYMS.items():
    _add_alias(alias, canon.strip().lower())

# Optional: Swedish→English map
SV_TO_EN = {sv.strip().lower(): en.strip().lower() for sv, en in FABRIC_PAIRS}

# ---------- Helpers ----------
def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return s
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def _alias_to_fragment(alias: str) -> str:
    """
    Convert alias like 'gore tex' / 'gore-tex' → regex allowing optional space/hyphen.
    """
    parts = re.split(r"[ \-]+", alias)
    parts = [p for p in parts if p]
    return r"[ -]?".join(re.escape(p) for p in parts) if parts else ""

# ---------- Build union regex once ----------
_alias_items: list[tuple[str, str, str, int, int]] = []
for alias, canon in _alias_to_canon.items():
    frag = _alias_to_fragment(alias)
    if not frag:
        continue
    token_cnt = len([p for p in re.split(r"[ \-]+", alias) if p])
    raw_len = len(alias.replace(" ", "").replace("-", ""))
    _alias_items.append((canon, alias, frag, token_cnt, raw_len))

seen: set[tuple[str, str]] = set()
uniq_items: list[tuple[str, str, str, int, int]] = []
for canon, alias, frag, token_cnt, raw_len in _alias_items:
    key = (canon, frag)
    if key in seen:
        continue
    seen.add(key)
    uniq_items.append((canon, alias, frag, token_cnt, raw_len))

# Sort: prefer multi-word & longer aliases to reduce overlaps
uniq_items.sort(key=lambda x: (-x[3], -x[4], x[1]))

ALNUM = "0-9a-z"   # after strip_accents
SUFFIX = r"(?:er|ar|or|en|et|n|s|es)?"

_group_to_canon: dict[str, str] = {}
_group_patterns: list[str] = []
for i, (canon, _alias, frag, *_rest) in enumerate(uniq_items):
    gname = f"g{i}"
    _group_to_canon[gname] = canon
    _group_patterns.append(fr"(?P<{gname}>{frag}{SUFFIX})")

UNION_PATTERN = re.compile(
    rf"(?<![{ALNUM}])(?:{'|'.join(_group_patterns)})(?![{ALNUM}])",
    flags=re.IGNORECASE,
)

# ---------- Core extractor ----------
def extract_fabrics(desc: str | None, *, return_empty_list: bool = False):
    """
    Return list of canonical Swedish fabric names in text order (deduped).
    If no match: pd.NA (default) or [] when return_empty_list=True.
    """
    if pd.isna(desc) or not isinstance(desc, str):
        return [] if return_empty_list else pd.NA

    text = strip_accents(desc.lower())
    matches: list[tuple[int, int, str]] = []
    for m in UNION_PATTERN.finditer(text):
        canon = _group_to_canon[m.lastgroup]
        s, e = m.span()
        matches.append((s, e, canon))

    if not matches:
        return [] if return_empty_list else pd.NA

    # Resolve overlaps: keep earliest; prefer longer span; dedup canonical
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    accepted: list[tuple[int, int, str]] = []
    seen_canon: set[str] = set()
    for s, e, c in matches:
        if any(not (e <= s2 or s >= e2) for s2, e2, _ in accepted):
            continue
        if c in seen_canon:
            continue
        accepted.append((s, e, c))
        seen_canon.add(c)

    accepted.sort(key=lambda x: x[0])
    out = [c for _, _, c in accepted]
    return out if out else ([] if return_empty_list else pd.NA)

# ---------- DataFrame-level helper ----------
def enrich_fabrics(
    articles: pd.DataFrame,
    *,
    desc_col: str = "description",
    list_col: str = "fabrics",
    primary_col: str = "fabric_primary",
    english_list_col: str = "fabrics_en",
    return_empty_list: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Adds:
      - list_col: list of canonical Swedish fabrics found in description
      - primary_col: first match (text order)
      - english_list_col: parallel English list (optional)
    Returns (df, stats)
    """
    df = articles.copy()
    if desc_col not in df.columns:
        return df, {"rows_scanned": 0, "with_any_fabric": 0, "with_primary": 0}

    # ensure string dtype for description
    df[desc_col] = df[desc_col].astype("string")

    df[list_col] = df[desc_col].apply(lambda x: extract_fabrics(x, return_empty_list=return_empty_list))
    df[primary_col] = df[list_col].apply(lambda xs: xs[0] if isinstance(xs, list) and xs else pd.NA)

    if english_list_col:
        df[english_list_col] = df[list_col].apply(
            lambda xs: ([SV_TO_EN.get(sv, sv) for sv in xs] if isinstance(xs, list) and xs else ( [] if return_empty_list else pd.NA))
        )

    stats = {
        "rows_scanned": int(len(df)),
        "with_any_fabric": int(df[list_col].apply(lambda v: isinstance(v, list) and len(v) > 0).sum()),
        "with_primary": int(df[primary_col].notna().sum()),
    }
    return df, stats