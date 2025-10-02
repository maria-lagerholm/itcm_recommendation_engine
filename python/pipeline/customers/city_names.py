from __future__ import annotations
import re
import pandas as pd

#------constants-----
S = pd.StringDtype()

COUNTRY_TAIL_RE = re.compile(
    r'(?:,\s*)?(Denmark|Danmark|Sweden|Sverige|Norway|Norge|Finland|Suomi)\s*$',
    re.IGNORECASE,
)
LEADING_POSTAL_RE = re.compile(
    r'^(?:[A-Z]{1,3}[-\s])?\d{2,3}\s?\d{2,3}\s+|^(?:[A-Z]{1,3}[-\s])?\d{3,6}\s+',
    re.IGNORECASE,
)
DIGITS_ANYWHERE_RE = re.compile(r'\d+')

#------sentence case-----
def _sentence_case_series(s: pd.Series) -> pd.Series:
    first = s.str.replace(r'^(.)', lambda m: m.group(1).upper(), regex=True)
    rest_lowered = first.str.replace(r'(?<=.)(.*)', lambda m: m.group(1).lower(), regex=True)
    return rest_lowered

#------main cleaning-----
def clean_city_series(s: pd.Series) -> pd.Series:
    s = s.astype(S)
    s = s.fillna("Unknown")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip(" ,")
    s = s.mask(s.eq(""), "Unknown")
    s = s.str.replace(LEADING_POSTAL_RE, "", regex=True).str.strip(" ,")
    s = s.str.replace(COUNTRY_TAIL_RE, "", regex=True).str.strip(" ,")
    s = s.str.strip()
    s = s.str.replace(DIGITS_ANYWHERE_RE, "", regex=True)
    s = s.str.strip()
    def _hyphen_rule(val: str) -> str:
        if "-" in val:
            parts = [p.strip() for p in val.split("-") if p.strip()]
            if len(parts) > 1:
                return parts[-1]
        return val
    s = s.apply(_hyphen_rule)
    s = s.str.replace(r"\b\w\b$", "", regex=True).str.strip(" ,")
    s = s.apply(lambda x: " ".join(w for w in x.split() if len(w) > 1)).str.strip(" ,")
    s = _sentence_case_series(s)
    s = s.where(~s.isin({"", "<NA>", "nan", "NaN"}), other="Unknown")
    return s