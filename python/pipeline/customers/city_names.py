import re
import pandas as pd

COUNTRY_TAIL_RE = re.compile(r'(?:,\s*)?(Denmark|Danmark|Sweden|Sverige|Norway|Norge|Finland|Suomi)\s*$', re.IGNORECASE)
LEADING_POSTAL_RE = re.compile(r'^(?:[A-Z]{1,3}[-\s])?\d{2,3}\s?\d{2,3}\s+|^(?:[A-Z]{1,3}[-\s])?\d{3,6}\s+', re.IGNORECASE)
DIGITS_ANYWHERE_RE = re.compile(r'\d+')

def clean_city_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna("Unknown")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip(" ,")
    s = s.mask(s.eq(""), "Unknown")
    s = s.str.replace(LEADING_POSTAL_RE, "", regex=True)
    s = s.str.replace(COUNTRY_TAIL_RE, "", regex=True)
    s = s.str.replace(DIGITS_ANYWHERE_RE, "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip(" ,")
    s = s.str.replace(r".*-\s*([^\-\s].+)$", r"\1", regex=True)
    s = s.str.replace(r"\b\w\b", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip(" ,")
    s = s.str.capitalize()
    s = s.where(~s.isin({"", "<NA>", "nan", "NaN"}), "Unknown")
    return s
