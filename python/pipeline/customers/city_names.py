from __future__ import annotations
import re
import pandas as pd

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

def _sentence_case_series(s: pd.Series) -> pd.Series:
    # Match behavior of sentense_case(): strip already handled upstream.
    # If empty -> stay empty; else first char upper, rest lower.
    first = s.str.replace(r'^(.)', lambda m: m.group(1).upper(), regex=True)
    rest_lowered = first.str.replace(r'(?<=.)(.*)', lambda m: m.group(1).lower(), regex=True)
    return rest_lowered

def clean_city_series(s: pd.Series) -> pd.Series:
    """
    Vectorized equivalent to `normalize_city`, step-for-step.
    """
    s = s.astype(S)

    # None/NA -> "Unknown"
    s = s.fillna("Unknown")

    # Collapse whitespace, trim spaces/commas
    s = s.str.replace(r"\s+", " ", regex=True).str.strip(" ,")

    # If empty -> "Unknown"
    s = s.mask(s.eq(""), "Unknown")

    # Remove leading postal and trailing country
    s = s.str.replace(LEADING_POSTAL_RE, "", regex=True).str.strip(" ,")
    s = s.str.replace(COUNTRY_TAIL_RE, "", regex=True).str.strip(" ,")

    # Strip again
    s = s.str.strip()

    # Remove digits anywhere
    s = s.str.replace(DIGITS_ANYWHERE_RE, "", regex=True)

    # Strip again
    s = s.str.strip()

    # Hyphen handling: only choose the last NON-EMPTY part if there are >= 2 non-empty parts
    # (exactly like the scalar implementation).
    def _hyphen_rule(val: str) -> str:
        if "-" in val:
            parts = [p.strip() for p in val.split("-") if p.strip()]
            if len(parts) > 1:
                return parts[-1]
        return val
    s = s.apply(_hyphen_rule)

    # Remove trailing one-letter word
    s = s.str.replace(r"\b\w\b$", "", regex=True).str.strip(" ,")

    # Remove any remaining one-letter words (keep words with len > 1)
    s = s.apply(lambda x: " ".join(w for w in x.split() if len(w) > 1)).str.strip(" ,")

    # Sentence case (scalar version strips before casing; we already stripped)
    s = _sentence_case_series(s)

    # Final mapping to "Unknown"
    s = s.where(~s.isin({"", "<NA>", "nan", "NaN"}), other="Unknown")

    return s