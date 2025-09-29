# python/pipeline/customers/name_last_name.py
from __future__ import annotations
import re
import pandas as pd

S = pd.StringDtype()
_NULL_RE = re.compile(r"(?i)^(?:|nan|null|none|n/?a|n\.a\.|-|\.|0)$")

def clean_series(s: pd.Series) -> pd.Series:
    """Normalize, collapse whitespace, trim, and mask null-ish tokens."""
    s = s.astype(S)
    s = s.str.normalize("NFKC").str.replace(r"\s+", " ", regex=True).str.strip()
    return s.mask(s.str.fullmatch(_NULL_RE, na=False))

def clean_customer_name_fields(
    df: pd.DataFrame,
    cols: tuple[str, ...] = ("shopUserId", "invoiceFirstName", "invoiceLastName", "invoiceZip"),
) -> pd.DataFrame:
    """Return a copy with selected columns cleaned if present."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = clean_series(out[c])
    return out