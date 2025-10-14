import re
import pandas as pd

_NULL_RE = re.compile(r"^(?:|nan|null|none|n/?a|n\.a\.|-|\.|0)$", re.IGNORECASE)

def clean_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.normalize("NFKC").str.replace(r"\s+", " ", regex=True).str.strip()
    return s.mask(s.str.fullmatch(_NULL_RE, na=False))

def clean_customer_name_fields(
    df: pd.DataFrame,
    cols: tuple[str, ...] = ("shopUserId", "invoiceFirstName", "invoiceLastName", "invoiceZip"),
) -> pd.DataFrame:
    out = df.copy()
    existing = [c for c in cols if c in out]
    if not existing:
        return out
    out[existing] = out[existing].apply(clean_series)
    return out
