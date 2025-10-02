# python/pipeline/customers/name_last_name.py

#------constants-----
from __future__ import annotations
import re
import pandas as pd

S = pd.StringDtype()
_NULL_RE = re.compile(r"(?i)^(?:|nan|null|none|n/?a|n\.a\.|-|\.|0)$")

#------clean series-----
def clean_series(s: pd.Series) -> pd.Series:
    s = s.astype(S)
    s = s.str.normalize("NFKC").str.replace(r"\s+", " ", regex=True).str.strip()
    return s.mask(s.str.fullmatch(_NULL_RE, na=False))

#------clean customer name fields-----
def clean_customer_name_fields(
    df: pd.DataFrame,
    cols: tuple[str, ...] = ("shopUserId", "invoiceFirstName", "invoiceLastName", "invoiceZip"),
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = clean_series(out[c])
    return out