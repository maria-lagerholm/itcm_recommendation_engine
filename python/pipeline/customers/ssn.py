# python/pipeline/customers/ssn.py
from __future__ import annotations
import re
from datetime import date
import pandas as pd

S = pd.StringDtype()

COUNTRY_MAP = {"58": "DK", "160": "NO", "205": "SE", "72": "FI"}

def _safe_date(y, m, d):
    try:
        return date(int(y), int(m), int(d))
    except Exception:
        return None

def _age_from_birthdate(born):
    if not born:
        return None
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

def _parse_birthdate(ssn_str: str, country: str):
    if pd.isna(ssn_str):
        return None
    digits = re.sub(r"\D", "", str(ssn_str))

    if country == "SE":
        if len(digits) >= 12:  # YYYYMMDDxxxx
            return _safe_date(digits[:4], digits[4:6], digits[6:8])
        if len(digits) >= 10:  # YYMMDDxxxx (+/- century logic)
            yy, mm, dd = int(digits[:2]), digits[2:4], digits[4:6]
            sep = "-" if "-" in str(ssn_str) else "+" if "+" in str(ssn_str) else None
            this_year = date.today().year
            if sep == "+":  # 100+ years old
                y = 1900 + yy if (1900 + yy) <= this_year - 100 else 1800 + yy
            else:
                y = 1900 + yy if (1900 + yy) > this_year - 100 else 2000 + yy
            return _safe_date(y, mm, dd)
        return None

    if country == "NO" and len(digits) == 11:
        dd, mm, yy = int(digits[0:2]), int(digits[2:4]), int(digits[4:6])
        individ = int(digits[6:9])
        if dd > 40:  # D-number
            dd -= 40
        if 0 <= individ <= 499:
            year = 1900 + yy
        elif 500 <= individ <= 749 and 54 <= yy <= 99:
            year = 1800 + yy
        elif 500 <= individ <= 999 and 0 <= yy <= 39:
            year = 2000 + yy
        elif 900 <= individ <= 999 and 40 <= yy <= 99:
            year = 1900 + yy
        else:
            year = (2000 + yy) if yy <= 24 else (1900 + yy)
        return _safe_date(year, mm, dd)

    if country == "DK" and len(digits) >= 10:
        dd, mm, yy = digits[0:2], digits[2:4], int(digits[4:6])
        year = (2000 + yy) if yy <= 24 else (1900 + yy)
        return _safe_date(year, mm, dd)

    if country == "FI":
        m = re.match(r"^(\d{2})(\d{2})(\d{2})([-+A])(\d{3})\w?$", str(ssn_str).strip(), re.I)
        if m:
            dd, mm, yy, cent = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4).upper()
            base = {"+": 1800, "-": 1900, "A": 2000}[cent]
            return _safe_date(base + yy, mm, dd)
        if len(digits) >= 10:  # numeric-only fallback
            dd, mm, yy = int(digits[0:2]), int(digits[2:4]), int(digits[4:6])
            year = (2000 + yy) if yy <= 24 else (1900 + yy)
            return _safe_date(year, mm, dd)
        return None

    return None

def get_gender_age_from_ssn(ssn, country_id):
    if pd.isna(ssn):
        return None, None
    ssn_str = str(ssn).strip()
    country = COUNTRY_MAP.get(str(country_id))
    if not country:
        return None, None

    digits = re.sub(r"\D", "", ssn_str)
    gender_digit = None
    if country == "SE" and len(digits) >= 10:
        gender_digit = int(digits[-4:][2])
    elif country == "NO" and len(digits) == 11:
        gender_digit = int(digits[8])
    elif country == "DK" and len(digits) >= 10:
        gender_digit = int(digits[-1])
    elif country == "FI":
        m = re.match(r"^\d{6}[-+A]\d{3}\w?$", ssn_str, re.I)
        if m:
            gender_digit = int(digits[8])
        elif len(digits) >= 10:
            gender_digit = int(digits[8])

    gender = None
    if gender_digit is not None:
        gender = "Male" if gender_digit % 2 else "Female"
    age = _age_from_birthdate(_parse_birthdate(ssn_str, country))
    return gender, age

def derive_gender_age(df: pd.DataFrame,
                      *, ssn_col="invoiceSSN", country_col="invoiceCountryId",
                      gender_col="Gender", age_col="Age") -> pd.DataFrame:
    """Return a copy with Gender/Age columns populated from SSN and country."""
    out = df.copy()
    out[[gender_col, age_col]] = out.apply(
        lambda r: pd.Series(get_gender_age_from_ssn(r[ssn_col], r[country_col])),
        axis=1
    )
    return out

def filter_age_range(df: pd.DataFrame, *, age_col="Age", lo=10, hi=105) -> pd.DataFrame:
    """Keep rows with Age between lo and hi inclusive, or Age is NA."""
    mask = df[age_col].isna() | ((df[age_col] >= lo) & (df[age_col] <= hi))
    return df.loc[mask].reset_index(drop=True)
