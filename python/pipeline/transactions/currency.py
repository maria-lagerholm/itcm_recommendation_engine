from __future__ import annotations
import pandas as pd
import requests

CURRENCYID_TO_COUNTRY = {
    "40":  "DK",
    "134": "SE",
    "103": "NO",
    "50":  "FI",
}

def fix_six_digit_prices(df: pd.DataFrame, *, price_col: str = "price") -> pd.DataFrame:
    out = df.copy()
    s = out[price_col].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    mask = s.str.fullmatch(r"\d{6}")
    out.loc[mask, price_col] = s.loc[mask].str[-3:].astype(float).values
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce").astype("Float64")
    return out

def fetch_sek_rates(timeout: int = 10) -> dict[str, float]:
    resp = requests.get(
        "https://api.frankfurter.app/latest",
        params={"from": "EUR", "to": "SEK,DKK,NOK"},
        timeout=timeout,
    )
    resp.raise_for_status()
    rates = resp.json()["rates"]
    sek_per_eur = float(rates["SEK"])
    dkk_per_eur = float(rates["DKK"])
    nok_per_eur = float(rates["NOK"])
    return {
        "DK": sek_per_eur / dkk_per_eur,
        "NO": sek_per_eur / nok_per_eur,
        "SE": 1.0,
        "FI": sek_per_eur,
    }

def unify_price_to_sek(
    df: pd.DataFrame,
    *,
    price_col: str = "price",
    currency_id_col: str = "currencyId",
    out_col: str = "price_sek",
    add_cols: bool = True,
    rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    cur = out[currency_id_col].astype("string").str.strip()
    country = cur.map(CURRENCYID_TO_COUNTRY)
    if rates is None:
        rates = fetch_sek_rates()
    rate = country.map(rates)
    price = pd.to_numeric(out[price_col], errors="coerce")
    price_sek = (price * rate).round(0).astype("Int64")
    if add_cols:
        out["currency_country"] = country
        out["sek_rate"] = rate
    out[out_col] = price_sek
    return out