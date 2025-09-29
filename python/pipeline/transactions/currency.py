# python/pipeline/transactions/currency.py
from __future__ import annotations
import pandas as pd
import requests

# Map your platform currencyId → ISO-like country code bucket
CURRENCYID_TO_COUNTRY = {
    "40":  "DK",  # DKK
    "134": "SE",  # SEK
    "103": "NO",  # NOK
    "50":  "FI",  # EUR (Finland)
}

def fix_six_digit_prices(df: pd.DataFrame, *, price_col: str = "price") -> pd.DataFrame:
    out = df.copy()
    s = out[price_col].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    mask = s.str.fullmatch(r"\d{6}")
    out.loc[mask, price_col] = s.loc[mask].str[-3:].astype(float).values
    # ensure the whole column is numeric afterwards
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce").astype("Float64")
    return out


def fetch_sek_rates(timeout: int = 10) -> dict[str, float]:
    """
    Returns rates to convert *1 unit of local currency* to SEK.
    Uses Frankfurter (EUR base), then derives:
      DK: SEK_per_EUR / DKK_per_EUR
      NO: SEK_per_EUR / NOK_per_EUR
      SE: 1.0
      FI: SEK_per_EUR  (EUR→SEK)
    """
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
        "DK": sek_per_eur / dkk_per_eur,  # DKK→SEK
        "NO": sek_per_eur / nok_per_eur,  # NOK→SEK
        "SE": 1.0,                        # SEK→SEK
        "FI": sek_per_eur,                # EUR→SEK
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
    """
    Convert price to SEK using live (or provided) rates.
    - price_col coerced to numeric
    - currencyId mapped via CURRENCYID_TO_COUNTRY
    - outputs nullable Int64 'price_sek' (rounded to 0 decimals)
    When add_cols=True, also adds 'currency_country' and 'sek_rate'.
    """
    out = df.copy()

    # Coerce currencyId to string for mapping
    cur = out[currency_id_col].astype("string").str.strip()
    country = cur.map(CURRENCYID_TO_COUNTRY)

    # Get or fetch rates
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