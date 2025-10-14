import pandas as pd
import requests

def fetch_sek_rates_from_frankfurter(timeout: int = 8) -> dict[str, float | str]:
    fb = {"EUR": 11.50, "NOK": 1.00, "DKK": 1.55, "asof": "fallback"}
    try:
        r = requests.get("https://api.frankfurter.app/latest",
                         params={"base": "EUR", "symbols": "SEK,NOK,DKK"},
                         timeout=timeout)
        r.raise_for_status()
        data = r.json()
        eur = float(data["rates"]["SEK"])
        nok = eur / float(data["rates"]["NOK"])
        dkk = eur / float(data["rates"]["DKK"])
        return {"EUR": eur, "NOK": nok, "DKK": dkk, "asof": data.get("date", "unknown")}
    except Exception:
        return fb

def _to_float_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "fi":
        return s.astype("float64")
    s = (s.astype("string").str.strip()
         .str.replace("\u00A0", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce").astype("float64")

def fill_priceSEK_no_decimals(
    articles: pd.DataFrame,
    *,
    price_cols=("priceSEK", "priceEUR", "priceNOK", "priceDKK"),
    fetch_timeout: int = 8,
    rates: dict[str, float] | None = None,
    sku_col: str = "sku",
    overrides_priceSEK: dict[str, int | str] | None = None,
) -> pd.DataFrame:
    pSEK, pEUR, pNOK, pDKK = price_cols
    df = articles.copy()

    for c in price_cols:
        if c not in df:
            df[c] = pd.NA

    r = rates or fetch_sek_rates_from_frankfurter(timeout=fetch_timeout)
    R = {"EUR": float(r["EUR"]), "NOK": float(r["NOK"]), "DKK": float(r["DKK"])}

    cand = (_to_float_series(df[pEUR]) * R["EUR"]) \
        .combine_first(_to_float_series(df[pNOK]) * R["NOK"]) \
        .combine_first(_to_float_series(df[pDKK]) * R["DKK"])

    existing = pd.to_numeric(df[pSEK], errors="coerce")
    df[pSEK] = existing.combine_first(cand).round(0).astype("Int64")

    if overrides_priceSEK:
        if sku_col not in df.columns:
            raise KeyError(f"SKU column '{sku_col}' not found in DataFrame.")
        ov_map = {str(k).lower(): v for k, v in overrides_priceSEK.items()}
        ov_series = df[sku_col].astype("string").str.lower().map(ov_map)
        idx = ov_series.notna()
        df.loc[idx, pSEK] = pd.to_numeric(ov_series[idx], errors="coerce").round(0).astype("Int64")

    return df.reset_index(drop=True)
