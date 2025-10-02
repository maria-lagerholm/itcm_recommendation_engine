from __future__ import annotations
import pandas as pd
import requests

# --- Live rates with fallback (SEK per unit) ---------------------------------
def fetch_sek_rates_from_frankfurter(timeout: int = 8) -> dict[str, float | str]:
    fallback: dict[str, float | str] = {"EUR": 11.50, "NOK": 1.00, "DKK": 1.55, "asof": "fallback"}
    try:
        r = requests.get(
            "https://api.frankfurter.app/latest",
            params={"base": "EUR", "symbols": "SEK,NOK,DKK"},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        eur = float(data["rates"]["SEK"])
        nok = eur / float(data["rates"]["NOK"])
        dkk = eur / float(data["rates"]["DKK"])
        return {"EUR": eur, "NOK": nok, "DKK": dkk, "asof": data.get("date", "unknown")}
    except Exception:
        return fallback

# --- Helpers -----------------------------------------------------------------
def _to_float_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "fi":
        return s.astype("float64")
    s = (
        s.astype("string")
         .str.strip()
         .str.replace("\u00A0", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s, errors="coerce").astype("float64")

# --- Main transform -----------------------------------------------------------
def fill_priceSEK_no_decimals(
    articles: pd.DataFrame,
    *,
    price_cols: tuple[str, ...] = ("priceSEK", "priceEUR", "priceNOK", "priceDKK"),
    fetch_timeout: int = 8,
    rates: dict[str, float] | None = None,
    sku_col: str = "sku",
    overrides_priceSEK: dict[str, int | str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    pSEK, pEUR, pNOK, pDKK = price_cols
    df = articles.copy()

    for col in (pSEK, pEUR, pNOK, pDKK):
        if col not in df.columns:
            df[col] = pd.NA

    if rates is None:
        r = fetch_sek_rates_from_frankfurter(timeout=fetch_timeout)
        rates_used = {"EUR": float(r["EUR"]), "NOK": float(r["NOK"]), "DKK": float(r["DKK"])}
        asof = str(r.get("asof", "unknown"))
    else:
        rates_used = {k: float(v) for k, v in rates.items()}
        asof = "provided"

    mask = df[pSEK].isna()
    initial_missing = int(mask.sum())
    filled_rows_before_overrides = 0

    if mask.any():
        eur_sek = _to_float_series(df.loc[mask, pEUR]) * rates_used["EUR"]
        nok_sek = _to_float_series(df.loc[mask, pNOK]) * rates_used["NOK"]
        dkk_sek = _to_float_series(df.loc[mask, pDKK]) * rates_used["DKK"]

        cand = eur_sek.fillna(nok_sek).fillna(dkk_sek)
        df.loc[mask, pSEK] = cand.round(0).astype("Int64").astype("string")
        filled_rows_before_overrides = int(df.loc[mask, pSEK].notna().sum())

    df[pSEK] = df[pSEK].astype("string")

    overrides_applied = 0
    if overrides_priceSEK:
        if sku_col not in df.columns:
            raise KeyError(f"SKU column '{sku_col}' not found in DataFrame.")
        skus = df[sku_col].astype("string").str.lower()
        ov = {
            str(k).lower(): (pd.NA if pd.isna(v) else str(int(v)))
            for k, v in overrides_priceSEK.items()
        }
        ov_series = skus.map(ov)
        idx = ov_series.notna()
        df.loc[idx, pSEK] = ov_series[idx]
        overrides_applied = int(idx.sum())

    stats = {
        "asof": asof,
        "initial_missing_priceSEK": initial_missing,
        "filled_rows": filled_rows_before_overrides,
        "overrides_applied": overrides_applied,
        "still_missing_priceSEK": int(df[pSEK].isna().sum()),
        "unique_priceSEK_nonnull": int(df[pSEK].dropna().nunique()),
    }
    return df.reset_index(drop=True), stats