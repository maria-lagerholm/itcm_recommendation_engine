# pipeline/combine/build_json.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Iterable

import numpy as np
import pandas as pd

NORDICS: list[str] = ["Sweden", "Denmark", "Finland", "Norway"]

#------helpers-----
def status_from_orders(n: int) -> str:
    return "New" if n <= 1 else ("Returning" if n <= 3 else "Loyal")


def mode_or_first(s: pd.Series):
    m = s.mode()
    if not m.empty:
        return m.iat[0]
    s = s.dropna()
    return s.iat[0] if not s.empty else None

def _pick(df: pd.DataFrame, names: Iterable[str], default=None) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([default] * len(df), index=df.index)

def _norm_id(s: pd.Series) -> pd.Series:
    s = s.astype("string[python]").str.strip()
    return s.str.replace(r"\.0+$", "", regex=True)

#------base normalization-----
def _build_base_tx(tx: pd.DataFrame) -> pd.DataFrame:
    country = (
        _pick(tx, ["country", "currency_country", "Country"], "Unknown")
        .astype("string[python]")
        .str.strip()
        .replace({"": pd.NA})
        .fillna("Unknown")
    )
    city = _pick(tx, ["invoiceCity", "city"], "Unknown").astype(object).fillna("Unknown")
    shop = _norm_id(tx["shopUserId"])
    order_id = tx["orderId"].astype("string[python]").str.strip()
    created_raw = tx["created"]
    created = created_raw if np.issubdtype(created_raw.dtype, np.datetime64) else pd.to_datetime(created_raw, errors="coerce")
    if isinstance(created, pd.Series) and np.issubdtype(created.dtype, np.datetime64):
        try:
            if getattr(created.dt, "tz", None) is not None:
                created = created.dt.tz_localize(None)
        except Exception:
            pass
    rev = pd.to_numeric(tx.get("line_total_sek"), errors="coerce").fillna(0)
    typ = tx["type"] if "type" in tx.columns else pd.Series([None] * len(tx), index=tx.index)
    price = tx["price"] if "price" in tx.columns else pd.Series([None] * len(tx), index=tx.index)
    age = (
        pd.to_numeric(_pick(tx, ["Age"], pd.NA), errors="coerce").astype("Float64")
        if "Age" in tx.columns else pd.Series([pd.NA] * len(tx), index=tx.index, dtype="Float64")
    )
    gender = tx["Gender"] if "Gender" in tx.columns else pd.Series([pd.NA] * len(tx), index=tx.index, dtype="object")
    return pd.DataFrame(
        {
            "country": country,
            "city": city,
            "shopUserId": shop,
            "orderId": order_id,
            "rev": rev,
            "created": created,
            "type": typ,
            "price": price,
            "Age": age,
            "Gender": gender,
        }
    )

def _prep_customers(customers: pd.DataFrame) -> pd.DataFrame:
    c = customers.copy()
    c["shopUserId_norm"] = _norm_id(c["shopUserId"])
    c_agg = (
        c.groupby("shopUserId_norm", dropna=False)
        .agg(Age=("Age", mode_or_first), Gender=("Gender", mode_or_first))
        .reset_index()
    )
    return c_agg

#------aggregations-----
def split_nordics(tx: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    base_tx = _build_base_tx(tx)
    return {c: base_tx[base_tx["country"] == c].copy() for c in NORDICS}

def _country_totals(tx: pd.DataFrame) -> Tuple[int, int, int, int | None]:
    total_revenue = int(np.rint(tx["rev"].sum()))
    customers_cnt = int(tx["shopUserId"].nunique())
    total_orders = int(tx["orderId"].nunique())
    aov_country = None if total_orders == 0 else int(round(float(total_revenue) / total_orders))
    return total_revenue, customers_cnt, total_orders, aov_country

def _agg_city(tx: pd.DataFrame):
    agg_city = (
        tx.groupby("city", dropna=False, sort=False)
          .agg(total_revenue_sek=("rev", "sum"), customers_count=("shopUserId", "nunique"))
    )
    agg_city["total_revenue_sek"] = np.rint(agg_city["total_revenue_sek"]).astype("int64")
    agg_city["customers_count"] = agg_city["customers_count"].astype("int64")
    city_orders = tx.groupby("city", dropna=False)["orderId"].nunique().rename("total_orders").astype("int64")
    return agg_city, city_orders

def _agg_customer(tx: pd.DataFrame):
    tx_cust = tx[tx["shopUserId"].notna()].copy()
    agg_customer = (
        tx_cust.groupby(["city", "shopUserId"], dropna=False, sort=False)
        .agg(
            total_spent_sek=("rev", "sum"),
            total_orders=("orderId", "nunique"),
            first_order=("created", "min"),
            last_order=("created", "max"),
            age=("Age", mode_or_first),
            gender=("Gender", mode_or_first),
        )
    )
    agg_customer["total_spent_sek"] = np.rint(agg_customer["total_spent_sek"]).astype("int64")
    return agg_customer.sort_index(level=["city", "shopUserId"])

def _agg_order(tx: pd.DataFrame):
    tx_cust = tx[tx["shopUserId"].notna()].copy()
    agg_order = (
        tx_cust.groupby(["city", "shopUserId", "orderId"], dropna=False, sort=False)
        .agg(
            order_total_sek=("rev", "sum"),
            n_items=("orderId", "size"),
            created=("created", "min"),
            order_type=("type", mode_or_first),
            price=("price", mode_or_first),
        )
    )
    agg_order["order_total_sek"] = np.rint(agg_order["order_total_sek"]).astype("int64")
    return agg_order.sort_index(level=["city", "shopUserId", "orderId"])

def _agg_city_monthly(tx: pd.DataFrame) -> dict[str, dict[str, int]]:
    txm = tx.copy()
    ym = txm["created"]
    if getattr(ym.dt, "tz", None) is not None:
        ym = ym.dt.tz_localize(None)
    txm["year_month"] = ym.dt.to_period("M").astype(str)
    g = (
        txm.groupby(["city", "year_month"], dropna=False)["rev"]
           .sum()
           .round()
           .astype("int64")
    )
    out: dict[str, dict[str, int]] = {}
    for (cty, ym_str), val in g.items():
        ckey = "Unknown" if pd.isna(cty) else str(cty)
        out.setdefault(ckey, {})[ym_str] = int(val)
    return out

def _customers_by_channel(tx: pd.DataFrame) -> dict[str, int]:
    if "type" not in tx.columns:
        return {}
    tmp = tx[["shopUserId", "type"]].dropna().copy()
    tmp["channel"] = tmp["type"]
    g = (
        tmp.dropna(subset=["channel", "shopUserId"])
           .groupby("channel")["shopUserId"]
           .nunique()
           .astype(int)
    )
    return {k: int(v) for k, v in g.items()}

#------items & serialization-----
def _build_items_grouped(
    tx_country: pd.DataFrame,
    tx: pd.DataFrame,
    articles: pd.DataFrame | None = None,
):
    item_cols = [
        "sku", "groupId", "created", "quantity", "price_sek",
        "name", "line_total_sek", "type", "brand", "category", "price",
    ]
    present = [c for c in item_cols if c in tx.columns]
    items_tx = pd.DataFrame(
        {"city": tx_country["city"], "shopUserId": tx_country["shopUserId"], "orderId": tx_country["orderId"]}
    )
    for c in present:
        if c == "created":
            col_created = tx_country["created"]
            try:
                if getattr(col_created.dt, "tz", None) is not None:
                    col_created = col_created.dt.tz_localize(None)
            except Exception:
                pass
            items_tx[c] = col_created
        else:
            items_tx[c] = tx.loc[tx_country.index, c] if c in tx.columns else None
    if articles is not None:
        art = articles.copy()
        for key in ("sku", "groupId"):
            if key in art.columns:
                art[key] = art[key].astype("string[python]").str.strip()
        if "sku" in items_tx.columns:
            items_tx["sku"] = items_tx["sku"].astype("string[python]").str.strip()
        if "groupId" in items_tx.columns:
            items_tx["groupId"] = items_tx["groupId"].astype("string[python]").str.strip()
        if "brand" not in items_tx.columns:
            items_tx["brand"] = pd.NA
        if "category" not in items_tx.columns:
            items_tx["category"] = pd.NA
        if "sku" in art.columns:
            if "brand" in art.columns:
                sku_brand = art.dropna(subset=["sku"]).drop_duplicates("sku").set_index("sku")["brand"]
                items_tx["brand"] = items_tx["brand"].fillna(items_tx.get("sku").map(sku_brand))
            if "category" in art.columns:
                sku_cat = art.dropna(subset=["sku"]).drop_duplicates("sku").set_index("sku")["category"]
                items_tx["category"] = items_tx["category"].fillna(items_tx.get("sku").map(sku_cat))
        if "groupId" in art.columns:
            if "brand" in art.columns:
                gid_brand = art.dropna(subset=["groupId"]).drop_duplicates("groupId").set_index("groupId")["brand"]
                items_tx["brand"] = items_tx["brand"].fillna(items_tx.get("groupId").map(gid_brand))
            if "category" in art.columns:
                gid_cat = art.dropna(subset=["groupId"]).drop_duplicates("groupId").set_index("groupId")["category"]
                items_tx["category"] = items_tx["category"].fillna(items_tx.get("groupId").map(gid_cat))
    items_tx = items_tx[tx_country["shopUserId"].notna()].copy()
    items_tx["city"] = items_tx["city"].fillna("Unknown")
    return items_tx.groupby(["city", "shopUserId", "orderId"], dropna=False, sort=False)

def _item_dict(row: pd.Series):
    cr = row.get("created")
    if isinstance(cr, pd.Timestamp):
        cr = cr.isoformat(sep=" ")
    def nz(v): return None if pd.isna(v) else v
    def to_int(v): return None if pd.isna(v) else int(v)
    def to_float(v): return None if pd.isna(v) else float(v)
    return {
        "sku": nz(row.get("sku")),
        "groupId": nz(row.get("groupId")),
        "created": nz(cr),
        "quantity": to_int(row.get("quantity")),
        "price_sek": to_int(row.get("price_sek")),
        "name": nz(row.get("name")),
        "line_total_sek": to_int(row.get("line_total_sek")),
        "type": nz(row.get("type")),
        "brand": nz(row.get("brand")),
        "category": nz(row.get("category")),
        "price": to_float(row.get("price")),
    }

def export_country_json(
    tx_country: pd.DataFrame,
    tx_full: pd.DataFrame,
    country_name: str,
    out_dir: str | Path = "/workspace/data/processed/processed_staging/parquet_out",
    articles: pd.DataFrame | None = None,
) -> None:
    tx_c = tx_country.copy()
    tx_c["city"] = tx_c["city"].fillna("Unknown")
    total_revenue, customers_cnt, total_orders, aov_country = _country_totals(tx_c)
    agg_city, city_orders = _agg_city(tx_c)
    agg_customer = _agg_customer(tx_c)
    agg_order = _agg_order(tx_c)
    city_monthly_map = _agg_city_monthly(tx_c)
    customers_by_channel = _customers_by_channel(tx_c)
    items_grouped = _build_items_grouped(tx_c, tx_full, articles=articles)
    top_key = country_name.lower()
    result = {
        top_key: {
            "total_revenue_sek": int(total_revenue),
            "customers_count": int(customers_cnt),
            "total_orders": int(total_orders),
            "avg_order_value_sek": aov_country,
            "customers_by_channel": customers_by_channel,
            "cities": {},
        }
    }
    for cty, row in agg_city.iterrows():
        ckey = "Unknown" if pd.isna(cty) else str(cty)
        orders_c = int(city_orders.get(cty, 0))
        rev_c = int(row["total_revenue_sek"])
        aov_c = None if orders_c == 0 else int(round(float(rev_c) / orders_c))
        result[top_key]["cities"][ckey] = {
            "total_revenue_sek": rev_c,
            "customers_count": int(row["customers_count"]),
            "total_orders": orders_c,
            "avg_order_value_sek": aov_c,
            "monthly_revenue_sek": city_monthly_map.get(ckey, {}),
            "customers": {},
        }
    for (cty, uid), row in agg_customer.iterrows():
        status = status_from_orders(int(row["total_orders"]))
        first_iso = row["first_order"].isoformat(sep=" ") if pd.notna(row["first_order"]) else None
        last_iso = row["last_order"].isoformat(sep=" ") if pd.notna(row["last_order"]) else None
        age_val = None
        if "age" in row and pd.notna(row["age"]):
            try:
                age_val = int(row["age"])
            except Exception:
                age_val = None
        gender_val = None if "gender" not in row or pd.isna(row["gender"]) else str(row["gender"])
        cust_node = {
            "summary": {
                "total_orders": int(row["total_orders"]),
                "total_spent_sek": int(row["total_spent_sek"]),
                "first_order": first_iso,
                "last_order": last_iso,
                "status": status,
                "age": age_val,
                "gender": gender_val,
            },
            "orders": {},
        }
        try:
            cust_orders = agg_order.loc[(cty, uid)]
            if isinstance(cust_orders, pd.Series):
                cust_orders = cust_orders.to_frame().T
            for oid, orow in cust_orders.iterrows():
                try:
                    items_for_order = items_grouped.get_group((cty, uid, oid))
                    items = [_item_dict(r) for _, r in items_for_order.iterrows()]
                except KeyError:
                    items = []
                cust_node["orders"][str(oid)] = {
                    "created": orow["created"].isoformat(sep=" ") if pd.notna(orow["created"]) else None,
                    "order_total_sek": int(orow["order_total_sek"]),
                    "n_items": int(orow["n_items"]),
                    "order_type": None if pd.isna(orow["order_type"]) else orow["order_type"],
                    "price": None if pd.isna(orow.get("price")) else float(orow.get("price")),
                    "items": items,
                }
        except KeyError:
            pass
        ckey = "Unknown" if pd.isna(cty) else str(cty)
        result[top_key]["cities"][ckey]["customers"][str(uid)] = cust_node
    out_path = Path(out_dir) / f"{country_name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)