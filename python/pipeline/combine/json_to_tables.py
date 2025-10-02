# pipeline/combine/json_to_tables.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
import pandas as pd

# ---- Stable column schemas ----
CS_COUNTRY = ["country","total_revenue_sek","customers_count","total_orders","avg_order_value_sek"]
CS_CITY    = ["country","city","total_revenue_sek","customers_count","total_orders","avg_order_value_sek"]
CS_CUST    = ["country","city","customer_id","total_orders","total_spent_sek","first_order","last_order","status","age","gender"]
CS_ORDERS  = ["country","city","customer_id","order_id","created","order_total_sek","n_items","order_type","price"]
CS_ITEMS   = ["country","city","customer_id","order_id","sku","groupId","created","quantity","price_sek","name","line_total_sek","type","brand","category","price"]
CS_CITY_MONTHLY = ["country","city","year_month","total_revenue_sek"]
CS_COUNTRY_CHANNEL = ["country","channel","customers_count"]

# ---- I/O helpers ----
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_parquet(tx: pd.DataFrame, path: Path) -> None:
    # Assumes parent directory exists (per your setup)
    tx.to_parquet(path, index=False)

# ---- DataFrame utilities ----
def _ensure(tx: pd.DataFrame | None, cols: List[str]) -> pd.DataFrame:
    if tx is None or tx.empty:
        return pd.DataFrame({c: pd.Series([], dtype="object") for c in cols})[cols]
    for c in cols:
        if c not in tx.columns:
            tx[c] = pd.NA
    return tx[cols]

# ---- JSON shape helpers ----
def _unwrap(obj: dict, hint: str) -> tuple[str, dict]:
    if isinstance(obj, dict) and len(obj) == 1 and isinstance(next(iter(obj.values())), dict):
        k = next(iter(obj.keys()))
        return k.capitalize(), next(iter(obj.values()))
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON must be an object/dict")
    return hint, obj

# ---- Flatten country JSON → row buckets ----
def flatten_country(obj: dict, country_hint: str) -> dict[str, list[dict]]:
    country, root = _unwrap(obj, country_hint)
    out = {
        "country_summary": [{
            "country": country,
            "total_revenue_sek": root.get("total_revenue_sek"),
            "customers_count": root.get("customers_count"),
            "total_orders": root.get("total_orders"),
            "avg_order_value_sek": root.get("avg_order_value_sek"),
        }],
        "country_channels": [],
        "city_summary": [],
        "city_monthly": [],
        "customer_summary": [],
        "orders": [],
        "order_items": [],
    }

    for ch, cnt in (root.get("customers_by_channel") or {}).items():
        out["country_channels"].append({
            "country": country,
            "channel": ch,
            "customers_count": cnt,
        })

    for city, cnode in (root.get("cities") or {}).items():
        out["city_summary"].append({
            "country": country, "city": city,
            "total_revenue_sek": cnode.get("total_revenue_sek"),
            "customers_count": cnode.get("customers_count"),
            "total_orders": cnode.get("total_orders"),
            "avg_order_value_sek": cnode.get("avg_order_value_sek"),
        })

        for ym, rev in (cnode.get("monthly_revenue_sek") or {}).items():
            out["city_monthly"].append({
                "country": country, "city": city,
                "year_month": ym,
                "total_revenue_sek": rev,
            })

        for cust_id, cst in (cnode.get("customers") or {}).items():
            summ = cst.get("summary") or {}
            out["customer_summary"].append({
                "country": country, "city": city, "customer_id": cust_id,
                "total_orders": summ.get("total_orders"),
                "total_spent_sek": summ.get("total_spent_sek"),
                "first_order": summ.get("first_order"),
                "last_order": summ.get("last_order"),
                "status": summ.get("status"),
                "age": summ.get("age"),
                "gender": summ.get("gender"),
            })
            for order_id, ordn in (cst.get("orders") or {}).items():
                out["orders"].append({
                    "country": country, "city": city, "customer_id": cust_id, "order_id": order_id,
                    "created": ordn.get("created"),
                    "order_total_sek": ordn.get("order_total_sek"),
                    "n_items": ordn.get("n_items"),
                    "order_type": ordn.get("order_type"),
                    "price": ordn.get("price"),
                })
                for it in (ordn.get("items") or []):
                    out["order_items"].append({
                        "country": country, "city": city, "customer_id": cust_id, "order_id": order_id,
                        "sku": it.get("sku"), "groupId": it.get("groupId"), "created": it.get("created"),
                        "quantity": it.get("quantity"), "price_sek": it.get("price_sek"), "name": it.get("name"),
                        "line_total_sek": it.get("line_total_sek"), "type": it.get("type"),
                        "brand": it.get("brand"), "category": it.get("category"), "price": it.get("price"),
                    })
    return out

# ---- Aggregation orchestrator ----
def collect_buckets(country_files: Dict[str, Path]) -> dict[str, list[dict]]:
    buckets = {k: [] for k in [
        "country_summary","country_channels","city_summary","city_monthly","customer_summary","orders","order_items"
    ]}
    for name, path in country_files.items():
        if not path.exists():
            continue
        rows = flatten_country(load_json(path), name)
        for k, v in rows.items():
            buckets[k].extend(v)
    return buckets

# ---- Materialize dataframes (fixed schemas) ----
def to_dataframes(buckets: dict[str, list[dict]]) -> dict[str, pd.DataFrame]:
    tx_country = _ensure(pd.DataFrame(buckets["country_summary"]),   CS_COUNTRY)
    tx_cc      = _ensure(pd.DataFrame(buckets["country_channels"]),  CS_COUNTRY_CHANNEL)
    tx_city    = _ensure(pd.DataFrame(buckets["city_summary"]),      CS_CITY)
    tx_city_m  = _ensure(pd.DataFrame(buckets["city_monthly"]),      CS_CITY_MONTHLY)
    tx_cust    = _ensure(pd.DataFrame(buckets["customer_summary"]),  CS_CUST)
    tx_orders  = _ensure(pd.DataFrame(buckets["orders"]),            CS_ORDERS)
    tx_items   = _ensure(pd.DataFrame(buckets["order_items"]),       CS_ITEMS)
    return {
        "country_summary": tx_country,
        "country_channels": tx_cc,
        "city_summary": tx_city,
        "city_monthly": tx_city_m,
        "customer_summary": tx_cust,
        "orders": tx_orders,
        "order_items": tx_items,
    }

# ---- Persist to parquet ----
def write_all_parquet(txs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    save_parquet(txs["country_summary"],   out_dir / "country_summary.parquet")
    save_parquet(txs["country_channels"],  out_dir / "country_customers_by_channel.parquet")
    save_parquet(txs["city_summary"],      out_dir / "city_summary.parquet")
    save_parquet(txs["city_monthly"],      out_dir / "city_monthly_revenue.parquet")
    save_parquet(txs["customer_summary"],  out_dir / "customer_summary.parquet")
    save_parquet(txs["orders"],            out_dir / "orders.parquet")
    save_parquet(txs["order_items"],       out_dir / "order_items.parquet")

# ---- Convenience wrapper used by combine.py ----
def build_tables_from_dir(input_dir: Path, output_dir: Path) -> None:
    country_files = {
        "Sweden":  input_dir / "Sweden.json",
        "Denmark": input_dir / "Denmark.json",
        "Finland": input_dir / "Finland.json",
        "Norway":  input_dir / "Norway.json",
    }
    buckets = collect_buckets(country_files)
    txs = to_dataframes(buckets)
    write_all_parquet(txs, output_dir)
