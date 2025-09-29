from __future__ import annotations
import re
import pandas as pd

# -----------------------
# Config / constants
# -----------------------
PRICELESS = ("priceSEK", "priceEUR", "priceNOK", "priceDKK")

RARE_COLOR_MERGES: list[tuple[str, str]] = [
    ("blush", "rosa"), ("cerise", "rosa"), ("grå-rosa", "rosa"), ("grålila", "lila"),
    ("havsblå", "blå"), ("jeansblå", "blå"), ("klarblå", "blå"), ("lavendel", "lila"),
    ("ljus beige", "beige"), ("ljus blå", "blå"), ("ljusgrå mix", "grå"), ("ljusturkos", "turkos"),
    ("marinblå", "marin"), ("mellanblå", "blå"), ("mellanbrun", "brun"), ("mellangrå", "grå"),
    ("mellanrosa", "rosa"), ("mintgrön", "grön"), ("mörkbrun", "brun"), ("mörkröd", "röd"),
    ("natur", "beige"), ("oblekt", "vit"), ("oliv", "grön"), ("orange mix", "orange"),
    ("puderrosa", "rosa"), ("rost", "röd"), ("svart-silver", "svart"), ("transparent", "vit"),
    ("violett", "lila"), ("off white", "offwhite"), ("silverfärgad", "silver"), ("guldgul", "gul"),
    ("ljusrosa", "rosa"), ("gråsvart", "svart"), ("gråbeige", "beige"), ("gråbrun", "brun"),
    ("vitblå", "blå"), ("grårosa", "rosa"), ("plommonlila", "lila"), ("vinröd", "röd"),
    ("multicolor", "multi"), ("flerfärgad", "multi"), ("jeans", "blå"), ("himmelblå", "ljusblå"),
    ("pärlvit", "vit"), ("naturvit", "vit"), ("sandfärgad", "sand"), ("kaffe", "brun"),
    ("kamel", "brun"), ("taupe", "brun"), ("offvit", "vit"), ("beigegrå", "beige"),
]

# -----------------------
# Helpers
# -----------------------
def _dedup_commalist(val) -> pd.StringDtype:
    if pd.isna(val):
        return pd.NA
    seen = set()
    toks = [x.strip() for x in str(val).split(",") if x.strip()]
    out = [t for t in toks if not (t in seen or seen.add(t))]
    return ",".join(out) if out else pd.NA

def _clean_color_name(color) -> pd.StringDtype:
    if pd.isna(color):
        return color
    return str(color).replace("/", "-").lower()

def _merge_comma_colors_to_mode(color: str | pd.NA, counts: pd.Series) -> pd.StringDtype:
    """If 'red, blue' → pick the variant with highest frequency in data."""
    if pd.isna(color) or "," not in str(color):
        return color
    parts = [c.strip() for c in str(color).split(",") if c.strip()]
    if not parts:
        return pd.NA
    # choose by global frequency (fallback to first)
    best = max(parts, key=lambda c: int(counts.get(c, 0))) if len(parts) > 1 else parts[0]
    return best

def _build_all_colors_set(df: pd.DataFrame) -> set[str]:
    base = {t for _, t in RARE_COLOR_MERGES}
    rare = {s for s, _ in RARE_COLOR_MERGES}
    existing = set(df["color"].dropna().astype(str).str.lower().unique()) if "color" in df else set()
    return {c.lower() for c in (base | rare | existing) if isinstance(c, str)}

def _extract_color_from_description(desc: str | None, all_colors: set[str]) -> pd.StringDtype:
    if pd.isna(desc) or not isinstance(desc, str):
        return pd.NA
    d = desc.lower()
    found = []
    for c in sorted(all_colors, key=lambda x: -len(x)):  # match longer tokens first
        if re.search(r"\b" + re.escape(c) + r"\b", d) or (c in d):
            found.append(c)
    if not found:
        return pd.NA
    pos = [(d.find(c), c) for c in found if d.find(c) != -1]
    if not pos:
        return pd.NA
    pos.sort()
    chosen = pos[0][1]
    # map rare→base if listed
    for rare, base in RARE_COLOR_MERGES:
        if chosen == rare:
            return base
    return chosen

# -----------------------
# Public API
# -----------------------
def report_color_stats(df: pd.DataFrame) -> dict:
    """Return stats useful for logging/printing."""
    out = {}
    if "color" not in df.columns or "colorId" not in df.columns:
        return out

    non_na = df["colorId"].notna()
    by_color = df.loc[non_na].groupby("color", dropna=False)["colorId"].agg(["unique", "count"])
    out["colors_with_multiple_colorIds"] = {
        color: [i for i in ids if pd.notna(i)]
        for color, ids in by_color["unique"].items()
        if len([i for i in ids if pd.notna(i)]) > 1
    }
    by_id = df.loc[non_na].groupby("colorId", dropna=False)["color"].agg(["unique", "count"])
    out["colorIds_used_by_multiple_colors"] = {
        cid: [c for c in cols if pd.notna(c)]
        for cid, cols in by_id["unique"].items()
        if len([c for c in cols if pd.notna(c)]) > 1
    }
    out["num_rows_without_colorId"] = int(df["colorId"].isna().sum())
    out["num_rows_without_color"] = int(df["color"].isna().sum())
    out["nunique_colorIds"] = int(df["colorId"].nunique(dropna=True))
    out["nunique_colors"] = int(df["color"].nunique(dropna=True))
    return out

def normalize_colors(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    - Dedup & lowercase color/colorId
    - Merge comma-separated colors to the most frequent variant
    - Merge rare → base colors (RARE_COLOR_MERGES)
    - For a given color, collapse to a single dominant colorId
    - Fill missing 'color' from 'description' when possible
    Returns (clean_df, stats_dict).
    """
    out = df.copy()
    for c in ("color", "colorId", "description"):
        if c in out.columns:
            out[c] = out[c].astype("string")

    # 1) dedup & clean color/colorId
    if "color" in out.columns:
        counts = out["color"].dropna().str.lower().value_counts()
        out["color"] = out["color"].apply(_dedup_commalist).apply(_clean_color_name)
        out["color"] = out["color"].apply(lambda c: _merge_comma_colors_to_mode(c, counts))
    if "colorId" in out.columns:
        out["colorId"] = out["colorId"].apply(_dedup_commalist).astype("string")

    # 2) merge rare → base colors
    merge_counts = {}
    if "color" in out.columns:
        for rare, target in RARE_COLOR_MERGES:
            mask = out["color"].eq(rare)
            n = int(mask.sum())
            if n:
                out.loc[mask, "color"] = target
                merge_counts[f"{rare}→{target}"] = n

    # 3) per-color: collapse colorIds to the most common id
    reassigned = {}
    if "color" in out.columns and "colorId" in out.columns:
        vc = out.groupby("color")["colorId"].apply(lambda s: s.value_counts(dropna=True))
        # for each color, pick the colorId with max frequency
        for color in out["color"].dropna().unique():
            ids = out.loc[out["color"] == color, "colorId"]
            if ids.dropna().empty:
                continue
            dominant = ids.value_counts(dropna=True).idxmax()
            # only if multiple different ids exist
            if ids.dropna().nunique() > 1:
                reassigned[color] = {
                    "from": sorted(set(ids.dropna().unique())),
                    "to": dominant,
                }
                out.loc[out["color"] == color, "colorId"] = dominant

    # 4) fill missing color from description (optional; only if columns exist)
    filled_from_desc = 0
    if "color" in out.columns and "description" in out.columns:
        all_colors = _build_all_colors_set(out)
        miss = out["color"].isna() | out["color"].eq("unknown")
        fill_vals = out.loc[miss, "description"].apply(lambda d: _extract_color_from_description(d, all_colors))
        filled_from_desc = int(fill_vals.notna().sum())
        out.loc[miss, "color"] = fill_vals

    stats = {
        "rare_merges": merge_counts,
        "reassigned_colorIds": reassigned,
        "filled_color_from_description": filled_from_desc,
        **report_color_stats(out),
    }
    return out.reset_index(drop=True), stats
