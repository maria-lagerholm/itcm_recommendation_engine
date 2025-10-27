from pathlib import Path
import os, re, unicodedata
import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

MISSING = {"", "unknown", "nan", "none", None}
PRICE_BINS = [0, 100, 300, 600, 1000, 2000, float("inf")]
PRICE_LABELS = ["Budget", "Value", "Popular", "Premium", "Luxury", "Exclusive"]
MODEL_ID = "Alibaba-NLP/gte-multilingual-base"

def canon(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\u00A0", " ", s)
    s = re.sub(r"[\u2010-\u2015\u2212\-]+", "-", s)
    return re.sub(r"\s+", " ", s).strip()

def norm_categories(x):
    toks = [canon(c) for c in str(x).split(",")]
    out, seen = [], set()
    for c in toks:
        cl = c.strip().lower()
        if not cl or cl in MISSING:
            continue
        if cl not in seen:
            seen.add(cl)
            out.append(c)
    return out

def short_desc(desc, max_words=30):
    if not desc:
        return ""
    first = re.split(r"(?<=[.!?])\s+", desc)[0]
    return " ".join(first.split()[:max_words])

def format_colors(col) -> str:
    # deduplicate, expand delimiters, filter missing/empty
    vals = []
    if isinstance(col, (list, tuple, pd.Series, np.ndarray)):
        for v in list(col):
            s = str(v).strip()
            if not s or s.lower() in MISSING:
                continue
            parts = re.split(r"\s*[,/|;]\s*", s) if any(sep in s for sep in ",/|;") else [s]
            vals.extend(parts)
    else:
        s = str(col).strip()
        if s and s.lower() not in MISSING:
            quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", s)
            if quoted:
                vals = [a or b for a, b in quoted]
            else:
                vals = re.split(r"\s*[,/|;]\s*", s) if any(sep in s for sep in ",/|;") else [s]
    out, seen = [], set()
    for v in vals:
        t = v.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return ", ".join(out)

def build_text_embed_clean(r):
    # Compose one text string from separate fields for embedding
    name = canon(r.get("name", ""))
    desc = short_desc(canon(r.get("description", "")), 30)
    brand_raw = r.get("brand", "")
    brand = canon(brand_raw) if str(brand_raw).strip() and str(brand_raw).strip().lower() not in MISSING else ""
    cats = r.get("categories", []) or []
    cols = r.get("colors_str", "")
    parts, attrs = [], []
    if name:
        parts.append(f"{name}.")
    if desc:
        parts.append(desc)
    if brand:
        attrs.append(brand)
    if cats:
        attrs.append(", ".join(cats))
    if cols:
        attrs.append(cols)
    if attrs:
        parts.append(" ".join(attrs) + ".")
    return re.sub(r"\s+", " ", " ".join(parts)).strip()

def _iter_batches(n, bs):
    # yields batch start/end indices for batch processing
    for i in range(0, n, bs):
        yield i, min(i + bs, n)

def run(
    processed_dir: Path,
    batch_size: int = 64,
    k: int = 10,
    cos_min: float = 0.60,
    min_price: float = 1.0,
    num_threads: int = 1,
) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    hf_logging.set_verbosity_error()
    torch.set_num_threads(max(1, num_threads))

    groups = pd.read_parquet(processed_dir.joinpath("articles_for_recs.parquet"))
    groups["priceSEK"] = pd.to_numeric(groups["priceSEK"], errors="coerce")
    groups = groups[groups["priceSEK"] >= min_price].copy()
    groups["priceband"] = pd.cut(groups["priceSEK"], bins=PRICE_BINS, labels=PRICE_LABELS, include_lowest=True)
    if "color" not in groups.columns:
        groups["color"] = ""
    cat_src = groups["category"] if "category" in groups.columns else ""
    groups["categories"] = pd.Series(cat_src).apply(norm_categories)
    groups["colors_str"] = groups["color"].apply(format_colors)
    groups["text"] = groups.apply(build_text_embed_clean, axis=1)

    group_df = groups[["groupId", "text", "color", "colors_str", "categories", "brand", "priceband"]].reset_index(drop=True)
    texts = group_df["text"].fillna("").tolist()
    N = len(texts)

    enc = SentenceTransformer(MODEL_ID, device="cpu", trust_remote_code=True)
    try:
        max_len = getattr(getattr(enc, "tokenizer", None), "model_max_length", 4096)
    except Exception:
        max_len = 4096
    enc.max_seq_length = min(4096, max_len)

    probe = enc.encode(texts[:1] or [""], batch_size=1, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False).astype("float32")
    d = int(probe.shape[1])
    index = faiss.IndexFlatIP(d)

    mmap_path = processed_dir.joinpath("semantic_embeddings.mmap")
    E_mm = np.memmap(mmap_path, dtype="float32", mode="w+", shape=(N, d))
    for s, e in _iter_batches(N, batch_size):
        Eb = enc.encode(
            texts[s:e],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        E_mm[s:e, :] = Eb
        index.add(Eb)
    del probe
    E_mm.flush()

    # Search
    I_all = np.empty((N, k + 1), dtype=np.int64)
    S_all = np.empty((N, k + 1), dtype=np.float32)
    E_ro = np.memmap(mmap_path, dtype="float32", mode="r", shape=(N, d))
    for s, e in _iter_batches(N, batch_size):
        Eb = np.array(E_ro[s:e, :], copy=False)
        S, I = index.search(Eb, k + 1)
        I_all[s:e, :] = I
        S_all[s:e, :] = S
    del E_ro

    # Build wide rows with Top/Score; skip items with <1 rec
    ids = group_df["groupId"].astype(str).to_numpy()
    rows = []
    for i in range(N):
        js, ss = I_all[i], S_all[i]
        # filter: non-self + above threshold
        m = (js != i) & (ss >= cos_min)
        js, ss = js[m], ss[m]
        if js.size == 0:
            continue  # <1 rec → skip this Product ID entirely
        # truncate
        js, ss = js[:k], ss[:k]
        # build row: [Product ID, Top1, Score1, ..., TopK, ScoreK] with None padding
        row = [ids[i]]
        for r in range(k):
            if r < js.size:
                row.append(ids[js[r]])
                row.append(float(ss[r]))
            else:
                row.append(None)
                row.append(None)
        rows.append(row)

    cols = ["Product ID"] + [c for r in range(1, k + 1) for c in (f"Top {r}", f"Score {r}")]
    wide = pd.DataFrame(rows, columns=cols)
    wide.to_parquet(processed_dir.joinpath("semantic_similarity_recs.parquet"), index=False)

    try:
        os.remove(mmap_path)
    except Exception:
        pass
