import pandas as pd

def dedup_size(s):
    if pd.isna(s): return pd.NA
    tokens, seen = [], set()
    for t in map(str.strip, str(s).split(',')):
        if t and t not in seen:
            seen.add(t); tokens.append(t)
    return ','.join(tokens) if tokens else pd.NA