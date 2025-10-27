import re
import pandas as pd

def clean_audience(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize/derive 'audience' and 'audienceId' from existing 'audience' and 'category' columns.
    Returns a new DataFrame with 'audience' cleaned, missing values classified from 'category',
    and a new 'audienceId' column inserted right after 'audience'.
    """
    AUD2ID = {'dam':'6','herr':'15','baby & barn':'12','barn & ungdom':'42','generic':'99','hemmet':'222'}

    def norm_audience(a):
        if pd.isna(a): 
            return pd.NA
        toks = {t.strip().lower() for t in str(a).split(',') if t.strip()}
        if any('dam' in t for t in toks):
            return 'dam'
        keep = [t for t in toks if t in AUD2ID]
        return ','.join(keep) if keep else pd.NA

    def to_ids(a):
        if pd.isna(a): 
            return pd.NA
        ids = sorted({AUD2ID[t] for t in a.split(',') if t in AUD2ID}, key=int)
        return ','.join(ids) if ids else pd.NA

    DAM_raw = [
        'dam','bh','trosor','underkläder','body','bodykorselett','korsett','korsetter',
        'klänning','klänningar','tunika','tunikor','topp','toppar','kjol','kjolar',
        'byxa','byxor','blus','blusar','nattlinne','bikinibh','bikini','t-shirt-bh',
        'minimizer','kofta','koftor','väst','västar','skor','väskor','sjalar',
        'Bh,Underkläder,Bygel-bh',
        'Bygel-bh,Bh,Underkläder',
        'Bh utan bygel,Bh,Underkläder',
        'Bh utan bygel,Framknäppt bh,Bh,Underkläder',
        'Framknäppt bh,Bh,Underkläder',
        'Bh,Underkläder,Sport-bh',
        'Sport-bh,Bh,Underkläder',
        'Minimizer,Bh,Underkläder',
        'Underkläder,Trosor',
        'Underkläder,Trosor & gördlar',
        'Underkläder,Trosor & gördlar,Trosor',
        'Trosor,Underkläder,Gördlar',
        'Underkjolar,Underkläder',
        'Underkläder,Underklänningar',
        'Underkläder,Mamelucker',
        'Strumpbyxor,Underkläder',
        'Baddräkter,Badkläder,Dam',
        'Badkläder,Dam',
        'Dam,Bikini,Badkläder',
        'Dam,Badkläder,Tankini',
        'Nattlinnen,Sovkläder,Dam',
        'Sovkläder,Dam'
    ]
    HEM_raw = [
        'frottéhanddukar','badlakan','bad','badrumsmattor','kökshanddukar','vaxdukar','dukar',
        'pläd','plädar','kanallängder','kanalkappa','gardiner','påslakanset','bädd',
        'lakan','örngott','hemtextil','kuddfodral','överkast','gardinstänger','kökshjälpmedel',
        'dekorationer','metervara','prydnadssaker','belysning','servetter',
        'Frottéhanddukar & badlakan',
        'Frottéhanddukar & badlakan,Bad',
        'Badrumsmattor,Bad',
        'Duschdraperier,Bad',
        'Kökshanddukar',
        'Vaxdukar',
        'Dukar',
        'Vaxdukar,Dukar',
        'Dukar,Vaxdukar',
        'Påslakanset',
        'Lakan & örngott,Bädd',
        'Bädd',
        'Bäddtillbehör,Bädd',
        'Innerkuddar,Bädd (linea),Kuddar',
        'Kuddar',
        'Plädar',
        'Gardinbåge',
        'Kanallängder',
        'Kanalkappa',
        'Panellängder',
        'Multibandslängder',
        'Multibandslängder,Mörkläggningsgardiner',
        'Öljettkappa',
        'Tabletter/underlägg/brickor',
        'Batteridrivna ljus',
        'Synhjälpmedel,Belysning',
        'Ljusstakar & lyktor,Juldekoration',
        'Servetter'
    ]
    GEN_raw = [
        'inkontinens','stödartiklar','vardagshjälpmedel','rollator','rollatorer','stödstrumpor',
        'skotillbehör','fotvård','hobbyhörnan','pussel','sytillbehör','symaskiner','lust',
        'massage','synhjälpmedel','medicin','böcker','halkskydd','träning & motion',
        'Vardagshjälpmedel',
        'Vardagshjälpmedel,Dynor & säten',
        'Stödartiklar',
        'Synhjälpmedel',
        'Gånghjälpmedel',
        'Rollatorer',
        'Inkontinens',
        'Intimvård',
        'Fotvård',
        'Skotillbehör',
        'Stödstrumpor,Underkläder',
        'Hobbyhörnan,Pussel',
        'Hobbyhörnan,Pysselset',
        'Sytillbehör,Symaskiner och tillbehör',
        'Symaskiner och tillbehör,Sytillbehör',
        'Tvätt & skötsel,Vardagshjälpmedel',
        'Tvätt & skötsel,Vardagshjälpmedel,Hushåll övrigt',
        'Träning & motion',
        'Träning & motion,Hälsa',
        'Massage,Kroppsvård,Hälsa',
        'Medicin,Hälsa',
        'Synhjälpmedel,Belysning,Vardagshjälpmedel',
        'Virknålar,Vardagshjälpmedel',
        'Halkskydd',
        'Halkskydd,Gånghjälpmedel'
    ]
    HERR_raw = [
        'herr','skjorta','skjortor','kostym','kavaj','boxer','kalsonger',
        'Skjortor,Herr',
        'Pyjamas,Herr,Sovkläder',
        'Herr,Överdelar,T-shirts',
        'Herr,Sovkläder,Nattskjortor',
        'Accessoarer,Herr,Kepsar & mössor'
    ]

    DAM = [s.lower() for s in DAM_raw]
    HEM = [s.lower() for s in HEM_raw]
    GEN = [s.lower() for s in GEN_raw]
    HERR = [s.lower() for s in HERR_raw]

    REA_TOKEN = re.compile(r'(^|,)\s*rea\s*(?=,|$)')

    def strip_rea(s):
        s = s.lower()
        s = REA_TOKEN.sub(lambda m: ',' if m.group(1) else '', s)
        return re.sub(r',+', ',', s).strip(', ').strip()

    def classify(cat):
        if pd.isna(cat): 
            return pd.NA
        s = strip_rea(str(cat))
        if not s: 
            return pd.NA
        if any(h in s for h in DAM): return 'dam'
        if any(h in s for h in HERR): return 'herr'
        if any(h in s for h in HEM): return 'hemmet'
        if any(h in s for h in GEN): return 'generic'
        return pd.NA

    def move_after(df_in, cols, after):
        cols_all = list(df_in.columns)
        for c in cols:
            if c in cols_all:
                cols_all.remove(c)
        i = cols_all.index(after) + 1 if after in cols_all else len(cols_all)
        return df_in[cols_all[:i] + cols + cols_all[i:]]

    out = df.copy()

    out['audience'] = out['audience'].apply(norm_audience).astype('string')

    na_mask = out['audience'].isna()
    fill = out.loc[na_mask, 'category'].apply(classify)
    idx = fill.dropna().index
    out.loc[idx, 'audience'] = fill.loc[idx]

    out['audienceId'] = out['audience'].apply(to_ids).astype('string')

    out = move_after(out, ['audienceId'], 'audience')

    return out
