from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import io
import traceback

app = Flask(__name__)

# ── In-memory cache (single-user local app) ────────────────────────────────────
_cached_df = None

# ── Constants ──────────────────────────────────────────────────────────────────

EPOCH = pd.Timestamp('2020-01-06')  # A known Monday far before any data

CAT_MAP = {
    'Clinical Knowledge and Concept Query': 'Clinical Knowledge',
    'Treatment Decision Support': 'Treatment Decision',
    'Psychotherapy and Mental Health Clinical Query': 'Mental Health',
    'Unintelligible, Duplicate, or Out-of-Scope Message': 'Out-of-Scope/Noise',
    'Drug Safety and Interactions Query': 'Drug Safety',
    'Complex or Ongoing Case Discussion': 'Complex Case',
    'Diagnostic Workup and Differential Diagnosis': 'Diagnostics',
    'Drug Dosing and Administration': 'Drug Dosing',
    'Evidence, Guidelines, and Literature Query': 'Evidence/Guidelines',
    'Tool Capability and Feature Question': 'Tool/Feature',
    'Rehabilitation, Physiotherapy, and Lifestyle Query': 'Rehab/Lifestyle',
    'Clinical Note or Document Generation': 'Clinical Docs',
    'Lab or Imaging Result Interpretation': 'Lab/Imaging',
    'Patient-Facing Content Generation': 'Patient Content',
    'Drug Selection and Treatment Decision Support': 'Drug Dosing',
    'Drug Administration and Special Populations': 'Drug Dosing',
    'Out-of-Scope Message': 'Out-of-Scope/Noise',
    'Out-of-Scope': 'Out-of-Scope/Noise',
}

SPECIALTY_MAP = {
    'Psiquiatra': 'psychiatrist', 'Psicólogo': 'psychologist',
    'Ginecologista': 'gynecologist', 'fizjoterapeuta': 'physiotherapist',
    'psiquiatra': 'psychiatrist', 'psicólogo': 'psychologist',
    'ginecologista': 'gynecologist', 'Clínico Geral': 'general practitioner',
    'clinico geral': 'general practitioner', 'médico general': 'general practitioner',
    'Médico General': 'general practitioner', 'Cardiólogo': 'cardiologist',
    'Dermatólogo': 'dermatologist', 'Nutricionista': 'nutritionist',
    'nutricionista': 'nutritionist', 'Pediatra': 'pediatrician',
    'pediatra': 'pediatrician', 'Ortopedista': 'orthopedist',
    'Internista': 'internist',
}

# ── Preprocessing ──────────────────────────────────────────────────────────────

def _auto_detect_cols(df):
    """Return best-guess column names for each expected field."""
    col_map = {c.lower().strip(): c for c in df.columns}

    def find(target):
        t = target.lower().strip()
        if t in col_map:
            return col_map[t]
        for k, v in col_map.items():
            if t in k or k in t:
                return v
        return None

    return {
        'doctor_id': find('doctor id') or find('doctor_id') or find('user id') or find('user_id'),
        'category':  find('category'),
        'timestamp': find('first message time') or find('first_message_time') or find('timestamp') or find('date'),
        'country':   find('country'),
        'specialty': find('specialty'),
        'source':    find('source'),
    }


def preprocess(df, col_mappings=None):
    """
    col_mappings: optional dict {doctor_id, category, timestamp, country, specialty, source}
                  with the actual CSV column names.  If None, auto-detection is used.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Resolve mappings
    if col_mappings:
        m = col_mappings
    else:
        m = _auto_detect_cols(df)

    doctor_col   = m.get('doctor_id')
    category_col = m.get('category')
    date_col     = m.get('timestamp')
    spec_col     = m.get('specialty')
    country_col  = m.get('country')
    source_col   = m.get('source')

    if not doctor_col:
        raise ValueError("Could not find a Doctor ID column. Please map it manually.")
    if not category_col:
        raise ValueError("Could not find a Category column. Please map it manually.")
    if not date_col:
        raise ValueError("Could not find a Timestamp column. Please map it manually.")

    df = df.rename(columns={
        doctor_col:   'doctor id',
        category_col: 'category',
        date_col:     'first message time',
    })

    for fmt in ['%B %d, %Y, %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M']:
        try:
            df['first_dt'] = pd.to_datetime(df['first message time'], format=fmt)
            break
        except Exception:
            continue
    else:
        df['first_dt'] = pd.to_datetime(df['first message time'], infer_datetime_format=True, errors='coerce')

    df = df.dropna(subset=['doctor id', 'first_dt'])
    df['doctor id'] = df['doctor id'].astype(str).str.strip()

    df['cat_short'] = df['category'].map(CAT_MAP).fillna(df['category'])

    if spec_col and spec_col in df.columns:
        df['specialty_norm'] = df[spec_col].map(
            lambda x: SPECIALTY_MAP.get(str(x).strip(), str(x).strip()) if pd.notna(x) else np.nan
        )
    else:
        df['specialty_norm'] = np.nan

    if country_col and country_col in df.columns:
        df['country'] = df[country_col]
    else:
        df['country'] = np.nan

    if source_col and source_col in df.columns:
        df['source'] = df[source_col]
    else:
        df['source'] = np.nan

    df['conv_week_num'] = ((df['first_dt'] - EPOCH).dt.days // 7).astype(int)

    doc_first_week = df.groupby('doctor id')['conv_week_num'].min().rename('cohort_week_num')
    df = df.merge(doc_first_week, on='doctor id', how='left')
    df['week_offset'] = df['conv_week_num'] - df['cohort_week_num']
    df['cohort_monday'] = (EPOCH + pd.to_timedelta(df['cohort_week_num'] * 7, unit='D')).dt.date

    doc_count = df.groupby('doctor id').size().reset_index(name='total_convs')
    df = df.merge(doc_count, on='doctor id', how='left')

    doc_meta = (df.sort_values('first_dt')
                  .drop_duplicates('doctor id', keep='first')
                  [['doctor id', 'country', 'specialty_norm', 'source']]
                  .rename(columns={'country': 'doctor_country',
                                   'specialty_norm': 'doctor_specialty',
                                   'source': 'doctor_source'}))
    df = df.merge(doc_meta, on='doctor id', how='left')

    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_float(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(round(v, 2))

def safe_int(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0
    return int(v)


# ── Analysis 1: meta ──────────────────────────────────────────────────────────

def get_meta(df):
    return {
        'total_conversations': int(len(df)),
        'total_doctors':       int(df['doctor id'].nunique()),
        'date_min':            str(df['first_dt'].min().date()),
        'date_max':            str(df['first_dt'].max().date()),
        'single_q_doctors':    int((df.groupby('doctor id').size() == 1).sum()),
        'repeat_doctors':      int((df.groupby('doctor id').size() > 1).sum()),
        'countries':           sorted([c for c in df['country'].dropna().unique().tolist() if c]),
        'specialties':         sorted([s for s in df['specialty_norm'].dropna().unique().tolist() if s]),
    }


# ── Analysis 2: category distribution ─────────────────────────────────────────

def get_cat_dist(df):
    counts = df['cat_short'].value_counts()
    pcts   = (counts / len(df) * 100).round(1)
    if 'confidence' in df.columns:
        conf = df.groupby(['cat_short', 'confidence']).size().unstack(fill_value=0)
        conf_data = {}
        for cat in counts.index:
            row = conf.loc[cat] if cat in conf.index else pd.Series(dtype=int)
            conf_data[cat] = {
                'high':   int(row.get('high', 0)),
                'medium': int(row.get('medium', 0)),
                'low':    int(row.get('low', 0)),
            }
    else:
        conf_data = {cat: {'high': 0, 'medium': 0, 'low': 0} for cat in counts.index}

    return {
        'categories': counts.index.tolist(),
        'counts':     [int(v)   for v in counts.values],
        'pcts':       [float(v) for v in pcts.values],
        'confidence': conf_data,
    }


# ── Summary distributions (country + specialty of doctors) ────────────────────

def get_summary_distributions(df):
    doc_first = df.sort_values('first_dt').drop_duplicates('doctor id', keep='first')
    total_docs = len(doc_first)

    def clean_series(s):
        s = s.dropna()
        s = s[s.astype(str).str.lower() != 'nan']
        return s.value_counts()

    country_counts   = clean_series(doc_first['doctor_country'])
    specialty_counts = clean_series(doc_first['doctor_specialty'])

    return {
        'country': {
            'labels': [str(x) for x in country_counts.index.tolist()],
            'counts': [int(v)   for v in country_counts.values],
            'pcts':   [round(v / total_docs * 100, 1) for v in country_counts.values],
        },
        'specialty': {
            'labels': [str(x) for x in specialty_counts.head(15).index.tolist()],
            'counts': [int(v)   for v in specialty_counts.head(15).values],
            'pcts':   [round(v / total_docs * 100, 1) for v in specialty_counts.head(15).values],
        },
    }


# ── Analysis 3: first question table ──────────────────────────────────────────

def get_first_q_table(df):
    doc_first  = df.sort_values('first_dt').drop_duplicates('doctor id', keep='first').copy()
    single_docs = doc_first[doc_first['total_convs'] == 1]
    repeat_docs = doc_first[doc_first['total_convs'] > 1]
    repeat_set  = set(repeat_docs['doctor id'])

    all_conv_counts = df['cat_short'].value_counts()
    first_counts    = doc_first['cat_short'].value_counts()
    single_counts   = single_docs['cat_short'].value_counts()
    repeat_counts   = repeat_docs['cat_short'].value_counts()

    doc_first['is_repeat'] = doc_first['doctor id'].isin(repeat_set)
    conv_rate = doc_first.groupby('cat_short')['is_repeat'].mean().mul(100).round(1)

    rows = []
    all_cats_union = list(dict.fromkeys(
        all_conv_counts.index.tolist() + first_counts.index.tolist()
    ))
    for cat in all_cats_union:
        ac = int(all_conv_counts.get(cat, 0))
        fc = int(first_counts.get(cat, 0))
        sc = int(single_counts.get(cat, 0))
        rc = int(repeat_counts.get(cat, 0))
        rows.append({
            'category':       cat,
            'all_conv_n':     ac,
            'all_conv_pct':   round(ac / len(df) * 100, 1)           if len(df) > 0          else 0,
            'first_q_n':      fc,
            'first_q_pct':    round(fc / len(doc_first) * 100, 1)    if len(doc_first) > 0   else 0,
            'single_n':       sc,
            'single_pct':     round(sc / len(single_docs) * 100, 1)  if len(single_docs) > 0 else 0,
            'repeat_n':       rc,
            'repeat_pct':     round(rc / len(repeat_docs) * 100, 1)  if len(repeat_docs) > 0 else 0,
            'diff':           round(
                (rc / len(repeat_docs) * 100 if len(repeat_docs) > 0 else 0) -
                (sc / len(single_docs) * 100 if len(single_docs) > 0 else 0), 1),
            'conversion_pct': float(conv_rate.get(cat, 0)),
        })

    rows.sort(key=lambda r: -r['all_conv_n'])

    return {
        'rows': rows,
        'totals': {
            'all_conv':    int(len(df)),
            'first_q_all': int(len(doc_first)),
            'single':      int(len(single_docs)),
            'repeat':      int(len(repeat_docs)),
        }
    }


# ── Last question raw data (client-side filtering) ────────────────────────────

def get_last_q_raw(df):
    """Return per-doctor last question {category, last_date} for JS-side date filtering."""
    doc_last = (df.sort_values('first_dt')
                  .drop_duplicates('doctor id', keep='last')
                  [['cat_short', 'first_dt']])
    doc_last = doc_last.copy()
    doc_last['last_date'] = doc_last['first_dt'].dt.strftime('%Y-%m-%d')
    return (doc_last[['cat_short', 'last_date']]
            .rename(columns={'cat_short': 'category'})
            .to_dict('records'))


# ── Analysis 4: retention cohorts ─────────────────────────────────────────────

def build_retention_list(data, max_weeks=12, min_size=1):
    """Cohort retention with pct_single (1-question only) and pct_w0_return (same-week return)."""
    if data.empty:
        return []
    max_week_num = int(data['conv_week_num'].max())
    cohorts      = sorted(data['cohort_monday'].unique())
    results      = []
    for cohort in cohorts:
        cohort_data     = data[data['cohort_monday'] == cohort]
        cohort_week_num = int(cohort_data['cohort_week_num'].iloc[0])
        cohort_docs     = set(cohort_data['doctor id'])
        size            = len(cohort_docs)
        if size < min_size:
            continue

        # % who asked only 1 question total
        doc_convs  = cohort_data.drop_duplicates('doctor id')['total_convs']
        pct_single = round((doc_convs == 1).sum() / size * 100, 1)

        # % who came back within week 0 (had >1 conversation with week_offset == 0)
        w0_per_doc    = cohort_data[cohort_data['week_offset'] == 0].groupby('doctor id').size()
        pct_w0_return = round((w0_per_doc > 1).sum() / size * 100, 1)

        weeks = []
        for w in range(1, max_weeks + 1):
            if cohort_week_num + w > max_week_num:
                weeks.append(None)
            else:
                active = set(data[
                    (data['cohort_monday'] == cohort) & (data['week_offset'] == w)
                ]['doctor id'])
                weeks.append(round(len(active) / size * 100, 1))
        results.append({
            'cohort':         str(cohort),
            'size':           size,
            'pct_single':     float(pct_single),
            'pct_w0_return':  float(pct_w0_return),
            'weeks':          weeks,
        })
    return results


def get_retention_by_segment(df, segment_col, min_cohort_size=15):
    segments = sorted([s for s in df[segment_col].dropna().unique() if s and str(s) != 'nan'])
    results  = {}
    for seg in segments:
        doc_ids = set(
            df.sort_values('first_dt')
              .drop_duplicates('doctor id', keep='first')
              .loc[lambda x: x[segment_col] == seg, 'doctor id']
        )
        seg_data = df[df['doctor id'].isin(doc_ids)]
        ret      = build_retention_list(seg_data, max_weeks=12, min_size=min_cohort_size)
        if ret:
            results[str(seg)] = {
                'total_doctors': int(seg_data['doctor id'].nunique()),
                'cohorts':       ret,
            }
    return results


def get_retention_aggregate_by_segment(df, segment_col, max_weeks=12, min_doctors=10):
    """One aggregated retention row per segment value (across all cohorts)."""
    doc_first    = df.sort_values('first_dt').drop_duplicates('doctor id', keep='first')
    max_week_num = int(df['conv_week_num'].max())

    valid_vals = doc_first[segment_col].dropna()
    valid_vals = valid_vals[valid_vals.astype(str).str.lower() != 'nan']
    segments   = sorted(valid_vals.unique().tolist())
    results    = []

    for seg in segments:
        doc_seg = doc_first[doc_first[segment_col] == seg]
        doc_ids = set(doc_seg['doctor id'])
        size    = len(doc_ids)
        if size < min_doctors:
            continue

        seg_data = df[df['doctor id'].isin(doc_ids)]

        pct_single    = round((doc_seg['total_convs'] == 1).mean() * 100, 1)
        w0_counts     = seg_data[seg_data['week_offset'] == 0].groupby('doctor id').size()
        pct_w0_return = round((w0_counts > 1).sum() / size * 100, 1)

        weeks = []
        for w in range(1, max_weeks + 1):
            # Only count doctors for whom week w is already observable
            obs_mask = seg_data['cohort_week_num'] + w <= max_week_num
            obs_docs = set(seg_data[obs_mask]['doctor id'])
            if not obs_docs:
                weeks.append(None)
            else:
                active = set(seg_data[seg_data['week_offset'] == w]['doctor id']) & obs_docs
                weeks.append(round(len(active) / len(obs_docs) * 100, 1))

        results.append({
            'segment':        str(seg),
            'size':           int(size),
            'pct_single':     float(pct_single),
            'pct_w0_return':  float(pct_w0_return),
            'weeks':          weeks,
        })

    return results


def get_top_specialty_cohorts(df, top_n=3, min_cohort_size=5):
    """Full cohort heatmap for the top-N specialties by doctor count."""
    doc_first    = df.sort_values('first_dt').drop_duplicates('doctor id', keep='first')
    spec_counts  = doc_first['doctor_specialty'].value_counts()
    spec_counts  = spec_counts[
        spec_counts.index.notna() &
        (spec_counts.index.astype(str).str.lower() != 'nan')
    ]
    top_specs = spec_counts.head(top_n).index.tolist()
    results   = {}
    for spec in top_specs:
        doc_ids   = set(doc_first[doc_first['doctor_specialty'] == spec]['doctor id'])
        spec_data = df[df['doctor id'].isin(doc_ids)]
        ret       = build_retention_list(spec_data, max_weeks=12, min_size=min_cohort_size)
        results[str(spec)] = {
            'total_doctors': int(len(doc_ids)),
            'cohorts':       ret,
        }
    return results


# ── Qualitative analyses ───────────────────────────────────────────────────────

def get_qual_stickiness(df):
    repeat_df = df[df['total_convs'] > 1].copy()
    if repeat_df.empty:
        return []
    repeat_df['conv_rank']  = repeat_df.groupby('doctor id')['first_dt'].rank(method='first').astype(int)
    first_cats_map = repeat_df[repeat_df['conv_rank'] == 1].set_index('doctor id')['cat_short'].to_dict()
    later_df       = repeat_df[repeat_df['conv_rank'] > 1].copy()
    later_df['first_cat'] = later_df['doctor id'].map(first_cats_map)
    later_df = later_df.dropna(subset=['first_cat'])

    rows = []
    for first_cat, grp in later_df.groupby('first_cat'):
        total = len(grp)
        same  = (grp['cat_short'] == first_cat).sum()
        rows.append({
            'first_cat':      first_cat,
            'followup_convs': int(total),
            'stickiness_pct': round(same / total * 100, 1) if total > 0 else 0,
        })
    rows.sort(key=lambda r: -r['stickiness_pct'])
    return rows


def get_qual_depth(df):
    doc_first = df.sort_values('first_dt').drop_duplicates('doctor id', keep='first').copy()
    rows = []
    for cat, grp in doc_first.groupby('cat_short'):
        convs = grp['total_convs']
        rows.append({
            'category':    cat,
            'n_doctors':   int(len(grp)),
            'mean_convs':  round(float(convs.mean()), 2),
            'median_convs':float(convs.median()),
            'pct_multi':   round((convs > 1).mean() * 100, 1),
            'pct_5plus':   round((convs >= 5).mean() * 100, 1),
            'pct_10plus':  round((convs >= 10).mean() * 100, 1),
        })
    rows.sort(key=lambda r: -r['mean_convs'])
    return rows


def get_qual_power_users(df, threshold=5):
    power_docs = set(df.groupby('doctor id').size().loc[lambda x: x >= threshold].index)
    all_dist   = (df['cat_short'].value_counts() / len(df) * 100).round(1)
    power_data = df[df['doctor id'].isin(power_docs)]
    power_dist = (power_data['cat_short'].value_counts() / len(power_data) * 100).round(1) \
                 if len(power_data) > 0 else pd.Series(dtype=float)

    all_cats = all_dist.index.union(power_dist.index)
    rows = []
    for cat in all_cats:
        ap = float(all_dist.get(cat, 0))
        pp = float(power_dist.get(cat, 0))
        rows.append({
            'category':   cat,
            'all_pct':    ap,
            'power_pct':  pp,
            'overindex':  round(pp - ap, 1),
        })
    rows.sort(key=lambda r: -r['overindex'])
    return {
        'power_user_count': int(len(power_docs)),
        'threshold':        threshold,
        'rows':             rows,
    }


def get_qual_sequences(df):
    repeat_df = df[df['total_convs'] > 1].copy()
    if repeat_df.empty:
        return {'categories': [], 'matrix': []}
    repeat_df['conv_rank'] = repeat_df.groupby('doctor id')['first_dt'].rank(method='first').astype(int)
    first_cats   = repeat_df[repeat_df['conv_rank'] == 1][['doctor id', 'cat_short']].rename(columns={'cat_short': 'first_cat'})
    second_cats  = repeat_df[repeat_df['conv_rank'] == 2][['doctor id', 'cat_short']].rename(columns={'cat_short': 'second_cat'})
    transitions  = first_cats.merge(second_cats, on='doctor id', how='inner')
    if transitions.empty:
        return {'categories': [], 'matrix': []}
    matrix = pd.crosstab(transitions['first_cat'], transitions['second_cat'], normalize='index') * 100
    cats   = sorted(list(set(matrix.index.tolist() + matrix.columns.tolist())))
    matrix = matrix.reindex(index=cats, columns=cats, fill_value=0)
    return {
        'categories': cats,
        'matrix':     [[round(float(matrix.loc[r, c]), 1) for c in cats] for r in cats],
    }


def get_qual_weekly_shift(df):
    weekly_cats  = df.groupby(['week_offset', 'cat_short']).size().unstack(fill_value=0)
    weekly_total = weekly_cats.sum(axis=1)
    weekly_pct   = weekly_cats.div(weekly_total, axis=0).round(4) * 100
    weekly_pct   = weekly_pct[weekly_pct.index <= 8]
    weekly_total = weekly_total[weekly_total.index <= 8]

    if 0 in weekly_pct.index and len(weekly_pct[weekly_pct.index >= 4]) > 0:
        w0     = weekly_pct.loc[0]
        w4plus = weekly_pct[weekly_pct.index >= 4].mean()
        shift  = (w4plus - w0).round(1).sort_values(ascending=False)
        shift_list = [{'category': str(cat), 'shift': float(val)} for cat, val in shift.items()]
    else:
        shift_list = []

    return {
        'weeks':      [int(w)  for w in weekly_pct.index.tolist()],
        'totals':     [int(v)  for v in weekly_total.tolist()],
        'categories': weekly_pct.columns.tolist(),
        'data':       [[round(float(v), 1) for v in row] for row in weekly_pct.values],
        'shift':      shift_list,
    }


def get_engagement_score(df):
    doc_first = df.sort_values('first_dt').drop_duplicates('doctor id', keep='first').copy()
    repeat_df = df[df['total_convs'] > 1].copy()
    if repeat_df.empty:
        return []
    repeat_df['conv_rank']  = repeat_df.groupby('doctor id')['first_dt'].rank(method='first').astype(int)
    first_cats_map = repeat_df[repeat_df['conv_rank'] == 1].set_index('doctor id')['cat_short'].to_dict()
    later_df       = repeat_df[repeat_df['conv_rank'] > 1].copy()
    later_df['first_cat'] = later_df['doctor id'].map(first_cats_map)
    later_df = later_df.dropna(subset=['first_cat'])

    stickiness_map = {}
    for fc, grp in later_df.groupby('first_cat'):
        stickiness_map[fc] = round((grp['cat_short'] == fc).mean() * 100, 1)

    rows = []
    for cat, grp in doc_first.groupby('cat_short'):
        convs = grp['total_convs']
        rows.append({
            'category':       cat,
            'conversion_pct': round((convs > 1).mean() * 100, 1),
            'avg_convs':      round(float(convs.mean()), 2),
            'pct_5plus':      round((convs >= 5).mean() * 100, 1),
            'stickiness_pct': stickiness_map.get(cat, 0),
        })

    if len(rows) > 1:
        for metric in ['conversion_pct', 'avg_convs', 'pct_5plus', 'stickiness_pct']:
            vals = np.array([r[metric] for r in rows], dtype=float)
            std  = vals.std()
            mean = vals.mean()
            zs   = (vals - mean) / std if std > 0 else np.zeros(len(vals))
            for r, z in zip(rows, zs):
                r[f'{metric}_z'] = float(z)
        for r in rows:
            z_cols = [r[f'{m}_z'] for m in ['conversion_pct', 'avg_convs', 'pct_5plus', 'stickiness_pct']]
            r['engagement_score'] = round(np.mean(z_cols), 2)
            for m in ['conversion_pct', 'avg_convs', 'pct_5plus', 'stickiness_pct']:
                del r[f'{m}_z']

    rows.sort(key=lambda r: -r.get('engagement_score', 0))
    return rows


# ── Consolidated retention (all cohorts pooled) ────────────────────────────────

def get_retention_consolidated(data, max_weeks=12):
    """Single retention curve aggregated across all cohorts.
    For each week W, denominator = only doctors for whom W is already observable."""
    if data.empty:
        return {'total_docs': 0, 'pct_single': 0.0, 'pct_w0_return': 0.0, 'weeks': []}

    max_week_num  = int(data['conv_week_num'].max())
    doc_first     = data.sort_values('first_dt').drop_duplicates('doctor id', keep='first')
    total_docs    = len(doc_first)

    pct_single    = round(float((doc_first['total_convs'] == 1).mean() * 100), 1)
    w0_counts     = data[data['week_offset'] == 0].groupby('doctor id').size()
    pct_w0_return = round((w0_counts > 1).sum() / total_docs * 100, 1) if total_docs else 0.0

    weeks = []
    for w in range(1, max_weeks + 1):
        obs_docs = set(data[data['cohort_week_num'] + w <= max_week_num]['doctor id'])
        if not obs_docs:
            weeks.append(None)
        else:
            active = set(data[data['week_offset'] == w]['doctor id']) & obs_docs
            weeks.append(round(len(active) / len(obs_docs) * 100, 1))

    return {
        'total_docs':     int(total_docs),
        'pct_single':     float(pct_single),
        'pct_w0_return':  float(pct_w0_return),
        'weeks':          weeks,
    }


# ── Inter-question gap distribution ──────────────────────────────────────────

def get_inter_question_gaps(df):
    """Days between each consecutive pair of questions from the same doctor."""
    df_s = df.sort_values(['doctor id', 'first_dt']).copy()
    df_s['prev_dt']  = df_s.groupby('doctor id')['first_dt'].shift(1)
    df_s['gap_days'] = (df_s['first_dt'] - df_s['prev_dt']).dt.days
    gaps = df_s.dropna(subset=['gap_days']).copy()
    gaps['gap_days'] = gaps['gap_days'].astype(int)

    if gaps.empty:
        return None

    bins   = [0, 1, 3, 7, 14, 28, 60, 90, 10_000]
    labels = ['Same day', '1–2d', '3–6d', '1–2w', '2–4w', '1–2mo', '2–3mo', '3mo+']

    gaps['bucket'] = pd.cut(gaps['gap_days'], bins=bins, labels=labels, right=False)
    per_q = gaps['bucket'].value_counts().reindex(labels).fillna(0)

    doc_avg = gaps.groupby('doctor id')['gap_days'].mean()
    doc_bucket = pd.cut(doc_avg, bins=bins, labels=labels, right=False)
    per_doc = doc_bucket.value_counts().reindex(labels).fillna(0)

    n_gaps = len(gaps)
    n_docs = len(doc_avg)

    return {
        'labels': labels,
        'per_question': {
            'counts': [int(v) for v in per_q.values],
            'pcts':   [round(v / n_gaps * 100, 1) if n_gaps else 0 for v in per_q.values],
        },
        'per_doctor': {
            'counts': [int(v) for v in per_doc.values],
            'pcts':   [round(v / n_docs * 100, 1) if n_docs else 0 for v in per_doc.values],
        },
        'stats': {
            'median_days': round(float(gaps['gap_days'].median()), 1),
            'mean_days':   round(float(gaps['gap_days'].mean()), 1),
            'p25_days':    round(float(gaps['gap_days'].quantile(0.25)), 1),
            'p75_days':    round(float(gaps['gap_days'].quantile(0.75)), 1),
            'total_gaps':  int(n_gaps),
            'repeat_docs': int(n_docs),
        },
    }


# ── Category × Specialty matrices ─────────────────────────────────────────────

def _cat_spec_matrix(df, top_n=12):
    """% of conversations per specialty that fall in each category (row-normalised)."""
    top_specs = (df.drop_duplicates('doctor id')['specialty_norm']
                   .dropna().value_counts().head(top_n).index.tolist())
    cats = df['cat_short'].value_counts().index.tolist()
    df_t = df[df['specialty_norm'].isin(top_specs)]
    rows = []
    for spec in top_specs:
        sub   = df_t[df_t['specialty_norm'] == spec]
        total = len(sub)
        rows.append([round((sub['cat_short'] == c).sum() / total * 100, 1) if total else 0
                     for c in cats])
    return {'specialties': top_specs, 'categories': cats, 'values': rows}

def get_cat_specialty_matrix(df, top_n=12):
    return _cat_spec_matrix(df, top_n)

def get_firstq_specialty_matrix(df, top_n=12):
    doc_first = df.sort_values('first_dt').drop_duplicates('doctor id', keep='first')
    return _cat_spec_matrix(doc_first, top_n)


# ── Category engagement drilldown ─────────────────────────────────────────────

def get_category_engagement_drilldown(df, min_cat_n=20, min_seg_n=10, top_weak=5):
    """
    For every category: % of doctors who asked it as their first question and returned.
    For the weakest categories (by that rate), cross-tab by country / specialty / source
    to surface which specific combinations drive the low conversion.
    """
    doc_first = (df.sort_values('first_dt')
                   .drop_duplicates('doctor id', keep='first')
                   .copy())
    doc_first['returned'] = (doc_first['total_convs'] > 1).astype(int)

    overall_conv = round(float(doc_first['returned'].mean() * 100), 1) if len(doc_first) > 0 else 0.0

    cat_summary = []
    for cat, grp in doc_first.groupby('cat_short'):
        n = len(grp)
        if n < min_cat_n:
            continue
        conv = round(float(grp['returned'].mean() * 100), 1)
        cat_summary.append({'category': cat, 'n': int(n), 'conversion_pct': conv})
    cat_summary.sort(key=lambda r: r['conversion_pct'])

    weak_cats = [r['category'] for r in cat_summary[:top_weak]]

    drilldowns = {}
    for cat in weak_cats:
        cat_df = doc_first[doc_first['cat_short'] == cat].copy()
        drilldowns[cat] = {}

        for dim, col in [('country',   'doctor_country'),
                         ('specialty', 'doctor_specialty'),
                         ('source',    'doctor_source')]:
            if col not in cat_df.columns or cat_df[col].isna().all():
                drilldowns[cat][dim] = []
                continue
            rows = []
            for seg, grp in cat_df.groupby(col):
                seg_str = str(seg).strip()
                if not seg_str or seg_str.lower() in ('nan', 'none', ''):
                    continue
                n = len(grp)
                if n < min_seg_n:
                    continue
                conv = round(float(grp['returned'].mean() * 100), 1)
                rows.append({'segment': seg_str, 'n': int(n), 'conversion_pct': conv})
            rows.sort(key=lambda r: r['conversion_pct'])
            drilldowns[cat][dim] = rows

    return {
        'cat_summary':        cat_summary,
        'weak_cats':          weak_cats,
        'drilldowns':         drilldowns,
        'overall_conversion': overall_conv,
    }


# ── Cohort recompute helper (used for category-filtered retention) ─────────────

def recompute_cohort_cols(df):
    """Recalculate cohort columns after any row-level filtering."""
    if df.empty:
        return df
    df = df.copy()
    df['conv_week_num'] = ((df['first_dt'] - EPOCH).dt.days // 7).astype(int)
    df = df.drop(columns=['cohort_week_num', 'week_offset', 'cohort_monday', 'total_convs'],
                 errors='ignore')
    doc_first_week = df.groupby('doctor id')['conv_week_num'].min().rename('cohort_week_num')
    df = df.merge(doc_first_week, on='doctor id', how='left')
    df['week_offset']    = df['conv_week_num'] - df['cohort_week_num']
    df['cohort_monday']  = (EPOCH + pd.to_timedelta(df['cohort_week_num'] * 7, unit='D')).dt.date
    doc_count = df.groupby('doctor id').size().reset_index(name='total_convs')
    df = df.merge(doc_count, on='doctor id', how='left')
    return df


# ── Shared result builder ──────────────────────────────────────────────────────

def build_results(df):
    cats = sorted(df['cat_short'].dropna().unique().tolist())
    return {
        'meta':                       get_meta(df),
        'category_names':             cats,
        'cat_dist':                   get_cat_dist(df),
        'cat_specialty_matrix':       get_cat_specialty_matrix(df),
        'firstq_specialty_matrix':    get_firstq_specialty_matrix(df),
        'summary_distributions':      get_summary_distributions(df),
        'first_q_table':              get_first_q_table(df),
        'last_q_raw':                 get_last_q_raw(df),
        'retention_consolidated':      get_retention_consolidated(df),
        'retention_consolidated_repeat': get_retention_consolidated(df[df['total_convs'] > 1].copy()),
        'retention_overall':          build_retention_list(df, max_weeks=12, min_size=1),
        'retention_overall_repeat':   build_retention_list(
                                          df[df['total_convs'] > 1].copy(),
                                          max_weeks=12, min_size=1),
        'retention_by_country':       get_retention_by_segment(df, 'doctor_country',   min_cohort_size=15),
        'retention_by_specialty':     get_retention_by_segment(df, 'doctor_specialty',  min_cohort_size=15),
        'retention_by_source':        get_retention_by_segment(df, 'doctor_source',     min_cohort_size=15),
        'retention_agg_by_country':   get_retention_aggregate_by_segment(df, 'doctor_country',  min_doctors=10),
        'retention_agg_by_specialty': get_retention_aggregate_by_segment(df, 'doctor_specialty', min_doctors=10),
        'retention_agg_by_source':    get_retention_aggregate_by_segment(df, 'doctor_source',   min_doctors=10),
        'top_specialty_cohorts':      get_top_specialty_cohorts(df, top_n=3),
        'inter_question_gaps':        get_inter_question_gaps(df),
        'qual_stickiness':            get_qual_stickiness(df),
        'qual_depth':                 get_qual_depth(df),
        'qual_power_users':           get_qual_power_users(df),
        'qual_sequences':             get_qual_sequences(df),
        'qual_weekly_shift':          get_qual_weekly_shift(df),
        'engagement_score':           get_engagement_score(df),
        'cat_engagement_drilldown':   get_category_engagement_drilldown(df),
    }


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


def _json_response(obj):
    return app.response_class(
        response=json.dumps(obj, default=lambda o:
            int(o)     if isinstance(o, np.integer)  else
            float(o)   if isinstance(o, np.floating) else
            o.tolist() if isinstance(o, np.ndarray)  else str(o)),
        mimetype='application/json'
    )


@app.route('/preview', methods=['POST'])
def preview():
    """Return CSV column names, auto-detected mappings, and 5 sample rows."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        file    = request.files['file']
        content = file.read().decode('utf-8', errors='replace')
        df      = pd.read_csv(io.StringIO(content), nrows=5)
        df.columns = df.columns.str.strip()
        columns    = df.columns.tolist()
        detected   = _auto_detect_cols(df)
        total_rows = content.count('\n') - 1  # fast row count
        sample     = df.head(5).fillna('').astype(str).to_dict('records')
        return jsonify({
            'columns':    columns,
            'detected':   detected,
            'sample':     sample,
            'total_rows': max(total_rows, 0),
            'filename':   file.filename,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    global _cached_df
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        content  = file.read().decode('utf-8', errors='replace')
        df_raw   = pd.read_csv(io.StringIO(content))
        # Accept explicit column mappings if provided
        mappings_json = request.form.get('mappings', '')
        col_mappings  = json.loads(mappings_json) if mappings_json else None
        df       = preprocess(df_raw, col_mappings=col_mappings)
        _cached_df = df
        return _json_response(build_results(df))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/demo')
def demo():
    """Load the sample CSV without requiring a file upload (for preview/testing)."""
    global _cached_df
    import os
    demo_path = os.path.expanduser(
        '~/Downloads/Evidence questions categorized - categorized conversations (1).csv'
    )
    if not os.path.exists(demo_path):
        return jsonify({'error': 'Demo file not found'}), 404
    df_raw     = pd.read_csv(demo_path)
    df         = preprocess(df_raw)
    _cached_df = df
    return _json_response(build_results(df))


@app.route('/filter_retention', methods=['POST'])
def filter_retention():
    """Return consolidated retention after excluding selected categories."""
    global _cached_df
    if _cached_df is None:
        return jsonify({'error': 'No data loaded — upload or open /demo first'}), 400
    try:
        body     = request.json or {}
        excluded = set(body.get('excluded', []))
        mode     = body.get('mode', 'all')   # 'all' | 'first'

        df = _cached_df.copy()

        if not excluded:
            df_f = df
        elif mode == 'all':
            # Drop every conversation of excluded categories
            df_f = df[~df['cat_short'].isin(excluded)].copy()
            df_f = recompute_cohort_cols(df_f)
        else:
            # 'first' mode: keep all questions but re-anchor cohort to first
            # non-excluded question per doctor
            non_excl = df[~df['cat_short'].isin(excluded)]
            doc_new_first = (non_excl.groupby('doctor id')['first_dt']
                                     .min().rename('_new_first_dt'))
            df_f = df.merge(doc_new_first, on='doctor id', how='inner').copy()
            df_f['conv_week_num']  = ((df_f['first_dt']       - EPOCH).dt.days // 7).astype(int)
            df_f['cohort_week_num']= ((df_f['_new_first_dt']  - EPOCH).dt.days // 7).astype(int)
            df_f['week_offset']    = df_f['conv_week_num'] - df_f['cohort_week_num']
            df_f['week_offset']    = df_f['week_offset'].clip(lower=0)
            df_f['cohort_monday']  = (EPOCH + pd.to_timedelta(
                                         df_f['cohort_week_num'] * 7, unit='D')).dt.date
            total_c = df_f.groupby('doctor id').size().reset_index(name='total_convs')
            df_f = df_f.drop(columns=['total_convs', '_new_first_dt'], errors='ignore')
            df_f = df_f.merge(total_c, on='doctor id', how='left')

        if df_f.empty:
            return jsonify({'error': 'No data after filtering'}), 400

        repeat_df = df_f[df_f['total_convs'] > 1].copy()
        result = {
            'meta':                          get_meta(df_f),
            'retention_consolidated':         get_retention_consolidated(df_f),
            'retention_consolidated_repeat':  get_retention_consolidated(repeat_df)
                                              if not repeat_df.empty else None,
        }
        return _json_response(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import socket
    local_ip = socket.gethostbyname(socket.gethostname())
    print("\n  Evidence Analysis Dashboard")
    print(f"  Local:   http://127.0.0.1:5000")
    print(f"  Network: http://{local_ip}:5000  ← share this with colleagues\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
