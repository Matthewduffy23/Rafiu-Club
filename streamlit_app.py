import re
from pathlib import Path
from typing import List, Set

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Scouting – Club Fit Finder", page_icon="⚽", layout="wide")
DATA_FILE = "club_profiles.csv"   # your exported combined CSV

# ---- Section rules (edit if your labels differ) ----
SECTION_MAP = {
    "CBs":  {"CB","RCB","LCB"},
    "FBs":  {"RB","LB","RWB","LWB","FB"},
    "CMs":  {"CM","DM","CDM","CAM","LCM","RCM","AMF","CMF"},
    "ATTs": {"LW","RW","LAMF","RAMF","WF","SS","AM","LWF","RWF","WMF"},
    "CF":   {"CF","ST","9","9.5"},
}

# ---- helpers ----
def _to_num(s):
    if pd.isna(s): return pd.NA
    if not isinstance(s, str): return s
    s = re.sub(r"[€$,%\s]", "", s).replace(",", "")
    return pd.to_numeric(s, errors="coerce")

def tokenize_positions(s: str) -> Set[str]:
    """Split 'CF, AMF, LWF' -> {'CF','AMF','LWF'}; tolerant to separators/spaces."""
    if not isinstance(s, str) or not s.strip():
        return set()
    # split on comma/semicolon/slash/pipe
    parts = re.split(r"[,\;/|]+", s)
    return {p.strip().upper() for p in parts if p.strip()}

def infer_section_from_tokens(tokens: Set[str]) -> str | None:
    """Pick first matching section whose set intersects the tokens."""
    for sec, allowed in SECTION_MAP.items():
        if tokens & allowed:
            return sec
    return None

@st.cache_data
def load(path: str):
    if not Path(path).exists():
        st.error(f"CSV not found: {path}")
        st.stop()

    # raw: keep exactly as strings for display
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    original_cols = raw.columns.tolist()

    # numeric shadow copy
    num = raw.copy()
    for c in ["Age","Market value","League Strength","Final Fit %","Club Fit %","Value Fit %"]:
        if c in num.columns:
            num[c+"__num"] = num[c].map(_to_num)

    # position tokens for each row (for proper filtering)
    pos_col = None
    for cand in ["Position","Pos","Role"]:
        if cand in raw.columns:
            pos_col = cand
            break
    if pos_col is None:
        st.error("No Position/Pos/Role column found in CSV.")
        st.stop()

    pos_tokens = raw[pos_col].apply(tokenize_positions)

    # ensure Section column exists; infer if missing/empty
    if "Section" not in raw.columns or raw["Section"].astype(str).str.strip().eq("").all():
        inferred = pos_tokens.apply(infer_section_from_tokens)
        raw.insert(0, "Section", inferred.fillna("Unclassified"))
        original_cols = raw.columns.tolist()  # keep this order (Section now present)

    # collect unique positions (tokens) seen in data
    all_pos_tokens = sorted(set().union(*pos_tokens.tolist()))

    return raw, num, original_cols, pos_col, pos_tokens, all_pos_tokens

raw, num, original_cols, POS_COL, POS_TOKENS, ALL_POS = load(DATA_FILE)

# ---------------- UI ----------------
st.title("Scouting – Club Fit Finder")
st.caption("Filter by section/position, age, league strength, and market value. Table keeps your CSV’s exact column order & formatting.")

top = st.columns([2,1,1])
with top[0]:
    query = st.text_input("Search (Team / League / Position)", value="").strip()
with top[1]:
    topk = st.number_input("Rows to show", min_value=5, max_value=500, value=25, step=5)
with top[2]:
    show_dl = st.checkbox("Enable CSV download", value=True)

row1 = st.columns([1, 2])
with row1[0]:
    sections = [sec for sec in ["CBs","FBs","CMs","ATTs","CF"] if (raw["Section"] == sec).any()]
    section = st.selectbox("Section", options=sections if sections else ["Unclassified"])
with row1[1]:
    # default to all tokens present in this section
    mask_sec = (raw["Section"] == section)
    tokens_in_section = sorted(set().union(*POS_TOKENS[mask_sec].tolist())) if mask_sec.any() else ALL_POS
    pos_sel = st.multiselect("Positions (tokenized)", options=tokens_in_section, default=tokens_in_section)

row2 = st.columns(4)
with row2[0]:
    # Age
    if "Age__num" in num.columns and num["Age__num"].notna().any():
        a_min = int(num["Age__num"].min(skipna=True)); a_max = int(num["Age__num"].max(skipna=True))
        age_rng = st.slider("Age", a_min, a_max, (a_min, a_max))
    else:
        age_rng = None

with row2[1]:
    # League Strength
    if "League Strength__num" in num.columns and num["League Strength__num"].notna().any():
        q_min = float(num["League Strength__num"].min(skipna=True)); q_max = float(num["League Strength__num"].max(skipna=True))
        lq_rng = st.slider("League Strength", float(q_min), float(q_max), (float(q_min), float(q_max)))
    else:
        lq_rng = None

with row2[2]:
    # Market value
    if "Market value__num" in num.columns and num["Market value__num"].notna().any():
        v_min = float(num["Market value__num"].min(skipna=True)); v_max = float(num["Market value__num"].max(skipna=True))
        val_rng = st.slider("Market value (units = CSV)", float(max(0, v_min)), float(max(v_max, v_min+1)),
                            (float(max(0, v_min)), float(v_max)))
    else:
        val_rng = None

with row2[3]:
    # Sort
    default_sort = "Final Fit %" if "Final Fit %" in original_cols else original_cols[0]
    sort_col = st.selectbox("Sort by", original_cols, index=original_cols.index(default_sort))
    asc = st.checkbox("Ascending", value=False)

# ---------------- Filtering ----------------
idx = pd.Series(True, index=raw.index)

# Section
if "Section" in raw.columns and section:
    idx &= (raw["Section"] == section)

# Positions (match tokens overlap)
if pos_sel:
    has_any = POS_TOKENS.apply(lambda toks: bool(set(pos_sel) & toks))
    idx &= has_any

# Search
if query:
    m = pd.Series(False, index=raw.index)
    for c in ["Team","League",POS_COL]:
        if c in raw.columns:
            m |= raw[c].astype(str).str.contains(query, case=False, na=False)
    idx &= m

# Numeric ranges
if age_rng and "Age__num" in num.columns:
    idx &= (num["Age__num"] >= age_rng[0]) & (num["Age__num"] <= age_rng[1])
if lq_rng and "League Strength__num" in num.columns:
    idx &= (num["League Strength__num"] >= lq_rng[0]) & (num["League Strength__num"] <= lq_rng[1])
if val_rng and "Market value__num" in num.columns:
    idx &= (num["Market value__num"] >= val_rng[0]) & (num["Market value__num"] <= val_rng[1])

# Build display df (exact CSV format/column order)
view = raw.loc[idx, original_cols].copy()

# Sort: try numeric sense first
if sort_col in view.columns:
    tmp = view[sort_col].map(_to_num)
    if pd.notna(tmp).mean() > 0.5:
        view = view.assign(_S=tmp).sort_values("_S", ascending=asc, kind="mergesort").drop(columns="_S")
    else:
        view = view.sort_values(sort_col, ascending=asc, kind="mergesort")

st.markdown(f"**Results:** {len(view):,} rows (showing first {min(len(view), int(topk)):,})")
st.dataframe(view.head(int(topk)), use_container_width=True)

if show_dl and len(view):
    st.download_button("⬇️ Download filtered CSV",
                       data=view.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_results.csv",
                       mime="text/csv")
