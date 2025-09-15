import re
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Scouting – Club Fit Finder", page_icon="⚽", layout="wide")
DATA_FILE = "club_profiles.csv"

# ----- helpers -----
def _to_num(s):
    if pd.isna(s): return pd.NA
    if not isinstance(s, str): return s
    s = re.sub(r"[€$,%\s]", "", s).replace(",", "")
    return pd.to_numeric(s, errors="coerce")

@st.cache_data
def load(path: str):
    if not Path(path).exists():
        st.error(f"CSV not found: {path}")
        st.stop()
    # keep strings for display to preserve your formatting & column order
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    cols = raw.columns.tolist()

    # numeric shadow copy for filters/sorts
    num = raw.copy()
    for c in ["Age", "Market value", "League Strength", "Final Fit %", "Club Fit %", "Value Fit %"]:
        if c in num.columns:
            num[c + "__num"] = num[c].map(_to_num)

    return raw, num, cols

raw, num, original_cols = load(DATA_FILE)

st.title("Scouting – Club Fit Finder")
st.caption("Filter by section/position, age, league strength, and market value. Table keeps your CSV’s exact column order & formatting.")

# ============== controls ==============
top = st.columns([2, 1, 1])
with top[0]:
    query = st.text_input("Search (Team / League / Position)", value="").strip()
with top[1]:
    topk = st.number_input("Rows to show", min_value=5, max_value=500, value=25, step=5)
with top[2]:
    show_dl = st.checkbox("Enable CSV download", value=True)

row1 = st.columns([1, 2])
with row1[0]:
    # Section comes from your CSV ("Section": CBs/FBs/CMs/ATTs/CF)
    sections = sorted(raw["Section"].unique()) if "Section" in raw.columns else []
    section = st.selectbox("Section", options=sections, index=0 if sections else None)

with row1[1]:
    # Positions available (within selected section)
    if "Position" in raw.columns:
        if section:
            pos_options = sorted(raw.loc[raw["Section"] == section, "Position"].unique())
        else:
            pos_options = sorted(raw["Position"].unique())
        pos_sel = st.multiselect("Positions", options=pos_options, default=pos_options)
    else:
        pos_sel = []

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
        val_rng = st.slider("Market value (same units as CSV)", float(max(0, v_min)), float(max(v_max, v_min+1)), (float(max(0, v_min)), float(v_max)))
    else:
        val_rng = None

with row2[3]:
    # Sort
    default_sort = "Final Fit %" if "Final Fit %" in original_cols else original_cols[0]
    sort_col = st.selectbox("Sort by", original_cols, index=original_cols.index(default_sort))
    asc = st.checkbox("Ascending", value=False)

# ============== filtering (on numeric copy) ==============
idx = pd.Series(True, index=raw.index)

if "Section" in raw.columns and section:
    idx &= raw["Section"] == section

if pos_sel and "Position" in raw.columns:
    idx &= raw["Position"].isin(pos_sel)

if query:
    mask = pd.Series(False, index=raw.index)
    for c in ["Team", "League", "Position"]:
        if c in raw.columns:
            mask |= raw[c].astype(str).str.contains(query, case=False, na=False)
    idx &= mask

if age_rng and "Age__num" in num.columns:
    idx &= (num["Age__num"] >= age_rng[0]) & (num["Age__num"] <= age_rng[1])

if lq_rng and "League Strength__num" in num.columns:
    idx &= (num["League Strength__num"] >= lq_rng[0]) & (num["League Strength__num"] <= lq_rng[1])

if val_rng and "Market value__num" in num.columns:
    idx &= (num["Market value__num"] >= val_rng[0]) & (num["Market value__num"] <= val_rng[1])

# exact-display dataframe (strings) in original order
view = raw.loc[idx, original_cols].copy()

# sort respecting numeric meaning when possible
if sort_col in view.columns:
    tmp = view[sort_col].map(_to_num)
    if pd.notna(tmp).mean() > 0.5:
        view = view.assign(_S=tmp).sort_values("_S", ascending=asc, kind="mergesort").drop(columns="_S")
    else:
        view = view.sort_values(sort_col, ascending=asc, kind="mergesort")

st.markdown(f"**Results:** {len(view):,} rows (showing first {min(len(view), int(topk)):,})")
st.dataframe(view.head(int(topk)), use_container_width=True)

if show_dl and len(view):
    st.download_button("⬇️ Download filtered CSV", data=view.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_results.csv", mime="text/csv")

with st.expander("Notes"):
    st.markdown(
        "- This app preserves your CSV’s formatting (€, % etc.) while filtering on a numeric copy.\n"
        "- To change how scores are computed, update your notebook and re-export `club_profiles.csv`.\n"
        "- Columns used here: Section, Team, Position, League, Age, Market value, League Strength, Final Fit %, Club Fit %, Value Fit %."
    )
