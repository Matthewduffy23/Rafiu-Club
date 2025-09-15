import re
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Club Fit – Finder", page_icon="⚽", layout="wide")

DATA_FILE = "results.csv"   # <-- your uploaded file
ID_COL = "Player"           # typed name filters this column
DISPLAY_COLS = [
    "Team","Position","League","Age","Market value",
    "League Strength","Final Fit %","Club Fit %","Value Fit %"
]

def _to_num(s):
    if pd.isna(s): return pd.NA
    if not isinstance(s, str): return s
    s = re.sub(r"[€$,%\s]", "", s).replace(",", "")
    return pd.to_numeric(s, errors="coerce")

@st.cache_data
def load_data(path: str):
    if not Path(path).exists():
        st.error(f"CSV not found: {path}")
        st.stop()
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]

    # make sure the columns we show exist (blank if missing)
    for c in [ID_COL] + DISPLAY_COLS:
        if c not in raw.columns:
            raw[c] = ""

    # numeric shadows for filters/sort
    num = raw.copy()
    for c in ["Age","Market value","League Strength","Final Fit %","Club Fit %","Value Fit %"]:
        if c in num.columns:
            num[c+"__num"] = num[c].map(_to_num)

    # only show the columns we want (Player first)
    show_cols = [ID_COL] + [c for c in DISPLAY_COLS if c in raw.columns]
    return raw[show_cols], num

raw, num = load_data(DATA_FILE)

st.title("Club Fit – Finder")
st.caption("Type a player’s name. Adjust League Strength & Age. The table shows only your chosen columns in the same order.")

# Controls
c1, c2, c3 = st.columns([2,1,1])
with c1:
    player = st.text_input("Player name", value="").strip()
with c2:
    topk = st.number_input("Rows to show", min_value=5, max_value=500, value=25, step=5)
with c3:
    enable_dl = st.checkbox("Enable CSV download", True)

f1, f2 = st.columns(2)
with f1:
    if "League Strength__num" in num.columns and num["League Strength__num"].notna().any():
        qmin = float(num["League Strength__num"].min(skipna=True)); qmax = float(num["League Strength__num"].max(skipna=True))
        lq = st.slider("League Strength", float(qmin), float(qmax), (float(qmin), float(qmax)))
    else:
        lq = None
with f2:
    if "Age__num" in num.columns and num["Age__num"].notna().any():
        amin = int(num["Age__num"].min(skipna=True)); amax = int(num["Age__num"].max(skipna=True))
        age = st.slider("Age", amin, amax, (amin, amax))
    else:
        age = None

# Sorting
sort_cols = [c for c in DISPLAY_COLS if c in raw.columns]
default_sort = "Final Fit %" if "Final Fit %" in sort_cols else sort_cols[0]
s1, s2 = st.columns([3,1])
with s1:
    sort_by = st.selectbox("Sort by", sort_cols, index=sort_cols.index(default_sort))
with s2:
    asc = st.checkbox("Ascending", False)

# Filtering
view = raw.copy()
if player:
    view = view[view[ID_COL].astype(str).str.contains(player, case=False, na=False)]

idx = view.index
if lq is not None and "League Strength__num" in num.columns:
    n = num.loc[idx, "League Strength__num"]
    idx = idx[(n >= lq[0]) & (n <= lq[1])]
if age is not None and "Age__num" in num.columns:
    n = num.loc[idx, "Age__num"]
    idx = idx[(n >= age[0]) & (n <= age[1])]
view = view.loc[idx]

# Numeric-aware sort
tmp = view[sort_by].map(_to_num) if sort_by in view.columns else None
if tmp is not None and pd.notna(tmp).mean() > 0.5:
    view = view.assign(_S=tmp).sort_values("_S", ascending=asc, kind="mergesort").drop(columns="_S")
else:
    view = view.sort_values(sort_by, ascending=asc, kind="mergesort")

st.markdown(f"**Results:** {len(view):,} rows (showing first {min(len(view), int(topk)):,})")
st.dataframe(view.head(int(topk)), use_container_width=True)

if enable_dl and len(view):
    st.download_button(
        "⬇️ Download filtered CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"{(player or 'results').replace(' ','_')}.csv",
        mime="text/csv",
    )
