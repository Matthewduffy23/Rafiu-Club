import streamlit as st
import pandas as pd

st.set_page_config(page_title="Similar Players / Club Fit", page_icon="⚽", layout="wide")

@st.cache_data
def load_data():
    """
    Tries to load a prepared CSV with club fit outputs.
    Put a file named 'club_profiles.csv' in the repo root for full results.
    Fallback: a tiny demo dataframe so the app still runs.
    """
    try:
        df = pd.read_csv("club_profiles.csv")
        # normalize column names a bit if needed
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        # --- Fallback demo so the app runs before you export real results
        df = pd.DataFrame({
            "Team": ["Borussia Dortmund", "RB Leipzig", "AC Milan", "Brighton"],
            "League": ["Bundesliga", "Bundesliga", "Serie A", "Premier League"],
            "League Strength": [88, 85, 83, 86],
            "Market value": [62, 58, 65, 55],
            "Club Fit %": [92, 90, 88, 86],
            "Value Fit %": [92, 89, 87, 85],
            "Final Fit %": [92, 90, 88, 86],
        })
        return df

df = load_data()

st.title("Similar Players / Club Fit Finder")
st.caption("Type a name and adjust filters to see best club fits. Replace the demo data by adding 'club_profiles.csv' to this repo.")

# ---- Controls
left, right = st.columns([2, 1])
with left:
    player_name = st.text_input("Player name", value="Rafiu Durosinmi").strip()
with right:
    topk = st.number_input("How many results?", min_value=3, max_value=50, value=10, step=1)

# Optional filters if your CSV has these columns
f1, f2, f3 = st.columns(3)
with f1:
    leagues = sorted(df["League"].dropna().unique()) if "League" in df.columns else []
    chosen_leagues = st.multiselect("League filter", leagues, default=leagues[:3] if leagues else [])
with f2:
    min_fit = st.slider("Min Final Fit %", 0, 100, 70)
with f3:
    sort_by = st.selectbox("Sort by", [c for c in df.columns if c.endswith("%")] or df.columns, index= ( [c.endswith("%") for c in df.columns].index(True) if any(c.endswith("%") for c in df.columns) else 0))
    ascending = st.checkbox("Ascending sort", value=False)

# ---- Filtering
df_show = df.copy()
if chosen_leagues and "League" in df_show.columns:
    df_show = df_show[df_show["League"].isin(chosen_leagues)]
if "Final Fit %" in df_show.columns:
    df_show = df_show[df_show["Final Fit %"] >= min_fit]

# You can optionally personalize results by name (e.g., position/age filters) — for now we just display.
df_show = df_show.sort_values(sort_by, ascending=ascending).head(int(topk))

st.subheader(f"Top Matches for: {player_name if player_name else '—'}")
st.dataframe(df_show, use_container_width=True)

with st.expander("How to plug in your real outputs"):
    st.markdown(
        """
        1. In your notebook, **export the final table** (the one with fit scores) to CSV:
           ```python
           club_profiles.to_csv("club_profiles.csv", index=False)
           ```
        2. Upload **club_profiles.csv** to the repo root on GitHub.
        3. Redeploy or refresh the app — it will automatically load your real data.
        """
    )
