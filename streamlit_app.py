import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIG (defaults you can edit)
# -----------------------------
DEFAULT_FEATURES = [
    'Defensive duels per 90', 'Aerial duels per 90', 'Aerial duels won, %', 'PAdj Interceptions',
    'Non-penalty goals per 90', 'xG per 90', 'Shots per 90', 'Shots on target, %', 'Goal conversion, %',
    'Crosses per 90', 'Accurate crosses, %', 'Dribbles per 90', 'Successful dribbles, %',
    'Offensive duels per 90', 'Touches in box per 90', 'Progressive runs per 90', 'Accelerations per 90',
    'Passes per 90', 'Accurate passes, %', 'xA per 90', 'Smart passes per 90', 'Key passes per 90',
    'Passes to final third per 90', 'Passes to penalty area per 90', 'Accurate passes to penalty area, %',
    'Deep completions per 90'
]

# IMPORTANT: keys must match feature names EXACTLY
DEFAULT_WEIGHT_FACTORS = {
    'Passes per 90': 2,
    'Accurate passes, %': 2,
    'Dribbles per 90': 2,
    'Non-penalty goals per 90': 2,   # fixed key (was "Non-Penalty..." in notebook)
    'Shots per 90': 2,
    'Successful dribbles, %': 2,
    'Aerial duels won, %': 2,
    'xA per 90': 2,
    'xG per 90': 2,
    'Touches in box per 90': 2,
}

# Reasonable league strengths; you can edit in the UI
DEFAULT_LEAGUE_STRENGTHS = {
    'England 1.': 100.00, 'Italy 1.': 97.14, 'Spain 1.': 94.29, 'Germany 1.': 94.29, 'France 1.': 91.43,
    'Brazil 1.': 82.86, 'England 2.': 71.43, 'Portugal 1.': 71.43, 'Argentina 1.': 71.43,
    'Belgium 1.': 68.57, 'Mexico 1.': 68.57, 'Turkey 1.': 65.71, 'Germany 2.': 65.71, 'Spain 2.': 65.71,
    'France 2.': 65.71, 'USA 1.': 65.71, 'Russia 1.': 65.71, 'Colombia 1.': 62.86, 'Netherlands 1.': 62.86,
    'Austria 1.': 62.86, 'Switzerland 1.': 62.86, 'Denmark 1.': 62.86, 'Croatia 1.': 62.86,
    'Japan 1.': 62.86, 'Korea 1.': 62.86, 'Italy 2.': 62.86, 'Czech 1.': 57.14, 'Norway 1.': 57.14,
    'Poland 1.': 57.14, 'Romania 1.': 57.14, 'Israel 1.': 57.14, 'Algeria 1.': 57.14, 'Paraguay 1.': 57.14,
    'Saudi 1.': 57.14, 'Uruguay 1.': 57.14, 'Morocco 1.': 57.00, 'Brazil 2.': 56.00, 'Ukraine 1.': 54.29,
    'Ecuador 1.': 54.29, 'Spain 3.': 54.29, 'Scotland 1.': 54.29, 'Chile 1.': 51.43, 'Cyprus 1.': 51.43,
    'Portugal 2.': 51.43, 'Slovakia 1.': 51.43, 'Australia 1.': 51.43, 'Hungary 1.': 51.43,
    'Egypt 1.': 51.43, 'England 3.': 51.43, 'France 3.': 48.00, 'Japan 2.': 48.00, 'Bulgaria 1.': 48.57,
    'Slovenia 1.': 48.57, 'Venezuela 1.': 48.00, 'Germany 3.': 45.71, 'Albania 1.': 44.00, 'Serbia 1.': 42.86,
    'Belgium 2.': 42.86, 'Bosnia 1.': 42.86, 'Kosovo 1.': 42.86, 'Nigeria 1.': 42.86, 'Azerbaijan 1.': 50.00,
    'Bolivia 1.': 50.00, 'Costa Rica 1.': 50.00, 'South Africa 1.': 50.00, 'UAE 1.': 50.00, 'Georgia 1.': 40.00,
    'Finland 1.': 40.00, 'Italy 3.': 40.00, 'Peru 1.': 40.00, 'Tunisia 1.': 40.00, 'USA 2.': 40.00,
    'Armenia 1.': 40.00, 'North Macedonia 1.': 40.00, 'Qatar 1.': 40.00, 'Uzbekistan 1.': 42.00,
    'Norway 2.': 42.00, 'Kazakhstan 1.': 42.00, 'Poland 2.': 38.00, 'Denmark 2.': 37.00, 'Czech 2.': 37.14,
    'Israel 2.': 37.14, 'Netherlands 2.': 37.14, 'Switzerland 2.': 37.14, 'Iceland 1.': 34.29,
    'Ireland 1.': 34.29, 'Sweden 2.': 34.29, 'Germany 4.': 34.29, 'Malta 1.': 30.00, 'Turkey 2.': 35,
    'Canada 1.': 28.57, 'England 4.': 28.57, 'Scotland 2.': 28.57, 'Moldova 1.': 28.57, 'Austria 2.': 25.71,
    'Lithuania 1.': 25.71, 'Brazil 3.': 25.00, 'England 7.': 25.00, 'Slovenia 2.': 22.00, 'Latvia 1.': 22.86,
    'Serbia 2.': 20.00, 'Slovakia 2.': 20.00, 'England 9.': 20.00, 'England 8.': 15.00, 'Montenegro 1.': 14.29,
    'Wales 1.': 12.00, 'Portugal 3.': 11.43, 'Northern Ireland 1.': 11.43, 'England 5.': 11.43,
    'Andorra 1.': 10.00, 'Estonia 1.': 8.57, 'England 10.': 5.00, 'Scotland 3.': 0.00, 'England 6.': 0.00
}

# -----------------------------
# Helpers
# -----------------------------
def compute_club_fits(
    df: pd.DataFrame,
    target_player: str,
    features: list,
    weight_factors: dict,
    included_leagues: list,
    league_strengths: dict,
    position_prefix: str = 'CF',
    league_weight: float = 0.3,
    market_value_weight: float = 0.3,
    manual_override: float | None = None,
    top_n: int = 15
) -> tuple[pd.DataFrame, dict]:
    # Filter to position + leagues + complete features
    df = df.copy()
    df = df[df['Position'].astype(str).str.startswith(position_prefix)]
    df = df[df['League'].isin(included_leagues)]
    df = df.dropna(subset=features)

    # Types
    df['Market value'] = pd.to_numeric(df['Market value'], errors='coerce')

    # Target row
    if target_player not in df['Player'].values:
        raise ValueError(f"Player '{target_player}' not found in filtered data. "
                         f"Check name, position prefix='{position_prefix}', and included leagues.")
    target_row = df.loc[df['Player'] == target_player].iloc[0]
    target_vector = target_row[features].values

    # Market value
    if manual_override is not None:
        target_mv = float(manual_override)
    else:
        mv = target_row['Market value']
        target_mv = 2_000_000 if (pd.isna(mv) or mv == 0) else float(mv)

    # Club profiles
    club_profiles = df.groupby(['Team', 'Position'])[features].mean().reset_index()

    scaler = StandardScaler()
    scaled_club_features = scaler.fit_transform(club_profiles[features])
    target_scaled = scaler.transform([target_vector])[0]

    # Similarity (weighted Euclidean)
    weights = np.array([weight_factors.get(f, 1.0) for f in features], dtype=float)
    diffs = (scaled_club_features - target_scaled) * weights
    distance = np.linalg.norm(diffs, axis=1)
    club_profiles['Distance'] = distance

    # Normalize distance → 0..100 (guard divide-by-zero)
    dmin, dmax = float(distance.min()), float(distance.max())
    if dmax - dmin == 0:
        club_fit = np.full_like(distance, 100.0)
    else:
        club_fit = (1 - (distance - dmin) / (dmax - dmin)) * 100
    club_profiles['Club Fit %'] = club_fit.round(2)

    # League mapping & strength
    team_league_map = df.groupby('Team')['League'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    club_profiles['League'] = club_profiles['Team'].map(team_league_map)
    club_profiles['League Strength'] = club_profiles['League'].map(league_strengths).fillna(0.0)

    target_ls = float(league_strengths.get(target_row['League'], 1.0))

    # Optional: basic league gating (here we keep all, but you can constrain if needed)
    club_profiles = club_profiles[(club_profiles['League Strength'] >= 0) & (club_profiles['League Strength'] <= 101)]

    # Adjust for league difference
    diff_ratio = (club_profiles['League Strength'] / target_ls).clip(0.5, 1.2)
    adjusted = (club_profiles['Club Fit %'] * (1 - league_weight) +
                club_profiles['Club Fit %'] * diff_ratio * league_weight)

    league_gap = (club_profiles['League Strength'] - target_ls).clip(lower=0)
    league_penalty = (1 - (league_gap / 100)).clip(lower=0.7)
    adjusted *= league_penalty
    club_profiles['Adjusted Fit %'] = adjusted

    # Market value fit
    team_mv = df.groupby('Team')['Market value'].mean()
    club_profiles['Avg Team Market Value'] = club_profiles['Team'].map(team_mv)

    club_profiles = club_profiles.dropna(subset=['Avg Team Market Value'])
    value_fit_ratio = (club_profiles['Avg Team Market Value'] / target_mv).clip(0.5, 1.5)
    value_fit_score = (1 - np.abs(1 - value_fit_ratio)) * 100

    final_fit = (club_profiles['Adjusted Fit %'] * (1 - market_value_weight) +
                 value_fit_score * market_value_weight)
    club_profiles['Final Fit %'] = final_fit

    # Output
    out_cols = ['Team', 'Position', 'League', 'League Strength', 'Avg Team Market Value',
                'Club Fit %', 'Adjusted Fit %', 'Final Fit %']
    results = club_profiles.sort_values('Final Fit %', ascending=False).head(top_n).reset_index(drop=True)

    info = {
        'target_player': target_row['Player'],
        'target_team': target_row['Team'],
        'target_position': target_row['Position'],
        'target_league': target_row['League'],
        'target_market_value': target_mv
    }
    return results[out_cols], info


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Club Fit Finder", layout="wide")
st.title("🔍 Club Fit Finder")

st.markdown(
    "Upload your player dataset and compute **club fit** for any player based on positional+feature similarity, "
    "with adjustments for **league strength** and **market value**."
)

# Data source
with st.expander("📁 Data source", expanded=True):
    up = st.file_uploader("Upload CSV with columns: Player, Team, Position, League, Market value, and feature columns", type=["csv"])
    path = st.text_input("...or type a local CSV path", value="")
    if up is not None:
        df = pd.read_csv(up)
    elif path.strip():
        df = pd.read_csv(path.strip())
    else:
        df = None

if df is not None:
    # Basic guards
    needed_cols = {'Player', 'Team', 'Position', 'League', 'Market value'}
    missing = needed_cols - set(df.columns)
    if missing:
        st.error(f"Your data is missing required columns: {sorted(missing)}")
        st.stop()

    # Feature selection (defaults pre-filled)
    with st.expander("🧩 Features & Weights", expanded=True):
        feature_options = [c for c in DEFAULT_FEATURES if c in df.columns]
        features = st.multiselect("Select features to use", options=feature_options, default=feature_options)

        # Editable weights table (only for selected features)
        wdf = pd.DataFrame({
            'Feature': features,
            'Weight': [DEFAULT_WEIGHT_FACTORS.get(f, 1.0) for f in features]
        })
        wdf = st.data_editor(wdf, num_rows="dynamic", use_container_width=True, key="weights_editor")
        weight_factors = {row.Feature: float(row.Weight) for _, row in wdf.iterrows() if row.Feature in features}

    # Leagues (available from the data)
    all_leagues_in_data = sorted(df['League'].dropna().unique().tolist())
    with st.expander("🏳️ Leagues", expanded=True):
        included_leagues = st.multiselect(
            "Include leagues", options=all_leagues_in_data,
            default=[l for l in all_leagues_in_data if l in DEFAULT_LEAGUE_STRENGTHS] or all_leagues_in_data
        )

        # Editable league strengths (only for included leagues)
        ls_rows = []
        for lg in included_leagues:
            ls_rows.append({
                'League': lg,
                'League Strength': float(DEFAULT_LEAGUE_STRENGTHS.get(lg, 0.0))
            })
        ls_df = pd.DataFrame(ls_rows)
        ls_df = st.data_editor(ls_df, num_rows="dynamic", use_container_width=True, key="league_strengths_editor")
        league_strengths = {row.League: float(row['League Strength']) for _, row in ls_df.iterrows()}

    # Player + knobs
    with st.expander("🎯 Target & Settings", expanded=True):
        # Player picker with typeahead
        player_list = sorted(df['Player'].dropna().unique().tolist())
        target_player = st.selectbox("Choose player", options=player_list)

        # Position filter prefix
        position_prefix = st.text_input("Position prefix (e.g., CF, ST, AMF, etc.)", value="CF")

        # Weights
        colw1, colw2, colw3 = st.columns(3)
        with colw1:
            league_weight = st.slider("League weight", 0.0, 1.0, 0.3, 0.05)
        with colw2:
            market_value_weight = st.slider("Market value weight", 0.0, 1.0, 0.3, 0.05)
        with colw3:
            top_n = st.number_input("How many results", min_value=5, max_value=50, value=15, step=1)

        # Market value override
        manual_override = st.number_input("Manual target market value override (€)", min_value=0.0, value=0.0, step=100000.0)
        manual_override = manual_override if manual_override > 0 else None

    # Run
    run = st.button("▶️ Run analysis", use_container_width=True)

    if run:
        # Validate features
        if not features:
            st.warning("Please select at least one feature.")
            st.stop()

        try:
            results, info = compute_club_fits(
                df=df,
                target_player=target_player,
                features=features,
                weight_factors=weight_factors,
                included_leagues=included_leagues,
                league_strengths=league_strengths,
                position_prefix=position_prefix,
                league_weight=league_weight,
                market_value_weight=market_value_weight,
                manual_override=manual_override,
                top_n=int(top_n)
            )

            # Header
            st.subheader("Top Club Fits")
            st.write(
                f"**Player:** {info['target_player']}  |  **Team:** {info['target_team']}  |  "
                f"**Pos:** {info['target_position']}  |  **League:** {info['target_league']}  |  "
                f"**Market Value Used:** €{info['target_market_value']:,.0f}"
            )

            st.dataframe(results, use_container_width=True)

            # Download
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download results as CSV", data=csv, file_name="club_fits.csv", mime="text/csv")

        except Exception as e:
            st.error(str(e))

else:
    st.info("Upload a CSV or enter a local path to begin.")
