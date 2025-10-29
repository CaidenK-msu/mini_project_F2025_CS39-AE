import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import os
from datetime import date

#Page setup
#-----------------------------------------------------------
st.set_page_config(page_title="Streaming Platform Insights", page_icon="📺", layout="wide")
st.title("Streaming Platform Insights")

st.caption(
    "Alt text: Dashboard with top-row filters and KPI cards, a watch-hours chart, "
    "a regional map of watch time, a summary text panel, and a Top-10 table."
)

#Safe Data Loader
#-----------------------------------------------------------
@st.cache_data
def load_data():
    """Loads viewing and catalog data safely and clearly."""
    try:
        va_path = "data/viewing_activity.csv"
        cat_path = "data/content_catalog.csv"

        #verify
        if not os.path.exists(va_path):
            raise FileNotFoundError(f"Missing file: {va_path}")
        if not os.path.exists(cat_path):
            raise FileNotFoundError(f"Missing file: {cat_path}")

        va = pd.read_csv(va_path, parse_dates=["date"])
        cat = pd.read_csv(cat_path)
        return va, cat, va_path, cat_path

    except Exception as e:
        st.error(f"⚠️ Could not load data: {e}")
        st.stop()

va, catalog, va_path, cat_path = load_data()
st.caption(f"Loaded: `{va_path}` and `{cat_path}`")

#Data prep
#-----------------------------------------------------------
va["day"] = va["date"].dt.date

# Top Row – Filters + KPIs
#-----------------------------------------------------------
filter_cols = st.columns([1.1, 1.1, 1.2, 1.6, 1.0, 1.0, 1.0])

#Filters
#-----------------------------------------------------------
with filter_cols[0]:
    min_d, max_d = va["day"].min(), va["day"].max()
    date_range = st.date_input("Date", value=(min_d, max_d), min_value=min_d, max_value=max_d)
with filter_cols[1]:
    genres = sorted(va["genre"].unique().tolist())
    sel_genres = st.multiselect("Genre", genres, default=genres)
with filter_cols[2]:
    regions = ["All"] + sorted(va["region"].unique().tolist())
    sel_region = st.selectbox("Region", regions)
with filter_cols[3]:
    st.write("")  # spacing
    st.download_button(
        "Download Viewer Data",
        data=va.to_csv(index=False).encode(),
        file_name="viewing_activity.csv",
        mime="text/csv"
    )

# Filtered data
start_d, end_d = (date_range if isinstance(date_range, tuple) else (date_range, date_range))
mask = (va["day"] >= start_d) & (va["day"] <= end_d)
if sel_genres:
    mask &= va["genre"].isin(sel_genres)
if sel_region != "All":
    mask &= (va["region"] == sel_region)
f = va.loc[mask].copy()

if f.empty:
    st.warning("No data for selected filters.")
    st.stop()

#KPI Cards
k1, k2, k3 = filter_cols[4], filter_cols[5], filter_cols[6]
with k1:
    active_users = f["user_id"].nunique()
    st.metric("Subscribers", f"{active_users:,}")
with k2:
    avg_watch = f["watch_time_minutes"].mean()
    st.metric("Avg. Watch", f"{avg_watch/60:.1f}h")
with k3:
    top_genre = f.groupby("genre")["watch_time_minutes"].sum().idxmax()
    st.metric("Top Genre", top_genre)

st.markdown("---")

#Middle Row – Watch Hours Chart + Map
#-----------------------------------------------------------
left, right = st.columns([1.4, 1.0])

with left:
    st.subheader("Watch Hours")
    daily = (
        f.groupby("day")["watch_time_minutes"]
        .sum()
        .reset_index()
        .rename(columns={"watch_time_minutes": "watch_minutes"})
    )
    daily["watch_hours"] = daily["watch_minutes"] / 60.0

    chart = (
        alt.Chart(daily, title="Total watch hours by day")
        .mark_area(line=True)
        .encode(
            x=alt.X("day:T", title="Date"),
            y=alt.Y("watch_hours:Q", title="Watch hours"),
            tooltip=[
                alt.Tooltip("day:T", title="Date"),
                alt.Tooltip("watch_hours:Q", title="Watch hours", format=".2f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart.interactive(), use_container_width=True)
    st.caption("Alt text: Area chart of total watch hours per day; zoom/hover to explore.")

with right:
    st.subheader("World Map (Watch Hours)")
    #approximate region coordinates
    region_coords = {
        "North America": (39.8, -98.6),
        "Europe": (54.5, 15.3),
        "Asia": (34.0, 100.0),
        "South America": (-15.6, -56.1),
        "Africa": (1.9, 17.3),
        "Oceania": (-25.3, 133.8),
    }
    region_agg = (
        f.groupby("region")["watch_time_minutes"]
        .sum()
        .reset_index()
        .rename(columns={"watch_time_minutes": "watch_minutes"})
    )
    region_agg["lat"] = region_agg["region"].map(lambda r: region_coords.get(r, (0, 0))[0])
    region_agg["lon"] = region_agg["region"].map(lambda r: region_coords.get(r, (0, 0))[1])
    region_agg["watch_hours"] = region_agg["watch_minutes"] / 60.0

    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.2)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=region_agg,
        get_position="[lon, lat]",
        get_radius="watch_hours * 200000",
        radius_min_pixels=4,
        pickable=True,
    )
    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=view_state,
            layers=[layer],
            tooltip={"text": "{region}\nWatch hours: {watch_hours}"},
        )
    )
    st.caption("Alt text: Bubble map showing total watch hours per region.")

st.markdown("---")

#Bottom Row – Summary + Top10 Table
#-----------------------------------------------------------
left2, right2 = st.columns([1.4, 1.0])

with left2:
    st.subheader("Hot genres & times summary")
    gsum = f.groupby("genre")["watch_time_minutes"].sum().sort_values(ascending=False)
    top_genre_name = gsum.index[0]
    top_genre_min = int(gsum.iloc[0])
    peak_day = daily.loc[daily["watch_minutes"].idxmax(), "day"]
    st.markdown(
        f"- **Top genre:** **{top_genre_name}** with **{top_genre_min} minutes** total\n"
        f"- **Peak day:** **{peak_day}**\n"
        f"- **Active subscribers:** **{active_users}**"
    )
    st.caption("Alt text: Text panel summarizing the top genre, peak day, and user count.")

with right2:
    st.subheader("Top10 Show Table")
    #Placeholder grouping
    top10 = (
        f.groupby(["genre", "user_id"])["watch_time_minutes"]
        .sum()
        .reset_index()
        .sort_values("watch_time_minutes", ascending=False)
        .head(10)
        .rename(columns={"watch_time_minutes": "watch_minutes"})
    )
    st.dataframe(top10, use_container_width=True)
    st.caption("Alt text: Table of the top items by watch minutes for the selected filters.")

#Accessibility Notes
#-----------------------------------------------------------
with st.expander("Accessibility & labeling notes"):
    st.write(
        "- All charts and metrics include clear titles and captions.\n"
        "- Captions double as alt text for accessibility.\n"
        "- Color is never the only encoding; text labels show values.\n"
        "- Layout matches the submitted wireframe (filters → KPIs → visuals → summary)."
    )
