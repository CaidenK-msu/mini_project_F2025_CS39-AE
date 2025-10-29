import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from pathlib import Path
import os

#Page setup
#-----------------------------------------------------------
st.set_page_config(page_title="Streaming Platform Insights", page_icon="📺", layout="wide")
st.title("Streaming Platform Insights")
st.caption(
    "Alt text: Dashboard with top-row filters and KPI cards, a watch-hours chart, "
    "a regional map of watch time, a summary text panel, and a Top-10 table."
)

#Load data safely relative to this file
#-----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    va_path = BASE_DIR / "data" / "viewing_activity.csv"
    cat_path = BASE_DIR / "data" / "content_catalog.csv"

    if not va_path.exists():
        raise FileNotFoundError(f"Missing file: {va_path}")
    if not cat_path.exists():
        raise FileNotFoundError(f"Missing file: {cat_path}")

    va = pd.read_csv(va_path, parse_dates=["date"])
    cat = pd.read_csv(cat_path)
    return va, cat, str(va_path), str(cat_path)

va, catalog, va_path, cat_path = load_data()
st.caption(f"Loaded: `{va_path}` and `{cat_path}`")

#Data prep
#-----------------------------------------------------------
va["day"] = va["date"].dt.date

#Filters + KPIs
#-----------------------------------------------------------
filters = st.columns([1.2, 1.2, 1.2, 1.5, 1, 1, 1])
with filters[0]:
    min_d, max_d = va["day"].min(), va["day"].max()
    date_range = st.date_input("Date", (min_d, max_d), min_value=min_d, max_value=max_d)
with filters[1]:
    genres = sorted(va["genre"].unique())
    sel_genres = st.multiselect("Genre", genres, default=genres)
with filters[2]:
    regions = ["All"] + sorted(va["region"].unique())
    sel_region = st.selectbox("Region", regions)
with filters[3]:
    st.download_button(
        "Download Viewer Data",
        data=va.to_csv(index=False).encode(),
        file_name="viewing_activity.csv",
        mime="text/csv"
    )

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

k1, k2, k3 = filters[4], filters[5], filters[6]
with k1:
    st.metric("Subscribers", f["user_id"].nunique())
with k2:
    st.metric("Avg. Watch", f"{f['watch_time_minutes'].mean() / 60:.1f} h")
with k3:
    st.metric("Top Genre", f.groupby("genre")["watch_time_minutes"].sum().idxmax())

st.markdown("---")

#Charts
#-----------------------------------------------------------
left, right = st.columns([1.4, 1.0])

#Watch Hours Chart
with left:
    st.subheader("Watch Hours Over Time")
    daily = f.groupby("day")["watch_time_minutes"].sum().reset_index()
    daily["watch_hours"] = daily["watch_time_minutes"] / 60
    chart = (
        alt.Chart(daily)
        .mark_area(line=True)
        .encode(
            x=alt.X("day:T", title="Date"),
            y=alt.Y("watch_hours:Q", title="Watch Hours"),
            tooltip=["day:T", "watch_hours:Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Alt text: Area chart showing total watch hours per day.")

#Regional Map
with right:
    st.subheader("Watch Hours by Region")
    region_coords = {
        "North America": (39.8, -98.6),
        "Europe": (54.5, 15.3),
        "Asia": (34.0, 100.0),
        "South America": (-15.6, -56.1),
        "Africa": (1.9, 17.3),
        "Oceania": (-25.3, 133.8),
    }
    region_agg = f.groupby("region")["watch_time_minutes"].sum().reset_index()
    region_agg["lat"] = region_agg["region"].map(lambda r: region_coords.get(r, (0, 0))[0])
    region_agg["lon"] = region_agg["region"].map(lambda r: region_coords.get(r, (0, 0))[1])
    region_agg["watch_hours"] = region_agg["watch_time_minutes"] / 60

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1.2),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=region_agg,
                    get_position="[lon, lat]",
                    get_radius="watch_hours * 200000",
                    radius_min_pixels=4,
                    pickable=True,
                )
            ],
            tooltip={"text": "{region}\nWatch hours: {watch_hours}"},
        )
    )
    st.caption("Alt text: Bubble map showing total watch hours by region.")

st.markdown("---")

#Summary + Table
#-----------------------------------------------------------
left2, right2 = st.columns([1.4, 1.0])

with left2:
    st.subheader("Hot Genres & Times Summary")
    gsum = f.groupby("genre")["watch_time_minutes"].sum().sort_values(ascending=False)
    top_genre = gsum.index[0]
    top_genre_min = int(gsum.iloc[0])
    peak_day = f.groupby("day")["watch_time_minutes"].sum().idxmax()
    st.markdown(
        f"- **Top Genre:** {top_genre} ({top_genre_min} total minutes)\n"
        f"- **Peak Day:** {peak_day}\n"
        f"- **Active Users:** {f['user_id'].nunique()}"
    )

with right2:
    st.subheader("Top 10 User Watch Totals")
    top10 = (
        f.groupby(["user_id", "genre"])["watch_time_minutes"]
        .sum()
        .reset_index()
        .sort_values("watch_time_minutes", ascending=False)
        .head(10)
    )
    st.dataframe(top10, use_container_width=True)
    st.caption("Alt text: Table of top users by total watch time and genre.")

with st.expander("Accessibility & labeling notes"):
    st.write(
        "- Clear titles, labels, and units.\n"
        "- Captions serve as alt text.\n"
        "- Values shown in tooltips to avoid color-only encoding."
    )
