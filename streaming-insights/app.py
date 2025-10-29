import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from datetime import date

st.set_page_config(page_title="Streaming Platform Insights", page_icon="📺", layout="wide")
st.title("Streaming Platform Insights")

st.caption(
    "Alt text: Dashboard with top-row filters and KPI cards, a watch-hours chart, "
    "a regional map of watch time, a summary text panel, and a Top-10 table."
)

#---------- Data ----------
@st.cache_data
def load_data():
    va = pd.read_csv("data/viewing_activity.csv", parse_dates=["date"])
    cat = pd.read_csv("data/content_catalog.csv")
    return va, cat

va, catalog = load_data()
va["day"] = va["date"].dt.date

#---------- Top row: filters (left) + KPI cards ----------
filter_cols = st.columns([1.1, 1.1, 1.2, 1.6, 1.0, 1.0, 1.0])  # control spacing
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
    st.write("")  # vertical nudge
    st.download_button(
        "Download Viewer Data",
        data=va.to_csv(index=False).encode(),
        file_name="viewing_activity.csv",
        mime="text/csv"
    )

#KPIs on the right
k1, k2, k3 = filter_cols[4], filter_cols[5], filter_cols[6]

#---------- Filtering ----------
start_d, end_d = (date_range if isinstance(date_range, tuple) else (date_range, date_range))
mask = (va["day"] >= start_d) & (va["day"] <= end_d)
if sel_genres:
    mask &= va["genre"].isin(sel_genres)
if sel_region != "All":
    mask &= (va["region"] == sel_region)
f = va.loc[mask].copy()

if f.empty:
    st.warning("No data for the selected filters.")
    st.stop()

#KPI metrics
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

#---------- Watch Hours + World Map ----------
left, right = st.columns([1.4, 1.0])

with left:
    st.subheader("Watch Hours")
    daily = (
        f.groupby("day")["watch_time_minutes"]
        .sum()
        .reset_index()
        .rename(columns={"watch_time_minutes": "watch_minutes"})
    )
    #minutes -> hours to match wireframe vibe
    daily["watch_hours"] = daily["watch_minutes"] / 60.0

    line = alt.Chart(daily, title="Total watch hours by day").mark_area(line=True).encode(
        x=alt.X("day:T", title="Date"),
        y=alt.Y("watch_hours:Q", title="Watch hours"),
        tooltip=[alt.Tooltip("day:T", title="Date"),
                 alt.Tooltip("watch_hours:Q", title="Watch hours", format=".2f")]
    ).properties(height=320)
    st.altair_chart(line.interactive(), use_container_width=True)
    st.caption("Alt text: Area chart of total watch hours per day; zoom/hover to explore.")

with right:
    st.subheader("World Map (Watch Hours)")
    #Minimal region centroids
    region_coords = {
        "North America": (39.8, -98.6),
        "Europe": (54.5, 15.3),
        "Asia": (34.0, 100.0),
        "South America": (-15.6, -56.1),
        "Africa": (1.9, 17.3),
        "Oceania": (-25.3, 133.8),
    }
    region_agg = (
        f.groupby("region")["watch_time_minutes"].sum().reset_index()
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
        get_radius="watch_hours * 200000",  # scales bubble by hours
        radius_min_pixels=4,
        pickable=True,
    )
    st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=[layer],
                             tooltip={"text": "{region}\nWatch hours: {watch_hours}"}))
    st.caption("Alt text: Bubble map sizing each region by total watch hours.")

st.markdown("---")

#---------- Bottom row: Summary + Top-10 Table ----------
left2, right2 = st.columns([1.4, 1.0])

with left2:
    st.subheader("Hot genres & times summary")
    gsum = f.groupby("genre")["watch_time_minutes"].sum().sort_values(ascending=False)
    top_genre_name = gsum.index[0]
    top_genre_min = int(gsum.iloc[0])
    peak_day = daily.loc[daily["watch_minutes"].idxmax(), "day"]
    st.markdown(
        f"- **Top genre**: **{top_genre_name}** with **{top_genre_min} minutes** in the selected range.\n"
        f"- **Peak day**: **{peak_day}**.\n"
        f"- **Active subscribers**: **{active_users}**."
    )
    st.caption("Alt text: Text panel summarizing key takeaways from filters.")

with right2:
    st.subheader("Top10 Show Table")
    #With tiny sample we fake “show” as genre & user
    top10 = (
        f.groupby(["genre", "user_id"])["watch_time_minutes"]
        .sum()
        .reset_index()
        .sort_values("watch_time_minutes", ascending=False)
        .head(10)
        .rename(columns={"watch_time_minutes": "watch_minutes"})
    )
    st.dataframe(top10, use_container_width=True)
    st.caption("Alt text: Table of the top items by watch minutes for the current filters.")

#---------- Accessibility notes ----------
with st.expander("Accessibility & labeling notes"):
    st.write(
        "- Charts have descriptive titles, axis labels, and units.\n"
        "- Captions above serve as alt text.\n"
        "- Values are visible via labels/tooltips (not color-only encoding).\n"
        "- Layout mirrors the submitted paper prototype."
    )
