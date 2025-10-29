import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from pathlib import Path

#Page setup
#-----------------------------------------------------------
st.set_page_config(page_title="Streaming Platform Insights", page_icon="📺", layout="wide")
st.title("Streaming Platform Insights")
st.caption(
    "Alt text: Dashboard with top-row filters and KPI cards, a watch-hours chart, "
    "a regional map of watch time, a summary text panel, and a Top-10 table."
)

BASE_DIR = Path(__file__).resolve().parent

#Synthetic dataset generators
#-----------------------------------------------------------
def make_demo_data(n_users=500, days=60, seed=7):
    """Prebuilt rich dataset spanning the last `days` days."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    dates = pd.date_range(start, periods=days, freq="D")
    regions = ["North America","Europe","Asia","South America","Africa","Oceania"]
    devices = ["TV","Mobile","Desktop"]
    genres  = ["Drama","Comedy","Action","Documentary","Sci-Fi","Reality"]

    rows = []
    for d in dates:
        active_count = rng.integers(n_users//6, n_users//3)
        actives = rng.choice(np.arange(1, n_users+1), size=active_count, replace=False)
        for u in actives:
            region = rng.choice(regions, p=[0.32,0.25,0.25,0.07,0.06,0.05])
            device = rng.choice(devices, p=[0.55,0.30,0.15])
            genre  = rng.choice(genres,  p=[0.26,0.22,0.20,0.12,0.12,0.08])
            wt     = max(5, int(rng.normal(46, 22)))  # minutes
            rows.append((d, f"u{u:04d}", region, device, genre, wt))
    df = pd.DataFrame(rows, columns=["date","user_id","region","device","genre","watch_time_minutes"])
    return df

def make_demo_data_for_range(start_d, end_d, n_users=800, seed=11):
    """Generate *just-in-time* synthetic rows for any chosen date range."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(pd.to_datetime(start_d), pd.to_datetime(end_d), freq="D")
    regions = ["North America","Europe","Asia","South America","Africa","Oceania"]
    devices = ["TV","Mobile","Desktop"]
    genres  = ["Drama","Comedy","Action","Documentary","Sci-Fi","Reality"]

    rows = []
    for d in dates:
        dow   = d.dayofweek
        base  = 40 + (10 if dow in (4,5) else 0)  #Fri/Sat bump
        #simulate “sessions” per day
        for _ in range(rng.integers(250, 500)):
            region = rng.choice(regions, p=[0.32,0.25,0.25,0.07,0.06,0.05])
            device = rng.choice(devices, p=[0.55,0.30,0.15])
            genre  = rng.choice(genres,  p=[0.26,0.22,0.20,0.12,0.12,0.08])
            wt     = max(5, int(rng.normal(base, 18)))
            uid    = f"u{rng.integers(1, n_users+1):04d}"
            rows.append((d.normalize(), uid, region, device, genre, wt))
    df = pd.DataFrame(rows, columns=["date","user_id","region","device","genre","watch_time_minutes"])
    return df

#Data loading (local CSVs)
#-----------------------------------------------------------
@st.cache_data
def load_local():
    va_path = BASE_DIR / "data" / "viewing_activity.csv"
    cat_path = BASE_DIR / "data" / "content_catalog.csv"
    va = pd.read_csv(va_path, parse_dates=["date"])
    cat = pd.read_csv(cat_path)
    return va, cat, str(va_path), str(cat_path)

#Sidebar controls
#-----------------------------------------------------------
with st.sidebar:
    st.subheader("Data Source")
    use_demo = st.toggle("Use demo dataset (richer & diverse)", value=False)
    if use_demo:
        demo_days  = st.slider("Demo days", 14, 120, 60, step=7)
        demo_users = st.slider("Demo users", 100, 2000, 600, step=50)
        demo_seed  = st.number_input("Demo seed", 0, 9999, 7, step=1)
    map_style = st.selectbox("Map style", ["light", "dark", "road", "satellite"], index=0)
    color_intensity = st.slider("Map color intensity", 0.5, 3.0, 1.4, step=0.1)

if use_demo:
    va = make_demo_data(n_users=demo_users, days=demo_days, seed=int(demo_seed))
    catalog = pd.DataFrame({"genre":["Drama","Comedy","Action","Documentary","Sci-Fi","Reality"],
                            "release_year":[2021,2022,2020,2023,2019,2018],
                            "rating":[4.2,4.0,3.8,4.5,4.1,3.7]})
    data_source_label = f"demo({demo_days}d, {demo_users} users, seed={demo_seed})"
else:
    va, catalog, va_path, cat_path = load_local()
    data_source_label = f"local CSVs → {va_path.split('/')[-2:]}, {cat_path.split('/')[-2:]}"

st.caption(f"Loaded data: {data_source_label}")

#Prep
#-----------------------------------------------------------
va["day"] = pd.to_datetime(va["date"]).dt.date

#Filters + KPIs
#-----------------------------------------------------------
filters = st.columns([1.2, 1.2, 1.2, 1.5, 1, 1, 1])

with filters[0]:
    #allow a larger UI window than the data actually contains
    data_min, data_max = va["day"].min(), va["day"].max()
    ui_min = (pd.to_datetime(data_min) - pd.Timedelta(days=90)).date()
    ui_max = (pd.to_datetime(data_max) + pd.Timedelta(days=90)).date()
    date_range = st.date_input("Date", (data_min, data_max), min_value=ui_min, max_value=ui_max)

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

val = date_range
if isinstance(val, (list, tuple)):
    if len(val) >= 2 and val[0] and val[1]:
        start_d, end_d = val[0], val[1]
    elif len(val) >= 1 and val[0]:
        start_d = end_d = val[0]
    else:
        start_d = end_d = data_min
else:
    start_d = end_d = val

#Filter CSV data first
mask = (va["day"] >= start_d) & (va["day"] <= end_d)
if sel_genres:
    mask &= va["genre"].isin(sel_genres)
if sel_region != "All":
    mask &= (va["region"] == sel_region)
f = va.loc[mask].copy()

if f.empty:
    f = make_demo_data_for_range(start_d, end_d, n_users=800, seed=11)
    # Respect current genre/region filters if possible
    if sel_genres:
        f = f[f["genre"].isin(sel_genres)]
    if sel_region != "All":
        f = f[f["region"] == sel_region]
    st.info("No rows in the CSV for this date range; generated synthetic data for the selected dates.")

#KPIs
k1, k2, k3 = filters[4], filters[5], filters[6]
with k1:
    st.metric("Subscribers", f["user_id"].nunique())
with k2:
    st.metric("Avg. Watch", f"{f['watch_time_minutes'].mean()/60:.1f} h")
with k3:
    st.metric("Top Genre", f.groupby("genre")["watch_time_minutes"].sum().idxmax())

st.markdown("---")

#Charts: line + colored map
#-----------------------------------------------------------
left, right = st.columns([1.4, 1.0])

with left:
    st.subheader("Watch Hours Over Time")
    daily = f.groupby(pd.to_datetime(f["date"]).dt.date)["watch_time_minutes"].sum().reset_index(name="watch_time_minutes")
    daily["watch_hours"] = daily["watch_time_minutes"] / 60
    chart = (
        alt.Chart(daily, title="Total watch hours by day")
        .mark_area(line=True)
        .encode(
            x=alt.X("index:T", title="Date").transform_calculate(index="toDate(datum['index'])")  # ensure temporal type
            if "index" in daily.columns else alt.X("index:T"),
            y=alt.Y("watch_hours:Q", title="Watch Hours"),
            tooltip=[alt.Tooltip("index:T", title="Date") if "index" in daily.columns else alt.Tooltip("index:T", title="Date"),
                     alt.Tooltip("watch_hours:Q", title="Watch hours", format=".2f")]
        )
        .properties(height=320)
    )
    daily = daily.rename(columns={"index":"day"}) if "index" in daily.columns else daily
    chart = (
        alt.Chart(daily, title="Total watch hours by day")
        .mark_area(line=True)
        .encode(
            x=alt.X("day:T", title="Date"),
            y=alt.Y("watch_hours:Q", title="Watch Hours"),
            tooltip=[alt.Tooltip("day:T", title="Date"),
                     alt.Tooltip("watch_hours:Q", title="Watch hours", format=".2f")]
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Alt text: Area chart showing total watch hours per day.")

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

    #Color bubbles
    max_h = float(region_agg["watch_hours"].max() or 1)
    scale = color_intensity
    region_agg["r"] = ((region_agg["watch_hours"]/max_h) * 40 * scale + 30).clip(0, 255)
    region_agg["g"] = ((region_agg["watch_hours"]/max_h) * 120 * scale + 70).clip(0, 255)
    region_agg["b"] = 210.0

    deck = pdk.Deck(
        map_style=map_style, # "light" | "dark" | "road" | "satellite"
        initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1.2),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=region_agg,
                get_position="[lon, lat]",
                get_radius="watch_hours * 200000",
                get_fill_color="[r, g, b, 180]",
                get_line_color=[25,25,25],
                radius_min_pixels=4,
                pickable=True,
            )
        ],
        tooltip={"text": "{region}\nWatch hours: {watch_hours:.2f}"},
    )
    st.pydeck_chart(deck)
    st.caption("Alt text: Colored bubble map sized and colored by watch hours; style and intensity adjustable.")

st.markdown("---")

#Summary + table
#-----------------------------------------------------------
left2, right2 = st.columns([1.4, 1.0])

with left2:
    st.subheader("Hot Genres & Times Summary")
    gsum = f.groupby("genre")["watch_time_minutes"].sum().sort_values(ascending=False)
    top_genre = gsum.index[0]
    top_genre_min = int(gsum.iloc[0])
    peak_day = pd.to_datetime(f["date"]).dt.date.value_counts().idxmax()
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
        "- Clear titles, labels, and units; captions are alt text.\n"
        "- Values shown in labels/tooltips (not color-only encoding).\n"
        "- If the CSV has no rows for the selected dates, synthetic data is generated automatically."
    )
