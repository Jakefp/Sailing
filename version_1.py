import streamlit as st
import pandas as pd
import gzip
from fitparse import FitFile
from io import BytesIO
from geopy.distance import geodesic
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import itertools
import os
import xml.etree.ElementTree as ET  # for GPX parsing


# Set Mapbox token
os.environ["MAPBOX_ACCESS_TOKEN"] = "pk.eyJ1IjoiamFrZWZwLXNhaWxpbmciLCJhIjoiY21ha2theHZzMTN2NTJqcHoxeXRwMmFnOCJ9.VLxCKga5g5XjCZNos-kqGw"

st.set_page_config(layout="wide")
st.title("Multi-Sailor GPS Analyzer")

# ---------- Helper Functions ----------
def compute_bearing(p1, p2):
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def add_kinematics(df):
    """
    Expects a DataFrame with columns: 'lat', 'lon', 'time' (tz-aware).
    Returns a new DataFrame with speed, speed_knots, heading, distance_m.
    """
    # Ensure chronological order
    df = df.sort_values("time").reset_index(drop=True)

    lats = df["lat"].values
    lons = df["lon"].values
    times = df["time"].values

    speeds = []
    headings = []
    distances = [0]  # pairwise segment distances

    for i in range(1, len(df)):
        p1 = (lats[i - 1], lons[i - 1])
        p2 = (lats[i], lons[i])

        dist = geodesic(p1, p2).meters
        duration = (times[i] - times[i - 1]) / np.timedelta64(1, "s")

        speeds.append(dist / duration if duration > 0 else 0)
        headings.append(compute_bearing(p1, p2))
        distances.append(dist)

    # Drop first row to match original behaviour
    df = df.iloc[1:].copy()
    df["speed"] = speeds
    df["speed_knots"] = df["speed"] * 1.94384
    df["heading"] = headings
    df["distance_m"] = distances[1:]
    return df

def generate_ladder_lines(lat_series, lon_series, wind_from_deg, spacing_m=50.0, extend_factor=1.1):
    """
    Creates ladder lines (crosswind-parallel lines) spaced along the wind axis.

    wind_from_deg: meteorological "from" direction in degrees (0=N, 90=E).
    spacing_m: distance between ladder lines along the wind axis (meters).
    extend_factor: how much longer than the data extent the lines should be.
    Returns: list of dicts with 'lats' and 'lons'
    """
    if len(lat_series) == 0:
        return []

    # Use a local tangent-plane approximation for bounding / placement
    lat0 = float(lat_series.mean())
    lon0 = float(lon_series.mean())

    lat = lat_series.to_numpy(dtype=float)
    lon = lon_series.to_numpy(dtype=float)

    # meters per degree
    m_per_deg_lat = 110540.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat0))

    # Convert to local XY meters (east=x, north=y)
    x = (lon - lon0) * m_per_deg_lon
    y = (lat - lat0) * m_per_deg_lat

    # Wind axis direction "to" (so offsets march downwind/upwind consistently)
    wind_to_deg = (wind_from_deg + 180) % 360

    # Unit vectors for wind axis (a) and perpendicular/crosswind axis (p)
    a = np.array([np.sin(np.radians(wind_to_deg)), np.cos(np.radians(wind_to_deg))])  # along-wind
    p_deg = (wind_to_deg + 90) % 360
    p = np.array([np.sin(np.radians(p_deg)), np.cos(np.radians(p_deg))])              # perpendicular to wind

    # Project points onto wind axis to find min/max extent
    t = x * a[0] + y * a[1]  # along-wind coordinate (meters)
    t_min, t_max = float(np.min(t)), float(np.max(t))

    if spacing_m <= 0:
        return []

    # Determine ladder offsets (meters) covering the extent
    start = np.floor(t_min / spacing_m) * spacing_m
    end = np.ceil(t_max / spacing_m) * spacing_m
    offsets = np.arange(start, end + spacing_m, spacing_m)

    # Determine crosswind span for line length
    s = x * p[0] + y * p[1]  # crosswind coordinate
    half_len = 0.5 * (float(np.max(s)) - float(np.min(s))) * extend_factor
    half_len = max(half_len, spacing_m * 2)  # ensure visible even if tight track

    lines = []
    center_point = (lat0, lon0)

    for off in offsets:
        # Line center point in XY meters relative to origin
        cx, cy = off * a[0], off * a[1]

        # Convert that center back to lat/lon approx
        clat = lat0 + (cy / m_per_deg_lat)
        clon = lon0 + (cx / m_per_deg_lon)

        # Compute line endpoints using geodesic destination for accuracy
        # Endpoints go along p_deg and opposite
        p1 = geodesic(meters=half_len).destination((clat, clon), p_deg)
        p2 = geodesic(meters=half_len).destination((clat, clon), (p_deg + 180) % 360)

        lines.append({
            "lats": [p1.latitude, p2.latitude],
            "lons": [p1.longitude, p2.longitude]
        })

    return lines

def add_stitched_time(df, time_ranges):
    """
    Adds a 't_play_s' column: seconds along a gapless timeline created by concatenating time_ranges.
    df must contain tz-aware 'time' column.
    Only rows within time_ranges should be passed in (or they'll become NaN).
    """
    # Ensure ranges are sorted by start time
    ranges = sorted(time_ranges, key=lambda x: x[0])

    # Precompute cumulative offsets
    offsets = []
    cum = 0.0
    for start_t, end_t in ranges:
        offsets.append((start_t, end_t, cum))
        cum += (end_t - start_t).total_seconds()

    t_play = np.full(len(df), np.nan, dtype=float)
    times = df["time"].to_numpy()

    for (start_t, end_t, base) in offsets:
        mask = (df["time"] >= start_t) & (df["time"] <= end_t)
        # seconds from start of this segment + cumulative base
        t_play[mask.to_numpy()] = base + (df.loc[mask, "time"] - start_t).dt.total_seconds().to_numpy()

    out = df.copy()
    out["t_play_s"] = t_play
    out = out.dropna(subset=["t_play_s"])
    return out

# ----------- Uploading functions -------------

def process_fit_gz(file_obj):
    with gzip.open(file_obj, 'rb') as f:
        fit_data = f.read()
    fitfile = FitFile(BytesIO(fit_data))
    records = []
    for record in fitfile.get_messages("record"):
        data = {field.name: field.value for field in record}
        if "position_lat" in data and "position_long" in data:
            records.append(data)
    df = pd.DataFrame(records)

    df["lat"] = df["position_lat"] * (180 / 2**31)
    df["lon"] = df["position_long"] * (180 / 2**31)

    # Drop invalid rows
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]

    # Fit timestamps are usually UTC
    df["time"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC").dt.tz_convert("Australia/Sydney")

    # Compute speed, heading, and distance
    df = add_kinematics(df)
    return df

def process_gpx(file_obj):
    # Read and parse XML
    content = file_obj.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")

    root = ET.fromstring(content)

    points = []
    for trkpt in root.findall(".//{*}trkpt"):
        lat = trkpt.attrib.get("lat")
        lon = trkpt.attrib.get("lon")
        time_el = trkpt.find("{*}time") or trkpt.find(".//{*}time")

        if lat is None or lon is None or time_el is None:
            continue

        points.append({
            "lat": float(lat),
            "lon": float(lon),
            "timestamp": time_el.text
        })

    df = pd.DataFrame(points)
    if df.empty:
        raise ValueError("No trackpoints found in GPX file")

    # GPX times are typically UTC (with 'Z')
    df["time"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Australia/Sydney")

    # Basic sanity filter on positions
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]

    df = add_kinematics(df)
    return df

def process_csv(file_obj):
    df_raw = pd.read_csv(file_obj)

    if df_raw.empty:
        raise ValueError("CSV file is empty")

    # Try to detect column names (case-insensitive)
    cols_lower = {c.lower(): c for c in df_raw.columns}

    lat_col = next((cols_lower[c] for c in ["lat", "latitude", "position_lat"]), None)
    lon_col = next((cols_lower[c] for c in ["lon", "lng", "longitude", "position_long"]), None)
    time_col = next((cols_lower[c] for c in ["time", "timestamp", "date_time", "datetime"]), None)

    if lat_col is None or lon_col is None or time_col is None:
        raise ValueError(
            "CSV must contain latitude, longitude, and time columns "
            "(e.g. 'lat'/'latitude', 'lon'/'longitude', 'time'/'timestamp')."
        )

    df = df_raw.copy()
    df["lat"] = df[lat_col].astype(float)
    df["lon"] = df[lon_col].astype(float)

    # Parse time; if tz-naive, assume Australia/Sydney; if tz-aware, convert.
    times = pd.to_datetime(df[time_col], errors="coerce")
    if times.dt.tz is None:
        times = times.dt.tz_localize("Australia/Sydney")
    else:
        times = times.dt.tz_convert("Australia/Sydney")
    df["time"] = times

    # Drop bad rows
    df = df.dropna(subset=["lat", "lon", "time"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]

    df = add_kinematics(df)
    return df


# ---------- Upload & Process ----------
uploaded_files = st.file_uploader(
    "Upload one or more GPS files (.fit.gz, .fit, .gpx, .csv)",
    type=["gz", "fit", "gpx", "csv"],
    accept_multiple_files=True
)

sailor_data = []
colors = itertools.cycle(px.colors.qualitative.Set2)

if uploaded_files:
    st.subheader("Sailor Assignments")
    for file in uploaded_files:
        filename = file.name
        sailor_name = st.text_input(
            f"Name for {filename}",
            value=filename.split('.')[0],
            key=filename
        )

        ext = filename.lower()

        try:
            if ext.endswith(".fit.gz") or ext.endswith(".fit"):
                df = process_fit_gz(file)
            elif ext.endswith(".gpx"):
                df = process_gpx(file)
            elif ext.endswith(".csv"):
                df = process_csv(file)
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            df["sailor"] = sailor_name.strip()
            color = next(colors)
            sailor_data.append({"name": sailor_name, "df": df, "color": color})

        except Exception as e:
            st.error(f"Failed to parse {filename}: {e}")

# ---------- If data loaded ----------
if sailor_data:
    all_times = pd.concat([s["df"]["time"] for s in sailor_data])
    all_lats = pd.concat([s["df"]["lat"] for s in sailor_data])
    all_lons = pd.concat([s["df"]["lon"] for s in sailor_data])
    start_time, end_time = all_times.min(), all_times.max()

        # ---------- Time range selection ----------
    st.markdown("### Time ranges")
    ranges_count = st.number_input(
        "Number of time ranges",
        min_value=1,
        max_value=4,
        value=1,
        step=1,
        format="%d",
    )

    time_ranges = []
    for i in range(ranges_count):
        start_i, end_i = st.slider(
            f"Time range {i + 1}",
            min_value=start_time.to_pydatetime(),
            max_value=end_time.to_pydatetime(),
            value=(start_time.to_pydatetime(), end_time.to_pydatetime()),
            format="MM-DD HH:mm:ss",
            step=timedelta(minutes=1),
            key=f"time_range_{i + 1}",
        )
        time_ranges.append((start_i, end_i))

        # ---------- Layout: Map and Polar ----------
    col1, col2, col3 = st.columns([3, 3, 3])
    with col1:
        st.subheader("Track Map")
        map_bearing = st.number_input(
        "Wind Direction (degrees)",
        min_value=0,
        max_value=359,
        value=0,
        step=1,
        format="%d",
        help="Rotates the map clockwise. 0 = North-up. 90 = East-up."
        )
        ladder_spacing_m = st.number_input(
        "Ladder line spacing (meters)",
        min_value=5.0,
        max_value=1000.0,
        value=50.0,
        step=5.0,
        format="%.0f"
        )
        show_ladder = st.checkbox("Show ladder lines (perpendicular to wind)", value=False)

    with col2:
        st.subheader("Polar Smoothing")
        smoothing = st.slider(
        "Polar smoothing amount", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="0 = no smoothing, 1 = full smoothing with neighbors"
    )

    with col3:
        st.subheader("Minimum Speed for Polars")
        # 🆕 Minimum boatspeed filter for polar plot
        min_speed_knots = st.number_input(
        "Minimum boatspeed to include in polars (knots)",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=0.01,
        format="%.2f",
        help="Speeds below this value are excluded from the polar averages"
    )

    # ---------- Polar Plot ----------
    polar_fig = go.Figure()
    polar_fig.update_layout(polar=dict(
        angularaxis=dict(
            direction="clockwise",
            rotation=90,
            tickmode="linear",
            tick0=0,
            dtick=30,
            tickvals=[0, 90, 180, 270],
            ticktext=["N", "E", "S", "W"]
        ),
        radialaxis=dict(title="Speed (knots)")
    ))

    # ---------- Map Plot ----------
    track_fig = go.Figure()
    track_fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=all_lats.mean(), lon=all_lons.mean()),
            zoom=15 if all_lats.std() + all_lons.std() < 0.01 else 13 if all_lats.std() + all_lons.std() < 0.05 else 12,
            bearing=map_bearing
        ),
        height=500,
        margin=dict(r=0, l=0, t=0, b=0)
    )

    if show_ladder:
        ladder_lines = generate_ladder_lines(all_lats, all_lons, map_bearing, spacing_m=ladder_spacing_m)

        for ln in ladder_lines:
            track_fig.add_trace(go.Scattermapbox(
                lat=ln["lats"],
                lon=ln["lons"],
                mode="lines",
                line=dict(width=0.8,color='lightgrey'),      # keep subtle
                opacity=0.9,
                hoverinfo="skip",
                showlegend=False
            ))


    # ---------- Process and Plot Each Sailor ----------
    summary_rows = []
    for sailor in sailor_data:
        df = sailor["df"]
        name = sailor["name"]
        color = sailor["color"]

        # ---------- Apply one or two time ranges ----------
        time_mask = pd.Series(False, index=df.index)
        for start_t, end_t in time_ranges:
            time_mask |= (df["time"] >= start_t) & (df["time"] <= end_t)

        filtered = df[time_mask].copy()
        if filtered.empty:
            continue

        # 🆕 Boatspeed filter for POLAR ONLY
        polar_filtered = filtered[filtered["speed_knots"] >= min_speed_knots].copy()

        # ---------- Polar (only if we have data after speed filter) ----------
        if not polar_filtered.empty:
            polar_filtered.loc[:, "dir_bin"] = (polar_filtered["heading"] // 5) * 5
            avg = polar_filtered.groupby("dir_bin")["speed_knots"].mean().reset_index()

            # Fill all 5° bins
            all_bins = pd.DataFrame({"dir_bin": np.arange(0, 360, 5)})
            avg = all_bins.merge(avg, on="dir_bin", how="left").fillna(0)

            # Smoothing weights
            center_weight = 1 - smoothing
            neighbor_weight = smoothing / 2

            # Apply weighted smoothing
            smoothed = []
            for i in range(len(avg)):
                prev = avg.iloc[i - 1]["speed_knots"] if i > 0 else avg.iloc[-1]["speed_knots"]
                curr = avg.iloc[i]["speed_knots"]
                next_ = avg.iloc[(i + 1) % len(avg)]["speed_knots"]
                smooth_val = neighbor_weight * prev + center_weight * curr + neighbor_weight * next_
                smoothed.append(smooth_val)

            avg["smoothed_speed"] = smoothed

        polar_fig.add_trace(go.Scatterpolar(
            r=avg["smoothed_speed"],
            theta=avg["dir_bin"],
            name=name,
            mode="lines+markers",
            line=dict(color=color)
        ))

        # Track
        track_fig.add_trace(go.Scattermapbox(
        lat=filtered["lat"],
        lon=filtered["lon"],
        mode="lines",
        name=name,
        line=dict(color=color),
        text=filtered["time"].dt.strftime("%H:%M:%S"),
        hoverinfo="text"
    ))

        # Summary values
        total_distance = filtered["distance_m"].sum() / 1000
        avg_speed = filtered["speed_knots"].mean()
        summary_rows.append({
            "Sailor": name,
            "Distance (km)": round(total_distance, 2),
            "Avg Speed (kn)": round(avg_speed, 2),
            "Color": color
        })

    summary_df = pd.DataFrame(summary_rows)

    # ---------- Layout: Map and Polar ----------
    col1, col2 = st.columns([2, 2])
    with col1:
        st.subheader("Track Map")
        st.plotly_chart(track_fig, use_container_width=True)

    with col2:
        st.subheader("Polar Diagram")
        st.plotly_chart(polar_fig, use_container_width=True)


    st.markdown("### Playback")

    enable_playback = st.checkbox("Enable playback", value=False)

    col1, col2, col3 = st.columns([3, 3, 3])
    with col1:
        tail_seconds = st.number_input(
        "Tail length (seconds)",
        min_value=0,
        max_value=600,
        value=30,
        step=5,
        format="%d"
    )
    with col2:
        playback_speed = st.selectbox(
        "Playback speed",
        options=[0.5, 1, 2, 4, 8],
        index=1
    )
    with col3:
        frame_step_s = st.number_input(
        "Playback timestep (seconds per frame)",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        format="%d",
        help="Bigger timestep = fewer frames = faster app."
    )
        
    if enable_playback:
        # ---- Build per-sailor filtered data using your existing time_mask logic ----
        playback_sailors = []
        for s in sailor_data:
            df = s["df"].copy()

            # Apply your multi-range time mask (same logic you already use)
            time_mask = pd.Series(False, index=df.index)
            for start_t, end_t in time_ranges:
                time_mask |= (df["time"] >= start_t) & (df["time"] <= end_t)
            df = df[time_mask].copy()
            if df.empty:
                continue

            # Add stitched timeline
            df = add_stitched_time(df, time_ranges)

            # OPTIONAL PERFORMANCE: resample to 1 point per frame_step_s (approx)
            # We'll bin by playback time seconds
            df["t_bin"] = (df["t_play_s"] // frame_step_s) * frame_step_s
            df = df.sort_values("t_play_s").groupby("t_bin", as_index=False).last()

            playback_sailors.append({
                "name": s["name"],
                "color": s["color"],
                "df": df.sort_values("t_play_s").reset_index(drop=True)
            })

        if len(playback_sailors) == 0:
            st.info("No playback data found in the selected time ranges.")
        else:
            # Determine global playback span
            t_min = min(ps["df"]["t_play_s"].min() for ps in playback_sailors)
            t_max = max(ps["df"]["t_play_s"].max() for ps in playback_sailors)

            # Frame times
            frame_times = np.arange(
                np.floor(t_min / frame_step_s) * frame_step_s,
                np.ceil(t_max / frame_step_s) * frame_step_s + frame_step_s,
                frame_step_s
            )

            # Base figure
            playback_fig = go.Figure()
            playback_fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(
                    center=dict(lat=all_lats.mean(), lon=all_lons.mean()),
                    zoom=15 if all_lats.std() + all_lons.std() < 0.01 else 13 if all_lats.std() + all_lons.std() < 0.05 else 12,
                    bearing=map_bearing
                ),
                height=550,
                margin=dict(r=0, l=0, t=10, b=0),
                showlegend=True,
                legend=dict(orientation="h")
            )

            # Initialize traces (2 per sailor: tail line + current marker)
            for ps in playback_sailors:
                name = ps["name"]
                color = ps["color"]

                playback_fig.add_trace(go.Scattermapbox(
                    lat=[], lon=[],
                    mode="lines",
                    name=f"{name} tail",
                    line=dict(color=color, width=2),
                    showlegend=False
                ))
                playback_fig.add_trace(go.Scattermapbox(
                    lat=[], lon=[],
                    mode="markers",
                    name=name,
                    marker=dict(size=14, color=color)
                ))

            # Pre-extract arrays for fast slicing
            pre = []
            for ps in playback_sailors:
                d = ps["df"]
                pre.append({
                    "t": d["t_play_s"].to_numpy(),
                    "lat": d["lat"].to_numpy(),
                    "lon": d["lon"].to_numpy(),
                    "time": d["time"].to_numpy()
                })

            # Build mapping from t_play_s to actual timestamp for slider labels
            # Use the first sailor's data as reference for timestamps
            ref_df = playback_sailors[0]["df"]
            t_to_time = dict(zip(ref_df["t_play_s"], ref_df["time"]))

            frames = []
            for t_now in frame_times:
                frame_data = []
                for idx, ps in enumerate(playback_sailors):
                    arr = pre[idx]
                    t_arr = arr["t"]

                    # tail window
                    t0 = t_now - tail_seconds
                    i0 = np.searchsorted(t_arr, t0, side="left")
                    i1 = np.searchsorted(t_arr, t_now, side="right")

                    tail_lat = arr["lat"][i0:i1].tolist()
                    tail_lon = arr["lon"][i0:i1].tolist()

                    # current point = last point within window
                    if i1 > 0:
                        cur_i = min(i1 - 1, len(t_arr) - 1)
                        cur_lat = [float(arr["lat"][cur_i])]
                        cur_lon = [float(arr["lon"][cur_i])]
                    else:
                        cur_lat, cur_lon = [], []

                    # Add in same order as traces: tail then marker
                    frame_data.append(go.Scattermapbox(lat=tail_lat, lon=tail_lon))
                    frame_data.append(go.Scattermapbox(lat=cur_lat, lon=cur_lon))

                frames.append(go.Frame(name=str(int(t_now)), data=frame_data))

            playback_fig.frames = frames

            # Control animation speed: Plotly frame duration in ms
            # "normal speed" means 1 playback second per real second, so frame duration = frame_step_s * 1000 / speed
            frame_duration_ms = int((frame_step_s * 1000) / float(playback_speed))

            playback_fig.update_layout(
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    x=0.0, y=1.08,
                    xanchor="left", yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, dict(
                                frame=dict(duration=frame_duration_ms, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate"
                            )]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0)
                            )]
                        )
                    ]
                )],
                sliders=[dict(
                    active=0,
                    x=0.0, y=0.0,
                    xanchor="left", yanchor="top",
                    len=1.0,
                    steps=[dict(
                        label=pd.Timestamp(t_to_time.get(t, t_to_time.get(min(t_to_time.keys(), key=lambda k: abs(k - t))))).strftime("%H:%M:%S") if t_to_time else str(int(t)),
                        method="animate",
                        args=[[str(int(t))], dict(
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                            mode="immediate"
                        )]
                    ) for t in frame_times]
                )]
            )

            st.subheader("Playback")
            st.plotly_chart(playback_fig, use_container_width=True)


# ========== Summary Table ============
st.subheader("Overalls")

summary_rows = []

for sailor in sailor_data:
    df = sailor["df"]
    name = sailor["name"]
    color = sailor["color"]

    # Apply the same time ranges as for the plots
    time_mask = pd.Series(False, index=df.index)
    for start_t, end_t in time_ranges:
        time_mask |= (df["time"] >= start_t) & (df["time"] <= end_t)

    filtered = df[time_mask].copy()
    if filtered.empty:
        continue

    total_distance = filtered["distance_m"].sum() / 1000  # km
    avg_speed = filtered["speed_knots"].mean()

    summary_rows.append({
        "Sailor": name,
        "Distance (km)": round(total_distance, 2),
        "Avg Speed (kn)": round(avg_speed, 2)
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True)