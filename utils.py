import io
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import plotly.express as px
import plotly.graph_objects as go

def process_gps_file(raw_data):
    # Assume CSV-like format: time, lat, lon
    df = pd.read_csv(io.StringIO(raw_data))
    df['time'] = pd.to_datetime(df['time'])
    
    lats = df['lat'].values
    lons = df['lon'].values
    times = df['time'].values

    speeds = []
    headings = []

    for i in range(1, len(df)):
        point1 = (lats[i-1], lons[i-1])
        point2 = (lats[i], lons[i])
        distance = geodesic(point1, point2).meters
        duration = (times[i] - times[i-1]).total_seconds()
        if duration > 0:
            speed = distance / duration
            heading = compute_bearing(point1, point2)
            speeds.append(speed)
            headings.append(heading)
        else:
            speeds.append(0)
            headings.append(0)

    df = df.iloc[1:].copy()
    df['speed'] = speeds
    df['heading'] = headings
    return df

def compute_bearing(p1, p2):
    import math
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def plot_polar(df):
    df['dir_bin'] = (df['heading'] // 10) * 10
    avg_speeds = df.groupby('dir_bin')['speed'].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_speeds['speed'],
        theta=avg_speeds['dir_bin'],
        mode='lines+markers'
    ))
    fig.update_layout(polar=dict(
        angularaxis=dict(direction="clockwise", rotation=90),
        radialaxis=dict(title="Speed (m/s)")
    ))
    return fig

def plot_track_map(df):
    fig = px.line_mapbox(df, lat="lat", lon="lon", hover_name="time",
                         zoom=12, height=500)
    fig.update_layout(mapbox_style="open-street-map")
    return fig