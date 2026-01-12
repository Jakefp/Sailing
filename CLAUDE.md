# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based GPS data analyzer for sailing activities. It allows users to upload GPS track files from multiple sailors, visualize their tracks on maps, and generate polar diagrams showing speed vs. heading.

## Running the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run the Streamlit app
streamlit run version_1.py
```

## Architecture

### Main Application (`version_1.py`)
- Single-file Streamlit application handling all UI and data processing
- Supports multiple file formats: `.fit.gz`, `.fit`, `.gpx`, `.csv`
- Core features:
  - Multi-sailor track comparison
  - Interactive map with configurable wind direction and ladder lines
  - Polar diagram with adjustable smoothing
  - Animated playback of tracks with configurable tail length and speed

### Data Pipeline
1. **File parsing**: `process_fit_gz()`, `process_gpx()`, `process_csv()` convert uploaded files to DataFrames with `lat`, `lon`, `time` columns
2. **Kinematics calculation**: `add_kinematics()` computes `speed`, `speed_knots`, `heading`, `distance_m` from GPS coordinates
3. **Time range filtering**: Multiple time ranges can be selected and stitched together for analysis
4. **Visualization**: Plotly-based map traces and polar diagrams

### Key Functions
- `compute_bearing(p1, p2)`: Calculates heading between two GPS points
- `generate_ladder_lines()`: Creates perpendicular reference lines for wind direction visualization
- `add_stitched_time()`: Concatenates multiple time ranges into a continuous playback timeline

### Utility Module (`utils.py`)
Contains shared GPS processing functions similar to main app but designed for standalone CSV processing.

## Dependencies

Key libraries: `streamlit`, `pandas`, `fitparse`, `geopy`, `numpy`, `plotly`

## Important Notes

- Timestamps are converted to Australia/Sydney timezone
- GPS coordinates use semicircle format from FIT files (converted via `* (180 / 2**31)`)
- Map uses Mapbox with a configured access token in the code
