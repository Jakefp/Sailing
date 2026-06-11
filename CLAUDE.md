# CLAUDE.md

Guidance for Claude Code (and other AI assistants) when working in this repo.

## What this is

A single-file **Streamlit** app, `version_1.py`, that analyses sailing GPS tracks.
Users upload `.fit` / `.fit.gz` / `.gpx` / `.csv` files; the app draws each track on
a map and builds a speed **polar diagram** per sailor.

## Run / verify

```bash
pip install -r requirements.txt
streamlit run version_1.py
```

Headless smoke test (no browser):

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("version_1.py").run()
assert not at.exception
```

Sample data for manual testing: `~/Downloads/Belmont_Sprints.gpx`
(a Strava GPX with ~2000 trackpoints, timezone Australia/Sydney).

## Architecture / key functions

- **Parsers** (`process_fit_gz`, `process_gpx`, `process_csv`) — each is
  `@st.cache_data` and takes raw `bytes` (call `file.read()` once and pass the
  bytes, never the file object, or caching breaks). Each returns a DataFrame with
  `lat, lon, time` (tz-aware, Australia/Sydney) then calls `add_kinematics`.
- **`add_kinematics(df)`** — vectorised haversine distance + great-circle bearing.
  Adds `speed`, `speed_knots`, `heading`, `distance_m`. Drops the first row (no
  prior point). Compute `dt` with `df["time"].diff().dt.total_seconds()` — a raw
  numpy subtraction on the tz-aware column raises a dtype error.
- **`generate_ladder_lines(...)`** — cached; crosswind reference lines along the
  wind axis. Takes lat/lon as **tuples** (hashable, for caching).
- **`vmg_line_polar(height, wind_dir, r_max, upwind)`** — returns `(theta, r)` for
  a horizontal VMG line. Maths: a horizontal screen line at vertical distance
  `height` is `r = height / cos(angle-from-top)`, clipped to `r_max`.

## Conventions / gotchas

- **Wind direction** (`wind_dir`, the "Wind Direction (degrees)" input) is the
  direction the wind comes **from**. It (a) sets the map `bearing`, and (b) rotates
  the polar angular axis to `(90 + wind_dir) % 360` so the wind sits at the top.
  Wind `090` → diagram spins 90° anti-clockwise. Keep these two uses in sync.
- The polar uses `direction="clockwise"`; headings are compass bearings
  (0 = N, 90 = E) and tick labels N/E/S/W ride along with the rotation.
- The polar trace **must** be added inside the `if not polar_filtered.empty:`
  block — `avg` is only defined there (a previous version had this bug).
- **Playback is intentionally disabled**: the whole block lives inside a
  `_PLAYBACK_DISABLED = ''' ... '''` string literal. To re-enable, remove the
  surrounding `'''`. `add_stitched_time` is only used by playback.
- Timezone is hard-coded to `Australia/Sydney`.
- The Mapbox token in the source is only used to set an env var; the map itself
  uses the free `open-street-map` style.

## When changing things

- After edits, run `python3 -m py_compile version_1.py` and the AppTest smoke test.
- Prefer vectorised numpy/pandas over per-row Python loops — large tracks are the
  common case and this is what keeps the app responsive.
