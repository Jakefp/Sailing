# Multi-Sailor GPS Analyzer

A Streamlit app for analysing sailing training data from GPS devices. Upload one
or more tracks and the app draws each sailor's path on a map and builds a speed
**polar diagram** so you can compare boat speed at every heading.

## Features

- **Multiple file formats** — `.fit`, `.fit.gz`, `.gpx` and `.csv`.
- **Multiple sailors** — upload several files at once and give each a name; tracks
  and polars are overlaid in different colours.
- **Track map** — every sailor's path drawn on an OpenStreetMap background.
- **Polar diagram** — average boat speed (knots) binned by heading, with an
  adjustable smoothing control.
- **Wind-aligned polar** — set the wind direction and the polar rotates so the
  wind always blows from the **top** of the diagram. For example, a wind of `090`
  spins the diagram 90° anti-clockwise.
- **VMG lines** — optional horizontal upwind / downwind velocity-made-good lines
  drawn straight across the polar. Use the sliders to move them up and down to
  find your best VMG angle (where the line just touches the polar curve).
- **Ladder lines** — optional crosswind reference lines spaced along the wind axis.
- **Time-range filtering** — analyse up to four separate time windows.
- **Minimum-speed filter** — drop slow/stationary points from the polar averages.

> The interactive **playback** feature is currently disabled (commented out in the
> code) so the app stays focused on tracks and polars. It can be re-enabled later.

## Running

```bash
pip install -r requirements.txt
streamlit run version_1.py
```

Then upload a GPS file (a sample `Belmont_Sprints.gpx` works well) and assign a
sailor name.

## How the polar / VMG maths works

- Boat speed and heading are computed between consecutive GPS points using a
  haversine distance and great-circle bearing (fully vectorised, so even multi-
  thousand-point files process in milliseconds).
- Headings are binned in 5° steps and averaged into the polar curve.
- The wind direction is the direction the wind blows **from**. The polar's angular
  axis is rotated to `(90 + wind) mod 360` so that heading appears at 12 o'clock.
- A VMG line at height *h* knots is the set of polar points where the on-screen
  vertical distance from the centre equals *h* — i.e. `r = h / cos(angle-from-top)`
  — which renders as a horizontal chord across the diagram.
