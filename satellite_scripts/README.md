# Satellite imagery scripts (run these yourself)

Download **real** monthly satellite rasters for Kolkata — VIIRS night-lights and
MODIS MAIAC AOD — and plot them. These need your own (free) Google Earth Engine
account; they are not run from inside this repo's pipeline.

## Datasets used
| Field | Earth Engine collection | Native res | Notes |
|---|---|---|---|
| Night-lights | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` (`avg_rad`) | ~500 m | monthly, stray-light corrected |
| AOD (550 nm) | `MODIS/061/MCD19A2_GRANULES` (`Optical_Depth_055`) | 1 km | MAIAC; scale factor 0.001, monthly mean |

Period: **2024-01 → 2026-03** (matches the project dataset). Output = one
multi-band GeoTIFF per field (one band per month).

## Option A — Code Editor (easiest, no install)
1. Sign up / sign in: <https://earthengine.google.com/signup> then
   <https://code.earthengine.google.com>.
2. Paste **`gee_kolkata_monthly.js`**, click **Run**.
3. Open the **Tasks** tab → **Run** the 2 export tasks.
4. GeoTIFFs appear in Google Drive → `kolkata_satellite/`. Download them here.

## Option B — Python API
```bash
pip install earthengine-api
earthengine authenticate
# edit PROJECT in gee_kolkata_monthly.py, then:
python gee_kolkata_monthly.py
```
Tasks run server-side; outputs land in the same Drive folder.

## Plot the downloaded rasters
```bash
pip install rasterio matplotlib numpy
python plot_satellite_geotiffs.py \
    --nl kolkata_nightlight_monthly.tif --aod kolkata_aod_monthly.tif
```
Writes `F18b_monthly_aod_satellite.png` and
`F19b_monthly_nightlight_satellite.png` into `../visualizations/` — the
true-raster counterparts of the dataset-interpolated F18/F19.

## Notes
- The scripts default to a bounding box around the 7 stations. To use the exact
  city polygon, upload `../geodata/kolkata.geojson` as a GEE asset and switch the
  `ROI`/`kolkata` line (commented in both scripts).
- For daytime true-colour / Black-Marble style imagery (your reference image 1),
  swap the VIIRS collection for `NASA/VIIRS/002/VNP46A3` (Black Marble monthly)
  or use Sentinel-2 `COPERNICUS/S2_SR_HARMONIZED` with a cloud filter — the same
  export pattern applies.
- MAIAC AOD has gaps under cloud; `.mean()` over each month fills most. Add the
  `cloudMask`/QA band if you want stricter quality control.
