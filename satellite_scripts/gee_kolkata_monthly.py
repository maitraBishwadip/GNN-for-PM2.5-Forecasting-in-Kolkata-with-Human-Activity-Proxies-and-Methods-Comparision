"""Kolkata monthly satellite fields via the Earth Engine Python API.

VIIRS DNB monthly night-lights + MODIS MAIAC 1-km AOD, 2024-01 .. 2026-03,
clipped to Kolkata, exported to Google Drive as multi-band GeoTIFFs
(one band per month).

SETUP (once):
    pip install earthengine-api
    earthengine authenticate          # opens a browser, sign in
    # set your Cloud project id below (or via: earthengine set_project <id>)

RUN:
    python gee_kolkata_monthly.py
Then check https://code.earthengine.google.com/tasks (or the prints) and let the
two export tasks finish. Files appear in Drive/kolkata_satellite, then run
plot_satellite_geotiffs.py.
"""
import ee

PROJECT = "your-ee-project-id"      # <-- EDIT: your Earth Engine Cloud project

# Bounding box around the 7 stations (immediately runnable).
ROI = ee.Geometry.Rectangle([88.30, 22.45, 88.46, 22.65])
# Or use your polygon asset:
# ROI = ee.FeatureCollection("projects/<proj>/assets/kolkata").geometry()

START = ee.Date("2024-01-01")
END = ee.Date("2026-04-01")          # exclusive


def monthly_stack(coll, scale_factor, n_months):
    months = ee.List.sequence(0, n_months.subtract(1))

    def one(m):
        m = ee.Number(m)
        s = START.advance(m, "month")
        e = s.advance(1, "month")
        return (coll.filterDate(s, e).mean()
                .multiply(scale_factor)
                .clip(ROI)
                .rename(s.format("YYYY_MM")))

    return ee.ImageCollection(months.map(one)).toBands()


def main():
    ee.Initialize(project=PROJECT)
    n = ee.Number(END.difference(START, "month")).round()

    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").select("avg_rad")
    aod = ee.ImageCollection("MODIS/061/MCD19A2_GRANULES").select("Optical_Depth_055")

    nl_stack = monthly_stack(viirs, 1, n)
    aod_stack = monthly_stack(aod, 0.001, n)

    t1 = ee.batch.Export.image.toDrive(
        image=nl_stack, description="kolkata_nightlight_monthly",
        folder="kolkata_satellite", fileNamePrefix="kolkata_nightlight_monthly",
        region=ROI, scale=500, crs="EPSG:4326", maxPixels=int(1e9))
    t2 = ee.batch.Export.image.toDrive(
        image=aod_stack, description="kolkata_aod_monthly",
        folder="kolkata_satellite", fileNamePrefix="kolkata_aod_monthly",
        region=ROI, scale=1000, crs="EPSG:4326", maxPixels=int(1e9))
    t1.start(); t2.start()
    print(f"Exporting {n.getInfo()} months/variable. Tasks started:")
    print(" night-lights:", t1.id)
    print(" AOD         :", t2.id)
    print("Monitor at https://code.earthengine.google.com/tasks "
          "-> outputs in Drive/kolkata_satellite")


if __name__ == "__main__":
    main()
