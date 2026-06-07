"""Plot the monthly satellite GeoTIFFs (exported by the GEE script) as panels.

Reads the multi-band GeoTIFFs (one band per month) and renders a small-multiples
grid for night-lights and AOD — the real-raster versions of figures F18/F19.

SETUP:
    pip install rasterio matplotlib numpy
RUN (after the GeoTIFFs are in this folder, or pass paths):
    python plot_satellite_geotiffs.py \
        --nl kolkata_nightlight_monthly.tif --aod kolkata_aod_monthly.tif
Outputs PNGs into ../visualizations/.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
VIZ = HERE.parent / "visualizations"


def plot_stack(tif, cmap, label, title, out, pmin=3, pmax=97):
    import rasterio
    with rasterio.open(tif) as src:
        arr = src.read(masked=True).astype("float32")   # [bands, H, W]
        names = list(src.descriptions)
        bounds = src.bounds
    n = arr.shape[0]
    names = [nm or f"band {i+1}" for i, nm in enumerate(names)]
    # strip GEE toBands index prefix like "0_2024_01" -> "2024-01"
    clean = []
    for nm in names:
        parts = nm.split("_")
        clean.append("-".join(parts[-2:]) if len(parts) >= 2 else nm)

    valid = arr.compressed()
    vmin, vmax = np.nanpercentile(valid, [pmin, pmax])
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    ext = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.8 * nrow))
    axes = np.array(axes).ravel()
    im = None
    for i in range(n):
        ax = axes[i]
        im = ax.imshow(arr[i], cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=ext, origin="upper")
        ax.set_title(clean[i], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, len(axes)):
        axes[j].axis("off")
    if im is not None:
        cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
        cb.set_label(label)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    VIZ.mkdir(exist_ok=True)
    fig.savefig(VIZ / out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", VIZ / out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl", default=str(HERE / "kolkata_nightlight_monthly.tif"))
    ap.add_argument("--aod", default=str(HERE / "kolkata_aod_monthly.tif"))
    args = ap.parse_args()
    if Path(args.nl).exists():
        plot_stack(args.nl, "inferno", "VIIRS night-light radiance (nW/cm²/sr)",
                   "Monthly VIIRS night-lights over Kolkata (satellite)",
                   "F19b_monthly_nightlight_satellite.png")
    else:
        print("night-light GeoTIFF not found:", args.nl)
    if Path(args.aod).exists():
        plot_stack(args.aod, "YlOrRd", "MODIS MAIAC AOD (550 nm)",
                   "Monthly MODIS AOD over Kolkata (satellite)",
                   "F18b_monthly_aod_satellite.png")
    else:
        print("AOD GeoTIFF not found:", args.aod)


if __name__ == "__main__":
    main()
