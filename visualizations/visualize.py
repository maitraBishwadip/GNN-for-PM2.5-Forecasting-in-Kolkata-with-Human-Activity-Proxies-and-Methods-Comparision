"""Generate publication-standard figures for the write-up.

EDA figures are built from the CSV; result figures from artefacts produced by
`run.py` (results/test_predictions.npz, results/metrics.json).

Usage:
    python -m visualizations.visualize
Figures are written as 300-dpi PNGs into the visualizations/ folder.
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

from models import config as C

OUT = C.VIZ_DIR
SEASON_ORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
PALETTE = {
    "Persistence": "#7f8c8d", "HistGBR": "#e67e22", "LSTM": "#27ae60",
    "Pure-STGNN": "#2980b9", "GNN-LSTM": "#c0392b",
}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.edgecolor": "#444444",
        "lines.linewidth": 1.6, "figure.autolayout": False,
    })


def _load_df():
    df = pd.read_csv(C.CSV_PATH, parse_dates=["datetime_IST"])
    df["month"] = df["datetime_IST"].dt.month
    df["hour"] = df["datetime_IST"].dt.hour
    df["season"] = df["month"].map(C.SEASONS)
    return df


# ----------------------------- EDA figures -----------------------------------
def fig_station_map(df):
    g = df.groupby("Station").agg(lat=("Latitude", "mean"), lon=("Longitude", "mean"),
                                  pm=("Ground_PM2.5", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    sc = ax.scatter(g.lon, g.lat, c=g.pm, s=300, cmap="YlOrRd",
                    edgecolor="k", linewidth=1.2, zorder=3)
    for _, r in g.iterrows():
        ax.annotate(r.Station.replace("_", " "), (r.lon, r.lat), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    fig.colorbar(sc, ax=ax, label="Mean ground PM$_{2.5}$ (µg/m³)")
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title("Kolkata monitoring network (7 stations)")
    _save(fig, "F1_station_map")


def fig_spatial_contour(df):
    """Smoothed spatial contour of mean PM2.5 interpolated across the city."""
    g = df.groupby("Station").agg(lat=("Latitude", "mean"), lon=("Longitude", "mean"),
                                  pm=("Ground_PM2.5", "mean")).reset_index()
    lon, lat, pm = g.lon.values, g.lat.values, g.pm.values
    pad = 0.02
    gx = np.linspace(lon.min() - pad, lon.max() + pad, 200)
    gy = np.linspace(lat.min() - pad, lat.max() + pad, 200)
    GX, GY = np.meshgrid(gx, gy)
    rbf = RBFInterpolator(np.c_[lon, lat], pm, kernel="thin_plate_spline", smoothing=0.5)
    GZ = rbf(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    cf = ax.contourf(GX, GY, GZ, levels=14, cmap="YlOrRd")
    cs = ax.contour(GX, GY, GZ, levels=7, colors="k", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
    ax.scatter(lon, lat, c="k", s=40, zorder=4)
    for _, r in g.iterrows():
        ax.annotate(r.Station.replace("_", " "), (r.lon, r.lat), fontsize=7,
                    xytext=(4, 4), textcoords="offset points", color="navy")
    fig.colorbar(cf, ax=ax, label="Mean ground PM$_{2.5}$ (µg/m³)")
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title("Spatial distribution of mean PM$_{2.5}$ over Kolkata")
    ax.grid(False)
    _save(fig, "F4_spatial_contour")


def fig_seasonality(df):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    mon = df.groupby("month")["Ground_PM2.5"].mean()
    bars = ax[0].bar(mon.index, mon.values, color="#c0392b", edgecolor="k", linewidth=0.4)
    ax[0].set_xticks(range(1, 13))
    ax[0].set_xlabel("Month"); ax[0].set_ylabel("Mean ground PM$_{2.5}$ (µg/m³)")
    ax[0].set_title("(a) Monthly mean PM$_{2.5}$")
    for b, v in zip(bars, mon.values):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}",
                   ha="center", fontsize=6)
    data = [df[df.season == s]["Ground_PM2.5"].dropna() for s in SEASON_ORDER]
    bp = ax[1].boxplot(data, tick_labels=[s.replace("-", "-\n") for s in SEASON_ORDER],
                       showfliers=False, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#2c5f8a", "#3a9", "#27ae60", "#e67e22"]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax[1].set_ylabel("Ground PM$_{2.5}$ (µg/m³)")
    ax[1].set_title("(b) Seasonal distribution")
    _save(fig, "F6_seasonality")


def fig_diurnal(df):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    h = df.groupby("hour")["Ground_PM2.5"].mean()
    ax.plot(h.index, h.values, "-o", color="#2c3e50", ms=4)
    ax.fill_between(h.index, h.values, alpha=0.12, color="#2c3e50")
    ax.set_xlabel("Hour of day (IST)"); ax.set_ylabel("Mean ground PM$_{2.5}$ (µg/m³)")
    ax.set_title("Diurnal cycle of PM$_{2.5}$")
    ax.set_xticks(range(0, 24, 2))
    _save(fig, "F7_diurnal_cycle")


def fig_nightlight_scatter(df):
    g = df.groupby("Station").agg(nl=("Night_Light", "mean"),
                                  pm=("Ground_PM2.5", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.scatter(g.nl, g.pm, s=200, c="#8e44ad", edgecolor="k", zorder=3)
    for _, r in g.iterrows():
        ax.annotate(r.Station.replace("_", " "), (r.nl, r.pm), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Mean night-time light (human-activity proxy)")
    ax.set_ylabel("Mean ground PM$_{2.5}$ (µg/m³)")
    ax.set_title("Human activity (Night_Light) vs PM$_{2.5}$ by station")
    _save(fig, "F8_nightlight_vs_pm")


def fig_nightlight_spatiotemporal(df):
    """Spatio-temporal CONTOUR (Hovmoller): time x station-latitude, filled by value.

    (a) Night_Light and (b) Ground PM2.5, so one can read off when and where
    night-light (human activity) changes and how PM2.5 responds.
    """
    import matplotlib.dates as mdates
    coords = df.groupby("Station")["Latitude"].mean()
    d = df.copy()
    d["ym"] = d["datetime_IST"].values.astype("datetime64[M]")
    nl = d.pivot_table(index="ym", columns="Station", values="Night_Light", aggfunc="mean")
    ao = d.pivot_table(index="ym", columns="Station", values="AOD", aggfunc="mean")
    pm = d.pivot_table(index="ym", columns="Station", values="Ground_PM2.5", aggfunc="mean")

    stations = sorted(coords.index, key=lambda s: coords[s])      # south -> north
    lat = np.array([coords[s] for s in stations])
    times = pd.to_datetime(nl.index)
    X = mdates.date2num(times)
    TX, TY = np.meshgrid(X, lat)

    def grid(piv):
        z = piv[stations].T                                       # [station, time]
        z = z.interpolate(axis=1, limit_direction="both")
        return z.fillna(z.stack().mean()).values

    nlZ, aoZ, pmZ = grid(nl), grid(ao), grid(pm)

    fig, ax = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    specs = [(ax[0], nlZ, "viridis", "Night-time light radiance",
              "(a) Night-time light — spatio-temporal contour (time × latitude)"),
             (ax[1], aoZ, "YlGnBu", "Aerosol Optical Depth (AOD)",
              "(b) Satellite AOD — spatio-temporal contour (time × latitude)"),
             (ax[2], pmZ, "YlOrRd", "Ground PM$_{2.5}$ (µg/m³)",
              "(c) Ground PM$_{2.5}$ — spatio-temporal contour (time × latitude)")]
    for a, Z, cmap, lab, title in specs:
        cf = a.contourf(TX, TY, Z, levels=14, cmap=cmap)
        a.contour(TX, TY, Z, levels=7, colors="k", linewidths=0.3, alpha=0.4)
        a.set_yticks(lat)
        a.set_yticklabels([s.replace("_", " ") for s in stations], fontsize=8)
        a.set_ylabel("Station (by latitude, S→N)")
        a.set_title(title)
        a.xaxis_date()
        a.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        a.grid(False)
        cb = fig.colorbar(cf, ax=a, fraction=0.04, pad=0.02)
        cb.set_label(lab)
    ax[2].set_xlabel("Date")
    fig.autofmt_xdate(rotation=30)
    _save(fig, "F12_nightlight_spatiotemporal")


def fig_sat_vs_ground(df):
    d = df[["PM25", "Ground_PM2.5"]].dropna()
    d = d.sample(min(20000, len(d)), random_state=0)
    fig, ax = plt.subplots(figsize=(6, 5.6))
    hb = ax.hexbin(d["Ground_PM2.5"], d["PM25"], gridsize=45, cmap="viridis",
                   mincnt=1, bins="log")
    lim = [0, float(d.max().max())]
    ax.plot(lim, lim, "w--", lw=1.4, label="1:1 line")
    bias = float((df["PM25"] - df["Ground_PM2.5"]).mean())
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Ground PM$_{2.5}$ (µg/m³)"); ax.set_ylabel("Satellite PM$_{2.5}$ (µg/m³)")
    ax.set_title(f"Satellite vs ground (mean bias = {bias:+.1f} µg/m³)")
    fig.colorbar(hb, ax=ax, label="log$_{10}$(count)")
    ax.legend(loc="upper left"); ax.grid(False)
    _save(fig, "F9_satellite_vs_ground")


# ---------------------------- result figures ---------------------------------
def fig_model_comparison():
    m = _metrics()
    if not m:
        return
    names = [n for n in PALETTE if n in m] + [n for n in m if n not in PALETTE]
    mae = [m[n]["overall"]["MAE"] for n in names]
    rmse = [m[n]["overall"]["RMSE"] for n in names]
    r2 = [m[n]["overall"]["R2"] for n in names]
    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    b1 = ax[0].bar(x - w / 2, mae, w, label="MAE", color="#2980b9", edgecolor="k", linewidth=0.4)
    b2 = ax[0].bar(x + w / 2, rmse, w, label="RMSE", color="#e67e22", edgecolor="k", linewidth=0.4)
    for bars in (b1, b2):
        for b in bars:
            ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                       f"{b.get_height():.1f}", ha="center", fontsize=7)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=18, ha="right")
    ax[0].set_ylabel("Error (µg/m³)"); ax[0].set_title("(a) Test error by model")
    ax[0].legend()
    bars = ax[1].bar(x, r2, color=[PALETTE.get(n, "#888") for n in names],
                     edgecolor="k", linewidth=0.4)
    for b in bars:
        ax[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                   f"{b.get_height():.3f}", ha="center", fontsize=7)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=18, ha="right")
    ax[1].set_ylabel("R²"); ax[1].set_ylim(0.6, 0.82); ax[1].set_title("(b) Coefficient of determination")
    _save(fig, "F_model_comparison")


def fig_error_vs_horizon():
    m = _metrics()
    if not m:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for n in [x for x in PALETTE if x in m] + [x for x in m if x not in PALETTE]:
        ph = m[n].get("per_horizon", {})
        if not ph:
            continue
        hs = [int(k[2:]) for k in ph]
        col = PALETTE.get(n, None)
        ax[0].plot(hs, [ph[k]["RMSE"] for k in ph], "-o", ms=4, color=col, label=n)
        ax[1].plot(hs, [ph[k]["R2"] for k in ph], "-o", ms=4, color=col, label=n)
    ax[0].set_xlabel("Forecast horizon (h)"); ax[0].set_ylabel("RMSE (µg/m³)")
    ax[0].set_title("(a) RMSE vs horizon"); ax[0].legend()
    ax[1].set_xlabel("Forecast horizon (h)"); ax[1].set_ylabel("R²")
    ax[1].set_title("(b) R² vs horizon"); ax[1].legend()
    _save(fig, "F10_error_vs_horizon")


def fig_pred_vs_obs_all():
    npz = C.RESULTS_DIR / "test_predictions.npz"
    if not npz.exists():
        return
    d = np.load(npz, allow_pickle=True)
    P, Y, M = d["P"], d["Y"], d["M"]
    stations = list(d["stations"]); horizons = list(d["horizons"])
    hi = horizons.index(1) if 1 in horizons else 0
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=False)
    axes = axes.ravel()
    for si, s in enumerate(stations):
        ax = axes[si]
        mask = M[:, si, hi]
        p, y = P[mask, si, hi], Y[mask, si, hi]
        n = min(350, len(p))
        ax.plot(y[:n], color="#2c3e50", lw=1.2, label="Observed")
        ax.plot(p[:n], color="#c0392b", lw=1.0, alpha=0.85, label="Predicted")
        mae = np.mean(np.abs(p - y)) if len(p) else float("nan")
        ax.set_title(f"{s.replace('_', ' ')}  (MAE={mae:.1f})", fontsize=10)
        ax.set_ylabel("PM$_{2.5}$")
        if si == 0:
            ax.legend(loc="upper right", fontsize=8)
    for j in range(len(stations), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"GNN-LSTM predicted vs observed — all stations (T+{horizons[hi]}h, test set)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, "F11_pred_vs_obs_all_stations", tight=False)


def _target_frame(times, te_ends, horizon, P, Y, M, hi):
    """Flatten (sample x station) observed/predicted with target timestamps."""
    N = P.shape[1]
    tgt = times[te_ends + horizon]                       # [S]
    tmat = np.repeat(tgt[:, None], N, axis=1)
    sel = M[:, :, hi]
    df = pd.DataFrame({
        "time": pd.to_datetime(tmat[sel]),
        "obs": Y[:, :, hi][sel],
        "pred": P[:, :, hi][sel],
    })
    df["hour"] = df["time"].dt.hour
    df["season"] = df["time"].dt.month.map(C.SEASONS)
    return df


def fig_trend_prediction(times):
    """Predicted vs observed temporal trends (diurnal + monthly) — GNN-LSTM."""
    npz = C.RESULTS_DIR / "test_predictions.npz"
    if not npz.exists():
        return
    d = np.load(npz, allow_pickle=True)
    P, Y, M = d["P"], d["Y"], d["M"]
    te_ends = d["te_ends"]; horizons = list(d["horizons"])
    hi = horizons.index(1) if 1 in horizons else 0
    df = _target_frame(times, te_ends, horizons[hi], P, Y, M, hi)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    # (a) diurnal trend
    gh = df.groupby("hour").agg(o=("obs", "mean"), om=("obs", "std"),
                                p=("pred", "mean")).reset_index()
    ax[0].plot(gh.hour, gh.o, "-o", color="#2c3e50", ms=4, label="Observed")
    ax[0].fill_between(gh.hour, gh.o - gh.om, gh.o + gh.om, color="#2c3e50", alpha=0.12)
    ax[0].plot(gh.hour, gh.p, "-s", color="#c0392b", ms=4, label="Predicted")
    ax[0].set_xlabel("Hour of day (IST)"); ax[0].set_ylabel("PM$_{2.5}$ (µg/m³)")
    ax[0].set_title("(a) Diurnal trend — predicted vs observed")
    ax[0].set_xticks(range(0, 24, 3)); ax[0].legend()
    # (b) monthly trend over test period
    ms = df.set_index("time").resample("MS").agg(o=("obs", "mean"), p=("pred", "mean"))
    ax[1].plot(ms.index, ms.o, "-o", color="#2c3e50", ms=4, label="Observed")
    ax[1].plot(ms.index, ms.p, "-s", color="#c0392b", ms=4, label="Predicted")
    ax[1].set_xlabel("Date"); ax[1].set_ylabel("Monthly mean PM$_{2.5}$ (µg/m³)")
    ax[1].set_title("(b) Monthly trend over test period")
    ax[1].legend(); fig.autofmt_xdate(rotation=30)
    _save(fig, "F13_trend_prediction")


def fig_seasonal_prediction(times):
    """Predicted vs observed seasonal variation + per-season bias — GNN-LSTM."""
    npz = C.RESULTS_DIR / "test_predictions.npz"
    if not npz.exists():
        return
    d = np.load(npz, allow_pickle=True)
    P, Y, M = d["P"], d["Y"], d["M"]
    te_ends = d["te_ends"]; horizons = list(d["horizons"])
    hi = horizons.index(1) if 1 in horizons else 0
    df = _target_frame(times, te_ends, horizons[hi], P, Y, M, hi)
    df["bias"] = df["pred"] - df["obs"]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    # (a) seasonal mean observed vs predicted
    gs = df.groupby("season").agg(o=("obs", "mean"), p=("pred", "mean")).reindex(SEASON_ORDER)
    x = np.arange(len(SEASON_ORDER)); w = 0.38
    b1 = ax[0].bar(x - w / 2, gs.o, w, label="Observed", color="#2c3e50", edgecolor="k", linewidth=0.4)
    b2 = ax[0].bar(x + w / 2, gs.p, w, label="Predicted", color="#c0392b", edgecolor="k", linewidth=0.4)
    for bars in (b1, b2):
        for b in bars:
            ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                       f"{b.get_height():.0f}", ha="center", fontsize=7)
    ax[0].set_xticks(x); ax[0].set_xticklabels([s.replace("-", "-\n") for s in SEASON_ORDER])
    ax[0].set_ylabel("Mean PM$_{2.5}$ (µg/m³)")
    ax[0].set_title("(a) Seasonal variation — predicted vs observed"); ax[0].legend()
    # (b) per-season bias boxplot
    data = [df[df.season == s]["bias"].dropna() for s in SEASON_ORDER]
    bp = ax[1].boxplot(data, tick_labels=[s.replace("-", "-\n") for s in SEASON_ORDER],
                       showfliers=False, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#2c5f8a", "#3aa", "#27ae60", "#e67e22"]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax[1].axhline(0, color="k", lw=0.8, ls="--")
    ax[1].set_ylabel("Prediction bias (pred − obs, µg/m³)")
    ax[1].set_title("(b) Forecast bias by season")
    _save(fig, "F14_seasonal_prediction")


def _load_boundary():
    geo = C.ROOT / "geodata" / "kolkata.geojson"
    if geo.exists():
        try:
            import geopandas as gpd
            return gpd.read_file(geo)
        except Exception:
            return None
    return None


def _rbf_grid(lon, lat, vals, gx, gy, smoothing=0.3):
    GX, GY = np.meshgrid(gx, gy)
    rbf = RBFInterpolator(np.c_[lon, lat], vals,
                          kernel="thin_plate_spline", smoothing=smoothing)
    return GX, GY, rbf(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)


def fig_daily_spatial_evolution(df, start="2025-01-01", n_days=12, cmap="YlOrRd"):
    """Small-multiples of the interpolated daily-mean PM2.5 field (like ref. img 1)."""
    coords = df.groupby("Station").agg(lat=("Latitude", "mean"),
                                       lon=("Longitude", "mean"))
    daily = (df.set_index("datetime_IST")
               .groupby("Station")["Ground_PM2.5"].resample("D").mean()
               .reset_index())
    days = pd.date_range(start, periods=n_days, freq="D")
    lon_all = coords["lon"].values; lat_all = coords["lat"].values
    pad = 0.012
    gx = np.linspace(lon_all.min() - pad, lon_all.max() + pad, 70)
    gy = np.linspace(lat_all.min() - pad, lat_all.max() + pad, 70)
    boundary = _load_boundary()

    # shared colour scale across all days
    sub = daily[daily["datetime_IST"].isin(days)]
    vmin, vmax = np.nanpercentile(sub["Ground_PM2.5"], [5, 95])

    ncol = 4
    nrow = int(np.ceil(n_days / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.9 * nrow))
    axes = np.array(axes).ravel()
    mesh = None
    for k, day in enumerate(days):
        ax = axes[k]
        dd = daily[daily["datetime_IST"] == day].set_index("Station")["Ground_PM2.5"]
        pts = [(coords.loc[s, "lon"], coords.loc[s, "lat"], dd[s])
               for s in coords.index if s in dd and np.isfinite(dd[s])]
        ax.set_title(day.strftime("%a %Y-%m-%d"), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if len(pts) >= 4:
            lo = np.array([p[0] for p in pts]); la = np.array([p[1] for p in pts])
            vv = np.array([p[2] for p in pts])
            GX, GY, GZ = _rbf_grid(lo, la, vv, gx, gy)
            mesh = ax.pcolormesh(GX, GY, GZ, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
            ax.scatter(lo, la, c="k", s=10, zorder=5)
        if boundary is not None:
            boundary.boundary.plot(ax=ax, color="#333", linewidth=0.6, zorder=4)
        ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[0], gy[-1])
    for j in range(n_days, len(axes)):
        axes[j].axis("off")
    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes.tolist(), fraction=0.02, pad=0.02)
        cb.set_label("Daily-mean ground PM$_{2.5}$ (µg/m³)")
    fig.suptitle(f"Daily PM$_{{2.5}}$ spatial evolution over Kolkata "
                 f"({days[0]:%d %b %Y} – {days[-1]:%d %b %Y})",
                 fontsize=13, fontweight="bold", y=0.98)
    _save(fig, "F16_daily_spatial_evolution", tight=False)


def fig_spatiotemporal_eof(df, n_comp=3):
    """EOF/PCA decomposition: temporal basis + spatial coeffs + modeled maps (ref. img 3)."""
    from sklearn.decomposition import PCA
    coords = df.groupby("Station").agg(lat=("Latitude", "mean"), lon=("Longitude", "mean"))
    stations = list(coords.index)
    mat = (df.pivot_table(index="datetime_IST", columns="Station",
                          values="Ground_PM2.5", aggfunc="mean")
             .resample("D").mean()[stations])
    mat = mat.interpolate(limit_direction="both").fillna(mat.mean())
    mu = mat.mean(); sd = mat.std().replace(0, 1)
    A = ((mat - mu) / sd).values                      # [time, space] anomalies
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(A)                     # [time, n_comp] temporal basis
    load = pca.components_                            # [n_comp, space] spatial coeffs
    evr = pca.explained_variance_ratio_ * 100
    t = mat.index
    lon = coords["lon"].values; lat = coords["lat"].values
    pad = 0.012
    gx = np.linspace(lon.min() - pad, lon.max() + pad, 90)
    gy = np.linspace(lat.min() - pad, lat.max() + pad, 90)
    boundary = _load_boundary()

    fig, axes = plt.subplots(3, n_comp, figsize=(4.6 * n_comp, 11))
    for c in range(n_comp):
        # row 0 — temporal basis
        a = axes[0, c]
        a.plot(t, scores[:, c], color="#555", lw=0.8)
        a.axhline(0, color="k", lw=0.5, ls="--")
        a.set_title(f"Temporal basis — component {c+1}  ({evr[c]:.0f}% var)", fontsize=10)
        a.set_ylabel("PC score" if c == 0 else "")
        a.tick_params(labelsize=7)
        # row 1 — standardized spatial coefficients (scatter)
        a = axes[1, c]
        vmax = np.abs(load[c]).max()
        sc = a.scatter(lon, lat, c=load[c], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       s=220, edgecolor="k", linewidth=0.8, zorder=5)
        if boundary is not None:
            boundary.boundary.plot(ax=a, color="#333", linewidth=0.6, zorder=4)
        a.set_aspect("equal"); a.set_xlim(gx[0], gx[-1]); a.set_ylim(gy[0], gy[-1])
        a.set_title(f"Spatial coefficients — comp {c+1}", fontsize=10)
        a.tick_params(labelsize=7)
        fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02)
        # row 2 — modeled (interpolated) spatial coefficients
        a = axes[2, c]
        GX, GY, GZ = _rbf_grid(lon, lat, load[c], gx, gy, smoothing=0.2)
        pm = a.pcolormesh(GX, GY, GZ, cmap="viridis", shading="auto")
        a.scatter(lon, lat, c="k", s=12, zorder=5)
        if boundary is not None:
            boundary.boundary.plot(ax=a, color="w", linewidth=0.6, zorder=4)
        a.set_aspect("equal"); a.set_xlim(gx[0], gx[-1]); a.set_ylim(gy[0], gy[-1])
        a.set_title(f"Modeled spatial coefficients — comp {c+1}", fontsize=10)
        a.tick_params(labelsize=7)
        fig.colorbar(pm, ax=a, fraction=0.046, pad=0.02)
    fig.suptitle("Spatio-temporal EOF/PCA decomposition of Kolkata PM$_{2.5}$",
                 fontsize=14, fontweight="bold", y=0.995)
    _save(fig, "F17_spatiotemporal_eof", tight=True)


def fig_architecture():
    """Submission-quality schematic of the proposed GNN-LSTM model."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(15.5, 8.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def panel(x, w, color, label):
        ax.add_patch(FancyBboxPatch((x, 9), w, 79, boxstyle="round,pad=0.6",
                     linewidth=0, facecolor=color, alpha=0.45, zorder=0))
        ax.text(x + w / 2, 92, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#2c3e50", zorder=1)

    def box(x, y, w, h, text, fc, fs=9.5, ec="#1b2631", title=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                     linewidth=1.4, facecolor=fc, edgecolor=ec, zorder=3))
        if title:
            ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center",
                    fontsize=fs + 0.5, fontweight="bold", zorder=4)
            ax.text(x + w / 2, y + (h - 4) / 2, text, ha="center", va="center",
                    fontsize=fs, zorder=4)
        else:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=fs, zorder=4)

    def arrow(x1, y1, x2, y2, style="-|>", rad=0.0, color="#2c3e50", lw=1.8, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                     mutation_scale=16, lw=lw, color=color, linestyle=ls,
                     connectionstyle=f"arc3,rad={rad}", zorder=2))

    def shape(x, y, txt):
        ax.text(x, y, txt, ha="center", va="center", fontsize=8,
                style="italic", color="#566573",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#ccc", lw=0.6), zorder=5)

    # ---- stage panels ----
    panel(0.5, 18.5, "#d5f5e3", "1.  Inputs")
    panel(20.5, 28.5, "#d6eaf8", "2.  Spatial encoder (GNN)")
    panel(50.5, 24.0, "#fdebd0", "3.  Temporal encoder")
    panel(75.5, 24.0, "#ebdef0", "4.  Prediction head")

    yc = 48          # core pipeline baseline (box bottom)
    h = 22

    # ---- ① inputs ----
    feat = ("• PM$_{2.5}$ (sat), AOD\n• Temp, RH, wind $u,v$\n"
            "• PBLH, rain, cloud\n• Night-light (activity)\n• time enc. (hr/dow/mon)")
    box(2.0, 52, 15.5, 30, feat, "#a9dfbf", fs=8.6, title="Node features")
    # dynamic graph box + glyph
    box(2.0, 14, 15.5, 26, "wind direction + inverse distance\n$\\rightarrow\\ A_t\\in\\mathbb{R}^{7\\times7}$  (hourly)",
        "#7dcea0", fs=8.4, title="Dynamic directed graph $G_t$")
    rng = np.random.default_rng(3)
    gpts = np.c_[5.0 + rng.uniform(0, 9.5, 7), 15.2 + rng.uniform(0, 3.4, 7)]
    for i in range(7):
        for j in range(7):
            if i != j and rng.random() < 0.28:
                arrow(gpts[i, 0], gpts[i, 1], gpts[j, 0], gpts[j, 1], color="#1e8449", lw=0.5)
    ax.scatter(gpts[:, 0], gpts[:, 1], s=26, c="#145a32", zorder=5)

    # ---- ② spatial encoder ----
    box(21.5, yc, 12.5, h, "Linear\n22 → 64", "#aed6f1", title="Node\nembedding", fs=9)
    box(36.0, yc, 13.0, h, "2 layers · 4 heads\nELU · wind-edge\nattention bias", "#7fb3d5",
        title="Spatial GAT", fs=8.8)

    # ---- ③ temporal encoder ----
    box(51.5, yc, 10.5, h, "GAT ⊕ raw\nfeatures\n(skip)", "#f8c471", title="Concat", fs=8.8)
    box(63.5, yc, 10.5, h, "2 layers\nhidden 64\nover H hours", "#f0a04b", title="LSTM", fs=8.8)
    # unrolled LSTM cells under the LSTM box
    for k, lab in enumerate(["t-2", "t-1", "t", "…"]):
        cx = 62.0 + k * 2.6
        ax.add_patch(FancyBboxPatch((cx, 33), 2.1, 5.5, boxstyle="round,pad=0.2",
                     fc="#fdebd0", ec="#b9770e", lw=0.8, zorder=3))
        ax.text(cx + 1.05, 35.8, lab, ha="center", va="center", fontsize=6.5, zorder=4)
        if k:
            arrow(cx - 0.6, 35.8, cx, 35.8, color="#b9770e", lw=0.9)
    arrow(68.7, 44, 68.7, yc, color="#b9770e", lw=1.0, style="-|>")

    # ---- ④ prediction head ----
    box(76.5, yc, 10.0, h, "64 → 64\n→ 5", "#c39bd3", title="FC head", fs=9)
    box(88.0, 38, 10.5, 42,
        "$P_F$ per station\n\nT+1 h\nT+3 h\nT+6 h\nT+12 h\nT+24 h", "#e6b0aa",
        title="Forecast", fs=8.8)

    # ---- main forward arrows + tensor shapes ----
    arrow(17.5, 63, 21.5, yc + h / 2); shape(19.5, 73, "[H,7,22]")
    arrow(34.0, yc + h / 2, 36.0, yc + h / 2)
    arrow(49.0, yc + h / 2, 51.5, yc + h / 2); shape(50.2, 73, "[H,7,64]")
    arrow(62.0, yc + h / 2, 63.5, yc + h / 2); shape(57.5, 73, "[H,7,86]")
    arrow(74.0, yc + h / 2, 76.5, yc + h / 2); shape(75.3, 73, "[7,64]")
    arrow(86.5, yc + h / 2, 88.0, yc + h / 2); shape(87.4, 73, "[7,5]")

    # graph A_t -> GAT (from inputs graph box up into GAT)
    arrow(17.5, 27, 42.5, yc, rad=-0.18, color="#145a32", lw=1.6)
    ax.text(30, 38, "$A_t$", fontsize=10, color="#145a32", fontweight="bold")

    # skip connection raw features -> concat
    arrow(9.8, 52, 56.7, yc, rad=-0.30, color="#7f8c8d", lw=1.5, ls="--")
    ax.text(34, 30, "skip connection (raw node features)", ha="center",
            fontsize=8.2, color="#7f8c8d", style="italic")

    ax.text(50, 97.5, "Proposed GNN-LSTM architecture for Kolkata PM$_{2.5}$ forecasting",
            ha="center", fontsize=14.5, fontweight="bold")
    ax.text(50, 4.2, "Pure-STGNN variant: replace the temporal LSTM (stage 3) with a "
            "temporal GAT over the hour axis — a fully graph-based, recurrence-free model.",
            ha="center", fontsize=9.2, style="italic", color="#555")
    _save(fig, "F3_architecture", tight=False)


def fig_annual_cycle(df):
    """Annual cycle of PM2.5: continuous record + day-of-year climatology."""
    d = df.set_index("datetime_IST")["Ground_PM2.5"]
    daily = d.resample("D").mean()
    roll = daily.rolling(7, center=True, min_periods=3).mean()

    fig, ax = plt.subplots(2, 1, figsize=(12.5, 8))
    # (a) continuous daily-mean time series with winter shading
    ax[0].plot(daily.index, daily.values, color="#b0bec5", lw=0.7, label="Daily mean")
    ax[0].plot(roll.index, roll.values, color="#c0392b", lw=1.8, label="7-day mean")
    for yr in sorted({t.year for t in daily.index}):
        ax[0].axvspan(pd.Timestamp(yr - 1, 12, 1), pd.Timestamp(yr, 2, 28),
                      color="#3498db", alpha=0.07)
    ax[0].set_ylabel("Ground PM$_{2.5}$ (µg/m³)")
    ax[0].set_title("(a) Continuous daily-mean PM$_{2.5}$ over the record "
                    "(winters shaded)")
    ax[0].legend(loc="upper right")
    ax[0].set_xlim(daily.index.min(), daily.index.max())

    # (b) day-of-year climatology, each year overlaid + mean ± std band
    dd = daily.to_frame("pm")
    dd["year"] = dd.index.year
    dd["doy"] = dd.index.dayofyear
    cmap = plt.get_cmap("viridis")
    years = sorted(dd["year"].unique())
    for i, y in enumerate(years):
        s = dd[dd.year == y].set_index("doy")["pm"].rolling(7, center=True, min_periods=3).mean()
        ax[1].plot(s.index, s.values, color=cmap(i / max(1, len(years) - 1)),
                   lw=1.4, label=str(y))
    clim = dd.groupby("doy")["pm"].agg(["mean", "std"])
    clim = clim.rolling(7, center=True, min_periods=3).mean()
    ax[1].plot(clim.index, clim["mean"], color="k", lw=2.4, label="Climatology")
    ax[1].fill_between(clim.index, clim["mean"] - clim["std"], clim["mean"] + clim["std"],
                       color="k", alpha=0.12)
    month_doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ax[1].set_xticks(month_doy)
    ax[1].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax[1].set_xlim(1, 366)
    ax[1].set_ylabel("Ground PM$_{2.5}$ (µg/m³)"); ax[1].set_xlabel("Month")
    ax[1].set_title("(b) Annual cycle — day-of-year climatology (years overlaid)")
    ax[1].legend(ncol=len(years) + 1, fontsize=8, loc="upper right")
    _save(fig, "F20_annual_cycle")


def fig_monthly_satellite_field(df, col, cmap, label, fname, title):
    """12 calendar-month interpolated maps of a satellite-derived field over Kolkata.

    NOTE: built from the satellite-derived point values in the dataset (the `col`
    column), interpolated across the 7 stations — not raw raster tiles (raster
    download needs Earth Engine / Earthdata credentials not available here).
    """
    coords = df.groupby("Station").agg(lat=("Latitude", "mean"), lon=("Longitude", "mean"))
    g = df.groupby(["Station", "month"])[col].mean()
    lon_all = coords["lon"].values; lat_all = coords["lat"].values
    pad = 0.012
    gx = np.linspace(lon_all.min() - pad, lon_all.max() + pad, 80)
    gy = np.linspace(lat_all.min() - pad, lat_all.max() + pad, 80)
    boundary = _load_boundary()

    allvals = g.dropna().values
    vmin, vmax = np.nanpercentile(allvals, [3, 97])
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 9.2))
    axes = axes.ravel(); mesh = None
    for m in range(1, 13):
        ax = axes[m - 1]
        pts = [(coords.loc[s, "lon"], coords.loc[s, "lat"], g.get((s, m), np.nan))
               for s in coords.index]
        pts = [(lo, la, v) for lo, la, v in pts if np.isfinite(v)]
        ax.set_title(names[m - 1], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if len(pts) >= 4:
            lo = np.array([p[0] for p in pts]); la = np.array([p[1] for p in pts])
            vv = np.array([p[2] for p in pts])
            GX, GY, GZ = _rbf_grid(lo, la, vv, gx, gy, smoothing=0.3)
            mesh = ax.pcolormesh(GX, GY, GZ, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
            ax.scatter(lo, la, c="cyan", s=12, edgecolor="k", linewidth=0.4, zorder=5)
        if boundary is not None:
            ec = "w" if cmap in ("inferno", "magma", "cividis") else "#333"
            boundary.boundary.plot(ax=ax, color=ec, linewidth=0.6, zorder=4)
        ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[0], gy[-1])
    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes.tolist(), fraction=0.02, pad=0.02)
        cb.set_label(label)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    _save(fig, fname, tight=False)


def _stmetric(pred, true):
    e = pred - true
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mape = float(np.mean(np.abs(e) / np.clip(np.abs(true), 1.0, None)) * 100)
    ss_res = np.sum(e ** 2)
    r2 = float(1 - ss_res / (np.sum((true - true.mean()) ** 2) + 1e-9))
    om = true.mean()
    ia = float(1 - ss_res / (np.sum((np.abs(pred - om) + np.abs(true - om)) ** 2) + 1e-9))
    return dict(IA=ia, R2=r2, RMSE=rmse, MAPE=mape)


def fig_metric_maps(df):
    """Chen-Fig.5-style per-station metric maps on the Kolkata boundary."""
    npz = C.RESULTS_DIR / "test_predictions.npz"
    if not npz.exists():
        return
    import matplotlib.gridspec as gridspec
    d = np.load(npz, allow_pickle=True)
    P, Y, M = d["P"], d["Y"], d["M"]
    stations = list(d["stations"]); horizons = list(d["horizons"])
    coords = df.groupby("Station").agg(lat=("Latitude", "mean"),
                                       lon=("Longitude", "mean"),
                                       pm=("Ground_PM2.5", "mean"))
    lon = np.array([coords.loc[s, "lon"] for s in stations])
    lat = np.array([coords.loc[s, "lat"] for s in stations])
    obs_pm = np.array([coords.loc[s, "pm"] for s in stations])

    # boundary (optional)
    boundary = None
    geo = C.ROOT / "geodata" / "kolkata.geojson"
    if geo.exists():
        try:
            import geopandas as gpd
            boundary = gpd.read_file(geo)
        except Exception:
            boundary = None

    def hidx(h):
        return horizons.index(h) if h in horizons else 0
    h_short = 1 if 1 in horizons else horizons[0]
    h_long = 24 if 24 in horizons else horizons[-1]

    # per-station metric arrays for both horizons
    def arrs(hi):
        out = {k: np.full(len(stations), np.nan) for k in ("IA", "R2", "RMSE", "MAPE")}
        for si in range(len(stations)):
            m = M[:, si, hi]
            if m.sum() < 5:
                continue
            mm = _stmetric(P[:, si, hi][m], Y[:, si, hi][m])
            for k in out:
                out[k][si] = mm[k]
        return out
    A_s, A_l = arrs(hidx(h_short)), arrs(hidx(h_long))

    metrics = [("IA", "IA", "RdYlGn"), ("R2", "R²", "RdYlGn"),
               ("RMSE", "RMSE", "YlOrRd"), ("MAPE", "MAPE (%)", "YlOrRd")]
    pad = 0.015
    xlim = (lon.min() - pad, lon.max() + pad)
    ylim = (lat.min() - pad, lat.max() + pad)
    if boundary is not None:
        bx = boundary.total_bounds
        xlim = (min(xlim[0], bx[0]), max(xlim[1], bx[2]))
        ylim = (min(ylim[0], bx[1]), max(ylim[1], bx[3]))

    # shared colour range per metric across both horizons
    vr = {}
    for key, _, _ in metrics:
        both = np.concatenate([A_s[key], A_l[key]])
        both = both[~np.isnan(both)]
        vr[key] = (float(both.min()), float(both.max()))

    # ordered, fully-filled 3x3 panel list (8 metric maps + observed mean)
    panels = []
    for key, lab, cmap in metrics:
        panels.append((A_s[key], cmap, vr[key][0], vr[key][1], f"{lab}, T+{h_short}h"))
    for key, lab, cmap in metrics:
        panels.append((A_l[key], cmap, vr[key][0], vr[key][1], f"{lab}, T+{h_long}h"))
    panels.append((obs_pm, "Spectral_r", float(obs_pm.min()), float(obs_pm.max()),
                   "Obs. mean PM$_{2.5}$"))

    letters = "abcdefghi"
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.6))
    axes = axes.ravel()
    for idx, (vals, cmap, vmin, vmax, title) in enumerate(panels):
        ax = axes[idx]
        if boundary is not None:
            boundary.plot(ax=ax, facecolor="#f4f4f4", edgecolor="#555",
                          linewidth=0.6, zorder=1)
        sc = ax.scatter(lon, lat, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=140, edgecolor="k", linewidth=0.7, zorder=5)
        ax.set_aspect("equal"); ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(f"({letters[idx]}) {title}", fontsize=9)
        ax.tick_params(labelsize=6); ax.grid(alpha=0.2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)
    fig.suptitle("GNN-LSTM per-station skill across Kolkata "
                 f"(T+{h_short} h and T+{h_long} h)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "F15_metric_maps", tight=False)


def _metrics():
    p = C.RESULTS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save(fig, name, tight=True):
    path = OUT / f"{name}.png"
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def main():
    set_style()
    # remove the now-unwanted correlation heatmap if present
    old = OUT / "F5_correlation_heatmap.png"
    if old.exists():
        old.unlink()
        print("  removed F5_correlation_heatmap.png")

    print("[viz] EDA figures ...")
    df = _load_df()
    # NOTE: F1 station map, F4 contour, F8 night-light scatter were removed by
    # the user and are intentionally NOT regenerated.
    fig_architecture()
    fig_seasonality(df)
    fig_diurnal(df)
    fig_annual_cycle(df)
    fig_nightlight_spatiotemporal(df)
    fig_sat_vs_ground(df)
    fig_daily_spatial_evolution(df)
    fig_spatiotemporal_eof(df)
    fig_monthly_satellite_field(
        df, "AOD", "YlOrRd", "Aerosol Optical Depth (AOD)",
        "F18_monthly_aod", "Monthly satellite AOD over Kolkata (climatology)")
    fig_monthly_satellite_field(
        df, "Night_Light", "inferno", "Night-time light radiance",
        "F19_monthly_nightlight", "Monthly satellite night-time lights over Kolkata (climatology)")
    print("[viz] result figures ...")
    fig_model_comparison()
    fig_error_vs_horizon()
    fig_pred_vs_obs_all()
    times = np.sort(df["datetime_IST"].unique())
    fig_trend_prediction(times)
    fig_seasonal_prediction(times)
    fig_metric_maps(df)
    print("[viz] done ->", OUT)


if __name__ == "__main__":
    main()
