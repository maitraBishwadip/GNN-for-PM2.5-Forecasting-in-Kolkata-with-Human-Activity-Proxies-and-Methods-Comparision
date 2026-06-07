"""Data loading, feature engineering, dynamic graph construction, and splitting.

Produces dense panel tensors of shape [T, N, F] plus a per-hour directed
adjacency [T, N, N], and block-based train/val/test window indices that
guarantee the test set is never-seen and spans all stations and all seasons.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as C

# Continuous features that get standardised (mean/std from TRAIN only).
CONT_FEATURES = [
    "PM25", "Temperature", "RH", "Wind_Speed", "wind_u", "wind_v",
    "PBLH", "Rainfall_mm", "Cloud_Cover", "AOD", "Night_Light",
    "gpm_lag", "lat", "lon",
]
# Flag / cyclical features passed through unscaled.
PASS_FEATURES = [
    "AOD_missing", "gpm_missing",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
]
FEATURES = CONT_FEATURES + PASS_FEATURES


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _bearing(lat1, lon1, lat2, lon2):
    """Initial bearing (deg, 0=N, clockwise) from point 1 to point 2."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def build_panel():
    """Return dict with panel arrays, time index, coords, and split labels."""
    df = pd.read_csv(C.CSV_PATH, parse_dates=["datetime_IST"])
    stations = C.STATIONS
    times = np.sort(df["datetime_IST"].unique())
    T, N = len(times), len(stations)
    t_index = {t: i for i, t in enumerate(times)}
    s_index = {s: i for i, s in enumerate(stations)}

    raw_cols = ["PM25", "Temperature", "RH", "Wind_Speed", "Wind_Direction",
                "PBLH", "Rainfall_mm", "Cloud_Cover", "AOD", "Night_Light",
                "Ground_PM2.5", "Latitude", "Longitude"]
    panel = {c: np.full((T, N), np.nan, dtype=np.float64) for c in raw_cols}
    ti = df["datetime_IST"].map(t_index).to_numpy()
    si = df["Station"].map(s_index).to_numpy()
    for c in raw_cols:
        panel[c][ti, si] = df[c].to_numpy()

    coords = np.zeros((N, 2))
    for s, i in s_index.items():
        coords[i, 0] = np.nanmean(panel["Latitude"][:, i])
        coords[i, 1] = np.nanmean(panel["Longitude"][:, i])

    # ---- feature engineering ----
    feat = {}
    feat["PM25"] = panel["PM25"]
    feat["Temperature"] = panel["Temperature"]
    feat["RH"] = panel["RH"]
    feat["Wind_Speed"] = panel["Wind_Speed"]
    wd = np.nan_to_num(panel["Wind_Direction"], nan=0.0)
    ws = np.nan_to_num(panel["Wind_Speed"], nan=0.0)
    feat["wind_u"] = ws * np.sin(np.radians(wd))
    feat["wind_v"] = ws * np.cos(np.radians(wd))
    feat["PBLH"] = panel["PBLH"]
    feat["Rainfall_mm"] = panel["Rainfall_mm"]
    feat["Cloud_Cover"] = panel["Cloud_Cover"]
    aod = panel["AOD"]
    feat["AOD_missing"] = np.isnan(aod).astype(np.float64)
    feat["AOD"] = _ffill_bfill(aod)
    feat["Night_Light"] = panel["Night_Light"]

    gpm = panel["Ground_PM2.5"]
    feat["gpm_missing"] = np.isnan(gpm).astype(np.float64)
    feat["gpm_lag"] = _ffill_bfill(gpm)   # past ground PM used autoregressively
    feat["lat"] = np.broadcast_to(coords[:, 0], (T, N)).copy()
    feat["lon"] = np.broadcast_to(coords[:, 1], (T, N)).copy()

    # temporal encodings (same across nodes)
    tt = pd.DatetimeIndex(times)
    hour = tt.hour.to_numpy()[:, None].astype(float)
    dow = tt.dayofweek.to_numpy()[:, None].astype(float)
    month = tt.month.to_numpy()[:, None].astype(float)
    feat["hour_sin"] = np.broadcast_to(np.sin(2 * np.pi * hour / 24), (T, N)).copy()
    feat["hour_cos"] = np.broadcast_to(np.cos(2 * np.pi * hour / 24), (T, N)).copy()
    feat["dow_sin"] = np.broadcast_to(np.sin(2 * np.pi * dow / 7), (T, N)).copy()
    feat["dow_cos"] = np.broadcast_to(np.cos(2 * np.pi * dow / 7), (T, N)).copy()
    feat["month_sin"] = np.broadcast_to(np.sin(2 * np.pi * month / 12), (T, N)).copy()
    feat["month_cos"] = np.broadcast_to(np.cos(2 * np.pi * month / 12), (T, N)).copy()

    # any remaining NaNs in continuous predictors -> column ffill/bfill then 0
    for c in CONT_FEATURES:
        feat[c] = np.nan_to_num(_ffill_bfill(feat[c]), nan=0.0)

    X = np.stack([feat[c] for c in FEATURES], axis=-1)   # [T, N, F]
    target = gpm.copy()                                  # [T, N] (NaN = missing)
    target_mask = ~np.isnan(target)

    adj = _dynamic_adjacency(coords, panel["Wind_Direction"], times)  # [T, N, N]

    split_of_t = _block_split(times)

    return dict(
        X=X, target=target, target_mask=target_mask, adj=adj,
        times=times, months=tt.month.to_numpy(), coords=coords,
        stations=stations, split_of_t=split_of_t, T=T, N=N, F=X.shape[-1],
    )


def _ffill_bfill(a):
    """Forward then backward fill along time (axis 0) per column."""
    a = a.copy()
    for j in range(a.shape[1]):
        col = a[:, j]
        idx = np.where(~np.isnan(col))[0]
        if idx.size == 0:
            continue
        # forward fill
        last = np.nan
        for i in range(len(col)):
            if not np.isnan(col[i]):
                last = col[i]
            elif not np.isnan(last):
                col[i] = last
        # backward fill leading NaNs
        first = col[idx[0]]
        col[:idx[0]] = first
        a[:, j] = col
    return a


def _dynamic_adjacency(coords, wind_dir, times):
    """Per-hour directed edge weights [T, N, N] from distance + wind alignment."""
    N = len(coords)
    dist = np.zeros((N, N))
    brng = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dist[i, j] = _haversine(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
            brng[i, j] = _bearing(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
    inv = np.zeros((N, N))
    nz = dist > 0
    inv[nz] = 1.0 / dist[nz]
    inv = inv / (inv.max() + 1e-9)            # normalise inverse-distance to [0,1]

    T = len(times)
    W = np.zeros((T, N, N), dtype=np.float32)
    wd = wind_dir  # [T, N] degrees (may be NaN)
    for t in range(T):
        for i in range(N):
            di = wd[t, i]
            for j in range(N):
                if i == j:
                    W[t, i, j] = 1.0       # self-loop
                    continue
                if np.isnan(di):
                    align = 0.5            # no wind info -> distance only
                else:
                    ang = np.radians(abs(((di - brng[i, j]) + 180) % 360 - 180))
                    align = 0.5 + 0.5 * max(0.0, np.cos(ang))
                W[t, i, j] = inv[i, j] * align
    return W


def _block_split(times):
    """Assign each timestamp a split label by cycling BLOCK_DAYS-day blocks."""
    t0 = times[0]
    hours = (times - t0) / np.timedelta64(1, "h")
    block = (hours // (C.BLOCK_DAYS * 24)).astype(int)
    cyc = C.SPLIT_CYCLE
    return np.array([cyc[b % len(cyc)] for b in block])


def make_window_indices(split_of_t, target_mask, split):
    """End-indices t of windows fully inside `split` with >=1 observed target."""
    T = len(split_of_t)
    H, maxh = C.HIST_LEN, C.MAX_H
    horizons = np.array(C.HORIZONS)
    ends = []
    for t in range(H - 1, T - maxh):
        span = split_of_t[t - H + 1: t + maxh + 1]
        if not np.all(span == split):
            continue
        if target_mask[t + horizons].any():
            ends.append(t)
    return np.array(ends, dtype=np.int64)


class WindowDataset(Dataset):
    """Lazily slices windows from the standardised full panel."""

    def __init__(self, data, ends, mu, sd, y_mu, y_sd):
        self.X = data["X"]
        self.adj = data["adj"]
        self.target = data["target"]
        self.tmask = data["target_mask"]
        self.ends = ends
        self.mu, self.sd = mu, sd
        self.y_mu, self.y_sd = y_mu, y_sd
        self.H = C.HIST_LEN
        self.hor = np.array(C.HORIZONS)

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, idx):
        t = self.ends[idx]
        xin = (self.X[t - self.H + 1: t + 1] - self.mu) / self.sd       # [H,N,F]
        ain = self.adj[t - self.H + 1: t + 1]                            # [H,N,N]
        yt = self.target[t + self.hor]                                   # [n_hor,N]
        ym = self.tmask[t + self.hor]                                    # [n_hor,N]
        y = np.nan_to_num(yt, nan=0.0)
        y_std = (y - self.y_mu) / self.y_sd
        return (
            torch.tensor(xin, dtype=torch.float32),
            torch.tensor(ain, dtype=torch.float32),
            torch.tensor(y_std.T, dtype=torch.float32),     # [N, n_hor]
            torch.tensor(ym.T, dtype=torch.float32),        # [N, n_hor]
        )


def compute_norm_stats(data, train_ends):
    """Feature mean/std and target mean/std over the TRAIN windows' history span."""
    H = C.HIST_LEN
    cont_idx = [FEATURES.index(c) for c in CONT_FEATURES]
    # sample feature stats from all train history timesteps
    mask = np.zeros(data["T"], dtype=bool)
    for t in train_ends:
        mask[t - H + 1: t + 1] = True
    Xtr = data["X"][mask]                       # [*, N, F]
    mu = Xtr.reshape(-1, data["F"]).mean(0)
    sd = Xtr.reshape(-1, data["F"]).std(0) + 1e-6
    # keep pass-through features unscaled
    keep = np.ones(data["F"], dtype=bool)
    keep[cont_idx] = False
    mu[keep] = 0.0
    sd[keep] = 1.0

    # target stats from observed train targets
    horizons = np.array(C.HORIZONS)
    ys = []
    for t in train_ends:
        yt = data["target"][t + horizons]
        m = data["target_mask"][t + horizons]
        ys.append(yt[m])
    ys = np.concatenate(ys)
    y_mu, y_sd = float(ys.mean()), float(ys.std() + 1e-6)
    return mu.astype(np.float32), sd.astype(np.float32), y_mu, y_sd
