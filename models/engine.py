"""Training loop, prediction collection, and evaluation metrics."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as C
from .model import masked_huber


def _loader(ds, shuffle):
    return DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=shuffle, num_workers=0)


def train_model(model, train_ds, val_ds, device, ckpt_path, log=print):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.EPOCHS)
    tl, vl = _loader(train_ds, True), _loader(val_ds, False)
    best, bad = float("inf"), 0
    for ep in range(1, C.EPOCHS + 1):
        model.train()
        tr = 0.0
        for x, a, y, m in tl:
            x, a, y, m = x.to(device), a.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            loss = masked_huber(model(x, a), y, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr += loss.item() * x.size(0)
        tr /= len(train_ds)
        sched.step()
        model.eval()
        va = 0.0
        with torch.no_grad():
            for x, a, y, m in vl:
                x, a, y, m = x.to(device), a.to(device), y.to(device), m.to(device)
                va += masked_huber(model(x, a), y, m).item() * x.size(0)
        va /= len(val_ds)
        log(f"  epoch {ep:02d}  train {tr:.4f}  val {va:.4f}")
        if va < best - 1e-4:
            best, bad = va, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= C.PATIENCE:
                log(f"  early stop at epoch {ep} (best val {best:.4f})")
                break
    model.load_state_dict(torch.load(ckpt_path))
    return model


@torch.no_grad()
def collect_predictions(model, ds, device, y_mu, y_sd):
    """Return preds, targets, masks in ORIGINAL ug/m3 units: [S, N, n_hor]."""
    model.eval()
    P, Y, M = [], [], []
    for x, a, y, m in _loader(ds, False):
        x, a = x.to(device), a.to(device)
        p = model(x, a).cpu().numpy()
        P.append(p)
        Y.append(y.numpy())
        M.append(m.numpy())
    P = np.concatenate(P) * y_sd + y_mu
    Y = np.concatenate(Y) * y_sd + y_mu
    M = np.concatenate(M).astype(bool)
    return P, Y, M


# ---------- metrics ----------
def _metrics(pred, true):
    e = pred - true
    mae = np.mean(np.abs(e))
    rmse = np.sqrt(np.mean(e ** 2))
    denom = np.clip(np.abs(true), 1.0, None)
    mape = np.mean(np.abs(e) / denom) * 100
    ss_res = np.sum(e ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2) + 1e-9
    r2 = 1 - ss_res / ss_tot
    o_mean = true.mean()
    ia = 1 - ss_res / (np.sum((np.abs(pred - o_mean) + np.abs(true - o_mean)) ** 2) + 1e-9)
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2, IA=ia, n=int(true.size))


def evaluate_all(P, Y, M, months_per_sample, station_names):
    """Full metric breakdown: overall, per-horizon, per-station, per-season, events."""
    horizons = C.HORIZONS
    out = {}
    p, y = P[M], Y[M]
    out["overall"] = _metrics(p, y)

    out["per_horizon"] = {}
    for hi, h in enumerate(horizons):
        m = M[:, :, hi]
        out["per_horizon"][f"T+{h}"] = _metrics(P[:, :, hi][m], Y[:, :, hi][m])

    out["per_station"] = {}
    for si, s in enumerate(station_names):
        m = M[:, si, :]
        if m.any():
            out["per_station"][s] = _metrics(P[:, si, :][m], Y[:, si, :][m])

    out["per_season"] = {}
    seasons = np.array([C.SEASONS[mo] for mo in months_per_sample])  # per-sample (window end)
    for name in ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]:
        sel = seasons == name
        if not sel.any():
            continue
        m = M[sel]
        if m.any():
            out["per_season"][name] = _metrics(P[sel][m], Y[sel][m])

    # event detection (RR / FAR) per horizon at EVENT_THRESHOLD
    out["events"] = {}
    thr = C.EVENT_THRESHOLD
    for hi, h in enumerate(horizons):
        m = M[:, :, hi]
        pt, yt = P[:, :, hi][m], Y[:, :, hi][m]
        obs, pos = yt >= thr, pt >= thr
        tp = np.sum(obs & pos); fn = np.sum(obs & ~pos)
        fp = np.sum(~obs & pos); tn = np.sum(~obs & ~pos)
        rr = 100 * tp / (tp + fn) if (tp + fn) else float("nan")
        far = 100 * fp / (fp + tn) if (fp + tn) else float("nan")
        out["events"][f"T+{h}"] = dict(RR=rr, FAR=far, n_events=int(tp + fn))
    return out
