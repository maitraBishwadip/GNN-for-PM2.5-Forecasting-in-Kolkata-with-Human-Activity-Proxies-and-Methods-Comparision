"""Non-graph baselines: persistence and gradient-boosted trees (per horizon).

Both consume the same windows as the GNN and are evaluated on the identical
never-seen test set, so the comparison table is apples-to-apples.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C
from .data import FEATURES


def _gather(data, ends):
    """Flatten windows -> (Xflat[S*N, H*F], Y[S,N,n_hor], M[S,N,n_hor], last_gpm)."""
    H = C.HIST_LEN
    hor = np.array(C.HORIZONS)
    X = data["X"]; tgt = data["target"]; tm = data["target_mask"]
    gpm_idx = FEATURES.index("gpm_lag")
    S, N = len(ends), data["N"]
    Xf = np.zeros((S, N, H * data["F"]), dtype=np.float32)
    last = np.zeros((S, N), dtype=np.float32)
    Y = np.zeros((S, N, len(hor)), dtype=np.float32)
    M = np.zeros((S, N, len(hor)), dtype=bool)
    for k, t in enumerate(ends):
        win = X[t - H + 1: t + 1]                 # [H,N,F]
        Xf[k] = win.transpose(1, 0, 2).reshape(N, -1)
        last[k] = win[-1, :, gpm_idx]
        Y[k] = tgt[t + hor].T
        M[k] = tm[t + hor].T
    return Xf, Y, M, last


def persistence(data, test_ends):
    """Predict last observed ground PM2.5 for every horizon."""
    _, Y, M, last = _gather(data, test_ends)
    P = np.repeat(last[:, :, None], len(C.HORIZONS), axis=2)
    return P, np.nan_to_num(Y), M


def hist_gbr(data, train_ends, test_ends, log=print):
    """One HistGradientBoostingRegressor per horizon, pooled over stations."""
    Xtr, Ytr, Mtr, _ = _gather(data, train_ends)
    Xte, Yte, Mte, _ = _gather(data, test_ends)
    Ntr = Xtr.reshape(-1, Xtr.shape[-1])
    Nte = Xte.reshape(-1, Xte.shape[-1])
    P = np.zeros_like(Yte)
    for hi, h in enumerate(C.HORIZONS):
        ytr = Ytr[:, :, hi].reshape(-1)
        mtr = Mtr[:, :, hi].reshape(-1)
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=8,
            l2_regularization=1.0, random_state=C.SEED)
        model.fit(Ntr[mtr], ytr[mtr])
        P[:, :, hi] = model.predict(Nte).reshape(Yte.shape[0], Yte.shape[1])
        log(f"  HistGBR fitted horizon T+{h}")
    return P, np.nan_to_num(Yte), Mte
