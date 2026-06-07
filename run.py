"""Entry point: train all models, evaluate on the never-seen test set, and
document results for the write-up.

Usage:
    python run.py                 # full run (all models)
    python run.py --quick         # smaller/faster run for a smoke test
    python run.py --models gnn    # subset: gnn, lstm, gbr, persistence

Outputs:
    results/metrics.json          # all metrics, every model & breakdown
    results/RESULTS.md            # human-readable results report for the paper
    results/test_predictions.npz  # GNN-LSTM test preds for visualisation
    results/split_summary.json    # proof the test spans all stations & seasons
"""
import argparse
import json
import time

import numpy as np
import torch

from models import config as C
from models import data as D
from models import engine as E
from models import baselines as B
from models.model import GNNLSTM, LSTMOnly, PureSTGNN


def jdefault(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--models", nargs="*",
                    default=["persistence", "gbr", "lstm", "stgnn", "gnn"])
    args = ap.parse_args()

    if args.quick:
        C.EPOCHS, C.PATIENCE = 3, 2

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print(f"[setup] device={device}  epochs={C.EPOCHS}")

    print("[data] building panel ...")
    data = D.build_panel()
    print(f"[data] T={data['T']} N={data['N']} F={data['F']} features={D.FEATURES}")

    tr_ends = D.make_window_indices(data["split_of_t"], data["target_mask"], "train")
    va_ends = D.make_window_indices(data["split_of_t"], data["target_mask"], "val")
    te_ends = D.make_window_indices(data["split_of_t"], data["target_mask"], "test")
    print(f"[split] windows  train={len(tr_ends)} val={len(va_ends)} test={len(te_ends)}")

    # ---- prove the test set spans all stations & all seasons ----
    te_months = data["months"][te_ends]
    te_seasons = sorted({C.SEASONS[m] for m in te_months})
    # stations present in test windows with >=1 observed target
    hor = np.array(C.HORIZONS)
    st_present = set()
    season_counts = {}
    for t in te_ends:
        m = data["target_mask"][t + hor]            # [n_hor, N]
        for si in range(data["N"]):
            if m[:, si].any():
                st_present.add(data["stations"][si])
        s = C.SEASONS[data["months"][t]]
        season_counts[s] = season_counts.get(s, 0) + 1
    split_summary = dict(
        test_windows=int(len(te_ends)),
        test_seasons=te_seasons,
        test_season_window_counts=season_counts,
        test_stations_with_targets=sorted(st_present),
        n_test_stations=len(st_present),
        block_days=C.BLOCK_DAYS, split_cycle=C.SPLIT_CYCLE,
    )
    (C.RESULTS_DIR / "split_summary.json").write_text(json.dumps(split_summary, indent=2))
    print(f"[split] test seasons = {te_seasons}")
    print(f"[split] test stations w/ targets = {len(st_present)}/{data['N']}")
    assert len(st_present) == data["N"], "test must cover all stations"
    assert len(te_seasons) == 4, "test must cover all four seasons"

    mu, sd, y_mu, y_sd = D.compute_norm_stats(data, tr_ends)
    print(f"[norm] target mean={y_mu:.2f} std={y_sd:.2f}")

    train_ds = D.WindowDataset(data, tr_ends, mu, sd, y_mu, y_sd)
    val_ds = D.WindowDataset(data, va_ends, mu, sd, y_mu, y_sd)
    test_ds = D.WindowDataset(data, te_ends, mu, sd, y_mu, y_sd)

    # merge into any existing metrics so single-model runs join the same table
    metrics_path = C.RESULTS_DIR / "metrics.json"
    results = {}
    if metrics_path.exists():
        try:
            results = json.loads(metrics_path.read_text())
        except Exception:
            results = {}
    saved_gnn_pred = None

    # ---------- persistence ----------
    if "persistence" in args.models:
        print("[model] persistence ...")
        P, Y, M = B.persistence(data, te_ends)
        results["Persistence"] = E.evaluate_all(P, Y, M, te_months, data["stations"])

    # ---------- gradient-boosted trees ----------
    if "gbr" in args.models:
        print("[model] HistGBR ...")
        P, Y, M = B.hist_gbr(data, tr_ends, te_ends)
        results["HistGBR"] = E.evaluate_all(P, Y, M, te_months, data["stations"])

    # ---------- LSTM-only (no graph) ----------
    if "lstm" in args.models:
        print("[model] LSTM-only (ablation: no graph) ...")
        m = LSTMOnly(data["F"], len(C.HORIZONS))
        m = E.train_model(m, train_ds, val_ds, device, C.CKPT_DIR / "lstm.pt")
        P, Y, M = E.collect_predictions(m, test_ds, device, y_mu, y_sd)
        results["LSTM"] = E.evaluate_all(P, Y, M, te_months, data["stations"])

    # ---------- Pure spatio-temporal GNN (no LSTM) ----------
    if "stgnn" in args.models:
        print("[model] Pure ST-GNN (no LSTM) ...")
        m = PureSTGNN(data["F"], len(C.HORIZONS))
        n_par = sum(p.numel() for p in m.parameters())
        print(f"  parameters: {n_par:,}")
        m = E.train_model(m, train_ds, val_ds, device, C.CKPT_DIR / "stgnn.pt")
        P, Y, M = E.collect_predictions(m, test_ds, device, y_mu, y_sd)
        results["Pure-STGNN"] = E.evaluate_all(P, Y, M, te_months, data["stations"])

    # ---------- GNN-LSTM (proposed) ----------
    if "gnn" in args.models:
        print("[model] GNN-LSTM (proposed) ...")
        m = GNNLSTM(data["F"], len(C.HORIZONS))
        n_par = sum(p.numel() for p in m.parameters())
        print(f"  parameters: {n_par:,}")
        m = E.train_model(m, train_ds, val_ds, device, C.CKPT_DIR / "gnn.pt")
        P, Y, M = E.collect_predictions(m, test_ds, device, y_mu, y_sd)
        results["GNN-LSTM"] = E.evaluate_all(P, Y, M, te_months, data["stations"])
        saved_gnn_pred = (P, Y, M)
        np.savez_compressed(
            C.RESULTS_DIR / "test_predictions.npz",
            P=P, Y=Y, M=M, te_ends=te_ends, te_months=te_months,
            stations=np.array(data["stations"]), horizons=np.array(C.HORIZONS),
        )

    (C.RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=jdefault))
    write_report(results, split_summary, time.time() - t0)
    print(f"[done] {time.time() - t0:.1f}s  -> results/RESULTS.md")


def write_report(results, split, secs):
    L = []
    L.append("# Kolkata PM2.5 GNN Forecasting — Results\n")
    L.append("_Auto-generated by `run.py`. Test set is strictly never-seen data "
             "(block split) covering all 7 stations and all 4 seasons._\n")

    L.append("## Test-set coverage (never-seen data)\n")
    L.append(f"- Test windows: **{split['test_windows']}**")
    L.append(f"- Seasons in test: **{', '.join(split['test_seasons'])}**")
    L.append(f"- Stations with targets in test: **{split['n_test_stations']}/7**")
    L.append(f"- Window counts per season: {split['test_season_window_counts']}\n")

    pref = ["Persistence", "HistGBR", "LSTM", "Pure-STGNN", "GNN-LSTM"]
    names = [n for n in pref if n in results] + [n for n in results if n not in pref]

    L.append("## Overall test performance (all horizons pooled)\n")
    L.append("| Model | MAE | RMSE | MAPE(%) | R² | IA |")
    L.append("|---|---|---|---|---|---|")
    for name in names:
        o = results[name]["overall"]
        L.append(f"| {name} | {o['MAE']:.2f} | {o['RMSE']:.2f} | {o['MAPE']:.1f} "
                 f"| {o['R2']:.3f} | {o['IA']:.3f} |")
    L.append("")

    for gname in [n for n in ("GNN-LSTM", "Pure-STGNN") if n in results]:
        g = results[gname]
        L.append(f"## {gname} — error vs forecast horizon\n")
        L.append("| Horizon | MAE | RMSE | MAPE(%) | R² | IA |")
        L.append("|---|---|---|---|---|---|")
        for h, m in g["per_horizon"].items():
            L.append(f"| {h} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['MAPE']:.1f} "
                     f"| {m['R2']:.3f} | {m['IA']:.3f} |")
        L.append("")
        L.append(f"## {gname} — per-station test performance\n")
        L.append("| Station | MAE | RMSE | R² | IA | n |")
        L.append("|---|---|---|---|---|---|")
        for s, m in g["per_station"].items():
            L.append(f"| {s} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.3f} "
                     f"| {m['IA']:.3f} | {m['n']} |")
        L.append("")
        L.append(f"## {gname} — per-season test performance\n")
        L.append("| Season | MAE | RMSE | R² | IA | n |")
        L.append("|---|---|---|---|---|---|")
        for s, m in g["per_season"].items():
            L.append(f"| {s} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.3f} "
                     f"| {m['IA']:.3f} | {m['n']} |")
        L.append("")
        L.append(f"## {gname} — pollution-event detection (PM2.5 ≥ "
                 f"{C.EVENT_THRESHOLD:.0f} µg/m³)\n")
        L.append("| Horizon | Recall Rate(%) | False Alarm Rate(%) | events |")
        L.append("|---|---|---|---|")
        for h, m in g["events"].items():
            L.append(f"| {h} | {m['RR']:.1f} | {m['FAR']:.1f} | {m['n_events']} |")
        L.append("")

    L.append(f"\n_Run time: {secs:.1f}s._\n")
    (C.RESULTS_DIR / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
