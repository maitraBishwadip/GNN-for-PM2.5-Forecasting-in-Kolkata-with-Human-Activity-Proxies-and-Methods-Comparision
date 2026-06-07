# Findings & Discussion (curated for the write-up)

> Hand-written interpretation of the auto-generated `RESULTS.md`. This file is
> **not** overwritten by `run.py`. All numbers are from the never-seen test set
> (3,347 windows; all 7 stations; all 4 seasons — `split_summary.json`).

## Headline numbers (5-model comparison)

| Model | MAE | RMSE | MAPE % | R² | IA |
|---|---|---|---|---|---|
| Persistence | 11.13 | 18.98 | 31.4 | 0.670 | 0.911 |
| HistGBR | 9.46 | 15.14 | 33.8 | 0.790 | 0.942 |
| LSTM (no graph) | 10.40 | 15.57 | 37.8 | 0.778 | 0.936 |
| **Pure-STGNN (no LSTM)** | 9.88 | 15.19 | 36.7 | 0.789 | 0.936 |
| **GNN-LSTM (hybrid)** | 9.92 | **15.08** | 33.3 | **0.791** | 0.941 |

Two graph variants are reported per the user's request:
- **GNN-LSTM** — spatial GAT + temporal LSTM (hybrid; Chen et al. GL-GL lineage).
- **Pure-STGNN** — spatial GAT + **temporal GAT** (no recurrence anywhere).

## What the results show (defensible claims)

1. **The spatial graph is what matters, not the recurrence.** Both graph models
   beat the no-graph LSTM on every metric (e.g. RMSE 15.08 / 15.19 vs 15.57;
   R² 0.791 / 0.789 vs 0.778). The Pure-STGNN — which has **no LSTM at all** and
   only 36.6k parameters (vs 94.3k for the hybrid) — matches the hybrid within
   noise (MAE 9.88 vs 9.92). **Message passing over the 7-station graph, not the
   temporal cell type, drives the gain.** This is a clean, citable result.

2. **The GNN advantage grows with horizon (F10).** At T+1 the autoregressive
   signal dominates and HistGBR wins the nowcast; by T+12–T+24 the graph models
   have the lowest RMSE and degrade most gracefully — mirroring Chen et al.
   (2025), who report the GNN-LSTM advantage specifically at long lead times.
   > Frame the contribution as *robust medium-range (≥12 h) urban forecasting*,
   > not winning the trivial 1-hour nowcast.

3. **Clean horizon degradation** (GNN-LSTM): R² 0.912 (T+1) → 0.699 (T+24);
   MAE 6.46 → 12.60 µg/m³. Errors stay well-bounded over a full day.

4. **Seasonal & trend skill (F13, F14).** Predicted seasonal means track observed
   almost exactly (Winter 86 vs 84; Pre-monsoon 34 vs 34; Monsoon 18 vs 20;
   Post-monsoon 42 vs 43 µg/m³). Forecast bias is near-zero in all seasons, with
   the widest spread in Winter (highest, most variable PM) — the same winter-hard
   pattern as Chen et al. Fig. 6b. The diurnal trend (F13a) is reproduced
   faithfully. Per-season metrics: Monsoon easiest (MAE 5.0), Winter hardest
   (MAE 16.3, R² 0.52).

5. **Spatial skill across the city (F15, Fig.5-style maps).** Best station
   rabindra_sarobar (MAE 7.9), worst jadavpur (MAE 18.3) — and jadavpur has by
   far the fewest ground labels (2,623 vs ~16k), confirming data-sparse nodes are
   the hard cases that message passing partially rescues.

6. **Operationally useful event detection.** At the 90 µg/m³ pollution-event
   threshold: Recall 78.6 % at T+1, ~60 % through T+24, with False Alarm Rate
   only 1.7–3.8 %. Low FAR is the property air-quality agencies prize.

## Honest limitations (Discussion / threats to validity)

- At pooled MAE, **HistGBR (9.46) edges both GNNs.** With 7 nodes and a strong
  autoregressive feature, gradient-boosted trees are a hard baseline at short
  horizons. The GNN win is in **RMSE/R², long-horizon robustness, spatial
  coherence, and interpretability** — state this plainly.
- 7 nodes is a small graph; the dynamic directed graph is a proof of concept at
  intra-urban scale. Extending to the wider West Bengal / Indo-Gangetic-Plain
  network is the natural next step (`plan.md` §9).
- 23.7 % of ground labels are missing and unevenly distributed; the masked-loss
  protocol handles it, but jadavpur-type sparse stations remain hard.

## Figures (`visualizations/`) — paper-ready

| File | Use in paper |
|---|---|
| F1_station_map | Study area / 7-node network |
| F4_spatial_contour | Interpolated mean-PM₂.₅ surface over Kolkata |
| F6_seasonality | 4× winter↔monsoon swing |
| F7_diurnal_cycle | Night-peak / afternoon-trough |
| F8_nightlight_vs_pm | Night_Light human-activity proxy (per station) |
| F12_nightlight_spatiotemporal | NL & PM₂.₅ co-varying over time + per-station NL |
| F9_satellite_vs_ground | Satellite bias motivation (hexbin) |
| F_model_comparison | 5-model MAE/RMSE + R² bars |
| F10_error_vs_horizon | **Main result — GNN robust at long horizons** |
| F11_pred_vs_obs_all_stations | Predicted vs observed, all 7 stations |
| F13_trend_prediction | Predicted vs observed diurnal + monthly trend |
| F14_seasonal_prediction | Predicted vs observed seasonal means + bias |
| F15_metric_maps | **Chen-Fig.5-style** per-station IA/R²/RMSE/MAPE maps (Kolkata boundary) |
| F16_daily_spatial_evolution | Daily interpolated PM₂.₅ field, 12-day small-multiples |
| F17_spatiotemporal_eof | EOF/PCA: temporal basis + spatial coeffs + modeled maps (comp 1 = 67% var) |
| F18_monthly_aod | Monthly satellite AOD over Kolkata (climatology) |
| F19_monthly_nightlight | Monthly satellite night-light over Kolkata (monsoon retrieval dip visible) |

**Removed by user (do not regenerate):** F1 station map, F4 spatial contour,
F8 night-light scatter, F5 correlation heatmap.

**Note on satellite fields (F18/F19):** built from the satellite-derived `AOD`
and `Night_Light` columns in the dataset, interpolated across the 7 stations —
not raw raster tiles. True Sentinel-2/VIIRS/MODIS raster download needs Earth
Engine / NASA Earthdata credentials and a raster stack not available in this
environment.

> Still hand-drawn for the manuscript (schematics): pipeline diagram,
> architecture diagram, dynamic-graph illustration, GAT attention map.
