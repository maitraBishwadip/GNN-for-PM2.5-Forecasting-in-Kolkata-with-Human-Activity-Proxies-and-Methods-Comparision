"""Central configuration for the Kolkata PM2.5 GNN forecasting project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "Kolkata_sat_ground.csv"
RESULTS_DIR = ROOT / "results"
VIZ_DIR = ROOT / "visualizations"
CKPT_DIR = RESULTS_DIR / "checkpoints"

for _d in (RESULTS_DIR, VIZ_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----- temporal windowing -----
HIST_LEN = 24                       # input history window (hours) = 1 diurnal cycle
HORIZONS = [1, 3, 6, 12, 24]        # forecast horizons (hours ahead)
MAX_H = max(HORIZONS)

# ----- train / val / test split (block-based, all-station & all-season coverage) -----
# The timeline is chopped into BLOCK_DAYS-day blocks; blocks are cycled into
# train/val/test so EVERY season appears in EVERY split. A forecast window is
# assigned to a split only if its ENTIRE input+horizon span lies in that split
# (straddling windows are dropped) -> the test set is strictly never-seen data.
BLOCK_DAYS = 15
SPLIT_CYCLE = ["train", "train", "train", "val", "test"]  # 60/20/20

# ----- target / event -----
TARGET = "Ground_PM2.5"
EVENT_THRESHOLD = 90.0              # ug/m3; pollution-event definition for RR/FAR

# ----- model hyperparameters -----
GAT_HIDDEN = 32
GAT_HEADS = 4
GAT_LAYERS = 2
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
DROPOUT = 0.15

# ----- training -----
EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
HUBER_DELTA = 1.0
SEED = 42

STATIONS = [
    "ballygunge", "bidhannagar", "fort_william", "jadavpur",
    "rabindra_bharati", "rabindra_sarobar", "victoria_memorial",
]

SEASONS = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
}
