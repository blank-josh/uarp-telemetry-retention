"""All settings live here. Edit this file, not the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # <-- put your datasets under this folder
RESULTS_DIR = ROOT / "results"

# Which datasets to run. Start with ["skab"]; add "pump" once SKAB works.
DATASETS = ["skab"]

DATASET_CONFIGS = {
    # SKAB: put the unzipped Kaggle files anywhere under data/skab/ (CSVs are found recursively).
    "skab": dict(
        path=DATA_DIR / "skab", kind="folder", sep=";",
        time_col="datetime", label_col="anomaly", label_positive=[1],
        # Columns that must NEVER be used as features (label leakage).
        never_features=["anomaly", "changepoint", "file_id"],
        feature_cols=["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
                      "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"],
        max_features=12, max_rows=None,
        split_by_file=True,      # split S/V/T on file boundaries
    ),
    # Pump Sensor Data: put the Kaggle CSV at data/pump/sensor.csv
    "pump": dict(
        path=DATA_DIR / "pump" / "sensor.csv", kind="file", sep=",",
        time_col="timestamp", label_col="machine_status",
        label_positive=["BROKEN", "RECOVERING"],
        never_features=["Unnamed: 0", "machine_status", "timestamp", "file_id"],
        feature_cols=None,       # None = auto-select
        max_features=12, max_rows=60000,
        split_by_file=False,
    ),
}

# ---- Protocol ---------------------------------------------------------------
SPLIT = (0.60, 0.15, 0.25)           # stream S / validation V / test T (chronological)
RETENTION_RATIOS = [0.005, 0.01, 0.02, 0.05]   # buffer capacity K as a fraction of |S|
HEADLINE_RATIO = 0.02
N_SEEDS = 10
MAX_TRIM = 0.05                      # each seed trims 0-5% from the start of S (seed 0 = no trim)
MIN_K = 20

# ---- Utility function --------------------------------------------------------
DEFAULT_WEIGHTS = dict(alpha=0.30, beta=0.35, gamma=0.15, delta=0.20)   # recency, anomaly, deviation, rarity
STRATIFIED_WEIGHTS = dict(alpha=0.0, beta=0.40, gamma=0.20, delta=0.40)
STRATIFIED_BASE_FRAC = 0.6
SCORER = dict(window=1000, ema_alpha=0.01, knn=10, stride=8, refresh=10)
ARES_EPS = 0.05                      # floor on the A-Res sampling weight

# ---- Experiments -------------------------------------------------------------
N_TUNING_TRIALS = 40
Z_THR = 3.5                          # modified-z cut-off for the robust-z detector
IF_TREES = 100
RF_TREES = 100
TIMING_ARRIVALS = 5000               # arrivals used for the latency/memory measurement
SNAPSHOTS = 60                       # points on the live-recall curve
