"""Dataset loading, chronological splitting and feature selection."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class Dataset:
    name: str
    X: np.ndarray            # (n, D) float features, chronological
    y: np.ndarray            # (n,) int 0/1 label
    file_id: np.ndarray      # (n,) int, which source file a row came from
    event_id: np.ndarray     # (n,) int, anomaly event id (-1 = normal)
    features: list
    cfg: dict
    notes: list = field(default_factory=list)


def _sort_key(p: Path):
    s = str(p).lower()
    group = 0 if ("anomaly-free" in s or "anomaly_free" in s) else 1
    num = int(p.stem) if p.stem.isdigit() else 10 ** 9
    return (group, p.parent.name, num, p.stem)


def _read_csv(path, sep):
    df = pd.read_csv(path, sep=sep)
    if df.shape[1] == 1:                      # wrong separator guess
        df = pd.read_csv(path, sep="," if sep == ";" else ";")
    return df


def _load_raw(cfg, notes):
    path = Path(cfg["path"])
    if cfg["kind"] == "folder":
        files = sorted(path.rglob("*.csv"), key=_sort_key)
        if not files:
            raise FileNotFoundError(
                f"No CSV files found under {path}. Put the unzipped dataset there (see README).")
        frames = []
        for f in files:
            df = _read_csv(f, cfg["sep"])
            if cfg["time_col"] not in df.columns:
                notes.append(f"skipped {f.name}: no '{cfg['time_col']}' column")
                continue
            df[cfg["time_col"]] = pd.to_datetime(df[cfg["time_col"]], errors="coerce")
            df = df.sort_values(cfg["time_col"], kind="stable")
            if cfg["label_col"] not in df.columns:
                df[cfg["label_col"]] = 0
                notes.append(f"{f.name}: no label column, treated as all-normal")
            df["file_id"] = len(frames)
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        notes.append(f"read {len(frames)} files from {path}")
    else:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Put the CSV there (see README).")
        df = _read_csv(path, cfg["sep"])
        if cfg["time_col"] in df.columns:
            df[cfg["time_col"]] = pd.to_datetime(df[cfg["time_col"]], errors="coerce")
            df = df.sort_values(cfg["time_col"], kind="stable").reset_index(drop=True)
        else:
            notes.append("no time column: using row order as time")
        df["file_id"] = 0
        notes.append(f"read {len(df)} rows from {path}")
    if cfg.get("max_rows"):
        df = df.iloc[: cfg["max_rows"]].reset_index(drop=True)
        notes.append(f"capped at the first {cfg['max_rows']} rows (chronological)")
    return df


def load_dataset(name, cfg, max_rows_override=None):
    cfg = dict(cfg)
    if max_rows_override:
        cfg["max_rows"] = min(max_rows_override, cfg["max_rows"] or max_rows_override)
    notes = []
    df = _load_raw(cfg, notes)

    if cfg["label_col"] not in df.columns:
        raise ValueError(f"Label column '{cfg['label_col']}' not in columns: {list(df.columns)}")
    y = df[cfg["label_col"]].isin(cfg["label_positive"]).astype(int).to_numpy()
    file_id = df["file_id"].to_numpy().astype(int)

    banned = set(cfg["never_features"]) | {cfg["time_col"], cfg["label_col"]}
    if cfg.get("feature_cols"):
        cols = [c for c in cfg["feature_cols"] if c in df.columns]
        if len(cols) < len(cfg["feature_cols"]):
            notes.append(f"missing configured features: {sorted(set(cfg['feature_cols']) - set(cols))}")
        if len(cols) < 2:
            cols = None
    else:
        cols = None
    if cols is None:
        cols = [c for c in df.columns
                if c not in banned and pd.api.types.is_numeric_dtype(df[c])]
        notes.append("features auto-detected (numeric columns)")
    cols = [c for c in cols if c not in banned]

    feat = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = [c for c in cols if feat[c].isna().mean() <= 0.30]
    if len(keep) < len(cols):
        notes.append(f"dropped >30% NaN columns: {sorted(set(cols) - set(keep))}")
    feat = feat[keep].ffill().bfill()
    nz = [c for c in keep if feat[c].std() > 0]
    if len(nz) < len(keep):
        notes.append(f"dropped constant columns: {sorted(set(keep) - set(nz))}")
    feat = feat[nz]
    X = feat.to_numpy(dtype=float)

    prev = np.r_[0, y[:-1]]
    prev_file = np.r_[-1, file_id[:-1]]
    start = (y == 1) & ((prev == 0) | (file_id != prev_file))
    event_id = np.where(y == 1, np.cumsum(start) - 1, -1)

    notes.append(f"{len(X)} rows, {X.shape[1]} features, anomaly rate {y.mean():.2%}, "
                 f"{int(start.sum())} anomaly events")
    return Dataset(name, X, y, file_id, event_id, list(nz), cfg, notes)


def make_splits(ds: Dataset, fracs):
    """Chronological S / V / T index arrays. Splits on file boundaries if configured."""
    n = len(ds.y)
    if ds.cfg.get("split_by_file") and len(np.unique(ds.file_id)) >= 5:
        files = np.unique(ds.file_id)
        nf = len(files)
        a = max(1, int(round(fracs[0] * nf)))
        b = max(a + 1, int(round((fracs[0] + fracs[1]) * nf)))
        b = min(b, nf - 1)
        S = np.where(ds.file_id < files[a])[0]
        V = np.where((ds.file_id >= files[a]) & (ds.file_id < files[b]))[0]
        T = np.where(ds.file_id >= files[b])[0]
    else:
        a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
        S, V, T = np.arange(0, a), np.arange(a, b), np.arange(b, n)
    for nm, idx in (("S", S), ("V", V), ("T", T)):
        if len(idx) == 0:
            raise ValueError(f"Split {nm} is empty; check the dataset size / SPLIT setting.")
    ev_S = np.unique(ds.event_id[S][ds.y[S] == 1])
    if len(ev_S) < 2 or ds.y[S].sum() < 20:
        raise ValueError(
            f"Stream S has only {len(ev_S)} anomaly events / {int(ds.y[S].sum())} anomalous rows. "
            "Change SPLIT or the dataset: the stream needs >=2 events and >=20 anomalous rows.")
    return dict(S=S, V=V, T=T)


def select_features(XS, k):
    """Label-free feature selection on S only: keep the k channels with the heaviest tails
    (std / IQR). Returns column indices."""
    D = XS.shape[1]
    if D <= k:
        return np.arange(D)
    q75, q25 = np.percentile(XS, [75, 25], axis=0)
    score = XS.std(axis=0) / (q75 - q25 + 1e-9)
    return np.sort(np.argsort(-score)[:k])
