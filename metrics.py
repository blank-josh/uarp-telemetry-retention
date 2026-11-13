"""Retention-quality metrics (computed on S) and downstream metrics (trained on the buffer, scored on T/V)."""
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import config

_GRID = np.linspace(0.0, 1.0, 512)


class RetentionEvaluator:
    """Everything that depends only on the stream S is precomputed once."""
    def __init__(self, XS, yS, eventS):
        self.XS, self.yS, self.eventS = XS, yS, eventS
        self.n = len(XS)
        self.std = XS.std(axis=0) + 1e-9
        self.q_full = np.quantile(XS, _GRID, axis=0)                       # (512, D)
        self.tail = (XS > np.percentile(XS, 95, axis=0)).any(axis=1)
        self.events = np.unique(eventS[yS == 1])

    def score(self, idx):
        yS = self.yS
        ret = yS[idx]
        q = np.quantile(self.XS[idx], _GRID, axis=0)
        wass = float(np.mean(np.abs(q - self.q_full).mean(axis=0) / self.std))
        covered = np.unique(self.eventS[idx][ret == 1])
        return dict(
            anomaly_recall=float(ret.sum() / max(1, yS.sum())),
            event_coverage=float(len(covered) / max(1, len(self.events))),
            wasserstein=wass,
            tail_coverage=float(self.tail[idx].sum() / max(1, self.tail.sum())),
            median_staleness=float((self.n - 1) - np.median(idx)),
        )


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _ap(y, s):
    return float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")


def downstream(idx, XS, yS, XE, yE, seed):
    """Train three detectors on the retained samples, score on the evaluation segment XE/yE."""
    Xb, yb = XS[idx], yS[idx]
    sc = StandardScaler().fit(Xb)
    Xbs, XEs = sc.transform(Xb), sc.transform(XE)

    iso = IsolationForest(n_estimators=config.IF_TREES, random_state=seed).fit(Xbs)
    s_if = -iso.score_samples(XEs)

    if len(np.unique(yb)) == 2:                          # supervised: needs both classes in the buffer
        rf = RandomForestClassifier(n_estimators=config.RF_TREES, class_weight="balanced",
                                    n_jobs=-1, random_state=seed).fit(Xbs, yb)
        s_rf = rf.predict_proba(XEs)[:, 1]
    else:                                                # buffer has one class: constant score (chance level)
        s_rf = np.zeros(len(XE))

    med = np.median(Xb, axis=0)
    mad = 1.4826 * np.median(np.abs(Xb - med), axis=0) + 1e-6
    pred = (np.abs(XE - med) / mad).max(axis=1) > config.Z_THR

    return dict(if_auroc=_auc(yE, s_if), if_auprc=_ap(yE, s_if),
                rf_auroc=_auc(yE, s_rf), rf_auprc=_ap(yE, s_rf),
                z_f1=float(f1_score(yE, pred, zero_division=0)))
