"""Online utility scorer (port of UtilityScorer from the notebook).

Utility of a sample x arriving at time t, with the buffer's current time `now`:
    U = alpha*recency + beta*anomaly + gamma*deviation + delta*rarity
recency   = exp(-(now - t) / tau)                       (computed by the buffer, changes with time)
anomaly   = tanh(max-channel robust z / 6)              (frozen at arrival)
deviation = tanh(max-channel |x - EMA| / robust scale / 6)   (frozen at arrival)
rarity    = 1 - exp(-mean kNN distance to a reference sample / 4)  (frozen at arrival)
Every channel is robust-standardised (median / MAD) first so the weights are comparable.
The scorer sees EVERY arrival (admitted or not) so its baseline tracks the stream, and it never sees labels.
"""
import numpy as np


class OnlineScorer:
    def __init__(self, D, window=1000, ema_alpha=0.01, knn=10, stride=8, refresh=10):
        self.W, self.ema_alpha, self.knn, self.stride, self.refresh = window, ema_alpha, knn, stride, refresh
        self.buf = np.empty((window, D))
        self.n = 0
        self.ema = None
        self.med = self.scale = self.ref = None

    def observe(self, x):
        """Update state with x and return its frozen components (anomaly, deviation, rarity)."""
        x = np.asarray(x, dtype=float)
        self.buf[self.n % self.W] = x
        self.n += 1
        self.ema = x.copy() if self.ema is None else (1 - self.ema_alpha) * self.ema + self.ema_alpha * x
        if self.n >= 30 and (self.med is None or self.n % self.refresh == 0):
            if self.n <= self.W:
                w = self.buf[: self.n]
            else:
                pos = self.n % self.W
                w = np.concatenate([self.buf[pos:], self.buf[:pos]])      # oldest -> newest
            self.med = np.median(w, axis=0)
            self.scale = 1.4826 * np.median(np.abs(w - self.med), axis=0) + 1e-6
            older = w[:-1][::-1][:: self.stride]                          # newest-first, excluding x itself
            self.ref = (older - self.med) / self.scale
        if self.med is None:
            return np.zeros(3)
        z = (x - self.med) / self.scale
        anomaly = np.tanh(np.abs(z).max() / 6.0)
        deviation = np.tanh((np.abs(x - self.ema) / self.scale).max() / 6.0)
        dist = np.linalg.norm(self.ref - z, axis=1)
        k = min(self.knn, len(dist))
        knn_d = np.partition(dist, k - 1)[:k].mean()
        rarity = 1.0 - np.exp(-knn_d / 4.0)
        return np.array([anomaly, deviation, rarity])


def precompute_components(X, **params):
    """Run the scorer over a stream once; row i = frozen (anomaly, deviation, rarity) of sample i."""
    sc = OnlineScorer(X.shape[1], **params)
    out = np.zeros((len(X), 3))
    for i in range(len(X)):
        out[i] = sc.observe(X[i])
    return out
