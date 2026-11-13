"""Bounded-buffer retention policies. Policies are label-free: observe(i, comp) only gets the arrival
position i (also used as its timestamp) and the frozen scorer components comp = (anomaly, deviation, rarity).
A buffer stores positions in the stream; indices() returns the retained positions."""
import heapq
from collections import deque
import numpy as np
import config


class Policy:
    name = "base"
    needs_comps = False

    def __init__(self, K, n=None, seed=0, **kw):
        self.K, self.n_total = int(K), n
        self.rng = np.random.default_rng(seed)

    def observe(self, i, comp):
        raise NotImplementedError

    def indices(self):
        raise NotImplementedError


class FIFO(Policy):
    name = "FIFO"
    def __init__(self, K, **kw):
        super().__init__(K, **kw); self.ids = np.full(self.K, -1); self.pos = 0
    def observe(self, i, comp):
        self.ids[self.pos] = i; self.pos = (self.pos + 1) % self.K
    def indices(self):
        return self.ids[self.ids >= 0].copy()


class LIFO(Policy):
    name = "LIFO"
    def __init__(self, K, **kw):
        super().__init__(K, **kw); self.ids = []
    def observe(self, i, comp):
        if len(self.ids) < self.K: self.ids.append(i)
    def indices(self):
        return np.array(self.ids, dtype=int)


class RandomEviction(Policy):
    name = "RandomEviction"
    def __init__(self, K, **kw):
        super().__init__(K, **kw); self.ids = []
    def observe(self, i, comp):
        if len(self.ids) < self.K: self.ids.append(i)
        else: self.ids[self.rng.integers(self.K)] = i
    def indices(self):
        return np.array(self.ids, dtype=int)


class Reservoir(Policy):
    name = "Reservoir"
    def __init__(self, K, **kw):
        super().__init__(K, **kw); self.ids = []; self.seen = 0
    def observe(self, i, comp):
        self.seen += 1
        if len(self.ids) < self.K: self.ids.append(i)
        else:
            j = self.rng.integers(self.seen)
            if j < self.K: self.ids[j] = i
    def indices(self):
        return np.array(self.ids, dtype=int)


class TimeStratified(Policy):
    """One uniformly chosen sample per equal-width time bin (needs the stream length n)."""
    name = "TimeStratified"
    def __init__(self, K, n, **kw):
        super().__init__(K, n=n, **kw)
        self.ids = np.full(self.K, -1); self.cnt = np.zeros(self.K, dtype=int)
    def observe(self, i, comp):
        b = min(self.K - 1, i * self.K // self.n_total)
        self.cnt[b] += 1
        if self.rng.random() < 1.0 / self.cnt[b]: self.ids[b] = i
    def indices(self):
        return self.ids[self.ids >= 0].copy()


class AResWeighted(Policy):
    """Efraimidis-Spirakis weighted reservoir; weight = eps + anomaly score."""
    name = "AResWeighted"
    needs_comps = True
    def __init__(self, K, **kw):
        super().__init__(K, **kw); self.h = []
    def observe(self, i, comp):
        w = config.ARES_EPS + comp[0]
        key = self.rng.random() ** (1.0 / w)
        if len(self.h) < self.K: heapq.heappush(self.h, (key, i))
        elif key > self.h[0][0]: heapq.heapreplace(self.h, (key, i))
    def indices(self):
        return np.array([i for _, i in self.h], dtype=int)


class UtilityAware(Policy):
    """Keep the K highest-utility samples. Non-recency utility is frozen at arrival; recency decays
    with age (tau = K arrivals). An arrival replaces the current lowest-utility slot if it beats it."""
    name = "UtilityAware"
    needs_comps = True
    def __init__(self, K, weights=None, tau=None, **kw):
        super().__init__(K, **kw)
        w = weights or config.DEFAULT_WEIGHTS
        self.a, self.b, self.g, self.d = w["alpha"], w["beta"], w["gamma"], w["delta"]
        self.tau = float(tau or self.K)
        self.ids = np.empty(self.K, dtype=int); self.nonrec = np.empty(self.K); self.m = 0
    def observe(self, i, comp):
        nr = self.b * comp[0] + self.g * comp[1] + self.d * comp[2]
        if self.m < self.K:
            self.ids[self.m] = i; self.nonrec[self.m] = nr; self.m += 1; return
        sc = self.a * np.exp(-(i - self.ids) / self.tau) + self.nonrec
        j = int(np.argmin(sc))
        if self.a + nr > sc[j]:
            self.ids[j] = i; self.nonrec[j] = nr
    def indices(self):
        return self.ids[: self.m].copy()


class ValueBiased(UtilityAware):
    """Anomaly score only, no recency."""
    name = "ValueBiased"
    def __init__(self, K, **kw):
        super().__init__(K, weights=dict(alpha=0.0, beta=1.0, gamma=0.0, delta=0.0), **kw)


class StratifiedUtility(Policy):
    """FIFO tier (recent history) + utility tier. Samples leaving the FIFO tier compete for the utility tier."""
    name = "StratifiedUtility"
    needs_comps = True
    def __init__(self, K, **kw):
        super().__init__(K, **kw)
        self.Kb = max(1, int(round(config.STRATIFIED_BASE_FRAC * self.K)))
        self.Ku = max(1, self.K - self.Kb)
        w = config.STRATIFIED_WEIGHTS
        self.b, self.g, self.d = w["beta"], w["gamma"], w["delta"]
        self.ring = deque(); self.uid = []; self.us = []
    def observe(self, i, comp):
        s = self.b * comp[0] + self.g * comp[1] + self.d * comp[2]
        self.ring.append((i, s))
        if len(self.ring) <= self.Kb: return
        oi, os_ = self.ring.popleft()
        if len(self.uid) < self.Ku:
            self.uid.append(oi); self.us.append(os_); return
        j = int(np.argmin(self.us))
        if os_ > self.us[j]:
            self.uid[j] = oi; self.us[j] = os_
    def indices(self):
        return np.array([i for i, _ in self.ring] + self.uid, dtype=int)


MAIN_POLICIES = ["FIFO", "LIFO", "RandomEviction", "Reservoir", "TimeStratified", "AResWeighted",
                 "ValueBiased", "UtilityAware", "StratifiedUtility", "UtilityAware-Tuned"]

_CLASSES = {c.name: c for c in [FIFO, LIFO, RandomEviction, Reservoir, TimeStratified, AResWeighted,
                                ValueBiased, UtilityAware, StratifiedUtility]}


def _norm(w):
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def ablation_specs():
    """name -> weights for the UtilityAware ablations."""
    base = config.DEFAULT_WEIGHTS
    out = {}
    for t in base:
        out[f"abl:no_{t}"] = _norm({k: (0.0 if k == t else v) for k, v in base.items()})
        out[f"abl:only_{t}"] = {k: (1.0 if k == t else 0.0) for k in base}
    return out


def make_policy(name, K, n, seed, tuned_weights=None, weights=None):
    if name == "UtilityAware-Tuned":
        return UtilityAware(K, weights=tuned_weights or config.DEFAULT_WEIGHTS, n=n, seed=seed)
    if name.startswith("abl:") or weights is not None:
        return UtilityAware(K, weights=weights, n=n, seed=seed)
    return _CLASSES[name](K, n=n, seed=seed)


def stream(policy, comps, snaps=None):
    """Feed positions 0..n-1 to the policy. If snaps is a set of positions, also return copies of the
    buffer contents at those positions."""
    snapshots = {}
    for i in range(len(comps)):
        policy.observe(i, comps[i])
        if snaps is not None and i in snaps:
            snapshots[i] = policy.indices().copy()
    return (policy.indices(), snapshots) if snaps is not None else policy.indices()
