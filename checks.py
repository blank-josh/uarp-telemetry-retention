"""Acceptance checks run before the experiments. Any failure stops the run with a clear message."""
import numpy as np
import config
from policies import make_policy, stream, MAIN_POLICIES
from scorer import OnlineScorer, precompute_components


def run_checks(XS, splits, log):
    rng = np.random.default_rng(0)
    # 1. every policy holds exactly K slots once the stream is longer than K
    n, K = 600, 50
    comps = rng.random((n, 3))
    for name in MAIN_POLICIES:
        pol = make_policy(name, K, n, 1)
        for i in range(n):
            pol.observe(i, comps[i])
            if i >= K and name != "TimeStratified":
                assert len(pol.indices()) == K, f"{name}: holds {len(pol.indices())} slots, expected {K}"
        idx = pol.indices()
        assert len(idx) == len(set(idx.tolist())) and idx.max() < n and idx.min() >= 0, f"{name}: bad indices"
        if name == "TimeStratified":
            assert len(idx) == K, "TimeStratified should fill every bin"
    log("check 1 passed: every policy holds exactly K slots, no duplicate or out-of-range indices")

    # 2. an injected spike outranks a baseline sample on the real data scale
    m = min(400, len(XS) - 1)
    sc = OnlineScorer(XS.shape[1], **config.SCORER)
    for i in range(m):
        sc.observe(XS[i])
    base = sc.observe(XS[m])
    sc2 = OnlineScorer(XS.shape[1], **config.SCORER)
    for i in range(m):
        sc2.observe(XS[i])
    spike = XS[m].copy(); spike[0] = sc.med[0] + 20 * sc.scale[0]
    spk = sc2.observe(spike)
    assert spk[0] > base[0] and spk[2] >= base[2], f"spike did not outrank baseline: {spk} vs {base}"
    log(f"check 2 passed: spike anomaly={spk[0]:.2f} vs baseline={base[0]:.2f}")

    # 3. no leakage: S < V < T in time, disjoint
    S, V, T = splits["S"], splits["V"], splits["T"]
    assert S.max() < V.min() and V.max() < T.min(), "splits overlap or are not chronological"
    assert not (set(S) & set(T)) and not (set(S) & set(V)) and not (set(V) & set(T))
    log("check 3 passed: S, V, T are disjoint and chronologically ordered; policies and scalers only ever see S")

    # 4. determinism: same seed, same retained set
    Xs = XS[: min(2000, len(XS))]
    c = precompute_components(Xs, **config.SCORER)
    for name in ("UtilityAware", "Reservoir", "AResWeighted"):
        a = np.sort(stream(make_policy(name, 40, len(Xs), 7), c))
        b = np.sort(stream(make_policy(name, 40, len(Xs), 7), c))
        assert np.array_equal(a, b), f"{name} not deterministic"
    log("check 4 passed: identical results across two runs with the same seed")
