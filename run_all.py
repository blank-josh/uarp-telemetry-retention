"""Run everything:  python run_all.py            (datasets from config.DATASETS)
                   python run_all.py --dataset skab pump
                   python run_all.py --fast     (1 seed, 1 buffer size, <=20k rows: for debugging)"""
import argparse, platform, sys, time, tracemalloc
import numpy as np
import pandas as pd
import sklearn

import config, figures, report
from checks import run_checks
from data import load_dataset, make_splits, select_features
from metrics import RetentionEvaluator, downstream
from policies import MAIN_POLICIES, ablation_specs, make_policy, stream
from scorer import OnlineScorer, precompute_components


def tune_weights(XS, yS, XV, yV, comps, K, n_trials, log):
    """Random search over Dirichlet weight vectors, scored on the validation segment V (never on T)."""
    rng = np.random.default_rng(123)
    cands = [dict(config.DEFAULT_WEIGHTS)]
    for v in rng.dirichlet(np.ones(4), size=n_trials):
        cands.append(dict(zip(["alpha", "beta", "gamma", "delta"], v.tolist())))
    rows = []
    for w in cands:
        idx = stream(make_policy("UtilityAware", K, len(XS), 0, weights=w), comps)
        d = downstream(idx, XS, yS, XV, yV, seed=0)
        rows.append(dict(**w, val_rf_auprc=d["rf_auprc"], val_if_auprc=d["if_auprc"]))
    df = pd.DataFrame(rows)
    key = df["val_rf_auprc"].fillna(-1) + 1e-6 * df["val_if_auprc"].fillna(-1)
    best = df.loc[key.idxmax()]
    if df["val_rf_auprc"].isna().all():
        log("WARNING: validation segment has no anomalies; keeping default weights")
        best = df.iloc[0]
    wbest = {k: float(best[k]) for k in ["alpha", "beta", "gamma", "delta"]}
    log(f"tuned weights (val RF-AUPRC {best['val_rf_auprc']:.3f}): {wbest}")
    return wbest, df


def time_policies(XS, K, tuned, log):
    """Per-arrival update latency (incl. scoring for utility-based policies) and peak memory."""
    na = min(config.TIMING_ARRIVALS, len(XS))
    zero = np.zeros(3)
    rows = []
    for name in MAIN_POLICIES:
        def run():
            pol = make_policy(name, K, na, 0, tuned_weights=tuned)
            sc = OnlineScorer(XS.shape[1], **config.SCORER) if pol.needs_comps else None
            for i in range(na):
                pol.observe(i, sc.observe(XS[i]) if sc else zero)
        t0 = time.perf_counter(); run(); dt = time.perf_counter() - t0
        tracemalloc.start(); run(); _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        rows.append(dict(policy=name, mean_latency_us=1e6 * dt / na, peak_memory_kb=peak / 1024.0,
                         includes_scoring=name in {"AResWeighted", "ValueBiased", "UtilityAware",
                                                   "StratifiedUtility", "UtilityAware-Tuned"}))
    log(f"timing measured over {na} arrivals at K={K}")
    return pd.DataFrame(rows)


def run_dataset(name, args, log_root):
    cfg = config.DATASET_CONFIGS[name]
    out = config.RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run_log.txt", "w")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log(f"=== dataset: {name} ===")
    log(f"python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, sklearn {sklearn.__version__}")
    ds = load_dataset(name, cfg, max_rows_override=20000 if args.fast else None)
    for note in ds.notes: log("data: " + note)
    splits = make_splits(ds, config.SPLIT)
    S, V, T = splits["S"], splits["V"], splits["T"]
    cols = select_features(ds.X[S], cfg["max_features"])
    X = ds.X[:, cols]; features = [ds.features[i] for i in cols]
    log(f"features used ({len(features)}): {features}")
    XS, yS, eS = X[S], ds.y[S], ds.event_id[S]
    XV, yV, XT, yT = X[V], ds.y[V], X[T], ds.y[T]
    log(f"|S|={len(S)} (anomaly rate {yS.mean():.2%}, events {len(np.unique(eS[yS == 1]))}), "
        f"|V|={len(V)} ({yV.mean():.2%}), |T|={len(T)} ({yT.mean():.2%})")
    if yT.sum() == 0 or yT.sum() == len(yT):
        raise ValueError("Test segment T has only one class; change SPLIT or dataset.")
    if not args.skip_checks:
        run_checks(XS, splits, log)

    ratios = [config.HEADLINE_RATIO] if args.fast else config.RETENTION_RATIOS
    n_seeds = 1 if args.fast else args.seeds
    n_trials = 8 if args.fast else config.N_TUNING_TRIALS
    Kh = max(config.MIN_K, int(round(config.HEADLINE_RATIO * len(XS))))

    # ---- weight tuning on V (seed 0, untrimmed S) -----------------------------------------------
    comps0 = precompute_components(XS, **config.SCORER)
    tuned, tuning_df = tune_weights(XS, yS, XV, yV, comps0, Kh, n_trials, log)
    tuning_df.to_csv(out / "weight_tuning.csv", index=False)

    # ---- main experiment --------------------------------------------------------------------------
    abl = ablation_specs()
    rows, final_idx, snap_curves = [], {}, {}
    t_start = time.time()
    for seed in range(n_seeds):
        trim = 0 if seed == 0 else int(np.random.default_rng(seed).uniform(0, config.MAX_TRIM) * len(XS))
        Xs, ys, es = XS[trim:], yS[trim:], eS[trim:]
        n = len(Xs)
        comps = comps0[trim:] if trim == 0 else precompute_components(Xs, **config.SCORER)
        ev = RetentionEvaluator(Xs, ys, es)

        d = downstream(np.arange(n), Xs, ys, XT, yT, seed)
        rows.append(dict(seed=seed, ratio=1.0, K=n, policy="FullData", **d))

        for ri, ratio in enumerate(ratios):
            K = max(config.MIN_K, int(round(ratio * n)))
            specs = [(p, None) for p in MAIN_POLICIES]
            if abs(ratio - config.HEADLINE_RATIO) < 1e-12:
                specs += [(a, w) for a, w in abl.items()]
            for pname, w in specs:
                pol = make_policy(pname, K, n, seed * 1000 + ri, tuned_weights=tuned, weights=w)
                want_snap = seed == 0 and abs(ratio - config.HEADLINE_RATIO) < 1e-12 and pname in MAIN_POLICIES
                if want_snap:
                    pos = set(np.linspace(0, n - 1, config.SNAPSHOTS, dtype=int).tolist())
                    idx, snaps = stream(pol, comps, snaps=pos)
                    xs = np.array(sorted(snaps)); cum = np.cumsum(ys)
                    snap_curves[pname] = (xs, np.array([ys[snaps[p]].sum() / max(1, cum[p]) for p in xs]))
                    final_idx[pname] = idx
                else:
                    idx = stream(pol, comps)
                if pname != "TimeStratified":
                    assert len(idx) == K, f"{pname}: {len(idx)} != {K}"
                rows.append(dict(seed=seed, ratio=ratio, K=K, policy=pname,
                                 **ev.score(idx), **downstream(idx, Xs, ys, XT, yT, seed)))
        log(f"seed {seed + 1}/{n_seeds} done ({time.time() - t_start:.0f}s elapsed)")

    runs = report.add_pct_of_full(pd.DataFrame(rows))
    runs.insert(0, "dataset", name)
    runs.to_csv(out / "runs_all.csv", index=False)

    # ---- tables -----------------------------------------------------------------------------------
    order = MAIN_POLICIES + list(abl) + ["FullData"]
    agg = report.aggregate(runs)
    agg["policy"] = pd.Categorical(agg["policy"], order, ordered=True)
    agg = agg.sort_values(["ratio", "policy"]).reset_index(drop=True)
    agg.to_csv(out / "metrics_by_K.csv", index=False)
    head = agg[np.isclose(agg.ratio, config.HEADLINE_RATIO) | (agg.policy == "FullData")]
    main = head[head.policy.isin(MAIN_POLICIES + ["FullData"])].copy()
    main["policy"] = main["policy"].astype(str)
    main.to_csv(out / "metrics_main.csv", index=False)
    names = {"anomaly_recall": "Anom. recall", "event_coverage": "Event cov.", "wasserstein": "W1 (low=good)",
             "if_auprc": "IF AUPRC", "rf_auprc": "RF AUPRC", "z_f1": "Robust-z F1",
             "if_auprc_pct_full": "IF % of full", "rf_auprc_pct_full": "RF % of full"}
    report.write_latex(main, report.MAIN_TABLE_COLS, out / "table_main.tex",
                       f"Retention policies on {name} at K={int(config.HEADLINE_RATIO * 100)}\\% (mean $\\pm$ std over {n_seeds} seeds).",
                       f"tab:main_{name}", names=names)
    ab = head[head.policy.astype(str).str.startswith("abl:") | (head.policy.astype(str) == "UtilityAware")].copy()
    ab["policy"] = ab["policy"].astype(str)
    ab.to_csv(out / "ablation.csv", index=False)
    report.write_latex(ab, ["event_coverage", "anomaly_recall", "wasserstein", "if_auprc", "rf_auprc"],
                       out / "table_ablation.tex", f"Ablation of the utility terms on {name}.", f"tab:abl_{name}", names=names)

    hl = runs[np.isclose(runs.ratio, config.HEADLINE_RATIO) & runs.policy.isin(MAIN_POLICIES)]
    sig = report.significance(hl)
    sig.to_csv(out / "significance.csv", index=False)

    sysdf = time_policies(XS, Kh, tuned, log)
    sysdf.to_csv(out / "systems.csv", index=False)

    # ---- figures ----------------------------------------------------------------------------------
    byK = agg[agg.policy.astype(str).isin(MAIN_POLICIES)].copy(); byK["policy"] = byK["policy"].astype(str)
    show = ["FIFO", "Reservoir", "TimeStratified", "AResWeighted", "ValueBiased", "UtilityAware", "UtilityAware-Tuned"]
    figures.live_recall(snap_curves, out / "fig_live_recall.png")
    if len(ratios) > 1:
        figures.metric_vs_K(byK, ["event_coverage", "if_auprc", "rf_auprc"], show, out / "fig_metrics_vs_K.png")
    figures.timeline(final_idx, yS, ["FIFO", "Reservoir", "UtilityAware"], out / "fig_timeline.png")
    figures.distributions(XS, final_idx, features, ["FIFO", "Reservoir", "UtilityAware"], out / "fig_distributions.png")
    figures.bars(main.assign(), ["event_coverage", "if_auprc", "rf_auprc", "wasserstein"], out / "fig_bars.png")

    summ = report.summary_text(main)
    log("\n" + summ)
    (out / "RESULTS_SUMMARY.txt").write_text(summ)
    report.save_json(out / "results.json", dict(
        dataset=name, features=features, sizes=dict(S=len(S), V=len(V), T=len(T)), seeds=n_seeds,
        headline_ratio=config.HEADLINE_RATIO, tuned_weights=tuned, default_weights=config.DEFAULT_WEIGHTS,
        scorer=config.SCORER, headline_table=main.to_dict(orient="records")))
    log(f"\nAll outputs written to {out}")
    logf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=config.DATASETS)
    ap.add_argument("--seeds", type=int, default=config.N_SEEDS)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()
    for d in args.dataset:
        run_dataset(d, args, None)
