"""Paper figures (300 dpi PNG)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HIGHLIGHT = {"UtilityAware": "#d62728", "UtilityAware-Tuned": "#ff7f0e", "StratifiedUtility": "#9467bd"}


_PALETTE = ["#1f77b4", "#2ca02c", "#8c564b", "#7f7f7f", "#17becf", "#bcbd22", "#e377c2", "#aec7e8", "#98df8a", "#c49c94"]


def _color(name, i):
    """Highlighted policies keep fixed colours; the others cycle through a palette without red/orange/purple."""
    return HIGHLIGHT.get(name, _PALETTE[(i // 2) % len(_PALETTE)])


def live_recall(snap_curves, path):
    plt.figure(figsize=(7, 4))
    for i, (name, (xs, ys)) in enumerate(snap_curves.items()):
        plt.plot(xs, ys, label=name, color=_color(name, i * 2), lw=2.2 if name in HIGHLIGHT else 1.2)
    plt.xlabel("Stream position (arrivals)"); plt.ylabel("Anomaly recall so far")
    plt.title("Live anomaly recall of the buffer"); plt.legend(fontsize=7, ncol=2); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def metric_vs_K(agg_byK, metrics, policies, path):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.8))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        for i, p in enumerate(policies):
            d = agg_byK[agg_byK.policy == p].sort_values("ratio")
            if d.empty: continue
            x = d["ratio"].astype(float) * 100
            mu, sd = d[m + "_mean"].to_numpy(), d[m + "_std"].to_numpy()
            ax.plot(x, mu, marker="o", ms=3, label=p, color=_color(p, i * 2), lw=2.2 if p in HIGHLIGHT else 1.1)
            ax.fill_between(x, mu - sd, mu + sd, alpha=.12, color=_color(p, i * 2))
        ax.set_xscale("log"); ax.set_xlabel("Retention ratio K / |S| (%)"); ax.set_ylabel(m); ax.grid(alpha=.3)
    axes[0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def timeline(final_idx, yS, policies, path, bins=60):
    n = len(yS)
    fig, axes = plt.subplots(len(policies), 1, figsize=(8, 1.9 * len(policies)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, policies):
        ax.fill_between(np.arange(n), 0, yS, step="mid", color="gold", alpha=.35, transform=ax.get_xaxis_transform())
        ax.hist(final_idx[p], bins=bins, range=(0, n), color=_color(p, 0), alpha=.85)
        ax.set_ylabel(p, fontsize=7)
    axes[-1].set_xlabel("Stream position (anomalies shaded)")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def distributions(XS, final_idx, features, policies, path, max_ch=4):
    ch = list(range(min(max_ch, XS.shape[1])))
    fig, axes = plt.subplots(len(policies), len(ch), figsize=(3 * len(ch), 2.2 * len(policies)), squeeze=False)
    for r, p in enumerate(policies):
        for c, j in enumerate(ch):
            ax = axes[r][c]
            lo, hi = np.percentile(XS[:, j], [0.5, 99.5])
            ax.hist(XS[:, j], bins=40, range=(lo, hi), density=True, color="lightgray", label="full stream S")
            ax.hist(XS[final_idx[p], j], bins=40, range=(lo, hi), density=True, histtype="step",
                    color=_color(p, 0), lw=1.6, label="retained")
            if r == 0: ax.set_title(str(features[j])[:18], fontsize=8)
            if c == 0: ax.set_ylabel(p, fontsize=7)
            ax.tick_params(labelsize=6)
    axes[0][0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def bars(agg_headline, metrics, path):
    a = agg_headline[agg_headline.policy != "FullData"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 4.2))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        ax.barh(a.policy, a[m + "_mean"], xerr=a[m + "_std"], color=[_color(p, i * 2) for i, p in enumerate(a.policy)])
        ax.set_title(m, fontsize=9); ax.invert_yaxis(); ax.tick_params(labelsize=7); ax.grid(axis="x", alpha=.3)
        if ax is not axes[0]: ax.set_yticklabels([])
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
