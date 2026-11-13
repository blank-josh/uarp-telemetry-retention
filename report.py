"""Aggregation, CSV/LaTeX tables, significance tests and the plain-language summary."""
import json
import numpy as np
import pandas as pd

METRICS = ["anomaly_recall", "event_coverage", "wasserstein", "tail_coverage", "median_staleness",
           "if_auroc", "if_auprc", "rf_auroc", "rf_auprc", "z_f1", "if_auprc_pct_full", "rf_auprc_pct_full"]
HIGHER_BETTER = {"anomaly_recall": 1, "event_coverage": 1, "tail_coverage": 1, "if_auroc": 1, "if_auprc": 1,
                 "rf_auroc": 1, "rf_auprc": 1, "z_f1": 1, "wasserstein": -1,
                 "if_auprc_pct_full": 1, "rf_auprc_pct_full": 1}
MAIN_TABLE_COLS = ["anomaly_recall", "event_coverage", "wasserstein", "if_auprc", "rf_auprc", "z_f1",
                   "if_auprc_pct_full", "rf_auprc_pct_full"]


def add_pct_of_full(runs):
    full = runs[runs.policy == "FullData"].set_index("seed")
    for m in ("if_auprc", "rf_auprc"):
        ref = runs["seed"].map(full[m])
        runs[m + "_pct_full"] = np.where(ref > 0, 100.0 * runs[m] / ref, np.nan)
    return runs


def aggregate(runs, by=("policy", "ratio")):
    g = runs.groupby(list(by), sort=False)
    mean = g[METRICS].mean().add_suffix("_mean")
    std = g[METRICS].std(ddof=1).fillna(0.0).add_suffix("_std")
    return pd.concat([mean, std], axis=1).reset_index()


def _tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_latex(df_mean_std, cols, path, caption, label, policy_col="policy", names=None):
    """Booktabs table of 'mean ± std' with the best value per column in bold."""
    names = names or {}
    rows = list(df_mean_std[policy_col])
    best = {}
    for c in cols:
        m = df_mean_std[c + "_mean"].to_numpy(dtype=float).copy()
        m[[str(r) == "FullData" for r in rows]] = np.nan       # the full-data reference is never "best"
        if np.all(np.isnan(m)):
            continue
        best[c] = np.nanargmax(m * HIGHER_BETTER.get(c, 1))
    head = " & ".join(["Policy"] + [_tex_escape(names.get(c, c)) for c in cols]) + r" \\"
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             head, r"\midrule"]
    for r, name in enumerate(rows):
        cells = [_tex_escape(name)]
        for c in cols:
            mu, sd = df_mean_std[c + "_mean"].iloc[r], df_mean_std[c + "_std"].iloc[r]
            if np.isnan(mu):
                cells.append("--"); continue
            txt = f"{mu:.3f} $\\pm$ {sd:.3f}"
            cells.append(rf"\textbf{{{txt}}}" if best.get(c) == r else txt)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    path.write_text("\n".join(lines))


def significance(runs_headline, ref="UtilityAware", metrics=("event_coverage", "if_auprc", "rf_auprc")):
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return pd.DataFrame()
    rows = []
    piv = {m: runs_headline.pivot_table(index="seed", columns="policy", values=m) for m in metrics}
    for m in metrics:
        for pol in piv[m].columns:
            if pol in (ref, "FullData") or ref not in piv[m].columns:
                continue
            d = (piv[m][ref] - piv[m][pol]).dropna()
            try:
                p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
            except Exception:
                p = float("nan")
            rows.append(dict(metric=m, baseline=pol, mean_diff=float(d.mean()), p_value=p, n_seeds=len(d)))
    return pd.DataFrame(rows)


def summary_text(agg_headline, ref="UtilityAware"):
    a = agg_headline[agg_headline.policy != "FullData"].set_index("policy")
    out = ["RESULTS SUMMARY (headline buffer size, mean over seeds)"]
    for m in ["event_coverage", "anomaly_recall", "if_auprc", "rf_auprc", "z_f1", "wasserstein"]:
        col = m + "_mean"
        if col not in a.columns or a[col].isna().all():
            continue
        sign = HIGHER_BETTER[m]
        order = (a[col] * sign).sort_values(ascending=False)
        best = order.index[0]
        rank = list(order.index).index(ref) + 1 if ref in order.index else None
        line = f"- {m}: best = {best} ({a.loc[best, col]:.3f})"
        if ref in a.index:
            line += f"; {ref} = {a.loc[ref, col]:.3f}, rank {rank}/{len(order)}"
            gap = abs(a.loc[best, col] - a.loc[ref, col])
            if best != ref and gap < 5e-4:
                line += " -> effectively tied with the best"
            elif best != ref:
                line += f" -> {ref} LOSES here by {gap:.3f}"
        out.append(line)
    return "\n".join(out)


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
