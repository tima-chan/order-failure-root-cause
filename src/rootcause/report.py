"""Turn the raw search output into a short list of drivers and README figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# palette: categorical slots 1–2, blue sequential ramp, recessive text/grid
BLUE, ORANGE = "#257de0", "#f47230"
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cce3fd", "#84b9f3", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 150,
})


def _mask(df, conds):
    return np.logical_and.reduce([(df[a] == v).to_numpy() for a, v in conds])


def pick_drivers(df: pd.DataFrame, res: pd.DataFrame, k: int = 8, max_overlap: float = 0.5,
                 min_depth: int = 1) -> pd.DataFrame:
    """Greedy de-duplication: walk drivers by excess failures and skip any segment
    that mostly sits inside one already picked (overlap / smaller size > max_overlap)."""
    picked, masks = [], []
    for _, row in res[res.driver & (res.depth >= min_depth)].iterrows():
        m = _mask(df, row.conds)
        if any((m & p).sum() / min(m.sum(), p.sum()) > max_overlap for p in masks):
            continue
        picked.append(row)
        masks.append(m)
        if len(picked) == k:
            break
    out = pd.DataFrame(picked)
    covered = np.logical_or.reduce(masks) if masks else np.zeros(len(df), bool)
    out.attrs["share_of_failures_covered"] = df.failed.to_numpy()[covered].sum() / df.failed.sum()
    out.attrs["share_of_orders_covered"] = covered.mean()
    return out


def fig_monthly(df: pd.DataFrame, path: Path):
    m = (df.assign(month=df.order_purchase_timestamp.dt.to_period("M").dt.to_timestamp())
           .groupby("month").agg(orders=("failed", "size"),
                                 late=("failure_type", lambda s: (s == "late").mean()),
                                 cancelled=("failure_type", lambda s: (s == "cancelled").mean())))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(m.index, m.late * 100, color=BLUE, lw=2, marker="o", ms=4, label="Late delivery")
    ax.plot(m.index, m.cancelled * 100, color=ORANGE, lw=2, marker="o", ms=4, label="Cancelled")
    for lab, col in (("late", BLUE), ("cancelled", ORANGE)):
        ax.annotate("Late delivery" if lab == "late" else "Cancelled",
                    (m.index[-1], m[lab].iloc[-1] * 100), xytext=(6, 0),
                    textcoords="offset points", va="center", color=TEXT, fontsize=9)
    ax.set_title("Failure rate by purchase month")
    ax.set_ylabel("% of orders")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return m


def fig_heatmap(df: pd.DataFrame, row: str, col: str, path: Path, title: str,
                top_rows: int | None = None, min_cell: int = 100, col_order: list | None = None):
    rows = df[row].value_counts().head(top_rows).index if top_rows else df[row].unique()
    d = df[df[row].isin(rows)]
    rate = d.pivot_table(index=row, columns=col, values="failed", aggfunc="mean") * 100
    n = d.pivot_table(index=row, columns=col, values="failed", aggfunc="size")
    rate = rate.where(n >= min_cell)
    rate = rate.loc[rate.mean(axis=1).sort_values(ascending=False).index]
    if col_order:
        rate = rate[[c for c in col_order if c in rate.columns]]
    fig, ax = plt.subplots(figsize=(1.1 * rate.shape[1] + 2.5, 0.42 * rate.shape[0] + 1.4))
    im = ax.imshow(rate.to_numpy(), cmap=SEQ, aspect="auto")
    ax.set_xticks(range(rate.shape[1]), rate.columns)
    ax.set_yticks(range(rate.shape[0]), rate.index)
    ax.grid(False)
    vmax = np.nanmax(rate.to_numpy())
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            v = rate.iat[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", color=MUTED, fontsize=8)
            else:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 * vmax else TEXT)
    ax.set_title(title)
    ax.set_xlabel(col.replace("_", " "))
    ax.set_ylabel(row.replace("_", " "))
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("failure rate, %")
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_drivers(drivers: pd.DataFrame, baseline: float, path: Path):
    d = drivers.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(d) + 1.2))
    ax.barh(range(len(d)), d.excess_failures, color=BLUE, height=0.6)
    ax.set_yticks(range(len(d)), [s.replace(" & ", "\n& ") if len(s) > 45 else s for s in d.segment],
                  fontsize=8.5)
    for i, (x, r, n) in enumerate(zip(d.excess_failures, d.fail_rate, d.orders)):
        ax.text(x + d.excess_failures.max() * 0.01, i,
                f"{r:.0%} fail rate · {n:,} orders", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, d.excess_failures.max() * 1.45)
    ax.set_xlabel(f"Excess failed orders vs. {baseline:.1%} baseline")
    ax.set_title("Top failure drivers (de-duplicated)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
