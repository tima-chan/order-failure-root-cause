"""Breadth-first search over attribute combinations.

Level 1 scores every single `attribute = value` segment. Each further level adds
one more condition to a segment from the previous level. Only segments that
fail *more* than the baseline are expanded, so the search stays small
(a few thousand segments on ~100K orders).

Every segment is tested against the rest of the population (two-proportion
z-test). Benjamini–Hochberg then corrects for the thousands of tests, so a
"driver" isn't just a lucky slice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


@dataclass
class SearchConfig:
    max_depth: int = 3
    min_support: int = 300        # smallest segment worth acting on
    min_lift: float = 1.25        # segment must fail ≥25% more than baseline
    min_gain: float = 1.10        # a combo must beat each of its sub-segments by ≥10%
    alpha: float = 0.05           # FDR level for Benjamini–Hochberg


@dataclass
class _Node:
    conds: tuple                  # ((attr, value), ...) in attribute order
    idx: np.ndarray               # row positions in the segment
    last_attr: int                # index of the last attribute used (canonical order)
    rate: float = field(default=0.0)


def _ztest(f1, n1, f2, n2):
    """Two-sided two-proportion z-test, vectorised."""
    p = (f1 + f2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (f1 / n1 - f2 / n2) / se
    return 2 * norm.sf(np.abs(np.nan_to_num(z)))


def _wilson(f, n, z=1.96):
    p = f / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def label(conds) -> str:
    return " & ".join(f"{a} = {v}" for a, v in conds)


def bfs_search(df: pd.DataFrame, attributes: list[str], target: str = "failed",
               config: SearchConfig | None = None) -> pd.DataFrame:
    """Return every evaluated segment with rate, lift, p-values and a `driver` flag."""
    cfg = config or SearchConfig()
    y = df[target].to_numpy()
    codes = {a: df[a].astype("category") for a in attributes}
    N, F = len(y), int(y.sum())
    base = F / N

    rows = []
    rate_of = {(): base}
    frontier = [_Node((), np.arange(N), -1, base)]

    for depth in range(1, cfg.max_depth + 1):
        nxt = []
        for node in frontier:
            for j in range(node.last_attr + 1, len(attributes)):
                attr = attributes[j]
                col = codes[attr].iloc[node.idx]
                g = pd.DataFrame({"v": col.to_numpy(), "y": y[node.idx]}).groupby("v", observed=True).y
                stats = g.agg(["size", "sum"])
                for val, (n, f) in stats.iterrows():
                    if n < cfg.min_support:
                        continue
                    conds = node.conds + ((attr, val),)
                    rate = f / n
                    rate_of[conds] = rate
                    rows.append((depth, conds, int(n), int(f), rate))
                    if rate > base and depth < cfg.max_depth:
                        child_idx = node.idx[(col == val).to_numpy()]
                        nxt.append(_Node(conds, child_idx, j, rate))
        frontier = nxt

    res = pd.DataFrame(rows, columns=["depth", "conds", "orders", "failures", "fail_rate"])
    n, f = res.orders.to_numpy(), res.failures.to_numpy()
    res["lift"] = res.fail_rate / base
    res["p_value"] = _ztest(f, n, F - f, N - n)
    res["p_adj"] = multipletests(res.p_value, method="fdr_bh")[1]
    res["ci_low"], res["ci_high"] = _wilson(f, n)
    # failures that would not have happened if the segment performed at baseline
    res["excess_failures"] = f - n * base

    # interaction check: a combo must beat every sub-segment one level down
    def best_parent(conds):
        if len(conds) == 1:
            return base
        subs = [conds[:i] + conds[i + 1:] for i in range(len(conds))]
        for s in subs:
            if s not in rate_of:  # sub-segment was pruned from the search: score it directly
                mask = np.logical_and.reduce([(df[a] == v).to_numpy() for a, v in s])
                rate_of[s] = y[mask].mean() if mask.any() else base
        return max(rate_of[s] for s in subs)

    res["best_parent_rate"] = res.conds.map(best_parent)
    res["gain_vs_parent"] = res.fail_rate / res.best_parent_rate
    res["driver"] = (
        (res.p_adj < cfg.alpha)
        & (res.lift >= cfg.min_lift)
        & ((res.depth == 1) | (res.gain_vs_parent >= cfg.min_gain))
    )
    res["segment"] = res.conds.map(label)
    res.attrs["baseline"] = base
    res.attrs["n_orders"] = N
    return res.sort_values("excess_failures", ascending=False).reset_index(drop=True)
