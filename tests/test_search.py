"""The search must recover a planted interaction and ignore pure noise."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rootcause.search import SearchConfig, bfs_search  # noqa: E402


def make_data(n=40_000, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "city": rng.choice(["A", "B", "C", "D"], n),
        "device": rng.choice(["ios", "android", "web"], n),
        "payment": rng.choice(["card", "cash"], n),
        "noise": rng.choice(list("pqrstu"), n),
    })
    p = np.full(n, 0.05)
    # planted root cause: city B on android fails 4x — neither attribute alone is extreme
    p[(df.city == "B") & (df.device == "android")] = 0.20
    df["failed"] = (rng.random(n) < p).astype(int)
    return df


def test_finds_planted_interaction():
    df = make_data()
    res = bfs_search(df, ["city", "device", "payment", "noise"], config=SearchConfig(max_depth=2))
    top = res[res.driver].iloc[0]
    assert set(top.conds) == {("city", "B"), ("device", "android")}


def test_noise_attribute_is_never_a_driver():
    df = make_data()
    res = bfs_search(df, ["city", "device", "payment", "noise"], config=SearchConfig(max_depth=2))
    drivers = res[res.driver]
    assert not any(a == "noise" and len(c) == 1 for c in drivers.conds for a, _ in c)


def test_respects_min_support():
    df = make_data()
    res = bfs_search(df, ["city", "device", "payment", "noise"], config=SearchConfig(min_support=2_000))
    assert res.orders.min() >= 2_000
