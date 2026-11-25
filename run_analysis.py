"""Find which order attributes drive failed deliveries.

Usage:
    python run_analysis.py --data-dir data/ --out-dir reports/
    python run_analysis.py --data-dir data/ --max-depth 2 --min-support 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rootcause.data import ATTRIBUTES, load_orders  # noqa: E402
from rootcause.report import fig_drivers, fig_heatmap, fig_monthly, pick_drivers  # noqa: E402
from rootcause.search import SearchConfig, bfs_search  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--min-support", type=int, default=300)
    p.add_argument("--min-lift", type=float, default=1.25)
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    orders = load_orders(args.data_dir)
    no_items = orders[~orders.has_items]
    df = orders[orders.has_items].reset_index(drop=True)
    base = df.failed.mean()
    print(f"{len(df):,} resolved orders · baseline failure rate {base:.2%} "
          f"({(df.failure_type == 'late').mean():.2%} late, {(df.failure_type == 'cancelled').mean():.2%} cancelled)")
    print(f"{len(no_items):,} cancelled orders have no item data and are reported separately")

    cfg = SearchConfig(max_depth=args.max_depth, min_support=args.min_support, min_lift=args.min_lift)
    res = bfs_search(df, ATTRIBUTES, config=cfg)
    print(f"Searched {len(res):,} segments · {res.driver.sum():,} pass significance + lift + interaction checks")

    drivers = pick_drivers(df, res)
    print(f"Top {len(drivers)} de-duplicated drivers cover "
          f"{drivers.attrs['share_of_orders_covered']:.0%} of orders and "
          f"{drivers.attrs['share_of_failures_covered']:.0%} of failures:\n")
    cols = ["segment", "orders", "fail_rate", "lift", "excess_failures", "p_adj"]
    print(drivers[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    # attributes that never produce a driver are just as useful to report
    level1 = res[res.depth == 1]
    no_signal = sorted(set(ATTRIBUTES) - {c[0][0] for c in level1[level1.driver].conds})
    print("\nNo single-attribute driver found in:", ", ".join(no_signal))

    res[res.driver].drop(columns="conds").to_csv(out / "significant_segments.csv", index=False)
    drivers.drop(columns="conds").to_csv(out / "top_drivers.csv", index=False)

    fig_monthly(df, out / "figures" / "monthly_failure_rate.png")
    fig_drivers(drivers, base, out / "figures" / "top_drivers.png")
    fig_heatmap(df, "customer_state", "purchase_quarter", out / "figures" / "heatmap_state_quarter.png",
                "Failure rate (%) by destination state and purchase quarter", top_rows=12)
    fig_heatmap(df, "route", "promised_days", out / "figures" / "heatmap_route_promise.png",
                "Failure rate (%) by route and promised delivery window",
                col_order=["<=14d", "15-21d", "22-30d", "31d+"])
    print(f"\nWrote tables and figures to {out}/")


if __name__ == "__main__":
    main()
