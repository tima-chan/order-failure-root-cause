"""Load the Olist e-commerce tables and build one row per order with
categorical attributes known at purchase time, plus the failure flag.

An order counts as *failed* if it was cancelled/unavailable, or if it was
delivered after the date promised to the customer. Orders still in flight
(shipped, invoiced, processing…) are excluded because their outcome is unknown.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REGION = {
    # North
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North",
    "RR": "North", "TO": "North",
    # Northeast
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast",
    "PB": "Northeast", "PE": "Northeast", "PI": "Northeast", "RN": "Northeast",
    "SE": "Northeast",
    # Center-West
    "DF": "Center-West", "GO": "Center-West", "MT": "Center-West", "MS": "Center-West",
    # Southeast
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    # South
    "PR": "South", "RS": "South", "SC": "South",
}

FAILED_STATUSES = {"canceled", "unavailable"}
RESOLVED_STATUSES = FAILED_STATUSES | {"delivered"}

# Attributes the search is allowed to combine. All are known when the order is placed.
ATTRIBUTES = [
    "customer_region",
    "customer_state",
    "seller_region",
    "route",
    "category",
    "payment_type",
    "installments",
    "basket_value",
    "freight_share",
    "weight",
    "n_items",
    "multi_seller",
    "promised_days",
    "seller_size",
    "purchase_weekday",
    "purchase_quarter",
]


def _read(data_dir: Path, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"{name}.csv", **kw)


def _bucket(s: pd.Series, bins, labels) -> pd.Series:
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(str)


def load_orders(data_dir: str | Path, start="2017-01-01", end="2018-08-31") -> pd.DataFrame:
    """Return one row per resolved order with categorical attributes and `failed` (0/1)."""
    data_dir = Path(data_dir)
    ts_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    orders = _read(data_dir, "olist_orders_dataset", parse_dates=ts_cols)
    items = _read(data_dir, "olist_order_items_dataset")
    payments = _read(data_dir, "olist_order_payments_dataset")
    customers = _read(data_dir, "olist_customers_dataset")
    sellers = _read(data_dir, "olist_sellers_dataset")
    products = _read(data_dir, "olist_products_dataset")
    cat_en = _read(data_dir, "product_category_name_translation")

    o = orders[
        orders.order_status.isin(RESOLVED_STATUSES)
        & orders.order_purchase_timestamp.between(start, f"{end} 23:59:59")
    ].copy()

    late = o.order_delivered_customer_date > o.order_estimated_delivery_date
    o["failed"] = (o.order_status.isin(FAILED_STATUSES) | late).astype(int)
    o["failure_type"] = np.select(
        [o.order_status.isin(FAILED_STATUSES), late], ["cancelled", "late"], "ok"
    )

    # --- customer
    o = o.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    o["customer_region"] = o.customer_state.map(REGION)

    # --- items, products, sellers (aggregate to order level)
    products = products.merge(cat_en, on="product_category_name", how="left")
    products["category"] = products.product_category_name_english.fillna("unknown")
    it = items.merge(products[["product_id", "category", "product_weight_g"]], on="product_id", how="left")
    it = it.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")

    seller_volume = items.groupby("seller_id").order_id.nunique()
    it["seller_orders"] = it.seller_id.map(seller_volume)

    # main line = most expensive item in the basket
    main = it.sort_values("price", ascending=False).drop_duplicates("order_id")
    agg = it.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
        basket=("price", "sum"),
        freight=("freight_value", "sum"),
        weight_g=("product_weight_g", "sum"),
    )
    main = main.set_index("order_id")[["category", "seller_state", "seller_orders"]]
    o = o.join(agg, on="order_id").join(main, on="order_id")

    # cancelled-before-checkout orders may have no items; keep them with "unknown"
    o["seller_region"] = o.seller_state.map(REGION).fillna("unknown")
    o["route"] = np.where(
        o.seller_state.isna(), "unknown",
        np.where(o.seller_state == o.customer_state, "same state",
                 np.where(o.seller_region == o.customer_region, "same region", "cross region")),
    )

    # top categories only; the long tail becomes "other"
    top = o.category.value_counts().head(15).index
    o["category"] = o.category.where(o.category.isin(top), "other").fillna("unknown")

    # --- payments
    pay = payments.sort_values("payment_value", ascending=False).drop_duplicates("order_id")
    pay = pay.set_index("order_id")[["payment_type", "payment_installments"]]
    o = o.join(pay, on="order_id")
    o["payment_type"] = o.payment_type.fillna("unknown")
    o["installments"] = _bucket(o.payment_installments.fillna(1), [0, 1, 3, 6, 24], ["1", "2-3", "4-6", "7+"])

    # --- numeric buckets
    o["basket_value"] = _bucket(o.basket.fillna(0), [0, 50, 100, 200, 500, 1e6],
                                ["<50", "50-100", "100-200", "200-500", "500+"])
    share = (o.freight / (o.basket + o.freight)).fillna(0)
    o["freight_share"] = _bucket(share, [0, .1, .2, .3, 1], ["<10%", "10-20%", "20-30%", "30%+"])
    o["weight"] = _bucket(o.weight_g.fillna(0), [0, 500, 2000, 10000, 1e9],
                          ["<0.5kg", "0.5-2kg", "2-10kg", "10kg+"])
    o["n_items"] = np.where(o.n_items.fillna(0) > 1, "2+", "1")
    o["multi_seller"] = np.where(o.n_sellers.fillna(1) > 1, "yes", "no")
    promised = (o.order_estimated_delivery_date - o.order_purchase_timestamp).dt.days
    o["promised_days"] = _bucket(promised, [-1, 14, 21, 30, 1000], ["<=14d", "15-21d", "22-30d", "31d+"])
    o["seller_size"] = _bucket(o.seller_orders.fillna(0), [0, 50, 500, 1e6], ["small", "mid", "large"])
    o["purchase_weekday"] = o.order_purchase_timestamp.dt.day_name()
    o["purchase_quarter"] = o.order_purchase_timestamp.dt.to_period("Q").astype(str)

    # Orders with no item rows (almost all "unavailable") carry no product, seller
    # or basket information, so the search can't explain them. Flag them so the
    # caller can report them separately instead of letting them pose as a segment.
    o["has_items"] = o.basket.notna()

    keep = ["order_id", "order_purchase_timestamp", "failed", "failure_type", "has_items"] + ATTRIBUTES
    return o[keep].reset_index(drop=True)
