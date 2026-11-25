#!/usr/bin/env bash
# Downloads the Brazilian E-Commerce Public Dataset by Olist (CC BY-NC-SA 4.0).
# Official source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# With the Kaggle CLI:   kaggle datasets download -d olistbr/brazilian-ecommerce -p data --unzip
# Without it, this script pulls the same CSVs from Olist's public GitHub mirror.
set -euo pipefail
cd "$(dirname "$0")"
BASE=https://raw.githubusercontent.com/Athospd/work-at-olist-data/master/datasets
for f in olist_customers_dataset olist_order_items_dataset olist_order_payments_dataset \
         olist_orders_dataset olist_products_dataset olist_sellers_dataset \
         product_category_name_translation; do
  curl -sSfL -o "$f.csv" "$BASE/$f.csv" && echo "✓ $f.csv"
done
