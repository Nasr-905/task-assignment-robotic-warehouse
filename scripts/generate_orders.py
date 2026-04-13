#!/usr/bin/env python3
"""
Synthetic order data generator.

Produces a CSV matching the format of order_data_sample.csv with
configurable SKU catalog, order volume, and probability distributions.

Timestamp model
---------------
Orders are grouped into 60-second windows.  Each window's order count
is drawn from Poisson(POISSON_LAMBDA).  The Timestamp column uses
HHMMSS encoding where window k (1-indexed) → k * 100
  e.g. window 1 → 100  (00:01:00 = 12:01:00 AM)
       window 2 → 200  (00:02:00 = 12:02:00 AM)
       window 60 → 6000 (01:00:00 = 1:00:00 AM)
"""

import csv
import os
import random
from collections.abc import Sequence
import numpy as np

# ============================================================
# CONFIGURATION — edit these values
# ============================================================

# Catalog & volume
NUM_SKUS        = 10    # unique SKUs in the catalogue
NUM_TIME_BUCKETS = 60   # number of 60-second windows to simulate

# Poisson rate: expected orders arriving per 60-second window.
# Total orders ≈ POISSON_LAMBDA * NUM_TIME_BUCKETS.
POISSON_LAMBDA  = 5.0

# SKU frequency distribution
# Each SKU is ranked 1..NUM_SKUS by popularity.
# The relative weight of rank r is  1 / r^SKU_DIST_EXPONENT.
#   1.0  → harmonic  (most common ≫ least common)
#   0.0  → uniform   (all SKUs equally likely)
#   2.0  → quadratic drop-off
SKU_DIST_EXPONENT = 1.0

# Relative probabilities for the number of SKU *lines* per order.
# Index 0 → 1 line, index 1 → 2 lines, index 2 → 3 lines, …
# Values do NOT need to sum to 1 — they are normalised automatically.
SKU_COUNT_WEIGHTS = [50, 20, 10, 5, 5, 5, 3, 3, 2, 2, 1, 1, 1]   # up to 13 lines per order

# Relative probabilities for the shipped quantity of each SKU line.
# Index 0 → qty 1, index 1 → qty 2, index 2 → qty 3, …
# Values do NOT need to sum to 1 — they are normalised automatically.
SHIPPED_QTY_WEIGHTS = [70, 20, 10, 500]   # up to qty 4

# Date written into every row  (YYYYMMDD)
ORDER_DATE = "20260101"

# Numeric base for Order IDs  (C + zero-padded 11-digit number)
ORDER_ID_BASE = 1557500000

# Output file  (written next to this script)
TARWARE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(TARWARE_DIR, "data", "processed")
OUTPUT_FILE   = os.path.join(PROCESSED_DIR, "generated_orders.csv")

# Reproducibility  (set to None for a different result each run)
RANDOM_SEED = 42

# ============================================================
# END CONFIGURATION
# ============================================================


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def hhmmss_to_timestamp_str(hhmmss_int: int) -> str:
    """Convert integer HHMMSS → 'HH:MM:SS AM/PM' display string."""
    hh = hhmmss_int // 10000
    mm = (hhmmss_int % 10000) // 100
    ss = hhmmss_int % 100
    period = "AM" if hh < 12 else "PM"
    display_hh = hh % 12 or 12
    return f"{display_hh}:{mm:02d}:{ss:02d} {period}"


def time_bucket_label(hhmmss_int: int) -> str:
    """Return a coarse 2-hour bucket label, e.g. '12AM-2AM'."""
    hh = hhmmss_int // 10000
    slot_start = (hh // 2) * 2          # floor to even hour
    slot_end   = slot_start + 2
    def fmt(h):
        if h == 0:   return "12AM"
        if h == 12:  return "12PM"
        if h < 12:   return f"{h}AM"
        return f"{h - 12}PM"
    return f"{fmt(slot_start)}-{fmt(slot_end)}"


def hhmmss_hour(hhmmss_int: int) -> int:
    return hhmmss_int // 10000


def build_sku_catalog(n: int, exponent: float, rng: random.Random) -> list[dict]:
    """
    Create n SKUs with IDs and pre-computed sampling weights.

    SKU IDs are random 6-digit integers (matching the sample data style).
    Rank-1 (most popular) gets weight 1, rank-r gets weight 1/r^exponent.
    """
    # Use a sorted set so IDs are unique
    ids: set[int] = set()
    while len(ids) < n:
        ids.add(rng.randint(10000, 999999))
    sorted_ids = sorted(ids)

    catalog = []
    for rank, sku_id in enumerate(sorted_ids, start=1):
        weight = 1.0 / (rank ** exponent)
        catalog.append({"sku_id": float(sku_id), "rank": rank, "weight": weight})

    # Normalise weights
    total = sum(c["weight"] for c in catalog)
    for c in catalog:
        c["weight"] /= total

    return catalog


def sample_sku(catalog: list[dict], rng: random.Random) -> float:
    """Sample one SKU ID according to catalog weights."""
    r = rng.random()
    cumulative = 0.0
    for c in catalog:
        cumulative += c["weight"]
        if r <= cumulative:
            return c["sku_id"]
    return catalog[-1]["sku_id"]   # floating-point safety


def sample_sku_count(weights: Sequence[float], rng: random.Random) -> int:
    """Sample number of lines (1-indexed) from normalised weight list."""
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return i + 1
    return len(weights)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def generate(
    num_skus: int,
    num_time_buckets: int,
    poisson_lambda: float,
    sku_dist_exponent: float,
    sku_count_weights: Sequence[float],
    shipped_qty_weights: Sequence[float],
    order_date: str,
    order_id_base: int,
    seed,
) -> list[dict]:
    """Return a list of row dicts ready for csv.DictWriter."""

    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    catalog = build_sku_catalog(num_skus, sku_dist_exponent, rng)

    rows         = []
    order_counter = order_id_base
    order_number  = 0   # running count for unique order IDs within a bucket

    for bucket in range(1, num_time_buckets + 1):
        n_orders_this_bucket = int(np_rng.poisson(poisson_lambda))

        # HHMMSS for this 60-second window
        hhmmss = bucket * 100           # e.g. bucket 1 → 100 = 00:01:00
        # Carry minutes into hours when MM ≥ 60
        raw_mm = hhmmss % 10000 // 100
        raw_hh = hhmmss // 10000
        extra_hours, mm = divmod(raw_mm, 60)
        hh = raw_hh + extra_hours
        hhmmss_norm = hh * 10000 + mm * 100   # SS always 00

        timestamp_str  = hhmmss_to_timestamp_str(hhmmss_norm)
        bucket_label   = time_bucket_label(hhmmss_norm)
        hour_val       = hhmmss_hour(hhmmss_norm)

        for _ in range(n_orders_this_bucket):
            order_number += 1
            order_id = f"C{order_counter + order_number:011d}"

            # Determine how many SKU lines this order has
            n_lines = sample_sku_count(sku_count_weights, rng)

            # Sample distinct SKUs for this order (no duplicates)
            chosen: list[float] = []
            attempts = 0
            while len(chosen) < n_lines and attempts < 10 * num_skus:
                candidate = sample_sku(catalog, rng)
                if candidate not in chosen:
                    chosen.append(candidate)
                attempts += 1

            shipped_qty = sample_sku_count(shipped_qty_weights, rng)

            for sku_id in chosen:
                # Synthetic unit cube value (loosely matching the sample range)
                unit_cube = round(rng.uniform(0.01, 2.0), 9)
                large_line = "large" if unit_cube > 1.0 else "OK"

                rows.append({
                    "Order #":          order_id,
                    "Date Created":     order_date,
                    "Time Created":     bucket * 60,   # seconds from midnight
                    "Shipped Quantity": shipped_qty,
                    "Ship Date":        order_date,
                    "Order Type":       "eCom",
                    "SKU":              sku_id,
                    "Unit cube F ":     unit_cube,
                    "Large Lines":      large_line,
                    "Timestamp":        timestamp_str,
                    "Time Bucket":      bucket_label,
                    "Hour":             hour_val,
                })

    return rows


def main():
    rows = generate(
        num_skus          = NUM_SKUS,
        num_time_buckets  = NUM_TIME_BUCKETS,
        poisson_lambda    = POISSON_LAMBDA,
        sku_dist_exponent = SKU_DIST_EXPONENT,
        sku_count_weights   = SKU_COUNT_WEIGHTS,
        shipped_qty_weights = SHIPPED_QTY_WEIGHTS,
        order_date          = ORDER_DATE,
        order_id_base     = ORDER_ID_BASE,
        seed              = RANDOM_SEED,
    )

    fieldnames = [
        "Order #", "Date Created", "Time Created",
        "Shipped Quantity", "Ship Date", "Order Type",
        "SKU", "Unit cube F ", "Large Lines",
        "Timestamp", "Time Bucket", "Hour",
    ]

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    unique_orders = len({r["Order #"] for r in rows})
    print(f"Generated {len(rows)} rows across {unique_orders} orders.")
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
