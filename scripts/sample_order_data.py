import pandas as pd
import os

# ── Parameters ────────────────────────────────────────────────
ORDERS_PER_TIMESTAMP = 1000   # Max rows to keep per discrete timestamp wave

# Time Created is in HHMMSS format (e.g. 352 = 00:03:52, 3500 = 00:35:00).
# Set to None to include all timestamps.
MAX_TIMESTAMP = 3500        # e.g. 3500 = 00:35:00, 86400 = end of day

# Date range — format YYYYMMDD. Rows outside this range are excluded.
START_DATE = 20251227
END_DATE   = 20251227

# ── File paths ────────────────────────────────────────────────
TARWARE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(TARWARE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(TARWARE_DIR, "data", "processed")
INPUT_FILE    = os.path.join(RAW_DIR, "Order Details UofT Capstone V2 (Order Data).csv")
OUTPUT_FILE   = os.path.join(PROCESSED_DIR, "order_data_sample.csv")
# ─────────────────────────────────────────────────────────────

print("Loading Order Data...")
df = pd.read_csv(INPUT_FILE)

# 1. Filter by date range
df = df[(df["Date Created"] >= START_DATE) & (df["Date Created"] <= END_DATE)]
print(f"Rows after date filter ({START_DATE}–{END_DATE}): {len(df)}")

# 2. Filter by max timestamp (HHMMSS)
if MAX_TIMESTAMP is not None:
    df = df[df["Time Created"] <= MAX_TIMESTAMP]
    print(f"Rows after timestamp filter (<= {MAX_TIMESTAMP}): {len(df)}")

# 3. Sample up to ORDERS_PER_TIMESTAMP rows from each discrete timestamp wave
sampled = (
    df.sort_values("Time Created")
      .groupby("Time Created", sort=True)
      .head(ORDERS_PER_TIMESTAMP)
      .reset_index(drop=True)
)

print(f"Timestamps (waves) included: {sampled['Time Created'].nunique()}")
print(f"Total rows in sample: {len(sampled)}")

os.makedirs(PROCESSED_DIR, exist_ok=True)
sampled.to_csv(OUTPUT_FILE, index=False)
print(f"Saved sample to: {OUTPUT_FILE}")
