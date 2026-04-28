import pandas as pd
import os
from datetime import datetime, timedelta

# ── Parameters ────────────────────────────────────────────────
# Date Created values (YYYYMMDD) to keep. All other rows are dropped.
KEEP_DATES = [
    20260114,
    20260111,
    20260108,
    20260107,
    20260110,
    20260119,
    20251227,
]

# After filtering, the kept dates are sorted ascending and remapped to
# consecutive days starting at REMAP_START_DATE.
REMAP_START_DATE = 20260101

# ── File paths ────────────────────────────────────────────────
TARWARE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(TARWARE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(TARWARE_DIR, "data", "processed")
INPUT_FILE    = os.path.join(RAW_DIR, "Order Details UofT Capstone V2 (Order Data).csv")
OUTPUT_FILE   = os.path.join(PROCESSED_DIR, "final_order_dataset.csv")
# ─────────────────────────────────────────────────────────────


def build_date_mapping(keep_dates: list[int], start_date: int) -> dict[int, int]:
    """Map sorted keep_dates → consecutive days beginning at start_date."""
    sorted_dates = sorted(set(keep_dates))
    start = datetime.strptime(str(start_date), "%Y%m%d")
    mapping: dict[int, int] = {}
    for i, original in enumerate(sorted_dates):
        new_dt = start + timedelta(days=i)
        mapping[original] = int(new_dt.strftime("%Y%m%d"))
    return mapping


def main() -> None:
    print("Loading Order Data...")
    df = pd.read_csv(INPUT_FILE)

    keep_set = set(KEEP_DATES)
    df = df[df["Date Created"].isin(keep_set)]
    print(f"Rows after Date Created filter ({len(keep_set)} dates): {len(df)}")

    before = len(df)
    df = df[df["Large Lines"].astype(str).str.strip().str.lower() != "large"]
    print(f"Rows after dropping Large Lines == 'large': {len(df)} (removed {before - len(df)})")

    date_map = build_date_mapping(KEEP_DATES, REMAP_START_DATE)
    print("Date remapping:")
    for original, remapped in sorted(date_map.items()):
        print(f"  {original} → {remapped}")

    # Sort so output is in chronological order before remapping.
    df = df.sort_values(["Date Created", "Time Created"]).reset_index(drop=True)

    df["Date Created"] = df["Date Created"].map(date_map)

    # The Ship Date column has stray whitespace in the raw header; locate it
    # by prefix to be robust.
    ship_date_col = next(
        (c for c in df.columns if c.strip().lower() == "ship date"),
        None,
    )
    if ship_date_col is not None:
        df[ship_date_col] = df[ship_date_col].map(lambda v: date_map.get(v, v))

    print(f"Total rows in final dataset: {len(df)}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved final dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
