#!/usr/bin/env python3
"""
Generate a shelf-level map that recreates the reference warehouse image,
along with a sibling JSON describing per-shelf bin capacity.

Pipeline
--------
1. Build a bay-level grid from the structural parameters below, using
   's' (storage), 'p' (pickwall), 'r' (replenishment) plus the aisle /
   packaging / highway codes ('3', '4', '6', '0').
2. Expand the bay grid into a shelf grid using the factors from
   data/maps/bays/map_format.json (same logic as
   scripts/convert_bay_map_to_shelf_map.py).
3. Write the shelf CSV and a matching .json metadata file.

Outputs
-------
- data/maps/shelves/image_recreation.csv
- data/maps/shelves/image_recreation.json
"""

import csv
import json
import math
import os

# ============================================================
# CONFIGURATION — edit these values
# ============================================================

TINY=False
SMALL=False
MEDIUM=False
LARGE=False
FULL=True

# File paths  (mirrors scripts/generate_orders.py)
TARWARE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR    = os.path.join(TARWARE_DIR, "data", "maps")
MAP_FORMAT_PATH = os.path.join(MAPS_DIR, "map_format.json")
OUTPUT_NAME     = "full_dhl"   # -> {output_name}.csv / {output_name}.json

# Bay-level structural parameters
N_SECTIONS           = 2    # number of horizontal sections (Default: 4)
SECTION_HEIGHT       = 4    # bay-rows per section (Default: 10)
SECTION_VERTICAL_GAP = 2    # bay-rows between sections (Default: 8)

# Outer margins
LEFT_MARGIN  = 2            # empty cols before the first storage block on the left (Default: 4)
RIGHT_MARGIN = 2            # empty cols after the last replenishment block on the right (Default: 4)

# AGV idle zone
AGV_IDLE_ZONE_HEIGHT = 0    # number of rows at the top of the map for AGV idle zone (Default: 0)

# Storage blocks
STORAGE_REPEATS     = 5    # number of storage blocks in each storage region (Default: 18) 
STORAGE_BLOCK_WIDTH = 1     # cols per storage block (Default: 1)
STORAGE_COLUMN_GAP  = 2     # empty cols between adjacent storage blocks (Default: 2)

# Spacing between the last storage block and the pickwall region (both sides)
STORAGE_TO_PICKWALL_GAP = 2  # empty cols between the storage and pickwall regions (Default: 5)
PICKWALL_TO_STORAGE_GAP = 2  # empty cols between the pickwall and right-side storage regions (Default: 5)

# Pickwall region
PICKWALL_REPEATS     = 4     # number of pickwall blocks (Default: 4)
PICKWALL_BLOCK_WIDTH = 1     # cols per pickwall block (Default: 1)
PICKWALL_AISLE_WIDTH = 1     # AGV-aisle cells on each side of the packaging strip
# Inter-pickwall gap widths  (length must be PICKWALL_REPEATS - 1).
# Layout per gap:
#   - Middle gap (AGV-only highway): `gap` empty cells (cross-aisles overlay '6').
#   - Other gaps: [AGV-aisle × PICKWALL_AISLE_WIDTH,
#                  packaging × (gap - 2 * PICKWALL_AISLE_WIDTH),
#                  AGV-aisle × PICKWALL_AISLE_WIDTH].
PICKWALL_GAPS        = [4, 2, 4]
MIDDLE_PICKWALL_GAP_INDEX = 1   # which entry of PICKWALL_GAPS is AGV-only

# Right-side storage + replenishment
RIGHT_STORAGE_REPEATS        = STORAGE_REPEATS
STORAGE_TO_REPLENISHMENT_GAP = 2    # empty cols between the rightmost storage block and the replenishment region (Default: 9)
REPLENISHMENT_REPEATS        = 1    # number of replenishment blocks (Default: 1)

# Override given configuration if TINY is True (for tiny_dhl)
if TINY:
    N_SECTIONS = 1
    SECTION_HEIGHT = 1
    STORAGE_REPEATS = 1

# Override given configuration if SMALL is True (for small_dhl)
if SMALL:
    N_SECTIONS = 1
    SECTION_HEIGHT = 2
    STORAGE_REPEATS = 2

# Override given configuration if MEDIUM is True (for medium_dhl)
if MEDIUM:
    N_SECTIONS = 2
    SECTION_HEIGHT = 2
    STORAGE_REPEATS = 3

# Override given configuration if LARGE is True (for large_dhl)
if LARGE:
    N_SECTIONS = 2
    SECTION_HEIGHT = 4
    STORAGE_REPEATS = 5

# Override given configuration if FULL is True (for full_dhl)
if FULL:
    N_SECTIONS = 4
    SECTION_HEIGHT = 10
    SECTION_VERTICAL_GAP = 8
    LEFT_MARGIN = 4
    RIGHT_MARGIN = 4
    AGV_IDLE_ZONE_HEIGHT = 0
    STORAGE_REPEATS = 18
    STORAGE_BLOCK_WIDTH = 1
    STORAGE_COLUMN_GAP = 2
    STORAGE_TO_PICKWALL_GAP = 5
    PICKWALL_TO_STORAGE_GAP = 5
    PICKWALL_REPEATS = 4
    PICKWALL_BLOCK_WIDTH = 1
    PICKWALL_AISLE_WIDTH = 1
    PICKWALL_GAPS = [4, 2, 4]
    MIDDLE_PICKWALL_GAP_INDEX = 1
    RIGHT_STORAGE_REPEATS = STORAGE_REPEATS
    STORAGE_TO_REPLENISHMENT_GAP = 9
    REPLENISHMENT_REPEATS = 1

# Ensure RIGHT_STORAGE_REPEATS matches STORAGE_REPEATS after all overrides
RIGHT_STORAGE_REPEATS = STORAGE_REPEATS

# ============================================================
# END CONFIGURATION
# ============================================================


# Bay-map cell codes
STORAGE_CHAR       = "s"
PICKWALL_CHAR      = "p"
REPLENISHMENT_CHAR = "r"
AGV_AISLE_CHAR     = "3"
PACKAGING_CHAR     = "4"
HIGHWAY_CHAR       = "6"
IDLE_ZONE_CHAR     = "7"
REPLENISHMENT_MARGIN_CHAR = "9"
EMPTY_CHAR         = "0"

# Shelf-map numeric codes produced by the expansion step
CHAR_TO_NUMERIC = {
    STORAGE_CHAR:       "1",
    PICKWALL_CHAR:      "2",
    REPLENISHMENT_CHAR: "5",
    IDLE_ZONE_CHAR:     "7",
    REPLENISHMENT_MARGIN_CHAR: "9",
}


# ------------------------------------------------------------------
# Bay-map construction
# ------------------------------------------------------------------

def add_storage_region(row, start_col, repeats):
    col = start_col
    for i in range(repeats):
        for dc in range(STORAGE_BLOCK_WIDTH):
            row[col + dc] = STORAGE_CHAR
        col += STORAGE_BLOCK_WIDTH
        if i != repeats - 1:
            col += STORAGE_COLUMN_GAP
    return col


def add_pickwall_region(row, start_col):
    col = start_col
    for i in range(PICKWALL_REPEATS):
        for dc in range(PICKWALL_BLOCK_WIDTH):
            row[col + dc] = PICKWALL_CHAR
        col += PICKWALL_BLOCK_WIDTH
        if i == PICKWALL_REPEATS - 1:
            continue

        gap = PICKWALL_GAPS[i]
        if i == MIDDLE_PICKWALL_GAP_INDEX:
            # AGV-only highway channel — leave cells as EMPTY; cross-aisle
            # bands overlay '6' on the intersecting rows.
            col += gap
        else:
            packaging = gap - 2 * PICKWALL_AISLE_WIDTH
            for dc in range(PICKWALL_AISLE_WIDTH):
                row[col + dc] = AGV_AISLE_CHAR
            col += PICKWALL_AISLE_WIDTH
            for dc in range(packaging):
                row[col + dc] = PACKAGING_CHAR
            col += packaging
            for dc in range(PICKWALL_AISLE_WIDTH):
                row[col + dc] = AGV_AISLE_CHAR
            col += PICKWALL_AISLE_WIDTH
    return col


def add_replenishment_region(row, start_col, repeats):
    col = start_col
    for i in range(repeats):
        row[col] = REPLENISHMENT_CHAR
        col += 1
        if i != repeats - 1:
            col += STORAGE_COLUMN_GAP
    return col


def build_bay_map():
    assert len(PICKWALL_GAPS) == PICKWALL_REPEATS - 1, (
        f"PICKWALL_GAPS must have length {PICKWALL_REPEATS - 1}"
    )

    left_storage_width = (
        STORAGE_REPEATS * STORAGE_BLOCK_WIDTH
        + (STORAGE_REPEATS - 1) * STORAGE_COLUMN_GAP
    )
    pickwall_width = (
        PICKWALL_REPEATS * PICKWALL_BLOCK_WIDTH
        + sum(PICKWALL_GAPS)
    )
    right_storage_width = (
        RIGHT_STORAGE_REPEATS * STORAGE_BLOCK_WIDTH
        + (RIGHT_STORAGE_REPEATS - 1) * STORAGE_COLUMN_GAP
    )
    replenishment_width = REPLENISHMENT_REPEATS

    width = (
        LEFT_MARGIN
        + left_storage_width
        + STORAGE_TO_PICKWALL_GAP
        + pickwall_width
        + PICKWALL_TO_STORAGE_GAP
        + right_storage_width
        + STORAGE_TO_REPLENISHMENT_GAP
        + replenishment_width
        + RIGHT_MARGIN
    )
    height = (
        AGV_IDLE_ZONE_HEIGHT
        + N_SECTIONS * SECTION_HEIGHT
        + (N_SECTIONS + 1) * SECTION_VERTICAL_GAP
    )

    grid = [[EMPTY_CHAR for _ in range(width)] for _ in range(height)]

    # Add the AGV idle zone at the top
    for r in range(AGV_IDLE_ZONE_HEIGHT):
        for c in range(width):
            grid[r][c] = IDLE_ZONE_CHAR

    pickwall_start_col = LEFT_MARGIN + left_storage_width + STORAGE_TO_PICKWALL_GAP
    pickwall_end_col   = pickwall_start_col + pickwall_width

    # Only the pickwall span of the horizontal cross-aisle bands is shared;
    # the rest stays AGV-only highway.
    for band in range(N_SECTIONS + 1):
        row_start = AGV_IDLE_ZONE_HEIGHT + band * (SECTION_HEIGHT + SECTION_VERTICAL_GAP)
        for r in range(row_start, row_start + SECTION_VERTICAL_GAP):
            if r >= height:
                continue

            gap_offset = r - row_start

            # Always set pickwall region to highway (shared section)
            for c in range(pickwall_start_col, pickwall_end_col):
                grid[r][c] = HIGHWAY_CHAR

            if SECTION_VERTICAL_GAP >= 2:
                # Check if this is a top or bottom divider
                is_top_divider = (band == 0)
                is_bottom_divider = (band == N_SECTIONS)
                is_first_row = (gap_offset == 0)
                is_last_row = (gap_offset == SECTION_VERTICAL_GAP - 1)

                # Determine if this row should get idle zones
                should_fill_idle = False

                if is_top_divider:
                    # Top divider: fill all rows except the last (which touches the section)
                    should_fill_idle = not is_last_row
                elif is_bottom_divider:
                    # Bottom divider: fill all rows except the first (which touches the section)
                    should_fill_idle = not is_first_row
                else:
                    # Middle dividers: all rows touch a section (sandwiched), so no idle zones
                    should_fill_idle = False

                if should_fill_idle:
                    for c in range(width):
                        if c < pickwall_start_col or c >= pickwall_end_col:
                            if grid[r][c] == EMPTY_CHAR:
                                grid[r][c] = IDLE_ZONE_CHAR

    for section in range(N_SECTIONS):
        row_start = AGV_IDLE_ZONE_HEIGHT + SECTION_VERTICAL_GAP + section * (
            SECTION_HEIGHT + SECTION_VERTICAL_GAP
        )
        for r in range(row_start, row_start + SECTION_HEIGHT):
            row = grid[r]
            col = LEFT_MARGIN
            col = add_storage_region(row, col, STORAGE_REPEATS)
            col += STORAGE_TO_PICKWALL_GAP
            col = add_pickwall_region(row, col)
            col += PICKWALL_TO_STORAGE_GAP
            col = add_storage_region(row, col, RIGHT_STORAGE_REPEATS)
            col += STORAGE_TO_REPLENISHMENT_GAP
            add_replenishment_region(row, col, REPLENISHMENT_REPEATS)

    # Fill the right margin with replenishment margin marker
    # Only fill to the right of replenishment cells in rows that have them
    for r in range(len(grid)):
        # Find the rightmost replenishment cell in this row
        rightmost_rep = -1
        for c in range(len(grid[r])):
            if grid[r][c] == REPLENISHMENT_CHAR:
                rightmost_rep = c

        # If this row has replenishment cells, fill to the right with margin marker
        if rightmost_rep != -1:
            for c in range(rightmost_rep + 1, width):
                if grid[r][c] == EMPTY_CHAR:
                    grid[r][c] = REPLENISHMENT_MARGIN_CHAR

    return grid


# ------------------------------------------------------------------
# Bay -> shelf expansion
# ------------------------------------------------------------------

def expand_bay_to_shelf(bay_map, map_format):
    bay_width_shelves   = map_format["bay_width_shelves"]
    char_depth = {
        STORAGE_CHAR:       map_format["storage_bay_depth_shelves"],
        PICKWALL_CHAR:      map_format["pickerwall_bay_depth_shelves"],
        REPLENISHMENT_CHAR: map_format["replenishment_bay_depth_shelves"],
    }

    num_cols = len(bay_map[0]) if bay_map else 0

    # Column expansion factor is set by the first s/p/r cell seen in the column.
    col_expansion = [1] * num_cols
    for col_idx in range(num_cols):
        for row in bay_map:
            cell = row[col_idx]
            if cell in char_depth:
                col_expansion[col_idx] = char_depth[cell]
                break

    shelf_map = []
    for bay_row in bay_map:
        expanded_row = []
        for col_idx, cell in enumerate(bay_row):
            factor = col_expansion[col_idx]
            if cell in CHAR_TO_NUMERIC:
                expanded_row.extend([CHAR_TO_NUMERIC[cell]] * factor)
            else:
                expanded_row.extend([cell] * factor)

        # Only rows that contain a bay marker (s/p/r) get expanded vertically.
        # Non-marker rows (pure aisle / highway / empty) are preserved once
        # instead of being dropped or duplicated.
        has_marker = any(c in char_depth for c in bay_row)
        repeats = bay_width_shelves if has_marker else 1
        for _ in range(repeats):
            shelf_map.append(expanded_row)

    return shelf_map


def compute_bins_per_shelf(map_format):
    bay_volume_bins = (
        map_format["bay_width_bins"]
        * map_format["bay_height_bins"]
        * map_format["storage_bay_depth_bins"]
    )
    shelf_area = (
        map_format["bay_width_shelves"]
        * map_format["storage_bay_depth_shelves"]
    )
    return math.floor(bay_volume_bins / shelf_area) if shelf_area > 0 else 0


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    with open(MAP_FORMAT_PATH, "r") as f:
        map_format = json.load(f)

    bay_map = build_bay_map()
    shelf_map = expand_bay_to_shelf(bay_map, map_format)

    os.makedirs(MAPS_DIR, exist_ok=True)

    csv_path = os.path.join(MAPS_DIR, f"{OUTPUT_NAME}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(shelf_map)

    metadata = {"bins_per_shelf": compute_bins_per_shelf(map_format)}
    json_path = os.path.join(MAPS_DIR, f"{OUTPUT_NAME}.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    cols = len(shelf_map[0]) if shelf_map else 0
    print(f"wrote {csv_path}")
    print(f"rows={len(shelf_map)} cols={cols}")
    print(f"wrote {json_path}  (bins_per_shelf={metadata['bins_per_shelf']})")


if __name__ == "__main__":
    main()
