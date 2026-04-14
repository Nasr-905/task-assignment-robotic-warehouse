# Map CSV Format

Each CSV file defines a warehouse grid row-by-row (row 0 = top of the map).
Blank/missing cells are treated as `0`.

## Tile Types

| Value | Name           | Description                                              |
|-------|----------------|----------------------------------------------------------|
| `0`   | HIGHWAY        | AGV walkable lane — no shelf placed here                 |
| `1`   | SHELF          | Shelf location — a shelf object is spawned here on reset |
| `2`   | PICKERWALL     | Goal slot — AGVs deposit shelves here for pickers        |
| `3`   | PICKER_HIGHWAY | Picker walkable aisle (picker zone)                      |
| `4`   | PACKAGING      | Packaging station — pickers walk here to complete orders |
| `5`   | REPLENISHMENT  | AGV walkable staging location for fresh stock shelves    |
| `6`   | SHARED_HIGHWAY | Shared AGV/picker aisle; AGVs yield to picker occupancy  |
| `9`   | BLANK          | Unused blocked cell                                      |

## Physical bin scaffold

Stage A1 added logical bin metadata for storage-like cells. Stage A2 assigns
one SKU per logical storage bin and computes quantity from `Unit cube F` and
usable bin volume. Stage A3 makes the current movable `Shelf` objects explicit
wrappers around logical bins: picker stock draws down the backing bin, and
replenishment shelves are backed by replenishment bins.

Default bin parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `TARWARE_BIN_VOLUME_FT3` | `2.68` | Physical volume of one bin |
| `TARWARE_BIN_USABLE_FRACTION` | `0.85` | Conservative usable fraction of bin volume |
| `BIN_LEVELS_PER_SIDE` | `5` | Vertical levels per accessible side |
| `BINS_PER_LEVEL` | `5` | Bins per level |

Derived default capacity:

| Cell tile | Cell type | Sides | Bins/cell | Usable volume/cell |
|-----------|-----------|-------|-----------|--------------------|
| `1` | Storage | 2 | 50 | `50 * 2.68 * 0.85 = 113.9 ft^3` |
| `2` | Pickerwall | 1 | 25 | `25 * 2.68 * 0.85 = 56.95 ft^3` |
| `5` | Replenishment | 1 | 25 | `25 * 2.68 * 0.85 = 56.95 ft^3` |

These bins are exposed as environment metadata:

- `env.bin_cells`: fixed map cells that contain logical bins.
- `env.logical_bins`: flattened list of all logical bins.
- `env.storage_logical_bins`: flattened list of storage-cell bins.
- `env.replenishment_logical_bins`: flattened list of replenishment-cell bins.
- `env.logical_bins_by_id[bin_id]`: lookup for a logical bin by id.
- `env.bin_cells_by_xy[(x, y)]`: lookup for the bin cell at a map coordinate.

Inventory assignment:

- Logical storage bins receive one SKU each.
- Bin quantity is `floor(usable_bin_volume_ft3 / sku_unit_cube_ft3)`.
- Bins with missing or non-positive SKU cube receive quantity `0`.
- Current `Shelf` objects act as movable/rendered wrappers around one backing
  logical bin, referenced by `shelf.bin_id`.
- Picker stock decrements the backing logical bin quantity and volume.
- Exhausted bins and wrappers are removed from SKU lookup indexes.
- Replenishment-spawned wrappers are backed by an available replenishment bin.
- Pickerwall bin metadata exists, but pickerwall cell capacity and handoff are
  still handled by the existing wrapper flow until later stages.

## Zone rules

- **AGV zone**: any row containing at least one AGV-side tile:
  `0`, `1`, `2`, `5`, or `6`.
- **Picker zone**: any connected or embedded aisle made from picker-walkable
  tiles `3`, `4`, or `6`. Older maps place this zone contiguously at the
  bottom, but test layouts may put picker aisles around a middle pickerwall.
- `num_pickers` is set at runtime (env parameter), not in the map file.
- **Shared cells**: tile `6` is walkable by both AGVs and pickers.
  Picker path planning ignores AGVs and avoids other pickers; AGVs plan around
  both AGVs and pickers and yield/replan around picker current and next
  positions at runtime.

## Files

| File             | Grid size  | Shelf rows | Shelf columns | Shelves | Goals |
|------------------|------------|------------|---------------|---------|-------|
| `tiny.csv`       | 20 × 14    | 1          | 3             | 48      | 6     |
| `small.csv`      | 30 × 14    | 2          | 3             | 96      | 6     |
| `medium.csv`     | 30 × 22    | 2          | 5             | 160     | 10    |
| `large.csv`      | 40 × 22    | 3          | 5             | 240     | 10    |
| `extralarge.csv` | 50 × 30    | 4          | 7             | 448     | 14    |

## Usage

```python
env = Warehouse(
    ...,
    map_csv_path="data/maps/tiny.csv",
    num_pickers=2,
)
```

When `map_csv_path` is provided the `shelf_columns`, `shelf_rows`, and
`column_height` constructor parameters are **ignored** — the grid geometry
comes entirely from the CSV.
