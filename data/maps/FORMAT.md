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
