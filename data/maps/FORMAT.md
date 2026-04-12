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

## Zone rules

- **AGV zone**: any row containing at least one tile of type `0`, `1`, or `2`.
- **Picker zone**: rows whose tiles are all `3` or `4` (must be contiguous at the bottom).
- `num_pickers` is set at runtime (env parameter), not in the map file.

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
