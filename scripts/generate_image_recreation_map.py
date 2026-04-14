from pathlib import Path


OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "maps" / "image_recreation.csv"

N_SECTIONS = 4
SECTION_HEIGHT = 10
HORIZONTAL_AISLE_HEIGHT = 2

STORAGE_REPEATS = 18
STORAGE_BLOCK_WIDTH = 2
STORAGE_AISLE_WIDTH = 2

PICKWALL_REPEATS = 4
PICKWALL_BLOCK_WIDTH = 1
PICKWALL_AISLE_WIDTH = 1
PACKAGING_WIDTH = 1

RIGHT_STORAGE_REPEATS = STORAGE_REPEATS
REPLENISHMENT_REPEATS = 1


def add_storage_region(row, start_col, repeats):
    col = start_col
    for i in range(repeats):
        for dc in range(STORAGE_BLOCK_WIDTH):
            row[col + dc] = 1
        col += STORAGE_BLOCK_WIDTH
        if i != repeats - 1:
            col += STORAGE_AISLE_WIDTH
    return col


def add_pickwall_region(row, start_col):
    col = start_col
    for i in range(PICKWALL_REPEATS):
        row[col] = 2
        col += PICKWALL_BLOCK_WIDTH
        if i != PICKWALL_REPEATS - 1:
            if i == 1:
                # The middle inter-pickwall channel is an AGV-only highway.
                col += 2 * PICKWALL_AISLE_WIDTH + PACKAGING_WIDTH
            else:
                row[col] = 3
                col += PICKWALL_AISLE_WIDTH
                row[col] = 4
                col += PACKAGING_WIDTH
                row[col] = 3
                col += PICKWALL_AISLE_WIDTH
    return col


def add_replenishment_region(row, start_col, repeats):
    col = start_col
    for i in range(repeats):
        row[col] = 5
        col += 1
        if i != repeats - 1:
            col += STORAGE_AISLE_WIDTH
    return col


def build_map():
    left_storage_width = (
        STORAGE_REPEATS * STORAGE_BLOCK_WIDTH
        + (STORAGE_REPEATS - 1) * STORAGE_AISLE_WIDTH
    )
    pickwall_width = (
        PICKWALL_REPEATS * PICKWALL_BLOCK_WIDTH
        + (PICKWALL_REPEATS - 1)
        * (2 * PICKWALL_AISLE_WIDTH + PACKAGING_WIDTH)
    )
    right_storage_width = (
        RIGHT_STORAGE_REPEATS * STORAGE_BLOCK_WIDTH
        + (RIGHT_STORAGE_REPEATS - 1) * STORAGE_AISLE_WIDTH
    )
    replenishment_width = REPLENISHMENT_REPEATS

    # Outer margins plus aisle bands between the functional regions.
    width = (
        1
        + left_storage_width
        + 2
        + pickwall_width
        + 2
        + right_storage_width
        + 2
        + replenishment_width
        + 1
    )
    height = (
        N_SECTIONS * SECTION_HEIGHT
        + (N_SECTIONS + 1) * HORIZONTAL_AISLE_HEIGHT
    )

    grid = [[0 for _ in range(width)] for _ in range(height)]

    pickwall_start_col = 1 + left_storage_width + 2
    pickwall_end_col = pickwall_start_col + pickwall_width

    # Only the pickwall span of the five horizontal cross-aisle bands is shared.
    # The rest of each cross-aisle remains AGV-only highway, so pickers cannot
    # wander into the storage fields.
    for band in range(N_SECTIONS + 1):
        row_start = band * (SECTION_HEIGHT + HORIZONTAL_AISLE_HEIGHT)
        for r in range(row_start, row_start + HORIZONTAL_AISLE_HEIGHT):
            if r >= height:
                continue
            for c in range(pickwall_start_col, pickwall_end_col):
                grid[r][c] = 6

    for section in range(N_SECTIONS):
        row_start = HORIZONTAL_AISLE_HEIGHT + section * (
            SECTION_HEIGHT + HORIZONTAL_AISLE_HEIGHT
        )
        for r in range(row_start, row_start + SECTION_HEIGHT):
            row = grid[r]
            col = 1
            col = add_storage_region(row, col, STORAGE_REPEATS)
            col += 2

            col = add_pickwall_region(row, col)
            col += 2
            col = add_storage_region(row, col, RIGHT_STORAGE_REPEATS)
            col += 2
            add_replenishment_region(row, col, REPLENISHMENT_REPEATS)

    return grid


def main():
    grid = build_map()
    OUT_PATH.write_text(
        "\n".join(",".join(str(cell) for cell in row) for row in grid) + "\n"
    )
    print(f"wrote {OUT_PATH}")
    print(f"rows={len(grid)} cols={len(grid[0])}")


if __name__ == "__main__":
    main()
