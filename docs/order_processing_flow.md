# Order Processing Flow

This diagram shows how order rows are digested into simulator data structures,
physical shelf requests, picker work, packaging progress, and replenishment.

```mermaid
flowchart TD
    A[Order CSV<br/>data/processed/order_data_sample.csv] --> B[OrderSequencer.__init__]

    B --> C[Order objects<br/>order_number, date, time, type, skus]
    B --> D[SKUEntry objects<br/>sku, quantity]
    B --> E[_pending<br/>List[Order], sorted by creation time]
    B --> F[_unique_skus<br/>SKU frequency order]

    G[Warehouse.reset] --> H[Create Shelf objects<br/>from map shelf_locs]
    H --> I[initialize_shelf_sku_map]
    F --> I
    I --> J[Shelf.sku assigned]
    I --> K[_sku_to_shelves<br/>Dict sku -> List[Shelf]]

    G --> L[release_pending_orders step 0]
    E --> L
    L --> M[_active_queue<br/>released orders]
    L --> N[_pending_sku_requests<br/>Deque of SKUEntry + Order]

    N --> O[next_order_shelf candidates]
    K --> O
    O --> P[Warehouse.request_queue<br/>List[Shelf] requested by active orders]
    O --> Q[_shelf_to_order<br/>shelf_id -> Order]

    P --> R[heuristic_episode]
    R --> S[Assign AGV PICKING mission<br/>go to requested shelf]
    S --> T[AGV loads shelf]
    T --> U[Assign AGV DELIVERING mission<br/>go to pickerwall]
    U --> V[AGV unloads shelf at goal / pickerwall]

    V --> W[_execute_unload detects<br/>delivered requested shelf]
    Q --> W
    W --> X[_pickerwall_pending<br/>Deque of shelf_id + SKUEntry + Order]
    W --> Y[request_queue replacement]
    N --> Y
    Y --> P

    X --> Z[_claim_items_for_picker]
    Z --> AA[PickerClaim<br/>shelf_id, SKUEntry, order_number, Order]
    Z --> AB[_packaging_slots<br/>order_number -> required / delivered]
    AA --> AC[_build_picker_task<br/>ordered claims for picker]

    AC --> AD[_advance_pickers]
    AD --> AE[Picker walks to pickerwall shelf]
    AE --> AF[Picker PICKING state]
    AF --> AG[Shelf.capacity decreases]
    AF --> AH[_maybe_mark_shelf_fulfilled]
    AG --> AI{Shelf depleted?}
    AI -->|yes| AJ[_issue_replenishment<br/>spawn new shelf at replenishment cell]
    AJ --> K
    AJ --> Y

    AH --> AK[Shelf.fulfilled = true]
    AK --> AL[Heuristic displacement<br/>AGV returns fulfilled pickerwall shelf to storage]

    AF --> AM[Picker walks to packaging]
    AM --> AN[AT_PACKAGING]
    AN --> AB
    AB --> AO{delivered >= required?}
    AO -->|yes| AP[Order complete<br/>delete packaging slot]
```

## Key Data Structures

`OrderSequencer._pending`: orders loaded from CSV but not yet released.

`OrderSequencer._active_queue`: orders whose simulated release time has arrived.

`OrderSequencer._pending_sku_requests`: SKU-level work exploded from active
orders. This queue feeds physical shelf requests.

`OrderSequencer._sku_to_shelves`: lookup from SKU to shelves currently assigned
that SKU.

`Warehouse.request_queue`: physical shelves that AGVs should fetch for active
SKU requests.

`Warehouse._shelf_to_order`: preserves the order identity after an order line
has been converted into a shelf request.

`Warehouse._pickerwall_pending`: delivered pickerwall shelves waiting to become
picker claims.

`PickerClaim` and `PickerTask`: picker-side work units and capacity-limited
task batches.

`Warehouse._packaging_slots`: order-level packaging progress, keyed by order
number.
