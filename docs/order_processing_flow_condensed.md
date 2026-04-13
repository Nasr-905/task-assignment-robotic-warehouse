# Order Processing Flow Condensed

This condensed diagram shows the main handoffs in the current order-processing
pipeline: CSV rows become order/SKU queues, then physical shelf requests, then
pickerwall work, picker tasks, packaging progress, and replenishment.

```mermaid
flowchart TD
    A[Order CSV] --> B[OrderSequencer<br/>group rows into Orders and SKUEntries]
    B --> C[Time-gated order release<br/>_pending to _active_queue]
    C --> D[SKU request queue<br/>_pending_sku_requests]

    E[Warehouse map CSV] --> F[Physical warehouse objects<br/>Shelves, goals, picker aisles, packaging, replenishment]
    F --> G[SKU-to-shelf assignment<br/>_sku_to_shelves]

    D --> H[Match next SKU request<br/>to an available Shelf]
    G --> H
    H --> I[AGV request queue<br/>Warehouse.request_queue]
    H --> J[Order lookup<br/>_shelf_to_order]

    I --> K[AGV policy / heuristic<br/>fetch requested shelves]
    K --> L[AGV delivers shelf<br/>to pickerwall]
    J --> L

    L --> M[Pickerwall work queue<br/>_pickerwall_pending]
    L --> N[Refill request queue<br/>with next SKU request]
    N --> I

    M --> O[Picker claims work<br/>PickerClaim]
    O --> P[Picker task batch<br/>PickerTask]
    P --> Q[Picker walks, picks,<br/>and reduces Shelf.capacity]

    Q --> R{Shelf depleted?}
    R -->|yes| S[Spawn replenishment shelf<br/>at replenishment location]
    S --> G
    S --> N

    Q --> T[Shelf fulfilled?]
    T -->|yes| U[AGV displacement<br/>return shelf from pickerwall to storage]

    Q --> V[Picker walks to packaging]
    V --> W[Packaging progress<br/>_packaging_slots]
    W --> X{delivered >= required?}
    X -->|yes| Y[Order complete]
```

## Reading Guide

The system digests orders in two conversions:

1. Abstract order demand becomes physical shelf movement:
   `Order -> SKUEntry -> Shelf -> AGV request`.
2. Delivered shelves become human/picker work:
   `Shelf at pickerwall -> PickerClaim -> PickerTask -> packaging progress`.

The highest-leverage queues to watch are:

- `_pending_sku_requests`
- `Warehouse.request_queue`
- `_pickerwall_pending`
- `_packaging_slots`
