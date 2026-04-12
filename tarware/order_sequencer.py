from __future__ import annotations

import collections
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import pandas as pd

if TYPE_CHECKING:
    from tarware.warehouse import Shelf

logger = logging.getLogger(__name__)


def _hhmmss_to_seconds(hhmmss: int) -> int:
    """Convert a HHMMSS integer to seconds from midnight."""
    hh = hhmmss // 10000
    mm = (hhmmss % 10000) // 100
    ss = hhmmss % 100
    return hh * 3600 + mm * 60 + ss


@dataclass
class Order:
    order_number: str
    date_created: int           # YYYYMMDD
    time_created_seconds: int   # seconds from midnight
    order_type: str
    skus: List[SKUEntry] = field(default_factory=list)
    priority: int = 0           # reserved for future priority queue support

    def __lt__(self, other: "Order") -> bool:
        return (self.priority, self.date_created, self.time_created_seconds) < (
            other.priority, other.date_created, other.time_created_seconds
        )


@dataclass
class SKUEntry:
    sku: int
    quantity: int


class OrderSequencer:
    """Bridges the order CSV and the Warehouse environment.

    Loads orders, assigns SKUs to shelves, and time-gates order release
    into an active FIFO queue keyed by simulated seconds.
    """

    def __init__(self, csv_path: str | Path, steps_per_simulated_second: float = 1.0):
        self._steps_per_second = steps_per_simulated_second

        df = pd.read_csv(csv_path).dropna(subset=["SKU", "Time Created"])
        all_orders_dict: dict[str, Order] = {}
        for _, row in df.iterrows():
            order_number = str(row["Order #"])
            if order_number not in all_orders_dict:
                date_created = int(row["Date Created"])
                time_created_seconds = _hhmmss_to_seconds(int(row["Time Created"]))
                order_type = str(row.get("Order Type", ""))
                order = Order(
                    order_number=order_number,
                    date_created=date_created,
                    time_created_seconds=time_created_seconds,
                    order_type=order_type,
                )
                all_orders_dict[order_number] = order
            order = all_orders_dict[order_number]
            sku = int(row["SKU"])
            quantity = int(row["Shipped Quantity"])
            order.skus.append(SKUEntry(sku=sku, quantity=quantity))

        all_orders = list(all_orders_dict.values())
        all_orders.sort(key=lambda o: (o.date_created, o.time_created_seconds))
        self._pending: List[Order] = all_orders

        self._active_queue: collections.deque[Order] = collections.deque()
        self._pending_sku_requests: collections.deque[Tuple[SKUEntry, Order]] = collections.deque()
        self._sku_to_shelves: Dict[int, List["Shelf"]] = {}

        # SKUs sorted by descending frequency so the most-requested are assigned first
        sku_counts = df["SKU"].dropna().astype(int).value_counts()
        self._unique_skus: List[int] = list(sku_counts.index)

        logger.info(
            "OrderSequencer loaded: orders=%d unique_skus=%d "
            "first_release_d=%d first_release_s=%d last_release_d=%d last_release_s=%d steps_per_second=%.2f",
            len(self._pending),
            len(self._unique_skus),
            self._pending[0].date_created if self._pending else 0,
            self._pending[0].time_created_seconds if self._pending else 0,
            self._pending[-1].date_created if self._pending else 0,
            self._pending[-1].time_created_seconds if self._pending else 0,
            self._steps_per_second,
        )

    def get_unique_skus(self) -> List[int]:
        """Return unique SKUs found in the order file."""
        return list(self._unique_skus)

    def initialize_shelf_sku_map(self, shelfs: Sequence["Shelf"]) -> None:
        """Assign one SKU per shelf and build the SKU->shelf lookup.

        Shelves are shuffled before assignment to scatter SKUs randomly
        across the warehouse rather than clustering by frequency.
        """
        self._sku_to_shelves = {}
        shuffled = list(shelfs)
        random.shuffle(shuffled)
        for i, shelf in enumerate(shuffled):
            sku = self._unique_skus[i % len(self._unique_skus)]
            shelf.sku = sku
            self._sku_to_shelves.setdefault(sku, []).append(shelf)

        covered = len(set(self._unique_skus) & set(self._sku_to_shelves.keys()))
        logger.info(
            "SKU->shelf map initialised: shelves=%d unique_skus_in_orders=%d skus_with_shelf=%d",
            len(shelfs),
            len(self._unique_skus),
            covered,
        )

    def release_pending_orders(self, current_step: int) -> List[Order]:
        """Move orders whose release time has been reached into the active queue.

        Each released order is exploded into per-SKU entries in
        _pending_sku_requests so the warehouse can request one shelf per SKU.
        """
        simulated_seconds = current_step / self._steps_per_second
        released: List[Order] = []
        while self._pending and self._pending[0].time_created_seconds <= simulated_seconds:
            order = self._pending.pop(0)
            self._active_queue.append(order)
            for sku_entry in order.skus:
                self._pending_sku_requests.append((sku_entry, order))
            released.append(order)
            logger.debug(
                "order_released: step=%d sim_s=%.1f order=%s skus=%s release_d=%d release_s=%d "
                "pending_remaining=%d active_queue=%d sku_requests=%d",
                current_step, simulated_seconds,
                order.order_number,
                [(se.sku, se.quantity) for se in order.skus],
                order.date_created, order.time_created_seconds,
                len(self._pending), len(self._active_queue),
                len(self._pending_sku_requests),
            )
        if released:
            logger.info(
                "step=%d sim_s=%.1f released=%d pending=%d active=%d sku_requests=%d",
                current_step, simulated_seconds, len(released),
                len(self._pending), len(self._active_queue),
                len(self._pending_sku_requests),
            )
        return released

    def reset(self) -> None:
        """Restore queues to initial state. Call from Warehouse.reset()."""
        requeued = len(self._active_queue)
        self._pending.extend(self._active_queue)
        self._active_queue.clear()
        self._pending_sku_requests.clear()
        self._pending.sort(key=lambda o: (o.date_created, o.time_created_seconds))
        logger.info(
            "OrderSequencer reset: re-queued=%d total_pending=%d",
            requeued, len(self._pending),
        )

    def next_order_shelf(
        self, candidates: Sequence["Shelf"]
    ) -> Optional[Tuple["Shelf", "Order"]]:
        """Pop the next pending SKU request and return the matching (shelf, order).

        Only shelves in candidates are eligible. Returns None if the queue is
        empty or no eligible shelf exists for the front-most SKU.
        """
        if not self._pending_sku_requests:
            logger.debug("next_order_shelf: sku request queue empty, returning None")
            return None

        sku_entry, order = self._pending_sku_requests.popleft()
        candidate_set = set(candidates)

        for shelf in self._sku_to_shelves.get(sku_entry.sku, []):
            if shelf in candidate_set:
                logger.debug(
                    "next_order_shelf: order=%s sku=%d qty=%d -> shelf_id=%d (sku_requests_remaining=%d)",
                    order.order_number, sku_entry.sku, sku_entry.quantity, shelf.id,
                    len(self._pending_sku_requests),
                )
                return shelf, order

        logger.warning(
            "next_order_shelf: order=%s sku=%d has no available shelf in %d candidates "
            "(shelves_for_sku=%d) -- SKU request consumed without fulfillment",
            order.order_number, sku_entry.sku, len(candidates),
            len(self._sku_to_shelves.get(sku_entry.sku, [])),
        )
        return None

    def sku_requests_for_sku(self, sku: int) -> bool:
        """Return True if any pending SKU request has the given SKU."""
        return any(se.sku == sku for se, _ in self._pending_sku_requests)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def active_count(self) -> int:
        return len(self._active_queue)
