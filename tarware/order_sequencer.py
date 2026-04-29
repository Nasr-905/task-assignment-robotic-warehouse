from __future__ import annotations

import collections
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Set

import pandas as pd

if TYPE_CHECKING:
    from tarware.warehouse import LogicalBin

logger = logging.getLogger(__name__)


# Operating-hour time model:
# The simulation only runs during operating hours. Stepping advances
# "active" sim seconds; the gap between ACTIVE_END_HOUR and
# ACTIVE_START_HOUR (next day) is collapsed to zero, and orders that
# arrived during the gap are released as a single batch at the start
# of the next operating window. Step 0 corresponds to ACTIVE_START_HOUR
# on the earliest order date.
ACTIVE_START_HOUR = 9    # 9am - sim begins each day at this hour
ACTIVE_END_HOUR   = 23   # 11pm - sim jumps to next day's start hour
SECONDS_PER_HOUR  = 3600
SECONDS_PER_DAY   = 24 * SECONDS_PER_HOUR


def _find_column(df: pd.DataFrame, stripped_name: str) -> Optional[str]:
    """Return the actual CSV column whose stripped name matches."""
    for column in df.columns:
        if str(column).strip() == stripped_name:
            return column
    return None


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
    # Absolute release time in simulated seconds, measured from the
    # sequencer epoch (midnight of the earliest date_created seen).
    # Set by OrderSequencer after loading; needed so the time gate keeps
    # working across day boundaries.
    release_seconds: int = 0

    def __lt__(self, other: "Order") -> bool:
        return (self.priority, self.release_seconds) < (
            other.priority, other.release_seconds
        )


@dataclass
class SKUEntry:
    sku: int
    quantity: int
    unit_cube: float = 0.0


class OrderSequencer:
    """Bridges the order CSV and the Warehouse environment.

    Loads orders, assigns SKUs to shelves, and time-gates order release
    into an active FIFO queue keyed by simulated seconds.
    """

    def __init__(
        self,
        csv_path: str | Path,
        steps_per_simulated_second: float = 1.0,
        active_start_hour: int = ACTIVE_START_HOUR,
        active_end_hour: int = ACTIVE_END_HOUR,
    ):
        if not 0 <= active_start_hour < active_end_hour <= 24:
            raise ValueError(
                f"Invalid operating window {active_start_hour}h-{active_end_hour}h"
            )
        self._steps_per_second = steps_per_simulated_second
        self._active_start_s = active_start_hour * SECONDS_PER_HOUR
        self._active_end_s = active_end_hour * SECONDS_PER_HOUR
        self._active_duration_s = self._active_end_s - self._active_start_s

        df = pd.read_csv(csv_path).dropna(subset=["SKU", "Time Created"])
        unit_cube_column = _find_column(df, "Unit cube F")
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
            unit_cube = float(row.get(unit_cube_column, 0.0) or 0.0) if unit_cube_column else 0.0
            order.skus.append(SKUEntry(sku=sku, quantity=quantity, unit_cube=unit_cube))

        all_orders = list(all_orders_dict.values())
        all_orders.sort(key=lambda o: (o.date_created, o.time_created_seconds))

        # Anchor sim time to ACTIVE_START_HOUR of the earliest order date.
        # Each order's release_seconds is the *active* sim time (operating
        # hours only) at which it should release. Orders arriving outside
        # the active window are batched to the next window's start.
        if all_orders:
            self._epoch_date: int = all_orders[0].date_created
            epoch_dt = datetime.strptime(str(self._epoch_date), "%Y%m%d").date()
            for order in all_orders:
                order_dt = datetime.strptime(str(order.date_created), "%Y%m%d").date()
                days_offset = (order_dt - epoch_dt).days
                order.release_seconds = self._wall_to_active_seconds(
                    days_offset, order.time_created_seconds
                )
            # release_seconds may not match the (date, time-of-day) order
            # because late-night orders get pushed to next day's window.
            all_orders.sort(key=lambda o: o.release_seconds)
        else:
            self._epoch_date = 0

        self._pending: List[Order] = all_orders

        self._active_queue: collections.deque[Order] = collections.deque()
        self._pending_sku_requests: collections.deque[Tuple[SKUEntry, Order]] = collections.deque()

        # SKUs sorted by descending frequency so the most-requested are assigned first
        sku_counts = df["SKU"].dropna().astype(int).value_counts()
        self._unique_skus: List[int] = list(sku_counts.index)
        if unit_cube_column:
            self._sku_unit_cube: Dict[int, float] = (
                df.dropna(subset=["SKU"])
                .assign(SKU=lambda d: d["SKU"].astype(int))
                .groupby("SKU")[unit_cube_column]
                .median()
                .fillna(0.0)
                .astype(float)
                .to_dict()
            )
        else:
            self._sku_unit_cube = {sku: 0.0 for sku in self._unique_skus}
        self._sku_to_bins: Dict[int, List["LogicalBin"]] = {}

        overnight_at_start = sum(
            1 for o in self._pending if o.release_seconds == 0
        )
        logger.info(
            "OrderSequencer loaded: orders=%d unique_skus=%d epoch_date=%d "
            "active_window=%dh-%dh active_day_s=%d "
            "first_release_d=%d first_release_s=%d first_active_s=%d "
            "last_release_d=%d last_release_s=%d last_active_s=%d "
            "overnight_at_step0=%d steps_per_second=%.2f",
            len(self._pending),
            len(self._unique_skus),
            self._epoch_date,
            self._active_start_s // SECONDS_PER_HOUR,
            self._active_end_s // SECONDS_PER_HOUR,
            self._active_duration_s,
            self._pending[0].date_created if self._pending else 0,
            self._pending[0].time_created_seconds if self._pending else 0,
            self._pending[0].release_seconds if self._pending else 0,
            self._pending[-1].date_created if self._pending else 0,
            self._pending[-1].time_created_seconds if self._pending else 0,
            self._pending[-1].release_seconds if self._pending else 0,
            overnight_at_start,
            self._steps_per_second,
        )

    def _wall_to_active_seconds(self, days_offset: int, t_of_day_s: int) -> int:
        """Map a wall-clock arrival ``(days_offset, t_of_day_s)`` to the
        active sim second at which the order should release.

        Orders arriving before the active window release at the start of
        that day's window; orders arriving after the active window release
        at the start of the next day's window; orders inside the window
        release at their offset into it.
        """
        if t_of_day_s < self._active_start_s:
            return days_offset * self._active_duration_s
        if t_of_day_s < self._active_end_s:
            return (
                days_offset * self._active_duration_s
                + (t_of_day_s - self._active_start_s)
            )
        return (days_offset + 1) * self._active_duration_s

    def get_unique_skus(self) -> List[int]:
        """Return unique SKUs found in the order file."""
        return list(self._unique_skus)

    def get_sku_unit_cube(self, sku: int) -> float:
        """Return the median unit cube observed for a SKU in the order file."""
        return float(self._sku_unit_cube.get(sku, 0.0))

    def _quantity_for_bin(self, bin_: "LogicalBin", sku: int) -> int:
        unit_cube = self.get_sku_unit_cube(sku)
        if unit_cube <= 0:
            return 0
        return max(0, int(math.floor(bin_.usable_volume_ft3 / unit_cube)))

    def release_pending_orders(self, current_step: int) -> List[Order]:
        """Move orders whose release time has been reached into the active queue.

        Each released order is exploded into per-SKU entries in
        _pending_sku_requests so the warehouse can request one shelf per SKU.
        """
        simulated_seconds = current_step / self._steps_per_second
        released: List[Order] = []
        while self._pending and self._pending[0].release_seconds <= simulated_seconds:
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
        self._pending.sort(key=lambda o: o.release_seconds)
        logger.info(
            "OrderSequencer reset: re-queued=%d total_pending=%d",
            requeued, len(self._pending),
        )

    def sort_pending_sku_requests(self, pickerwall_skus: Set[int]) -> None:
        """Reorder pending SKU requests so those with SKUs in pickerwall_skus come first."""
        if pickerwall_skus is None:
            return
        pending_skus = list(self._pending_sku_requests)
        pending_skus.sort(key=lambda pair: (0 if pair[0].sku in pickerwall_skus else 1))
        self._pending_sku_requests.clear()
        self._pending_sku_requests.extend(pending_skus)

    def next_order_bin(
        self, candidates: Sequence["LogicalBin"]
    ) -> Optional[Tuple["LogicalBin", "Order"]]:
        """Pop the next pending SKU request and return the matching (bin, order).

        Only bins in ``candidates`` are eligible. Returns None if the queue is
        empty or no eligible bin exists for the front-most SKU.
        """
        if not self._pending_sku_requests:
            logger.debug("next_order_bin: sku request queue empty, returning None")
            return None

        sku_entry, order = self._pending_sku_requests.popleft()
        candidate_set = set(candidates)

        for bin_ in self._sku_to_bins.get(sku_entry.sku, []):
            if bin_ in candidate_set:
                logger.debug(
                    "next_order_bin: order=%s sku=%d qty=%d -> bin_id=%d (sku_requests_remaining=%d)",
                    order.order_number, sku_entry.sku, sku_entry.quantity, bin_.id,
                    len(self._pending_sku_requests),
                )
                return bin_, order

        logger.warning(
            "next_order_bin: order=%s sku=%d has no available bin in %d candidates "
            "(bins_for_sku=%d) -- SKU request consumed without fulfillment",
            order.order_number, sku_entry.sku, len(candidates),
            len(self._sku_to_bins.get(sku_entry.sku, [])),
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

    @property
    def has_future_orders(self) -> bool:
        """True while orders remain time-gated awaiting release."""
        return len(self._pending) > 0

    @property
    def is_fully_drained(self) -> bool:
        """True when the CSV is fully consumed: no future releases and no
        SKU requests waiting to be claimed by an AGV. Caller should also
        verify the warehouse-side pickerwall queue is empty before calling
        the run finished.
        """
        return len(self._pending) == 0 and len(self._pending_sku_requests) == 0
