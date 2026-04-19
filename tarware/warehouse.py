import json
import logging
from collections import deque
import math
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from pathlib import Path

import gymnasium as gym
import networkx as nx
import pandas as pd
import numpy as np
import pyastar2d
from gymnasium import spaces
from tarware.definitions import (Action, AgentType, Direction,
                                 RewardType, CollisionLayers)
from tarware.spaces import observation_map
from tarware.utils import find_sections, get_next_micro_action
from tarware.order_sequencer import Order, OrderSequencer, SKUEntry
from tarware.human_factors import (
    HumanFactorsConfig,
    PhysicalTimeConfig,
    PickerEffortProfile,
    PickerHumanFactorsState,
)

logger = logging.getLogger(__name__)

_FIXING_CLASH_TIME = 4
_STUCK_THRESHOLD = 5
_PICKER_BLOCKED_REROUTE_THRESHOLD = 4  # consecutive blocked steps before a picker detours
_PICK_TICKS = 3  # Steps a picker spends picking from a shelf
# Tiles:
# - 0: AGV highway
# - 1: shelf/storage
# - 2: pickerwall
# - 3: picker highway
# - 4: packaging
# - 5: replenishment (AGV take-only)
# - 6: shared highway
# - 7: AGV idle zone (non-shared zone that AGVs can idle in)
_AGV_WALKABLE_TILES = {0, 6, 7}
_AGV_LOADABLE_TILES = {0}
_PICKER_WALKABLE_TILES = {3, 4, 6}
_PICKER_LOADABLE_TILES = {3, 4}
_SHARED_HIGHWAY_TILE = 6
BIN_VOLUME_FT3 = 2.68
BIN_USABLE_FRACTION = 0.85

# Pickerwall pending priority weights (higher score = picked sooner)
PICKER_PRIORITY_COMPLETION_WEIGHT = 0.0   # boost orders close to finishing
PICKER_PRIORITY_COMPLETION_EXPONENT = 0.5 # non-linear curve (sqrt)
PICKER_PRIORITY_SKU_BATCH_WEIGHT = 0.0    # reward batching same-SKU picks
PICKER_PRIORITY_SKU_BATCH_CAP = 5         # normalize sku count up to this value
PICKER_PRIORITY_AGE_WEIGHT = 1.0          # prevent starvation of old orders


class BinCellType(Enum):
    STORAGE = "storage"
    PICKERWALL = "pickerwall"
    REPLENISHMENT = "replenishment"


@dataclass(eq=False)
class LogicalBin:
    """Mobile bin carried one-at-a-time by an AGV.

    A bin lives in a Shelf slot until an AGV picks it up, is carried to a
    destination shelf (pickerwall / replenishment / storage), and is then
    deposited into a free slot there. SKU + quantity track inventory inside
    the bin; volume fields cap how much stock the bin can hold.

    ``shelf_id`` / ``slot_index`` are set while the bin is resting in a
    shelf and cleared while the bin is being carried by an AGV.
    """
    id: int
    x: int
    y: int
    cell_type: BinCellType          # type of the shelf this bin was spawned in
    volume_ft3: float
    usable_fraction: float
    sku: Optional[int] = None
    quantity: int = 0
    used_volume_ft3: float = 0.0
    shelf_id: Optional[int] = None
    slot_index: Optional[int] = None
    depleted: bool = False
    fulfilled: bool = False
    from_replenishment: bool = False  # True for a bin that has just been spawned at replenishment
    on_grid: bool = True              # False once the bin has been retired from circulation

    @property
    def usable_volume_ft3(self) -> float:
        return self.volume_ft3 * self.usable_fraction

    @property
    def remaining_volume_ft3(self) -> float:
        return max(0.0, self.usable_volume_ft3 - self.used_volume_ft3)

    @property
    def on_shelf(self) -> bool:
        return self.shelf_id is not None

class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y

class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, dir_: Direction, agent_type: AgentType):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.dir = dir_
        self.req_action: Optional[Action] = None
        self.carrying_bin: Optional["LogicalBin"] = None
        self.canceled_action = None
        self.has_delivered = False
        self.path = None
        self.busy = False
        self.fixing_clash = 0
        self.type = agent_type
        self.target = 0
        self.motion_credit_cells: float = 0.0
        self.speed_limited_this_step: bool = False

    def req_location(self, grid_size) -> Tuple[int, int]:
        if self.req_action != Action.FORWARD:
            return self.x, self.y
        elif self.dir == Direction.UP:
            return self.x, max(0, self.y - 1)
        elif self.dir == Direction.DOWN:
            return self.x, min(grid_size[0] - 1, self.y + 1)
        elif self.dir == Direction.LEFT:
            return max(0, self.x - 1), self.y
        elif self.dir == Direction.RIGHT:
            return min(grid_size[1] - 1, self.x + 1), self.y

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.req_action == Action.RIGHT:
            return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        elif self.req_action == Action.LEFT:
            return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        else:
            return self.dir

class Shelf(Entity):
    """Static shelf fixture with a fixed number of bin slots.

    A shelf is pinned to one grid cell for the lifetime of the warehouse.
    Bins move in and out of its slots via ``place_bin`` / ``remove_bin``;
    the shelf itself never moves. Storage, pickerwall, and replenishment
    cells are all represented as Shelf instances distinguished by
    ``cell_type``.
    """
    counter = 0

    def __init__(self, x: int, y: int, cell_type: "BinCellType", num_slots: int):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)
        self.cell_type: "BinCellType" = cell_type
        self.bin_slots: List[Optional["LogicalBin"]] = [None] * num_slots

    @property
    def num_slots(self) -> int:
        return len(self.bin_slots)

    @property
    def bin_count(self) -> int:
        return sum(1 for b in self.bin_slots if b is not None)

    @property
    def is_full(self) -> bool:
        return all(b is not None for b in self.bin_slots)

    @property
    def is_empty(self) -> bool:
        return all(b is None for b in self.bin_slots)

    def first_empty_slot(self) -> Optional[int]:
        for i, b in enumerate(self.bin_slots):
            if b is None:
                return i
        return None

    def place_bin(self, bin_: "LogicalBin", slot_index: Optional[int] = None) -> int:
        if slot_index is None:
            slot_index = self.first_empty_slot()
        assert slot_index is not None, f"shelf {self.id} is full"
        assert self.bin_slots[slot_index] is None, (
            f"shelf {self.id} slot {slot_index} already occupied"
        )
        self.bin_slots[slot_index] = bin_
        bin_.shelf_id = self.id
        bin_.slot_index = slot_index
        bin_.x = self.x
        bin_.y = self.y
        return slot_index

    def remove_bin(self, slot_index: int) -> "LogicalBin":
        bin_ = self.bin_slots[slot_index]
        assert bin_ is not None, f"shelf {self.id} slot {slot_index} is empty"
        self.bin_slots[slot_index] = None
        bin_.shelf_id = None
        bin_.slot_index = None
        return bin_

class StuckCounter:
    def __init__(self, position: Tuple[int, int]):
        self.position = position
        self.count = 0

    def update(self, new_position: Tuple[int, int]):
        if new_position == self.position:
            self.count += 1
        else:
            self.count = 0
            self.position = new_position

    def reset(self, position=None):
        self.count = 0
        if position:
            self.position = position

class PickerState(Enum):
    IDLE = 0
    WALKING_TO_SHELF = 1
    PICKING = 2
    WAITING_FOR_SHELF = 3   # next SKU's shelf not yet at pickerwall; picker idles
    WALKING_TO_PACKAGING = 4
    AT_PACKAGING = 5
    DISTRACTED = 6  # reserved for future human-factors extension


@dataclass
class PickerClaim:
    """One SKU-quantity claim a picker has taken from the pending request queue."""
    bin_id: int            # id of the LogicalBin to pick from; -1 if not yet at pickerwall
    sku_entry: "SKUEntry"  # which SKU and how many units to pick
    order_number: str      # which order this fulfils
    order: "Order"         # full Order object (needed for packaging slot setup)
    picked: bool = False   # set to True once the picker physically picks this item


@dataclass
class PickerTask:
    """Capacity-limited batch of claims spanning potentially multiple orders.

    Claims are ordered by the simple picker policy, with same-shelf claims
    grouped together. ``current_claim_index`` tracks which claim the picker is
    currently working towards.
    """
    claims: List["PickerClaim"]
    current_claim_index: int = 0


class Picker(Entity):
    counter = 0
    CAPACITY: int = 10  # default number of item-units a picker can carry per trip

    def __init__(self, x: int, y: int, dir_: Direction):
        Picker.counter += 1
        super().__init__(Picker.counter, x, y)
        self.dir = dir_
        self.type = AgentType.PICKER
        self.state: PickerState = PickerState.IDLE
        self.task: Optional[PickerTask] = None
        self.path: List = []
        self.pick_ticks_remaining: int = 0    # counts down the steps spent picking the current claim's SKU; set to 0 when not actively picking
        self.capacity: int = Picker.CAPACITY
        self.blocked_ticks: int = 0           # consecutive steps spent waiting on a blocked cell
        self.fixing_clash: int = 0            # cooldown after rerouting; mirrors Agent.fixing_clash
        self.home_zone: Optional[int] = None
        self.stalled: bool = False
        self.packaging_location: Optional[Tuple[int, int]] = None
        self.motion_credit_cells: float = 0.0
        self.speed_limited_this_step: bool = False


class Warehouse(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        map_csv_path: Path,
        map_json_path: Path,
        order_csv_path: Path,
        num_agvs: int,
        num_pickers: int,
        observation_type: str,
        request_queue_size: int,
        steps_per_simulated_second: float,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType,
        normalised_coordinates: bool = False,
        fit_width: Optional[int] = None,
        fit_height: Optional[int] = None,
    ):
        """Multi-agent robotic warehouse gym environment."""
        self.steps_per_simulated_second = max(1e-6, float(steps_per_simulated_second))
        self.time_config = PhysicalTimeConfig.from_env(self.steps_per_simulated_second)
        self.bin_volume_ft3 = float(os.getenv("TARWARE_BIN_VOLUME_FT3", str(BIN_VOLUME_FT3)))
        self.bin_usable_fraction = float(
            os.getenv("TARWARE_BIN_USABLE_FRACTION", str(BIN_USABLE_FRACTION))
        )
        self._make_order_sequencer_from_csv(order_csv_path, self.steps_per_simulated_second)
        self._make_layout_from_csv(map_csv_path, map_json_path, fit_width, fit_height)

        self.num_agvs = num_agvs
        self.num_pickers = num_pickers
        self.num_agents = num_agvs  # Only AGVs receive macro actions
    
        self._agent_types = [AgentType.AGV for _ in range(num_agvs)]

        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps
        self.picker_policy = os.getenv("TARWARE_PICKER_POLICY", "fifo").lower()
        self.picker_zone_overflow = os.getenv("TARWARE_PICKER_ZONE_OVERFLOW", "adjacent").lower()
        self.picker_stall_probability = float(os.getenv("TARWARE_PICKER_STALL_PROBABILITY", "0.0"))
        self.use_sku_size_pick_time = os.getenv("TARWARE_PICKER_USE_SKU_SIZE_TIME", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self.pick_base_ticks = int(os.getenv("TARWARE_PICK_BASE_TICKS", str(_PICK_TICKS)))
        self.pick_unit_cube_tick_scale = float(os.getenv("TARWARE_PICK_UNIT_CUBE_TICK_SCALE", "1.0"))
        self.human_factors_config = HumanFactorsConfig.from_env(
            map_name=Path(map_csv_path).stem,
            time_config=self.time_config,
            fallback_pick_base_ticks=self.pick_base_ticks,
            fallback_pick_unit_cube_tick_scale=self.pick_unit_cube_tick_scale,
        )

        self.action_size = len(self.action_id_to_coords_map) + 1
        self.action_space = spaces.Tuple(tuple(self.num_agents * [spaces.Discrete(self.action_size)]))

        self.observation_space_mapper = observation_map[observation_type](
            self.num_agvs,
            self.grid_size,
            self.num_non_goal_actions,
            self.num_pickerwall_actions,
            normalised_coordinates,
        )
        self.observation_space = spaces.Tuple(tuple(self.observation_space_mapper.ma_spaces))

        self.request_queue_size = request_queue_size
        self.request_queue = []
        self._step_deliveries = 0

        # rack_groups clusters physical rack cells; dedupe the per-slot action entries.
        self.rack_groups = find_sections(
            [(r, c) for (r, c) in self.shelf_locs]
            + [(y, x) for (x, y) in self.replenishment_locs]
        )
        self.agents: List[Agent] = []
        self.pickers: List[Picker] = []
        self.stuck_counters = []
        self.renderer = None
        self._bin_to_order: Dict[int, list] = {}  # bin_id -> [(sku_entry, Order), ...]
        self._packaging_slots: Dict[str, Dict[str, Any]] = {}  # order_number -> {required, delivered, station}
        self._pickerwall_pending: deque = deque()  # (bin_id, SKUEntry, Order) populated on delivery
        self._reserved_slots: Dict[Tuple[int, int], int] = {}  # (shelf_id, slot_idx) -> bin_id
        self._next_bin_id: int = 0
        self._bins_by_id: Dict[int, "LogicalBin"] = {}
        self._picker_hf_profile_by_id: Dict[int, PickerEffortProfile] = {}
        self._picker_hf_state_by_id: Dict[int, PickerHumanFactorsState] = {}
        self._picker_hf_episode_delay_steps = 0
        self._picker_hf_episode_failed_pick_delays = 0
        self._latest_picker_diagnostics: List[Dict[str, Any]] = []
        self._replenishment_request_step_by_bin: Dict[int, int] = {}
        self._configure_motion_model()

    def steps_to_simulated_seconds(self, steps: int) -> float:
        return max(0, int(steps)) * self.time_config.simulated_seconds_per_step

    def steps_to_real_seconds(self, steps: int) -> float:
        return max(0, int(steps)) * self.time_config.real_seconds_per_step

    def simulated_seconds_to_steps(self, seconds: float, ceil: bool = True) -> int:
        return self.time_config.simulated_seconds_to_steps(seconds, ceil=ceil)

    def rate_per_second_to_per_step(self, value_per_second: float) -> float:
        return self.time_config.per_second_to_per_step(value_per_second)

    def agv_nominal_cells_per_step(self) -> float:
        return self.time_config.agv_nominal_cells_per_step()

    def picker_nominal_cells_per_step(self) -> float:
        return self.time_config.picker_nominal_cells_per_step()

    def _configure_motion_model(self) -> None:
        self._use_physical_speed_model = os.getenv(
            "TARWARE_USE_PHYSICAL_SPEEDS", "0"
        ).lower() in ("1", "true", "yes")

        if self._use_physical_speed_model:
            self._agv_cells_per_step_configured = max(0.0, self.agv_nominal_cells_per_step())
            self._picker_cells_per_step_configured = max(0.0, self.picker_nominal_cells_per_step())
        else:
            self._agv_cells_per_step_configured = max(
                0.0, float(os.getenv("TARWARE_AGV_CELLS_PER_STEP", "1.0"))
            )
            self._picker_cells_per_step_configured = max(
                0.0, float(os.getenv("TARWARE_PICKER_CELLS_PER_STEP", "1.0"))
            )

        # This engine executes at most one tile move per entity per env step.
        self._agv_cells_per_step_effective = min(1.0, self._agv_cells_per_step_configured)
        self._picker_cells_per_step_effective = min(1.0, self._picker_cells_per_step_configured)

        if self._agv_cells_per_step_configured > 1.0 or self._picker_cells_per_step_configured > 1.0:
            logger.warning(
                "motion config requests >1.0 cells/step; clamped to 1.0 in discrete step engine "
                "(agv=%.3f picker=%.3f)",
                self._agv_cells_per_step_configured,
                self._picker_cells_per_step_configured,
            )

    def _consume_motion_credit(self, entity: Entity, cells_per_step: float) -> bool:
        if cells_per_step <= 0.0:
            return False
        current = float(getattr(entity, "motion_credit_cells", 0.0))
        current = min(1.0, current + float(cells_per_step))
        if current + 1e-9 < 1.0:
            setattr(entity, "motion_credit_cells", current)
            return False
        setattr(entity, "motion_credit_cells", max(0.0, current - 1.0))
        return True

    @property
    def targets_agvs(self):
        return [agent.target for agent in self.agents]

    def _make_order_sequencer_from_csv(self, order_csv_path: Path, steps_per_simulated_second: float) -> "OrderSequencer":
        self.order_sequencer = OrderSequencer(
            order_csv_path, steps_per_simulated_second=steps_per_simulated_second
        )

        logger.info(
            "Warehouse: built OrderSequencer from path=%s", order_csv_path
        )

    def _make_shelves(self) -> None:
        """Create static Shelf fixtures for every storage, pickerwall, and
        replenishment cell in the map.

        Each shelf has ``self.bins_per_shelf`` empty bin slots. Bins are
        not created here - they are populated later by the inventory
        initialisation step, one LogicalBin per assigned SKU, placed into
        storage shelf slots.
        """
        Shelf.counter = 0
        self.shelfs: List[Shelf] = []
        self.shelves_by_xy: Dict[Tuple[int, int], Shelf] = {}
        self.shelves_by_id: Dict[int, Shelf] = {}

        def add_shelf(x: int, y: int, cell_type: BinCellType) -> None:
            shelf = Shelf(x, y, cell_type, self.bins_per_shelf)
            self.shelfs.append(shelf)
            self.shelves_by_xy[(x, y)] = shelf
            self.shelves_by_id[shelf.id] = shelf

        for row, col in self.shelf_locs:
            add_shelf(col, row, BinCellType.STORAGE)
        for x, y in self.goals:
            add_shelf(x, y, BinCellType.PICKERWALL)
        for x, y in self.replenishment_locs:
            add_shelf(x, y, BinCellType.REPLENISHMENT)

        self.storage_shelves: List[Shelf] = [
            s for s in self.shelfs if s.cell_type == BinCellType.STORAGE
        ]
        self.pickerwall_shelves: List[Shelf] = [
            s for s in self.shelfs if s.cell_type == BinCellType.PICKERWALL
        ]
        self.replenishment_shelves: List[Shelf] = [
            s for s in self.shelfs if s.cell_type == BinCellType.REPLENISHMENT
        ]

    def _make_layout_from_csv(self, map_csv_path: Path, map_json_path: Path, fit_width: Optional[int] = None, fit_height: Optional[int] = None) -> None:
        """Build the warehouse layout from a CSV tile map.

        Tile encoding: 0=highway, 1=shelf/storage, 2=pickerwall, 3=picker_highway,
        4=packaging, 5=replenishment (take-only), 6=shared_highway, 9=blank.
        """

        df = pd.read_csv(map_csv_path, header=None).fillna(0)
        tile_grid = df.values.astype(int)  # shape (num_rows, num_cols)
        num_rows, num_cols = tile_grid.shape

        with open(map_json_path, "r") as f:
            map_data = json.load(f)
        self.bins_per_shelf: int = int(map_data["bins_per_shelf"])

        self.grid_size = (num_rows, num_cols)
        self.grid = np.zeros((len(CollisionLayers), *self.grid_size), dtype=np.int32)

        # Highways: tile 6 is shared, so it is marked walkable for both groups.
        self.highways = np.zeros(self.grid_size, dtype=np.int32)
        self.picker_highways = np.zeros(self.grid_size, dtype=np.int32)
        for r in range(num_rows):
            for c in range(num_cols):
                t = tile_grid[r, c]
                if t in _AGV_WALKABLE_TILES:
                    self.highways[r, c] = 1
                if t in _PICKER_WALKABLE_TILES:
                    self.picker_highways[r, c] = 1
        self.shared_highway_locs: List[Tuple[int, int]] = [
            (c, r)
            for r in range(num_rows)
            for c in range(num_cols)
            if tile_grid[r, c] == _SHARED_HIGHWAY_TILE
        ]

        # AGV idle zones (tile 7): non-shared AGV-walkable cells where AGVs can
        # park without blocking pickers or pickerwall entries.
        self.agv_idle_zones = np.zeros(self.grid_size, dtype=np.int32)
        for r in range(num_rows):
            for c in range(num_cols):
                if tile_grid[r, c] == 7:
                    self.agv_idle_zones[r, c] = 1

        # Loadable tiles: loadings and unloadings should only occur in loadable areas,
        # which is dfiferent from highways (the shared space is ignored).
        self.loadable_tiles = np.zeros(self.grid_size, dtype=np.int32)
        self.picker_loadable_tiles = np.zeros(self.grid_size, dtype=np.int32)
        for r in range(num_rows):
            for c in range(num_cols):
                t = tile_grid[r, c]
                if t in _AGV_LOADABLE_TILES:
                    self.loadable_tiles[r, c] = 1
                if t in _PICKER_LOADABLE_TILES:
                    self.picker_loadable_tiles[r, c] = 1

        # Goals (pickerwall), packaging locations (stored as (x, y) = (col, row))
        self.goals: List[Tuple[int, int]] = [
            (c, r)
            for r in range(num_rows)
            for c in range(num_cols)
            if tile_grid[r, c] == 2
        ]
        self.num_goals = len(self.goals)
        self._build_pickerwall_zones()

        self.packaging_locations: List[Tuple[int, int]] = [
            (c, r)
            for r in range(num_rows)
            for c in range(num_cols)
            if tile_grid[r, c] == 4
        ]
        packaging_set = set(self.packaging_locations)

        # Shelf cells: column-major ordering
        self.shelf_locs: List[Tuple[int, int]] = [
            (r, c)
            for c in range(num_cols)
            for r in range(num_rows)
            if tile_grid[r, c] == 1
        ]
        # Quick (col, row) lookup set for shelf cells
        self._shelf_locs_xy_set: set = {(c, r) for (r, c) in self.shelf_locs}

        # Replenishment cells: staging zone for fresh stock (AGV take-only)
        self.replenishment_locs: List[Tuple[int, int]] = [
            (c, r)
            for r in range(num_rows)
            for c in range(num_cols)
            if tile_grid[r, c] == 5
        ]
        self._replenishment_locs_set: set = set(self.replenishment_locs)
        
        # Off-limit cells: blank tiles where no agents can go
        self.off_limit_locs: List[Tuple[int, int]] = [
            (c, r)
            for r in range(num_rows)
            for c in range(num_cols)
            if tile_grid[r, c] == 9
        ]
        
        self._make_shelves()
        self.agv_spawn_locs = np.argwhere(self.highways == 1)  # (row, col)

        # Picker spawn cells exclude packaging stations. Picker aisles may be
        # embedded in the map, not only in a contiguous bottom picker zone.
        self.picker_spawn_locs = np.array([
            (r, c)
            for r in range(num_rows)
            for c in range(num_cols)
            if self.picker_highways[r, c] == 1 and (c, r) not in packaging_set
        ], dtype=int)

        # Per-goal picker entry points: picker_highway cells adjacent to each goal
        self._goal_to_picker_entry: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for (gx, gy) in self.goals:
            entries = [
                (nx, ny)
                for (nx, ny) in [(gx - 1, gy), (gx + 1, gy), (gx, gy - 1), (gx, gy + 1)]
                if 0 <= ny < num_rows and 0 <= nx < num_cols
                and self.picker_loadable_tiles[ny, nx] == 1
            ]
            self._goal_to_picker_entry[(gx, gy)] = entries

        # Per-goal AGV entry points: highway cells adjacent to each goal (for side drop-off).
        # AGVs deposit shelves from an adjacent highway cell rather than entering the goal cell.
        self._goal_to_agv_entry: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for (gx, gy) in self.goals:
            entries = [
                (nx, ny)
                for (nx, ny) in [(gx - 1, gy), (gx + 1, gy), (gx, gy - 1), (gx, gy + 1)]
                if 0 <= ny < num_rows and 0 <= nx < num_cols
                and self.loadable_tiles[ny, nx] == 1       # AGV-walkable
            ]
            self._goal_to_agv_entry[(gx, gy)] = entries

        # Reverse: AGV-entry (col, row) -> goal (col, row)
        self._agv_entry_to_goal: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for goal_xy, entries in self._goal_to_agv_entry.items():
            for entry_xy in entries:
                self._agv_entry_to_goal[entry_xy] = goal_xy

        # Action IDs: one action per bin slot, in order:
        #   pickerwall slots, then storage slots, then replenishment slots.
        # action_id_to_coords_map: action_id -> (row, col) of the containing shelf cell
        # action_id_to_slot:       action_id -> (shelf_id, slot_index)
        self.action_id_to_coords_map: Dict[int, Tuple[int, int]] = {}
        self.action_id_to_slot: Dict[int, Tuple[int, int]] = {}

        action_id = 1
        self._pickerwall_action_id_base = action_id
        for (x, y) in self.goals:
            shelf = self.shelves_by_xy[(x, y)]
            # Define 1 action per slot in each pickerwall shelf
            for slot_idx in range(self.bins_per_shelf):
                self.action_id_to_coords_map[action_id] = (y, x)
                self.action_id_to_slot[action_id] = (shelf.id, slot_idx)
                action_id += 1
        self.num_pickerwall_actions = action_id - self._pickerwall_action_id_base

        self._storage_action_id_base = action_id
        for (r, c) in self.shelf_locs:
            shelf = self.shelves_by_xy[(c, r)]
            # Define 1 action per slot in each storage shelf
            for slot_idx in range(self.bins_per_shelf):
                self.action_id_to_coords_map[action_id] = (r, c)
                self.action_id_to_slot[action_id] = (shelf.id, slot_idx)
                action_id += 1
        self.num_storage_actions = action_id - self._storage_action_id_base

        self._replenishment_action_id_base = action_id
        for (x, y) in self.replenishment_locs:
            shelf = self.shelves_by_xy[(x, y)]
            # Define 1 action per slot in each replenishment shelf
            for slot_idx in range(self.bins_per_shelf):
                self.action_id_to_coords_map[action_id] = (y, x)
                self.action_id_to_slot[action_id] = (shelf.id, slot_idx)
                action_id += 1
        self.num_replenishment_actions = action_id - self._replenishment_action_id_base
        self.num_non_goal_actions = self.num_storage_actions + self.num_replenishment_actions

        # Idle-zone cells (tile 7): one macro action per cell so AGVs can be
        # dispatched to wait here via the normal path planner. These actions are
        # not bin slots, so TOGGLE_LOAD on arrival is a no-op (see _execute_load),
        # which cleanly terminates the macro action without side effects.
        self._idle_action_id_base = action_id
        self.idle_zone_locs: List[Tuple[int, int]] = [
            (c, r)
            for r in range(num_rows)
            for c in range(num_cols)
            if tile_grid[r, c] == 7
        ]
        self.idle_zone_action_ids: List[int] = []
        self._idle_cell_to_action_id: Dict[Tuple[int, int], int] = {}
        for (x, y) in self.idle_zone_locs:
            self.action_id_to_coords_map[action_id] = (y, x)
            self.idle_zone_action_ids.append(action_id)
            self._idle_cell_to_action_id[(x, y)] = action_id
            action_id += 1
        self.num_idle_actions = action_id - self._idle_action_id_base

        # Reverse lookup: (shelf_id, slot_idx) -> action_id. Needed by the heuristic
        # to target a specific bin slot rather than collapsing by cell.
        self.slot_to_action_id: Dict[Tuple[int, int], int] = {
            slot: aid for aid, slot in self.action_id_to_slot.items()
        }

        # longest shelf column run, used by stuck-agent resolver timeouts
        max_run = 1
        for c in range(num_cols):
            run = 0
            for r in range(num_rows):
                if tile_grid[r, c] == 1:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
        self.column_height = max_run

        # Fill out the fit widths and heights if requested
        self.render_tile_size: Optional[int] = None
        if fit_width is not None and fit_height is not None:
            self.render_tile_size = max(4, min((fit_width - 1) // num_cols - 1, (fit_height - 1) // num_rows - 1))

        logger.info(
            "Map loaded from CSV: %s | grid=%s "
            "goals=%d storage_shelves=%d replenishment_shelves=%d packaging=%d "
            "total_shelves=%d bins_per_shelf=%d bin_usable_volume_ft3=%.3f column_height=%d",
            map_csv_path, self.grid_size,
            self.num_goals, len(self.storage_shelves), len(self.replenishment_shelves),
            len(self.packaging_locations), len(self.shelfs), self.bins_per_shelf,
            self.bin_volume_ft3 * self.bin_usable_fraction, self.column_height,
        )

    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]

    def cancel_agv_motion(self, agv: Agent) -> None:
        """Abort an AGV's in-flight macro action so a new one can be dispatched.

        Safe to call only when the AGV is not carrying a bin: a carried bin has
        a reserved destination slot that would otherwise leak. Clears the path,
        busy flag, target, and pending micro action so the next call to
        attribute_macro_actions treats this AGV as fresh.
        """
        if agv.carrying_bin is not None:
            raise ValueError(
                f"cancel_agv_motion: AGV id={agv.id} is carrying a bin; "
                "cancelling would leak its reserved destination slot."
            )
        agv.path = None
        agv.busy = False
        agv.target = 0
        agv.req_action = Action.NOOP

    def find_agv_path(self, start: Tuple[int, int], goal: Tuple[int, int], care_for_agents: bool = True) -> List[Tuple[int, int]]:
        """A* path for an AGV on the highway grid. Returns [] if no path exists."""
        grid = np.zeros(self.grid_size)
        if care_for_agents:
            grid += self.grid[CollisionLayers.AGVS]
            grid += self.grid[CollisionLayers.PICKERS]
        grid[goal[0], goal[1]] = 0

        # AGVs travel only on highways but can access their specific target cell
        grid += (1 - self.highways)
        grid[goal[0], goal[1]] -= not self._is_highway(goal[1], goal[0])

        start_fix = (0, 0)
        grid[start[0]+start_fix[0], start[1]+start_fix[1]] = 0
        grid = [list(map(int, l)) for l in (grid!=0)]
        grid = np.array(grid, dtype=np.float32)
        grid[np.where(grid == 1)] = np.inf
        grid[np.where(grid == 0)] = 1
        astar_path = pyastar2d.astar_path(grid, np.add(start, start_fix), goal, allow_diagonal=False)
        if astar_path is not None:
            astar_path = [tuple(x) for x in list(astar_path)]
            astar_path = astar_path[1 - int(grid[start[0], start[1]] > 1):]

        if astar_path:
            return [(x, y) for y, x in astar_path]
        else:
            return []
        
    def find_agv_path_through_adjacent_loc(self, start: Tuple[int, int], goal: Tuple[int, int], care_for_agents: bool = True) -> List[Tuple[int, int]]:
        """Find path from start to goal that goes through via. Returns [] if no path exists."""
        gr, gc = goal
        rows, cols = self.grid_size
        adjacent = [
            (gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1)
        ]
        entries = [
            (r, c) for (r, c) in adjacent
            if 0 <= r < rows and 0 <= c < cols
            and self.highways[r, c] == 1
        ]
        best_via = None
        best_path: List[Tuple[int, int]] = []
        for entry_rc in entries:
            p = self.find_agv_path(start, entry_rc, care_for_agents)
            if p and (not best_path or len(p) < len(best_path)):
                best_via = entry_rc
                best_path = p
        via = best_via
        path_to_via = best_path
        if not path_to_via:
            return []
        path_from_via = self.find_agv_path(via, goal, care_for_agents)
        if not path_from_via:
            return []
        return path_to_via + path_from_via
    
    def find_agv_path_to_target_entry(self, start: Tuple[int, int], target_rc: Tuple[int, int], care_for_agents: bool = True) -> List[Tuple[int, int]]:
        """Route an AGV to the nearest highway cell adjacent to target_rc (row, col).

        Used when the target is a non-highway cell (shelf tile 1) so the AGV
        stops next to it and interacts sideways.  start is (row, col).
        Returns [] when no adjacent highway cell is reachable.
        """
        tr, tc = target_rc  # (row, col)
        rows, cols = self.grid_size
        adjacent = [
            (tr - 1, tc), (tr + 1, tc), (tr, tc - 1), (tr, tc + 1)
        ]
        entries = [
            (r, c) for (r, c) in adjacent
            if 0 <= r < rows and 0 <= c < cols
            and self.highways[r, c] == 1
            and (c, r) not in self.goals   # not a goal slot
        ]
        best_path: List[Tuple[int, int]] = []
        for entry_rc in entries:
            p = self.find_agv_path(start, entry_rc, care_for_agents)
            if p and (not best_path or len(p) < len(best_path)):
                best_path = p
        return best_path

    def find_agv_path_to_goal_entry(self, start: Tuple[int, int], goal_xy: Tuple[int, int], care_for_agents: bool = True) -> List[Tuple[int, int]]:
        """Route an AGV to the nearest highway cell adjacent to goal_xy (col, row).

        Used for side drop-off: the AGV stops at a highway cell next to the
        pickerwall slot and deposits the shelf sideways.  start is (row, col).
        Returns [] when no entry cell is reachable.
        """
        entries = self._goal_to_agv_entry.get(goal_xy, [])
        best_path: List[Tuple[int, int]] = []
        for (ex, ey) in entries:          # entries stored as (col, row)
            p = self.find_agv_path(start, (ey, ex), care_for_agents)
            if p and (not best_path or len(p) < len(best_path)):
                best_path = p
        return best_path

    def find_picker_path(self, start: Tuple[int, int], goal: Tuple[int, int], care_for_agents: bool = True) -> List[Tuple[int, int]]:
        """A* path for a picker on picker_highways. Returns [] if no path exists."""
        grid = np.zeros(self.grid_size)
        if care_for_agents:
            grid += self.grid[CollisionLayers.PICKERS]
        # Pickers can only traverse picker_highways cells
        grid += (1 - self.picker_highways)
        grid[goal[0], goal[1]] = 0
        grid[start[0], start[1]] = 0

        grid = [list(map(int, l)) for l in (grid != 0)]
        grid = np.array(grid, dtype=np.float32)
        grid[np.where(grid == 1)] = np.inf
        grid[np.where(grid == 0)] = 1

        astar_path = pyastar2d.astar_path(grid, np.array(start), np.array(goal), allow_diagonal=False)
        if astar_path is not None:
            astar_path = [tuple(x) for x in list(astar_path)]
            astar_path = astar_path[1:]  # skip start position

        if astar_path:
            return [(x, y) for y, x in astar_path]
        else:
            return []

    def find_picker_path_through_adjacent_loc(self, start: Tuple[int, int], goal: Tuple[int, int], care_for_agents: bool = True) -> List[Tuple[int, int]]:
        """Find path from start to goal that goes through via. Returns [] if no path exists."""
        gr, gc = goal
        rows, cols = self.grid_size
        adjacent = [
            (gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1)
        ]
        entries = [
            (r, c) for (r, c) in adjacent
            if 0 <= r < rows and 0 <= c < cols
            and self.picker_highways[r, c] == 1
        ]
        best_via = None
        best_path: List[Tuple[int, int]] = []
        for entry_rc in entries:
            p = self.find_picker_path(start, entry_rc, care_for_agents)
            if p and (not best_path or len(p) < len(best_path)):
                best_via = entry_rc
                best_path = p
        via = best_via
        path_to_via = best_path
        if not path_to_via:
            return []
        path_from_via = self.find_picker_path(via, goal, care_for_agents)
        if not path_from_via:
            return []
        return path_to_via + path_from_via

    def _recalc_grid(self) -> None:
        self.grid.fill(0)

        # Shelves are static fixtures: they always occupy their cells.
        for shelf in self.shelfs:
            self.grid[CollisionLayers.SHELVES, shelf.y, shelf.x] = shelf.id
        for agent in self.agents:
            self.grid[CollisionLayers.AGVS, agent.y, agent.x] = agent.id
            if agent.carrying_bin:
                self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = agent.carrying_bin.id
        for picker in self.pickers:
            self.grid[CollisionLayers.PICKERS, picker.y, picker.x] = picker.id

    def get_carrying_bin_information(self):
        return [agent.carrying_bin != None for agent in self.agents[:self.num_agvs]]

    def get_shelf_request_information(self) -> np.ndarray[int]:
        """Per non-pickerwall slot: 1 if the bin in that slot is currently requested."""
        request_item_map = np.zeros(self.num_non_goal_actions, dtype=np.int32)
        requested_bin_ids = {getattr(item, "id", None) for item in self.request_queue}
        for action_id, (shelf_id, slot_idx) in self.action_id_to_slot.items():
            if action_id < self._storage_action_id_base:
                continue
            shelf = self.shelves_by_id[shelf_id]
            bin_ = shelf.bin_slots[slot_idx]
            if bin_ is not None and bin_.id in requested_bin_ids:
                request_item_map[action_id - self._storage_action_id_base] = 1
        return request_item_map

    def get_empty_shelf_information(self) -> np.ndarray[int]:
        """Per non-pickerwall slot: 1 if a storage slot is empty (bin can be placed there)."""
        empty_item_map = np.zeros(self.num_non_goal_actions, dtype=np.int32)
        for action_id, (shelf_id, slot_idx) in self.action_id_to_slot.items():
            if action_id < self._storage_action_id_base:
                continue
            if action_id >= self._replenishment_action_id_base:
                continue  # replenishment slots handled separately
            shelf = self.shelves_by_id[shelf_id]
            if shelf.bin_slots[slot_idx] is None:
                empty_item_map[action_id - self._storage_action_id_base] = 1
        return empty_item_map

    def get_empty_replenishment_information(self) -> np.ndarray[int]:
        """Per non-pickerwall slot: 1 if a replenishment slot is empty."""
        empty_item_map = np.zeros(self.num_non_goal_actions, dtype=np.int32)
        for action_id, (shelf_id, slot_idx) in self.action_id_to_slot.items():
            if action_id < self._replenishment_action_id_base:
                continue
            shelf = self.shelves_by_id[shelf_id]
            if shelf.bin_slots[slot_idx] is None:
                empty_item_map[action_id - self._storage_action_id_base] = 1
        return empty_item_map

    def get_pickerwall_info(self) -> np.ndarray:
        """Per pickerwall slot: 1 if a bin is currently parked in that slot."""
        occupied = np.zeros(self.num_pickerwall_actions, dtype=np.int32)
        for action_id, (shelf_id, slot_idx) in self.action_id_to_slot.items():
            if action_id >= self._storage_action_id_base:
                break
            shelf = self.shelves_by_id[shelf_id]
            if shelf.bin_slots[slot_idx] is not None:
                occupied[action_id - self._pickerwall_action_id_base] = 1
        return occupied

    def get_pickerwall_displacement_info(self) -> np.ndarray:
        """Per pickerwall slot: 1 if the bin in that slot is fulfilled and eligible for
        AGV displacement back to storage / replenishment."""
        displaceable = np.zeros(self.num_pickerwall_actions, dtype=np.int32)
        for action_id, (shelf_id, slot_idx) in self.action_id_to_slot.items():
            if action_id >= self._storage_action_id_base:
                break
            shelf = self.shelves_by_id[shelf_id]
            bin_ = shelf.bin_slots[slot_idx]
            if bin_ is not None and bin_.fulfilled:
                displaceable[action_id - self._pickerwall_action_id_base] = 1
        return displaceable

    def attribute_macro_actions(self, macro_actions: List[int]) -> int:
        agvs_distance_travelled = 0
        for agent, macro_action in zip(self.agents, macro_actions):
            agent.req_action = Action.NOOP
            if agent.fixing_clash > 0:
                agent.fixing_clash -= 1
            if not agent.busy:
                agent.target = 0
                if macro_action != 0:
                    target_rc = self.action_id_to_coords_map[macro_action]  # (row, col)
                    target_xy = (target_rc[1], target_rc[0])                # (col, row)

                    def _already_at_interaction_position() -> bool:
                        if target_xy in self.goals:
                            # Goal interactions happen from designated adjacent entry cells.
                            return (agent.x, agent.y) in self._goal_to_agv_entry.get(target_xy, [])
                        if not self._is_highway(target_rc[1], target_rc[0]):
                            # Non-highway targets (storage/replenishment slots) are serviced
                            # from an adjacent highway tile.
                            return abs(agent.x - target_xy[0]) + abs(agent.y - target_xy[1]) <= 1
                        # Highway/idle targets are serviced on-cell.
                        return (agent.y, agent.x) == target_rc

                    if target_xy in self.goals:
                        # Pickerwall target (pick-up or drop-off): approach from adjacent
                        # highway cell so the AGV never enters the tile-2 slot.
                        agent.path = self.find_agv_path_to_goal_entry(
                            (agent.y, agent.x), target_xy, care_for_agents=True
                        )
                        if not agent.path:
                            agent.path = self.find_agv_path_to_goal_entry(
                                (agent.y, agent.x), target_xy, care_for_agents=False
                            )
                    elif not self._is_highway(target_rc[1], target_rc[0]):
                        # Non-highway target (shelf tile): approach from an adjacent
                        # highway cell so the AGV never enters the shelf cell directly.
                        agent.path = self.find_agv_path_to_target_entry(
                            (agent.y, agent.x), target_rc, care_for_agents=True
                        )
                        if not agent.path:
                            agent.path = self.find_agv_path_to_target_entry(
                                (agent.y, agent.x), target_rc, care_for_agents=False
                            )
                    else:
                        agent.path = self.find_agv_path((agent.y, agent.x), target_rc, care_for_agents=True)
                        if not agent.path:
                            # Congestion blocked the agent-aware path; fall back to ignoring agents
                            # so the agent can at least start moving. resolve_move_conflict handles
                            # any resulting step-level collisions.
                            agent.path = self.find_agv_path((agent.y, agent.x), target_rc, care_for_agents=False)

                    if agent.path:
                        agent.busy = True
                        agent.target = macro_action
                        agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                        if agent.req_action == Action.FORWARD:
                            # Check if the agent has enough credit to move (e.g. due to configured speed)
                            can_move = self._consume_motion_credit(
                                agent,
                                self._agv_cells_per_step_effective,
                            )
                            if not can_move:
                                agent.req_action = Action.NOOP
                                agent.speed_limited_this_step = True
                        self.stuck_counters[agent.id - 1].reset((agent.x, agent.y))
                    elif macro_action in self.action_id_to_slot and _already_at_interaction_position():
                        # Path can be empty when the AGV is already correctly positioned
                        # next to the target slot. Enter busy mode so the regular busy
                        # branch issues TOGGLE_LOAD.
                        agent.path = []
                        agent.busy = True
                        agent.target = macro_action
                        self.stuck_counters[agent.id - 1].reset((agent.x, agent.y))
            else:
                if agent.path == []:
                    # Before issuing TOGGLE_LOAD, ensure the agent faces the target cell
                    # so the visual representation is correct.
                    target_rc = self.action_id_to_coords_map.get(agent.target)
                    required_dir = None
                    if target_rc is not None:
                        tx, ty = target_rc[1], target_rc[0]   # (col, row)
                        dx, dy = tx - agent.x, ty - agent.y
                        if (abs(dx) + abs(dy)) == 1:
                            dir_map = {
                                (0, -1): Direction.UP,
                                (0, 1):  Direction.DOWN,
                                (-1, 0): Direction.LEFT,
                                (1, 0):  Direction.RIGHT,
                            }
                            required_dir = dir_map.get((dx, dy))
                    if required_dir is not None and agent.dir != required_dir:
                        # Issue a single rotation step so the agent faces the target,
                        # then TOGGLE_LOAD will fire on the next step.
                        turn_order = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
                        diff = (turn_order.index(required_dir) - turn_order.index(agent.dir)) % 4
                        agent.req_action = Action.RIGHT if diff <= 2 else Action.LEFT
                    else:
                        agent.req_action = Action.TOGGLE_LOAD
                else:
                    agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                    if agent.req_action == Action.FORWARD:
                        # Check if the agent has enough credit to move (e.g. due to configured speed)
                        can_move = self._consume_motion_credit(
                            agent,
                            self._agv_cells_per_step_effective,
                        )
                        if can_move:
                            agvs_distance_travelled += 1
                        else:
                            agent.req_action = Action.NOOP
                            agent.speed_limited_this_step = True
                if len(agent.path) == 1 and agent.carrying_bin:
                    slot = self.action_id_to_slot.get(agent.target)
                    if slot is not None:
                        shelf_id, slot_idx = slot
                        shelf = self.shelves_by_id.get(shelf_id)
                        reserver = self._reserved_slots.get((shelf_id, slot_idx))
                        occupied = shelf is not None and shelf.bin_slots[slot_idx] is not None
                        stolen = reserver is not None and reserver != agent.carrying_bin.id
                        if occupied or stolen:
                            shelf_type = shelf.cell_type.value if shelf is not None else "unknown"
                            logger.info(
                                "step=%d agv_id=%d: blocked approaching target action_id=%d "
                                "(shelf_id=%d slot=%d type=%s occupied=%s stolen=%s) "
                                "carrying_bin_id=%d sku=%s depleted=%s",
                                self._cur_steps,
                                agent.id,
                                agent.target,
                                shelf_id,
                                slot_idx,
                                shelf_type,
                                occupied,
                                stolen,
                                agent.carrying_bin.id,
                                agent.carrying_bin.sku,
                                agent.carrying_bin.depleted,
                            )
                            agent.req_action = Action.NOOP
                            agent.busy = False
        return agvs_distance_travelled

    def resolve_move_conflict(self, agent_list):
        commited_agents = set()
        G = nx.DiGraph()
        for agent in agent_list:
            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)
            G.add_edge(start, target)
        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
        for comp in wcomps:
            try:
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:
                    # [A] <-> [B] swap is physically impossible; skip
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = self.grid[CollisionLayers.AGVS, start_node[1], start_node[0]]
                    if agent_id > 0:
                        commited_agents.add(agent_id)
                        continue
            except nx.NetworkXNoCycle:
                longest_path = nx.algorithms.dag_longest_path(comp)
                for x, y in longest_path:
                    agent_id = self.grid[CollisionLayers.AGVS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)
        clashes = 0
        for agent in agent_list:
            for other in agent_list:
                if agent.id != other.id:
                    agent_new_x, agent_new_y = agent.req_location(self.grid_size)
                    other_new_x, other_new_y = other.req_location(self.grid_size)
                    if agent.path and ((agent_new_x, agent_new_y) in [(other.x, other.y), (other_new_x, other_new_y)]):
                        if (agent_new_x, agent_new_y) == (other.x, other.y):
                            agent.req_action = Action.NOOP
                            if (other_new_x, other_new_y) in [(agent.x, agent.y), (agent_new_x, agent_new_y)] and other.req_action not in (Action.LEFT, Action.RIGHT):
                                if other.fixing_clash == 0:
                                    clashes+=1
                                    agent.fixing_clash = _FIXING_CLASH_TIME
                                    new_path = self.find_agv_path((agent.y, agent.x), (agent.path[-1][1] ,agent.path[-1][0]), True)
                                    if len(new_path) == 1:
                                        # Agents that are stuck and are 1 cell away from their target are likely competing with an AGV that has
                                        # the same dilemma, so, re-route agent to the target through an adjacent cell to the target.
                                        new_path = self.find_agv_path_through_adjacent_loc((agent.y, agent.x), (agent.path[-1][1] ,agent.path[-1][0]), True)
                                    if new_path != []:
                                        agent.path = new_path
                                    else:
                                        agent.fixing_clash = 0
                        elif (agent_new_x, agent_new_y) == (other_new_x, other_new_y) and (agent_new_x, agent_new_y) != (agent.x, agent.y):
                            if agent.fixing_clash == 0 and other.fixing_clash == 0:
                                agent.req_action = Action.NOOP
                                agent.fixing_clash = _FIXING_CLASH_TIME

        commited_agents = set([self.agents[id_ - 1] for id_ in commited_agents])
        failed_agents = set(agent_list) - commited_agents
        for agent in failed_agents:
            agent.req_action = Action.NOOP
        return clashes

    def _predict_picker_reserved_positions(self) -> set[Tuple[int, int]]:
        """Cells AGVs should treat as picker-controlled for this step.

        Pickers are modelled as less controllable than AGVs, so AGVs yield to
        both current picker locations and the next cell on each picker path.
        """
        reserved = {(picker.x, picker.y) for picker in self.pickers}
        for picker in self.pickers:
            is_walking = picker.state in (
                PickerState.WALKING_TO_SHELF,
                PickerState.WALKING_TO_PACKAGING,
            )
            if is_walking and picker.path:
                reserved.add(tuple(picker.path[0]))
        return reserved

    def _apply_picker_yield_to_agvs(self) -> int:
        """Stop and replan AGVs whose next move conflicts with picker space."""
        reserved = self._predict_picker_reserved_positions()
        yields = 0
        for agent in self.agents:
            if agent.req_action != Action.FORWARD:
                continue
            next_xy = agent.req_location(self.grid_size)
            if next_xy not in reserved:
                continue

            agent.req_action = Action.NOOP
            yields += 1
            if agent.fixing_clash == 0:
                agent.fixing_clash = _FIXING_CLASH_TIME

            if agent.path:
                new_path = self.find_agv_path(
                    (agent.y, agent.x),
                    (agent.path[-1][1], agent.path[-1][0]),
                    care_for_agents=True,
                )
                if len(new_path) == 1:
                    # Agents that are stuck and are 1 cell away from their target are likely competing with a picker that has
                    # the same dilemma, so, re-route agent to the target through an adjacent cell to the target.
                    new_path = self.find_agv_path_through_adjacent_loc(
                        (agent.y, agent.x),
                        (agent.path[-1][1], agent.path[-1][0]),
                        care_for_agents=True,
                    )
                if new_path:
                    agent.path = new_path

        return yields

    def resolve_picker_conflicts(self) -> int:
        """Graph-based conflict resolution for picker-vs-picker movement.

        Builds a directed graph of (current_pos -> next_pos) for all walking
        pickers, then resolves collisions:

        - **2-cycles (head-on swap)**: physically impossible - reroute one
          picker; if no alternate path exists, force it to wait.
        - **longer cycles**: all pickers in the cycle can advance safely.
        - **non-cycle collisions** (converging on same cell): one picker
          waits; if stuck for several ticks, reroute.
        """
        walking_pickers = [
            p for p in self.pickers
            if p.state in (PickerState.WALKING_TO_SHELF, PickerState.WALKING_TO_PACKAGING)
            and p.path
            and p.fixing_clash == 0
        ]
        if len(walking_pickers) < 2:
            return 0

        # Map each walking picker's current and intended next position
        picker_at: dict[Tuple[int, int], "Picker"] = {}
        picker_next: dict["Picker", Tuple[int, int]] = {}
        for p in walking_pickers:
            pos = (p.x, p.y)
            picker_at[pos] = p
            picker_next[p] = tuple(p.path[0])

        # Build directed graph: current_pos -> next_pos
        G = nx.DiGraph()
        for p in walking_pickers:
            G.add_edge((p.x, p.y), picker_next[p])

        clashes = 0
        rerouted: set = set()

        # Detect cycles
        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
        for comp in wcomps:
            try:
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:
                    # Head-on swap: physically impossible. Reroute one picker.
                    node_a = cycle[0][0]
                    picker_a = picker_at.get(node_a)
                    if picker_a is None or picker_a in rerouted:
                        continue
                    dest_col, dest_row = picker_a.path[-1]
                    new_path = self.find_picker_path(
                        (picker_a.y, picker_a.x), (dest_row, dest_col),
                        care_for_agents=True,
                    )
                    if len(new_path) == 1:
                        new_path = self.find_picker_path_through_adjacent_loc(
                            (picker_a.y, picker_a.x), (dest_row, dest_col),
                            care_for_agents=True,
                        )
                    if new_path:
                        picker_a.path = new_path
                        picker_a.fixing_clash = _FIXING_CLASH_TIME
                    else:
                        picker_a.blocked_ticks += 1
                    rerouted.add(picker_a)
                    clashes += 1
            except nx.NetworkXNoCycle:
                pass

        # Detect convergence: two pickers targeting the same cell
        target_pickers: dict[Tuple[int, int], list] = {}
        for p in walking_pickers:
            if p in rerouted:
                continue
            target_pickers.setdefault(picker_next[p], []).append(p)

        for target, pickers_list in target_pickers.items():
            if len(pickers_list) < 2:
                continue
            # Let the closest picker proceed, reroute the rest
            pickers_list.sort(
                key=lambda p: abs(p.x - target[0]) + abs(p.y - target[1])
            )
            for p in pickers_list[1:]:
                if p in rerouted:
                    continue
                dest_col, dest_row = p.path[-1]
                new_path = self.find_picker_path(
                    (p.y, p.x), (dest_row, dest_col),
                    care_for_agents=True,
                )
                if len(new_path) == 1:
                    new_path = self.find_picker_path_through_adjacent_loc(
                        (p.y, p.x), (dest_row, dest_col),
                        care_for_agents=True,
                    )
                if new_path:
                    p.path = new_path
                    p.fixing_clash = _FIXING_CLASH_TIME
                else:
                    p.blocked_ticks += 1
                rerouted.add(p)
                clashes += 1

        return clashes

    def resolve_stuck_agents(self) -> None:
        overall_stucks = 0
        moving_agents = [
            agent
            for agent in self.agents
            if agent.busy
            and agent.req_action not in (Action.LEFT, Action.RIGHT) # Don't count changing directions
            and (agent.req_action!=Action.TOGGLE_LOAD or (agent.x, agent.y) in self.goals or (agent.x, agent.y) in self._agv_entry_to_goal) # Don't count loading at goal or entry cell
        ]
        for agent in moving_agents:
            agent_stuck_count = self.stuck_counters[agent.id - 1]
            agent_stuck_count.update((agent.x, agent.y))
            if _STUCK_THRESHOLD < agent_stuck_count.count < _STUCK_THRESHOLD + self.column_height + 2:  # Time to get out of aisle
                overall_stucks += 1
                agent_stuck_count.reset((agent.x, agent.y))
                agent.req_action = Action.NOOP
                agent.busy = False
        return overall_stucks

    def _execute_forward(self, agent: Agent) -> None:
        agent.x, agent.y = agent.req_location(self.grid_size)
        agent.path = agent.path[1:]
        if agent.carrying_bin:
            agent.carrying_bin.x, agent.carrying_bin.y = agent.x, agent.y

    def _execute_rotation(self, agent: Agent) -> None:
        agent.dir = agent.req_direction()

    def _execute_load(self, agent: Agent, rewards: np.ndarray[int]) -> np.ndarray[int]:
        """Pick up the bin at agent.target's (shelf_id, slot_idx)."""
        slot = self.action_id_to_slot.get(agent.target)
        if slot is None:
            agent.busy = False
            return rewards
        shelf_id, slot_idx = slot
        shelf = self.shelves_by_id.get(shelf_id)
        if shelf is None:
            agent.busy = False
            return rewards
        bin_ = shelf.bin_slots[slot_idx]
        if bin_ is None:
            agent.busy = False
            return rewards

        shelf.remove_bin(slot_idx)

        agent.carrying_bin = bin_
        bin_.x, bin_.y = agent.x, agent.y
        self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = bin_.id
        self._release_reserved_slot_for_bin(bin_)
        if shelf.cell_type == BinCellType.REPLENISHMENT:
            bin_.from_replenishment = False
        agent.busy = False
        if self.reward_type == RewardType.GLOBAL:
            rewards += 0.5
        elif self.reward_type == RewardType.INDIVIDUAL:
            rewards[agent.id - 1] += 0.1
        return rewards

    def _deliver_to_pickerwall(self, agent: Agent, pickerwall_shelf: "Shelf",
                               slot_idx: int, rewards: np.ndarray[int]) -> np.ndarray[int]:
        """Place the bin carried by *agent* into a pickerwall shelf slot."""
        bin_ = agent.carrying_bin
        if pickerwall_shelf.bin_slots[slot_idx] is not None:
            agent.busy = False
            return rewards
        bin_.fulfilled = False
        pickerwall_shelf.place_bin(bin_, slot_idx)
        self._release_reserved_slot_for_bin(bin_)
        self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = 0
        agent.carrying_bin = None
        agent.busy = False
        agent.has_delivered = False

        if bin_ not in self.request_queue:
            return rewards

        if self.reward_type == RewardType.GLOBAL:
            rewards += 1
        elif self.reward_type == RewardType.INDIVIDUAL:
            rewards[agent.id - 1] += 1

        pending_entries = self._bin_to_order.pop(bin_.id, [])
        for sku_entry, delivered_order in pending_entries:
            self._pickerwall_pending.append((bin_.id, sku_entry, delivered_order))
        if pending_entries:
            logger.info(
                "step=%d delivery: bin_id=%d sku=%s arrived - queued %d pick(s) "
                "for order(s)=%s (pickerwall_pending=%d)",
                self._cur_steps, bin_.id, bin_.sku, len(pending_entries),
                [o.order_number for _, o in pending_entries],
                len(self._pickerwall_pending),
            )

        self.request_queue.remove(bin_)
        self._refill_request_queue()
        self._step_deliveries += 1
        return rewards

    def _storage_bin_candidates(self) -> List["LogicalBin"]:
        """Bins resting in storage shelves that are not already queued or carried."""
        carried_ids = {a.carrying_bin.id for a in self.agents if a.carrying_bin}
        queued_ids = {b.id for b in self.request_queue if b is not None}
        candidates: List[LogicalBin] = []
        for shelf in self.storage_shelves:
            for bin_ in shelf.bin_slots:
                if bin_ is None:
                    continue
                if bin_.id in carried_ids or bin_.id in queued_ids:
                    continue
                candidates.append(bin_)
        candidates.sort(key=lambda b: b.id)
        return candidates

    def _deposit_to_storage(self, agent: Agent, shelf: "Shelf", slot_idx: int,
                            rewards: np.ndarray[int]) -> np.ndarray[int]:
        """Place the bin carried by *agent* into a storage shelf slot."""
        bin_ = agent.carrying_bin
        if shelf.bin_slots[slot_idx] is not None:
            agent.busy = False
            return rewards
        shelf.place_bin(bin_, slot_idx)
        bin_.fulfilled = False
        self._release_reserved_slot_for_bin(bin_)
        self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = 0
        agent.carrying_bin = None
        agent.busy = False
        agent.has_delivered = False
        if self.reward_type == RewardType.GLOBAL:
            rewards += 0.5
        elif self.reward_type == RewardType.INDIVIDUAL:
            rewards[agent.id - 1] += 0.1
        return rewards

    def _return_depleted_to_replenishment(
        self,
        agent: Agent,
        shelf: "Shelf",
        slot_idx: int,
        rewards: np.ndarray[int],
    ) -> np.ndarray[int]:
        """Drop a depleted bin into a replenishment slot; it will be refilled in-place."""
        bin_ = agent.carrying_bin
        if shelf.bin_slots[slot_idx] is not None:
            agent.busy = False
            return rewards
        self._remove_bin_from_sku_lookup(bin_)
        original_sku = bin_.sku
        if original_sku is not None:
            self._assign_sku_to_bin(bin_, original_sku)
        shelf.place_bin(bin_, slot_idx)
        self._release_reserved_slot_for_bin(bin_)
        bin_.from_replenishment = True
        self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = 0
        agent.carrying_bin = None
        agent.busy = False
        agent.has_delivered = False
        self._log_replenishment_filled(bin_, shelf, slot_idx)
        return rewards

    def _execute_unload(self, agent: Agent, rewards: np.ndarray[int]) -> np.ndarray[int]:
        """Drop the carried bin into the slot given by agent.target."""
        slot = self.action_id_to_slot.get(agent.target)
        if slot is None:
            agent.busy = False
            return rewards
        shelf_id, slot_idx = slot
        shelf = self.shelves_by_id.get(shelf_id)
        if shelf is None:
            agent.busy = False
            return rewards

        if shelf.cell_type == BinCellType.PICKERWALL:
            return self._deliver_to_pickerwall(agent, shelf, slot_idx, rewards)
        if shelf.cell_type == BinCellType.REPLENISHMENT:
            return self._return_depleted_to_replenishment(agent, shelf, slot_idx, rewards)
        return self._deposit_to_storage(agent, shelf, slot_idx, rewards)

    def execute_micro_actions(self, rewards: np.ndarray[int]) -> np.ndarray[int]:
        for agent in self.agents:
            if agent.req_action == Action.FORWARD:
                self._execute_forward(agent)
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                self._execute_rotation(agent)
            elif agent.req_action == Action.TOGGLE_LOAD:
                if not agent.carrying_bin:
                    rewards = self._execute_load(agent, rewards)
                else:
                    rewards = self._execute_unload(agent, rewards)
        return rewards

    def process_shelf_deliveries(self, rewards: np.ndarray[int]) -> np.ndarray[int]:
        shelf_deliveries = self._step_deliveries
        if shelf_deliveries:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1
        return rewards, shelf_deliveries

    def _select_evenly_spaced_spawn_cells(
        self,
        candidates: List[Tuple[int, int]],
        count: int,
    ) -> List[Tuple[int, int]]:
        """Choose deterministic spawn cells spread across the candidate list."""
        if count <= 0 or not candidates:
            return []
        candidates = sorted(candidates)
        if count >= len(candidates):
            return candidates[:count]

        chosen: List[Tuple[int, int]] = []
        used_indices = set()
        positions = np.linspace(0, len(candidates) - 1, count + 2)[1:-1]
        for position in positions:
            idx = int(round(position))
            while idx in used_indices and idx + 1 < len(candidates):
                idx += 1
            while idx in used_indices and idx - 1 >= 0:
                idx -= 1
            if idx not in used_indices:
                used_indices.add(idx)
                chosen.append(candidates[idx])
        return chosen

    def _balanced_picker_spawn_specs(self) -> List[Tuple[int, int, Optional[int]]]:
        """Return deterministic picker spawn specs as (row, col, home_zone)."""
        all_candidates = [
            (int(row), int(col))
            for row, col in self.picker_spawn_locs
        ]
        if not all_candidates:
            return []

        if not self.pickerwall_zones:
            cells = self._select_evenly_spaced_spawn_cells(
                all_candidates,
                min(self.num_pickers, len(all_candidates)),
            )
            return [(row, col, None) for row, col in cells]

        zone_counts = [0 for _ in self.pickerwall_zones]
        for picker_idx in range(self.num_pickers):
            zone_counts[picker_idx % len(zone_counts)] += 1

        used_cells = set()
        specs: List[Tuple[int, int, Optional[int]]] = []
        for zone_id, count in enumerate(zone_counts):
            if count <= 0:
                continue
            zmin, zmax = self.pickerwall_zones[zone_id]
            local_candidates = [
                cell
                for cell in all_candidates
                if cell not in used_cells and zmin <= cell[0] <= zmax
            ]
            selected = self._select_evenly_spaced_spawn_cells(local_candidates, count)

            if len(selected) < count:
                fallback_candidates = [
                    cell
                    for cell in all_candidates
                    if cell not in used_cells and cell not in selected
                ]
                selected.extend(
                    self._select_evenly_spaced_spawn_cells(
                        fallback_candidates,
                        count - len(selected),
                    )
                )

            for row, col in selected:
                used_cells.add((row, col))
                specs.append((row, col, zone_id))

        return specs

    def _bin_quantity_for_sku(self, sku: int) -> int:
        """Return how many units of a SKU fit in one usable bin."""
        if self.order_sequencer is None:
            return 0
        unit_cube = self.order_sequencer.get_sku_unit_cube(sku)
        if unit_cube <= 0:
            return 0
        usable_volume = self.bin_volume_ft3 * self.bin_usable_fraction
        return max(0, int(math.floor(usable_volume / unit_cube)))

    def _assign_sku_to_bin(self, bin_: LogicalBin, sku: int) -> int:
        quantity = self._bin_quantity_for_sku(sku)
        unit_cube = self.order_sequencer.get_sku_unit_cube(sku) if self.order_sequencer else 0.0
        bin_.sku = sku
        bin_.quantity = quantity
        bin_.used_volume_ft3 = quantity * unit_cube
        bin_.depleted = False
        bin_.fulfilled = False
        if self.order_sequencer is not None and quantity > 0:
            bins_for_sku = self.order_sequencer._sku_to_bins.setdefault(sku, [])
            if bin_ not in bins_for_sku:
                bins_for_sku.append(bin_)
        return quantity

    def _remove_bin_from_sku_lookup(self, bin_: LogicalBin) -> None:
        if self.order_sequencer is None or bin_.sku is None:
            return
        bins_for_sku = self.order_sequencer._sku_to_bins.get(bin_.sku)
        if not bins_for_sku:
            return
        remaining = [
            candidate
            for candidate in bins_for_sku
            if candidate.id != bin_.id
        ]
        # This is left in to determine if replenishment is working properly
        self.order_sequencer._sku_to_bins[bin_.sku] = remaining
        if not remaining:
            logger.warning(
                "step=%d: SKU %s has NO remaining bins after removing "
                "bin_id=%d (depleted=%s)",
                self._cur_steps, bin_.sku, bin_.id, bin_.depleted,
            )

    def _make_new_bin(self, cell_type: BinCellType, x: int, y: int) -> LogicalBin:
        """Build a fresh LogicalBin with a unique id."""
        self._next_bin_id += 1
        return LogicalBin(
            id=self._next_bin_id,
            x=x,
            y=y,
            cell_type=cell_type,
            volume_ft3=self.bin_volume_ft3,
            usable_fraction=self.bin_usable_fraction,
        )

    def _initialize_bin_inventory(self) -> None:
        """Populate storage shelves with one bin per unique SKU.

        Bins are spread randomly across storage shelves and their empty slots.
        Any leftover storage capacity is left empty (not all slots are filled).
        """
        if self.order_sequencer is None:
            return

        unique_skus = self.order_sequencer.get_unique_skus()
        self.order_sequencer._sku_to_bins = {}
        self._bins_by_id: Dict[int, LogicalBin] = {}

        storage_slots: List[Tuple[Shelf, int]] = [
            (shelf, slot_idx)
            for shelf in self.storage_shelves
            for slot_idx in range(shelf.num_slots)
        ]
        if not storage_slots:
            return

        n_bins = min(len(unique_skus), len(storage_slots))
        slot_indices = np.random.choice(len(storage_slots), size=n_bins, replace=False)

        for sku, idx in zip(unique_skus[:n_bins], slot_indices):
            shelf, slot_idx = storage_slots[int(idx)]
            bin_ = self._make_new_bin(BinCellType.STORAGE, shelf.x, shelf.y)
            self._assign_sku_to_bin(bin_, sku)
            shelf.place_bin(bin_, slot_idx)
            self._bins_by_id[bin_.id] = bin_

        logger.info(
            "Bin inventory initialised: bins=%d unique_skus=%d storage_slots=%d slots_used=%d",
            len(self._bins_by_id), len(unique_skus),
            len(storage_slots), n_bins,
        )

    def _reserve_slot_for_bin(
        self,
        bin_: LogicalBin,
        shelf: Shelf,
    ) -> Optional[int]:
        """Reserve the first empty slot on ``shelf`` for ``bin_``.

        Returns the slot_index on success, None if the shelf is full.
        Reservations prevent two AGVs from targeting the same destination slot
        before either one actually arrives and deposits.
        """
        for slot_idx in range(shelf.num_slots):
            key = (shelf.id, slot_idx)
            if shelf.bin_slots[slot_idx] is None and key not in self._reserved_slots:
                self._reserved_slots[key] = bin_.id
                return slot_idx
        return None

    def _release_reserved_slot_for_bin(self, bin_: LogicalBin) -> None:
        keys = [k for k, v in self._reserved_slots.items() if v == bin_.id]
        for k in keys:
            del self._reserved_slots[k]

    def _requeue_sku_request(self, sku_entry: SKUEntry, order: Order) -> None:
        if self.order_sequencer is None:
            return
        self.order_sequencer._pending_sku_requests.appendleft(
            (
                SKUEntry(
                    sku=sku_entry.sku,
                    quantity=sku_entry.quantity,
                    unit_cube=sku_entry.unit_cube,
                ),
                order,
            )
        )

    def _drain_stale_pickerwall_work_for_bin(self, bin_: LogicalBin) -> None:
        """Requeue all unpicked work still tied to a depleted pickerwall bin."""
        removed_pending_entries = 0
        removed_task_claims = 0
        requeued_units = 0

        kept_pending = deque()
        while self._pickerwall_pending:
            entry_bin_id, sku_entry, order = self._pickerwall_pending.popleft()
            if entry_bin_id != bin_.id:
                kept_pending.append((entry_bin_id, sku_entry, order))
                continue
            removed_pending_entries += 1
            if sku_entry.quantity > 0:
                self._requeue_sku_request(sku_entry, order)
                requeued_units += int(sku_entry.quantity)
        self._pickerwall_pending = kept_pending

        for picker in self.pickers:
            task = picker.task
            if task is None:
                continue
            for claim in task.claims:
                if claim.picked or claim.bin_id != bin_.id:
                    continue
                removed_task_claims += 1
                if claim.sku_entry.quantity > 0:
                    self._requeue_sku_request(claim.sku_entry, claim.order)
                    requeued_units += int(claim.sku_entry.quantity)
                claim.sku_entry = SKUEntry(
                    sku=claim.sku_entry.sku,
                    quantity=0,
                    unit_cube=claim.sku_entry.unit_cube,
                )
                claim.picked = True

        if removed_pending_entries or removed_task_claims:
            logger.info(
                "step=%d: bin_id=%d sku=%s drained stale pickerwall work "
                "(pending_entries=%d task_claims=%d requeued_units=%d)",
                self._cur_steps,
                bin_.id,
                bin_.sku,
                removed_pending_entries,
                removed_task_claims,
                requeued_units,
            )

    def _pending_units_for_sku(self, sku: Optional[int]) -> int:
        if self.order_sequencer is None or sku is None:
            return 0
        return sum(
            sku_entry.quantity
            for sku_entry, _ in self.order_sequencer._pending_sku_requests
            if sku_entry.sku == sku
        )

    def _log_replenishment_requested(self, bin_: LogicalBin, trigger: str) -> None:
        """Log one replenishment-request event per depleted-bin cycle."""
        if bin_.id in self._replenishment_request_step_by_bin:
            return
        self._replenishment_request_step_by_bin[bin_.id] = self._cur_steps
        logger.info(
            "step=%d replenishment_requested: sku=%s bin_id=%d trigger=%s qty=%d pending_units_for_sku=%d",
            self._cur_steps,
            bin_.sku,
            bin_.id,
            trigger,
            bin_.quantity,
            self._pending_units_for_sku(bin_.sku),
        )

    def _log_replenishment_filled(
        self,
        bin_: LogicalBin,
        shelf: "Shelf",
        slot_idx: int,
    ) -> None:
        """Log replenishment fill completion and clear any outstanding request marker."""
        request_step = self._replenishment_request_step_by_bin.pop(bin_.id, None)
        if request_step is None:
            logger.info(
                "step=%d replenishment_filled: sku=%s bin_id=%d shelf_id=%d slot=%d qty=%d request_age_steps=unknown",
                self._cur_steps,
                bin_.sku,
                bin_.id,
                shelf.id,
                slot_idx,
                bin_.quantity,
            )
            return

        age_steps = max(0, self._cur_steps - request_step)
        logger.info(
            "step=%d replenishment_filled: sku=%s bin_id=%d shelf_id=%d slot=%d qty=%d request_age_steps=%d request_age_sim_seconds=%.2f",
            self._cur_steps,
            bin_.sku,
            bin_.id,
            shelf.id,
            slot_idx,
            bin_.quantity,
            age_steps,
            self.steps_to_simulated_seconds(age_steps),
        )

    def _decrement_bin_inventory(self, bin_: LogicalBin, quantity: int) -> int:
        """Pick ``quantity`` units from ``bin_``; return remaining stock."""
        bin_.quantity = max(0, bin_.quantity - quantity)
        unit_cube = self.order_sequencer.get_sku_unit_cube(bin_.sku) if (
            self.order_sequencer is not None and bin_.sku is not None
        ) else 0.0
        bin_.used_volume_ft3 = bin_.quantity * unit_cube
        if bin_.quantity <= 0:
            bin_.depleted = True
            self._remove_bin_from_sku_lookup(bin_)
            self._drain_stale_pickerwall_work_for_bin(bin_)
            self._maybe_mark_bin_fulfilled(bin_)
            self._log_replenishment_requested(bin_, trigger="inventory_depleted")
        return bin_.quantity

    def reset(self, seed=None, options=None)-> Tuple:
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0
        self._step_deliveries = 0
        self.seed(seed)
        self._bin_to_order = {}
        self._packaging_slots = {}
        self._pickerwall_pending = deque()
        self._reserved_slots = {}
        self._next_bin_id = 0
        self._bins_by_id = {}
        self._picker_hf_profile_by_id = {}
        self._picker_hf_state_by_id = {}
        self._picker_hf_episode_delay_steps = 0
        self._picker_hf_episode_failed_pick_delays = 0
        self._latest_picker_diagnostics = []
        self._replenishment_request_step_by_bin = {}

        # Clear all static shelves' slots so re-runs start empty.
        for shelf in self.shelfs:
            for slot_idx in range(shelf.num_slots):
                shelf.bin_slots[slot_idx] = None

        # Spawn AGVs preferentially in idle zones (tile 7). If there aren't
        # enough idle cells for every AGV, fill the rest from the regular
        # highway, excluding cells already chosen as idle spawns.
        idle_spawn_locs = np.argwhere(self.agv_idle_zones == 1)  # (row, col)
        n_idle = min(self.num_agents, len(idle_spawn_locs))
        n_highway = self.num_agents - n_idle

        if n_idle > 0:
            idle_pick_ids = np.random.choice(len(idle_spawn_locs), size=n_idle, replace=False)
            idle_pick = idle_spawn_locs[idle_pick_ids]
        else:
            idle_pick = np.empty((0, 2), dtype=int)

        if n_highway > 0:
            idle_set = {tuple(rc) for rc in idle_pick}
            highway_candidates = np.array(
                [rc for rc in self.agv_spawn_locs if tuple(rc) not in idle_set],
                dtype=int,
            )
            hw_pick_ids = np.random.choice(len(highway_candidates), size=n_highway, replace=False)
            highway_pick = highway_candidates[hw_pick_ids]
        else:
            highway_pick = np.empty((0, 2), dtype=int)

        chosen_rc = np.concatenate([idle_pick, highway_pick], axis=0) if (n_idle + n_highway) else np.empty((0, 2), dtype=int)
        agent_locs = [chosen_rc[:, 0], chosen_rc[:, 1]]
        agent_dirs = np.random.choice([d for d in Direction], size=self.num_agents)
        self.agents = [
            Agent(x, y, dir_, agent_type=agent_type)
            for y, x, dir_, agent_type in zip(*agent_locs, agent_dirs, self._agent_types)
        ]

        self.stuck_counters = [StuckCounter((agent.x, agent.y)) for agent in self.agents]

        Picker.counter = 0
        self.pickers = []
        if len(self.picker_spawn_locs) > 0:
            picker_spawn_specs = self._balanced_picker_spawn_specs()
            for picker_index, (py, px, home_zone) in enumerate(picker_spawn_specs):
                picker = Picker(int(px), int(py), Direction.UP)
                picker.state = PickerState.IDLE
                picker.task = None
                picker.path = []
                picker.pick_ticks_remaining = 0
                picker.home_zone = home_zone
                picker.stalled = False
                picker.packaging_location = None
                self.pickers.append(picker)
                profile = self.human_factors_config.profile_for_picker_index(picker_index)
                self._picker_hf_profile_by_id[picker.id] = profile
                self._picker_hf_state_by_id[picker.id] = PickerHumanFactorsState(
                    profile_name=profile.name,
                    fatigue=self.human_factors_config.fatigue_min,
                )
            logger.info(
                "reset: spawned %d picker(s) evenly across %d zone(s) and %d picker-highway cells",
                len(self.pickers), len(self.pickerwall_zones), len(self.picker_spawn_locs),
            )
            if self.human_factors_config.enabled:
                logger.info(
                    "reset: human factors enabled default_profile=%s picker_overrides=%s",
                    self.human_factors_config.default_profile,
                    self.human_factors_config.picker_profile_overrides,
                )

        self._recalc_grid()

        if self.order_sequencer is not None:
            logger.info(
                "reset: order_sequencer present - using time-gated SKU-based request queue"
            )
            self.order_sequencer.reset()
            self._initialize_bin_inventory()
            released = self.order_sequencer.release_pending_orders(0)
            logger.info("reset: released %d orders at t=0 (queue capacity=%d)", len(released), self.request_queue_size)
            self.request_queue = []
            self._refill_request_queue()
            logger.info(
                "reset: initial request_queue filled=%d/%d skus=%s",
                len(self.request_queue), self.request_queue_size,
                [b.sku for b in self.request_queue],
            )
        else:
            logger.info(
                "reset: no order_sequencer - filling request queue with %d random bins",
                self.request_queue_size,
            )
            storage_bins = [b for b in self._bins_by_id.values() if b.on_shelf]
            size = min(self.request_queue_size, len(storage_bins))
            self.request_queue = list(
                np.random.choice(storage_bins, size=size, replace=False)
            ) if size > 0 else []

        self.observation_space_mapper.extract_environment_info(self)
        return tuple([self.observation_space_mapper.observation(agent) for agent in self.agents])

    def _next_order_assignment(
        self,
    ) -> Optional[Tuple["LogicalBin", "SKUEntry", "Order", str]]:
        """Find the first currently-fulfillable SKU request and remove it.

        Scans ``_pending_sku_requests`` and looks up the (typically single)
        non-depleted bin for each SKU via ``_sku_to_bins``.  The bin's
        current location determines the match type:

        - **in_transit** - being carried to the pickerwall (carried + in
          ``request_queue``).  Picker can be queued ahead of arrival.
        - **pickerwall** - already resting on the pickerwall.
        - **storage** - on a storage shelf, available for AGV fetch.
        - **returning** - being carried *away* from the pickerwall (e.g.
          displacement).  The bin is un-fulfilled so the heuristic cancels
          the displacement mission; the request is left in the deque for
          retry once the bin lands back on a shelf.

        Returns ``(bin, sku_entry, order, match_type)`` or None.
        """
        carried_ids = {a.carrying_bin.id for a in self.agents if a.carrying_bin}
        queued_ids = {b.id for b in self.request_queue if b is not None}

        for i in range(len(self.order_sequencer._pending_sku_requests)):
            sku_entry, order = self.order_sequencer._pending_sku_requests[i]
            bins_for_sku = self.order_sequencer._sku_to_bins.get(sku_entry.sku, [])

            for bin_ in bins_for_sku:
                if bin_.depleted:
                    continue

                # Being carried to the pickerwall for another order
                if bin_.id in carried_ids and bin_.id in queued_ids:
                    del self.order_sequencer._pending_sku_requests[i]
                    return bin_, sku_entry, order, "in_transit"

                # Being carried away (displacement) - un-fulfill so the
                # heuristic cancels the mission; leave request for retry.
                if bin_.id in carried_ids:
                    if bin_.fulfilled:
                        bin_.fulfilled = False
                        logger.info(
                            "step=%d: bin_id=%d sku=%s un-fulfilled - "
                            "being displaced but needed by order=%s",
                            self._cur_steps, bin_.id, bin_.sku,
                            order.order_number,
                        )
                    break  # can't use this bin yet; skip to next request

                if not bin_.on_shelf:
                    continue
                shelf = self.shelves_by_id.get(bin_.shelf_id)
                if shelf is None:
                    continue

                if shelf.cell_type == BinCellType.PICKERWALL:
                    del self.order_sequencer._pending_sku_requests[i]
                    return bin_, sku_entry, order, "pickerwall"

                if (shelf.cell_type in (BinCellType.STORAGE, BinCellType.REPLENISHMENT)
                        and bin_.id not in queued_ids):
                    del self.order_sequencer._pending_sku_requests[i]
                    return bin_, sku_entry, order, "storage"

        return None

    def _refill_request_queue(self) -> None:
        """Fill the request queue up to request_queue_size using _next_order_assignment.

        Match types from ``_next_order_assignment``:

        - **in_transit**: bin is being carried to pickerwall for another order.
          Add pick to ``_pickerwall_pending`` so the picker is ready when it
          arrives.  No queue slot consumed.
        - **pickerwall**: bin already on pickerwall.  Un-fulfill it and add
          pick to ``_pickerwall_pending``.  No queue slot consumed.
        - **storage**: bin in storage.  Add to ``request_queue`` for AGV fetch.
        """
        if self.order_sequencer is None:
            return

        # Sort so requests whose SKU is already on the pickerwall come first -
        # they don't consume queue slots, leaving more room for storage fetches.
        pw_skus: set = set()
        for shelf in self.pickerwall_shelves:
            for bin_ in shelf.bin_slots:
                if bin_ is not None and bin_.sku is not None and not bin_.depleted:
                    pw_skus.add(bin_.sku)
        self.order_sequencer.sort_pending_sku_requests(pw_skus)

        resolved_without_slot = 0
        while True:
            result = self._next_order_assignment()
            if result is None:
                break
            bin_, sku_entry, order, match_type = result

            if match_type == "in_transit":
                # Bin already being carried to pickerwall - stash alongside
                # the original storage entry so _deliver_to_pickerwall adds
                # ALL pending picks to _pickerwall_pending in one shot.
                self._bin_to_order.setdefault(bin_.id, []).append((sku_entry, order))
                resolved_without_slot += 1
                logger.info(
                    "step=%d _refill: bin_id=%d sku=%s in transit to pickerwall - "
                    "stashed pick for order=%s (pending_for_bin=%d)",
                    self._cur_steps, bin_.id, bin_.sku, order.order_number,
                    len(self._bin_to_order.get(bin_.id, [])),
                )
                continue

            if match_type == "pickerwall":
                # Already on the pickerwall - resolve directly
                if bin_.fulfilled:
                    bin_.fulfilled = False
                    logger.info(
                        "step=%d: bin_id=%d sku=%s un-fulfilled - "
                        "SKU needed by order=%s (already on pickerwall)",
                        self._cur_steps, bin_.id, bin_.sku, order.order_number,
                    )
                if self.num_pickers > 0:
                    self._pickerwall_pending.append((bin_.id, sku_entry, order))
                resolved_without_slot += 1
                continue

            # Storage - queue for AGV fetch (only if there's room)
            if len(self.request_queue) >= self.request_queue_size:
                # Put the request back - we can't queue it yet
                self.order_sequencer._pending_sku_requests.appendleft(
                    (sku_entry, order)
                )
                break
            self._bin_to_order.setdefault(bin_.id, []).append((sku_entry, order))
            self.request_queue.append(bin_)
            logger.info(
                "step=%d _refill: added bin_id=%d sku=%d to request_queue "
                "(queue=%d/%d pending=%d active=%d)",
                self._cur_steps, bin_.id, bin_.sku,
                len(self.request_queue), self.request_queue_size,
                self.order_sequencer.pending_count, self.order_sequencer.active_count,
            )
        if resolved_without_slot:
            logger.info(
                "step=%d _refill: resolved %d SKU requests without queue slot "
                "(in_transit + pickerwall) remaining_sku_requests=%d queue=%d/%d",
                self._cur_steps, resolved_without_slot,
                len(self.order_sequencer._pending_sku_requests),
                len(self.request_queue), self.request_queue_size,
            )

    def _resolve_pickerwall_bin_for_sku(self, sku: int) -> int:
        """Return the bin_id of a bin resting in a pickerwall shelf with the given SKU.

        Returns -1 if no matching bin is currently on pickerwall.
        """
        for shelf in self.pickerwall_shelves:
            for bin_ in shelf.bin_slots:
                if bin_ is not None and bin_.sku == sku:
                    return bin_.id
        return -1

    def _nearest_pickerwall_zone(self, row: int) -> Optional[int]:
        if not self.pickerwall_zones:
            return None
        return min(
            range(len(self.pickerwall_zones)),
            key=lambda zone_id: abs(
                row - (self.pickerwall_zones[zone_id][0] + self.pickerwall_zones[zone_id][1]) / 2
            ),
        )

    def _bin_pickerwall_zone(self, bin_id: int) -> Optional[int]:
        bin_ = self._bins_by_id.get(bin_id)
        if bin_ is None or bin_.shelf_id is None:
            return None
        shelf = self.shelves_by_id.get(bin_.shelf_id)
        if shelf is None or shelf.cell_type != BinCellType.PICKERWALL:
            return None
        return self._goal_to_zone.get((shelf.x, shelf.y))

    def _picker_allowed_zones(self, picker: "Picker") -> List[Optional[int]]:
        if picker.home_zone is None or self.picker_policy != "zone":
            return [None]

        allowed = [picker.home_zone]
        if self.picker_zone_overflow in ("adjacent", "global"):
            if picker.home_zone - 1 >= 0:
                allowed.append(picker.home_zone - 1)
            if picker.home_zone + 1 < len(self.pickerwall_zones):
                allowed.append(picker.home_zone + 1)
        if self.picker_zone_overflow == "global":
            allowed.extend(
                zone_id
                for zone_id in range(len(self.pickerwall_zones))
                if zone_id not in allowed
            )
        return allowed

    def _pop_pickerwall_entry_for_zone(
        self,
        zone_id: Optional[int],
    ) -> Optional[Tuple[int, SKUEntry, Order]]:
        skipped = deque()
        selected = None

        while self._pickerwall_pending:
            bin_id, sku_entry, order = self._pickerwall_pending.popleft()
            bin_ = self._bins_by_id.get(bin_id)

            if bin_ is None or bin_.shelf_id is None:
                logger.info(
                    "step=%d _claim: bin_id=%d no longer on shelf - "
                    "returning sku=%d order=%s to pending_sku_requests",
                    self._cur_steps, bin_id, sku_entry.sku, order.order_number,
                )
                if self.order_sequencer is not None:
                    self.order_sequencer._pending_sku_requests.appendleft(
                        (sku_entry, order)
                    )
                continue
            shelf = self.shelves_by_id.get(bin_.shelf_id)
            if shelf is None or shelf.cell_type != BinCellType.PICKERWALL:
                logger.info(
                    "step=%d _claim: bin_id=%d no longer on pickerwall - "
                    "returning sku=%d order=%s to pending_sku_requests",
                    self._cur_steps, bin_id, sku_entry.sku, order.order_number,
                )
                if self.order_sequencer is not None:
                    self.order_sequencer._pending_sku_requests.appendleft(
                        (sku_entry, order)
                    )
                continue

            if bin_.depleted or bin_.quantity <= 0:
                if bin_.quantity <= 0 and not bin_.depleted:
                    # Canonicalize zero-stock bins so future assignment skips this bin
                    # until replenishment returns it to service.
                    bin_.depleted = True
                    self._remove_bin_from_sku_lookup(bin_)
                self._log_replenishment_requested(bin_, trigger="pickerwall_claim_no_stock")
                logger.info(
                    "step=%d _claim: bin_id=%d sku=%d has no stock - "
                    "returning order=%s qty=%d to pending_sku_requests",
                    self._cur_steps, bin_id, sku_entry.sku,
                    order.order_number, sku_entry.quantity,
                )
                if self.order_sequencer is not None:
                    self.order_sequencer._pending_sku_requests.appendleft(
                        (
                            SKUEntry(
                                sku=sku_entry.sku,
                                quantity=sku_entry.quantity,
                                unit_cube=sku_entry.unit_cube,
                            ),
                            order,
                        )
                    )
                # We just removed one stale pickerwall claim for this bin; re-check
                # displacement eligibility so depleted bins can be evacuated.
                self._maybe_mark_bin_fulfilled(bin_)
                continue

            if zone_id is None or self._bin_pickerwall_zone(bin_id) == zone_id:
                selected = (bin_id, sku_entry, order)
                break
            skipped.append((bin_id, sku_entry, order))

        while skipped:
            self._pickerwall_pending.appendleft(skipped.pop())
        return selected

    def _append_claim_for_entry(
        self,
        claims: List["PickerClaim"],
        entry: Tuple[int, SKUEntry, Order],
        remaining_cap: int,
    ) -> int:
        bin_id, sku_entry, order = entry
        if sku_entry.quantity <= remaining_cap:
            claims.append(PickerClaim(
                bin_id=bin_id,
                sku_entry=sku_entry,
                order_number=order.order_number,
                order=order,
            ))
            return remaining_cap - sku_entry.quantity

        claim_entry = SKUEntry(
            sku=sku_entry.sku,
            quantity=remaining_cap,
            unit_cube=sku_entry.unit_cube,
        )
        remaining_entry = SKUEntry(
            sku=sku_entry.sku,
            quantity=sku_entry.quantity - remaining_cap,
            unit_cube=sku_entry.unit_cube,
        )
        claims.append(PickerClaim(
            bin_id=bin_id,
            sku_entry=claim_entry,
            order_number=order.order_number,
            order=order,
        ))
        self._pickerwall_pending.appendleft((bin_id, remaining_entry, order))
        return 0

    def _claim_items_for_picker(self, picker: "Picker") -> List["PickerClaim"]:
        """Claim up to picker.capacity item-units from pickerwall work.

        FIFO mode takes the global queue in arrival order. Zone mode fills from
        the picker's home zone first, then optional overflow zones.
        """
        if not self._pickerwall_pending:
            return []

        claims: List[PickerClaim] = []
        remaining_cap = picker.capacity
        allowed_zones = self._picker_allowed_zones(picker)

        for zone_id in allowed_zones:
            while self._pickerwall_pending and remaining_cap > 0:
                entry = self._pop_pickerwall_entry_for_zone(zone_id)
                if entry is None:
                    break
                remaining_cap = self._append_claim_for_entry(
                    claims,
                    entry,
                    remaining_cap,
                )

        for claim in claims:
            if claim.order_number not in self._packaging_slots:
                total_qty = sum(se.quantity for se in claim.order.skus)
                required_per_sku: Dict[int, int] = {}
                for se in claim.order.skus:
                    required_per_sku[se.sku] = required_per_sku.get(se.sku, 0) + se.quantity
                self._packaging_slots[claim.order_number] = {
                    "required": total_qty,
                    "delivered": 0,
                    "station": None,
                    "required_per_sku": required_per_sku,
                    "delivered_per_sku": {},
                }
        return claims

    def _build_picker_task(self, picker: "Picker", claims: List["PickerClaim"]) -> "PickerTask":
        """Preserve policy claim order while grouping same-bin work."""
        bin_groups: Dict[int, List[PickerClaim]] = {}
        bin_order: List[int] = []
        unresolved: List[PickerClaim] = []
        for claim in claims:
            if claim.bin_id == -1:
                unresolved.append(claim)
                continue
            if claim.bin_id not in bin_groups:
                bin_groups[claim.bin_id] = []
                bin_order.append(claim.bin_id)
            bin_groups[claim.bin_id].append(claim)

        ordered_claims: List[PickerClaim] = []
        for bin_id in bin_order:
            ordered_claims.extend(bin_groups[bin_id])
        ordered_claims.extend(unresolved)
        return PickerTask(claims=ordered_claims)

    def _maybe_mark_bin_fulfilled(self, bin_: "LogicalBin") -> None:
        """Mark bin.fulfilled = True if no pending or in-flight picks remain for it."""
        bid = bin_.id
        for entry_bin_id, _sku_entry, _order in self._pickerwall_pending:
            if entry_bin_id == bid:
                bin_.fulfilled = False
                return
        for picker in self.pickers:
            if picker.task is None:
                continue
            for claim in picker.task.claims:
                if not claim.picked and claim.bin_id == bid:
                    bin_.fulfilled = False
                    return
        bin_.fulfilled = True
        logger.info(
            "step=%d: bin_id=%d sku=%s marked fulfilled - eligible for displacement",
            self._cur_steps, bin_.id, bin_.sku,
        )

    def _packaging_locations_for_picker(self, picker: "Picker") -> List[Tuple[int, int]]:
        """Return packaging stations local to the picker's home zone when possible."""
        if self.picker_policy != "zone" or picker.home_zone is None:
            return self.packaging_locations

        zmin, zmax = self.pickerwall_zones[picker.home_zone]
        local_packaging = [
            loc
            for loc in self.packaging_locations
            if zmin <= loc[1] <= zmax
        ]
        return local_packaging or self.packaging_locations

    def _choose_packaging_station_for_task(
        self,
        picker: "Picker",
        packaging_candidates: List[Tuple[int, int]],
    ) -> Tuple[int, int]:
        """Choose a stable packaging station for a picker task.

        Once an order has a packaging station, later picker trips for that
        order must return to the same station so progress does not appear to
        jump between cells in the renderer.
        """
        assigned_stations = []
        for claim in picker.task.claims:
            slot = self._packaging_slots.get(claim.order_number)
            if slot is not None and slot.get("station") is not None:
                assigned_stations.append(tuple(slot["station"]))

        if assigned_stations:
            return min(
                sorted(set(assigned_stations)),
                key=lambda loc: abs(loc[0] - picker.x) + abs(loc[1] - picker.y),
            )

        return min(
            packaging_candidates,
            key=lambda loc: abs(loc[0] - picker.x) + abs(loc[1] - picker.y),
        )

    def _start_walk_to_packaging(self, picker: "Picker") -> None:
        """Route the picker to the closest packaging station in its local zone."""
        packaging_candidates = self._packaging_locations_for_picker(picker)
        if not packaging_candidates:
            picker.path = []
            picker.state = PickerState.AT_PACKAGING
            logger.info(
                "step=%d picker_id=%d: all claims picked -> AT_PACKAGING "
                "without packaging station",
                self._cur_steps, picker.id,
            )
            return
        closest_pkg = self._choose_packaging_station_for_task(picker, packaging_candidates)
        picker.packaging_location = closest_pkg
        for claim in picker.task.claims:
            slot = self._packaging_slots.get(claim.order_number)
            if slot is not None and slot.get("station") is None:
                slot["station"] = closest_pkg
        pkg_target = (closest_pkg[1], closest_pkg[0])  # (row, col) for find_picker_path
        picker.path = self.find_picker_path(
            (picker.y, picker.x), pkg_target, care_for_agents=True
        )
        picker.state = PickerState.WALKING_TO_PACKAGING
        orders = list({c.order_number for c in picker.task.claims})
        logger.info(
            "step=%d picker_id=%d: all claims picked -> WALKING_TO_PACKAGING "
            "order(s)=%s pkg=(col=%d row=%d) home_zone=%s local_candidates=%d path_len=%d",
            self._cur_steps, picker.id, orders,
            closest_pkg[0], closest_pkg[1], picker.home_zone,
            len(packaging_candidates), len(picker.path),
        )

    def _picker_walk_to_next_shelf(self, picker: "Picker") -> None:
        """Route the picker to the pickerwall shelf holding the next bin, or to packaging."""
        task = picker.task

        while (task.current_claim_index < len(task.claims)
               and task.claims[task.current_claim_index].picked):
            task.current_claim_index += 1

        if task.current_claim_index >= len(task.claims):
            self._start_walk_to_packaging(picker)
            return

        claim = task.claims[task.current_claim_index]

        def _bin_on_pickerwall(bid: int) -> bool:
            b = self._bins_by_id.get(bid)
            if b is None or b.shelf_id is None:
                return False
            s = self.shelves_by_id.get(b.shelf_id)
            return s is not None and s.cell_type == BinCellType.PICKERWALL

        if claim.bin_id == -1 or not _bin_on_pickerwall(claim.bin_id):
            resolved_id = self._resolve_pickerwall_bin_for_sku(claim.sku_entry.sku)
            if resolved_id == -1:
                logger.info(
                    "step=%d picker_id=%d: no pickerwall bin for sku=%d - WAITING_FOR_SHELF",
                    self._cur_steps, picker.id, claim.sku_entry.sku,
                )
                picker.state = PickerState.WAITING_FOR_SHELF
                return
            for c in task.claims:
                if not c.picked and c.sku_entry.sku == claim.sku_entry.sku:
                    c.bin_id = resolved_id
            claim.bin_id = resolved_id

        bin_ = self._bins_by_id[claim.bin_id]
        shelf = self.shelves_by_id[bin_.shelf_id]
        goal_xy = (shelf.x, shelf.y)

        entries = self._goal_to_picker_entry.get(goal_xy, [])
        if not entries:
            logger.warning(
                "step=%d picker_id=%d: no picker entry adjacent to goal %s for bin_id=%d - skipping",
                self._cur_steps, picker.id, goal_xy, claim.bin_id,
            )
            for c in task.claims:
                if c.bin_id == claim.bin_id and not c.picked:
                    c.picked = True
            self._picker_walk_to_next_shelf(picker)
            return

        entry_xy = min(entries, key=lambda e: abs(e[0] - picker.x) + abs(e[1] - picker.y))
        target = (entry_xy[1], entry_xy[0])
        picker.path = self.find_picker_path(
            (picker.y, picker.x), target, care_for_agents=True
        )
        picker.state = PickerState.WALKING_TO_SHELF
        logger.info(
            "step=%d picker_id=%d: ->WALKING_TO_SHELF bin_id=%d sku=%d entry=%s path_len=%d",
            self._cur_steps, picker.id, claim.bin_id, claim.sku_entry.sku,
            entry_xy, len(picker.path),
        )

    def _build_pickerwall_zones(self) -> None:
        """Group pickerwall rows into contiguous zones for simple picker policies."""
        goal_rows = sorted({y for _, y in self.goals})
        self.pickerwall_zones: List[Tuple[int, int]] = []
        self._goal_to_zone: Dict[Tuple[int, int], int] = {}
        if not goal_rows:
            return

        start = prev = goal_rows[0]
        for row in goal_rows[1:]:
            if row == prev + 1:
                prev = row
                continue
            self.pickerwall_zones.append((start, prev))
            start = prev = row
        self.pickerwall_zones.append((start, prev))

        for goal in self.goals:
            _, gy = goal
            for zone_id, (zmin, zmax) in enumerate(self.pickerwall_zones):
                if zmin <= gy <= zmax:
                    self._goal_to_zone[goal] = zone_id
                    break

    def _picker_next_cell_blocked(
        self,
        picker: "Picker",
        next_xy: Tuple[int, int],
    ) -> bool:
        """Return True when a picker should wait before entering next_xy.

        Picker routes intentionally ignore AGVs, but physical movement still
        avoids stepping into an occupied cell. AGVs remain responsible for
        yielding because they reserve picker current and next positions.
        """
        for other in self.pickers:
            if other is not picker and (other.x, other.y) == next_xy:
                return True
        for agent in self.agents:
            if (agent.x, agent.y) == next_xy:
                return True
        return False

    def _picker_stalls_this_step(self, picker: "Picker") -> bool:
        if self.picker_stall_probability <= 0:
            picker.stalled = False
            return False
        picker.stalled = bool(np.random.random() < self.picker_stall_probability)
        return picker.stalled

    def _picker_hf_context(
        self,
        picker: "Picker",
    ) -> Tuple[Optional[PickerHumanFactorsState], Optional[PickerEffortProfile]]:
        state = self._picker_hf_state_by_id.get(picker.id)
        profile = self._picker_hf_profile_by_id.get(picker.id)
        return state, profile

    def _picker_blocking_entity(self, picker: "Picker", next_xy: Tuple[int, int]) -> str:
        for other in self.pickers:
            if other is picker:
                continue
            if (other.x, other.y) == next_xy:
                return f"picker:{other.id}"
        for agv in self.agents:
            if (agv.x, agv.y) == next_xy:
                return f"agv:{agv.id}"
        return ""

    def _collect_picker_diagnostics(self) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []
        for picker in self.pickers:
            path_len = len(picker.path)
            next_xy = tuple(picker.path[0]) if path_len > 0 else None
            blocked_by = self._picker_blocking_entity(picker, next_xy) if next_xy is not None else ""
            hf_state, hf_profile = self._picker_hf_context(picker)

            current_claim = None
            current_bin_id = -1
            current_order_number = ""
            if picker.task is not None and picker.task.current_claim_index < len(picker.task.claims):
                current_claim = picker.task.claims[picker.task.current_claim_index]
                current_bin_id = int(current_claim.bin_id)
                current_order_number = str(current_claim.order_number)

            reason_code = "unknown"
            if picker.state == PickerState.IDLE:
                if picker.task is None and len(self._pickerwall_pending) == 0:
                    reason_code = "idle_no_pending_work"
                elif picker.task is None:
                    reason_code = "idle_ready_to_claim"
                else:
                    reason_code = "idle_with_task"
            elif picker.state == PickerState.WAITING_FOR_SHELF:
                reason_code = "waiting_for_shelf_arrival"
            elif picker.state in (PickerState.WALKING_TO_SHELF, PickerState.WALKING_TO_PACKAGING):
                if path_len == 0:
                    reason_code = "walking_arrived_transition"
                elif picker.speed_limited_this_step:
                    reason_code = "speed_limited_by_tick"
                elif blocked_by.startswith("agv:"):
                    reason_code = "blocked_by_agv"
                elif blocked_by.startswith("picker:"):
                    reason_code = "blocked_by_picker"
                elif picker.stalled:
                    reason_code = "stalled_or_movement_delay"
                else:
                    reason_code = "walking"
            elif picker.state == PickerState.PICKING:
                if picker.pick_ticks_remaining > 0:
                    reason_code = "picking_or_failed_pick_delay"
                else:
                    reason_code = "picking_transition"
            elif picker.state == PickerState.AT_PACKAGING:
                reason_code = "delivering_at_packaging"
            elif picker.state == PickerState.DISTRACTED:
                reason_code = "distracted"

            fatigue = float(hf_state.fatigue) if hf_state is not None else 0.0
            energy_expended = float(hf_state.energy_expended) if hf_state is not None else 0.0
            fatigue_ratio = self._picker_fatigue_ratio(hf_state) if hf_state is not None else 0.0
            profile_name = hf_state.profile_name if hf_state is not None else ""
            movement_delay_events = int(hf_state.movement_delay_events) if hf_state is not None else 0
            failed_pick_delay_events = int(hf_state.failed_pick_delay_events) if hf_state is not None else 0
            cumulative_delay_steps = int(hf_state.cumulative_delay_steps) if hf_state is not None else 0
            cumulative_recovery_seconds = (
                float(hf_state.cumulative_recovery_seconds) if hf_state is not None else 0.0
            )

            movement_delay_probability = 0.0
            failed_pick_probability = 0.0
            if self.human_factors_config.enabled and hf_state is not None and hf_profile is not None:
                movement_delay_probability = min(
                    1.0,
                    max(0.0, hf_profile.movement_delay_base_prob)
                    + max(0.0, hf_profile.movement_delay_fatigue_prob_gain) * fatigue_ratio,
                )
                failed_pick_probability = min(
                    1.0,
                    max(0.0, hf_profile.failed_pick_base_prob)
                    + max(0.0, hf_profile.failed_pick_fatigue_prob_gain) * fatigue_ratio,
                )

            diagnostics.append({
                "picker_id": int(picker.id),
                "state": picker.state.name,
                "x": int(picker.x),
                "y": int(picker.y),
                "path_len": int(path_len),
                "blocked_ticks": int(picker.blocked_ticks),
                "fixing_clash": int(picker.fixing_clash),
                "pick_ticks_remaining": int(picker.pick_ticks_remaining),
                "stalled": bool(picker.stalled),
                "speed_limited": bool(picker.speed_limited_this_step),
                "blocked_by": blocked_by,
                "current_bin_id": int(current_bin_id),
                "current_order": current_order_number,
                "pending_pickerwall_claims": int(len(self._pickerwall_pending)),
                "reason_code": reason_code,
                "hf_enabled": bool(self.human_factors_config.enabled),
                "profile_name": profile_name,
                "fatigue": fatigue,
                "fatigue_ratio": float(fatigue_ratio),
                "energy_expended": energy_expended,
                "movement_delay_probability": float(movement_delay_probability),
                "failed_pick_probability": float(failed_pick_probability),
                "movement_delay_events": movement_delay_events,
                "failed_pick_delay_events": failed_pick_delay_events,
                "cumulative_delay_steps": cumulative_delay_steps,
                "cumulative_recovery_seconds": cumulative_recovery_seconds,
            })
        return diagnostics

    def _apply_picker_effort(
        self,
        state: PickerHumanFactorsState,
        profile: PickerEffortProfile,
        metabolic_rate_per_second: float,
    ) -> None:
        effort = max(0.0, metabolic_rate_per_second) * self.time_config.simulated_seconds_per_step
        state.energy_expended += effort
        fatigue_delta = effort * max(0.0, profile.fatigue_gain_per_effort)
        state.fatigue = min(self.human_factors_config.fatigue_max, state.fatigue + fatigue_delta)

    def _recover_picker_fatigue(
        self,
        state: PickerHumanFactorsState,
        profile: PickerEffortProfile,
    ) -> None:
        recovery = max(0.0, profile.fatigue_recovery_per_second) * self.time_config.simulated_seconds_per_step
        if recovery <= 0:
            return
        state.fatigue = max(self.human_factors_config.fatigue_min, state.fatigue - recovery)
        state.cumulative_recovery_seconds += self.time_config.simulated_seconds_per_step

    def _picker_fatigue_ratio(self, state: PickerHumanFactorsState) -> float:
        span = self.human_factors_config.fatigue_max - self.human_factors_config.fatigue_min
        if span <= 0:
            return 0.0
        return min(1.0, max(0.0, (state.fatigue - self.human_factors_config.fatigue_min) / span))

    def _picker_movement_delay_this_step(
        self,
        picker: "Picker",
        state: Optional[PickerHumanFactorsState],
        profile: Optional[PickerEffortProfile],
    ) -> bool:
        if not self.human_factors_config.enabled or state is None or profile is None:
            return False
        fatigue_ratio = self._picker_fatigue_ratio(state)
        delay_prob = (
            max(0.0, profile.movement_delay_base_prob)
            + max(0.0, profile.movement_delay_fatigue_prob_gain) * fatigue_ratio
        )
        if np.random.random() >= min(1.0, delay_prob):
            return False
        state.movement_delay_events += 1
        state.cumulative_delay_steps += 1
        self._picker_hf_episode_delay_steps += 1
        picker.stalled = True
        return True

    def _picker_failed_pick_delay_this_step(
        self,
        picker: "Picker",
        state: Optional[PickerHumanFactorsState],
        profile: Optional[PickerEffortProfile],
    ) -> bool:
        if not self.human_factors_config.enabled or state is None or profile is None:
            return False
        fatigue_ratio = self._picker_fatigue_ratio(state)
        fail_prob = (
            max(0.0, profile.failed_pick_base_prob)
            + max(0.0, profile.failed_pick_fatigue_prob_gain) * fatigue_ratio
        )
        if np.random.random() >= min(1.0, fail_prob):
            return False
        delay_steps = max(1, self.simulated_seconds_to_steps(profile.failed_pick_delay_seconds))
        picker.pick_ticks_remaining = delay_steps
        state.failed_pick_delay_events += 1
        state.cumulative_delay_steps += delay_steps
        self._picker_hf_episode_delay_steps += delay_steps
        self._picker_hf_episode_failed_pick_delays += 1
        return True

    def _pick_ticks_for_current_shelf(self, picker: "Picker") -> int:
        base_seconds = self.human_factors_config.pick_base_seconds
        if not self.use_sku_size_pick_time or not picker.task:
            return max(1, self.simulated_seconds_to_steps(base_seconds))

        current_bin_id = picker.task.claims[picker.task.current_claim_index].bin_id
        unit_cube = 0.0
        quantity = 0
        for claim in picker.task.claims:
            if claim.picked or claim.bin_id != current_bin_id:
                continue
            unit_cube = max(unit_cube, float(claim.sku_entry.unit_cube or 0.0))
            quantity += claim.sku_entry.quantity

        size_seconds = unit_cube * self.human_factors_config.pick_unit_cube_seconds_scale
        quantity_seconds = max(0, quantity - 1) * self.human_factors_config.pick_quantity_extra_seconds
        total_seconds = base_seconds + size_seconds + quantity_seconds

        hf_state, hf_profile = self._picker_hf_context(picker)
        if self.human_factors_config.enabled and hf_state is not None and hf_profile is not None:
            fatigue_ratio = self._picker_fatigue_ratio(hf_state)
            total_seconds *= 1.0 + fatigue_ratio * max(0.0, hf_profile.pick_duration_fatigue_gain)

        return max(1, self.simulated_seconds_to_steps(total_seconds))

    def _pickerwall_pending_priority(
        self,
        entry: Tuple[int, "SKUEntry", "Order"],
        sku_counts: Dict[int, int],
        min_time: float,
        time_range: float,
        simulated_seconds: float,
    ) -> float:
        """Score a pickerwall_pending entry for sorting (higher = picked sooner).

        Override this method to swap in a different priority strategy.

        Parameters
        ----------
        entry : (bin_id, sku_entry, order)
        sku_counts : mapping of sku -> number of pending picks for that sku
        min_time : earliest order creation time among all pending entries
        time_range : span between earliest and latest creation times (>0)
        simulated_seconds : current simulation time in seconds
        """
        _bin_id, sku_entry, order = entry

        # Compute completion ratio for the order, the number of units filled / total units
        slot = self._packaging_slots.get(order.order_number)
        if slot and slot["required"] > 0:
            completion_ratio = slot["delivered"] / slot["required"]
        else:
            completion_ratio = 0.0
        completion_score = completion_ratio ** PICKER_PRIORITY_COMPLETION_EXPONENT

        # Compute SKU batching score to prioritize picking multiple items of the same SKU together
        sku_count = sku_counts.get(sku_entry.sku, 1)
        sku_batch_score = min(sku_count / PICKER_PRIORITY_SKU_BATCH_CAP, 1.0)

        # Compute the amount of time the order has been in the system
        if time_range > 0:
            age_score = 1.0 - (order.time_created_seconds - min_time) / time_range
        else:
            age_score = 0.5

        return (
            PICKER_PRIORITY_COMPLETION_WEIGHT * completion_score
            + PICKER_PRIORITY_SKU_BATCH_WEIGHT * sku_batch_score
            + PICKER_PRIORITY_AGE_WEIGHT * age_score
        )

    def _sort_pickerwall_pending(self) -> None:
        """Sort _pickerwall_pending so the highest-priority picks come first."""
        if len(self._pickerwall_pending) < 2:
            return

        # Pre-compute shared context once
        sku_counts: Dict[int, int] = {}
        times: List[float] = []
        for _bid, se, order in self._pickerwall_pending:
            sku_counts[se.sku] = sku_counts.get(se.sku, 0) + 1
            times.append(order.time_created_seconds)

        min_time = min(times)
        max_time = max(times)
        time_range = max_time - min_time if max_time > min_time else 1.0
        sim_seconds = self._cur_steps / max(1e-6, self.steps_per_simulated_second)

        items = list(self._pickerwall_pending)
        items.sort(
            key=lambda e: self._pickerwall_pending_priority(
                e, sku_counts, min_time, time_range, sim_seconds
            ),
            reverse=True,
        )
        self._pickerwall_pending.clear()
        self._pickerwall_pending.extend(items)

    def _advance_pickers(self) -> None:
        """Drive all picker agents through their state machine each step."""
        if len(self.pickers) > 1:
            self.resolve_picker_conflicts()
        self._sort_pickerwall_pending()
        for picker in self.pickers:
            hf_state, hf_profile = self._picker_hf_context(picker)

            if picker.state in (PickerState.IDLE, PickerState.WAITING_FOR_SHELF):
                if self.human_factors_config.enabled and hf_state is not None and hf_profile is not None:
                    self._recover_picker_fatigue(hf_state, hf_profile)

            if picker.state != PickerState.IDLE and self._picker_stalls_this_step(picker):
                continue

            if picker.fixing_clash > 0:
                picker.fixing_clash -= 1

            # IDLE
            if picker.state == PickerState.IDLE:
                picker.stalled = False
                claims = self._claim_items_for_picker(picker)
                if not claims:
                    continue
                picker.task = self._build_picker_task(picker, claims)
                logger.info(
                    "step=%d picker_id=%d: claimed %d item(s) across %d order(s)",
                    self._cur_steps, picker.id,
                    sum(c.sku_entry.quantity for c in claims),
                    len({c.order_number for c in claims}),
                )
                self._picker_walk_to_next_shelf(picker)

            # WAITING_FOR_SHELF
            elif picker.state == PickerState.WAITING_FOR_SHELF:
                self._picker_walk_to_next_shelf(picker)

            # WALKING_TO_SHELF
            elif picker.state == PickerState.WALKING_TO_SHELF:
                if picker.path:
                    if not self._consume_motion_credit(picker, self._picker_cells_per_step_effective):
                        picker.speed_limited_this_step = True
                        continue
                    if self._picker_movement_delay_this_step(picker, hf_state, hf_profile):
                        continue
                    next_xy = picker.path[0]  # (col, row) = (x, y)
                    if self._picker_next_cell_blocked(picker, next_xy):
                        picker.blocked_ticks += 1
                        if picker.blocked_ticks >= _PICKER_BLOCKED_REROUTE_THRESHOLD and len(picker.path) > 1:
                            blocking_picker = next(
                                (p for p in self.pickers if p is not picker and (p.x, p.y) == next_xy), None
                            )
                            if blocking_picker is None or blocking_picker.fixing_clash == 0:
                                dest_col, dest_row = picker.path[-1]
                                new_path = self.find_picker_path(
                                    (picker.y, picker.x), (dest_row, dest_col), care_for_agents=True
                                )
                                if new_path:
                                    logger.debug(
                                        "step=%d picker_id=%d: deadlock detour after %d blocked steps",
                                        self._cur_steps, picker.id, picker.blocked_ticks,
                                    )
                                    picker.path = new_path
                                    picker.fixing_clash = _FIXING_CLASH_TIME
                                picker.blocked_ticks = 0
                        continue
                    picker.x, picker.y = next_xy[0], next_xy[1]
                    picker.path = picker.path[1:]
                    picker.blocked_ticks = 0
                    if self.human_factors_config.enabled and hf_state is not None and hf_profile is not None:
                        self._apply_picker_effort(hf_state, hf_profile, hf_profile.metabolic_rate_walking)
                else:
                    picker.state = PickerState.PICKING
                    picker.pick_ticks_remaining = self._pick_ticks_for_current_shelf(picker)
                    claim = picker.task.claims[picker.task.current_claim_index]
                    logger.info(
                        "step=%d picker_id=%d: arrived -> PICKING bin_id=%d sku=%d (pick_ticks=%d)",
                        self._cur_steps, picker.id, claim.bin_id, claim.sku_entry.sku,
                        picker.pick_ticks_remaining,
                    )

            # PICKING
            elif picker.state == PickerState.PICKING:
                picker.pick_ticks_remaining -= 1
                if self.human_factors_config.enabled and hf_state is not None and hf_profile is not None:
                    self._apply_picker_effort(hf_state, hf_profile, hf_profile.metabolic_rate_picking)
                if picker.pick_ticks_remaining <= 0:
                    if self._picker_failed_pick_delay_this_step(picker, hf_state, hf_profile):
                        logger.debug(
                            "step=%d picker_id=%d: failed pick attempt -> delay_steps=%d",
                            self._cur_steps,
                            picker.id,
                            picker.pick_ticks_remaining,
                        )
                        continue
                    current_bin_id = picker.task.claims[picker.task.current_claim_index].bin_id
                    bin_ = self._bins_by_id.get(current_bin_id)
                    # Pick every unpicked claim for this bin in a single visit
                    for claim in picker.task.claims:
                        if not claim.picked and claim.bin_id == current_bin_id:
                            claim.picked = True
                            if bin_ is not None:
                                remaining_stock = self._decrement_bin_inventory(
                                    bin_,
                                    claim.sku_entry.quantity,
                                )
                                logger.info(
                                    "step=%d picker_id=%d: picked %d unit(s) from bin_id=%d "
                                    "sku=%s for order=%s (stock_remaining=%d)",
                                    self._cur_steps, picker.id, claim.sku_entry.quantity,
                                    bin_.id, bin_.sku, claim.order_number, remaining_stock,
                                )
                    if bin_ is not None:
                        self._maybe_mark_bin_fulfilled(bin_)
                    self._picker_walk_to_next_shelf(picker)

            # WALKING_TO_PACKAGING
            elif picker.state == PickerState.WALKING_TO_PACKAGING:
                if picker.path:
                    if not self._consume_motion_credit(picker, self._picker_cells_per_step_effective):
                        picker.speed_limited_this_step = True
                        continue
                    if self._picker_movement_delay_this_step(picker, hf_state, hf_profile):
                        continue
                    next_xy = picker.path[0]
                    if self._picker_next_cell_blocked(picker, next_xy):
                        picker.blocked_ticks += 1
                        if picker.blocked_ticks >= _PICKER_BLOCKED_REROUTE_THRESHOLD and len(picker.path) > 1:
                            blocking_picker = next(
                                (p for p in self.pickers if p is not picker and (p.x, p.y) == next_xy), None
                            )
                            if blocking_picker is None or blocking_picker.fixing_clash == 0:
                                dest_col, dest_row = picker.path[-1]
                                new_path = self.find_picker_path(
                                    (picker.y, picker.x), (dest_row, dest_col), care_for_agents=True
                                )
                                if new_path:
                                    logger.debug(
                                        "step=%d picker_id=%d: deadlock detour after %d blocked steps (packaging)",
                                        self._cur_steps, picker.id, picker.blocked_ticks,
                                    )
                                    picker.path = new_path
                                    picker.fixing_clash = _FIXING_CLASH_TIME
                                picker.blocked_ticks = 0
                        continue
                    picker.x, picker.y = next_xy[0], next_xy[1]
                    picker.path = picker.path[1:]
                    picker.blocked_ticks = 0
                    if self.human_factors_config.enabled and hf_state is not None and hf_profile is not None:
                        self._apply_picker_effort(hf_state, hf_profile, hf_profile.metabolic_rate_walking)
                else:
                    picker.state = PickerState.AT_PACKAGING
                    logger.info(
                        "step=%d picker_id=%d: arrived at packaging -> AT_PACKAGING order(s)=%s",
                        self._cur_steps, picker.id,
                        list({c.order_number for c in picker.task.claims}),
                    )

            # AT_PACKAGING
            elif picker.state == PickerState.AT_PACKAGING:
                if self.human_factors_config.enabled and hf_state is not None and hf_profile is not None:
                    self._apply_picker_effort(hf_state, hf_profile, hf_profile.metabolic_rate_idle)
                for claim in picker.task.claims:
                    slot = self._packaging_slots.get(claim.order_number)
                    if slot is None:
                        continue  # order already completed by a previous deposit
                    slot["delivered"] += claim.sku_entry.quantity
                    per_sku = slot.setdefault("delivered_per_sku", {})
                    per_sku[claim.sku_entry.sku] = (
                        per_sku.get(claim.sku_entry.sku, 0) + claim.sku_entry.quantity
                    )
                    if slot["delivered"] >= slot["required"]:
                        logger.info(
                            "step=%d: order=%s COMPLETE at packaging station",
                            self._cur_steps, claim.order_number,
                        )
                        del self._packaging_slots[claim.order_number]
                logger.info(
                    "step=%d picker_id=%d: AT_PACKAGING->IDLE",
                    self._cur_steps, picker.id,
                )
                picker.packaging_location = None
                picker.task = None
                picker.state = PickerState.IDLE

    def step(
        self, macro_actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], Dict]:
        for agent in self.agents:
            agent.speed_limited_this_step = False
        for picker in self.pickers:
            picker.speed_limited_this_step = False

        if self.order_sequencer is not None:
            newly_released = self.order_sequencer.release_pending_orders(self._cur_steps)
            if newly_released:
                logger.info(
                    "step=%d: released %d new orders into active queue "
                    "(pending=%d active=%d queue=%d/%d)",
                    self._cur_steps, len(newly_released),
                    self.order_sequencer.pending_count,
                    self.order_sequencer.active_count,
                    len(self.request_queue), self.request_queue_size,
                )
            self._refill_request_queue()

        agvs_distance_travelled = self.attribute_macro_actions(macro_actions)
        clashes_count = self.resolve_move_conflict(self.agents)
        picker_yields_count = self._apply_picker_yield_to_agvs()
        stucks_count = self.resolve_stuck_agents()

        rewards = np.zeros(self.num_agents)
        rewards -= 0.001
        self._step_deliveries = 0
        rewards = self.execute_micro_actions(rewards)
        self._advance_pickers()
        rewards, shelf_deliveries = self.process_shelf_deliveries(rewards)

        self._recalc_grid()
        self._cur_steps += 1
        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps):
            terminateds = truncateds = self.num_agents * [True]
        else:
            terminateds = truncateds =  self.num_agents * [False]

        self.observation_space_mapper.extract_environment_info(self)
        new_obs = tuple([self.observation_space_mapper.observation(agent) for agent in self.agents])
        episode_done = bool(all(terminateds) or all(truncateds))
        info = self._build_info(
            agvs_distance_travelled,
            clashes_count,
            picker_yields_count,
            stucks_count,
            shelf_deliveries,
            episode_done=episode_done,
        )
        return new_obs, list(rewards), terminateds, terminateds, info

    def _build_info(
        self,
        agvs_distance_travelled: int,
        clashes_count: int,
        picker_yields_count: int,
        stucks_count: int,
        shelf_deliveries: int,
        episode_done: bool = False,
    ) -> Dict[str, np.ndarray]:
        info = {}
        agvs_idle_time = sum([int(agent.req_action in (Action.NOOP, Action.TOGGLE_LOAD)) for agent in self.agents])
        info["vehicles_busy"] = [agent.busy for agent in self.agents]
        info["shelf_deliveries"] = shelf_deliveries
        info["clashes"] = clashes_count
        info["picker_yields"] = picker_yields_count
        info["stucks"] = stucks_count
        info["agvs_distance_travelled"] = agvs_distance_travelled
        info["agvs_idle_time"] = agvs_idle_time
        info["steps_per_simulated_second"] = self.time_config.steps_per_simulated_second
        info["simulated_seconds"] = self.steps_to_simulated_seconds(self._cur_steps)
        info["real_seconds"] = self.steps_to_real_seconds(self._cur_steps)
        info["agv_nominal_cells_per_step"] = self.agv_nominal_cells_per_step()
        info["picker_nominal_cells_per_step"] = self.picker_nominal_cells_per_step()
        info["motion_speed_model"] = "physical_m_s" if self._use_physical_speed_model else "cells_per_step"
        info["agv_cells_per_step_configured"] = float(self._agv_cells_per_step_configured)
        info["agv_cells_per_step_effective"] = float(self._agv_cells_per_step_effective)
        info["picker_cells_per_step_configured"] = float(self._picker_cells_per_step_configured)
        info["picker_cells_per_step_effective"] = float(self._picker_cells_per_step_effective)
        info["agv_nominal_speed_m_s"] = float(self.time_config.agv_nominal_speed_m_s)
        info["picker_nominal_speed_m_s"] = float(self.time_config.picker_nominal_speed_m_s)
        info["agv_speed_limited_count"] = int(sum(1 for a in self.agents if a.speed_limited_this_step))
        info["human_factors_model"] = self.human_factors_config.model_name

        if self.pickers:
            picker_diagnostics = self._collect_picker_diagnostics()
            self._latest_picker_diagnostics = picker_diagnostics

            picker_fatigue = []
            picker_energy_expended = []
            picker_profiles = []
            for picker in self.pickers:
                state = self._picker_hf_state_by_id.get(picker.id)
                picker_fatigue.append(float(state.fatigue) if state else 0.0)
                picker_energy_expended.append(float(state.energy_expended) if state else 0.0)
                picker_profiles.append(state.profile_name if state else "")
            info["picker_fatigue"] = np.array(picker_fatigue, dtype=np.float32)
            info["picker_energy_expended"] = np.array(picker_energy_expended, dtype=np.float32)
            info["picker_profiles"] = picker_profiles
            info["picker_delay_steps"] = self._picker_hf_episode_delay_steps
            info["picker_failed_pick_delay_events"] = self._picker_hf_episode_failed_pick_delays
            info["picker_fatigue_mean"] = float(np.mean(info["picker_fatigue"]))
            info["picker_fatigue_max"] = float(np.max(info["picker_fatigue"]))
            info["picker_energy_total"] = float(np.sum(info["picker_energy_expended"]))
            info["picker_states"] = [d["state"] for d in picker_diagnostics]
            info["picker_reason_codes"] = [d["reason_code"] for d in picker_diagnostics]
            info["picker_positions"] = [(d["x"], d["y"]) for d in picker_diagnostics]
            info["picker_path_lengths"] = np.array([d["path_len"] for d in picker_diagnostics], dtype=np.int32)
            info["picker_blocked_ticks"] = np.array([d["blocked_ticks"] for d in picker_diagnostics], dtype=np.int32)
            info["picker_blocked_by"] = [d["blocked_by"] for d in picker_diagnostics]
            info["picker_stalled"] = [d["stalled"] for d in picker_diagnostics]
            info["picker_speed_limited"] = [d["speed_limited"] for d in picker_diagnostics]
            info["picker_hf_enabled"] = [d["hf_enabled"] for d in picker_diagnostics]
            info["picker_hf_profile_name"] = [d["profile_name"] for d in picker_diagnostics]
            info["picker_hf_fatigue"] = np.array([d["fatigue"] for d in picker_diagnostics], dtype=np.float32)
            info["picker_hf_fatigue_ratio"] = np.array([d["fatigue_ratio"] for d in picker_diagnostics], dtype=np.float32)
            info["picker_hf_energy_expended"] = np.array([d["energy_expended"] for d in picker_diagnostics], dtype=np.float32)
            info["picker_hf_movement_delay_probability"] = np.array(
                [d["movement_delay_probability"] for d in picker_diagnostics], dtype=np.float32
            )
            info["picker_hf_failed_pick_probability"] = np.array(
                [d["failed_pick_probability"] for d in picker_diagnostics], dtype=np.float32
            )
            info["picker_hf_movement_delay_events"] = np.array(
                [d["movement_delay_events"] for d in picker_diagnostics], dtype=np.int32
            )
            info["picker_hf_failed_pick_delay_events_per_picker"] = np.array(
                [d["failed_pick_delay_events"] for d in picker_diagnostics], dtype=np.int32
            )
            info["picker_hf_cumulative_delay_steps"] = np.array(
                [d["cumulative_delay_steps"] for d in picker_diagnostics], dtype=np.int32
            )
            info["picker_hf_cumulative_recovery_seconds"] = np.array(
                [d["cumulative_recovery_seconds"] for d in picker_diagnostics], dtype=np.float32
            )
            info["picker_diagnostics"] = picker_diagnostics

            if episode_done:
                info["episode_human_factors_summary"] = {
                    "model": self.human_factors_config.model_name,
                    "picker_count": len(self.pickers),
                    "delay_steps": int(self._picker_hf_episode_delay_steps),
                    "failed_pick_delay_events": int(self._picker_hf_episode_failed_pick_delays),
                    "fatigue_mean": float(np.mean(info["picker_fatigue"])),
                    "fatigue_max": float(np.max(info["picker_fatigue"])),
                    "energy_total": float(np.sum(info["picker_energy_expended"])),
                    "simulated_seconds": float(info["simulated_seconds"]),
                }
        return info

    def compute_valid_action_masks(self, block_conflicting_actions=True):
        requested_items = self.get_shelf_request_information()
        empty_items = self.get_empty_shelf_information()
        carrying_bin_info = self.get_carrying_bin_information()
        pickerwall_occupied = self.get_pickerwall_info()
        pickerwall_displaceable = self.get_pickerwall_displacement_info()

        non_goal_base = self._storage_action_id_base
        targets_agvs = [
            target - non_goal_base
            for target in self.targets_agvs
            if target >= non_goal_base
        ]

        valid_location_list_agvs = np.array([
            empty_items if carrying_bin else requested_items for carrying_bin in carrying_bin_info
        ])
        valid_goal_list_agvs = np.array([
            (1 - pickerwall_occupied) if carrying_bin else pickerwall_displaceable
            for carrying_bin in carrying_bin_info
        ])

        if block_conflicting_actions:
            valid_location_list_agvs[:, targets_agvs] = 0

        valid_action_masks = np.ones((self.num_agents, self.action_size))
        valid_action_masks[:, non_goal_base:] = valid_location_list_agvs
        valid_action_masks[:, 1:non_goal_base] = valid_goal_list_agvs
        return valid_action_masks

    def render(self, mode="human"):
        if not self.renderer:
            from tarware.rendering import Viewer
            if not self.render_tile_size:
                self.render_tile_size = int(os.getenv("TARWARE_RENDER_TILE_SIZE", "30"))
            self.renderer = Viewer(self.render_tile_size, self.grid_size)
        return self.renderer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
