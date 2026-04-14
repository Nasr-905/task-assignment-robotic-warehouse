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
_AGV_WALKABLE_TILES = {0, 5, 6}
_PICKER_WALKABLE_TILES = {3, 4, 6}
_SHARED_HIGHWAY_TILE = 6
BIN_VOLUME_FT3 = 2.68
BIN_USABLE_FRACTION = 0.85
BIN_LEVELS_PER_SIDE = 5
BINS_PER_LEVEL = 5


class BinCellType(Enum):
    STORAGE = "storage"
    PICKERWALL = "pickerwall"
    REPLENISHMENT = "replenishment"


BIN_SIDES_BY_CELL_TYPE = {
    BinCellType.STORAGE: 2,
    BinCellType.PICKERWALL: 1,
    BinCellType.REPLENISHMENT: 1,
}


@dataclass
class LogicalBin:
    """Physical bin metadata and Stage A2 inventory assignment.

    Stage A2 assigns one SKU per logical bin and computes quantity from
    SKU unit cube and usable bin volume. Movement still uses Shelf objects as
    a bridge until bin-level AGV tasks are implemented.
    """
    id: int
    cell_id: int
    x: int
    y: int
    cell_type: BinCellType
    side: int
    level: int
    slot: int
    volume_ft3: float
    usable_fraction: float
    sku: Optional[int] = None
    quantity: int = 0
    used_volume_ft3: float = 0.0

    @property
    def usable_volume_ft3(self) -> float:
        return self.volume_ft3 * self.usable_fraction

    @property
    def remaining_volume_ft3(self) -> float:
        return max(0.0, self.usable_volume_ft3 - self.used_volume_ft3)


@dataclass
class BinCell:
    """Fixed map cell that contains a conglomerate of logical bins."""
    id: int
    x: int
    y: int
    cell_type: BinCellType
    bins: List[LogicalBin]

    @property
    def bin_count(self) -> int:
        return len(self.bins)

    @property
    def usable_volume_ft3(self) -> float:
        return sum(bin_.usable_volume_ft3 for bin_ in self.bins)

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
        self.carrying_shelf: Optional[Shelf] = None
        self.canceled_action = None
        self.has_delivered = False
        self.path = None
        self.busy = False
        self.fixing_clash = 0
        self.type = agent_type
        self.target = 0

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
    counter = 0
    DEFAULT_CAPACITY = 10

    def __init__(self, x, y):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)
        self.sku: Optional[int] = None  # assigned by OrderSequencer.initialize_shelf_sku_map
        self.capacity: int = Shelf.DEFAULT_CAPACITY   # remaining stock units
        self.initial_capacity: int = Shelf.DEFAULT_CAPACITY
        self.fulfilled: bool = False  # True once all orders referencing this pickerwall shelf have been picked
        self.depleted: bool = False        # stock hit 0; shelf should be removed when returned
        self.on_grid: bool = True          # False after a depleted shelf is removed from the warehouse
        self.from_replenishment: bool = False  # spawned in the replenishment zone
        self.bin_id: Optional[int] = None  # Stage A2 bridge to the representative logical bin

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
    shelf_id: int          # pickerwall shelf_id to visit; -1 if not yet at pickerwall
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
        self.pick_ticks_remaining: int = 0
        self.capacity: int = Picker.CAPACITY
        self.blocked_ticks: int = 0   # consecutive steps spent waiting on a blocked cell
        self.fixing_clash: int = 0    # cooldown after rerouting; mirrors Agent.fixing_clash
        self.home_zone: Optional[int] = None
        self.stalled: bool = False
        self.packaging_location: Optional[Tuple[int, int]] = None


class Warehouse(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        map_csv_path: Path,
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
    ):
        """Multi-agent robotic warehouse gym environment."""
        self.steps_per_simulated_second = max(1e-6, float(steps_per_simulated_second))
        self.time_config = PhysicalTimeConfig.from_env(self.steps_per_simulated_second)
        self.bin_volume_ft3 = float(os.getenv("TARWARE_BIN_VOLUME_FT3", str(BIN_VOLUME_FT3)))
        self.bin_usable_fraction = float(
            os.getenv("TARWARE_BIN_USABLE_FRACTION", str(BIN_USABLE_FRACTION))
        )
        self._make_order_sequencer_from_csv(order_csv_path, self.steps_per_simulated_second)
        self._make_layout_from_csv(map_csv_path)

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
            len(self.action_id_to_coords_map)-len(self.goals),
            len(self.goals),
            normalised_coordinates,
        )
        self.observation_space = spaces.Tuple(tuple(self.observation_space_mapper.ma_spaces))

        self.request_queue_size = request_queue_size
        self.request_queue = []
        self._step_deliveries = 0
        
        self.rack_groups = find_sections(list([loc for loc in self.action_id_to_coords_map.values() if (loc[1], loc[0]) not in self.goals]))
        self.agents: List[Agent] = []
        self.pickers: List[Picker] = []
        self.stuck_counters = []
        self.renderer = None
        self._shelf_to_order: Dict[int, Any] = {}  # shelf_id -> Order that triggered its fetch
        self._packaging_slots: Dict[str, Dict[str, Any]] = {}  # order_number -> {required, delivered, station}
        self._pickerwall_pending: deque = deque()  # (shelf_id, SKUEntry, Order) populated on delivery
        self._pickerwall_slot_by_shelf_id: Dict[int, int] = {}
        self._replenishment_slot_by_shelf_id: Dict[int, int] = {}
        self._slot_occupancy_by_slot_id: Dict[int, int] = {}
        self._picker_hf_profile_by_id: Dict[int, PickerEffortProfile] = {}
        self._picker_hf_state_by_id: Dict[int, PickerHumanFactorsState] = {}
        self._picker_hf_episode_delay_steps = 0
        self._picker_hf_episode_failed_pick_delays = 0

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

    def _make_bin_cell(
        self,
        cell_id: int,
        x: int,
        y: int,
        cell_type: BinCellType,
        first_bin_id: int,
    ) -> BinCell:
        """Build the logical bin conglomerate for one fixed map cell."""
        bins: List[LogicalBin] = []
        sides = BIN_SIDES_BY_CELL_TYPE[cell_type]
        for side in range(sides):
            for level in range(BIN_LEVELS_PER_SIDE):
                for slot in range(BINS_PER_LEVEL):
                    bins.append(
                        LogicalBin(
                            id=first_bin_id + len(bins),
                            cell_id=cell_id,
                            x=x,
                            y=y,
                            cell_type=cell_type,
                            side=side,
                            level=level,
                            slot=slot,
                            volume_ft3=self.bin_volume_ft3,
                            usable_fraction=self.bin_usable_fraction,
                        )
                    )
        return BinCell(
            id=cell_id,
            x=x,
            y=y,
            cell_type=cell_type,
            bins=bins,
        )

    def _make_bin_cells(self) -> None:
        """Create Stage A1 logical bin metadata for storage-like map cells."""
        self.bin_cells: List[BinCell] = []
        self.bin_cells_by_xy: Dict[Tuple[int, int], BinCell] = {}
        next_bin_id = 1

        def add_cell(x: int, y: int, cell_type: BinCellType) -> None:
            nonlocal next_bin_id
            cell = self._make_bin_cell(
                len(self.bin_cells) + 1,
                x,
                y,
                cell_type,
                next_bin_id,
            )
            next_bin_id += len(cell.bins)
            self.bin_cells.append(cell)
            self.bin_cells_by_xy[(x, y)] = cell

        for row, col in self.shelf_locs:
            add_cell(col, row, BinCellType.STORAGE)
        for x, y in self.goals:
            add_cell(x, y, BinCellType.PICKERWALL)
        for x, y in self.replenishment_locs:
            add_cell(x, y, BinCellType.REPLENISHMENT)

        self.logical_bins: List[LogicalBin] = [
            bin_
            for cell in self.bin_cells
            for bin_ in cell.bins
        ]
        self.logical_bins_by_id: Dict[int, LogicalBin] = {
            bin_.id: bin_
            for bin_ in self.logical_bins
        }
        self.storage_logical_bins: List[LogicalBin] = [
            bin_
            for cell in self.bin_cells
            if cell.cell_type == BinCellType.STORAGE
            for bin_ in cell.bins
        ]
        self.pickerwall_logical_bins: List[LogicalBin] = [
            bin_
            for cell in self.bin_cells
            if cell.cell_type == BinCellType.PICKERWALL
            for bin_ in cell.bins
        ]
        self.replenishment_logical_bins: List[LogicalBin] = [
            bin_
            for cell in self.bin_cells
            if cell.cell_type == BinCellType.REPLENISHMENT
            for bin_ in cell.bins
        ]

    def _make_layout_from_csv(self, map_csv_path: Path) -> None:
        """Build the warehouse layout from a CSV tile map.

        Tile encoding: 0=highway, 1=shelf/storage, 2=pickerwall, 3=picker_highway,
        4=packaging, 5=replenishment (take-only), 6=shared_highway, 9=blank.
        """

        df = pd.read_csv(map_csv_path, header=None).fillna(0)
        tile_grid = df.values.astype(int)  # shape (num_rows, num_cols)
        num_rows, num_cols = tile_grid.shape

        # AGV zone ends at the last row containing any AGV-side tile.
        agv_zone_height = num_rows
        for r in range(num_rows - 1, -1, -1):
            if any(tile_grid[r, c] in (0, 1, 2, 5, 6) for c in range(num_cols)):
                agv_zone_height = r + 1
                break
        picker_zone_rows = num_rows - agv_zone_height
        self._picker_zone_rows = picker_zone_rows
        self.agv_zone_height = agv_zone_height
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
        self._make_bin_cells()

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
                and self.picker_highways[ny, nx] == 1
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
                and self.highways[ny, nx] == 1       # AGV-walkable
                and (nx, ny) not in self.goals        # not another goal slot
            ]
            self._goal_to_agv_entry[(gx, gy)] = entries

        # Reverse: AGV-entry (col, row) → goal (col, row)
        self._agv_entry_to_goal: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for goal_xy, entries in self._goal_to_agv_entry.items():
            for entry_xy in entries:
                self._agv_entry_to_goal[entry_xy] = goal_xy

        # Action IDs: 1..num_goals (pickerwall), then storage, then replenishment
        self.action_id_to_coords_map = {i + 1: (r, c) for i, (c, r) in enumerate(self.goals)}
        item_loc_index = len(self.action_id_to_coords_map) + 1
        for (r, c) in self.shelf_locs:
            self.action_id_to_coords_map[item_loc_index] = (r, c)
            item_loc_index += 1
        self._replenishment_action_id_base = item_loc_index
        for (x, y) in self.replenishment_locs:
            self.action_id_to_coords_map[item_loc_index] = (y, x)  # stored as (row, col)
            item_loc_index += 1

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

        logger.info(
            "Map loaded from CSV: %s | grid=%s agv_zone=%d picker_zone=%d "
            "goals=%d shelves=%d replenishment=%d packaging=%d bin_cells=%d "
            "logical_bins=%d bin_usable_volume_ft3=%.3f column_height=%d",
            map_csv_path, self.grid_size, agv_zone_height, picker_zone_rows,
            self.num_goals, len(self.shelf_locs), len(self.replenishment_locs),
            len(self.packaging_locations), len(self.bin_cells), len(self.logical_bins),
            self.bin_volume_ft3 * self.bin_usable_fraction, self.column_height,
        )

    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]

    def find_agv_path(self, start, goal: Tuple[int], agent: Tuple[int], care_for_agents: bool = True) -> List[Tuple[int]]:
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
        
    def find_agv_path_through_adjacent_loc(self, start, goal: Tuple[int], agent: Tuple[int], care_for_agents: bool = True) -> List[Tuple[int]]:
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
            p = self.find_agv_path(start, entry_rc, agent, care_for_agents)
            if p and (not best_path or len(p) < len(best_path)):
                best_via = entry_rc
                best_path = p
        via = best_via
        path_to_via = best_path
        if not path_to_via:
            return []
        path_from_via = self.find_agv_path(via, goal, agent, care_for_agents)
        if not path_from_via:
            return []
        return path_to_via + path_from_via
    
    def find_agv_path_to_target_entry(self, start: Tuple[int, int], target_rc: Tuple[int, int],
                                      agent: "Agent", care_for_agents: bool = True) -> List[Tuple[int, int]]:
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
            p = self.find_agv_path(start, entry_rc, agent, care_for_agents)
            if p and (not best_path or len(p) < len(best_path)):
                best_path = p
        return best_path

    def find_agv_path_to_goal_entry(self, start: Tuple[int, int], goal_xy: Tuple[int, int],
                                    agent: "Agent", care_for_agents: bool = True) -> List[Tuple[int, int]]:
        """Route an AGV to the nearest highway cell adjacent to goal_xy (col, row).

        Used for side drop-off: the AGV stops at a highway cell next to the
        pickerwall slot and deposits the shelf sideways.  start is (row, col).
        Returns [] when no entry cell is reachable.
        """
        entries = self._goal_to_agv_entry.get(goal_xy, [])
        best_path: List[Tuple[int, int]] = []
        for (ex, ey) in entries:          # entries stored as (col, row)
            p = self.find_agv_path(start, (ey, ex), agent, care_for_agents)
            if p and (not best_path or len(p) < len(best_path)):
                best_path = p
        return best_path

    def find_picker_path(self, start, goal: Tuple[int, int], picker: "Picker", care_for_agents: bool = True) -> List[Tuple[int, int]]:
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

    def _recalc_grid(self) -> None:
        self.grid.fill(0)

        carried_shelf_ids = {agent.carrying_shelf.id for agent in self.agents if agent.carrying_shelf}
        for shelf in self.shelfs:
            if shelf.on_grid and shelf.id not in carried_shelf_ids:
                self.grid[CollisionLayers.SHELVES, shelf.y, shelf.x] = shelf.id
        for agent in self.agents:
            self.grid[CollisionLayers.AGVS, agent.y, agent.x] = agent.id
            if agent.carrying_shelf:
                self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = agent.carrying_shelf.id
        for picker in self.pickers:
            self.grid[CollisionLayers.PICKERS, picker.y, picker.x] = picker.id

    def get_carrying_shelf_information(self):
        return [agent.carrying_shelf != None for agent in self.agents[:self.num_agvs]]

    def get_shelf_request_information(self) -> np.ndarray[int]:
        n_non_goal = len(self.action_id_to_coords_map) - self.num_goals
        request_item_map = np.zeros(n_non_goal)
        requested_shelf_ids = [shelf.id for shelf in self.request_queue]
        for id_, coords in self.action_id_to_coords_map.items():
            if (coords[1], coords[0]) not in self.goals:
                if self.grid[CollisionLayers.SHELVES, coords[0], coords[1]] in requested_shelf_ids:
                    request_item_map[id_ - self.num_goals - 1] = 1
        return request_item_map

    def get_empty_shelf_information(self) -> np.ndarray[int]:
        n_non_goal = len(self.action_id_to_coords_map) - self.num_goals
        empty_item_map = np.zeros(n_non_goal)
        for id_, coords in self.action_id_to_coords_map.items():
            if (coords[1], coords[0]) in self.goals:
                continue
            if (coords[1], coords[0]) in self._replenishment_locs_set:
                continue  # replenishment zone is take-only; AGVs cannot deposit here
            if self.grid[CollisionLayers.SHELVES, coords[0], coords[1]] == 0 and (
                self.grid[CollisionLayers.CARRIED_SHELVES, coords[0], coords[1]] == 0
                or self.agents[
                    self.grid[CollisionLayers.AGVS, coords[0], coords[1]] - 1
                ].req_action
                not in [Action.NOOP, Action.TOGGLE_LOAD]
            ):
                empty_item_map[id_ - self.num_goals - 1] = 1
        return empty_item_map

    def get_pickerwall_info(self) -> np.ndarray:
        """Returns a boolean array of length num_goals: 1 where a shelf is deposited at that goal slot."""
        occupied = np.zeros(self.num_goals, dtype=np.int32)
        for i, (x, y) in enumerate(self.goals):
            if self.grid[CollisionLayers.SHELVES, y, x] != 0:
                occupied[i] = 1
        return occupied

    def get_pickerwall_displacement_info(self) -> np.ndarray:
        """Returns a boolean array of length num_goals: 1 where a pickerwall shelf is fulfilled
        (all picker claims satisfied) and eligible for AGV displacement back to storage."""
        displaceable = np.zeros(self.num_goals, dtype=np.int32)
        for i, (x, y) in enumerate(self.goals):
            shelf_id = self.grid[CollisionLayers.SHELVES, y, x]
            if shelf_id == 0:
                continue
            shelf = self.shelfs[shelf_id - 1]
            if shelf.fulfilled:
                displaceable[i] = 1
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

                    if target_xy in self.goals:
                        # Pickerwall target (pick-up or drop-off): approach from adjacent
                        # highway cell so the AGV never enters the tile-2 slot.
                        agent.path = self.find_agv_path_to_goal_entry(
                            (agent.y, agent.x), target_xy, agent, care_for_agents=True
                        )
                        if not agent.path:
                            agent.path = self.find_agv_path_to_goal_entry(
                                (agent.y, agent.x), target_xy, agent, care_for_agents=False
                            )
                    elif not self._is_highway(target_rc[1], target_rc[0]):
                        # Non-highway target (shelf tile): approach from an adjacent
                        # highway cell so the AGV never enters the shelf cell directly.
                        agent.path = self.find_agv_path_to_target_entry(
                            (agent.y, agent.x), target_rc, agent, care_for_agents=True
                        )
                        if not agent.path:
                            agent.path = self.find_agv_path_to_target_entry(
                                (agent.y, agent.x), target_rc, agent, care_for_agents=False
                            )
                    else:
                        agent.path = self.find_agv_path((agent.y, agent.x), target_rc, agent, care_for_agents=True)
                        if not agent.path:
                            # Congestion blocked the agent-aware path; fall back to ignoring agents
                            # so the agent can at least start moving. resolve_move_conflict handles
                            # any resulting step-level collisions.
                            agent.path = self.find_agv_path((agent.y, agent.x), target_rc, agent, care_for_agents=False)

                    if agent.path:
                        agent.busy = True
                        agent.target = macro_action
                        agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
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
                    agvs_distance_travelled += 1
                if len(agent.path) == 1 and agent.carrying_shelf:
                    # If the deposit target is already occupied by a resting shelf, abort.
                    # The deposit target is the action target cell, not the entry cell.
                    target_rc = self.action_id_to_coords_map.get(agent.target)
                    if target_rc is not None:
                        tx, ty = target_rc[1], target_rc[0]
                        if self.grid[CollisionLayers.SHELVES, ty, tx]:
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
                                    new_path = self.find_agv_path((agent.y, agent.x), (agent.path[-1][1] ,agent.path[-1][0]), agent, True)
                                    if len(new_path) == 1:
                                        # Agents that are stuck and are 1 cell away from their target are likely competing with an AGV that has
                                        # the same dilemma, so, re-route agent to the target through an adjacent cell to the target.
                                        new_path = self.find_agv_path_through_adjacent_loc((agent.y, agent.x), (agent.path[-1][1] ,agent.path[-1][0]), agent, True)
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
                    agent,
                    care_for_agents=True,
                )
                if new_path:
                    agent.path = new_path

        return yields

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
        if agent.carrying_shelf:
            agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y

    def _execute_rotation(self, agent: Agent) -> None:
        agent.dir = agent.req_direction()

    def _execute_load(self, agent: Agent, rewards: np.ndarray[int]) -> np.ndarray[int]:
        # Determine which cell the shelf is in.  If the agent is adjacent to its
        # target (side pick-up from shelf rack or pickerwall), look there;
        # otherwise fall back to the agent's own cell (legacy / replenishment).
        load_x, load_y = agent.x, agent.y
        target_rc = self.action_id_to_coords_map.get(agent.target)
        if target_rc is not None:
            tx, ty = target_rc[1], target_rc[0]  # (col, row)
            if abs(tx - agent.x) + abs(ty - agent.y) == 1:
                load_x, load_y = tx, ty

        shelf_id = self.grid[CollisionLayers.SHELVES, load_y, load_x]
        if shelf_id:
            agent.carrying_shelf = self.shelfs[shelf_id - 1]
            self.grid[CollisionLayers.SHELVES, load_y, load_x] = 0
            self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = shelf_id
            self._release_reserved_slot_for_shelf(agent.carrying_shelf)
            agent.busy = False
            if self.reward_type == RewardType.GLOBAL:
                rewards += 0.5
            elif self.reward_type == RewardType.INDIVIDUAL:
                rewards[agent.id - 1] += 0.1
        else:
            agent.busy = False
        return rewards

    def _deliver_to_pickerwall(self, agent: Agent, gx: int, gy: int,
                               rewards: np.ndarray[int]) -> np.ndarray[int]:
        """Place the shelf carried by *agent* into pickerwall slot (gx, gy) and
        update the request queue and picker-wall pending list.  The agent must
        already be at its final cell (entry or goal) when this is called."""
        shelf = agent.carrying_shelf
        if not self._reserve_cell_slot_for_shelf(shelf, gx, gy, BinCellType.PICKERWALL):
            logger.warning(
                "step=%d delivery blocked: no pickerwall bin slot for shelf_id=%d at (%d,%d)",
                self._cur_steps, shelf.id, gx, gy,
            )
            agent.busy = False
            return rewards

        self.grid[CollisionLayers.SHELVES, gy, gx] = shelf.id
        self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = 0
        shelf.x, shelf.y = gx, gy
        agent.carrying_shelf = None
        agent.busy = False
        agent.has_delivered = False

        if shelf not in self.request_queue:
            self._release_reserved_slot_for_shelf(shelf)
            return rewards

        if self.reward_type == RewardType.GLOBAL:
            rewards += 1
        elif self.reward_type == RewardType.INDIVIDUAL:
            rewards[agent.id - 1] += 1

        carried_shelves = {a.carrying_shelf for a in self.agents if a.carrying_shelf}
        pickerwall_shelf_ids = {
            self.grid[CollisionLayers.SHELVES, yy, xx]
            for (xx, yy) in self.goals
            if self.grid[CollisionLayers.SHELVES, yy, xx] != 0
        }
        pickerwall_shelves = {self.shelfs[sid - 1] for sid in pickerwall_shelf_ids}
        new_shelf_candidates = [
            s for s in self.shelfs
            if s.on_grid
            and s not in self.request_queue
            and s not in carried_shelves
            and s not in pickerwall_shelves
        ]
        new_shelf_candidates.sort(key=lambda s: s.id)
        delivered_order = self._shelf_to_order.pop(shelf.id, None)
        if delivered_order is not None:
            matched_sku_entry = next(
                (se for se in delivered_order.skus if se.sku == shelf.sku), None
            )
            if matched_sku_entry is not None and self.num_pickers > 0:
                self._pickerwall_pending.append((shelf.id, matched_sku_entry, delivered_order))
            logger.info(
                "step=%d delivery: shelf_id=%d sku=%d arrived for order=%s "
                "(pickerwall_pending=%d)",
                self._cur_steps, shelf.id, shelf.sku, delivered_order.order_number,
                len(self._pickerwall_pending),
            )

        result = self.order_sequencer.next_order_shelf(new_shelf_candidates)
        if result is not None:
            new_request, new_order = result
            self._shelf_to_order[new_request.id] = new_order
            logger.info(
                "step=%d delivery: shelf_id=%d sku=%s delivered - "
                "replaced in queue with shelf_id=%d sku=%d "
                "(pending=%d active=%d)",
                self._cur_steps, shelf.id, shelf.sku,
                new_request.id, new_request.sku,
                self.order_sequencer.pending_count, self.order_sequencer.active_count,
            )
        else:
            new_request = None
            logger.info(
                "step=%d delivery: shelf_id=%d sku=%s delivered - "
                "no active orders available, queue shrinks to %d "
                "(pending=%d active=%d)",
                self._cur_steps, shelf.id, shelf.sku,
                len(self.request_queue) - 1,
                self.order_sequencer.pending_count, self.order_sequencer.active_count,
            )

        if new_request is not None:
            self.request_queue[self.request_queue.index(shelf)] = new_request
        else:
            self.request_queue.remove(shelf)
        self._step_deliveries += 1
        return rewards

    def _deposit_to_shelf_cell(self, agent: Agent, sx: int, sy: int,
                               rewards: np.ndarray[int]) -> np.ndarray[int]:
        """Place the shelf carried by *agent* into rack slot (sx, sy) and
        clear the CARRIED_SHELVES layer at the agent's current cell."""
        shelf = agent.carrying_shelf
        if shelf.depleted:
            shelf.on_grid = False
            self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = 0
            agent.carrying_shelf = None
            agent.busy = False
            agent.has_delivered = False
            logger.info(
                "step=%d: depleted shelf_id=%d sku=%d removed from warehouse",
                self._cur_steps, shelf.id, shelf.sku,
            )
        else:
            self.grid[CollisionLayers.SHELVES, sy, sx] = shelf.id
            self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = 0
            shelf.x, shelf.y = sx, sy
            agent.carrying_shelf = None
            agent.busy = False
            agent.has_delivered = False
            if self.reward_type == RewardType.GLOBAL:
                rewards += 0.5
            elif self.reward_type == RewardType.INDIVIDUAL:
                rewards[agent.id - 1] += 0.1
        return rewards

    def _execute_unload(self, agent: Agent, rewards: np.ndarray[int]) -> np.ndarray[int]:
        # Side drop-off to pickerwall: AGV is at a highway entry cell adjacent to a goal.
        goal_xy = self._agv_entry_to_goal.get((agent.x, agent.y))
        if goal_xy is not None:
            gx, gy = goal_xy
            if self.grid[CollisionLayers.SHELVES, gy, gx] != 0:
                # Goal slot still occupied; can't deposit yet
                agent.busy = False
                return rewards
            return self._deliver_to_pickerwall(agent, gx, gy, rewards)

        # Side deposit to rack: AGV is on a highway cell adjacent to its target shelf slot.
        target_rc = self.action_id_to_coords_map.get(agent.target)
        if target_rc is not None:
            tx, ty = target_rc[1], target_rc[0]   # (col, row) of the rack slot
            if (abs(tx - agent.x) + abs(ty - agent.y) == 1
                    and not self._is_highway(tx, ty)
                    and (tx, ty) not in self.goals):
                if self.grid[CollisionLayers.SHELVES, ty, tx] != 0:
                    # Rack slot occupied; can't deposit
                    agent.busy = False
                    return rewards
                return self._deposit_to_shelf_cell(agent, tx, ty, rewards)

        # Legacy / fallback: agent is at the target cell itself.

        # Can't deposit if another shelf is already sitting at this cell
        if self.grid[CollisionLayers.SHELVES, agent.y, agent.x] != 0:
            agent.busy = False
            return rewards

        # Replenishment zone is take-only - AGVs may never deposit here
        if (agent.x, agent.y) in self._replenishment_locs_set:
            agent.busy = False
            return rewards

        if (agent.x, agent.y) in self.goals:
            return self._deliver_to_pickerwall(agent, agent.x, agent.y, rewards)

        if not self._is_highway(agent.x, agent.y):
            return self._deposit_to_shelf_cell(agent, agent.x, agent.y, rewards)

        return rewards

    def execute_micro_actions(self, rewards: np.ndarray[int]) -> np.ndarray[int]:
        for agent in self.agents:
            if agent.req_action == Action.FORWARD:
                self._execute_forward(agent)
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                self._execute_rotation(agent)
            elif agent.req_action == Action.TOGGLE_LOAD:
                if not agent.carrying_shelf:
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

    def _initialize_bin_backed_shelf_inventory(self) -> None:
        """Bridge Stage A2 bin inventory into legacy Shelf objects.

        Logical bins hold the volume-aware SKU quantities. Until AGV tasks move
        bins directly, each storage-cell Shelf inherits SKU/capacity from the
        first stocked bin in that cell so existing order and movement code can
        continue to run.
        """
        if self.order_sequencer is None:
            return

        self.order_sequencer.initialize_bin_sku_map(self.storage_logical_bins)
        self.order_sequencer._sku_to_shelves = {}

        for shelf in self.shelfs:
            cell = self.bin_cells_by_xy.get((shelf.x, shelf.y))
            stocked_bin = None
            if cell is not None:
                stocked_bin = next(
                    (bin_ for bin_ in cell.bins if bin_.sku is not None and bin_.quantity > 0),
                    None,
                )

            if stocked_bin is None:
                shelf.sku = None
                shelf.capacity = 0
                shelf.initial_capacity = 0
                shelf.bin_id = None
                continue

            shelf.sku = stocked_bin.sku
            shelf.capacity = stocked_bin.quantity
            shelf.initial_capacity = stocked_bin.quantity
            shelf.bin_id = stocked_bin.id
            self.order_sequencer._sku_to_shelves.setdefault(stocked_bin.sku, []).append(shelf)

        logger.info(
            "Stage A2 shelf bridge initialised: shelves=%d storage_bins=%d skus_with_shelf=%d",
            len(self.shelfs),
            len(self.storage_logical_bins),
            len(self.order_sequencer._sku_to_shelves),
        )

    def _bin_quantity_for_sku(self, sku: int) -> int:
        """Return how many units of a SKU fit in one usable bin."""
        if self.order_sequencer is None:
            return Shelf.DEFAULT_CAPACITY
        unit_cube = self.order_sequencer.get_sku_unit_cube(sku)
        if unit_cube <= 0:
            return 0
        usable_volume = self.bin_volume_ft3 * self.bin_usable_fraction
        return max(0, int(math.floor(usable_volume / unit_cube)))

    def _logical_bin_for_shelf(self, shelf: "Shelf") -> Optional[LogicalBin]:
        if shelf.bin_id is None:
            return None
        return self.logical_bins_by_id.get(shelf.bin_id)

    def _assign_sku_to_bin(self, bin_: LogicalBin, sku: int) -> int:
        quantity = self._bin_quantity_for_sku(sku)
        unit_cube = self.order_sequencer.get_sku_unit_cube(sku) if self.order_sequencer else 0.0
        bin_.sku = sku
        bin_.quantity = quantity
        bin_.used_volume_ft3 = quantity * unit_cube
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
        self.order_sequencer._sku_to_bins[bin_.sku] = [
            candidate
            for candidate in bins_for_sku
            if candidate.id != bin_.id
        ]

    def _remove_shelf_from_sku_lookup(self, shelf: "Shelf") -> None:
        if self.order_sequencer is None or shelf.sku is None:
            return
        shelves_for_sku = self.order_sequencer._sku_to_shelves.get(shelf.sku)
        if not shelves_for_sku:
            return
        self.order_sequencer._sku_to_shelves[shelf.sku] = [
            candidate
            for candidate in shelves_for_sku
            if candidate.id != shelf.id
        ]

    def _reserve_cell_slot_for_shelf(
        self,
        shelf: "Shelf",
        x: int,
        y: int,
        cell_type: BinCellType,
    ) -> bool:
        """Reserve one logical slot inside a pickerwall/replenishment cell."""
        cell = self.bin_cells_by_xy.get((x, y))
        if cell is None or cell.cell_type != cell_type:
            return False
        if shelf.bin_id is None:
            return False

        for slot in cell.bins:
            if slot.id not in self._slot_occupancy_by_slot_id:
                self._slot_occupancy_by_slot_id[slot.id] = shelf.bin_id
                if cell_type == BinCellType.PICKERWALL:
                    self._pickerwall_slot_by_shelf_id[shelf.id] = slot.id
                elif cell_type == BinCellType.REPLENISHMENT:
                    self._replenishment_slot_by_shelf_id[shelf.id] = slot.id
                return True
        return False

    def _release_reserved_slot_for_shelf(self, shelf: "Shelf") -> None:
        slot_id = self._pickerwall_slot_by_shelf_id.pop(shelf.id, None)
        if slot_id is None:
            slot_id = self._replenishment_slot_by_shelf_id.pop(shelf.id, None)
        if slot_id is not None:
            self._slot_occupancy_by_slot_id.pop(slot_id, None)

    def _decrement_shelf_bin_inventory(self, shelf: "Shelf", quantity: int) -> int:
        """Pick units from the logical bin backing a movable Shelf wrapper."""
        bin_ = self._logical_bin_for_shelf(shelf)
        if bin_ is None:
            shelf.capacity = max(0, shelf.capacity - quantity)
            if shelf.capacity <= 0:
                self._remove_shelf_from_sku_lookup(shelf)
            return shelf.capacity

        bin_.quantity = max(0, bin_.quantity - quantity)
        unit_cube = self.order_sequencer.get_sku_unit_cube(bin_.sku) if (
            self.order_sequencer is not None and bin_.sku is not None
        ) else 0.0
        bin_.used_volume_ft3 = bin_.quantity * unit_cube
        shelf.capacity = bin_.quantity
        if bin_.quantity <= 0:
            self._remove_bin_from_sku_lookup(bin_)
            self._remove_shelf_from_sku_lookup(shelf)
        return shelf.capacity

    def reset(self, seed=None, options=None)-> Tuple:
        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0
        self._step_deliveries = 0
        self.seed(seed)
        self._shelf_to_order = {}
        self._packaging_slots = {}
        self._pickerwall_pending = deque()
        self._pickerwall_slot_by_shelf_id = {}
        self._replenishment_slot_by_shelf_id = {}
        self._slot_occupancy_by_slot_id = {}
        self._picker_hf_profile_by_id = {}
        self._picker_hf_state_by_id = {}
        self._picker_hf_episode_delay_steps = 0
        self._picker_hf_episode_failed_pick_delays = 0

        self.shelfs = [Shelf(x, y) for (y, x) in self.shelf_locs]

        agent_loc_ids = np.random.choice(len(self.agv_spawn_locs), size=self.num_agents, replace=False)
        agent_locs = [self.agv_spawn_locs[agent_loc_ids, 0], self.agv_spawn_locs[agent_loc_ids, 1]]
        agent_dirs = np.random.choice([d for d in Direction], size=self.num_agents)
        self.agents = [
            Agent(x, y, dir_, agent_type=agent_type)
            for y, x, dir_, agent_type in zip(*agent_locs, agent_dirs, self._agent_types)
        ]

        self.stuck_counters = [StuckCounter((agent.x, agent.y)) for agent in self.agents]

        Picker.counter = 0
        self.pickers = []
        if self.num_pickers > 0 and len(self.picker_spawn_locs) > 0:
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
            self._initialize_bin_backed_shelf_inventory()
            released = self.order_sequencer.release_pending_orders(0)
            logger.info("reset: released %d orders at t=0 (queue capacity=%d)", len(released), self.request_queue_size)
            self.request_queue = []
            for _ in range(self.request_queue_size):
                carried = {a.carrying_shelf for a in self.agents if a.carrying_shelf}
                candidates = list(set(self.shelfs) - set(self.request_queue) - carried)
                result = self.order_sequencer.next_order_shelf(candidates)
                if result is not None:
                    shelf, order = result
                    self._shelf_to_order[shelf.id] = order
                    self.request_queue.append(shelf)
            logger.info(
                "reset: initial request_queue filled=%d/%d skus=%s",
                len(self.request_queue), self.request_queue_size,
                [s.sku for s in self.request_queue],
            )
        else:
            logger.info(
                "reset: no order_sequencer - filling request queue with %d random shelves",
                self.request_queue_size,
            )
            self.request_queue = list(
                np.random.choice(self.shelfs, size=self.request_queue_size, replace=False)
            )

        self.observation_space_mapper.extract_environment_info(self)
        return tuple([self.observation_space_mapper.observation(agent) for agent in self.agents])

    def _refill_request_queue(self) -> None:
        """Fill the request queue from active orders up to request_queue_size."""
        if self.order_sequencer is None:
            return
        carried = {a.carrying_shelf for a in self.agents if a.carrying_shelf}
        pickerwall_shelf_ids = {
            self.grid[CollisionLayers.SHELVES, yy, xx]
            for (xx, yy) in self.goals
            if self.grid[CollisionLayers.SHELVES, yy, xx] != 0
        }
        pickerwall_shelves = {self.shelfs[sid - 1] for sid in pickerwall_shelf_ids}
        while len(self.request_queue) < self.request_queue_size:
            candidates = [
                s for s in self.shelfs
                if s.on_grid
                and s not in self.request_queue
                and s not in carried
                and s not in pickerwall_shelves
            ]
            result = self.order_sequencer.next_order_shelf(candidates)
            if result is None:
                logger.debug(
                    "step=%d _refill: active queue empty or no matching shelf - "
                    "queue stays at %d/%d",
                    self._cur_steps, len(self.request_queue), self.request_queue_size,
                )
                break
            shelf, order = result
            self._shelf_to_order[shelf.id] = order
            logger.info(
                "step=%d _refill: added shelf_id=%d sku=%d to request_queue "
                "(queue=%d/%d pending=%d active=%d)",
                self._cur_steps, shelf.id, shelf.sku,
                len(self.request_queue) + 1, self.request_queue_size,
                self.order_sequencer.pending_count, self.order_sequencer.active_count,
            )
            self.request_queue.append(shelf)

    def _issue_replenishment(self, sku: int) -> Optional["Shelf"]:
        """Spawn a fresh shelf with the given SKU at a free replenishment slot. Returns None if no slot is free."""
        if not self.replenishment_locs:
            return None

        free_slot = None
        for (rx, ry) in self.replenishment_locs:
            if (self.grid[CollisionLayers.SHELVES, ry, rx] == 0
                    and self.grid[CollisionLayers.CARRIED_SHELVES, ry, rx] == 0):
                free_slot = (rx, ry)
                break

        if free_slot is None:
            logger.warning(
                "step=%d _issue_replenishment: sku=%d - no free replenishment slot available",
                self._cur_steps, sku,
            )
            return None

        rx, ry = free_slot
        new_shelf = Shelf(rx, ry)
        new_shelf.sku = sku
        new_shelf.from_replenishment = True
        replenishment_bin = next(
            (
                bin_
                for bin_ in self.replenishment_logical_bins
                if (bin_.x, bin_.y) == (rx, ry)
                and (bin_.sku is None or bin_.quantity <= 0)
            ),
            None,
        )
        if replenishment_bin is not None:
            replenishment_quantity = self._assign_sku_to_bin(replenishment_bin, sku)
            new_shelf.bin_id = replenishment_bin.id
        else:
            replenishment_quantity = self._bin_quantity_for_sku(sku)
        new_shelf.capacity = replenishment_quantity
        new_shelf.initial_capacity = replenishment_quantity
        if new_shelf.bin_id is not None:
            self._reserve_cell_slot_for_shelf(
                new_shelf,
                rx,
                ry,
                BinCellType.REPLENISHMENT,
            )
        self.shelfs.append(new_shelf)
        self.grid[CollisionLayers.SHELVES, ry, rx] = new_shelf.id
        if self.order_sequencer is not None:
            self.order_sequencer._sku_to_shelves.setdefault(sku, []).append(new_shelf)

        logger.info(
            "step=%d replenishment: new shelf_id=%d sku=%d spawned at (%d,%d) "
            "(total_shelves=%d)",
            self._cur_steps, new_shelf.id, sku, rx, ry, len(self.shelfs),
        )

        self._refill_request_queue()
        return new_shelf

    def _resolve_pickerwall_shelf_for_sku(self, sku: int) -> int:
        """Return the shelf_id of a shelf sitting at the pickerwall with the given SKU.

        Returns -1 if no matching shelf is currently at a pickerwall slot.
        """
        for (x, y) in self.goals:
            shelf_id = self.grid[CollisionLayers.SHELVES, y, x]
            if shelf_id != 0 and self.shelfs[shelf_id - 1].sku == sku:
                return shelf_id
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

    def _shelf_pickerwall_zone(self, shelf_id: int) -> Optional[int]:
        shelf = self.shelfs[shelf_id - 1]
        if (shelf.x, shelf.y) not in self.goals:
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
            shelf_id, sku_entry, order = self._pickerwall_pending.popleft()
            shelf = self.shelfs[shelf_id - 1]

            if (shelf.x, shelf.y) not in self.goals or \
                    self.grid[CollisionLayers.SHELVES, shelf.y, shelf.x] != shelf_id:
                logger.debug(
                    "step=%d _claim: skipping stale pickerwall_pending shelf_id=%d",
                    self._cur_steps, shelf_id,
                )
                continue

            if zone_id is None or self._shelf_pickerwall_zone(shelf_id) == zone_id:
                selected = (shelf_id, sku_entry, order)
                break
            skipped.append((shelf_id, sku_entry, order))

        while skipped:
            self._pickerwall_pending.appendleft(skipped.pop())
        return selected

    def _append_claim_for_entry(
        self,
        claims: List["PickerClaim"],
        entry: Tuple[int, SKUEntry, Order],
        remaining_cap: int,
    ) -> int:
        shelf_id, sku_entry, order = entry
        if sku_entry.quantity <= remaining_cap:
            claims.append(PickerClaim(
                shelf_id=shelf_id,
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
            shelf_id=shelf_id,
            sku_entry=claim_entry,
            order_number=order.order_number,
            order=order,
        ))
        self._pickerwall_pending.appendleft((shelf_id, remaining_entry, order))
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
                self._packaging_slots[claim.order_number] = {
                    "required": total_qty,
                    "delivered": 0,
                    "station": None,
                }
        return claims

    def _build_picker_task(self, picker: "Picker", claims: List["PickerClaim"]) -> "PickerTask":
        """Preserve policy claim order while grouping same-shelf work."""
        shelf_groups: Dict[int, List[PickerClaim]] = {}
        shelf_order: List[int] = []
        unresolved: List[PickerClaim] = []
        for claim in claims:
            if claim.shelf_id == -1:
                unresolved.append(claim)
                continue
            if claim.shelf_id not in shelf_groups:
                shelf_groups[claim.shelf_id] = []
                shelf_order.append(claim.shelf_id)
            shelf_groups[claim.shelf_id].append(claim)

        ordered_claims: List[PickerClaim] = []
        for shelf_id in shelf_order:
            ordered_claims.extend(shelf_groups[shelf_id])
        ordered_claims.extend(unresolved)
        return PickerTask(claims=ordered_claims)

    def _maybe_mark_shelf_fulfilled(self, shelf: "Shelf") -> None:
        """Mark shelf.fulfilled = True if no pending or in-flight picks remain for it."""
        sid = shelf.id
        for entry_shelf_id, _sku_entry, _order in self._pickerwall_pending:
            if entry_shelf_id == sid:
                shelf.fulfilled = False
                return
        for picker in self.pickers:
            if picker.task is None:
                continue
            for claim in picker.task.claims:
                if not claim.picked and claim.shelf_id == sid:
                    shelf.fulfilled = False
                    return
        shelf.fulfilled = True
        logger.info(
            "step=%d: shelf_id=%d sku=%d marked fulfilled - eligible for displacement",
            self._cur_steps, shelf.id, shelf.sku,
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
            (picker.y, picker.x), pkg_target, picker, care_for_agents=True
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
        """Route the picker to the next unpicked shelf, or to packaging if all claims are done."""
        task = picker.task

        # Skip claims that have already been picked
        while (task.current_claim_index < len(task.claims)
               and task.claims[task.current_claim_index].picked):
            task.current_claim_index += 1

        if task.current_claim_index >= len(task.claims):
            self._start_walk_to_packaging(picker)
            return

        claim = task.claims[task.current_claim_index]

        # Resolve shelf_id=-1 (shelf wasn't at pickerwall when claim was made)
        if claim.shelf_id == -1:
            resolved_id = self._resolve_pickerwall_shelf_for_sku(claim.sku_entry.sku)
            if resolved_id == -1:
                picker.state = PickerState.WAITING_FOR_SHELF
                return
            # Propagate resolved shelf_id to all unpicked claims for this SKU
            for c in task.claims:
                if not c.picked and c.sku_entry.sku == claim.sku_entry.sku:
                    c.shelf_id = resolved_id
            claim.shelf_id = resolved_id

        shelf = self.shelfs[claim.shelf_id - 1]
        goal_xy = (shelf.x, shelf.y)

        # Verify the shelf is still sitting at a pickerwall slot
        if goal_xy not in self.goals or self.grid[CollisionLayers.SHELVES, shelf.y, shelf.x] != claim.shelf_id:
            logger.info(
                "step=%d picker_id=%d: shelf_id=%d (sku=%d) not at pickerwall - WAITING_FOR_SHELF",
                self._cur_steps, picker.id, claim.shelf_id, claim.sku_entry.sku,
            )
            picker.state = PickerState.WAITING_FOR_SHELF
            return

        entries = self._goal_to_picker_entry.get(goal_xy, [])
        if not entries:
            logger.warning(
                "step=%d picker_id=%d: no picker entry adjacent to goal %s for shelf_id=%d - skipping",
                self._cur_steps, picker.id, goal_xy, claim.shelf_id,
            )
            # Force-skip all claims for this shelf
            for c in task.claims:
                if c.shelf_id == claim.shelf_id and not c.picked:
                    c.picked = True
            self._picker_walk_to_next_shelf(picker)
            return

        entry_xy = min(entries, key=lambda e: abs(e[0] - picker.x) + abs(e[1] - picker.y))
        target = (entry_xy[1], entry_xy[0])  # (row, col) for find_picker_path
        picker.path = self.find_picker_path(
            (picker.y, picker.x), target, picker, care_for_agents=True
        )
        picker.state = PickerState.WALKING_TO_SHELF
        logger.info(
            "step=%d picker_id=%d: ->WALKING_TO_SHELF shelf_id=%d sku=%d entry=%s path_len=%d",
            self._cur_steps, picker.id, claim.shelf_id, claim.sku_entry.sku,
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

        current_shelf_id = picker.task.claims[picker.task.current_claim_index].shelf_id
        unit_cube = 0.0
        quantity = 0
        for claim in picker.task.claims:
            if claim.picked or claim.shelf_id != current_shelf_id:
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

    def _advance_pickers(self) -> None:
        """Drive all picker agents through their state machine each step."""
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
                                    (picker.y, picker.x), (dest_row, dest_col), picker, care_for_agents=True
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
                        "step=%d picker_id=%d: arrived -> PICKING shelf_id=%d sku=%d (pick_ticks=%d)",
                        self._cur_steps, picker.id, claim.shelf_id, claim.sku_entry.sku,
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
                    current_shelf_id = picker.task.claims[picker.task.current_claim_index].shelf_id
                    shelf = self.shelfs[current_shelf_id - 1]
                    # Pick every unpicked claim for this shelf in a single visit
                    for claim in picker.task.claims:
                        if not claim.picked and claim.shelf_id == current_shelf_id:
                            claim.picked = True
                            remaining_stock = self._decrement_shelf_bin_inventory(
                                shelf,
                                claim.sku_entry.quantity,
                            )
                            logger.info(
                                "step=%d picker_id=%d: picked %d unit(s) from shelf_id=%d "
                                "bin_id=%s sku=%d for order=%s (stock_remaining=%d)",
                                self._cur_steps, picker.id, claim.sku_entry.quantity,
                                shelf.id, shelf.bin_id, shelf.sku, claim.order_number,
                                remaining_stock,
                            )
                    if shelf.capacity <= 0 and not shelf.depleted:
                        shelf.depleted = True
                        logger.info(
                            "step=%d: shelf_id=%d sku=%d stock exhausted - issuing replenishment",
                            self._cur_steps, shelf.id, shelf.sku,
                        )
                        self._issue_replenishment(shelf.sku)
                    self._maybe_mark_shelf_fulfilled(shelf)
                    self._picker_walk_to_next_shelf(picker)

            # WALKING_TO_PACKAGING
            elif picker.state == PickerState.WALKING_TO_PACKAGING:
                if picker.path:
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
                                    (picker.y, picker.x), (dest_row, dest_col), picker, care_for_agents=True
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
        if self.num_pickers > 0:
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
        info["human_factors_model"] = self.human_factors_config.model_name

        if self.pickers:
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
        carrying_shelf_info = self.get_carrying_shelf_information()
        pickerwall_occupied = self.get_pickerwall_info()
        pickerwall_displaceable = self.get_pickerwall_displacement_info()
        targets_agvs = [target - len(self.goals) - 1 for target in self.targets_agvs if target > len(self.goals)]

        valid_location_list_agvs = np.array([
            empty_items if carrying_shelf else requested_items for carrying_shelf in carrying_shelf_info
        ])
        valid_goal_list_agvs = np.array([
            (1 - pickerwall_occupied) if carrying_shelf else pickerwall_displaceable
            for carrying_shelf in carrying_shelf_info
        ])

        if block_conflicting_actions:
            valid_location_list_agvs[:, targets_agvs] = 0

        valid_action_masks = np.ones((self.num_agents, self.action_size))
        valid_action_masks[:, 1 + len(self.goals):] = valid_location_list_agvs
        valid_action_masks[:, 1 : 1 + len(self.goals)] = valid_goal_list_agvs
        return valid_action_masks

    def render(self, mode="human"):
        if not self.renderer:
            from tarware.rendering import Viewer
            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
