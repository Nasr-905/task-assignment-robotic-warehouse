import logging
from collections import deque
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

logger = logging.getLogger(__name__)

_FIXING_CLASH_TIME = 4
_STUCK_THRESHOLD = 5
_PICK_TICKS = 3  # Steps a picker spends picking from a shelf
_AGV_WALKABLE_TILES = {0, 2, 5, 6}
_PICKER_WALKABLE_TILES = {3, 4, 6}
_SHARED_HIGHWAY_TILE = 6

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

    Claims are ordered so that consecutive entries with the same ``shelf_id``
    are serviced in a single pickerwall visit.  ``current_claim_index`` tracks
    which claim the picker is currently working towards.
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
        self._make_order_sequencer_from_csv(order_csv_path, steps_per_simulated_second)
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
        self._packaging_slots: Dict[str, Dict[str, int]] = {}  # order_number -> {required, delivered}
        self._pickerwall_pending: deque = deque()  # (shelf_id, SKUEntry, Order) populated on delivery

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

        # Replenishment cells: staging zone for fresh stock (AGV take-only)
        self.replenishment_locs: List[Tuple[int, int]] = [
            (c, r)
            for r in range(num_rows)
            for c in range(num_cols)
            if tile_grid[r, c] == 5
        ]
        self._replenishment_locs_set: set = set(self.replenishment_locs)

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
            "goals=%d shelves=%d replenishment=%d packaging=%d column_height=%d",
            map_csv_path, self.grid_size, agv_zone_height, picker_zone_rows,
            self.num_goals, len(self.shelf_locs), len(self.replenishment_locs),
            len(self.packaging_locations), self.column_height,
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
                    agent.path = self.find_agv_path((agent.y, agent.x), self.action_id_to_coords_map[macro_action], agent, care_for_agents=True)
                    if agent.path:
                        agent.busy = True
                        agent.target = macro_action
                        agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                        self.stuck_counters[agent.id - 1].reset((agent.x, agent.y))
            else:
                if agent.path == []:
                    agent.req_action = Action.TOGGLE_LOAD
                else:
                    agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                    agvs_distance_travelled += 1
                if len(agent.path) == 1:
                    # If carrying and the target cell is already occupied by a resting shelf, abort
                    if agent.carrying_shelf and self.grid[CollisionLayers.SHELVES, agent.path[-1][1], agent.path[-1][0]]:
                        # target cell occupied by a resting shelf - abort
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
                            if (other_new_x, other_new_y) in [(agent.x, agent.y), (agent_new_x, agent_new_y)] and not other.req_action in (Action.LEFT, Action.RIGHT):
                                if other.fixing_clash == 0:
                                    clashes+=1
                                    agent.fixing_clash = _FIXING_CLASH_TIME
                                    new_path = self.find_agv_path((agent.y, agent.x), (agent.path[-1][1] ,agent.path[-1][0]), agent)
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
            and (agent.req_action!=Action.TOGGLE_LOAD or (agent.x, agent.y) in self.goals) # Don't count loading or changing directions / if at goal
        ]
        for agent in moving_agents:
            agent_stuck_count = self.stuck_counters[agent.id - 1]
            agent_stuck_count.update((agent.x, agent.y))
            if _STUCK_THRESHOLD < agent_stuck_count.count < _STUCK_THRESHOLD + self.column_height + 2:  # Time to get out of aisle
                agent.req_action = Action.NOOP
                if agent.path:
                    new_path = self.find_agv_path((agent.y, agent.x), (agent.path[-1][1], agent.path[-1][0]), agent)
                    if new_path:
                        agent.path = new_path
                        if len(agent.path) == 1:
                            continue
                        agent_stuck_count.reset((agent.x, agent.y))
                        continue
                else:
                    overall_stucks += 1
                    agent.busy = False
                    agent_stuck_count.reset()
            if agent_stuck_count.count > _STUCK_THRESHOLD + self.column_height + 2:  # Time to get out of aisle
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
        shelf_id = self.grid[CollisionLayers.SHELVES, agent.y, agent.x]
        if shelf_id:
            agent.carrying_shelf = self.shelfs[shelf_id - 1]
            self.grid[CollisionLayers.SHELVES, agent.y, agent.x] = 0
            self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = shelf_id
            agent.busy = False
            if self.reward_type == RewardType.GLOBAL:
                rewards += 0.5
            elif self.reward_type == RewardType.INDIVIDUAL:
                rewards[agent.id - 1] += 0.1
        else:
            agent.busy = False
        return rewards

    def _execute_unload(self, agent: Agent, rewards: np.ndarray[int]) -> np.ndarray[int]:
        # Can't deposit if another shelf is already sitting at this cell
        if self.grid[CollisionLayers.SHELVES, agent.y, agent.x] != 0:
            agent.busy = False
            return rewards

        # Replenishment zone is take-only - AGVs may never deposit here
        if (agent.x, agent.y) in self._replenishment_locs_set:
            agent.busy = False
            return rewards

        if (agent.x, agent.y) in self.goals:
            # Deposit shelf at pickerwall slot
            shelf = agent.carrying_shelf
            self.grid[CollisionLayers.SHELVES, agent.y, agent.x] = shelf.id
            self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = 0
            shelf.x, shelf.y = agent.x, agent.y
            agent.carrying_shelf = None
            agent.busy = False
            agent.has_delivered = False
            if shelf in self.request_queue:
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
                        self._cur_steps,
                        shelf.id,
                        shelf.sku,
                        new_request.id,
                        new_request.sku,
                        self.order_sequencer.pending_count,
                        self.order_sequencer.active_count,
                    )
                else:
                    new_request = None
                    logger.info(
                        "step=%d delivery: shelf_id=%d sku=%s delivered - "
                        "no active orders available, queue shrinks to %d "
                        "(pending=%d active=%d)",
                        self._cur_steps,
                        shelf.id,
                        shelf.sku,
                        len(self.request_queue) - 1,
                        self.order_sequencer.pending_count,
                        self.order_sequencer.active_count,
                    )

                if new_request is not None:
                    self.request_queue[self.request_queue.index(shelf)] = new_request
                else:
                    self.request_queue.remove(shelf)
                self._step_deliveries += 1
            return rewards

        if not self._is_highway(agent.x, agent.y):
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
                self.grid[CollisionLayers.SHELVES, agent.y, agent.x] = shelf.id
                self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = 0
                shelf.x, shelf.y = agent.x, agent.y
                agent.carrying_shelf = None
                agent.busy = False
                agent.has_delivered = False
                if self.reward_type == RewardType.GLOBAL:
                    rewards += 0.5
                elif self.reward_type == RewardType.INDIVIDUAL:
                    rewards[agent.id - 1] += 0.1
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
            picker_loc_ids = np.random.choice(len(self.picker_spawn_locs), size=self.num_pickers, replace=False)
            picker_dirs = np.random.choice([d for d in Direction], size=self.num_pickers)
            for loc_id, dir_ in zip(picker_loc_ids, picker_dirs):
                py, px = self.picker_spawn_locs[loc_id]
                picker = Picker(int(px), int(py), dir_)
                picker.state = PickerState.IDLE
                picker.task = None
                picker.path = []
                picker.pick_ticks_remaining = 0
                self.pickers.append(picker)
            logger.info("reset: spawned %d picker(s) across %d picker-highway cells",
                        self.num_pickers, len(self.picker_spawn_locs))

        self._recalc_grid()

        if self.order_sequencer is not None:
            logger.info(
                "reset: order_sequencer present - using time-gated SKU-based request queue"
            )
            self.order_sequencer.reset()
            self.order_sequencer.initialize_shelf_sku_map(self.shelfs)
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
        new_shelf.capacity = Shelf.DEFAULT_CAPACITY
        new_shelf.initial_capacity = Shelf.DEFAULT_CAPACITY
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

    def _claim_items_for_picker(self, picker: "Picker") -> List["PickerClaim"]:
        """Claim up to picker.capacity item-units from _pickerwall_pending. Splits oversized entries in-place."""
        if not self._pickerwall_pending:
            return []

        claims: List[PickerClaim] = []
        remaining_cap = picker.capacity

        while self._pickerwall_pending and remaining_cap > 0:
            shelf_id, sku_entry, order = self._pickerwall_pending[0]
            shelf = self.shelfs[shelf_id - 1]

            # Skip stale entries (shelf moved away from pickerwall)
            if (shelf.x, shelf.y) not in self.goals or \
                    self.grid[CollisionLayers.SHELVES, shelf.y, shelf.x] != shelf_id:
                self._pickerwall_pending.popleft()
                logger.debug(
                    "step=%d _claim: skipping stale pickerwall_pending shelf_id=%d",
                    self._cur_steps, shelf_id,
                )
                continue

            self._pickerwall_pending.popleft()
            if sku_entry.quantity <= remaining_cap:
                claims.append(PickerClaim(
                    shelf_id=shelf_id,
                    sku_entry=sku_entry,
                    order_number=order.order_number,
                    order=order,
                ))
                remaining_cap -= sku_entry.quantity
            else:
                # Split: claim what fits, put the remainder back at the front
                claim_entry = SKUEntry(sku=sku_entry.sku, quantity=remaining_cap)
                remaining_entry = SKUEntry(sku=sku_entry.sku,
                                           quantity=sku_entry.quantity - remaining_cap)
                claims.append(PickerClaim(
                    shelf_id=shelf_id,
                    sku_entry=claim_entry,
                    order_number=order.order_number,
                    order=order,
                ))
                self._pickerwall_pending.appendleft((shelf_id, remaining_entry, order))
                remaining_cap = 0

        for claim in claims:
            if claim.order_number not in self._packaging_slots:
                total_qty = sum(se.quantity for se in claim.order.skus)
                self._packaging_slots[claim.order_number] = {
                    "required": total_qty,
                    "delivered": 0,
                }
        return claims

    def _build_picker_task(self, picker: "Picker", claims: List["PickerClaim"]) -> "PickerTask":
        """Group claims by SKU/shelf, ordered by greedy nearest-next proximity."""
        sku_groups: Dict[int, List[PickerClaim]] = {}
        for c in claims:
            sku_groups.setdefault(c.sku_entry.sku, []).append(c)

        resolved = [(grp[0].shelf_id, grp) for grp in sku_groups.values() if grp[0].shelf_id != -1]
        unresolved = [(grp[0].shelf_id, grp) for grp in sku_groups.values() if grp[0].shelf_id == -1]

        ordered_claims: List[PickerClaim] = []
        remaining = list(resolved)
        cx, cy = picker.x, picker.y
        while remaining:
            dists = [
                abs(self.shelfs[sid - 1].x - cx) + abs(self.shelfs[sid - 1].y - cy)
                for sid, _ in remaining
            ]
            best_idx = int(np.argmin(dists))
            sid, grp = remaining.pop(best_idx)
            ordered_claims.extend(grp)
            s = self.shelfs[sid - 1]
            cx, cy = s.x, s.y

        for _, grp in unresolved:
            ordered_claims.extend(grp)

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

    def _start_walk_to_packaging(self, picker: "Picker") -> None:
        """Route the picker to the closest packaging station."""
        closest_pkg = min(
            self.packaging_locations,
            key=lambda loc: abs(loc[0] - picker.x) + abs(loc[1] - picker.y),
        )
        pkg_target = (closest_pkg[1], closest_pkg[0])  # (row, col) for find_picker_path
        picker.path = self.find_picker_path(
            (picker.y, picker.x), pkg_target, picker, care_for_agents=True
        )
        picker.state = PickerState.WALKING_TO_PACKAGING
        orders = list({c.order_number for c in picker.task.claims})
        logger.info(
            "step=%d picker_id=%d: all claims picked -> WALKING_TO_PACKAGING "
            "order(s)=%s pkg=(col=%d row=%d) path_len=%d",
            self._cur_steps, picker.id, orders,
            closest_pkg[0], closest_pkg[1], len(picker.path),
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

    def _advance_pickers(self) -> None:
        """Drive all picker agents through their state machine each step."""
        for picker in self.pickers:

            # IDLE
            if picker.state == PickerState.IDLE:
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
                    next_xy = picker.path[0]  # (col, row) = (x, y)
                    if self._picker_next_cell_blocked(picker, next_xy):
                        continue
                    picker.x, picker.y = next_xy[0], next_xy[1]
                    picker.path = picker.path[1:]
                else:
                    picker.state = PickerState.PICKING
                    picker.pick_ticks_remaining = _PICK_TICKS
                    claim = picker.task.claims[picker.task.current_claim_index]
                    logger.info(
                        "step=%d picker_id=%d: arrived -> PICKING shelf_id=%d sku=%d (pick_ticks=%d)",
                        self._cur_steps, picker.id, claim.shelf_id, claim.sku_entry.sku, _PICK_TICKS,
                    )

            # PICKING
            elif picker.state == PickerState.PICKING:
                picker.pick_ticks_remaining -= 1
                if picker.pick_ticks_remaining <= 0:
                    current_shelf_id = picker.task.claims[picker.task.current_claim_index].shelf_id
                    shelf = self.shelfs[current_shelf_id - 1]
                    # Pick every unpicked claim for this shelf in a single visit
                    for claim in picker.task.claims:
                        if not claim.picked and claim.shelf_id == current_shelf_id:
                            claim.picked = True
                            shelf.capacity = max(0, shelf.capacity - claim.sku_entry.quantity)
                            logger.info(
                                "step=%d picker_id=%d: picked %d unit(s) from shelf_id=%d "
                                "sku=%d for order=%s (stock_remaining=%d)",
                                self._cur_steps, picker.id, claim.sku_entry.quantity,
                                shelf.id, shelf.sku, claim.order_number, shelf.capacity,
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
                    next_xy = picker.path[0]
                    if self._picker_next_cell_blocked(picker, next_xy):
                        continue
                    picker.x, picker.y = next_xy[0], next_xy[1]
                    picker.path = picker.path[1:]
                else:
                    picker.state = PickerState.AT_PACKAGING
                    logger.info(
                        "step=%d picker_id=%d: arrived at packaging -> AT_PACKAGING order(s)=%s",
                        self._cur_steps, picker.id,
                        list({c.order_number for c in picker.task.claims}),
                    )

            # AT_PACKAGING
            elif picker.state == PickerState.AT_PACKAGING:
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
        info = self._build_info(
            agvs_distance_travelled,
            clashes_count,
            picker_yields_count,
            stucks_count,
            shelf_deliveries,
        )
        return new_obs, list(rewards), terminateds, terminateds, info

    def _build_info(
        self,
        agvs_distance_travelled: int,
        clashes_count: int,
        picker_yields_count: int,
        stucks_count: int,
        shelf_deliveries: int,
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
