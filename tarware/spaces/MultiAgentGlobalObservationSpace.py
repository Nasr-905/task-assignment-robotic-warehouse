import numpy as np
from gymnasium import spaces

from tarware.definitions import Action, CollisionLayers
from tarware.spaces.MultiAgentBaseObservationSpace import (
    MultiAgentBaseObservationSpace, _VectorWriter)


class MultiAgentGlobalObservationSpace(MultiAgentBaseObservationSpace):
    def __init__(self, num_agvs, grid_size, shelf_locations, num_goals=0, normalised_coordinates=False):
        # Pass 0 for num_pickers to the base class (pickers are removed)
        super(MultiAgentGlobalObservationSpace, self).__init__(num_agvs, 0, grid_size, shelf_locations, normalised_coordinates)
        self.num_goals = num_goals

        self._define_obs_length()
        self.obs_lengths = [self.obs_length for _ in range(self.num_agents)]
        self._current_agents_info = []
        self._current_shelves_info = []

        ma_spaces = []
        for obs_length in self.obs_lengths:
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(obs_length,),
                    dtype=np.float32,
                )
            ]

        self.ma_spaces = spaces.Tuple(tuple(ma_spaces))

    def _define_obs_length(self):
        # Per agent: [carrying, shelf_requested, toggle_load, pos_x, pos_y, target_x, target_y] = 7 values
        self.obs_bits_per_agent = 7
        self.obs_bits_for_agents = self.obs_bits_per_agent * self.num_agvs
        # Per rack shelf location: [occupied, requested] = 2 values
        self.obs_bits_for_shelves = 2 * self.shelf_locations
        # Per pickerwall slot: [occupied, shelf_is_requested] = 2 values
        self.obs_bits_for_pickerwall = 2 * self.num_goals
        self.obs_length = (
            self.obs_bits_for_agents
            + self.obs_bits_for_shelves
            + self.obs_bits_for_pickerwall
        )

    def extract_environment_info(self, environment):
        self._current_agents_info = []
        self._current_shelves_info = []

        # Extract per-agent info
        for agent in environment.agents:
            agent_info = []
            if agent.carrying_shelf:
                agent_info.extend([1, int(agent.carrying_shelf in environment.request_queue)])
            else:
                agent_info.extend([0, 0])
            agent_info.append(int(agent.req_action == Action.TOGGLE_LOAD))
            agent_info.extend(self.process_coordinates((agent.y, agent.x), environment))
            if agent.target:
                agent_info.extend(self.process_coordinates(environment.action_id_to_coords_map[agent.target], environment))
            else:
                agent_info.extend([0, 0])
            self._current_agents_info.append(agent_info)

        # Extract rack shelves info: [occupied, requested] per rack location
        for group in environment.rack_groups:
            for (x, y) in group:
                shelf_id = environment.grid[CollisionLayers.SHELVES, x, y]
                if shelf_id != 0:
                    self._current_shelves_info.extend([1.0, int(environment.shelfs[shelf_id - 1] in environment.request_queue)])
                else:
                    self._current_shelves_info.extend([0.0, 0.0])

        # Extract pickerwall slot info: [occupied, shelf_is_requested] per goal slot
        for (x, y) in environment.goals:
            shelf_id = environment.grid[CollisionLayers.SHELVES, y, x]
            if shelf_id != 0:
                self._current_shelves_info.extend([1.0, int(environment.shelfs[shelf_id - 1] in environment.request_queue)])
            else:
                self._current_shelves_info.extend([0.0, 0.0])

    def observation(self, agent):
        obs = _VectorWriter(self.ma_spaces[agent.id - 1].shape[0])
        obs.write(self._current_agents_info[agent.id - 1])
        for agent_id, agent_info in enumerate(self._current_agents_info):
            if agent_id != agent.id - 1:
                obs.write(agent_info)
        obs.write(self._current_shelves_info)
        return obs.vector
