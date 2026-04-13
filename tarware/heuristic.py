from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np

from tarware.warehouse import Agent


class MissionType(Enum):
    PICKING = 1     # travel to a location (rack or replenishment zone) and pick up the shelf
    RETURNING = 2   # carry a shelf to an empty rack slot and deposit it
    DELIVERING = 3  # carry a shelf to an empty pickerwall slot and deposit it


@dataclass
class Mission:
    mission_type: MissionType
    location_id: int
    location_x: int
    location_y: int
    assigned_time: int
    at_location: bool = False


def heuristic_episode(env, render=False, seed=None, render_start=0, render_sleep=0.0):
    # non_goal_location_ids aligns with the index ordering of get_empty_shelf_information
    non_goal_location_ids = []
    for id_, coords in env.action_id_to_coords_map.items():
        if (coords[1], coords[0]) not in env.goals:
            non_goal_location_ids.append(id_)
    non_goal_location_ids = np.array(non_goal_location_ids)
    location_map = env.action_id_to_coords_map
    _ = env.reset(seed=seed)
    done = False
    all_infos = []
    timestep = 0

    agents = env.agents
    agvs = agents
    coords_original_loc_map = {v: k for k, v in env.action_id_to_coords_map.items()}

    assigned_agvs: dict[Agent, Mission] = OrderedDict({})
    assigned_items: dict[Agent, int] = OrderedDict({})  # AGV -> shelf_id being fetched
    global_episode_return = 0
    episode_returns = np.zeros(env.num_agents)

    while not done:
        request_queue = env.request_queue
        goal_locations = env.goals  # list of (x, y) = (col, row)
        actions = {k: 0 for k in agents}

        # update at_location flags
        for agv in agvs:
            if agv in assigned_agvs:
                m = assigned_agvs[agv]
                if m.mission_type == MissionType.DELIVERING:
                    # DELIVERING: AGV stops at an entry cell adjacent to the goal.
                    goal_xy = (m.location_x, m.location_y)
                    entry_cells = env._goal_to_agv_entry.get(goal_xy, [])
                    if (agv.x, agv.y) in entry_cells:
                        assigned_agvs[agv].at_location = True
                else:
                    # PICKING / RETURNING: AGV stops adjacent to its target cell.
                    # Accept distance-1 (adjacent) or 0 (at same cell, fallback).
                    dist = abs(agv.x - m.location_x) + abs(agv.y - m.location_y)
                    if dist <= 1:
                        assigned_agvs[agv].at_location = True

        # handle completed / transitioning missions
        for agv in agvs:
            if agv not in assigned_agvs or agv.busy:
                continue

            mission = assigned_agvs[agv]

            if mission.mission_type == MissionType.PICKING and mission.at_location and agv.carrying_shelf:
                was_at_goal = (mission.location_x, mission.location_y) in goal_locations

                if was_at_goal:
                    # displacement pick-up: return shelf to an empty rack slot
                    empty_shelves = env.get_empty_shelf_information()
                    empty_location_ids = list(non_goal_location_ids[empty_shelves > 0])
                    busy_loc_ids = [m.location_id for m in assigned_agvs.values()]
                    empty_location_ids = [i for i in empty_location_ids if i not in busy_loc_ids]
                    if empty_location_ids:
                        empty_yx = [location_map[i] for i in empty_location_ids]
                        dists = [len(p) if p else float('inf')
                                 for p in (env.find_agv_path((agv.y, agv.x), yx, agv, care_for_agents=False)
                                           for yx in empty_yx)]
                        best_id = empty_location_ids[np.argmin(dists)]
                        best_yx = location_map[best_id]
                        assigned_agvs.pop(agv)
                        assigned_agvs[agv] = Mission(MissionType.RETURNING, best_id,
                                                     best_yx[1], best_yx[0], timestep)
                else:
                    # normal fetch: deliver to the closest empty pickerwall slot
                    pickerwall_occupied = env.get_pickerwall_info()
                    available_goals = [
                        (x, y) for i, (x, y) in enumerate(goal_locations)
                        if not pickerwall_occupied[i]
                    ]
                    busy_goal_xys = {(m.location_x, m.location_y) for m in assigned_agvs.values()
                                     if m.mission_type == MissionType.DELIVERING}
                    available_goals = [g for g in available_goals if g not in busy_goal_xys]

                    if available_goals:
                        # Route to AGV-entry cells adjacent to each goal for accurate distances.
                        goal_paths = [
                            env.find_agv_path_to_goal_entry(
                                (agv.y, agv.x), (x, y), agv, care_for_agents=False
                            )
                            for (x, y) in available_goals
                        ]
                        goal_dists = [len(p) if p else float('inf') for p in goal_paths]
                        closest = available_goals[np.argmin(goal_dists)]  # (x, y) of goal
                        goal_id = coords_original_loc_map[(closest[1], closest[0])]
                        assigned_agvs.pop(agv)
                        # location_x/y stored as the goal (col, row) so at_location can look
                        # up the correct entry cells via env._goal_to_agv_entry.
                        assigned_agvs[agv] = Mission(MissionType.DELIVERING, goal_id,
                                                     closest[0], closest[1], timestep)
                    # if no slot free, AGV waits; displacement will clear one

            elif mission.mission_type == MissionType.DELIVERING and mission.at_location and not agv.carrying_shelf:
                assigned_agvs.pop(agv)
                assigned_items.pop(agv, None)

            elif mission.mission_type == MissionType.RETURNING and mission.at_location and not agv.carrying_shelf:
                assigned_agvs.pop(agv)
                assigned_items.pop(agv, None)

        # displacement: clear fulfilled pickerwall slots when above occupancy threshold.
        # Accounts for in-progress removals to avoid sending redundant AGVs.
        DISPLACEMENT_THRESHOLD = 0.7

        pickerwall_occupied = env.get_pickerwall_info()
        occupied_count = int(np.sum(pickerwall_occupied))

        displaceable = env.get_pickerwall_displacement_info()
        displaceable_goals = [(x, y) for i, (x, y) in enumerate(goal_locations) if displaceable[i]]
        already_targeted = {(m.location_x, m.location_y) for m in assigned_agvs.values()
                            if m.mission_type == MissionType.PICKING
                            and (m.location_x, m.location_y) in goal_locations}
        displaceable_goals = [g for g in displaceable_goals if g not in already_targeted]

        effective_occupied = occupied_count - len(already_targeted)

        if displaceable_goals and effective_occupied / env.num_goals >= DISPLACEMENT_THRESHOLD:
            free_agvs = [a for a in agvs if not a.busy and not a.carrying_shelf
                         and a not in assigned_agvs]
            for goal_xy in displaceable_goals:
                if not free_agvs:
                    break
                if effective_occupied / env.num_goals < DISPLACEMENT_THRESHOLD:
                    break
                agv_paths = [env.find_agv_path_to_goal_entry(
                                (a.y, a.x), goal_xy, a, care_for_agents=False)
                             for a in free_agvs]
                agv_dists = [len(p) if p else float('inf') for p in agv_paths]
                if all(d == float('inf') for d in agv_dists):
                    break  # no AGV can reach this displacement goal; skip
                chosen = free_agvs[np.argmin(agv_dists)]
                goal_id = coords_original_loc_map[(goal_xy[1], goal_xy[0])]
                assigned_agvs[chosen] = Mission(MissionType.PICKING, goal_id,
                                                goal_xy[0], goal_xy[1], timestep)
                effective_occupied -= 1
                free_agvs.remove(chosen)

        # fetch missions: assign remaining free AGVs to requested shelves
        for item in request_queue:
            if item.id in assigned_items.values():
                continue

            available_agvs = [a for a in agvs if not a.busy and not a.carrying_shelf
                               and a not in assigned_agvs]
            if not available_agvs:
                continue

            agv_paths = [env.find_agv_path((a.y, a.x), (item.y, item.x), a, care_for_agents=False)
                         for a in available_agvs]
            agv_dists = [len(p) if p else float('inf') for p in agv_paths]
            if all(d == float('inf') for d in agv_dists):
                continue  # no AGV can reach this item; try next
            closest_agv = available_agvs[np.argmin(agv_dists)]
            item_location_id = coords_original_loc_map[(item.y, item.x)]
            assigned_agvs[closest_agv] = Mission(MissionType.PICKING, item_location_id,
                                                  item.x, item.y, timestep)
            assigned_items[closest_agv] = item.id

        for agv, mission in assigned_agvs.items():
            actions[agv] = mission.location_id if not agv.busy else 0

        _, reward, terminated, truncated, info = env.step(list(actions.values()))
        done = terminated or truncated
        episode_returns += np.array(reward, dtype=np.float64)
        global_episode_return += np.sum(reward)
        done = all(done)
        all_infos.append(info)

        if render and timestep >= render_start:
            env.render(mode="human")
            if render_sleep:
                time.sleep(render_sleep)
        timestep += 1

    return all_infos, global_episode_return, episode_returns
