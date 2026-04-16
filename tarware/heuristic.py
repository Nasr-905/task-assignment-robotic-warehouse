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


def heuristic_episode(env, render=False, seed=None, render_start=0, render_skip=0, render_sleep=0.0):
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

    def _pickerwall_action_for_empty_slot(goal_xy, reserved_ids=()):
        """Return the action_id of the first empty pickerwall slot at this cell
        that has not already been reserved by another AGV's DELIVERING mission."""
        shelf = env.shelves_by_xy.get(goal_xy)
        if shelf is None:
            return None
        reserved = set(reserved_ids)
        for slot_idx, bin_ in enumerate(shelf.bin_slots):
            if bin_ is not None:
                continue
            aid = env.slot_to_action_id.get((shelf.id, slot_idx))
            if aid is not None and aid not in reserved:
                return aid
        return None

    def _pickerwall_action_for_fulfilled_bin(goal_xy, reserved_ids=()):
        shelf = env.shelves_by_xy.get(goal_xy)
        if shelf is None:
            return None
        reserved = set(reserved_ids)
        for slot_idx, bin_ in enumerate(shelf.bin_slots):
            if bin_ is None or not bin_.fulfilled:
                continue
            aid = env.slot_to_action_id.get((shelf.id, slot_idx))
            if aid is not None and aid not in reserved:
                return aid
        return None

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

            if mission.mission_type == MissionType.PICKING and mission.at_location and agv.carrying_bin:
                was_at_goal = (mission.location_x, mission.location_y) in goal_locations

                if was_at_goal:
                    # displacement pick-up: depleted bins return to replenishment;
                    # non-depleted bins return to an empty rack slot.
                    if agv.carrying_bin.depleted:
                        empty_locations = env.get_empty_replenishment_information()
                    else:
                        empty_locations = env.get_empty_shelf_information()
                    empty_location_ids = list(non_goal_location_ids[empty_locations > 0])
                    busy_loc_ids = [m.location_id for m in assigned_agvs.values()]
                    empty_location_ids = [i for i in empty_location_ids if i not in busy_loc_ids]
                    if empty_location_ids:
                        empty_yx = [location_map[i] for i in empty_location_ids]
                        dists = [len(p) if p else float('inf')
                                 for p in (env.find_agv_path((agv.y, agv.x), yx, care_for_agents=False)
                                           for yx in empty_yx)]
                        best_id = empty_location_ids[np.argmin(dists)]
                        best_yx = location_map[best_id]
                        assigned_agvs.pop(agv)
                        assigned_agvs[agv] = Mission(MissionType.RETURNING, best_id,
                                                     best_yx[1], best_yx[0], timestep)
                else:
                    # normal fetch: deliver to the closest pickerwall cell with
                    # an empty slot that is not already reserved by another AGV.
                    reserved_delivering_ids = {
                        m.location_id for m in assigned_agvs.values()
                        if m.mission_type == MissionType.DELIVERING
                    }
                    candidate_goals = [
                        goal_xy for goal_xy in goal_locations
                        if _pickerwall_action_for_empty_slot(goal_xy, reserved_delivering_ids) is not None
                    ]

                    if candidate_goals:
                        goal_paths = [
                            env.find_agv_path_to_goal_entry(
                                (agv.y, agv.x), goal_xy, care_for_agents=False
                            )
                            for goal_xy in candidate_goals
                        ]
                        goal_dists = [len(p) if p else float('inf') for p in goal_paths]
                        closest = candidate_goals[np.argmin(goal_dists)]  # (x, y) of cell
                        goal_id = _pickerwall_action_for_empty_slot(closest, reserved_delivering_ids)
                        if goal_id is None:
                            continue
                        assigned_agvs.pop(agv)
                        # location_x/y stored as the goal (col, row) so at_location can look
                        # up the correct entry cells via env._goal_to_agv_entry.
                        assigned_agvs[agv] = Mission(MissionType.DELIVERING, goal_id,
                                                     closest[0], closest[1], timestep)
                    # if no slot free, AGV waits; displacement will clear one

            elif mission.mission_type == MissionType.DELIVERING and mission.at_location and not agv.carrying_bin:
                assigned_agvs.pop(agv)
                assigned_items.pop(agv, None)

            elif mission.mission_type == MissionType.RETURNING and mission.at_location and not agv.carrying_bin:
                assigned_agvs.pop(agv)
                assigned_items.pop(agv, None)

        # displacement: clear fulfilled pickerwall slots when above occupancy threshold.
        # Occupancy and threshold are computed at slot granularity (not cell).
        DISPLACEMENT_THRESHOLD = 0.7

        pickerwall_occupied = env.get_pickerwall_info()
        occupied_count = int(np.sum(pickerwall_occupied))
        total_slots = env.num_pickerwall_actions

        displacement_reserved_ids = {
            m.location_id for m in assigned_agvs.values()
            if m.mission_type == MissionType.PICKING
            and (m.location_x, m.location_y) in goal_locations
        }
        displaceable_goals = [
            goal_xy for goal_xy in goal_locations
            if _pickerwall_action_for_fulfilled_bin(goal_xy, displacement_reserved_ids) is not None
        ]

        effective_occupied = occupied_count - len(displacement_reserved_ids)

        if displaceable_goals and total_slots and effective_occupied / total_slots >= DISPLACEMENT_THRESHOLD:
            free_agvs = [a for a in agvs if not a.busy and not a.carrying_bin
                         and a not in assigned_agvs]
            for goal_xy in displaceable_goals:
                if not free_agvs:
                    break
                if effective_occupied / total_slots < DISPLACEMENT_THRESHOLD:
                    break
                goal_id = _pickerwall_action_for_fulfilled_bin(goal_xy, displacement_reserved_ids)
                if goal_id is None:
                    continue
                agv_paths = [env.find_agv_path_to_goal_entry(
                                (a.y, a.x), goal_xy, care_for_agents=False)
                             for a in free_agvs]
                agv_dists = [len(p) if p else float('inf') for p in agv_paths]
                if all(d == float('inf') for d in agv_dists):
                    continue  # no AGV can reach this displacement goal; try next
                chosen = free_agvs[np.argmin(agv_dists)]
                assigned_agvs[chosen] = Mission(MissionType.PICKING, goal_id,
                                                goal_xy[0], goal_xy[1], timestep)
                displacement_reserved_ids.add(goal_id)
                effective_occupied -= 1
                free_agvs.remove(chosen)

        # fetch missions: assign remaining free AGVs to requested shelves
        for item in request_queue:
            if item.id in assigned_items.values():
                continue
            if item.shelf_id is None or item.slot_index is None:
                continue  # bin is in transit (carried); skip

            available_agvs = [a for a in agvs if not a.busy and not a.carrying_bin
                               and a not in assigned_agvs]
            if not available_agvs:
                continue

            agv_paths = [env.find_agv_path((a.y, a.x), (item.y, item.x), care_for_agents=False)
                         for a in available_agvs]
            agv_dists = [len(p) if p else float('inf') for p in agv_paths]
            if all(d == float('inf') for d in agv_dists):
                continue  # no AGV can reach this item; try next
            closest_agv = available_agvs[np.argmin(agv_dists)]
            item_location_id = env.slot_to_action_id.get((item.shelf_id, item.slot_index))
            if item_location_id is None:
                continue
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

        if render and timestep >= render_start and timestep % (render_skip + 1) == 0:
            env.render(mode="human")
            if render_sleep:
                time.sleep(render_sleep)
        timestep += 1

    return all_infos, global_episode_return, episode_returns
