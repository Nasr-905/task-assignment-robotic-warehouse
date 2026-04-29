import csv
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np


ENTITY_FIELD_NAMES = (
    "episode",
    "step",
    "simulated_seconds",
    "entity_kind",
    "entity_index",
    "entity_id",
    "x",
    "y",
    "prev_x",
    "prev_y",
    "dir",
    "busy",
    "target",
    "macro_action",
    "macro_target_x",
    "macro_target_y",
    "req_action",
    "path_length",
    "fixing_clash",
    "speed_limited",
    "motion_credit_cells",
    "carrying_bin_id",
    "carrying_sku",
    "carrying_quantity",
    "carrying_depleted",
    "state",
    "blocked_ticks",
    "stalled",
    "home_zone",
    "pick_ticks_remaining",
    "task_claim_count",
    "task_claim_index",
    "packaging_x",
    "packaging_y",
    "hf_profile",
    "hf_fatigue",
    "hf_energy_expended",
)

STEP_FIELD_NAMES = (
    "episode",
    "step",
    "simulated_seconds",
    "real_seconds",
    "reward_sum",
    "reward_mean",
    "terminated",
    "truncated",
    "request_queue_len",
    "order_pending_count",
    "order_active_count",
    "shelf_deliveries",
    "clashes",
    "picker_yields",
    "stucks",
    "agvs_distance_travelled",
    "agvs_idle_time",
    "vehicles_busy_count",
    "agv_speed_limited_count",
    "motion_speed_model",
    "picker_fatigue_mean",
    "picker_fatigue_max",
    "picker_energy_total",
    "items_picked",
)

DEFAULT_ENTITY_FIELDS = (
    "episode",
    "step",
    "simulated_seconds",
    "entity_kind",
    "entity_id",
    "x",
    "y",
    "macro_action",
    "macro_target_x",
    "macro_target_y",
    "req_action",
    "path_length",
    "busy",
    "carrying_bin_id",
    "carrying_sku",
    "state",
)

DEFAULT_STEP_FIELDS = (
    "episode",
    "step",
    "simulated_seconds",
    "real_seconds",
    "reward_sum",
    "request_queue_len",
    "order_pending_count",
    "order_active_count",
    "shelf_deliveries",
    "clashes",
    "picker_yields",
    "stucks",
    "agvs_distance_travelled",
    "agvs_idle_time",
    "items_picked",
)

VALID_ROLES = ("agv", "picker")


def format_available_fields() -> str:
    return "\n".join(
        [
            "Entity fields: " + ", ".join(ENTITY_FIELD_NAMES),
            "Step fields: " + ", ".join(STEP_FIELD_NAMES),
            "Roles: " + ", ".join(VALID_ROLES),
        ]
    )


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "stats"


def _parse_list(raw: Optional[str]) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_requested_fields(
    raw: Optional[str],
    available: Sequence[str],
    default: Sequence[str],
) -> list[str]:
    requested = _parse_list(raw)
    if not requested or requested == ["default"]:
        return list(default)
    if requested == ["none"]:
        return []
    if requested == ["all"]:
        return list(available)

    available_set = set(available)
    unknown = [field for field in requested if field not in available_set]
    if unknown:
        raise ValueError(f"Unknown stats field(s): {', '.join(unknown)}")

    resolved: list[str] = []
    seen: set[str] = set()
    for field in requested:
        if field not in seen:
            resolved.append(field)
            seen.add(field)
    return resolved


def _resolve_roles(raw: Optional[str]) -> list[str]:
    requested = _parse_list(raw)
    if not requested or requested == ["default"] or requested == ["all"]:
        return list(VALID_ROLES)

    valid = set(VALID_ROLES)
    unknown = [role for role in requested if role not in valid]
    if unknown:
        raise ValueError(f"Unknown stats role(s): {', '.join(unknown)}")
    return requested


def _to_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.name.lower()
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _reward_stats(reward: Any) -> tuple[Optional[float], Optional[float]]:
    if reward is None:
        return None, None
    if isinstance(reward, (list, tuple, np.ndarray)):
        arr = np.asarray(reward, dtype=np.float64)
        if arr.size == 0:
            return 0.0, 0.0
        return float(arr.sum()), float(arr.mean())
    value = float(reward)
    return value, value


def _all_true(value: Any) -> bool:
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(np.all(value))
    return bool(value)


@dataclass
class StatsConfig:
    enabled: bool
    output_dir: Path
    prefix: str
    every_n_steps: int
    roles: list[str]
    entity_fields: list[str]
    step_fields: list[str]
    write_entity_csv: bool
    write_step_csv: bool
    write_heatmaps: bool


class SimulationStatsTracker:
    def __init__(self, config: StatsConfig, run_name: str, env_id: str):
        self.config = config
        self.enabled = config.enabled
        self.run_name = run_name
        self.env_id = env_id

        self._episode_number: Optional[int] = None
        self._entity_handle = None
        self._step_handle = None
        self._entity_writer = None
        self._step_writer = None
        self._agv_heatmap: Optional[np.ndarray] = None
        self._picker_heatmap: Optional[np.ndarray] = None
        self._manifest_written = False

    @classmethod
    def from_args(cls, args, *, run_name: str, env_id: str) -> "SimulationStatsTracker":
        entity_fields = _resolve_requested_fields(
            getattr(args, "stats_entity_fields", None),
            ENTITY_FIELD_NAMES,
            DEFAULT_ENTITY_FIELDS,
        )
        step_fields = _resolve_requested_fields(
            getattr(args, "stats_step_fields", None),
            STEP_FIELD_NAMES,
            DEFAULT_STEP_FIELDS,
        )
        roles = _resolve_roles(getattr(args, "stats_roles", None))

        write_entity_csv = bool(getattr(args, "stats_write_entity_csv", True))
        write_step_csv = bool(getattr(args, "stats_write_step_csv", True))
        write_heatmaps = bool(getattr(args, "stats_heatmaps", False))
        enabled = bool(getattr(args, "stats_enable", False)) and (
            write_entity_csv or write_step_csv or write_heatmaps
        )

        prefix = getattr(args, "stats_prefix", None)
        if prefix:
            prefix = _sanitize_name(prefix)
        else:
            prefix = _sanitize_name(f"{run_name}_{env_id}")

        config = StatsConfig(
            enabled=enabled,
            output_dir=Path(getattr(args, "stats_dir", "stats")).expanduser(),
            prefix=prefix,
            every_n_steps=max(1, int(getattr(args, "stats_every", 1))),
            roles=roles,
            entity_fields=entity_fields,
            step_fields=step_fields,
            write_entity_csv=write_entity_csv and bool(entity_fields),
            write_step_csv=write_step_csv and bool(step_fields),
            write_heatmaps=write_heatmaps,
        )
        return cls(config=config, run_name=run_name, env_id=env_id)

    def start_episode(self, env, episode_number: int) -> None:
        if not self.enabled:
            return

        self.close_episode()
        self._episode_number = int(episode_number)
        output_dir = self.config.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        self._write_manifest(output_dir, env)

        stem = f"{self.config.prefix}_ep{self._episode_number:03d}"
        if self.config.write_entity_csv:
            entity_path = output_dir / f"{stem}_entities.csv"
            self._entity_handle = entity_path.open("w", newline="", encoding="utf-8")
            self._entity_writer = csv.DictWriter(self._entity_handle, fieldnames=self.config.entity_fields)
            self._entity_writer.writeheader()

        if self.config.write_step_csv:
            step_path = output_dir / f"{stem}_steps.csv"
            self._step_handle = step_path.open("w", newline="", encoding="utf-8")
            self._step_writer = csv.DictWriter(self._step_handle, fieldnames=self.config.step_fields)
            self._step_writer.writeheader()

        if self.config.write_heatmaps:
            self._agv_heatmap = np.zeros(env.grid_size, dtype=np.int64) if "agv" in self.config.roles else None
            self._picker_heatmap = np.zeros(env.grid_size, dtype=np.int64) if "picker" in self.config.roles else None

    def close_episode(self) -> None:
        if self._entity_handle is not None:
            self._entity_handle.close()
        if self._step_handle is not None:
            self._step_handle.close()

        self._entity_handle = None
        self._step_handle = None
        self._entity_writer = None
        self._step_writer = None
        self._episode_number = None

    def record_step(
        self,
        env,
        info: Optional[dict],
        *,
        macro_actions: Optional[Sequence[int]] = None,
        reward: Any = None,
        terminated: Any = None,
        truncated: Any = None,
    ) -> None:
        if not self.enabled or self._episode_number is None:
            return

        step = int(getattr(env, "_cur_steps", 0))
        if step <= 0:
            return
        if (step - 1) % self.config.every_n_steps != 0:
            return

        info = info or {}
        reward_sum, reward_mean = _reward_stats(reward)
        action_list = list(macro_actions) if macro_actions is not None else []
        context = {
            "episode": self._episode_number,
            "step": step,
            "info": info,
            "reward_sum": reward_sum,
            "reward_mean": reward_mean,
            "terminated": _all_true(terminated) if terminated is not None else None,
            "truncated": _all_true(truncated) if truncated is not None else None,
            "action_list": action_list,
        }

        if self._step_writer is not None:
            self._step_writer.writerow(
                {field: _to_scalar(self._step_field_value(field, env, context)) for field in self.config.step_fields}
            )

        if self._entity_writer is not None or self.config.write_heatmaps:
            self._record_entities(env, context)

    def finalize_episode(self, env) -> None:
        if not self.enabled or self._episode_number is None:
            return

        if self.config.write_heatmaps:
            output_dir = self.config.output_dir.resolve()
            stem = f"{self.config.prefix}_ep{self._episode_number:03d}"
            if self._agv_heatmap is not None:
                np.save(output_dir / f"{stem}_agv_heatmap.npy", self._agv_heatmap)
                np.savetxt(output_dir / f"{stem}_agv_heatmap.csv", self._agv_heatmap, fmt="%d", delimiter=",")
            if self._picker_heatmap is not None:
                np.save(output_dir / f"{stem}_picker_heatmap.npy", self._picker_heatmap)
                np.savetxt(output_dir / f"{stem}_picker_heatmap.csv", self._picker_heatmap, fmt="%d", delimiter=",")

        self.close_episode()
        self._agv_heatmap = None
        self._picker_heatmap = None

    def close(self) -> None:
        self.close_episode()

    def _record_entities(self, env, context: dict[str, Any]) -> None:
        for idx, agent in enumerate(getattr(env, "agents", [])):
            if "agv" not in self.config.roles:
                break
            action = context["action_list"][idx] if idx < len(context["action_list"]) else None
            if self._agv_heatmap is not None:
                self._agv_heatmap[agent.y, agent.x] += 1
            if self._entity_writer is not None:
                row = {
                    field: _to_scalar(self._entity_field_value(field, env, "agv", idx, agent, action))
                    for field in self.config.entity_fields
                }
                self._entity_writer.writerow(row)

        for idx, picker in enumerate(getattr(env, "pickers", [])):
            if "picker" not in self.config.roles:
                break
            if self._picker_heatmap is not None:
                self._picker_heatmap[picker.y, picker.x] += 1
            if self._entity_writer is not None:
                row = {
                    field: _to_scalar(self._entity_field_value(field, env, "picker", idx, picker, None))
                    for field in self.config.entity_fields
                }
                self._entity_writer.writerow(row)

    def _write_manifest(self, output_dir: Path, env) -> None:
        if self._manifest_written:
            return

        manifest = {
            "run_name": self.run_name,
            "env_id": self.env_id,
            "output_dir": str(output_dir),
            "every_n_steps": self.config.every_n_steps,
            "roles": list(self.config.roles),
            "entity_fields": list(self.config.entity_fields),
            "step_fields": list(self.config.step_fields),
            "write_entity_csv": self.config.write_entity_csv,
            "write_step_csv": self.config.write_step_csv,
            "write_heatmaps": self.config.write_heatmaps,
            "grid_size": list(getattr(env, "grid_size", ())),
            "num_agvs": int(getattr(env, "num_agvs", len(getattr(env, "agents", [])))),
            "num_pickers": int(getattr(env, "num_pickers", len(getattr(env, "pickers", [])))),
        }
        manifest_path = output_dir / f"{self.config.prefix}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._manifest_written = True

    def _step_field_value(self, field: str, env, context: dict[str, Any]) -> Any:
        info = context["info"]
        if field == "episode":
            return context["episode"]
        if field == "step":
            return context["step"]
        if field == "simulated_seconds":
            return info.get("simulated_seconds")
        if field == "real_seconds":
            return info.get("real_seconds")
        if field == "reward_sum":
            return context["reward_sum"]
        if field == "reward_mean":
            return context["reward_mean"]
        if field == "terminated":
            return context["terminated"]
        if field == "truncated":
            return context["truncated"]
        if field == "request_queue_len":
            return len(getattr(env, "request_queue", []))
        if field == "order_pending_count":
            order_sequencer = getattr(env, "order_sequencer", None)
            return getattr(order_sequencer, "pending_count", None) if order_sequencer is not None else None
        if field == "order_active_count":
            order_sequencer = getattr(env, "order_sequencer", None)
            return getattr(order_sequencer, "active_count", None) if order_sequencer is not None else None
        if field == "shelf_deliveries":
            return info.get("shelf_deliveries")
        if field == "clashes":
            return info.get("clashes")
        if field == "picker_yields":
            return info.get("picker_yields")
        if field == "stucks":
            return info.get("stucks")
        if field == "agvs_distance_travelled":
            return info.get("agvs_distance_travelled")
        if field == "agvs_idle_time":
            return info.get("agvs_idle_time")
        if field == "vehicles_busy_count":
            return int(sum(1 for busy in info.get("vehicles_busy", []) if busy))
        if field == "agv_speed_limited_count":
            return info.get("agv_speed_limited_count")
        if field == "motion_speed_model":
            return info.get("motion_speed_model")
        if field == "picker_fatigue_mean":
            return info.get("picker_fatigue_mean")
        if field == "picker_fatigue_max":
            return info.get("picker_fatigue_max")
        if field == "picker_energy_total":
            return info.get("picker_energy_total")
        if field == "items_picked":
            return info.get("items_picked")
        raise ValueError(f"Unsupported step stats field: {field}")

    def _entity_field_value(
        self,
        field: str,
        env,
        entity_kind: str,
        entity_index: int,
        entity,
        macro_action: Optional[int],
    ) -> Any:
        macro_target_x = None
        macro_target_y = None
        if macro_action not in (None, 0):
            target_yx = getattr(env, "action_id_to_coords_map", {}).get(macro_action)
            if target_yx is not None:
                macro_target_y = target_yx[0]
                macro_target_x = target_yx[1]

        carrying_bin = getattr(entity, "carrying_bin", None)
        picker_hf_state = getattr(env, "_picker_hf_state_by_id", {}).get(entity.id) if entity_kind == "picker" else None
        packaging_location = getattr(entity, "packaging_location", None)
        task = getattr(entity, "task", None)

        if field == "episode":
            return self._episode_number
        if field == "step":
            return int(getattr(env, "_cur_steps", 0))
        if field == "simulated_seconds":
            return env.steps_to_simulated_seconds(getattr(env, "_cur_steps", 0))
        if field == "entity_kind":
            return entity_kind
        if field == "entity_index":
            return entity_index
        if field == "entity_id":
            return entity.id
        if field == "x":
            return entity.x
        if field == "y":
            return entity.y
        if field == "prev_x":
            return entity.prev_x
        if field == "prev_y":
            return entity.prev_y
        if field == "dir":
            return getattr(entity, "dir", None)
        if field == "busy":
            return getattr(entity, "busy", None)
        if field == "target":
            return getattr(entity, "target", None)
        if field == "macro_action":
            return macro_action
        if field == "macro_target_x":
            return macro_target_x
        if field == "macro_target_y":
            return macro_target_y
        if field == "req_action":
            return getattr(entity, "req_action", None)
        if field == "path_length":
            path = getattr(entity, "path", None)
            return len(path) if path is not None else 0
        if field == "fixing_clash":
            return getattr(entity, "fixing_clash", None)
        if field == "speed_limited":
            return getattr(entity, "speed_limited_this_step", None)
        if field == "motion_credit_cells":
            return getattr(entity, "motion_credit_cells", None)
        if field == "carrying_bin_id":
            return getattr(carrying_bin, "id", None)
        if field == "carrying_sku":
            return getattr(carrying_bin, "sku", None)
        if field == "carrying_quantity":
            return getattr(carrying_bin, "quantity", None)
        if field == "carrying_depleted":
            return getattr(carrying_bin, "depleted", None)
        if field == "state":
            return getattr(entity, "state", None)
        if field == "blocked_ticks":
            return getattr(entity, "blocked_ticks", None)
        if field == "stalled":
            return getattr(entity, "stalled", None)
        if field == "home_zone":
            return getattr(entity, "home_zone", None)
        if field == "pick_ticks_remaining":
            return getattr(entity, "pick_ticks_remaining", None)
        if field == "task_claim_count":
            return len(task.claims) if task is not None else None
        if field == "task_claim_index":
            return getattr(task, "current_claim_index", None) if task is not None else None
        if field == "packaging_x":
            return packaging_location[0] if packaging_location is not None else None
        if field == "packaging_y":
            return packaging_location[1] if packaging_location is not None else None
        if field == "hf_profile":
            return getattr(picker_hf_state, "profile_name", None)
        if field == "hf_fatigue":
            return getattr(picker_hf_state, "fatigue", None)
        if field == "hf_energy_expended":
            return getattr(picker_hf_state, "energy_expended", None)
        raise ValueError(f"Unsupported entity stats field: {field}")
