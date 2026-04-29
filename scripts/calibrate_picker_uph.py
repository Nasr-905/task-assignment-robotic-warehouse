"""Calibrate single-picker throughput against a 150 UPH target.

This script runs the heuristic controller on a medium map while sweeping:
- AGV count (to ensure picker-side saturation)
- picker base pick ticks (_PICK_TICKS via env override)
- human-factors profile overrides

It writes:
- CSV with all trial metrics
- Markdown summary with closest-to-target configs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Iterator, List

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_EXT_SRC = _PROJECT_ROOT / "external" / "pyastar2d_TARWARE" / "src"
for _path in (_PROJECT_ROOT, _EXT_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Fallback for local calibration runs where the C++ pyastar extension is not built.
if "pyastar2d" not in sys.modules:
    pyastar2d_stub = types.ModuleType("pyastar2d")

    def _manhattan_path(_grid, start, goal, allow_diagonal=False):
        start = np.asarray(start, dtype=np.int64)
        goal = np.asarray(goal, dtype=np.int64)
        y, x = int(start[0]), int(start[1])
        gy, gx = int(goal[0]), int(goal[1])
        path = [[y, x]]
        while y != gy:
            y += 1 if gy > y else -1
            path.append([y, x])
        while x != gx:
            x += 1 if gx > x else -1
            path.append([y, x])
        return np.asarray(path, dtype=np.int64)

    pyastar2d_stub.astar_path = _manhattan_path
    sys.modules["pyastar2d"] = pyastar2d_stub

from tarware.definitions import RewardType
from tarware.heuristic import heuristic_episode
from tarware.warehouse import Warehouse


@dataclass(frozen=True)
class HFOverridePreset:
    name: str
    low_profile_override: Dict[str, float]


@dataclass
class TrialResult:
    agvs: int
    pick_ticks: int
    pick_base_seconds: float
    hf_preset: str
    mean_uph: float
    min_uph: float
    max_uph: float
    mean_items: float
    mean_sim_seconds: float
    seed_count: int


HF_PRESETS: List[HFOverridePreset] = [
    HFOverridePreset("zhao_low_default", {}),
    HFOverridePreset(
        "low_fast",
        {
            "pick_duration_fatigue_gain": 0.0,
            "movement_delay_base_prob": 0.0,
            "movement_delay_fatigue_prob_gain": 0.0,
            "failed_pick_base_prob": 0.0,
            "failed_pick_fatigue_prob_gain": 0.0,
            "failed_pick_delay_seconds": 0.25,
            "fatigue_recovery_per_second": 0.50,
        },
    ),
    HFOverridePreset(
        "low_moderate",
        {
            "pick_duration_fatigue_gain": 0.10,
            "movement_delay_base_prob": 0.002,
            "movement_delay_fatigue_prob_gain": 0.001,
            "failed_pick_base_prob": 0.001,
            "failed_pick_fatigue_prob_gain": 0.001,
            "failed_pick_delay_seconds": 0.50,
            "fatigue_recovery_per_second": 0.40,
        },
    ),
    HFOverridePreset(
        "low_slow",
        {
            "pick_duration_fatigue_gain": 0.80,
            "movement_delay_base_prob": 0.08,
            "movement_delay_fatigue_prob_gain": 0.02,
            "failed_pick_base_prob": 0.10,
            "failed_pick_fatigue_prob_gain": 0.05,
            "failed_pick_delay_seconds": 2.0,
            "fatigue_recovery_per_second": 0.20,
        },
    ),
]


@contextmanager
def temporary_env(overrides: Dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def parse_int_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_seed_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def run_trial(
    *,
    map_csv_path: Path,
    map_json_path: Path,
    order_csv_path: Path,
    num_agvs: int,
    pick_ticks: int,
    pick_base_seconds: float,
    hf_preset: HFOverridePreset,
    seeds: Iterable[int],
    max_steps: int,
    request_queue_size: int,
    steps_per_simulated_second: float,
) -> TrialResult:
    uph_values: List[float] = []
    picked_values: List[int] = []
    sim_seconds_values: List[float] = []

    env_overrides = {
        "TARWARE_PICK_BASE_TICKS": str(pick_ticks),
        "TARWARE_PICK_BASE_SECONDS": str(pick_base_seconds),
        "TARWARE_PICKER_POLICY": "fifo",
        "TARWARE_PICKER_USE_SKU_SIZE_TIME": "0",
        "TARWARE_PICKER_STALL_PROBABILITY": "0.0",
        "TARWARE_HF_ENABLED": "1",
        "TARWARE_HF_DEFAULT_PROFILE": "low",
        "TARWARE_HF_PICKER_PROFILE_OVERRIDES": '{"0":"low"}',
        "TARWARE_HF_PROFILE_LOW": json.dumps(hf_preset.low_profile_override),
    }

    with temporary_env(env_overrides):
        for seed in seeds:
            env = Warehouse(
                map_csv_path=map_csv_path,
                map_json_path=map_json_path,
                order_csv_path=order_csv_path,
                num_agvs=num_agvs,
                num_pickers=1,
                observation_type="partial",
                request_queue_size=request_queue_size,
                steps_per_simulated_second=steps_per_simulated_second,
                max_inactivity_steps=None,
                max_steps=max_steps,
                reward_type=RewardType.GLOBAL,
            )
            try:
                infos, _global_return, _episode_return = heuristic_episode(env, seed=seed)
            finally:
                env.close()

            total_items = int(sum(int(info.get("items_picked", 0)) for info in infos))
            sim_seconds = float(infos[-1].get("simulated_seconds", len(infos) / steps_per_simulated_second)) if infos else 0.0
            uph = (total_items * 3600.0 / sim_seconds) if sim_seconds > 0 else 0.0

            uph_values.append(uph)
            picked_values.append(total_items)
            sim_seconds_values.append(sim_seconds)

    return TrialResult(
        agvs=num_agvs,
        pick_ticks=pick_ticks,
        pick_base_seconds=pick_base_seconds,
        hf_preset=hf_preset.name,
        mean_uph=mean(uph_values),
        min_uph=min(uph_values),
        max_uph=max(uph_values),
        mean_items=mean(picked_values),
        mean_sim_seconds=mean(sim_seconds_values),
        seed_count=len(uph_values),
    )


def write_results_csv(path: Path, results: List[TrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "agvs",
                "pick_ticks",
                "pick_base_seconds",
                "hf_preset",
                "mean_uph",
                "min_uph",
                "max_uph",
                "mean_items",
                "mean_sim_seconds",
                "seed_count",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.agvs,
                    row.pick_ticks,
                    f"{row.pick_base_seconds:.3f}",
                    row.hf_preset,
                    f"{row.mean_uph:.3f}",
                    f"{row.min_uph:.3f}",
                    f"{row.max_uph:.3f}",
                    f"{row.mean_items:.3f}",
                    f"{row.mean_sim_seconds:.3f}",
                    row.seed_count,
                ]
            )


def write_summary_md(path: Path, results: List[TrialResult], target_uph: float, tolerance_uph: float) -> None:
    ranked = sorted(results, key=lambda r: abs(r.mean_uph - target_uph))
    near_target = [r for r in ranked if abs(r.mean_uph - target_uph) <= tolerance_uph]
    top = ranked[:10]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Picker UPH Calibration Findings\n\n")
        handle.write(f"Target UPH: {target_uph:.1f} (+/- {tolerance_uph:.1f})\n\n")
        handle.write("## Near-target configs\n\n")
        if near_target:
            handle.write("| agvs | pick_ticks | pick_base_seconds | hf_preset | mean_uph | min_uph | max_uph |\n")
            handle.write("|---:|---:|---:|---|---:|---:|---:|\n")
            for r in near_target[:12]:
                handle.write(
                    f"| {r.agvs} | {r.pick_ticks} | {r.pick_base_seconds:.2f} | {r.hf_preset} | "
                    f"{r.mean_uph:.2f} | {r.min_uph:.2f} | {r.max_uph:.2f} |\n"
                )
        else:
            handle.write("No configuration landed within the requested tolerance.\n")

        handle.write("\n## Closest configs\n\n")
        handle.write("| agvs | pick_ticks | pick_base_seconds | hf_preset | mean_uph | abs_error |\n")
        handle.write("|---:|---:|---:|---|---:|---:|\n")
        for r in top:
            handle.write(
                f"| {r.agvs} | {r.pick_ticks} | {r.pick_base_seconds:.2f} | {r.hf_preset} | "
                f"{r.mean_uph:.2f} | {abs(r.mean_uph - target_uph):.2f} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate single-picker throughput on medium map")
    parser.add_argument("--target-uph", type=float, default=150.0)
    parser.add_argument("--tolerance-uph", type=float, default=20.0)
    parser.add_argument("--agvs", type=str, default="24,32,40,48")
    parser.add_argument("--pick-ticks", type=str, default="1,2,3")
    parser.add_argument("--pick-base-seconds", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument(
        "--hf-presets",
        type=str,
        default=",".join(p.name for p in HF_PRESETS),
        help="Comma-separated subset of HF presets to evaluate.",
    )
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--steps-per-sim-second", type=float, default=1.0)
    parser.add_argument("--request-queue-size", type=int, default=120)
    parser.add_argument(
        "--map-csv",
        type=Path,
        default=_PROJECT_ROOT / "data" / "maps" / "medium_dhl.csv",
    )
    parser.add_argument(
        "--map-json",
        type=Path,
        default=_PROJECT_ROOT / "data" / "maps" / "medium_dhl.json",
    )
    parser.add_argument(
        "--order-csv",
        type=Path,
        default=_PROJECT_ROOT / "data" / "processed" / "order_data_full.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=_PROJECT_ROOT / "docs" / "picker_uph_calibration_results.csv",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=_PROJECT_ROOT / "docs" / "picker_uph_calibration_findings.md",
    )
    args = parser.parse_args()

    agv_values = parse_int_list(args.agvs)
    pick_tick_values = parse_int_list(args.pick_ticks)
    pick_base_seconds_values = [float(part.strip()) for part in args.pick_base_seconds.split(",") if part.strip()]
    seeds = parse_seed_list(args.seeds)
    selected_names = {name.strip() for name in args.hf_presets.split(",") if name.strip()}
    selected_hf_presets = [p for p in HF_PRESETS if p.name in selected_names]
    if not selected_hf_presets:
        raise ValueError("No valid HF presets selected. Use names from: " + ", ".join(p.name for p in HF_PRESETS))

    results: List[TrialResult] = []
    total = len(agv_values) * len(pick_tick_values) * len(pick_base_seconds_values) * len(selected_hf_presets)
    idx = 0

    for agvs in agv_values:
        for pick_ticks in pick_tick_values:
            for pick_base_seconds in pick_base_seconds_values:
                for hf_preset in selected_hf_presets:
                    idx += 1
                    print(
                        f"[{idx}/{total}] agvs={agvs} pick_ticks={pick_ticks} "
                        f"pick_base_seconds={pick_base_seconds:.2f} hf={hf_preset.name}"
                    )
                    result = run_trial(
                        map_csv_path=args.map_csv,
                        map_json_path=args.map_json,
                        order_csv_path=args.order_csv,
                        num_agvs=agvs,
                        pick_ticks=pick_ticks,
                        pick_base_seconds=pick_base_seconds,
                        hf_preset=hf_preset,
                        seeds=seeds,
                        max_steps=args.max_steps,
                        request_queue_size=args.request_queue_size,
                        steps_per_simulated_second=args.steps_per_sim_second,
                    )
                    results.append(result)

    write_results_csv(args.out_csv, results)
    write_summary_md(args.out_md, results, args.target_uph, args.tolerance_uph)

    ranked = sorted(results, key=lambda r: abs(r.mean_uph - args.target_uph))
    best = ranked[0]
    print("\nBest configuration:")
    print(
        f"agvs={best.agvs}, pick_ticks={best.pick_ticks}, pick_base_seconds={best.pick_base_seconds:.2f}, "
        f"hf={best.hf_preset}, mean_uph={best.mean_uph:.2f}"
    )
    print(f"CSV: {args.out_csv}")
    print(f"Summary: {args.out_md}")


if __name__ == "__main__":
    main()
