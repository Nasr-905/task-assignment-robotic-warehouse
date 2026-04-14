"""Integration tests for warehouse human factors runtime behavior.

These tests focus on Phase 3-5 requirements:
- Per-picker profile assignment on reset
- Per-step diagnostics in info
- Episode-level human factors summary at termination
- Model registry fallback behavior
"""

import os
import sys
import unittest
import types
from pathlib import Path

import numpy as np

# Make local pyastar2d source importable for warehouse module imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXTERNAL_PYASTAR_SRC = _REPO_ROOT / "external" / "pyastar2d_TARWARE" / "src"
if str(_EXTERNAL_PYASTAR_SRC) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_PYASTAR_SRC))


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
from tarware.warehouse import Warehouse


class TestWarehouseHumanFactorsIntegration(unittest.TestCase):
    def setUp(self):
        self._old_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def _make_env(self, max_steps: int = 3, num_pickers: int = 1) -> Warehouse:
        map_path = _REPO_ROOT / "data" / "maps" / "medium.csv"
        order_path = _REPO_ROOT / "data" / "processed" / "order_data_sample.csv"
        env = Warehouse(
            map_csv_path=map_path,
            order_csv_path=order_path,
            num_agvs=1,
            num_pickers=num_pickers,
            observation_type="partial",
            request_queue_size=4,
            steps_per_simulated_second=5.0,
            max_inactivity_steps=None,
            max_steps=max_steps,
            reward_type=RewardType.GLOBAL,
        )
        return env

    def test_reset_assigns_picker_profiles(self):
        os.environ["TARWARE_HF_ENABLED"] = "1"
        os.environ["TARWARE_HF_DEFAULT_PROFILE"] = "medium"
        os.environ["TARWARE_HF_PICKER_PROFILE_OVERRIDES"] = '{"0": "low"}'

        env = self._make_env(max_steps=2, num_pickers=1)
        try:
            env.reset(seed=7)
            self.assertEqual(len(env.pickers), 1)
            picker = env.pickers[0]
            profile = env._picker_hf_profile_by_id.get(picker.id)
            state = env._picker_hf_state_by_id.get(picker.id)

            self.assertIsNotNone(profile)
            self.assertIsNotNone(state)
            self.assertEqual(profile.name, "low")
            self.assertEqual(state.profile_name, "low")
        finally:
            env.close()

    def test_step_info_contains_hf_diagnostics(self):
        os.environ["TARWARE_HF_ENABLED"] = "1"

        env = self._make_env(max_steps=4, num_pickers=1)
        try:
            env.reset(seed=11)
            _, _, terminated, truncated, info = env.step([0])

            self.assertIn("human_factors_model", info)
            self.assertIn("picker_fatigue", info)
            self.assertIn("picker_energy_expended", info)
            self.assertIn("picker_delay_steps", info)
            self.assertIn("picker_failed_pick_delay_events", info)
            self.assertIn("picker_fatigue_mean", info)
            self.assertIn("picker_fatigue_max", info)
            self.assertIn("picker_energy_total", info)
            self.assertIn("picker_states", info)
            self.assertIn("picker_reason_codes", info)
            self.assertIn("picker_positions", info)
            self.assertIn("picker_path_lengths", info)
            self.assertIn("picker_blocked_ticks", info)
            self.assertIn("picker_blocked_by", info)
            self.assertIn("picker_stalled", info)
            self.assertIn("picker_diagnostics", info)
            self.assertIn("picker_hf_enabled", info)
            self.assertIn("picker_hf_profile_name", info)
            self.assertIn("picker_hf_fatigue", info)
            self.assertIn("picker_hf_fatigue_ratio", info)
            self.assertIn("picker_hf_energy_expended", info)
            self.assertIn("picker_hf_movement_delay_probability", info)
            self.assertIn("picker_hf_failed_pick_probability", info)
            self.assertIn("picker_hf_movement_delay_events", info)
            self.assertIn("picker_hf_failed_pick_delay_events_per_picker", info)
            self.assertIn("picker_hf_cumulative_delay_steps", info)
            self.assertIn("picker_hf_cumulative_recovery_seconds", info)

            self.assertEqual(info["human_factors_model"], "zhao")
            self.assertGreaterEqual(float(info["picker_fatigue_mean"]), 0.0)
            self.assertGreaterEqual(float(info["picker_fatigue_max"]), 0.0)
            self.assertGreaterEqual(float(info["picker_energy_total"]), 0.0)
            self.assertEqual(len(info["picker_states"]), 1)
            self.assertEqual(len(info["picker_reason_codes"]), 1)
            self.assertEqual(len(info["picker_diagnostics"]), 1)
            self.assertFalse(all(terminated) or all(truncated))
        finally:
            env.close()

    def test_terminal_step_emits_episode_hf_summary(self):
        os.environ["TARWARE_HF_ENABLED"] = "1"

        env = self._make_env(max_steps=1, num_pickers=1)
        try:
            env.reset(seed=5)
            _, _, terminated, truncated, info = env.step([0])

            self.assertTrue(all(terminated) or all(truncated))
            self.assertIn("episode_human_factors_summary", info)
            summary = info["episode_human_factors_summary"]
            self.assertEqual(summary["model"], "zhao")
            self.assertEqual(summary["picker_count"], 1)
            self.assertIn("delay_steps", summary)
            self.assertIn("failed_pick_delay_events", summary)
            self.assertIn("fatigue_mean", summary)
            self.assertIn("fatigue_max", summary)
            self.assertIn("energy_total", summary)
            self.assertIn("simulated_seconds", summary)
        finally:
            env.close()

    def test_unknown_model_falls_back_to_zhao(self):
        os.environ["TARWARE_HF_ENABLED"] = "1"
        os.environ["TARWARE_HF_MODEL"] = "does_not_exist"

        env = self._make_env(max_steps=2, num_pickers=1)
        try:
            env.reset(seed=3)
            self.assertEqual(env.human_factors_config.model_name, "zhao")
            _, _, _, _, info = env.step([0])
            self.assertEqual(info["human_factors_model"], "zhao")
        finally:
            env.close()

    def test_speed_controls_publish_motion_info(self):
        os.environ["TARWARE_USE_PHYSICAL_SPEEDS"] = "0"
        os.environ["TARWARE_AGV_CELLS_PER_STEP"] = "0.25"
        os.environ["TARWARE_PICKER_CELLS_PER_STEP"] = "0.5"

        env = self._make_env(max_steps=3, num_pickers=1)
        try:
            env.reset(seed=9)
            _, _, _, _, info = env.step([0])

            self.assertEqual(info["motion_speed_model"], "cells_per_step")
            self.assertAlmostEqual(float(info["agv_cells_per_step_configured"]), 0.25)
            self.assertAlmostEqual(float(info["agv_cells_per_step_effective"]), 0.25)
            self.assertAlmostEqual(float(info["picker_cells_per_step_configured"]), 0.5)
            self.assertAlmostEqual(float(info["picker_cells_per_step_effective"]), 0.5)
            self.assertIn("picker_speed_limited", info)
            self.assertIn("agv_speed_limited_count", info)
        finally:
            env.close()

    def test_movement_credit_supports_substep_speeds(self):
        os.environ["TARWARE_USE_PHYSICAL_SPEEDS"] = "0"
        os.environ["TARWARE_AGV_CELLS_PER_STEP"] = "0.25"

        env = self._make_env(max_steps=2, num_pickers=0)
        try:
            env.reset(seed=21)
            agent = env.agents[0]
            self.assertAlmostEqual(env._agv_cells_per_step_effective, 0.25)

            self.assertFalse(env._consume_motion_credit(agent, env._agv_cells_per_step_effective))
            self.assertFalse(env._consume_motion_credit(agent, env._agv_cells_per_step_effective))
            self.assertFalse(env._consume_motion_credit(agent, env._agv_cells_per_step_effective))
            self.assertTrue(env._consume_motion_credit(agent, env._agv_cells_per_step_effective))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
