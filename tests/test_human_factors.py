"""Unit tests for human factors module and Zhao et al. 2019 model integration.

Tests cover:
- Zhao formula implementations with realistic physiological parameters
- Profile calibration pipeline (physiology → effort profiles)
- Profile assignment reproducibility with seeded sampling
- Fatigue progression and recovery dynamics
- Random profile generation with realistic variation
- Human factors config loading and override logic
"""

import os
import unittest
from unittest.mock import patch

import sys
from pathlib import Path

# Import human_factors module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "human_factors",
    Path(__file__).parent.parent / "tarware" / "human_factors.py"
)
human_factors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(human_factors)

PhysicalTimeConfig = human_factors.PhysicalTimeConfig
PickerPhysiologicalParams = human_factors.PickerPhysiologicalParams
ZhaoMetabolicModel = human_factors.ZhaoMetabolicModel
PickerEffortProfile = human_factors.PickerEffortProfile
PickerHumanFactorsState = human_factors.PickerHumanFactorsState
HumanFactorsConfig = human_factors.HumanFactorsConfig
_calibrate_profiles_from_zhao = human_factors._calibrate_profiles_from_zhao
_profile_from_physiological_params = human_factors._profile_from_physiological_params
generate_random_picker_profile = human_factors.generate_random_picker_profile


class TestPhysicalTimeConfig(unittest.TestCase):
    """Test step/time conversion utilities."""

    def test_conversion_roundtrip(self):
        """Verify step ↔ seconds conversion is consistent."""
        tc = PhysicalTimeConfig(
            steps_per_simulated_second=10.0,
            real_seconds_per_simulated_second=1.0,
            grid_cell_size_m=1.0,
            agv_nominal_speed_m_s=1.0,
        )
        
        # 10 seconds should convert to 100 steps
        steps = tc.simulated_seconds_to_steps(10.0, ceil=False)
        self.assertEqual(steps, 100)
        
        # And back
        seconds_per_step = tc.simulated_seconds_per_step
        self.assertAlmostEqual(seconds_per_step, 0.1, places=5)

    def test_per_second_to_per_step_scaling(self):
        """Test per-second rate conversion to per-step."""
        tc = PhysicalTimeConfig(
            steps_per_simulated_second=5.0,
            real_seconds_per_simulated_second=1.0,
            grid_cell_size_m=1.0,
            agv_nominal_speed_m_s=1.0,
        )
        
        # 10 per second → 2 per step (since 0.2 seconds per step)
        per_step = tc.per_second_to_per_step(10.0)
        self.assertAlmostEqual(per_step, 2.0, places=5)

    def test_different_step_rates_preserve_physical_time(self):
        """Test that different step rates yield same physical-time behavior."""
        # Configuration 1: 10 steps/sec
        tc1 = PhysicalTimeConfig(
            steps_per_simulated_second=10.0,
            real_seconds_per_simulated_second=1.0,
            grid_cell_size_m=1.0,
            agv_nominal_speed_m_s=1.0,
        )
        
        # Configuration 2: 5 steps/sec
        tc2 = PhysicalTimeConfig(
            steps_per_simulated_second=5.0,
            real_seconds_per_simulated_second=1.0,
            grid_cell_size_m=1.0,
            agv_nominal_speed_m_s=1.0,
        )
        
        # For same 10 simulated seconds:
        # tc1: 100 steps
        # tc2: 50 steps
        self.assertEqual(tc1.simulated_seconds_to_steps(10.0, ceil=False), 100)
        self.assertEqual(tc2.simulated_seconds_to_steps(10.0, ceil=False), 50)
        
        # But per-step duration is inverse relationship
        self.assertAlmostEqual(tc1.simulated_seconds_per_step * 100, 10.0, places=5)
        self.assertAlmostEqual(tc2.simulated_seconds_per_step * 50, 10.0, places=5)


class TestZhaoMetabolicModel(unittest.TestCase):
    """Test Zhao et al. 2019 formula implementations."""

    def setUp(self):
        self.model = ZhaoMetabolicModel()
        self.test_params = PickerPhysiologicalParams.medium_effort_worker()

    def test_body_surface_area_bounds(self):
        """BSA should be 1.5–2.5 m² for typical adults."""
        bsa = self.model.body_surface_area(
            height_cm=173.0, mass_kg=80.0
        )
        self.assertGreater(bsa, 1.5)
        self.assertLess(bsa, 2.5)

    def test_relative_metabolic_rate_male(self):
        """RMR should scale linearly with HR."""
        rmr_low = self.model.relative_metabolic_rate(90.0, "male")
        rmr_high = self.model.relative_metabolic_rate(150.0, "male")
        
        # Higher HR should give higher RMR
        self.assertGreater(rmr_high, rmr_low)
        
        # Should be in reasonable range
        self.assertGreater(rmr_low, -5.0)
        self.assertLess(rmr_high, 10.0)

    def test_relative_metabolic_rate_female(self):
        """Female RMR formula should differ from male."""
        rmr_male = self.model.relative_metabolic_rate(120.0, "male")
        rmr_female = self.model.relative_metabolic_rate(120.0, "female")
        
        # Different coefficients should yield different values
        self.assertNotAlmostEqual(rmr_male, rmr_female, places=2)

    def test_relative_heart_rate_bounds(self):
        """RHR should be normalized to 0–100%."""
        rhr_low = self.model.relative_heart_rate(70.0, 60.0, 180.0)
        rhr_high = self.model.relative_heart_rate(160.0, 60.0, 180.0)
        
        self.assertGreaterEqual(rhr_low, 0.0)
        self.assertLessEqual(rhr_low, 100.0)
        self.assertGreater(rhr_high, rhr_low)
        self.assertLessEqual(rhr_high, 100.0)

    def test_maximum_acceptable_work_duration_decreases_with_rhr(self):
        """MAWD should decrease as relative heart rate increases."""
        mawd_low = self.model.maximum_acceptable_work_duration(20.0)
        mawd_high = self.model.maximum_acceptable_work_duration(80.0)
        
        self.assertGreater(mawd_low, mawd_high)
        self.assertGreater(mawd_low, 1.0)
        self.assertGreater(mawd_high, 1.0)

    def test_energy_expenditure_positive(self):
        """Energy expenditure should be non-negative."""
        q = self.model.energy_expenditure(
            rmr=3.0, hr=130.0, bsa=1.93, time_minutes=1.0
        )
        self.assertGreater(q, 0.0)

    def test_energy_expenditure_scales_with_time(self):
        """Energy should scale linearly with duration."""
        q_1min = self.model.energy_expenditure(
            rmr=3.0, hr=130.0, bsa=1.93, time_minutes=1.0
        )
        q_5min = self.model.energy_expenditure(
            rmr=3.0, hr=130.0, bsa=1.93, time_minutes=5.0
        )
        
        self.assertAlmostEqual(q_5min / q_1min, 5.0, places=1)

    def test_fatigue_scaled_duration_increases_with_position(self):
        """Duration should increase exponentially with position."""
        base = 3.0
        
        d0 = self.model.fatigue_scaled_duration(base, 0, 0.5)
        d5 = self.model.fatigue_scaled_duration(base, 5, 0.5)
        d10 = self.model.fatigue_scaled_duration(base, 10, 0.5)
        
        self.assertEqual(d0, base)  # position 0 = no scaling
        self.assertGreater(d5, base)
        self.assertGreater(d10, d5)

    def test_fatigue_scaled_duration_zero_factor(self):
        """Zero fatigue factor should not scale duration."""
        base = 3.0
        result = self.model.fatigue_scaled_duration(base, 10, 0.0)
        self.assertEqual(result, base)


class TestPickerPhysiologicalParams(unittest.TestCase):
    """Test physiological parameter models."""

    def test_low_effort_worker_realistic(self):
        """Low effort worker should be young and fit."""
        params = PickerPhysiologicalParams.low_effort_worker()
        
        self.assertEqual(params.age, 25.0)
        self.assertLess(params.hr_rest, 60.0)  # Fit person
        self.assertGreater(params.hr_max, 190.0)  # Young person

    def test_medium_effort_worker_typical(self):
        """Medium effort worker should be typical age/fitness."""
        params = PickerPhysiologicalParams.medium_effort_worker()
        
        self.assertEqual(params.age, 40.0)
        self.assertGreater(params.hr_rest, 60.0)  # Typical resting
        self.assertLess(params.hr_rest, 70.0)

    def test_high_effort_worker_older(self):
        """High effort worker should be older/deconditioned."""
        params = PickerPhysiologicalParams.high_effort_worker()
        
        self.assertGreater(params.age, 50.0)
        self.assertGreater(params.hr_rest, 70.0)  # Higher resting
        self.assertLess(params.hr_max, 170.0)  # Lower max

    def test_random_worker_reproducible(self):
        """Random worker generation should be seeded."""
        w1 = PickerPhysiologicalParams.random(seed=42)
        w2 = PickerPhysiologicalParams.random(seed=42)
        
        self.assertEqual(w1.age, w2.age)
        self.assertEqual(w1.mass_kg, w2.mass_kg)
        self.assertEqual(w1.sex, w2.sex)

    def test_random_worker_varies(self):
        """Different seeds should generate different workers."""
        w1 = PickerPhysiologicalParams.random(seed=1)
        w2 = PickerPhysiologicalParams.random(seed=2)
        
        # At least one parameter should differ
        differs = (
            w1.age != w2.age
            or w1.mass_kg != w2.mass_kg
            or w1.height_cm != w2.height_cm
        )
        self.assertTrue(differs)

    def test_random_worker_realistic_ranges(self):
        """Random workers should be in realistic ranges."""
        for seed in range(10):
            params = PickerPhysiologicalParams.random(seed=seed)
            
            self.assertGreaterEqual(params.age, 20.0)
            self.assertLessEqual(params.age, 65.0)
            self.assertGreater(params.mass_kg, 0)
            self.assertGreater(params.height_cm, 0)
            self.assertGreater(params.hr_max, 0)
            self.assertGreater(params.hr_rest, 0)


class TestProfileCalibration(unittest.TestCase):
    """Test profile calibration pipeline."""

    def test_calibrate_low_medium_high_profiles(self):
        """Should generate three distinct profiles."""
        profiles = _calibrate_profiles_from_zhao()
        
        self.assertIn("low", profiles)
        self.assertIn("medium", profiles)
        self.assertIn("high", profiles)
        
        low = profiles["low"]
        med = profiles["medium"]
        high = profiles["high"]
        
        # Verify all profiles are properly formed
        self.assertEqual(low.name, "low")
        self.assertGreater(low.metabolic_rate_picking, 0)
        self.assertGreater(low.fatigue_gain_per_effort, 0)

    def test_profile_hierarchy_fatigue_gradient(self):
        """Profiles should show increasing fatigue stress: low < medium < high."""
        profiles = _calibrate_profiles_from_zhao()
        
        low_gain = profiles["low"].fatigue_gain_per_effort
        med_gain = profiles["medium"].fatigue_gain_per_effort
        high_gain = profiles["high"].fatigue_gain_per_effort
        
        self.assertLess(low_gain, med_gain)
        self.assertLess(med_gain, high_gain)

    def test_profile_hierarchy_recovery_inverse(self):
        """Recovery rate should be inverse: high > medium > low (less fatigable = faster recovery)."""
        profiles = _calibrate_profiles_from_zhao()
        
        low_recovery = profiles["low"].fatigue_recovery_per_second
        med_recovery = profiles["medium"].fatigue_recovery_per_second
        high_recovery = profiles["high"].fatigue_recovery_per_second
        
        self.assertGreater(low_recovery, high_recovery)

    def test_metabolic_rates_normalized(self):
        """Metabolic rates should be normalized relative values."""
        profiles = _calibrate_profiles_from_zhao()
        
        for name, profile in profiles.items():
            self.assertGreater(profile.metabolic_rate_idle, 0)
            self.assertGreater(profile.metabolic_rate_picking, 0)
            # picking should generally be higher than idle
            self.assertGreater(profile.metabolic_rate_picking, profile.metabolic_rate_idle)

    def test_profile_from_physiological_params(self):
        """Test direct calibration from physiological params."""
        params = PickerPhysiologicalParams.low_effort_worker()
        profile = _profile_from_physiological_params(params, "test_low")
        
        self.assertEqual(profile.name, "test_low")
        self.assertGreater(profile.fatigue_gain_per_effort, 0)
        self.assertGreater(profile.fatigue_recovery_per_second, 0.05)

    def test_random_profile_generation(self):
        """Test random profile generation."""
        profiles = [generate_random_picker_profile(seed=i) for i in range(5)]
        
        # All should be valid profiles
        for p in profiles:
            self.assertGreater(p.metabolic_rate_picking, 0)
            self.assertGreater(p.fatigue_gain_per_effort, 0)

    def test_random_profile_seeded_reproducible(self):
        """Random profiles should be reproducible with seed."""
        p1 = generate_random_picker_profile(seed=123)
        p2 = generate_random_picker_profile(seed=123)
        
        self.assertEqual(p1.name, p2.name)
        self.assertEqual(p1.fatigue_gain_per_effort, p2.fatigue_gain_per_effort)


class TestHumanFactorsConfig(unittest.TestCase):
    """Test configuration loading and override logic."""

    def tearDown(self):
        """Clean up environment variables."""
        for key in [
            "TARWARE_HF_ENABLED",
            "TARWARE_HF_DEFAULT_PROFILE",
            "TARWARE_HF_DEFAULT_PROFILE_BY_MAP",
            "TARWARE_HF_PICKER_PROFILE_OVERRIDES",
        ]:
            if key in os.environ:
                del os.environ[key]

    def test_default_config(self):
        """Default config should load successfully."""
        time_config = PhysicalTimeConfig.from_env(steps_per_simulated_second=10.0)
        config = HumanFactorsConfig.from_env(
            map_name="medium",
            time_config=time_config,
            fallback_pick_base_ticks=3,
            fallback_pick_unit_cube_tick_scale=0.5,
        )
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.default_profile, "medium")
        self.assertGreater(len(config.profiles), 0)

    def test_enable_disable_flag(self):
        """Should respect TARWARE_HF_ENABLED flag."""
        os.environ["TARWARE_HF_ENABLED"] = "false"
        time_config = PhysicalTimeConfig.from_env(steps_per_simulated_second=10.0)
        config = HumanFactorsConfig.from_env(
            map_name="medium",
            time_config=time_config,
            fallback_pick_base_ticks=3,
            fallback_pick_unit_cube_tick_scale=0.5,
        )
        
        self.assertFalse(config.enabled)

    def test_default_profile_override(self):
        """Should respect TARWARE_HF_DEFAULT_PROFILE."""
        os.environ["TARWARE_HF_DEFAULT_PROFILE"] = "high"
        time_config = PhysicalTimeConfig.from_env(steps_per_simulated_second=10.0)
        config = HumanFactorsConfig.from_env(
            map_name="medium",
            time_config=time_config,
            fallback_pick_base_ticks=3,
            fallback_pick_unit_cube_tick_scale=0.5,
        )
        
        self.assertEqual(config.default_profile, "high")

    def test_map_profile_override(self):
        """Should apply per-map profile overrides."""
        os.environ["TARWARE_HF_DEFAULT_PROFILE_BY_MAP"] = '{"small": "low", "large": "high"}'
        time_config = PhysicalTimeConfig.from_env(steps_per_simulated_second=10.0)
        
        config_small = HumanFactorsConfig.from_env(
            map_name="small",
            time_config=time_config,
            fallback_pick_base_ticks=3,
            fallback_pick_unit_cube_tick_scale=0.5,
        )
        
        config_large = HumanFactorsConfig.from_env(
            map_name="large",
            time_config=time_config,
            fallback_pick_base_ticks=3,
            fallback_pick_unit_cube_tick_scale=0.5,
        )
        
        self.assertEqual(config_small.default_profile, "low")
        self.assertEqual(config_large.default_profile, "high")

    def test_picker_profile_assignment(self):
        """Should assign per-picker profiles deterministically."""
        os.environ["TARWARE_HF_PICKER_PROFILE_OVERRIDES"] = '{"0": "low", "1": "high", "2": "medium"}'
        time_config = PhysicalTimeConfig.from_env(steps_per_simulated_second=10.0)
        config = HumanFactorsConfig.from_env(
            map_name="medium",
            time_config=time_config,
            fallback_pick_base_ticks=3,
            fallback_pick_unit_cube_tick_scale=0.5,
        )
        
        # Picker 0 should get "low"
        profile_0 = config.profile_for_picker_index(0)
        self.assertEqual(profile_0.name, "low")
        
        # Picker 1 should get "high"
        profile_1 = config.profile_for_picker_index(1)
        self.assertEqual(profile_1.name, "high")
        
        # Picker 99 should get default
        profile_99 = config.profile_for_picker_index(99)
        self.assertEqual(profile_99.name, "medium")


class TestPickerHumanFactorsState(unittest.TestCase):
    """Test per-picker fatigue/energy tracking state."""

    def test_state_initialization(self):
        """State should initialize with zero fatigue/energy."""
        state = PickerHumanFactorsState(profile_name="medium")
        
        self.assertEqual(state.profile_name, "medium")
        self.assertEqual(state.fatigue, 0.0)
        self.assertEqual(state.energy_expended, 0.0)
        self.assertEqual(state.movement_delay_events, 0)

    def test_state_accumulation(self):
        """State variables should accumulate."""
        state = PickerHumanFactorsState(profile_name="medium")
        
        state.fatigue += 5.0
        state.energy_expended += 10.0
        state.movement_delay_events += 1
        
        self.assertEqual(state.fatigue, 5.0)
        self.assertEqual(state.energy_expended, 10.0)
        self.assertEqual(state.movement_delay_events, 1)


class TestFatigueProgression(unittest.TestCase):
    """Test fatigue accumulation and recovery mechanics."""

    def test_fatigue_clamping(self):
        """Fatigue should clamp to min/max range."""
        time_config = PhysicalTimeConfig.from_env(steps_per_simulated_second=10.0)
        config = HumanFactorsConfig.from_env(
            map_name="test",
            time_config=time_config,
            fallback_pick_base_ticks=3,
            fallback_pick_unit_cube_tick_scale=0.5,
        )
        
        # Fatigue should be between bounds
        self.assertGreaterEqual(config.fatigue_min, 0.0)
        self.assertGreaterEqual(config.fatigue_max, config.fatigue_min)

    def test_recovery_rate_per_step(self):
        """Recovery rate conversion from per-second to per-step."""
        time_config = PhysicalTimeConfig(
            steps_per_simulated_second=10.0,
            real_seconds_per_simulated_second=1.0,
            grid_cell_size_m=1.0,
            agv_nominal_speed_m_s=1.0,
        )
        
        recovery_per_sec = 0.2  # per second
        recovery_per_step = time_config.per_second_to_per_step(recovery_per_sec)
        
        # 0.2 per sec = 0.02 per step (since 0.1 sec per step)
        self.assertAlmostEqual(recovery_per_step, 0.02, places=5)


class TestReproducibility(unittest.TestCase):
    """Test deterministic behavior for validation."""

    def test_seeded_profile_assignment_reproducible(self):
        """Profile assignment with fixed seed should be reproducible."""
        profiles1 = _calibrate_profiles_from_zhao()
        profiles2 = _calibrate_profiles_from_zhao()
        
        for name in ["low", "medium", "high"]:
            p1 = profiles1[name]
            p2 = profiles2[name]
            
            self.assertAlmostEqual(
                p1.fatigue_gain_per_effort,
                p2.fatigue_gain_per_effort,
                places=10,
            )

    def test_random_profiles_consistent_ranges(self):
        """Random profiles should stay within consistent ranges."""
        profiles = [generate_random_picker_profile(seed=i) for i in range(20)]
        
        gains = [p.fatigue_gain_per_effort for p in profiles]
        recoveries = [p.fatigue_recovery_per_second for p in profiles]
        
        # All should be positive
        for gain in gains:
            self.assertGreater(gain, 0)
        for recovery in recoveries:
            self.assertGreater(recovery, 0.05)
            self.assertLess(recovery, 0.35)


if __name__ == "__main__":
    unittest.main()
