import json
import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class PhysicalTimeConfig:
    """Warehouse-wide step/time conversion parameters.

    This conversion layer is shared by picker and AGV subsystems to support a
    consistent physical-time view across the warehouse.
    """

    steps_per_simulated_second: float
    real_seconds_per_simulated_second: float
    grid_cell_size_m: float
    agv_nominal_speed_m_s: float
    picker_nominal_speed_m_s: float = 1.0

    @classmethod
    def from_env(cls, steps_per_simulated_second: float) -> "PhysicalTimeConfig":
        return cls(
            steps_per_simulated_second=max(1e-6, float(steps_per_simulated_second)),
            real_seconds_per_simulated_second=max(
                1e-6,
                float(os.getenv("TARWARE_REAL_SECONDS_PER_SIM_SECOND", "1.0")),
            ),
            grid_cell_size_m=max(1e-6, float(os.getenv("TARWARE_GRID_CELL_SIZE_M", "1.0"))),
            agv_nominal_speed_m_s=max(0.0, float(os.getenv("TARWARE_AGV_NOMINAL_SPEED_M_S", "1.0"))),
            picker_nominal_speed_m_s=max(0.0, float(os.getenv("TARWARE_PICKER_NOMINAL_SPEED_M_S", "1.0"))),
        )

    @property
    def simulated_seconds_per_step(self) -> float:
        return 1.0 / self.steps_per_simulated_second

    @property
    def real_seconds_per_step(self) -> float:
        return self.simulated_seconds_per_step * self.real_seconds_per_simulated_second

    def simulated_seconds_to_steps(self, seconds: float, *, ceil: bool = True) -> int:
        raw_steps = max(0.0, seconds) * self.steps_per_simulated_second
        return int(math.ceil(raw_steps) if ceil else math.floor(raw_steps))

    def per_second_to_per_step(self, per_second_value: float) -> float:
        return float(per_second_value) * self.simulated_seconds_per_step

    def agv_nominal_cells_per_step(self) -> float:
        # Placeholder for future AGV motion coupling in physical units.
        if self.grid_cell_size_m <= 0:
            return 0.0
        meters_per_step = self.agv_nominal_speed_m_s * self.simulated_seconds_per_step
        return meters_per_step / self.grid_cell_size_m

    def picker_nominal_cells_per_step(self) -> float:
        if self.grid_cell_size_m <= 0:
            return 0.0
        meters_per_step = self.picker_nominal_speed_m_s * self.simulated_seconds_per_step
        return meters_per_step / self.grid_cell_size_m


@dataclass(frozen=True)
class PickerEffortProfile:
    name: str
    metabolic_rate_idle: float
    metabolic_rate_walking: float
    metabolic_rate_picking: float
    fatigue_gain_per_effort: float
    fatigue_recovery_per_second: float
    movement_delay_base_prob: float
    movement_delay_fatigue_prob_gain: float
    pick_duration_fatigue_gain: float
    failed_pick_base_prob: float
    failed_pick_fatigue_prob_gain: float
    failed_pick_delay_seconds: float


@dataclass
class PickerHumanFactorsState:
    profile_name: str
    fatigue: float = 0.0
    energy_expended: float = 0.0
    movement_delay_events: int = 0
    failed_pick_delay_events: int = 0
    cumulative_delay_steps: int = 0
    cumulative_recovery_seconds: float = 0.0


@dataclass(frozen=True)
class HumanFactorsConfig:
    enabled: bool
    model_name: str
    fatigue_min: float
    fatigue_max: float
    default_profile: str
    map_profile_overrides: Dict[str, str]
    picker_profile_overrides: Dict[int, str]
    profiles: Dict[str, PickerEffortProfile]
    pick_base_seconds: float
    pick_unit_cube_seconds_scale: float
    pick_quantity_extra_seconds: float

    @classmethod
    def from_env(
        cls,
        *,
        map_name: str,
        time_config: PhysicalTimeConfig,
        fallback_pick_base_ticks: int,
        fallback_pick_unit_cube_tick_scale: float,
    ) -> "HumanFactorsConfig":
        enabled = os.getenv("TARWARE_HF_ENABLED", "1").lower() in ("1", "true", "yes")
        model_name = os.getenv("TARWARE_HF_MODEL", "zhao").strip().lower()
        if model_name not in _profile_registry():
            model_name = "zhao"
        fatigue_min = float(os.getenv("TARWARE_HF_FATIGUE_MIN", "0.0"))
        fatigue_max = float(os.getenv("TARWARE_HF_FATIGUE_MAX", "100.0"))

        profiles = _load_profiles_from_env(model_name=model_name)

        map_profile_overrides = _read_json_dict_str("TARWARE_HF_DEFAULT_PROFILE_BY_MAP")
        default_profile = os.getenv("TARWARE_HF_DEFAULT_PROFILE", "medium").strip().lower()
        if map_name in map_profile_overrides:
            default_profile = map_profile_overrides[map_name].strip().lower()

        picker_profile_overrides_raw = _read_json_dict_str("TARWARE_HF_PICKER_PROFILE_OVERRIDES")
        picker_profile_overrides: Dict[int, str] = {}
        for key, value in picker_profile_overrides_raw.items():
            try:
                picker_profile_overrides[int(key)] = str(value).strip().lower()
            except ValueError:
                continue

        if default_profile not in profiles:
            default_profile = "medium"

        for picker_index, profile_name in list(picker_profile_overrides.items()):
            if profile_name not in profiles:
                picker_profile_overrides.pop(picker_index)

        pick_base_seconds = float(
            os.getenv(
                "TARWARE_PICK_BASE_SECONDS",
                str(float(fallback_pick_base_ticks)),
            )
        )
        pick_unit_cube_seconds_scale = float(
            os.getenv(
                "TARWARE_PICK_UNIT_CUBE_SECONDS_SCALE",
                str(float(fallback_pick_unit_cube_tick_scale)),
            )
        )
        pick_quantity_extra_seconds = float(
            os.getenv("TARWARE_PICK_QUANTITY_EXTRA_SECONDS", "1.0")
        )

        # Keep timing values valid and positive for stable conversion.
        pick_base_seconds = max(time_config.simulated_seconds_per_step, pick_base_seconds)
        pick_unit_cube_seconds_scale = max(0.0, pick_unit_cube_seconds_scale)
        pick_quantity_extra_seconds = max(0.0, pick_quantity_extra_seconds)

        return cls(
            enabled=enabled,
            model_name=model_name,
            fatigue_min=min(fatigue_min, fatigue_max),
            fatigue_max=max(fatigue_min, fatigue_max),
            default_profile=default_profile,
            map_profile_overrides=map_profile_overrides,
            picker_profile_overrides=picker_profile_overrides,
            profiles=profiles,
            pick_base_seconds=pick_base_seconds,
            pick_unit_cube_seconds_scale=pick_unit_cube_seconds_scale,
            pick_quantity_extra_seconds=pick_quantity_extra_seconds,
        )

    def profile_for_picker_index(self, picker_index: int) -> PickerEffortProfile:
        profile_name = self.picker_profile_overrides.get(picker_index, self.default_profile)
        return self.profiles.get(profile_name, self.profiles[self.default_profile])


def _read_json_dict_str(env_name: str) -> Dict[str, str]:
    raw = os.getenv(env_name, "{}").strip()
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {str(k): str(v) for k, v in decoded.items()}


@dataclass(frozen=True)
class PickerPhysiologicalParams:
    """Human physiological parameters used in Zhao et al. 2019 model.
    
    All metrics in SI units unless noted.
    """
    age: float  # years
    mass_kg: float  # kilograms
    height_cm: float  # centimeters
    sex: str  # "male" or "female"
    hr_rest: float  # resting heart rate (bpm)
    hr_max: float  # maximum heart rate (bpm)
    hr_work_low: float  # working heart rate during low-intensity picking (bpm)
    hr_work_pick: float  # working heart rate during picking (bpm)
    hr_work_high: float  # working heart rate during high-intensity picking (bpm)

    @classmethod
    def low_effort_worker(cls) -> "PickerPhysiologicalParams":
        """Fit, young worker with strong cardiovascular reserve."""
        return cls(
            age=25.0,
            mass_kg=70.0,
            height_cm=175.0,
            sex="male",
            hr_rest=55.0,
            hr_max=195.0,
            hr_work_low=100.0,
            hr_work_pick=115.0,
            hr_work_high=140.0,
        )

    @classmethod
    def medium_effort_worker(cls) -> "PickerPhysiologicalParams":
        """Average worker in typical condition."""
        return cls(
            age=40.0,
            mass_kg=80.0,
            height_cm=173.0,
            sex="male",
            hr_rest=65.0,
            hr_max=180.0,
            hr_work_low=110.0,
            hr_work_pick=130.0,
            hr_work_high=155.0,
        )

    @classmethod
    def high_effort_worker(cls) -> "PickerPhysiologicalParams":
        """Older or deconditioned worker, more taxing effort."""
        return cls(
            age=55.0,
            mass_kg=90.0,
            height_cm=170.0,
            sex="male",
            hr_rest=75.0,
            hr_max=165.0,
            hr_work_low=120.0,
            hr_work_pick=145.0,
            hr_work_high=160.0,
        )

    @classmethod
    def random(cls, seed: Optional[int] = None) -> "PickerPhysiologicalParams":
        """Generate a random realistic worker profile."""
        if seed is not None:
            random.seed(seed)
        sex = random.choice(["male", "female"])
        age = random.uniform(20.0, 65.0)
        
        if sex == "male":
            mass_kg = random.uniform(60.0, 100.0)
            height_cm = random.uniform(165.0, 190.0)
            hr_max = 220.0 - age
        else:
            mass_kg = random.uniform(50.0, 85.0)
            height_cm = random.uniform(155.0, 180.0)
            hr_max = 226.0 - age
        
        hr_rest = random.uniform(50.0, 80.0)
        hr_range = hr_max - hr_rest
        hr_work_low = hr_rest + hr_range * random.uniform(0.35, 0.55)
        hr_work_pick = hr_rest + hr_range * random.uniform(0.55, 0.70)
        hr_work_high = hr_rest + hr_range * random.uniform(0.75, 0.90)
        
        return cls(
            age=age,
            mass_kg=mass_kg,
            height_cm=height_cm,
            sex=sex,
            hr_rest=hr_rest,
            hr_max=hr_max,
            hr_work_low=hr_work_low,
            hr_work_pick=hr_work_pick,
            hr_work_high=hr_work_high,
        )


class ZhaoMetabolicModel:
    """Implements the Zhao et al. 2019 energy expenditure model.
    
    Reference: Zhao et al., "Research on the Work-rest Scheduling in the Manual 
    Order Picking Systems to Consider Human Factors"
    
    Key formula: Q = (RMR + 1.2) × (HR × BSA ÷ 60) × T
    Where:
    - Q: Energy expenditure (kJ)
    - RMR: Relative metabolic rate
    - HR: Heart rate (bpm)
    - BSA: Body surface area (m²)
    - T: Time (minutes)
    """
    THRESHOLD_ENERGY_PER_MINUTE = 41.9  # kJ/min (10.056 kcal/min)
    
    @staticmethod
    def body_surface_area(height_cm: float, mass_kg: float) -> float:
        """Compute BSA using Zhao et al. formula (3).
        
        BSA = 0.0061 × B + 0.0128 × M - 0.1529
        where B = height (cm), M = mass (kg)
        """
        return 0.0061 * height_cm + 0.0128 * mass_kg - 0.1529

    @staticmethod
    def relative_metabolic_rate(hr_picking: float, sex: str) -> float:
        """Compute RMR using Zhao et al. formulas (4) and (5).
        
        RMR_male = 0.072 × HR - 5.608
        RMR_female = 0.065 × HR - 4.932
        """
        if sex.lower() == "female":
            return 0.065 * hr_picking - 4.932
        else:
            return 0.072 * hr_picking - 5.608

    @staticmethod
    def basal_metabolic_rate(
        age: float, mass_kg: float, height_cm: float, sex: str, bsa: float
    ) -> float:
        """Compute BMR using Zhao et al. formulas (6) and (7).
        
        BMR_male = (13.7×M + 5.0×B - 6.8×Age + 66) ÷ (24×BSA)
        BMR_female = (9.6×M + 1.8×B - 4.7×Age + 655) ÷ (24×BSA)
        """
        if sex.lower() == "female":
            numerator = 9.6 * mass_kg + 1.8 * height_cm - 4.7 * age + 655.0
        else:
            numerator = 13.7 * mass_kg + 5.0 * height_cm - 6.8 * age + 66.0
        
        if bsa <= 0:
            return 1.0
        return max(1.0, numerator / (24.0 * bsa))

    @staticmethod
    def relative_heart_rate(
        hr_work: float, hr_rest: float, hr_max: float
    ) -> float:
        """Compute RHR using Zhao et al. formula (9).
        
        RHR = (HRwork - HRrest) ÷ (HRmax - HRrest) × 100%
        """
        denominator = hr_max - hr_rest
        if denominator <= 0:
            return 0.0
        return min(100.0, max(0.0, (hr_work - hr_rest) / denominator * 100.0))

    @staticmethod
    def maximum_acceptable_work_duration(rhr: float) -> float:
        """Compute MAWD using Zhao et al. formula (8).
        
        MAWD = -2.67 + e^(7.02 - 5.72 × RHR)
        Returns minutes until worker reaches fatigue threshold.
        """
        rhr_bounded = min(100.0, max(0.01, rhr))
        exponent = 7.02 - 5.72 * (rhr_bounded / 100.0)
        return max(1.0, -2.67 + math.exp(exponent))

    @staticmethod
    def energy_expenditure(
        rmr: float, hr: float, bsa: float, time_minutes: float
    ) -> float:
        """Compute energy expenditure Q using Zhao et al. formula (1).
        
        Q = (RMR + 1.2) × (HR × BSA ÷ 60) × T
        Returns energy in kJ.
        """
        if bsa <= 0 or hr <= 0:
            return 0.0
        return (rmr + 1.2) * (hr * bsa / 60.0) * time_minutes

    @staticmethod
    def fatigue_scaled_duration(
        base_duration_seconds: float, position: int, fatigue_factor: float
    ) -> float:
        """Scale duration by fatigue using Zhao et al. formula (11).
        
        p(i) = (1 + α)^(position) × p_base
        where α is fatigue factor (0 ≤ α ≤ 1), position is 0-indexed.
        Returns scaled duration in seconds.
        """
        alpha_bounded = min(1.0, max(0.0, fatigue_factor))
        if position <= 0:
            return base_duration_seconds
        scale = (1.0 + alpha_bounded) ** max(0, position - 1)
        return base_duration_seconds * scale


def _profile_registry() -> Dict[str, Callable[[], Dict[str, PickerEffortProfile]]]:
    return {
        "zhao": _calibrate_profiles_from_zhao,
    }


def _load_profiles_from_env(model_name: str = "zhao") -> Dict[str, PickerEffortProfile]:
    """Load picker effort profiles, with Zhao model-calibrated defaults."""
    registry = _profile_registry()
    profile_loader = registry.get(model_name, registry["zhao"])
    defaults = profile_loader()


    for profile_name in ("low", "medium", "high"):
        env_key = f"TARWARE_HF_PROFILE_{profile_name.upper()}"
        override_raw = os.getenv(env_key, "").strip()
        if not override_raw:
            continue
        try:
            override = json.loads(override_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(override, dict):
            continue

        profile = defaults[profile_name]
        defaults[profile_name] = PickerEffortProfile(
            name=profile_name,
            metabolic_rate_idle=float(override.get("metabolic_rate_idle", profile.metabolic_rate_idle)),
            metabolic_rate_walking=float(override.get("metabolic_rate_walking", profile.metabolic_rate_walking)),
            metabolic_rate_picking=float(override.get("metabolic_rate_picking", profile.metabolic_rate_picking)),
            fatigue_gain_per_effort=float(override.get("fatigue_gain_per_effort", profile.fatigue_gain_per_effort)),
            fatigue_recovery_per_second=float(override.get("fatigue_recovery_per_second", profile.fatigue_recovery_per_second)),
            movement_delay_base_prob=float(override.get("movement_delay_base_prob", profile.movement_delay_base_prob)),
            movement_delay_fatigue_prob_gain=float(
                override.get(
                    "movement_delay_fatigue_prob_gain",
                    profile.movement_delay_fatigue_prob_gain,
                )
            ),
            pick_duration_fatigue_gain=float(override.get("pick_duration_fatigue_gain", profile.pick_duration_fatigue_gain)),
            failed_pick_base_prob=float(override.get("failed_pick_base_prob", profile.failed_pick_base_prob)),
            failed_pick_fatigue_prob_gain=float(
                override.get(
                    "failed_pick_fatigue_prob_gain",
                    profile.failed_pick_fatigue_prob_gain,
                )
            ),
            failed_pick_delay_seconds=float(override.get("failed_pick_delay_seconds", profile.failed_pick_delay_seconds)),
        )
    return defaults


def _calibrate_profiles_from_zhao() -> Dict[str, PickerEffortProfile]:
    """Generate picker profiles using Zhao et al. 2019 physiological model.
    
    Maps metabolic rates from actual heart rate and body parameters to
    effort costs, fatigue dynamics, and error/delay probabilities.
    """
    low_params = PickerPhysiologicalParams.low_effort_worker()
    med_params = PickerPhysiologicalParams.medium_effort_worker()
    high_params = PickerPhysiologicalParams.high_effort_worker()
    
    return {
        "low": _profile_from_physiological_params(low_params, "low"),
        "medium": _profile_from_physiological_params(med_params, "medium"),
        "high": _profile_from_physiological_params(high_params, "high"),
    }


def _profile_from_physiological_params(
    params: PickerPhysiologicalParams, profile_name: str
) -> PickerEffortProfile:
    """Convert physiological parameters to a PickerEffortProfile using Zhao formulas.
    
    Maps energy expenditure (Q) and heart rate data to:
    - Metabolic rates: RMR-based effort cost for each activity
    - Fatigue dynamics: Recovery rate, fatigue gain per effort
    - Error/delay probability: Based on relative workload (RHR) vs capacity (MAWD)
    """
    model = ZhaoMetabolicModel()
    
    # Compute baseline physiology
    bsa = model.body_surface_area(params.height_cm, params.mass_kg)
    bmr = model.basal_metabolic_rate(
        params.age, params.mass_kg, params.height_cm, params.sex, bsa
    )
    
    # RMR at different work intensities
    rmr_low = model.relative_metabolic_rate(params.hr_work_low, params.sex)
    rmr_pick = model.relative_metabolic_rate(params.hr_work_pick, params.sex)
    rmr_high = model.relative_metabolic_rate(params.hr_work_high, params.sex)
    
    # Relative heart rates define workload stress
    rhr_low = model.relative_heart_rate(
        params.hr_work_low, params.hr_rest, params.hr_max
    )
    rhr_pick = model.relative_heart_rate(
        params.hr_work_pick, params.hr_rest, params.hr_max
    )
    rhr_high = model.relative_heart_rate(
        params.hr_work_high, params.hr_rest, params.hr_max
    )
    
    # Maximum acceptable work durations at each intensity
    mawd_low = model.maximum_acceptable_work_duration(rhr_low)
    mawd_pick = model.maximum_acceptable_work_duration(rhr_pick)
    mawd_high = model.maximum_acceptable_work_duration(rhr_high)
    
    # Energy expenditure per minute at each intensity
    q_low = model.energy_expenditure(rmr_low, params.hr_work_low, bsa, 1.0) / 60.0
    q_pick = model.energy_expenditure(rmr_pick, params.hr_work_pick, bsa, 1.0) / 60.0
    q_high = model.energy_expenditure(rmr_high, params.hr_work_high, bsa, 1.0) / 60.0
    
    # Normalize metabolic rates: idle baseline + activity increments
    max_rate = max(q_low, q_pick, q_high)
    if max_rate > 0:
        metabolic_rate_idle = q_low / max_rate
        metabolic_rate_walking = q_low / max_rate
        metabolic_rate_picking = q_pick / max_rate
    else:
        metabolic_rate_idle = 0.5
        metabolic_rate_walking = 0.8
        metabolic_rate_picking = 1.0
    
    # Fatigue gain: steeper for high-stress workers (higher RHR)
    mean_rhr = (rhr_low + rhr_pick + rhr_high) / 3.0
    fatigue_gain_per_effort = 0.015 + 0.015 * (mean_rhr / 100.0)
    
    # Recovery rate: inverse of fatigue accumulation
    fatigue_recovery_per_second = max(0.10, 0.35 - 0.15 * (mean_rhr / 100.0))
    
    # Movement delay probability: increases with workload
    movement_delay_base_prob = max(0.0, 0.01 * (mean_rhr / 70.0 - 1.0))
    movement_delay_fatigue_prob_gain = 0.003 + 0.002 * (mean_rhr / 100.0)
    
    # Pick duration fatigue gain: high-workload workers show more slowdown
    pick_duration_fatigue_gain = 0.30 + 0.30 * (mean_rhr / 100.0)
    
    # Error rate: based on proximity to MAWD threshold
    underutil_ratio = min(1.0, 50.0 / max(1.0, mawd_pick))
    failed_pick_base_prob = max(0.0, 0.005 * underutil_ratio)
    failed_pick_fatigue_prob_gain = 0.002 + 0.002 * (mean_rhr / 100.0)
    failed_pick_delay_seconds = 1.0 + 0.5 * (mean_rhr / 100.0)
    
    return PickerEffortProfile(
        name=profile_name,
        metabolic_rate_idle=metabolic_rate_idle,
        metabolic_rate_walking=metabolic_rate_walking,
        metabolic_rate_picking=metabolic_rate_picking,
        fatigue_gain_per_effort=fatigue_gain_per_effort,
        fatigue_recovery_per_second=fatigue_recovery_per_second,
        movement_delay_base_prob=movement_delay_base_prob,
        movement_delay_fatigue_prob_gain=movement_delay_fatigue_prob_gain,
        pick_duration_fatigue_gain=pick_duration_fatigue_gain,
        failed_pick_base_prob=failed_pick_base_prob,
        failed_pick_fatigue_prob_gain=failed_pick_fatigue_prob_gain,
        failed_pick_delay_seconds=failed_pick_delay_seconds,
    )


def generate_random_picker_profile(
    seed: Optional[int] = None, profile_name: Optional[str] = None
) -> PickerEffortProfile:
    """Generate a random, physically justified picker profile using Zhao model.
    
    Samples random physiological parameters and derives effort profile using
    the Zhao et al. 2019 metabolic model, ensuring values are realistic and
    scientifically grounded.
    
    Args:
        seed: Random seed for reproducibility.
        profile_name: Name for the generated profile (default: auto-generated).
    
    Returns:
        PickerEffortProfile with Zhao-calibrated parameters.
    """
    params = PickerPhysiologicalParams.random(seed=seed)
    auto_name = profile_name or f"random_{params.sex}_{int(params.age)}y"
    
    profile = _profile_from_physiological_params(params, auto_name)
    return profile
