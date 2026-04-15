"""Diagnostic figure generation for human factors model and profiles.

Generates matplotlib visualizations of:
- Profile parameter comparisons
- Zhao model outputs across worker types
- Fatigue/energy progression over an episode
- Metabolic rate scaling
- Work capacity vs heart rate relationships
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import human factors module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "human_factors",
    Path(__file__).parent.parent / "tarware" / "human_factors.py"
)
human_factors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(human_factors)

PickerPhysiologicalParams = human_factors.PickerPhysiologicalParams
ZhaoMetabolicModel = human_factors.ZhaoMetabolicModel
_calibrate_profiles_from_zhao = human_factors._calibrate_profiles_from_zhao
generate_random_picker_profile = human_factors.generate_random_picker_profile

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


class HumanFactorsDiagnostics:
    """Generate diagnostic figures for human factors model."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize diagnostics.
        
        Args:
            output_dir: Directory for saving figures. Defaults to ./diagnostics/
        """
        self.output_dir = Path(output_dir or "./diagnostics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = ZhaoMetabolicModel()

    def figure_profile_comparison(self) -> Optional[str]:
        """Generate figure comparing low/medium/high effort profiles.
        
        Returns:
            Path to saved figure, or None if matplotlib unavailable.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        profiles = _calibrate_profiles_from_zhao()
        
        names = list(profiles.keys())
        metrics = {
            "Metabolic Idle": [profiles[n].metabolic_rate_idle for n in names],
            "Metabolic Pick": [profiles[n].metabolic_rate_picking for n in names],
            "Fatigue Gain": [profiles[n].fatigue_gain_per_effort for n in names],
            "Recovery Rate": [profiles[n].fatigue_recovery_per_second for n in names],
            "Pick Fatigue Gain": [profiles[n].pick_duration_fatigue_gain for n in names],
            "Error Rate": [profiles[n].failed_pick_base_prob for n in names],
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Picker Effort Profiles: Low vs Medium vs High", fontsize=16, fontweight="bold")
        
        axes = axes.flatten()
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # Green, orange, red
        
        for idx, (metric, values) in enumerate(metrics.items()):
            ax = axes[idx]
            bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
            ax.set_ylabel(metric, fontweight="bold")
            ax.set_title(metric)
            ax.grid(True, alpha=0.3, axis="y")
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height,
                    f"{val:.4f}",
                    ha="center", va="bottom", fontsize=10
                )
        
        plt.tight_layout()
        path = self.output_dir / "profile_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return str(path)

    def figure_metabolic_energy_model(self) -> Optional[str]:
        """Generate figure showing energy expenditure across worker types.
        
        Visualizes Zhao formula 1: Q = (RMR + 1.2) × (HR × BSA ÷ 60) × T
        
        Returns:
            Path to saved figure, or None if matplotlib unavailable.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        low_params = PickerPhysiologicalParams.low_effort_worker()
        med_params = PickerPhysiologicalParams.medium_effort_worker()
        high_params = PickerPhysiologicalParams.high_effort_worker()
        
        # Compute energy expenditure over time durations
        durations = np.linspace(0.5, 60.0, 50)  # 0.5 to 60 minutes
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Energy Expenditure Model (Zhao et al.)", fontsize=16, fontweight="bold")
        
        # Plot 1: Energy per worker type at picking intensity
        worker_types = [
            ("Low Effort (fit, young)", low_params, "#2ecc71"),
            ("Medium Effort (typical)", med_params, "#f39c12"),
            ("High Effort (older)", high_params, "#e74c3c"),
        ]
        
        ax = axes[0]
        for label, params, color in worker_types:
            bsa = self.model.body_surface_area(params.height_cm, params.mass_kg)
            rmr = self.model.relative_metabolic_rate(params.hr_work_pick, params.sex)
            
            energies = [
                self.model.energy_expenditure(rmr, params.hr_work_pick, bsa, t)
                for t in durations
            ]
            ax.plot(durations, energies, label=label, linewidth=2.5, color=color, marker="o", markersize=4)
        
        ax.set_xlabel("Duration (minutes)", fontweight="bold")
        ax.set_ylabel("Energy Expenditure (kJ)", fontweight="bold")
        ax.set_title("Energy by Worker Type at Picking Intensity")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Energy vs relative heart rate for medium worker
        ax = axes[1]
        hrs = np.linspace(20, 90, 30)  # 20% to 90% relative heart rate
        bsa = self.model.body_surface_area(med_params.height_cm, med_params.mass_kg)
        
        for pick_time in [5, 10, 20, 30]:
            mawd_hrs = []
            mawd_times = []
            for rhr in hrs:
                mawd = self.model.maximum_acceptable_work_duration(rhr)
                mawd_hrs.append(rhr)
                mawd_times.append(mawd)
            
            ax.plot(hrs, mawd_times, label=f"MAWD", linewidth=2.5, marker="o", markersize=5)
        
        ax.set_xlabel("Relative Heart Rate (%)", fontweight="bold")
        ax.set_ylabel("Maximum Acceptable Work Duration (minutes)", fontweight="bold")
        ax.set_title("Work Capacity (MAWD) by Workload Intensity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / "metabolic_energy_model.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return str(path)

    def figure_fatigue_progression(
        self,
        profile_name: str = "medium",
        episode_picks: int = 200,
        fatigue_rate: float = 0.025,
        recovery_rate: float = 0.2,
    ) -> Optional[str]:
        """Generate figure showing fatigue progression during episode.
        
        Args:
            profile_name: Profile to simulate ("low", "medium", "high")
            episode_picks: Number of picks in episode
            fatigue_rate: Fatigue gain per pick (per effort)
            recovery_rate: Recovery per idle step (per second)
        
        Returns:
            Path to saved figure, or None if matplotlib unavailable.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        # Simulate fatigue progression
        fatigue = [0.0]
        energy = [0.0]
        
        # Alternate between picking (fatigue) and recovery
        for pick_idx in range(episode_picks):
            # Picking phase: accumulate fatigue and energy
            new_fatigue = min(100.0, fatigue[-1] + fatigue_rate * 10)
            new_energy = energy[-1] + 0.5  # Arbitrary energy increment
            
            fatigue.append(new_fatigue)
            energy.append(new_energy)
            
            # Every 20 picks, add recovery phase
            if (pick_idx + 1) % 20 == 0:
                for _ in range(30):  # 30 steps of recovery
                    new_fatigue = max(0.0, fatigue[-1] - recovery_rate * 0.1)
                    fatigue.append(new_fatigue)
                    energy.append(energy[-1])

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(
            f"Fatigue & Energy Progression (Profile: {profile_name.upper()})",
            fontsize=16, fontweight="bold"
        )
        
        steps = range(len(fatigue))
        
        # Plot 1: Fatigue over episode
        ax = axes[0]
        ax.fill_between(steps, fatigue, alpha=0.3, color="#e74c3c")
        ax.plot(steps, fatigue, linewidth=2, color="#c0392b", label="Fatigue Level")
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Moderate Fatigue Threshold")
        ax.set_ylabel("Fatigue (%)", fontweight="bold")
        ax.set_title("Fatigue Accumulation and Recovery")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Plot 2: Energy expenditure
        ax = axes[1]
        ax.fill_between(steps, energy, alpha=0.3, color="#3498db")
        ax.plot(steps, energy, linewidth=2, color="#2980b9", label="Cumulative Energy")
        ax.set_xlabel("Simulation Step", fontweight="bold")
        ax.set_ylabel("Cumulative Energy (kJ)", fontweight="bold")
        ax.set_title("Energy Expenditure Over Episode")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / f"fatigue_progression_{profile_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return str(path)

    def figure_heart_rate_zones(self) -> Optional[str]:
        """Generate figure showing heart rate zones and work capacity.
        
        Returns:
            Path to saved figure, or None if matplotlib unavailable.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        workers = [
            ("Low Effort (25y)", PickerPhysiologicalParams.low_effort_worker(), "#2ecc71"),
            ("Medium Effort (40y)", PickerPhysiologicalParams.medium_effort_worker(), "#f39c12"),
            ("High Effort (55y)", PickerPhysiologicalParams.high_effort_worker(), "#e74c3c"),
        ]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Heart Rate Zones and Work Capacity", fontsize=16, fontweight="bold")
        
        # Plot 1: Heart rate zones by worker
        ax = axes[0]
        y_pos = np.arange(len(workers))
        
        for idx, (label, params, color) in enumerate(workers):
            hr_rest = params.hr_rest
            hr_max = params.hr_max
            hr_low = params.hr_work_low
            hr_pick = params.hr_work_pick
            hr_high = params.hr_work_high
            
            # Zone widths
            zones = [
                ("Rest", hr_rest, "#ecf0f1"),
                ("Low Work", hr_low - hr_rest, "#2ecc71"),
                ("Pick Work", hr_pick - hr_low, "#f39c12"),
                ("High Work", hr_high - hr_pick, "#e74c3c"),
                ("Reserve", hr_max - hr_high, "#aaa"),
            ]
            
            x_start = 0
            for zone_name, zone_width, zone_color in zones:
                if zone_width > 0:
                    ax.barh(idx, zone_width, left=x_start, color=zone_color, 
                            edgecolor="black", linewidth=1)
                    x_start += zone_width
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([w[0] for w in workers])
        ax.set_xlabel("Heart Rate (bpm)", fontweight="bold")
        ax.set_title("Heart Rate Zone Distribution")
        ax.grid(True, alpha=0.3, axis="x")
        
        # Plot 2: Relative heart rate vs maximum acceptable work duration
        ax = axes[1]
        
        for label, params, color in workers:
            rhrs = np.linspace(10, 95, 50)
            mawds = [self.model.maximum_acceptable_work_duration(rhr) for rhr in rhrs]
            ax.plot(rhrs, mawds, label=label, linewidth=2.5, color=color, marker="o", markersize=4)
        
        ax.set_xlabel("Relative Heart Rate (%)", fontweight="bold")
        ax.set_ylabel("Maximum Acceptable Work Duration (minutes)", fontweight="bold")
        ax.set_title("Work Duration Capacity vs. Workload Intensity")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 300])
        
        plt.tight_layout()
        path = self.output_dir / "heart_rate_zones.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return str(path)

    def figure_fatigue_scaling_formula(self) -> Optional[str]:
        """Generate figure showing fatigue-scaled pick duration formula.
        
        Visualizes Zhao formula 11: p(i) = (1 + α)^(i-1) × p_base
        
        Returns:
            Path to saved figure, or None if matplotlib unavailable.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        
        base_duration = 3.0  # seconds
        positions = np.arange(0, 51)  # 50 items in sequence
        alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(alpha_values)))
        
        for alpha, color in zip(alpha_values, colors):
            durations = [
                self.model.fatigue_scaled_duration(base_duration, pos, alpha)
                for pos in positions
            ]
            ax.plot(
                positions, durations,
                label=f"α = {alpha:.1f}", linewidth=2.5, color=color, marker="o", markersize=5
            )
        
        ax.set_xlabel("Position in Pick Sequence", fontweight="bold")
        ax.set_ylabel("Pick Duration (seconds)", fontweight="bold")
        ax.set_title(
            f"Fatigue-Scaled Pick Duration\n"
            f"p(i) = (1 + α)^(i-1) × p_base, where p_base = {base_duration}s",
            fontsize=14, fontweight="bold"
        )
        ax.legend(loc="upper left", title="Fatigue Factor (α)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        
        plt.tight_layout()
        path = self.output_dir / "fatigue_scaling_formula.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return str(path)

    def figure_random_profile_distribution(self, num_samples: int = 100) -> Optional[str]:
        """Generate figure showing random profile parameter distribution.
        
        Args:
            num_samples: Number of random profiles to sample
        
        Returns:
            Path to saved figure, or None if matplotlib unavailable.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        # Generate random profiles
        profiles = [generate_random_picker_profile(seed=i) for i in range(num_samples)]
        
        # Extract metrics
        fatigue_gains = [p.fatigue_gain_per_effort for p in profiles]
        recovery_rates = [p.fatigue_recovery_per_second for p in profiles]
        metabolic_picks = [p.metabolic_rate_picking for p in profiles]
        movement_delays = [p.movement_delay_base_prob for p in profiles]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Random Profile Distribution (n={num_samples} samples)", 
                    fontsize=16, fontweight="bold")
        
        # Plot 1: Fatigue gain histogram
        ax = axes[0, 0]
        ax.hist(fatigue_gains, bins=20, color="#e74c3c", alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(fatigue_gains), color="darkred", linestyle="--", linewidth=2,
                  label=f"Mean: {np.mean(fatigue_gains):.4f}")
        ax.set_xlabel("Fatigue Gain per Effort", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title("Fatigue Gain Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        
        # Plot 2: Recovery rate histogram
        ax = axes[0, 1]
        ax.hist(recovery_rates, bins=20, color="#3498db", alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(recovery_rates), color="darkblue", linestyle="--", linewidth=2,
                  label=f"Mean: {np.mean(recovery_rates):.4f}")
        ax.set_xlabel("Recovery Rate (per second)", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title("Recovery Rate Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        
        # Plot 3: Metabolic rate histogram
        ax = axes[1, 0]
        ax.hist(metabolic_picks, bins=20, color="#2ecc71", alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(metabolic_picks), color="darkgreen", linestyle="--", linewidth=2,
                  label=f"Mean: {np.mean(metabolic_picks):.4f}")
        ax.set_xlabel("Metabolic Rate (Picking)", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title("Picking Metabolic Rate Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        
        # Plot 4: 2D scatter (fatigue vs recovery)
        ax = axes[1, 1]
        scatter = ax.scatter(
            fatigue_gains, recovery_rates,
            c=metabolic_picks, s=50, alpha=0.6, cmap="viridis", edgecolor="black", linewidth=0.5
        )
        ax.set_xlabel("Fatigue Gain per Effort", fontweight="bold")
        ax.set_ylabel("Recovery Rate (per second)", fontweight="bold")
        ax.set_title("Fatigue vs Recovery (colored by Metabolic Rate)")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Metabolic Rate", fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / "random_profile_distribution.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return str(path)

    def generate_all_figures(self) -> Dict[str, str]:
        """Generate all diagnostic figures.
        
        Returns:
            Dictionary mapping description to file paths.
        """
        figures = {}
        
        print(f"Generating diagnostic figures in {self.output_dir}...")
        
        path = self.figure_profile_comparison()
        if path:
            figures["Profile Comparison"] = path
            print(f"  ✓ {path}")
        
        path = self.figure_metabolic_energy_model()
        if path:
            figures["Energy Expenditure Model"] = path
            print(f"  ✓ {path}")
        
        for profile_name in ["low", "medium", "high"]:
            path = self.figure_fatigue_progression(profile_name=profile_name)
            if path:
                figures[f"Fatigue Progression ({profile_name})"] = path
                print(f"  ✓ {path}")
        
        path = self.figure_heart_rate_zones()
        if path:
            figures["Heart Rate Zones"] = path
            print(f"  ✓ {path}")
        
        path = self.figure_fatigue_scaling_formula()
        if path:
            figures["Fatigue Scaling Formula"] = path
            print(f"  ✓ {path}")
        
        path = self.figure_random_profile_distribution()
        if path:
            figures["Random Profile Distribution"] = path
            print(f"  ✓ {path}")
        
        print(f"\nGenerated {len(figures)} figures")
        return figures


def main():
    """Generate all diagnostic figures."""
    diagnostics = HumanFactorsDiagnostics(output_dir="./diagnostics")
    figures = diagnostics.generate_all_figures()
    
    print("\nSummary:")
    for name, path in figures.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
