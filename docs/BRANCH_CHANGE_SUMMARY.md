# Branch Change Summary

This document summarizes the human-factors work added on branch `andre/shared-layout-rendering` after trimming non-essential artifacts.

## Scope

Implemented and validated the remaining Phase 3-5 items for the picker human-factors system:

- model extensibility and fallback behavior,
- runtime diagnostics and episode summaries,
- renderer overlay integration,
- warehouse-level integration testing.

Generated image artifacts and intermediate report files were removed to keep the branch review-friendly.

## Added Files

- `tarware/human_factors.py`
  - Zhao-based physiological model and profile calibration.
  - `HumanFactorsConfig.model_name` for model selection.
  - Model registry with fallback to `zhao`.
  - Random profile generator based on physiological sampling.

- `tarware/rendering_overlays.py`
  - Overlay drawing helpers for picker fatigue visualization.
  - Uses runtime human-factors state per picker.

- `tests/test_human_factors.py`
  - Unit tests for formulas, profile calibration, config loading, state behavior, and reproducibility.

- `tests/test_warehouse_hf_integration.py`
  - Integration tests for warehouse runtime behavior.
  - Covers reset profile assignment, per-step diagnostics, terminal episode summaries, and unknown-model fallback.
  - Includes a local pathfinding stub for environments without compiled `pyastar2d`.

- `scripts/validate_zhao_calibration.py`
  - Standalone script to verify Zhao formula calculations and profile calibration sanity.

- `scripts/generate_diagnostics.py`
  - Script to generate diagnostic figures for profile behavior and fatigue progression.

## Modified Files

- `tarware/warehouse.py`
  - Plumbed terminal-step detection into info building.
  - Added `human_factors_model` to `info`.
  - Added aggregate diagnostics:
    - `picker_fatigue_mean`
    - `picker_fatigue_max`
    - `picker_energy_total`
  - Added terminal `episode_human_factors_summary` payload.

- `tarware/rendering.py`
  - Added optional overlay import.
  - Added overlay draw call in render flow after picker rendering.

## Validation

Executed:

`python -m unittest tests.test_human_factors tests.test_warehouse_hf_integration -v`

Result:

- 40 tests passed.

## Branch Trimming Performed

Removed non-essential/generated files:

- `diagnostics/*` image outputs,
- `validation_output.txt`,
- `tests/__pycache__/`,
- redundant temporary markdown reports superseded by this file.

## Notes

- Existing local modifications in `.env` and `scripts/run_heuristic.py` were intentionally left untouched.
- This summary is intended to be the single source of branch-level change context for review.