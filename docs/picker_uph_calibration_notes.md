# Picker 150 UPH Calibration Notes

## Goal

Calibrate a single picker to approximately 150 UPH on a medium warehouse map while using a high AGV count (oversaturated AGV supply).

## Framework

Script: `simulation/tarware/scripts/calibrate_picker_uph.py`

What it does:
- Runs heuristic policy episodes on `medium_dhl` map.
- Forces `num_pickers=1` and configurable AGV count.
- Sweeps `_PICK_TICKS` (via `TARWARE_PICK_BASE_TICKS`) and human-factors profile overrides.
- Outputs CSV + Markdown summaries.

Default outputs:
- `simulation/tarware/docs/picker_uph_calibration_results.csv`
- `simulation/tarware/docs/picker_uph_calibration_findings.md`

## Key findings from runs

All runs below used one picker and `max_steps=1800` unless noted.

| AGVs | pick_ticks | pick_base_seconds | HF preset | extra settings | observed UPH |
|---:|---:|---:|---|---|---:|
| 40 | 2 | 2.0 | low_slow | picker speed 0.5 m/s | 282 |
| 40 | 4 | 5.0 | low_slow | picker speed 0.5 m/s, quantity extra 8.0 s | 236 |
| 40 | 8 | 10.0 | low_slow | picker speed 0.4 m/s, quantity extra 15.0 s | 190 |
| 40 | 10 | 12.0 | low_slow | picker speed 0.35 m/s, quantity extra 18.0 s | 172 |
| 40 | 11 | 13.0 | low_slow | picker speed 0.32 m/s, quantity extra 20.0 s | 164 |

Target band was 150 +/- 20 UPH. The best observed point in-band was:

- AGVs: 40
- `_PICK_TICKS`: 11
- `TARWARE_PICK_BASE_SECONDS`: 13.0
- `TARWARE_PICKER_NOMINAL_SPEED_M_S`: 0.32
- `TARWARE_PICK_QUANTITY_EXTRA_SECONDS`: 20.0
- HF preset: `low_slow`
- Observed throughput: 164 UPH

## AGV saturation note

AGV count was highly non-monotonic in this setup due to traffic effects. For the same picker calibration profile, 40 AGVs sustained high throughput while 32 and 48 AGVs produced severe drops in one-seed tests. In practice, use 40 AGVs for this calibration target and verify with multi-seed runs before locking a production default.

## Example command (near-target profile)

From `simulation/tarware` in PowerShell:

```powershell
$env:TARWARE_PICKER_NOMINAL_SPEED_M_S="0.32"
$env:TARWARE_PICK_QUANTITY_EXTRA_SECONDS="20.0"
python scripts\calibrate_picker_uph.py --agvs 40 --pick-ticks 11 --pick-base-seconds 13.0 --hf-presets low_slow --seeds 0 --max-steps 1800
```
