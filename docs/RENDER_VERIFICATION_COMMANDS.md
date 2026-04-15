# Render Verification Commands

These commands exercise the full TA-RWARE pipeline with different map presets, speed models, and simulation tick rates.

## 1) Run the full trial matrix (non-render, fast verification)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify_pipeline_trials.ps1
```

## 2) Run the full trial matrix with rendering enabled

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify_pipeline_trials.ps1 -Render
```

## 3) Run a single rendered trial (physical speed mode)

```powershell
$env:TARWARE_MAX_STEPS='60'
$env:TARWARE_STEPS_PER_SIMULATED_SECOND='2'
$env:TARWARE_USE_PHYSICAL_SPEEDS='1'
$env:TARWARE_GRID_CELL_SIZE_M='1.0'
$env:TARWARE_AGV_NOMINAL_SPEED_M_S='1.0'
$env:TARWARE_PICKER_NOMINAL_SPEED_M_S='0.8'
$env:TARWARE_RENDER_PHYSICAL_TIME_OVERLAY='1'
$env:TARWARE_RENDER_PICKER_DIAGNOSTICS_WINDOW='1'
python -m tarware.main classical eval --episodes 1 --seed 333 --size tiny --agvs 3 --pickers 2 --obs-type partial --render
```

## 4) Per-preset rendered commands (manual spot checks)

### tiny, cells baseline
```powershell
$env:TARWARE_MAX_STEPS='120'; $env:TARWARE_STEPS_PER_SIMULATED_SECOND='1'; $env:TARWARE_USE_PHYSICAL_SPEEDS='0'; $env:TARWARE_AGV_CELLS_PER_STEP='1.0'; $env:TARWARE_PICKER_CELLS_PER_STEP='1.0'; $env:TARWARE_RENDER_PHYSICAL_TIME_OVERLAY='1'; $env:TARWARE_RENDER_PICKER_DIAGNOSTICS_WINDOW='1'; python -m tarware.main classical eval --episodes 1 --seed 100 --size tiny --agvs 3 --pickers 2 --obs-type partial --render
```

### tiny, cells slow
```powershell
$env:TARWARE_MAX_STEPS='120'; $env:TARWARE_STEPS_PER_SIMULATED_SECOND='4'; $env:TARWARE_USE_PHYSICAL_SPEEDS='0'; $env:TARWARE_AGV_CELLS_PER_STEP='0.25'; $env:TARWARE_PICKER_CELLS_PER_STEP='0.25'; $env:TARWARE_RENDER_PHYSICAL_TIME_OVERLAY='1'; $env:TARWARE_RENDER_PICKER_DIAGNOSTICS_WINDOW='1'; python -m tarware.main classical eval --episodes 1 --seed 101 --size tiny --agvs 3 --pickers 2 --obs-type partial --render
```

### small, physical even
```powershell
$env:TARWARE_MAX_STEPS='140'; $env:TARWARE_STEPS_PER_SIMULATED_SECOND='2'; $env:TARWARE_USE_PHYSICAL_SPEEDS='1'; $env:TARWARE_GRID_CELL_SIZE_M='1.0'; $env:TARWARE_AGV_NOMINAL_SPEED_M_S='1.0'; $env:TARWARE_PICKER_NOMINAL_SPEED_M_S='1.0'; $env:TARWARE_RENDER_PHYSICAL_TIME_OVERLAY='1'; $env:TARWARE_RENDER_PICKER_DIAGNOSTICS_WINDOW='1'; python -m tarware.main classical eval --episodes 1 --seed 102 --size small --agvs 4 --pickers 2 --obs-type partial --render
```

### medium, physical mixed
```powershell
$env:TARWARE_MAX_STEPS='160'; $env:TARWARE_STEPS_PER_SIMULATED_SECOND='3'; $env:TARWARE_USE_PHYSICAL_SPEEDS='1'; $env:TARWARE_GRID_CELL_SIZE_M='0.8'; $env:TARWARE_AGV_NOMINAL_SPEED_M_S='1.2'; $env:TARWARE_PICKER_NOMINAL_SPEED_M_S='0.8'; $env:TARWARE_RENDER_PHYSICAL_TIME_OVERLAY='1'; $env:TARWARE_RENDER_PICKER_DIAGNOSTICS_WINDOW='1'; python -m tarware.main classical eval --episodes 1 --seed 103 --size medium --agvs 5 --pickers 3 --obs-type partial --render
```

### large, cells mixed
```powershell
$env:TARWARE_MAX_STEPS='180'; $env:TARWARE_STEPS_PER_SIMULATED_SECOND='5'; $env:TARWARE_USE_PHYSICAL_SPEEDS='0'; $env:TARWARE_AGV_CELLS_PER_STEP='0.5'; $env:TARWARE_PICKER_CELLS_PER_STEP='0.75'; $env:TARWARE_RENDER_PHYSICAL_TIME_OVERLAY='1'; $env:TARWARE_RENDER_PICKER_DIAGNOSTICS_WINDOW='1'; python -m tarware.main classical eval --episodes 1 --seed 104 --size large --agvs 6 --pickers 3 --obs-type partial --render
```

### extralarge, physical slow
```powershell
$env:TARWARE_MAX_STEPS='220'; $env:TARWARE_STEPS_PER_SIMULATED_SECOND='4'; $env:TARWARE_USE_PHYSICAL_SPEEDS='1'; $env:TARWARE_GRID_CELL_SIZE_M='1.0'; $env:TARWARE_AGV_NOMINAL_SPEED_M_S='0.9'; $env:TARWARE_PICKER_NOMINAL_SPEED_M_S='0.6'; $env:TARWARE_RENDER_PHYSICAL_TIME_OVERLAY='1'; $env:TARWARE_RENDER_PICKER_DIAGNOSTICS_WINDOW='1'; python -m tarware.main classical eval --episodes 1 --seed 105 --size extralarge --agvs 8 --pickers 4 --obs-type partial --render
```

## Notes

- Trial logs are written under `logs/trial_runs/`.
- The matrix script fails immediately on any non-zero trial exit code.
- Rendering includes the physical timing overlay and diagnostics window by default in these commands.
