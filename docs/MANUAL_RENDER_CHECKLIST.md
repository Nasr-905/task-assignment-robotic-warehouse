# Manual Render Checklist

Use these commands to start a rendered simulation and watch for visual or runtime issues.

## 1) Default quick check

```powershell
$env:TARWARE_RENDER_PHYSICAL_TIME_OVERLAY='1'
$env:TARWARE_RENDER_PICKER_DIAGNOSTICS_WINDOW='1'
python -m tarware.main classical eval --episodes 1 --seed 21 --size tiny --agvs 3 --pickers 2 --obs-type partial --render
```

## 2) Slightly heavier check

```powershell
$env:TARWARE_RENDER_PHYSICAL_TIME_OVERLAY='1'
$env:TARWARE_RENDER_PICKER_DIAGNOSTICS_WINDOW='1'
python -m tarware.main classical eval --episodes 1 --seed 22 --size medium --agvs 5 --pickers 3 --obs-type partial --render
```

## What to watch for

- The main warehouse window should draw normally without flickering.
- The physical timing HUD should appear in the main window.
- The picker diagnostics window should open beside the main window.
- The simulation should complete and close cleanly.
