# GUI Elements Added

This branch added a few runtime display elements to make simulation behavior easier to inspect.

## Main window timing HUD

A small overlay now appears in the main render window showing:

- Current simulation step
- Elapsed simulated time
- Elapsed real time
- The current speed model in use
- Effective AGV and picker movement rates per step

This is useful when changing `steps_per_simulated_second` or switching between physical-speed and cells-per-step modes.

## Picker diagnostics window

A separate side window now shows live picker runtime state, including:

- Picker state and position
- Blocking and stall status
- Current shelf/order information
- Human factors profile, fatigue, energy, and delay metrics

This window is meant for side-by-side debugging while the main warehouse animation is running.

## Human factors overlays

The map view still includes picker-level human factors overlays, such as:

- Fatigue rings around pickers
- Fatigue progress bars above pickers

These overlays help visualize how picker load changes over time during the simulation.

## Environment toggles

The main display features can be controlled with environment variables:

- `TARWARE_RENDER_PHYSICAL_TIME_OVERLAY`
- `TARWARE_RENDER_PICKER_DIAGNOSTICS_WINDOW`
- `TARWARE_RENDER_HUMAN_FACTORS_OVERLAY`
