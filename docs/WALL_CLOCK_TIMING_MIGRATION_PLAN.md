# Wall-Clock Timing Migration Plan

## Objective

Make wall-clock time the authoritative driver for both movement and human-factors computation.

Today, the simulation is still fundamentally step-driven:

- `Warehouse.step()` advances the world by one discrete simulation step.
- AGV and picker motion are executed as one cell per step, with some speed gating layered on top.
- Zhao-style human-factors updates are scaled by simulated seconds per step.
- The GUI displays real time, but real time is not the runtime source of truth.

The goal of this migration is to invert that relationship so the simulation advances according to elapsed wall-clock time, while preserving deterministic behavior as much as possible.

## Core Design Decision

Use a monotonic wall-clock source as the master timeline.

That means the simulation loop should:

1. Read the current monotonic time.
2. Compute `delta_seconds` since the previous update.
3. Feed that `delta_seconds` into motion, fatigue, recovery, delay, pick duration, and event scheduling.
4. Advance discrete state only when enough wall-clock time has accumulated.

The key point is that wall-clock time should not just be shown in the HUD. It should directly drive state progression.

## Current Constraints To Remove

The following assumptions need to be removed or rewritten:

- One environment step equals one physical step.
- One AGV or picker move is one full tile per simulation tick.
- Fatigue and recovery are scaled by step duration rather than actual elapsed time.
- Pick durations and failure delays are stored as step counts only.
- Collision handling assumes movement only occurs at discrete boundaries.
- Rendering and simulation progression are tied together too closely.

## Migration Phases

### Phase 1: Define the Timing Authority

Create a single timing source for the runtime.

Needed work:

- Introduce a monotonic clock in the top-level runtime loop.
- Store the last update timestamp.
- Compute `delta_seconds` on every tick.
- Clamp pathological deltas, for example after pauses or debugger breaks.
- Add a configurable maximum delta to avoid huge jumps.

Expected changes:

- `tarware/main.py`
- `tarware/rendering.py`
- Potentially a new runtime helper module if the logic grows

Acceptance criteria:

- The simulation can compute elapsed wall-clock time independently of render frame count.
- No game logic assumes a fixed step duration when `delta_seconds` is available.

### Phase 2: Split Simulation Time From Render Frames

Decouple render refresh frequency from simulation advancement.

Needed work:

- Stop assuming every render frame advances exactly one simulation step.
- Introduce an accumulator for elapsed wall-clock time.
- Run one or more simulation updates when the accumulator exceeds the update quantum.
- Allow rendering to happen more frequently than simulation updates, if necessary.

Two implementation options exist:

1. Fixed simulation tick with wall-clock accumulation
   - Keep a target simulation quantum in seconds.
   - Accumulate wall time and advance the environment only when enough time is present.
   - This is simpler and usually the best first step.

2. Fully variable-step simulation
   - Apply `delta_seconds` directly to all dynamics every update.
   - This is more faithful to wall-clock time but requires more invasive changes.

Recommended path:

- Start with fixed-quantum accumulation using real wall-clock time.
- Move to finer-grained variable-time mechanics only if the discrete update model becomes too limiting.

Acceptance criteria:

- Simulation progression is no longer implicitly tied to rendering frequency.
- Faster or slower GUI refresh does not change the physical behavior.

### Phase 3: Convert Motion to Time-Based Progress

This is the largest behavioral change.

Needed work:

- Replace single-tile-per-tick movement with progress accumulation measured in distance or cells.
- Express AGV and picker speed as cells per second or meters per second.
- Maintain partial progress toward the next grid cell across updates.
- Only finalize a grid-cell transition when enough wall-clock time has accumulated.

Likely model:

- Each moving entity has a floating-point progress value.
- Each update adds `speed * delta_seconds`.
- When progress crosses 1.0 cell, the entity enters the next cell and progress is reduced by 1.0.
- If speed is low, the entity may take several updates to traverse one cell.

Files likely affected:

- `tarware/warehouse.py`
- `tarware/human_factors.py`
- `tarware/rendering.py`
- `tarware/rendering_overlays.py`

Key design questions:

- Do AGVs and pickers move continuously between tiles or only in discrete jumps once a cell threshold is crossed?
- Is a picker considered to occupy the source cell, destination cell, or an interpolated position during motion?
- How should path planning behave while progress is mid-edge?

Recommended first implementation:

- Keep discrete occupancy at cell boundaries.
- Allow partial motion internally, but only update the visible cell when a threshold is crossed.
- This preserves most of the current pathing and collision logic while still respecting wall-clock speed.

Acceptance criteria:

- Lower speed causes slower arrival at the next cell.
- Higher speed causes faster arrival.
- Speeds can differ between AGVs and pickers.
- Motion remains stable under different render frame rates.

### Phase 4: Convert Human Factors To Real Elapsed Time

Zhao-style effort, fatigue, and recovery should use real elapsed time directly.

Needed work:

- Replace step-based effort accumulation with time-based effort accumulation.
- Use the actual elapsed seconds since the last update.
- Express recovery as a per-second rate and apply it using `delta_seconds`.
- Convert pick-duration logic from step counts to seconds.
- Convert movement-delay logic from step events to time-based stochastic delays.

Current issue:

- The code now scales effort and recovery by simulated seconds per step.
- That is mathematically consistent for a fixed-step simulation, but it is not the same as wall-clock authority.

Target behavior:

- If the wall-clock loop slows down, the fatigue model should advance by the actual elapsed seconds.
- If the wall-clock loop speeds up, the model should advance less per rendered frame, but still reflect elapsed time accurately.

Implementation details:

- `PickerHumanFactorsState` should likely store time-based accumulators.
- `fatigue`, `energy_expended`, `cumulative_recovery_seconds`, and delay durations should all be updated from real elapsed time.
- Any probability that is currently "per step" should become "per second" or "per update interval" derived from `delta_seconds`.

Files likely affected:

- `tarware/human_factors.py`
- `tarware/warehouse.py`

Acceptance criteria:

- Two runs with different frame rates but the same wall-clock schedule produce similar fatigue/recovery trajectories.
- The human-factors model is driven by elapsed real seconds, not update count.

### Phase 5: Rewrite Delay and Pick Timing State

Pick delays and movement delays are currently expressed in step units.

Needed work:

- Replace `pick_ticks_remaining` with a time-remaining field, likely in seconds.
- Replace delay counters with seconds remaining.
- Any temporary stall or failed-pick cooldown should be decremented by `delta_seconds`.
- Only transition state when the remaining time reaches zero.

Likely fields to revisit:

- `Picker.pick_ticks_remaining`
- `Picker.blocked_ticks`
- `Picker.fixing_clash`
- `PickerHumanFactorsState.cumulative_delay_steps`
- Any place that increments or decrements by `1` step

Recommended new representation:

- `pick_time_remaining_s`
- `movement_delay_remaining_s`
- `recovery_remaining_s` if needed
- `blocked_time_s` or `blocked_duration_s`

Acceptance criteria:

- Pick and delay durations are no longer quantized to integer steps.
- The same real duration produces the same behavior even if the runtime update frequency changes.

### Phase 6: Rework Collision, Reservation, and Reroute Logic

Once movement becomes time-based, the current conflict logic needs a careful review.

Needed work:

- Define when a cell is considered occupied during motion.
- Decide whether moving entities reserve their next cell for the entire transit duration or only near the boundary.
- Update AGV and picker conflict checks to use the new motion state.
- Preserve reroute and deadlock handling under partial motion.

Important concern:

- The current resolver assumes synchronous discrete moves.
- A time-driven system needs either stronger reservations or more formal interpolation rules.

Potential approaches:

1. Conservative reservation model
   - Reserve the source and destination cells until motion completes.
   - Simplest to reason about.
   - May reduce concurrency.

2. Time-window occupancy model
   - Each entity claims a cell for a time interval.
   - More accurate.
   - More complex.

Recommended first step:

- Implement conservative reservation semantics first.
- Improve granularity only if needed for throughput.

Acceptance criteria:

- Two entities do not silently pass through the same space because of fractional timing.
- Deadlock and rerouting logic still works after the motion refactor.

### Phase 7: Update Rendering To Show Interpolated Progress

The GUI should make wall-clock motion understandable.

Needed work:

- Show motion progress in the main warehouse view if partial movement is introduced.
- Keep the timing HUD, but make it explicit that the displayed times are wall-clock driven.
- Update picker diagnostics to show time-based delays rather than step-based counts.
- Consider showing speed, elapsed motion time, and remaining action time.

Files likely affected:

- `tarware/rendering.py`
- `tarware/rendering_overlays.py`
- `docs/GUI_ELEMENTS.md`

Acceptance criteria:

- The GUI clearly reflects time-based movement.
- Diagnostics match the real-time state of the simulation.

### Phase 8: Update Configuration Surface

Add explicit controls for wall-clock timing.

Likely new or revised settings:

- Maximum delta clamp
- Target simulation update quantum
- AGV speed in m/s
- Picker speed in m/s
- Human-factors update rates expressed in seconds
- Optional fixed-step compatibility mode

Potential environment variables:

- `TARWARE_USE_WALL_CLOCK_TIMING`
- `TARWARE_MAX_DELTA_SECONDS`
- `TARWARE_TARGET_UPDATE_SECONDS`
- `TARWARE_AGV_NOMINAL_SPEED_M_S`
- `TARWARE_PICKER_NOMINAL_SPEED_M_S`
- `TARWARE_REAL_SECONDS_PER_SIM_SECOND`

Acceptance criteria:

- A user can configure the real-time behavior without editing code.
- Default behavior remains sensible and documented.

### Phase 9: Replace or Rework Tests

The current tests validate step-based and simulated-time behavior.

Needed work:

- Add wall-clock driven tests for motion and fatigue.
- Add tests that simulate different update cadences.
- Keep existing Zhao-model unit tests, but reinterpret them around elapsed real seconds.
- Add integration tests for update-interval independence.

New test categories:

- Same physical duration, different frame cadence, same fatigue outcome
- Same physical duration, different frame cadence, same traversal distance
- Delay state decreases by elapsed seconds, not count
- Rendering mode remains stable under variable update cadence

Acceptance criteria:

- The new wall-clock architecture is covered by tests.
- Regression tests fail if behavior accidentally reverts to step-driven timing.

### Phase 10: Deprecate Step-Driven Assumptions

Once the new model is stable, remove the old assumptions cleanly.

Needed work:

- Remove or isolate any code path that still assumes one update equals one time step.
- Remove step-counter logic that is now redundant.
- Simplify diagnostics so they report actual elapsed time where appropriate.
- Update docs to avoid implying that steps are the authoritative timing source.

Acceptance criteria:

- No core simulation logic depends on fixed step duration unless explicitly configured as compatibility mode.

## Risk Areas

### 1. Path and collision semantics

This is the highest-risk area because the current warehouse logic is very discrete. Introducing wall-clock motion without rethinking occupancy can create subtle collision bugs.

### 2. Human-factors time consistency

Zhao-style formulas are time-based, but the current implementation mixes simulated seconds and step counts in several places. That needs a careful pass to avoid double-scaling or under-scaling.

### 3. GUI/simulation desynchronization

If rendering is decoupled from simulation progression, the GUI must not imply state changes that have not actually occurred.

### 4. Regression stability

Any change to timing semantics can affect throughput, delays, and deadlock patterns. Expect existing episodes to change materially.

## Suggested Implementation Order

1. Add wall-clock timing infrastructure in the runtime loop.
2. Keep discrete simulation logic, but drive it from elapsed real time.
3. Convert human-factors updates to use `delta_seconds`.
4. Convert pick/delay counters to seconds.
5. Rework AGV and picker motion to use real-time progress accumulation.
6. Revisit collision handling and reservations.
7. Update GUI and diagnostics.
8. Add regression tests.
9. Remove old step-driven compatibility paths once stable.

## Definition of Done

The migration is complete when:

- Wall-clock time is the source of truth for simulation advancement.
- AGV and picker movement rates are physically meaningful.
- Zhao-style fatigue, recovery, energy, and delays update according to elapsed real time.
- Simulation output is stable across different render frame rates.
- Tests validate time invariance and motion correctness.
- Documentation clearly distinguishes wall-clock authority from any compatibility step mode.

## Recommendation

This should be treated as a major architecture change, not a local fix.

The minimum safe approach is to keep the existing warehouse logic intact while introducing a real-time accumulator and time-based state fields first. After that, move motion and collision semantics incrementally. The motion/collision refactor is the part most likely to require substantial code churn.
