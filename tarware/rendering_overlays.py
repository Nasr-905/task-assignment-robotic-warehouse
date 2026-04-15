"""Rendering overlays for human factors model visualization.

This module provides utilities for rendering human factors overlays on top of
the warehouse visualization, including fatigue indicators, energy bars, and
work state visualization.

Integration Guide:
- Import this module in tarware/rendering.py
- Call draw_human_factors_overlays(env, viewer, gl) from Viewer.render() after _draw_pickers()
- Add TARWARE_RENDER_HUMAN_FACTORS_OVERLAY env var to enable/disable overlays
"""

import os
import math

try:
    import pyglet
    from pyglet.gl import gl
    PYGLET_AVAILABLE = True
except ImportError:
    PYGLET_AVAILABLE = False


# Color constants for overlays
_FATIGUE_LOW = (50, 200, 50)       # Green: low fatigue
_FATIGUE_MEDIUM = (255, 200, 50)   # Yellow: medium fatigue
_FATIGUE_HIGH = (255, 100, 50)     # Orange: moderate fatigue
_FATIGUE_CRITICAL = (255, 50, 50)  # Red: critical fatigue

_ENERGY_COLOR = (100, 150, 255)     # Blue for energy indicators
_OVERLAY_ALPHA = 0.7


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def draw_physical_time_overlay(env, viewer, gl_module) -> None:
    """Render physical simulation timing on the main render window."""
    if not PYGLET_AVAILABLE:
        return

    enabled = os.getenv("TARWARE_RENDER_PHYSICAL_TIME_OVERLAY", "1").lower() in (
        "1", "true", "yes"
    )
    if not enabled:
        return

    steps = int(getattr(env, "_cur_steps", 0))
    simulated_seconds_per_step = float(
        getattr(getattr(env, "time_config", None), "simulated_seconds_per_step", 0.0)
    )
    real_seconds_per_step = float(
        getattr(getattr(env, "time_config", None), "real_seconds_per_step", 0.0)
    )

    simulated_seconds = float(steps) * simulated_seconds_per_step
    real_seconds = float(steps) * real_seconds_per_step
    agv_cells_per_step = float(getattr(env, "_agv_cells_per_step_effective", 1.0))
    picker_cells_per_step = float(getattr(env, "_picker_cells_per_step_effective", 1.0))
    speed_model = "physical" if bool(getattr(env, "_use_physical_speed_model", False)) else "cells"

    lines = [
        "Physical Timing",
        f"step: {steps}",
        f"sim time: {_format_duration(simulated_seconds)} ({simulated_seconds:.1f}s)",
        f"real time: {_format_duration(real_seconds)} ({real_seconds:.1f}s)",
        (
            "step scale: "
            f"sim={simulated_seconds_per_step:.3f}s "
            f"real={real_seconds_per_step:.3f}s"
        ),
        f"speed model: {speed_model}",
        f"move rate: agv={agv_cells_per_step:.3f} picker={picker_cells_per_step:.3f} cells/step",
    ]

    panel_margin = 8
    line_height = 13
    panel_width = 470
    panel_height = line_height * len(lines) + 10
    x1 = viewer.width - panel_margin
    x0 = max(panel_margin, x1 - panel_width)
    y1 = viewer.height - panel_margin
    y0 = max(panel_margin, y1 - panel_height)

    bg = pyglet.graphics.vertex_list(4, ("v2f", [
        x0, y0,
        x1, y0,
        x1, y1,
        x0, y1,
    ]))
    gl_module.glColor4ub(10, 16, 22, 215)
    bg.draw(gl_module.GL_POLYGON)

    border = pyglet.graphics.vertex_list(4, ("v2f", [
        x0, y0,
        x1, y0,
        x1, y1,
        x0, y1,
    ]))
    gl_module.glColor4ub(175, 200, 220, 255)
    border.draw(gl_module.GL_LINE_LOOP)

    y = y1 - 5
    for idx, text in enumerate(lines):
        color = (235, 235, 235, 255)
        if idx == 0:
            color = (255, 255, 255, 255)
        label = pyglet.text.Label(
            text,
            font_name="Courier New",
            font_size=10,
            x=x0 + 7,
            y=y,
            anchor_x="left",
            anchor_y="top",
            color=color,
        )
        label.draw()
        y -= line_height


def draw_picker_diagnostics_dashboard(env, viewer, gl_module) -> None:
    """Render a small real-time dashboard with picker state diagnostics."""
    if not PYGLET_AVAILABLE:
        return

    enabled = os.getenv("TARWARE_RENDER_PICKER_DIAGNOSTICS", "1").lower() in ("1", "true", "yes")
    if not enabled:
        return

    diagnostics = getattr(env, "_latest_picker_diagnostics", None)
    if not diagnostics:
        return

    lines = [
        "Picker diagnostics",
        "id state pos path blk stall reason",
    ]
    for row in diagnostics:
        state = str(row.get("state", ""))
        reason = str(row.get("reason_code", ""))
        line = (
            f"{row.get('picker_id', -1):>2} "
            f"{state[:10]:<10} "
            f"({row.get('x', -1):>2},{row.get('y', -1):>2}) "
            f"{row.get('path_len', -1):>3} "
            f"{row.get('blocked_ticks', -1):>3} "
            f"{int(bool(row.get('stalled', False))):>5} "
            f"{reason[:28]}"
        )
        lines.append(line)

    panel_margin = 6
    line_height = 12
    panel_width = min(viewer.width - panel_margin * 2, 560)
    panel_height = line_height * len(lines) + 8
    x0 = panel_margin
    x1 = x0 + panel_width
    y1 = viewer.height - panel_margin
    y0 = max(panel_margin, y1 - panel_height)

    bg = pyglet.graphics.vertex_list(4, ("v2f", [
        x0, y0,
        x1, y0,
        x1, y1,
        x0, y1,
    ]))
    gl_module.glColor4ub(15, 20, 25, 210)
    bg.draw(gl_module.GL_POLYGON)

    border = pyglet.graphics.vertex_list(4, ("v2f", [
        x0, y0,
        x1, y0,
        x1, y1,
        x0, y1,
    ]))
    gl_module.glColor4ub(180, 180, 180, 255)
    border.draw(gl_module.GL_LINE_LOOP)

    y = y1 - 4
    for idx, text in enumerate(lines):
        color = (230, 230, 230, 255) if idx != 0 else (255, 255, 255, 255)
        label = pyglet.text.Label(
            text,
            font_name="Courier New",
            font_size=9,
            x=x0 + 6,
            y=y,
            anchor_x="left",
            anchor_y="top",
            color=color,
        )
        label.draw()
        y -= line_height


def draw_fatigue_ring(
    cx: float,
    cy: float,
    radius: float,
    fatigue_ratio: float,
    gl_module,
    resolution: int = 32,
) -> None:
    """Draw a fatigue indicator ring around a picker.
    
    Args:
        cx: Center x coordinate
        cy: Center y coordinate
        radius: Ring radius
        fatigue_ratio: Fatigue as ratio 0.0–1.0
        resolution: Number of vertices for ring
    """
    if not PYGLET_AVAILABLE:
        return
    
    # Color gradient based on fatigue
    if fatigue_ratio < 0.33:
        color = _FATIGUE_LOW
    elif fatigue_ratio < 0.67:
        color = _FATIGUE_MEDIUM
    elif fatigue_ratio < 0.95:
        color = _FATIGUE_HIGH
    else:
        color = _FATIGUE_CRITICAL
    
    # Draw ring (outer circle)
    verts = []
    for i in range(resolution):
        angle = 2 * math.pi * i / resolution
        x = radius * math.cos(angle) + cx
        y = radius * math.sin(angle) + cy
        verts += [x, y]
    
    ring = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
    gl_module.glColor4ub(color[0], color[1], color[2], int(255 * _OVERLAY_ALPHA))
    ring.draw(gl_module.GL_LINE_LOOP)


def draw_fatigue_bar(
    cx: float,
    cy: float,
    bar_width: float,
    bar_height: float,
    fatigue_ratio: float,
    gl_module,
    vertical: bool = True,
) -> None:
    """Draw a fatigue progress bar.
    
    Args:
        cx: Center x coordinate
        cy: Center y coordinate
        bar_width: Bar width
        bar_height: Bar height
        fatigue_ratio: Fatigue as ratio 0.0–1.0
        vertical: If True, bar fills from bottom; if False, from left
    """
    if not PYGLET_AVAILABLE:
        return
    
    # Background (empty bar)
    x = cx - bar_width / 2
    y = cy - bar_height / 2
    verts = [
        x, y,
        x + bar_width, y,
        x + bar_width, y + bar_height,
        x, y + bar_height,
    ]
    bg = pyglet.graphics.vertex_list(4, ("v2f", verts))
    gl_module.glColor4ub(200, 200, 200, int(255 * _OVERLAY_ALPHA * 0.5))
    bg.draw(gl_module.GL_POLYGON)
    
    # Filled portion (fatigue indicator)
    if vertical:
        fill_height = bar_height * fatigue_ratio
        verts = [
            x, y,
            x + bar_width, y,
            x + bar_width, y + fill_height,
            x, y + fill_height,
        ]
    else:
        fill_width = bar_width * fatigue_ratio
        verts = [
            x, y,
            x + fill_width, y,
            x + fill_width, y + bar_height,
            x, y + bar_height,
        ]
    
    # Color by fatigue level
    if fatigue_ratio < 0.33:
        color = _FATIGUE_LOW
    elif fatigue_ratio < 0.67:
        color = _FATIGUE_MEDIUM
    elif fatigue_ratio < 0.95:
        color = _FATIGUE_HIGH
    else:
        color = _FATIGUE_CRITICAL
    
    fill = pyglet.graphics.vertex_list(4, ("v2f", verts))
    gl_module.glColor4ub(color[0], color[1], color[2], int(255 * _OVERLAY_ALPHA))
    fill.draw(gl_module.GL_POLYGON)
    
    # Border
    frame = pyglet.graphics.vertex_list(4, ("v2f", [
        x, y,
        x + bar_width, y,
        x + bar_width, y + bar_height,
        x, y + bar_height,
    ]))
    gl_module.glColor4ub(0, 0, 0, 255)
    frame.draw(gl_module.GL_LINE_LOOP)


def draw_human_factors_overlays(env, viewer, gl_module) -> None:
    """Draw human factors overlays for all pickers.
    
    Called from Viewer.render() after drawing pickers.
    
    Args:
        env: Warehouse environment
        viewer: Viewer instance (for grid_size, rows, etc.)
        gl_module: pyglet.gl.gl module
    """
    if not PYGLET_AVAILABLE:
        return
    
    # Check if overlays are enabled
    overlay_enabled = os.getenv("TARWARE_RENDER_HUMAN_FACTORS_OVERLAY", "1").lower() in ("1", "true", "yes")
    if not overlay_enabled:
        return
    
    # Check if environment has human factors info
    if not hasattr(env, "pickers") or not env.pickers:
        return
    
    if not hasattr(env, "_picker_hf_state_by_id"):
        return
    
    # Draw overlay for each picker
    for picker in env.pickers:
        state = env._picker_hf_state_by_id.get(picker.id)
        if state is None:
            continue
        fatigue = float(state.fatigue)
        fatigue_min = float(getattr(env.human_factors_config, "fatigue_min", 0.0))
        fatigue_max = float(getattr(env.human_factors_config, "fatigue_max", 100.0))
        span = max(1e-6, fatigue_max - fatigue_min)
        
        # Normalize fatigue to 0–1 ratio
        fatigue_ratio = min(1.0, max(0.0, (fatigue - fatigue_min) / span))
        
        # Position
        col, row = picker.x, picker.y
        row = viewer.rows - row - 1  # Reverse for pyglet
        cx = (viewer.grid_size + 1) * col + viewer.grid_size // 2 + 1
        cy = (viewer.grid_size + 1) * row + viewer.grid_size // 2 + 1
        
        # Draw fatigue ring
        ring_radius = viewer.grid_size / 2.5
        draw_fatigue_ring(cx, cy, ring_radius, fatigue_ratio, gl_module)
        
        # Draw fatigue bar (above picker)
        bar_y = cy + viewer.grid_size // 2 + 5
        draw_fatigue_bar(
            cx,
            bar_y,
            viewer.grid_size * 0.6,
            4,
            fatigue_ratio,
            gl_module,
            vertical=False,
        )

    # Picker diagnostics are rendered in a dedicated side window by Viewer.


def get_fatigue_color(fatigue_ratio: float) -> tuple:
    """Get RGB color for fatigue level.
    
    Args:
        fatigue_ratio: Fatigue as ratio 0.0–1.0
    
    Returns:
        (R, G, B) tuple
    """
    if fatigue_ratio < 0.33:
        return _FATIGUE_LOW
    elif fatigue_ratio < 0.67:
        return _FATIGUE_MEDIUM
    elif fatigue_ratio < 0.95:
        return _FATIGUE_HIGH
    else:
        return _FATIGUE_CRITICAL

