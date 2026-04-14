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

