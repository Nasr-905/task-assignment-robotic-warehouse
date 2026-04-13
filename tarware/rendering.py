"""2D rendering of the warehouse environment using pyglet."""

import math
import os
import sys

import numpy as np
import six
from gymnasium import error

from tarware.warehouse import AgentType, Direction

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import gl
except ImportError as e:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_LIGHTORANGE = (255, 200, 0)
_RED = (255, 0, 0)
_ORANGE = (255, 165, 0)
_DARKORANGE = (255, 140, 0)
_DARKSLATEBLUE = (72, 61, 139)
_TEAL = (0, 128, 128)
_MAROON = (128, 0, 0)
_BLUE = (30,144,255)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK
_SHELF_COLOR = _DARKSLATEBLUE
_SHELF_REQ_COLOR = _TEAL
_AGENT_COLOR = _DARKORANGE
_AGENT_LOADED_COLOR = _RED
_AGENT_DIR_COLOR = _BLACK
_GOAL_COLOR = (60, 60, 60)
_SHELF_FULFILLED_COLOR = (0, 160, 80)   # green: pickerwall shelf fully picked, ready to displace
_CARRIER_COLOR = _MAROON
_LOADER_AGENT = _BLUE

_SHELF_LOC_COLOR = (200, 200, 200)    # light gray for shelf location markers
_PICKER_ZONE_COLOR = (170, 230, 255)   # light blue background for picker zone
_SHARED_HIGHWAY_COLOR = (205, 235, 205)  # green-blue tint: shared AGV/picker aisle
_REPLENISHMENT_ZONE_COLOR = (220, 255, 220)  # light green background for replenishment zone
_REPLENISHMENT_SHELF_COLOR = (50, 200, 50)   # bright green: fresh stock shelf in replenishment zone
_DEPLETED_SHELF_COLOR = (180, 60, 60)        # muted red: shelf with exhausted stock
_PACKAGING_COLOR = (255, 215, 0)       # gold for packaging locations (idle)
_PACKAGING_PARTIAL_COLOR = (255, 140, 0)   # dark orange: station has partial orders in progress
_PACKAGING_FILL_COLOR = (200, 80, 0)       # deep orange: filled portion of in-progress bar
_PICKER_COLOR = (0, 180, 90)           # green for idle/walking pickers
_PICKER_PICKING_COLOR = (255, 100, 0)  # orange while actively picking

_SHELF_PADDING = 2


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols = world_size

        self.grid_size = int(os.getenv("TARWARE_RENDER_TILE_SIZE", "30"))
        self.icon_size = max(4, int(self.grid_size * 2 / 3))

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def render(self, env, return_rgb_array=False):
        gl.glClearColor(*_BACKGROUND_COLOR, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_shelf_locs(env)
        self._draw_picker_zone(env)
        self._draw_shared_highways(env)
        self._draw_replenishment_zone(env)
        self._draw_goals(env)
        self._draw_packaging(env)
        self._draw_shelfs(env)
        self._draw_agents(env)
        self._draw_pickers(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        # VERTICAL LINES
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,  # LEFT X
                        (self.grid_size + 1) * r + 1,  # Y
                        (self.grid_size + 1) * self.cols,  # RIGHT X
                        (self.grid_size + 1) * r + 1,  # Y
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )

        # HORIZONTAL LINES
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * c + 1,  # X
                        0,  # BOTTOM Y
                        (self.grid_size + 1) * c + 1,  # X
                        (self.grid_size + 1) * self.rows,  # TOP Y
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )
        batch.draw()

    def _draw_shelf_locs(self, env):
        if not hasattr(env, "shelf_locs") or not env.shelf_locs:
            return
        batch = pyglet.graphics.Batch()
        gs = self.grid_size + 1
        for row, col in env.shelf_locs:
            y = self.rows - row - 1
            batch.add(
                4, gl.GL_QUADS, None,
                ("v2f", (
                    gs * col + 1,        gs * y + 1,
                    gs * (col + 1),      gs * y + 1,
                    gs * (col + 1),      gs * (y + 1),
                    gs * col + 1,        gs * (y + 1),
                )),
                ("c3B", 4 * _SHELF_LOC_COLOR),
            )
        batch.draw()

    def _draw_shelfs(self, env):
        batch = pyglet.graphics.Batch()

        for shelf in env.shelfs:
            if not shelf.on_grid:
                continue
            x, y = shelf.x, shelf.y
            y = self.rows - y - 1  # pyglet rendering is reversed
            if shelf.depleted:
                shelf_color = _DEPLETED_SHELF_COLOR
            elif shelf.from_replenishment:
                shelf_color = _REPLENISHMENT_SHELF_COLOR
            else:
                shelf_color = _SHELF_REQ_COLOR if shelf in env.request_queue else _SHELF_COLOR

            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + _SHELF_PADDING + 1,  # TL - X
                        (self.grid_size + 1) * y + _SHELF_PADDING + 1,  # TL - Y
                        (self.grid_size + 1) * (x + 1) - _SHELF_PADDING,  # TR - X
                        (self.grid_size + 1) * y + _SHELF_PADDING + 1,  # TR - Y
                        (self.grid_size + 1) * (x + 1) - _SHELF_PADDING,  # BR - X
                        (self.grid_size + 1) * (y + 1) - _SHELF_PADDING,  # BR - Y
                        (self.grid_size + 1) * x + _SHELF_PADDING + 1,  # BL - X
                        (self.grid_size + 1) * (y + 1) - _SHELF_PADDING,  # BL - Y
                    ),
                ),
                ("c3B", 4 * shelf_color),
            )
        batch.draw()

        # Draw capacity labels on each shelf
        for shelf in env.shelfs:
            if not shelf.on_grid:
                continue
            sx, sy = shelf.x, shelf.y
            sy = self.rows - sy - 1
            cx = (self.grid_size + 1) * sx + self.grid_size // 2 + 1
            cy = (self.grid_size + 1) * sy + self.grid_size // 2 + 1
            label = pyglet.text.Label(
                str(shelf.capacity),
                font_name="Arial",
                font_size=7,
                x=cx, y=cy,
                anchor_x="center", anchor_y="center",
                color=(255, 255, 255, 255),
            )
            label.draw()

    def _draw_goals(self, env):
        batch = pyglet.graphics.Batch()

        fulfilled_positions = set()
        if hasattr(env, "shelfs") and hasattr(env, "grid"):
            from tarware.definitions import CollisionLayers
            for gx, gy in env.goals:
                shelf_id = env.grid[CollisionLayers.SHELVES, gy, gx]
                if shelf_id != 0:
                    shelf = env.shelfs[shelf_id - 1]
                    if shelf.fulfilled:
                        fulfilled_positions.add((gx, gy))

        for goal in env.goals:
            x, y = goal
            color = _SHELF_FULFILLED_COLOR if (x, y) in fulfilled_positions else _GOAL_COLOR
            y = self.rows - y - 1  # pyglet rendering is reversed
            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * x + 1,  # TL - X
                        (self.grid_size + 1) * y + 1,  # TL - Y
                        (self.grid_size + 1) * (x + 1),  # TR - X
                        (self.grid_size + 1) * y + 1,  # TR - Y
                        (self.grid_size + 1) * (x + 1),  # BR - X
                        (self.grid_size + 1) * (y + 1),  # BR - Y
                        (self.grid_size + 1) * x + 1,  # BL - X
                        (self.grid_size + 1) * (y + 1),  # BL - Y
                    ),
                ),
                ("c3B", 4 * color),
            )
        batch.draw()

    def _draw_picker_zone(self, env):
        if not hasattr(env, "picker_highways") or not env.picker_highways.any():
            return
        batch = pyglet.graphics.Batch()
        gs = self.grid_size + 1
        for row in range(env.grid_size[0]):
            y = self.rows - row - 1  # pyglet reversed
            for col in range(env.grid_size[1]):
                if env.picker_highways[row, col] != 1:
                    continue
                batch.add(
                    4, gl.GL_QUADS, None,
                    ("v2f", (
                        gs * col + 1,        gs * y + 1,
                        gs * (col + 1),      gs * y + 1,
                        gs * (col + 1),      gs * (y + 1),
                        gs * col + 1,        gs * (y + 1),
                    )),
                    ("c3B", 4 * _PICKER_ZONE_COLOR),
                )
        batch.draw()

    def _draw_shared_highways(self, env):
        if not hasattr(env, "shared_highway_locs") or not env.shared_highway_locs:
            return
        batch = pyglet.graphics.Batch()
        gs = self.grid_size + 1
        for col, row in env.shared_highway_locs:
            y = self.rows - row - 1
            batch.add(
                4, gl.GL_QUADS, None,
                ("v2f", (
                    gs * col + 1,        gs * y + 1,
                    gs * (col + 1),      gs * y + 1,
                    gs * (col + 1),      gs * (y + 1),
                    gs * col + 1,        gs * (y + 1),
                )),
                ("c3B", 4 * _SHARED_HIGHWAY_COLOR),
            )
        batch.draw()

    def _draw_replenishment_zone(self, env):
        if not hasattr(env, "replenishment_locs") or not env.replenishment_locs:
            return
        batch = pyglet.graphics.Batch()
        gs = self.grid_size + 1
        for (rx, ry) in env.replenishment_locs:
            y = self.rows - ry - 1
            batch.add(
                4, gl.GL_QUADS, None,
                ("v2f", (
                    gs * rx + 1,        gs * y + 1,
                    gs * (rx + 1),      gs * y + 1,
                    gs * (rx + 1),      gs * (y + 1),
                    gs * rx + 1,        gs * (y + 1),
                )),
                ("c3B", 4 * _REPLENISHMENT_ZONE_COLOR),
            )
        batch.draw()

    def _draw_packaging(self, env):
        if not hasattr(env, "packaging_locations"):
            return

        partial_slots = []
        if hasattr(env, "_packaging_slots"):
            partial_slots = sorted(
                ((order_num, s["delivered"], s["required"])
                 for order_num, s in env._packaging_slots.items()
                 if s["required"] > 0),
                key=lambda t: t[0],
            )

        gs = self.grid_size + 1
        pad = _SHELF_PADDING

        for i, (px, py) in enumerate(env.packaging_locations):
            y = self.rows - py - 1

            x0 = gs * px + pad + 1
            x1 = gs * (px + 1) - pad
            y0 = gs * y + pad + 1
            y1 = gs * (y + 1) - pad

            # Each square holds at most one partial order (assigned by sorted index)
            order_info = partial_slots[i] if i < len(partial_slots) else None

            base_color = _PACKAGING_PARTIAL_COLOR if order_info else _PACKAGING_COLOR
            base_batch = pyglet.graphics.Batch()
            base_batch.add(
                4, gl.GL_QUADS, None,
                ("v2f", (x0, y0, x1, y0, x1, y1, x0, y1)),
                ("c3B", 4 * base_color),
            )
            base_batch.draw()

            if order_info:
                _order_num, delivered, required = order_info
                fill_ratio = delivered / required if required > 0 else 0.0

                if fill_ratio > 0:
                    # Vertical progress bar filling from the bottom up
                    fill_y1 = y0 + (y1 - y0) * fill_ratio
                    fill_batch = pyglet.graphics.Batch()
                    fill_batch.add(
                        4, gl.GL_QUADS, None,
                        ("v2f", (x0, y0, x1, y0, x1, fill_y1, x0, fill_y1)),
                        ("c3B", 4 * _PACKAGING_FILL_COLOR),
                    )
                    fill_batch.draw()

                # Label: "delivered/required" centred in the square
                cx = (x0 + x1) // 2
                cy = (y0 + y1) // 2
                pyglet.text.Label(
                    f"{delivered}/{required}",
                    font_name="Arial",
                    font_size=6,
                    bold=True,
                    x=cx, y=cy,
                    anchor_x="center", anchor_y="center",
                    color=(255, 255, 255, 255),
                ).draw()

    def _draw_pickers(self, env):
        if not hasattr(env, "pickers") or not env.pickers:
            return
        from tarware.warehouse import PickerState
        radius = self.grid_size / 3
        for picker in env.pickers:
            col, row = picker.x, picker.y
            row = self.rows - row - 1
            cx = (self.grid_size + 1) * col + self.grid_size // 2 + 1
            cy = (self.grid_size + 1) * row + self.grid_size // 2 + 1
            # Diamond shape (4 vertices)
            verts = [
                cx, cy + radius,    # top
                cx + radius, cy,    # right
                cx, cy - radius,    # bottom
                cx - radius, cy,    # left
            ]
            diamond = pyglet.graphics.vertex_list(4, ("v2f", verts))
            color = _PICKER_PICKING_COLOR if picker.state == PickerState.PICKING else _PICKER_COLOR
            gl.glColor3ub(*color)
            diamond.draw(gl.GL_POLYGON)

    def _draw_agents(self, env):
        agents = []
        batch = pyglet.graphics.Batch()

        radius = self.grid_size / 3

        for agent in env.agents:

            col, row = agent.x, agent.y
            row = self.rows - row - 1  # pyglet rendering is reversed
            
            if agent.type == AgentType.AGV:
                resolution = 6
            elif agent.type == AgentType.PICKER:
                resolution = 4
            elif agent.type == AgentType.AGENT:
                resolution = 8
            else:
                raise ValueError("Agent type not recognized by environment.")
            
            verts = []
            for i in range(resolution):
                angle = 2 * math.pi * i / resolution
                x = (
                    radius * math.cos(angle)
                    + (self.grid_size + 1) * col
                    + self.grid_size // 2
                    + 1
                )
                y = (
                    radius * math.sin(angle)
                    + (self.grid_size + 1) * row
                    + self.grid_size // 2
                    + 1
                )
                verts += [x, y]
            circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))

            draw_color = _AGENT_LOADED_COLOR if agent.carrying_shelf else _AGENT_COLOR

            gl.glColor3ub(*draw_color)
            circle.draw(gl.GL_POLYGON)

        for agent in env.agents:

            col, row = agent.x, agent.y
            row = self.rows - row - 1  # pyglet rendering is reversed

            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * col
                        + self.grid_size // 2
                        + 1,  # CENTER X
                        (self.grid_size + 1) * row
                        + self.grid_size // 2
                        + 1,  # CENTER Y
                        (self.grid_size + 1) * col
                        + self.grid_size // 2
                        + 1
                        + (
                            radius if agent.dir.value == Direction.RIGHT.value else 0
                        )  # DIR X
                        + (
                            -radius if agent.dir.value == Direction.LEFT.value else 0
                        ),  # DIR X
                        (self.grid_size + 1) * row
                        + self.grid_size // 2
                        + 1
                        + (
                            radius if agent.dir.value == Direction.UP.value else 0
                        )  # DIR Y
                        + (
                            -radius if agent.dir.value == Direction.DOWN.value else 0
                        ),  # DIR Y
                    ),
                ),
                ("c3B", (*_AGENT_DIR_COLOR, *_AGENT_DIR_COLOR)),
            )
        batch.draw()

    def _draw_badge(self, row, col, level):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * self.grid_size + (3 / 4) * self.grid_size
        badge_y = self.height - self.grid_size * (row + 1) + (1 / 4) * self.grid_size

        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        gl.glColor3ub(*_BLACK)
        circle.draw(gl.GL_POLYGON)
        gl.glColor3ub(*_WHITE)
        circle.draw(gl.GL_LINE_LOOP)
        label = pyglet.text.Label(
            str(level),
            font_name="Times New Roman",
            font_size=12,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
        )
        label.draw()
