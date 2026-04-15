import os
from pathlib import Path

import gymnasium as gym

from tarware.warehouse import RewardType

# CSV Path to the warehouse map data to be used in the environment. Can be set via 
# the TARWARE_MAP_NAME environment variable, or defaults to "medium" in the 
# data/maps directory.
_MAP_NAME = os.getenv("TARWARE_MAP_NAME", "medium")
_MAP_CSV_FILENAME = f"{_MAP_NAME}.csv"
_MAP_JSON_FILENAME = f"{_MAP_NAME}.json"
_MAP_CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "maps" / _MAP_CSV_FILENAME
)
_MAP_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "maps" / _MAP_JSON_FILENAME
)
# CSV Path of the order data to be used in the environment. Can be set via the 
# TARWARE_ORDER_DATA environment variable, or defaults to "order_data_sample" 
# in the data/processed directory.
_ORDER_DATA = os.getenv("TARWARE_ORDER_DATA", "order_data_sample")
_ORDER_CSV_FILENAME = f"{_ORDER_DATA}.csv"
_ORDER_CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "processed" / _ORDER_CSV_FILENAME
)
# CSV Path of the ABC analysis data to be used in the environment. Can be set via the 
# TARWARE_ABC_CSV environment variable, or defaults to "abc_data_sample.csv" in the 
# data/processed directory.
_ABC_CSV_FILENAME = os.getenv("TARWARE_ABC_CSV", "abc_data_sample.csv")
_ABC_CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "processed" / _ABC_CSV_FILENAME
)

# Number of AGVs (defaults to 3)
_NUM_AGVS = int(os.getenv("TARWARE_AGVS", 3))
# Number of pickers (defaults to 4)
_NUM_PICKERS = int(os.getenv("TARWARE_PICKERS", 4))
# Observation type (defaults to "partial")
_OBS_TYPE = os.getenv("TARWARE_OBS_TYPE", "partial")
# Request queue size (defaults to 20)
_REQUEST_QUEUE_SIZE = int(os.getenv("TARWARE_REQUEST_QUEUE_SIZE", 20))
# Steps per simulated second (defaults to 1.0)
_STEPS_PER_SIMULATED_SECOND = float(os.getenv("TARWARE_STEPS_PER_SIMULATED_SECOND", 1.0))
# Maximum inactivity steps before episode termination (defaults to None, meaning no 
# termination based on inactivity)
_MAX_INACTIVITY_STEPS = os.getenv("TARWARE_MAX_INACTIVITY_STEPS", None)
if _MAX_INACTIVITY_STEPS is not None:
    _MAX_INACTIVITY_STEPS = int(_MAX_INACTIVITY_STEPS)
# Maximum steps per episode (defaults to 500)
_MAX_STEPS = int(os.getenv("TARWARE_MAX_STEPS", 500))
# Reward type (defaults to "global")
_REWARD_TYPE = os.getenv("TARWARE_REWARD_TYPE", "global")
if _REWARD_TYPE == "global":
    _REWARD_TYPE = RewardType.GLOBAL
elif _REWARD_TYPE == "individual":
    _REWARD_TYPE = RewardType.INDIVIDUAL
elif _REWARD_TYPE == "two_stage":
    _REWARD_TYPE = RewardType.TWO_STAGE
else:
    raise ValueError(f"Invalid reward type: {_REWARD_TYPE}. Must be 'individual', 'global', or 'two_stage'.")
# Fit width for rendering
_FIT_WIDTH = os.getenv("TARWARE_RENDER_WIDTH", None)
if _FIT_WIDTH is not None:
    _FIT_WIDTH = int(_FIT_WIDTH)
# Fit height for rendering
_FIT_HEIGHT = os.getenv("TARWARE_RENDER_HEIGHT", None)
if _FIT_HEIGHT is not None:
    _FIT_HEIGHT = int(_FIT_HEIGHT)

ENV_ID: str = f"tarware/map-{_MAP_NAME}_order-{_ORDER_DATA}_agvs-{_NUM_AGVS}_pickers-{_NUM_PICKERS}_obs-{_OBS_TYPE}_v1"

gym.register(
    id=ENV_ID,
    entry_point="tarware.warehouse:Warehouse",
    kwargs={
        "map_csv_path": _MAP_CSV_PATH,
        "map_json_path": _MAP_JSON_PATH,
        "order_csv_path": _ORDER_CSV_PATH,
        "num_agvs": _NUM_AGVS,
        "num_pickers": _NUM_PICKERS,
        "observation_type": _OBS_TYPE,
        "request_queue_size": _REQUEST_QUEUE_SIZE,
        "steps_per_simulated_second": _STEPS_PER_SIMULATED_SECOND,
        "max_inactivity_steps": _MAX_INACTIVITY_STEPS,
        "max_steps": _MAX_STEPS,
        "reward_type": _REWARD_TYPE,
        "fit_width": _FIT_WIDTH,
        "fit_height": _FIT_HEIGHT,
    },
)
