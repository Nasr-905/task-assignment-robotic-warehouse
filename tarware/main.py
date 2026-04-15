import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
if __name__ == "__main__":
    dotenv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    sys.path.append(os.path.dirname(dotenv_path))
    load_dotenv(dotenv_path)

import tarware
from tarware.heuristic import heuristic_episode

LOGGER = logging.getLogger(__name__)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc

def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {value!r}") from exc


def unpack_reset(reset_out):
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out
    return reset_out, {}


def unpack_step(step_out):
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        return obs, reward, terminated, truncated, info
    if len(step_out) == 4:
        obs, reward, done, info = step_out
        return obs, reward, done, done, info
    raise RuntimeError(f"Unexpected step return length: {len(step_out)}")


def all_done(flags: Sequence[bool]) -> bool:
    try:
        return bool(all(flags))
    except TypeError:
        return bool(flags)


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def build_env_id(size: str, agvs: int, pickers: int, obs_type: str = "partial") -> str:
    return f"tarware-{size}-{agvs}agvs-{pickers}pickers-{obs_type}obs-v1"


def _map_csv_path_for_size(size: str) -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "data"
        / "maps"
        / f"{size}.csv"
    )


def make_base_env(args) -> gym.Env:
    map_csv_path = _map_csv_path_for_size(args.size)
    if not map_csv_path.exists():
        raise FileNotFoundError(f"Map CSV not found for size={args.size!r}: {map_csv_path}")

    return gym.make(
        tarware.ENV_ID,
        map_csv_path=map_csv_path,
        num_agvs=args.agvs,
        num_pickers=args.pickers,
        observation_type=args.obs_type,
        disable_env_checker=not args.enable_env_checker,
    )


def get_env_and_id(args):
    return make_base_env(args), build_env_id(args.size, args.agvs, args.pickers, args.obs_type)


class JointWarehouseWrapper(gym.Wrapper):
    """Convert TA-RWARE multi-agent API into a single-agent API for standard RL libs."""

    def __init__(self, env: gym.Env, reward_aggregation: str = "sum"):
        super().__init__(env)
        self.reward_aggregation = reward_aggregation

        action_spaces = list(env.action_space.spaces)
        self._n_agents = len(action_spaces)
        self._nvec = np.array([space.n for space in action_spaces], dtype=np.int64)
        self.action_space = gym.spaces.MultiDiscrete(self._nvec)

        sample_obs, _ = unpack_reset(env.reset(seed=0))
        flat = self._flatten_obs(sample_obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat.shape[0],),
            dtype=np.float32,
        )

    def _flatten_obs(self, obs) -> np.ndarray:
        if isinstance(obs, tuple):
            parts = [np.asarray(o, dtype=np.float32).reshape(-1) for o in obs]
            return np.concatenate(parts, axis=0)
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _aggregate_reward(self, reward_list: Sequence[float]) -> float:
        reward_array = np.asarray(reward_list, dtype=np.float32)
        if self.reward_aggregation == "mean":
            return float(np.mean(reward_array))
        return float(np.sum(reward_array))

    def reset(self, **kwargs):
        obs, info = unpack_reset(self.env.reset(**kwargs))
        return self._flatten_obs(obs), info

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.size != self._n_agents:
            raise ValueError(f"Expected {self._n_agents} actions, got {action.size}")

        step_out = self.env.step(tuple(int(a) for a in action))
        obs, reward, terminated, truncated, info = unpack_step(step_out)

        info = dict(info) if isinstance(info, dict) else {"raw_info": info}
        info["reward_per_agent"] = list(reward)
        info["terminated_per_agent"] = list(terminated)
        info["truncated_per_agent"] = list(truncated)

        return (
            self._flatten_obs(obs),
            self._aggregate_reward(reward),
            all_done(terminated),
            all_done(truncated),
            info,
        )

def add_common_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--size",
        default=os.getenv("TARWARE_SIZE", "small"),
        choices=["tiny", "small", "medium", "large", "extralarge"],
        help="Warehouse map size.",
    )
    parser.add_argument(
        "--agvs",
        type=int,
        default=env_int("TARWARE_AGVS", 3),
        help="Number of AGV agents.",
    )
    parser.add_argument(
        "--pickers",
        type=int,
        default=env_int("TARWARE_PICKERS", 0),
        help="Number of picker agents.",
    )
    parser.add_argument(
        "--obs-type",
        default=os.getenv("TARWARE_OBS_TYPE", "partial"),
        choices=["partial", "global"],
        help="Observation type for the environment id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=env_int("TARWARE_SEED", 21),
        help="Base random seed.",
    )
    parser.add_argument(
        "--enable-env-checker",
        action="store_true",
        help="Enable Gymnasium passive env checker warnings.",
    )


def run_classical_eval(args) -> None:
    env = gym.make(tarware.ENV_ID)
    LOGGER.info("classical_eval env_id=%s", tarware.ENV_ID)
    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            infos, global_return, agent_returns = heuristic_episode(
                env.unwrapped,
                render=args.render,
                seed=episode_seed,
            )
            total_deliveries = sum(info.get("shelf_deliveries", 0) for info in infos)
            total_clashes = sum(info.get("clashes", 0) for info in infos)
            total_stucks = sum(info.get("stucks", 0) for info in infos)

            LOGGER.info(
                "episode=%s length=%s global_return=%.3f deliveries=%s clashes=%s stucks=%s agent_returns=%s",
                episode + 1,
                len(infos),
                float(global_return),
                total_deliveries,
                total_clashes,
                total_stucks,
                np.asarray(agent_returns, dtype=np.float64).round(3).tolist(),
            )
    finally:
        env.close()
        LOGGER.info("closed=True")


def run_rl_train(args) -> None:
    import stable_baselines3 as sb3

    PPO = sb3.PPO
    base_env, env_id = get_env_and_id(args)
    env = JointWarehouseWrapper(base_env, reward_aggregation=args.reward_aggregation)
    LOGGER.info("rl_train env_id=%s", env_id)
    LOGGER.info("config agvs=%d pickers=%d size=%s obs=%s", args.agvs, args.pickers, args.size, args.obs_type)
    LOGGER.info("total_timesteps=%s reward_aggregation=%s", args.total_timesteps, args.reward_aggregation)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        verbose=1 if logging.getLogger().isEnabledFor(logging.INFO) else 0,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.total_timesteps)

    model_path = Path(args.model_path).expanduser().resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    LOGGER.info("saved_model=%s", model_path)

    env.close()
    LOGGER.info("closed=True")


def run_rl_eval(args) -> None:
    import stable_baselines3 as sb3

    PPO = sb3.PPO
    base_env, env_id = get_env_and_id(args)
    env = JointWarehouseWrapper(base_env, reward_aggregation=args.reward_aggregation)
    LOGGER.info("rl_eval env_id=%s", env_id)
    LOGGER.info("config agvs=%d pickers=%d size=%s obs=%s", args.agvs, args.pickers, args.size, args.obs_type)

    model_path = Path(args.model_path).expanduser().resolve()
    print(env.observation_space)
    model = PPO.load(str(model_path), env=env)

    episode_returns = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            done = False
            truncated = False
            episode_return = 0.0
            steps = 0

            while not (done or truncated) and steps < args.max_steps:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, done, truncated, info = env.step(action)
                episode_return += float(reward)
                steps += 1

                if args.render:
                    base_env.render()
                    time.sleep(args.render_sleep)

                LOGGER.debug(
                    "episode=%s step=%s reward=%.4f done=%s truncated=%s deliveries=%s",
                    episode + 1,
                    steps,
                    float(reward),
                    done,
                    truncated,
                    info.get("shelf_deliveries"),
                )

            episode_returns.append(episode_return)
            LOGGER.info(
                "episode=%s steps=%s return=%.3f",
                episode + 1,
                steps,
                episode_return,
            )

        arr = np.asarray(episode_returns, dtype=np.float64)
        LOGGER.info("eval_summary episodes=%s mean_return=%.3f std_return=%.3f", len(arr), arr.mean(), arr.std())
    finally:
        env.close()
        LOGGER.info("closed=True")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TA-RWARE runner with classical and RL subcommands.")
    parser.add_argument(
        "--log-level",
        default=os.getenv("TARWARE_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )

    subparsers = parser.add_subparsers(dest="framework", required=True)

    classical_parser = subparsers.add_parser("classical", help="Classical (non-RL) baseline.")
    classical_sub = classical_parser.add_subparsers(dest="classical_cmd", required=True)
    classical_eval = classical_sub.add_parser("eval", help="Run heuristic baseline evaluation.")
    add_common_env_args(classical_eval)
    classical_eval.add_argument(
        "--steps",
        type=int,
        default=env_int("TARWARE_STEPS", 1000),
        help="Maximum steps per episode.",
    )
    classical_eval.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes.")
    classical_eval.add_argument("--render", action="store_true", help="Render while evaluating.")
    classical_eval.set_defaults(func=run_classical_eval)

    rl_parser = subparsers.add_parser("rl", help="RL workflows.")
    rl_sub = rl_parser.add_subparsers(dest="rl_cmd", required=True)

    rl_train = rl_sub.add_parser("train", help="Train RL policy.")
    add_common_env_args(rl_train)
    rl_train.add_argument(
        "--total-timesteps",
        type=int,
        default=env_int("TARWARE_STEPS", 100_000),
        help="Training timesteps.",
    )
    rl_train.add_argument("--learning-rate", type=float, default=3e-4)
    rl_train.add_argument("--gamma", type=float, default=0.99)
    rl_train.add_argument("--model-path", type=str, default="models/tarware_ppo")
    rl_train.add_argument("--reward-aggregation", choices=["sum", "mean"], default="sum")
    rl_train.set_defaults(func=run_rl_train)

    rl_eval = rl_sub.add_parser("eval", help="Evaluate trained RL policy.")
    add_common_env_args(rl_eval)
    rl_eval.add_argument("--model-path", type=str, required=True)
    rl_eval.add_argument("--episodes", type=int, default=5)
    rl_eval.add_argument("--max-steps", type=int, default=env_int("TARWARE_STEPS", 500))
    rl_eval.add_argument("--deterministic", action="store_true")
    rl_eval.add_argument("--render", action="store_true")
    rl_eval.add_argument("--render-sleep", type=float, default=env_float("TARWARE_RENDER_SLEEP", 0.05))
    rl_eval.add_argument("--reward-aggregation", choices=["sum", "mean"], default="sum")
    rl_eval.set_defaults(func=run_rl_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    if getattr(args, "agvs", 1) < 1:
        parser.error("--agvs must be >= 1")

    args.func(args)


if __name__ == "__main__":
    main()