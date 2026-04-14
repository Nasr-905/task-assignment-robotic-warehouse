import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd


def choose_actions(env, rng):
    masks = env.unwrapped.compute_valid_action_masks()
    actions = []
    for mask in masks:
        valid = np.flatnonzero(mask)
        actions.append(int(rng.choice(valid)) if len(valid) else 0)
    return actions


def main():
    parser = argparse.ArgumentParser(description="Run a live rendered TA-RWARE simulation.")
    parser.add_argument("--map-name", default="extralarge")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.03)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--fit-width", type=int, default=1200)
    parser.add_argument("--fit-height", type=int, default=800)
    parser.add_argument("--agvs", type=int, default=14)
    parser.add_argument("--pickers", type=int, default=7)
    parser.add_argument("--render-start", type=int, default=55)
    parser.add_argument("--picker-policy", choices=["fifo", "zone"], default="fifo")
    parser.add_argument(
        "--picker-zone-overflow",
        choices=["none", "adjacent", "global"],
        default="adjacent",
    )
    parser.add_argument("--picker-stall-probability", type=float, default=0.0)
    parser.add_argument("--sku-size-pick-time", action="store_true")
    parser.add_argument("--policy", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--action-refresh", type=int, default=25)
    args = parser.parse_args()

    os.environ["TARWARE_MAP_NAME"] = args.map_name
    os.environ["TARWARE_MAX_STEPS"] = str(args.steps)
    os.environ["TARWARE_AGVS"] = str(args.agvs)
    os.environ["TARWARE_PICKERS"] = str(args.pickers)
    os.environ["TARWARE_PICKER_POLICY"] = args.picker_policy
    os.environ["TARWARE_PICKER_ZONE_OVERFLOW"] = args.picker_zone_overflow
    os.environ["TARWARE_PICKER_STALL_PROBABILITY"] = str(args.picker_stall_probability)
    os.environ["TARWARE_PICKER_USE_SKU_SIZE_TIME"] = "1" if args.sku_size_pick_time else "0"

    tile_size = args.tile_size
    if tile_size is None:
        map_path = Path(__file__).resolve().parents[1] / "data" / "maps" / f"{args.map_name}.csv"
        rows, cols = pd.read_csv(map_path, header=None).shape
        tile_size = max(4, min((args.fit_width - 1) // cols - 1, (args.fit_height - 1) // rows - 1))
    os.environ["TARWARE_RENDER_TILE_SIZE"] = str(tile_size)

    import tarware

    rng = np.random.default_rng(args.seed)
    env = gym.make(tarware.ENV_ID)
    env.reset(seed=args.seed)
    actions = [0 for _ in range(env.unwrapped.num_agents)]

    print(f"running {tarware.ENV_ID}")
    print(f"shared cells: {len(env.unwrapped.shared_highway_locs)}")
    print(f"tile size: {tile_size}px")
    print(f"agents: {args.agvs} AGVs, {args.pickers} pickers")
    print(
        "picker model: {policy}, overflow={overflow}, stall_p={stall}, "
        "sku_size_time={sku_time}".format(
            policy=args.picker_policy,
            overflow=args.picker_zone_overflow,
            stall=args.picker_stall_probability,
            sku_time=args.sku_size_pick_time,
        )
    )
    print(f"policy: {args.policy}")
    print(f"render starts at step: {args.render_start}")
    print("close the render window or press Ctrl+C to stop")

    try:
        if args.policy == "heuristic":
            from tarware.heuristic import heuristic_episode

            infos, global_return, _ = heuristic_episode(
                env.unwrapped,
                render=True,
                seed=args.seed,
                render_start=args.render_start,
                render_sleep=args.sleep,
            )
            if infos:
                last_info = infos[-1]
                print(
                    "done steps={steps} return={ret:.3f} deliveries={deliveries} "
                    "picker_yields={picker_yields} clashes={clashes}".format(
                        steps=len(infos),
                        ret=float(global_return),
                        deliveries=sum(info.get("shelf_deliveries", 0) for info in infos),
                        picker_yields=sum(info.get("picker_yields", 0) for info in infos),
                        clashes=sum(info.get("clashes", 0) for info in infos),
                    )
                )
            return

        for step in range(args.steps):
            if step % args.action_refresh == 0:
                actions = choose_actions(env, rng)

            _, _, terminated, truncated, info = env.step(actions)
            env.unwrapped.render(mode="human")

            if step % args.action_refresh == 0:
                print(
                    "step={step} deliveries={deliveries} picker_yields={picker_yields} "
                    "clashes={clashes}".format(
                        step=step,
                        deliveries=info.get("shelf_deliveries"),
                        picker_yields=info.get("picker_yields"),
                        clashes=info.get("clashes"),
                    )
                )

            if all(terminated) or all(truncated):
                break
            time.sleep(args.sleep)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
