"""Profile `python3 ./tarware/main.py classical eval` with cProfile.

Usage:
    python3 scripts/profile_classical_eval.py
    python3 scripts/profile_classical_eval.py --output profiles/run.prof --top 40
    python3 scripts/profile_classical_eval.py -- --episodes 3 --size medium

Anything after `--` is forwarded to `tarware.main` as CLI args (after
`classical eval`). The profile is written as a pstats `.prof` file that can
be opened with snakeviz (`snakeviz profiles/run.prof`) or gprof2dot.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables from .env file
if __name__ == "__main__":
    dotenv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    sys.path.append(os.path.dirname(dotenv_path))
    load_dotenv(dotenv_path)


from tarware.main import build_parser, configure_logging  # noqa: E402


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "profiles" / "classical_eval.prof",
        help="Path to write the binary pstats profile (default: profiles/classical_eval.prof).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of rows to show in each text summary (default: 30).",
    )
    parser.add_argument(
        "--sort",
        default="cumulative",
        choices=["cumulative", "tottime", "ncalls", "percall"],
        help="Primary sort key for the printed summary (default: cumulative).",
    )
    return parser.parse_known_args()


def main() -> None:
    opts, forwarded = parse_args()

    # Build the same argparse as tarware.main, but inject the subcommand.
    tarware_argv = ["classical", "eval", *forwarded]
    tarware_parser = build_parser()
    tarware_args = tarware_parser.parse_args(tarware_argv)
    configure_logging(tarware_args.log_level)

    opts.output.parent.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        tarware_args.func(tarware_args)
    finally:
        profiler.disable()

    profiler.dump_stats(str(opts.output))
    print(f"\nwrote profile to {opts.output}")
    print(f"view with: snakeviz {opts.output}\n")

    stats = pstats.Stats(profiler).strip_dirs().sort_stats(opts.sort)
    print(f"=== top {opts.top} by {opts.sort} ===")
    stats.print_stats(opts.top)

    print(f"=== top {opts.top} by tottime (self time) ===")
    stats.sort_stats("tottime").print_stats(opts.top)

    print(f"=== callers of top {min(opts.top, 15)} cumulative functions ===")
    stats.sort_stats("cumulative").print_callers(min(opts.top, 15))


if __name__ == "__main__":
    main()
