import argparse
from pathlib import Path

import numpy as np


def load_heatmap(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".csv":
        return np.loadtxt(path, delimiter=",")
    raise ValueError(f"Unsupported heatmap format: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Display a TA-RWARE heatmap from CSV or NPY.")
    parser.add_argument("path", type=Path, help="Path to a heatmap .csv or .npy file.")
    parser.add_argument("--title", type=str, default=None, help="Plot title.")
    parser.add_argument("--cmap", type=str, default="hot", help="Matplotlib colormap.")
    parser.add_argument("--origin", choices=["upper", "lower"], default="upper")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the image instead of only showing it.")
    parser.add_argument("--dpi", type=int, default=180, help="DPI for saved figures.")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install it with `pip install matplotlib`.") from exc

    heatmap_path = args.path.expanduser().resolve()
    data = load_heatmap(heatmap_path)

    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap=args.cmap, origin=args.origin, interpolation="nearest")
    plt.colorbar(label="Visits")
    plt.title(args.title or heatmap_path.stem)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    if args.save is not None:
        output_path = args.save.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=args.dpi)

    plt.show()


if __name__ == "__main__":
    main()
