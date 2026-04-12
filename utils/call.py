from pathlib import Path

import numpy as np

from .networkBuilder import generate_degrees


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 500
    mean_k = 6
    var_k = 20
    np.random.seed(42)

    degs = generate_degrees(n=n, mean_k=mean_k, var_k=var_k)
    out_path = out_dir / "degrees.npy"
    np.save(out_path, degs)
    print(f"Saved degree sequence ({degs.size} nodes) to {out_path}")


if __name__ == "__main__":
    main()
