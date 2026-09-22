"""Print a compact convergence summary from the Conan CE VisualDL records.

Used by scripts/package_delivery.sh to embed final metrics into the delivery
README. Reads ``logs/content_extractor_ce/ocean_logs`` relative to the project
root (resolved from this file), so it can be run from anywhere:

    python scripts/pack_metrics.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import paddle  # noqa: E402

paddle.set_device("cpu")

from utils.vdl_metrics import parse_vdl_dir  # noqa: E402

VDL_DIR = PROJECT_ROOT / "logs" / "content_extractor_ce" / "ocean_logs"


def main() -> None:
    curves = parse_vdl_dir(str(VDL_DIR))
    for name in ("train/loss", "val/loss", "val/acc", "train/lr"):
        points = curves.get(name, [])
        if not points:
            print(f"{name}: empty")
            continue
        first, last = points[0], points[-1]
        line = (
            f"{name}: n={len(points)} "
            f"first@{first[0]}={first[1]:.4f} last@{last[0]}={last[1]:.4f}"
        )
        if name == "val/loss":
            best = min(points, key=lambda p: p[1])
            line += f" BEST@{best[0]}={best[1]:.4f}"
        elif name == "val/acc":
            best = max(points, key=lambda p: p[1])
            line += f" BEST@{best[0]}={best[1]:.4f}"
        print(line)


if __name__ == "__main__":
    main()
