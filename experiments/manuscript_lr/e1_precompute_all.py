#!/usr/bin/env python3
"""Запуск всех precompute для каждого из 4 вариантов обучения."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.manuscript_lr.e1_config import VARIANT_NAMES

SCRIPTS = (
    "e1_precompute_fixed_point_probability.py",
    "e1_precompute_amplitude_basins.py",
    "e1_precompute_neuromap_scan_residual.py",
)


def main() -> None:
    exp = Path(__file__).resolve().parent
    py = sys.executable
    for variant in VARIANT_NAMES:
        print(f"\n### variant={variant}")
        for script in SCRIPTS:
            cmd = [py, str(exp / script), "--variant", variant]
            print(">", " ".join(cmd))
            subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
