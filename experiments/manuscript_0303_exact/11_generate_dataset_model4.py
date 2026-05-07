#!/usr/bin/env python3
from pathlib import Path

import numpy as np

from systems.vdp_mod2 import vdp_mod2_rk4
from utils import generate_pairs_dataset_finite

OUT_DIR = Path("experiments/manuscript_0303_exact/artifacts/model4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DT = 0.01
NUM_IN_TRAJ = 10
NUM_OF_TRAJ_TRAIN = 100_000
NUM_OF_TRAJ_VAL = 25_000

VARIABLES_RANGES = [(-3.3, 3.3), (-29.0, 29.0)]
PARAMETERS_RANGES = [(-3.0, 1.0), (-1.0, 4.0)]  # lambda, mu


def save_dataset(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    np.savez(path, X=X, y=y)
    print(f"Saved: {path} | X={X.shape}, y={y.shape}")


def main() -> None:
    print("Generate TRAIN dataset (model 4)")
    X_train, y_train = generate_pairs_dataset_finite(
        evolution_operator=vdp_mod2_rk4,
        variables_ranges=VARIABLES_RANGES,
        parameters_ranges=PARAMETERS_RANGES,
        num_of_traj=NUM_OF_TRAJ_TRAIN,
        num_in_traj=NUM_IN_TRAJ,
        dt=DT,
        seed=SEED,
    )
    save_dataset(OUT_DIR / "train_dataset.npz", X_train, y_train)

    print("Generate VAL dataset (model 4)")
    X_val, y_val = generate_pairs_dataset_finite(
        evolution_operator=vdp_mod2_rk4,
        variables_ranges=VARIABLES_RANGES,
        parameters_ranges=PARAMETERS_RANGES,
        num_of_traj=NUM_OF_TRAJ_VAL,
        num_in_traj=NUM_IN_TRAJ,
        dt=DT,
        seed=SEED + 1,
    )
    save_dataset(OUT_DIR / "val_dataset.npz", X_val, y_val)


if __name__ == "__main__":
    main()
